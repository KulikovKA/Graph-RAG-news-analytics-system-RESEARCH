"""Stage 3A.5.7: read-only, resumable end-to-end retrieval benchmark."""

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from inspect import signature
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

from integrated_benchmark_reporting import atomic_text, encode_json, write_json
from validate_draft import HERE, LOCK_SHA256, SOURCE, documents, jsonl, require, validate


ROOT = HERE.parents[2]
GOLD = HERE / "questions_draft.jsonl"
HOLDOUT = HERE / "validate_promote_c/holdout_original_c.jsonl"
DATASET_HASHES = {
    "approved": "2533e252f1d2654689034494a6b96f8aa68ed33821fe5e8b2573768150f67124",
    "holdout": "cfddb313dfe97d989021a76f83136ff3f4dcb504450b8d3a6127cdd60622265f",
}
GROUPS = {"A": 8, "B": 7, "C": 10, "HOLDOUT": 10}
BFS = {"max_hops": 5, "max_neighbors": 50, "max_relationships": 1000,
       "max_seeds": 32, "timeout_seconds": 10.0}
LINKING = {"min_score": 0.85, "fallback": "baseline", "max_fallback_seeds": 6}
GLINER_THRESHOLD = 0.15
RERANK_TOP_K = 5
TORCH_SEED = 42
CPU_THREADS = 4
SCHEMA_VERSION = 1
VARIANTS = {
    name: {"reranking": policy, "rerank_top_k": RERANK_TOP_K, "gliner_threshold": GLINER_THRESHOLD,
           "linking": {**LINKING, "top_k": top_k},
           "bfs": {**BFS, "max_candidates": candidates}, "seed_fair": False}
    for name, policy, top_k, candidates in (
        ("A", "A", 5, 200), ("B", "C", 5, 200), ("C", "C", 10, 200), ("D", "C", 10, 100))
}
DEPENDENCIES = {
    "torch": "2.11.0", "gliner2": "2.1.0", "transformers": "5.1.0", "tokenizers": "0.22.2",
    "huggingface-hub": "1.28.0", "safetensors": "0.7.0", "numpy": "2.4.1",
    "neo4j": "6.1.0", "tqdm": "4.70.1",
}
STAGES = {"ner": "GLiNER", "e5": "E5 / cosine", "linking": "Entity Linking",
          "bfs": "Neo4j Retrieval", "evidence": "Evidence Extraction", "rerank": "Reranking"}
NEO4J_UNAVAILABLE = {"ServiceUnavailable", "SessionExpired", "RoutingServiceUnavailable", "ConnectionRefusedError"}


class BenchmarkError(ValueError):
    """Диагностика из проверенных сообщений runner."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(encode_json(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


@contextmanager
def run_lock(directory):
    # Блокировка снимается ОС даже после завершения процесса извне.
    with (directory / ".lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            acquire()
        except OSError:
            raise BenchmarkError("The run directory is locked by another process.") from None
        yield


@contextmanager
def model_output(log_path):
    from transformers.utils import logging as transformers_logging

    # Загрузчик весов пишет собственный tqdm и предупреждения. Оставляем терминал полосам runner.
    logger = logging.getLogger("transformers")
    handlers, propagate = logger.handlers[:], logger.propagate
    progress_enabled = transformers_logging.is_progress_bar_enabled()
    with closing(logging.FileHandler(log_path, encoding="utf-8")) as handler:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.handlers[:] = [handler]
        logger.propagate = False
        transformers_logging.disable_progress_bar()
        try:
            yield
        finally:
            logger.handlers[:] = handlers
            logger.propagate = propagate
            if progress_enabled:
                transformers_logging.enable_progress_bar()


@dataclass(frozen=True)
class Question:
    group: str
    question_id: str
    text: str

    @property
    def key(self):
        return f"{self.group}/{self.question_id}"


def load_questions(corpus):
    for label, path in (("approved", GOLD), ("holdout", HOLDOUT)):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == DATASET_HASHES[label],
                f"Unexpected frozen dataset SHA-256 for {label}")
    groups = {group: [] for group in GROUPS}
    for row in jsonl(GOLD):
        if row["category"] in ("A", "B", "C"):
            groups[row["category"]].append(row)
    groups["HOLDOUT"] = list(jsonl(HOLDOUT))
    questions, references = [], {}
    for group, rows in groups.items():
        category = "C" if group == "HOLDOUT" else group
        expected_ids = {f"3A52-{category}{i:02d}" for i in range(1, GROUPS[group] + 1)}
        require(len(rows) == len(expected_ids) and {r["question_id"] for r in rows} == expected_ids,
                f"Unexpected IDs for {group}")
        for row in rows:
            if group != "HOLDOUT":
                validate(row, corpus)
            else:
                for span in row["evidence"]:
                    doc = corpus[span["document_id"]]
                    require(doc["revision_id"] == span["revision_id"]
                            and doc["text"][span["start"]:span["end"]] == span["quote"],
                            f"Historical source mismatch for {row['question_id']}")
            question = Question(group, row["question_id"], row["question"])
            questions.append(question)
            references[question.key] = {
                **row, "evidence": [{**span, "evidence_id": span.get("evidence_id", f"E{i:02d}")}
                                    for i, span in enumerate(row["evidence"], 1)],
            }
    return questions, references


def diagnostic(exc):
    # Исключения библиотек могут содержать URI, настройки и переменные окружения.
    code = getattr(exc, "code", None)
    code = code if isinstance(code, str) and re.fullmatch(r"Neo\.[A-Za-z.]+", code) else None
    kind = type(exc).__name__
    if isinstance(exc, BenchmarkError):
        return str(exc)
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"Required file missing ({Path(exc.filename).name}). See the assets section of the README."
    if kind in NEO4J_UNAVAILABLE:
        return f"{kind}. Start the existing Neo4j server and check NEO4J_URI and the Bolt port."
    if code == "Neo.ClientError.Database.DatabaseNotFound":
        return f"{code}. Set NEO4J_DATABASE to the database containing the frozen namespace."
    if code and ".Security." in code:
        return f"{code}. Verify the credentials and read permissions for the selected database."
    if code and "Timeout" in code:
        return f"{code}. Check Neo4j availability and server load, then resume with the same limits."
    return f"{kind}" + (f" ({code})" if code else "")


def verify_graph(driver, database, corpus=None):
    from neo4j import READ_ACCESS
    import smoke_retrieval_3a1 as audit

    graph = {"namespace": audit.TARGET_NAMESPACE, "database": database}
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        try:
            audit.verify(session, graph)
        except ValueError as exc:
            reason = str(exc)
            if not re.fullmatch(r"[a-z_]+", reason):
                raise
            if reason == "node_counts" and not graph["node_counts"]:
                raise BenchmarkError(f"Namespace {audit.TARGET_NAMESPACE} is absent. "
                                     "Select the existing Stage 2G.3 database in NEO4J_DATABASE.") from None
            raise BenchmarkError(f"Frozen graph check failed ({reason}). Check the Stage 2G.3 database.") from None
        nodes = {}
        if corpus is not None:
            rows = session.run(audit.stage.graph.READ_NODES, ns=audit.TARGET_NAMESPACE).data()
            live_docs = {r["properties"]["document_id"]: r["properties"] for r in rows
                         if "RELEXDocument" in r["labels"]}
            if live_docs.keys() != corpus.keys() or any(
                live_docs[key][field] != doc[field] for key, doc in corpus.items()
                for field in ("revision_id", "text_sha256", "text")
            ):
                raise BenchmarkError("Neo4j documents differ from the frozen corpus.")
            nodes = {row["properties"]["key"]: row["properties"] for row in rows}
    return graph, nodes


def preflight(args, notify):
    if sys.version_info[:2] != (3, 11):
        raise BenchmarkError("Python 3.11 is required. Activate the project environment.")
    versions = {"python": platform.python_version()}
    for name, expected in DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise BenchmarkError(f"Missing dependency {name}. Run uv sync --frozen --extra benchmark --extra relex.") from None
        if actual.split("+")[0] != expected:
            notify(f"Preflight | {name} {actual} differs from pinned {expected}. "
                   "Use the frozen environment for comparable runs.")
        versions[name] = actual

    from gliner2 import AutoExtractor
    from transformers import AutoModel
    from tokenizers import Tokenizer
    import torch
    from cgraph.entity_linking import EntityLinker, _sha, load_entity_cache
    from cgraph.graph_retrieval import GraphRetriever
    from cgraph.query_analysis import E3Extractor, E3_HASHES, MODEL_ID
    import smoke_retrieval_3a1 as audit

    # Импорты проверяют API моделей без загрузки весов.
    if not all((AutoExtractor, AutoModel, Tokenizer, torch)):
        raise BenchmarkError("The pinned runtime could not load the model APIs.")
    if not hasattr(EntityLinker, "_score") or "seed_fair" not in signature(GraphRetriever.retrieve).parameters:
        raise BenchmarkError("Merged retrieval API required. Update design-v2-sequential-kg to 99dc3d0 or later.")
    notify("Preflight | validating frozen corpus and both reference sets")
    try:
        corpus = documents()
        questions, references = load_questions(corpus)
    except (KeyError, ValueError, TypeError) as exc:
        raise BenchmarkError(f"Frozen corpus or reference validation failed ({type(exc).__name__}). "
                             "Check the pinned files in the README.") from None

    notify("Preflight | verifying entity vectors and GLiNER/E5 hashes")
    try:
        cache = load_entity_cache(ROOT / ".cache/semantic_concept_sweep")
    except ValueError:
        raise BenchmarkError("Frozen entity cache validation failed. Restore .cache/semantic_concept_sweep.") from None
    e3 = E3Extractor(ROOT / "artifacts/ner_v2/models/gliner25")
    description = cache.model
    original_root = PureWindowsPath(description["root"])

    def e5_path(name):
        if args.e5_model_dir is None:
            return Path(name)
        return args.e5_model_dir.resolve() / PureWindowsPath(name).relative_to(original_root)

    files = [{**item, "path": str(e5_path(item["path"]))} for item in description["files"]]
    model = {**description, "files": files, **{
        field: str(e5_path(description[field])) for field in ("root", "config", "tokenizer", "weight")
    }}
    e5_hashes = {PureWindowsPath(item["path"]).relative_to(original_root).as_posix(): item["sha256"]
                 for item in description["files"]}
    for path, expected in [(e3.model_path / name, sha) for name, sha in E3_HASHES.items()] + [
        (Path(item["path"]), item["sha256"]) for item in files
    ]:
        if _sha(path) != expected:
            raise BenchmarkError(f"Model checksum mismatch for {path.parent.name}/{path.name}. Restore the frozen files.")
    cache = replace(cache, model=model)

    notify("Preflight | checking Neo4j connection and namespace snapshot")
    core = audit.stage.graph.base.core
    if args.neo4j_env is not None and not args.neo4j_env.is_file():
        raise BenchmarkError("The --neo4j-env file was not found. Supply the existing Neo4j .env path.")
    try:
        driver, database = core.connect_neo4j(args.neo4j_env)
    except core.Neo4jConnectionError as exc:
        raise BenchmarkError(str(exc)) from None
    except ValueError:
        if not os.getenv("NEO4J_PASSWORD"):
            raise BenchmarkError("NEO4J_PASSWORD is missing. Set it in PowerShell or pass --neo4j-env.") from None
        raise BenchmarkError("Neo4j connection settings are invalid. Check NEO4J_URI and NEO4J_DATABASE.") from None
    try:
        graph, nodes = verify_graph(driver, database, corpus)
        tracked_sources = [Path(__file__), HERE / "integrated_benchmark_reporting.py",
                           HERE / "run_block_c.py", HERE / "run_document_aware.py", HERE / "validate_draft.py"]
        tracked_sources.extend(ROOT / "src/cgraph" / f"{name}.py" for name in (
            "query_analysis", "entity_linking", "graph_retrieval", "evidence_extraction", "path_reranking",
            "document_ranking", "retrieval_types", "semantic_concepts", "ner_v2/detection"))
        identity = {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "versions": versions, "graph": graph, "dataset_sha256": DATASET_HASHES,
            "corpus_sha256": _sha(SOURCE), "document_lock_sha256": LOCK_SHA256,
            "models": {"GLiNER": {"id": MODEL_ID, "sha256": E3_HASHES},
                       "E5": {"id": model["model"], "sha256": e5_hashes,
                              "entity_cache_fingerprint": cache.fingerprint}},
            "source_sha256": {path.relative_to(ROOT).as_posix(): _sha(path) for path in tracked_sources},
        }
        return identity, questions, references, driver, database, nodes, e3, cache
    except BaseException:
        driver.close()
        raise


class Progress:
    def __init__(self, enabled, notify, questions, variants, saved):
        self.notify, self.bars, self.stage_name = notify, [], "Preflight"
        self.counts = Counter(row["variant"] for row in saved.values())
        self.question_total = len(questions)
        self.restored = len(saved)
        completed = sum(all((q.key, variant) in saved for variant in variants) for q in questions)
        if enabled:
            from tqdm import tqdm

            for position, (label, total, initial, unit, bar_format) in enumerate((
                ("Overall benchmark", len(questions) * len(variants), len(saved), "eval",
                 "{desc}: {n_fmt}/{total_fmt} evaluations |{bar}| [{elapsed}<{remaining}]{postfix}"),
                ("Questions", len(questions), completed, "q", None),
                ("Variants", len(questions), 0, "eval", "{desc}: {n_fmt}/{total_fmt} |{bar}|"),
            )):
                self.bars.append(tqdm(total=total, initial=initial, desc=label, unit=unit,
                                      position=position, dynamic_ncols=True, ascii=True, file=sys.stderr,
                                      bar_format=bar_format))
            self.bars[0].set_postfix_str(f"restored={self.restored}")

    def question(self, question):
        self.current = question
        self.started = perf_counter()

    def variant(self, variant):
        if self.bars:
            bar = self.bars[2]
            bar.set_description_str(f"Variant {variant}", refresh=False)
            bar.n = self.counts[variant]
            bar.refresh()

    def stage(self, name):
        self.stage_name = name
        if self.bars:
            detail = f"{self.current.group}/{self.current.question_id[5:]} | {name}"
            self.bars[1].set_postfix_str(f"{detail} | {perf_counter() - self.started:.1f}s")

    def saved(self, row):
        variant = row["variant"]
        self.counts[variant] += 1
        if self.bars:
            self.bars[0].update()
            self.bars[2].update()
        metrics = row["metrics"]
        self.notify(f"{row['key']} | Variant {variant} | "
                    f"GoldDocs {len(metrics['gold_documents_in_top_5'])}/{row['gold_count']} | "
                    f"Factual {len(metrics['factual_gold_documents_in_top_5'])}/{row['gold_count']} | "
                    f"Paths {row['candidate_count']} | {row['wall_seconds']:.2f}s")

    def question_done(self):
        if self.bars:
            self.bars[1].update()

    def close(self):
        for bar in reversed(self.bars):
            bar.close()


class StageCache:
    def __init__(self, directory, fingerprint, progress):
        self.directory, self.fingerprint, self.progress = directory, fingerprint, progress
        self.memory = {}

    def stage(self, name, inputs, operation):
        self.progress.stage(STAGES[name])
        key = digest([self.fingerprint, name, inputs])
        path = self.directory / "cache" / name / f"{key}.json"
        origin = "memory"
        record = self.memory.get(key)
        if record is None and path.is_file():
            origin = "disk"
            record = read_record(path, key)
        if record is None:
            origin = "computed"
            started = perf_counter()
            value = operation()
            record = {"value": value, "seconds": perf_counter() - started}
            write_json(path, {"key": key, "sha256": digest(record), "record": record})
        self.memory[key] = record
        return record["value"], {"key": key, "seconds": record["seconds"], "origin": origin,
                                 "file": path.relative_to(self.directory).as_posix()}


class Pipeline:
    def __init__(self, stages, entity_cache, encoder, analyzer, retriever, extractor, rank, graph_fingerprint):
        import numpy as np
        from cgraph.entity_linking import EntityLinker

        self.stages, self.analyzer = stages, analyzer
        self.retriever, self.extractor, self.rank = retriever, extractor, rank
        self.graph_fingerprint = graph_fingerprint
        encoding_seconds = 0.0

        def timed_encode(texts):
            nonlocal encoding_seconds
            started = perf_counter()
            vectors = encoder.encode(texts)
            encoding_seconds = perf_counter() - started
            return vectors

        class CachedLinker(EntityLinker):
            # Точка подстановки score cache, используемая в sweep Stage 3A.5.5.
            def _score(linker, texts):
                parent_score = super()._score

                def compute():
                    started = perf_counter()
                    scores = parent_score(texts)
                    return {"dtype": str(scores.dtype), "scores": scores.tolist(),
                            "encoding_seconds": encoding_seconds,
                            "cosine_and_validation_seconds": perf_counter() - started - encoding_seconds}

                started = perf_counter()
                value, linker.score_stage = self.stages.stage("e5", [linker.query_hash, texts], compute)
                for key in ("encoding_seconds", "cosine_and_validation_seconds"):
                    linker.score_stage[key] = value[key]
                linker.score_wall = perf_counter() - started
                self.stages.progress.stage(STAGES["linking"])
                return np.array(value["scores"], dtype=value["dtype"])

        self.linker = CachedLinker(entity_cache, SimpleNamespace(encode=timed_encode), top_k=10, **LINKING)

    def evaluate(self, question, variant):
        from cgraph.graph_retrieval import TraversalLimits
        from cgraph.retrieval_types import validate_seed

        config = VARIANTS[variant]
        query_hash = digest(question)
        query, ner = self.stages.stage("ner", query_hash, lambda: self.analyzer.analyze(question))
        self.linker.query_hash = query_hash

        def link():
            self.linker.score_stage, self.linker.score_wall = None, 0.0
            result = self.linker.link(query, top_k=config["linking"]["top_k"])
            return {"result": result, "e5": self.linker.score_stage, "score_wall": self.linker.score_wall}

        linked, linking = self.stages.stage("linking", [ner["key"], config["linking"]], link)
        linking["seconds"] = max(0.0, linking["seconds"] - linked["score_wall"])
        e5 = linked["e5"]
        if e5 and linking["origin"] != "computed":
            e5 = {**e5, "origin": linking["origin"]}
        limits = TraversalLimits(**config["bfs"])
        seeds = linked["result"]["seeds"][:limits.max_seeds]
        # FIFO использует порядок первых node_key и document_id, а reranking использует также scores.
        physical_seeds = {}
        for seed in seeds:
            validate_seed(seed, self.linker.namespace)
            document = physical_seeds.setdefault(seed["node_key"], seed["document_id"])
            require(document == seed["document_id"], "Conflicting seed documents")
        def retrieve():
            result = self.retriever.retrieve(seeds, limits, seed_fair=config["seed_fair"])
            if result["graph_fingerprint"] != self.graph_fingerprint:
                raise BenchmarkError("Graph fingerprint changed during traversal. Verify the frozen Neo4j namespace.")
            return result

        retrieval, bfs = self.stages.stage(
            "bfs", [query_hash, list(physical_seeds.items()), config["bfs"], config["seed_fair"]],
            retrieve)
        bundles, evidence = self.stages.stage(
            "evidence", [retrieval["namespace"], retrieval["graph_fingerprint"], digest(retrieval["paths"])],
            lambda: self.extractor.extract(retrieval))
        ranking, rerank = self.stages.stage(
            "rerank", [query_hash, bfs["key"], evidence["key"], seeds, config["reranking"], config["rerank_top_k"]],
            lambda: self.rank(question, retrieval, bundles, seeds, policy=config["reranking"],
                              top_k=config["rerank_top_k"]))
        timings = {"ner": ner, "e5": e5, "linking": linking, "bfs": bfs, "evidence": evidence, "rerank": rerank}
        linking_trace = {"seeds": seeds, "seeds_before_cap": len(linked["result"]["seeds"]),
                         "linking_candidates": len(linked["result"]["candidates"]),
                         "unresolved_mentions": linked["result"]["unresolved_mentions"],
                         "abstention_reason": linked["result"]["abstention_reason"]}
        return retrieval, bundles, ranking, timings, linking_trace


def read_record(path, key):
    try:
        envelope = read_json(path)
        if envelope["key"] == key and envelope["sha256"] == digest(envelope["record"]):
            return envelope["record"]
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass
    return None


def checkpoint_path(directory, question, variant):
    return directory / "checkpoints" / question.group / f"{question.question_id}-{variant}.json"


def restore_checkpoints(directory, fingerprint, questions, variants, notify):
    restored, checked = {}, {}
    for question in questions:
        for variant in variants:
            path = checkpoint_path(directory, question, variant)
            if not path.is_file():
                continue
            key = digest([fingerprint, question.key, variant])
            row = read_record(path, key)
            valid = row is not None and row.get("key") == question.key and row.get("variant") == variant
            if valid:
                for stage in row["timings"].values():
                    if stage and stage["key"] not in checked:
                        checked[stage["key"]] = read_record(directory / stage["file"], stage["key"]) is not None
                    if stage and not checked[stage["key"]]:
                        valid = False
            if valid:
                restored[question.key, variant] = row
            else:
                notify(f"{question.key} | Variant {variant} | invalid checkpoint will be retried")
    return restored


def execute(directory, fingerprint, questions, variants, references, nodes, pipeline, progress, saved, error):
    from integrated_benchmark_reporting import evaluate_metrics

    for group in GROUPS:
        grouped = [question for question in questions if question.group == group]
        if not grouped:
            continue
        for question in grouped:
            if all((question.key, variant) in saved for variant in variants):
                continue
            pipeline.stages.memory.clear()
            progress.question(question)
            for variant in variants:
                if (question.key, variant) in saved:
                    continue
                progress.variant(variant)
                started = perf_counter()
                try:
                    retrieval, bundles, ranking, timings, linking_trace = pipeline.evaluate(question.text, variant)
                    progress.stage("Metrics")
                    row = evaluate_metrics(question, variant, retrieval, bundles, ranking,
                                           references[question.key], nodes, timings, linking_trace)
                    row["wall_seconds"] = perf_counter() - started
                except Exception as exc:
                    error(progress.stage_name, exc, question.key, variant)
                    if type(exc).__name__ in NEO4J_UNAVAILABLE:
                        return False
                    continue
                progress.stage("Saving results")
                key = digest([fingerprint, question.key, variant])
                write_json(checkpoint_path(directory, question, variant),
                           {"key": key, "record": row, "sha256": digest(row)})
                saved[question.key, variant] = row
                progress.saved(row)
            if all((question.key, variant) in saved for variant in variants):
                progress.question_done()
        summaries = []
        for variant in variants:
            rows = [saved[q.key, variant] for q in grouped if (q.key, variant) in saved]
            hits = sum(len(row["metrics"]["gold_documents_in_top_5"]) for row in rows)
            total = sum(row["gold_count"] for row in rows)
            recall = f"{hits}/{total} ({hits / total:.1%})" if total else "N/A"
            summaries.append(f"{variant} {recall}, {len(rows)}/{len(grouped)} q")
        progress.notify(f"Group {group} | Gold Recall micro | " + " | ".join(summaries))
    return True


def run_benchmark(args, directory, config, driver, database, nodes, e3, cache,
                  questions, references, notify):
    from cgraph.entity_linking import E5Encoder
    from cgraph.evidence_extraction import EvidenceExtractor
    from cgraph.graph_retrieval import GraphRetriever
    from cgraph.path_reranking import rerank_paths
    from cgraph.query_analysis import QueryAnalyzer
    import torch
    from integrated_benchmark_reporting import write_reports

    manifest_path = directory / "manifest.json"
    config_path = directory / "config.json"
    variants = config["selection"]["variants"]
    fingerprint = digest(config)
    with run_lock(directory):
        if args.resume:
            if read_json(config_path) != config:
                raise BenchmarkError("Resume fingerprint mismatch. Check commit, assets, environment and filters.")
            manifest = read_json(manifest_path) if manifest_path.is_file() else {}
            if manifest.get("status") == "COMPLETE":
                raise BenchmarkError("This experiment is complete. Select a new output directory.")
            if manifest and manifest.get("fingerprint") != fingerprint:
                raise BenchmarkError("Manifest fingerprint mismatch.")
        else:
            write_json(config_path, config)
            manifest = {}
        notify("Run directory " + str(directory))
        log_path = directory / "execution.log"
        errors_path = directory / "errors.jsonl"
        errors_path.touch(exist_ok=True)

        def event(message):
            notify(message)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"{utc_now()} {message}\n")

        def error(stage, exc, key=None, variant=None):
            item = {"timestamp_utc": utc_now(), "stage": stage, "key": key, "variant": variant,
                    "error_type": type(exc).__name__, "reason": diagnostic(exc)}
            atomic_text(errors_path, errors_path.read_text(encoding="utf-8") + encode_json(item) + "\n")
            event(" | ".join(value for value in (key, variant, stage, item["reason"]) if value))

        started = perf_counter()
        start_time = utc_now()
        previous_attempts = manifest.get("attempts", [])
        manifest.update(started_at_utc=manifest.get("started_at_utc", start_time),
                        fingerprint=fingerprint, status="RUNNING", expected_evaluations=len(questions) * len(variants))
        write_json(manifest_path, manifest)
        saved = restore_checkpoints(directory, fingerprint, questions, variants, event) if args.resume else {}
        progress = Progress(not args.no_progress and sys.stderr.isatty(), event, questions, variants, saved)
        status, exit_code = "PARTIAL", 3
        try:
            event(f"Preflight PASS | restored {len(saved)}/{manifest['expected_evaluations']} evaluations")
            event("WARNING: LIVE NEO4J AND SEQUENTIAL GLINER/E5 CPU INFERENCE MAY TAKE TIME.")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            torch.manual_seed(TORCH_SEED)
            torch.set_num_threads(CPU_THREADS)
            stages = StageCache(directory, fingerprint, progress)
            graph = config["identity"]["graph"]
            pipeline = Pipeline(stages, cache, E5Encoder(cache.model),
                                QueryAnalyzer(e3, threshold=GLINER_THRESHOLD),
                                GraphRetriever(driver, database=database, namespace=graph["namespace"]),
                                EvidenceExtractor(driver, database=database, namespace=graph["namespace"]),
                                rerank_paths, graph["graph_fingerprint"])
            with model_output(log_path):
                finished_evaluations = execute(
                    directory, fingerprint, questions, variants, references, nodes, pipeline, progress, saved, error)
            if finished_evaluations:
                progress.stage_name = "Postflight"
                after, _ = verify_graph(driver, database)
                if after != graph:
                    raise BenchmarkError("Graph snapshot changed during the run.")
                if len(saved) == manifest["expected_evaluations"]:
                    status, exit_code = "COMPLETE", 0
        except KeyboardInterrupt as exc:
            status, exit_code = "INTERRUPTED", 130
            error(progress.stage_name, exc)
        except Exception as exc:
            error(progress.stage_name, exc)
        finally:
            progress.close()
            finished = utc_now()
            manifest.update(status=status, finished_at_utc=finished,
                            completed_evaluations=len(saved),
                            completed_questions=sum(all((q.key, v) in saved for v in variants) for q in questions),
                            expected_questions=len(questions),
                            attempts=[*previous_attempts, {"started_at_utc": start_time, "finished_at_utc": finished,
                                                           "wall_seconds": perf_counter() - started,
                                                           "restored_evaluations": progress.restored, "status": status}])
            event("Saving results | generating JSON, CSV and Markdown")
            write_reports(directory, config, manifest, questions, saved)
            write_json(manifest_path, manifest)
        event(f"{status} | {len(saved)}/{manifest['expected_evaluations']} evaluations")
        return exit_code


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="verify environment and frozen assets without inference")
    parser.add_argument("--output-dir", type=Path, help="exact run directory (default: integrated_runs/<UTC run ID>)")
    parser.add_argument("--resume", action="store_true", help="resume an unfinished --output-dir with the same inputs")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), help="default: A B C D")
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), help="default: A B C HOLDOUT")
    parser.add_argument("--question-id", type=str.upper, help="single ID such as C08 or 3A52-C08; use --groups for one C set")
    parser.add_argument("--no-progress", action="store_true", help="disable dynamic progress bars")
    parser.add_argument("--neo4j-env", type=Path, help="existing Neo4j .env (process environment takes precedence)")
    parser.add_argument("--e5-model-dir", type=Path, help="directory with the same pinned E5 files at a different location")
    args = parser.parse_args(argv)
    if args.resume and (args.output_dir is None or args.preflight):
        parser.error("--resume requires --output-dir and a benchmark run")
    if args.question_id and not re.fullmatch(r"(3A52-)?[ABC][0-9]{2}", args.question_id):
        parser.error("--question-id must look like C08 or 3A52-C08")
    return args


def main(argv=None):
    args = parse_args(argv)
    if ROOT != Path.cwd().resolve():
        sys.stderr.write("Run this script from the repository root.\n")
        return 2
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

    def notify(message):
        try:
            from tqdm import tqdm
        except ImportError:
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
        else:
            tqdm.write(message)
            sys.stdout.flush()

    directory = args.output_dir or HERE / "integrated_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
    directory = directory.resolve()
    driver = None
    phase = "Preflight"
    try:
        stored_selection = {}
        if args.resume:
            if not (directory / "config.json").is_file():
                raise BenchmarkError("--resume requires an existing run with config.json.")
            stored_selection = read_json(directory / "config.json")["selection"]
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file() and read_json(manifest_path).get("status") == "COMPLETE":
                raise BenchmarkError("This experiment is complete. Select a new output directory.")
        elif directory.exists() and not args.preflight:
            raise BenchmarkError("Output directory exists. Use --resume or choose a new directory.")
        requested_variants = args.variants or stored_selection.get("variants", VARIANTS)
        requested_groups = args.groups or stored_selection.get("groups", GROUPS)
        question_id = args.question_id or stored_selection.get("question_id")
        if question_id and not question_id.startswith("3A52-"):
            question_id = f"3A52-{question_id}"
        selection = {"variants": [v for v in VARIANTS if v in requested_variants],
                     "groups": [g for g in GROUPS if g in requested_groups], "question_id": question_id}
        notify("Preflight | checking dependencies and pinned runtime")
        identity, questions, references, driver, database, nodes, e3, cache = preflight(args, notify)
        questions = [q for q in questions if q.group in selection["groups"]
                     and (question_id is None or q.question_id == question_id)]
        if not questions or not selection["variants"]:
            raise BenchmarkError("No evaluations match the supplied filters.")
        config = {"schema_version": SCHEMA_VERSION, "stage": "3A.5.7", "identity": identity,
                  "selection": selection, "question_keys": [q.key for q in questions], "variants": VARIANTS,
                  "runtime": {"cpu_threads": CPU_THREADS, "torch_seed": TORCH_SEED}}
        if args.preflight:
            graph = identity["graph"]
            notify(f"Preflight PASS | {graph['namespace']} | {graph['graph_fingerprint']} | "
                   f"{len(questions)} questions × {len(selection['variants'])} variants")
            return 0
        phase = "Benchmark"
        if not args.resume:
            directory.mkdir(parents=True, exist_ok=False)
        return run_benchmark(args, directory, config, driver, database, nodes, e3, cache,
                             questions, references, notify)
    except KeyboardInterrupt:
        notify(f"{phase} interrupted | {directory}")
        return 130
    except Exception as exc:
        notify(f"{phase} FAILED | {diagnostic(exc)}")
        return 2 if phase == "Preflight" else 3
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
