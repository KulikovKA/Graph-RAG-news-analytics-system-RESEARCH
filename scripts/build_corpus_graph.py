"""Build and publish the isolated Stage 3A.5.11 Corpus100 knowledge graph."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import subprocess
import sys
import time
import traceback

from filelock import FileLock
from tqdm import tqdm

import build_relex_graphs as relex
import corpus_graph_inference as inference
import corpus_graph_payload as graph
from cgraph.ner_v2.io import read_documents
from relex_experiment import MemorySampler

ROOT = relex.ROOT
STAGE = "3A.5.11.2"
NAMESPACE = "relex_corpus100_stage_3a5_11_v1"
DEFAULT_OUTPUT = ROOT / "artifacts/corpus_graph_3a5_11"
CORPUS = ROOT / "data/retrieval/benchmark_3a5_11/corpus100/abstracts.jsonl"
FROZEN = ROOT / "reports/retrieval/benchmark_3a5_11/corpus100"
SPEC = ROOT / "docs/architecture/STAGE_3A5_11_CORPUS100_GRAPH.md"
ER_HISTORY = ROOT / "reports/entity_resolution/relex_base_t035"
SEED = 42
E3_PRE_PUNCTUATION_FIX = "9bdda1adf663eef6f367113625ee50420454d1ddc7170bc49893938ac907cb7e"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    # Временный файл находится на том же томе, что и итоговый checkpoint.
    relex.atomic_json(path, value)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def source_identity(document):
    return {key: document[key] for key in ("document_id", "revision_id", "text_sha256")}


def checkpoint(path, identity, results, seconds):
    body = dict(identity, status="complete", results=results, seconds=seconds, completed_at=stamp())
    atomic_json(path, body | dict(checksum=signature(body)))


def load_checkpoint(path, identity, verify):
    try:
        value = read(path)
        checksum = value.pop("checksum")
        if (checksum != signature(value) or value["status"] != "complete"
                or any(value.get(k) != v for k, v in identity.items())
                or not math.isfinite(value["seconds"]) or value["seconds"] < 0):
            return None
        verify(value["results"])
        return value["results"]
    except (FileNotFoundError, UnicodeError, ValueError, KeyError, TypeError, IndexError, AttributeError,
            OverflowError):
        return None


def configure(threads, original_spec_sha256=None):
    e3 = read(relex.SOURCE / "E3/config.json")
    rel = read(ROOT / "reports/re_v2/relex_comparison/threshold035/e3_base/experiment_config.json")
    hist_er = read(ER_HISTORY / "global_kg_2f3/manifest.json")
    e5 = read(ER_HISTORY / "semantic_concept_sweep_extended/manifest.json")["embedding"]["model"]
    lock = {name: row["sha256"] for name, row in e3["checkpoint"]["files"].items()}
    model_files = dict(ner=lock, re=rel["models"]["base"],
                       re_common={name: value for name, value in rel["models"]["large"].items()
                                  if name != "model.safetensors"},
                       e5={row["path"].replace("\\", "/").split("intfloat__e5-small-v2/")[1]: row["sha256"]
                           for row in e5["files"]})
    versions = {name: version(name) for name in (
        "torch", "gliner", "gliner2", "transformers", "numpy", "spacy", "en-core-web-sm",
        "huggingface-hub", "tokenizers", "sentencepiece", "peft", "neo4j", "tqdm", "psutil")}
    graph.require(e3["seed"] == SEED and sha(relex.ONTOLOGY) == e3["ontology_sha256"],
                  "E3 seed or ontology differs from the historical experiment")
    ontology = read(relex.ONTOLOGY)
    graph.require(ontology == rel["ontology"] and relex.PREDICATES == rel["predicates"],
                  "RelEx prompts or ontology drift")
    ner_model = e3["checkpoint"]["repo_id"] + "@" + e3["checkpoint"]["revision"]
    return dict(stage=STAGE, namespace=NAMESPACE, seed=SEED, workers=2, threads=threads, device="cpu",
                python=sys.version.split()[0], versions=versions, model_files=model_files,
                corpus_commit="750d03a801fd771879b956cdc32324404e72ce57", corpus_sha256=sha(CORPUS),
                spec_sha256=original_spec_sha256 or sha(SPEC),
                ner=dict(model=ner_model, threshold=.35, decoder_threshold=e3["minimum_inference_threshold"],
                         **{k: e3[k] for k in ("labels", "window_words", "overlap_words", "max_input_tokens",
                                               "normalization", "dtype")}),
                re=dict(model=rel["model_ids"]["base"], threshold=rel["relation_threshold"],
                        ontology=ontology, predicates=relex.PREDICATES, token_limit=rel["token_limit"],
                        overlap=rel["overlap"], adapter=relex.VERSION),
                er=dict(strategy=hist_er["ner_confidence_strategy"], threshold=hist_er["ner_threshold"],
                        policy="exact-source-abbreviation-pairs-v1"),
                embeddings=dict(model=e5["model"], dimension=e5["dimension"], prefix=e5["prefix"],
                                pooling=e5["pooling"], mode="text_name", batch=8, token_limit=512,
                                semantic_threshold="0.950", hybrid_threshold="0.830", hybrid_top_entities=20,
                                hybrid_top_concepts=10))


def verify_models(config, paths, names):
    for name in names:
        for relative, expected in config["model_files"][name].items():
            path = Path(paths[name]) / relative
            if sha(path) != expected:
                raise ValueError(f"Model checksum mismatch: {name}/{relative}")


def worker(stage, factory, config, paths, tasks, events, slot):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with FileLock(str(slot)):
            if not mp.parent_process().is_alive():
                return
            infer = factory(config, paths)
            for doc, upstream, path, identity in tasks:
                if not mp.parent_process().is_alive():
                    return
                try:
                    start = time.perf_counter()
                    results = infer(doc, upstream)
                    validate_document_result(stage, doc, results, upstream, config)
                    if not mp.parent_process().is_alive():
                        return
                    checkpoint(path, identity, results, time.perf_counter()-start)
                    events.put((doc["document_id"], None))
                except Exception:
                    events.put((doc["document_id"], traceback.format_exc()))
    except Exception:
        events.put((None, traceback.format_exc()))


def validate_document_result(stage, doc, results, upstream, config):
    if stage == "ner":
        inference.validate_ner(doc, results, config["ner"])
    elif stage == "re":
        inference.validate_re(doc, results, upstream, config["re"])


class Run:
    def __init__(self, output, resume, config, memory):
        self.output, self.config, self.memory = output, config, memory
        self.fingerprint = signature(config)
        self.manifest = output / "manifest.json"
        self.progress = output / "progress.json"
        if self.manifest.exists():
            old = read(self.manifest)
            if not resume or old["fingerprint"] != self.fingerprint or old["configuration"] != config:
                raise ValueError("Existing output requires --resume and the same frozen configuration")
        elif resume:
            raise ValueError("--resume requires an existing run manifest")
        else:
            atomic_json(self.manifest, dict(fingerprint=self.fingerprint, configuration=config, created_at=stamp()))
        self.state = read(self.progress) if self.progress.exists() else dict(
            stages={}, peak_rss_bytes=0, status="running", errors=0)
        self.state.update(status="running", last_invocation=stamp())
        self.save()

    def save(self):
        self.state["peak_rss_bytes"] = max(self.state["peak_rss_bytes"], self.memory.peak)
        atomic_json(self.progress, self.state | dict(updated_at=stamp(), fingerprint=self.fingerprint))

    def error(self, stage, unit, message):
        with (self.output / "errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(stage=stage, document_id=unit, time=stamp(), error=message),
                                    ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.state["errors"] += 1
        self.save()

    def stage(self, name, finished, total, recovered, previous, start, errors=0):
        self.state["current_stage"] = name
        self.state["stages"][name] = dict(processed=finished, remaining=total-finished, total=total,
                                          recovered=recovered, errors=errors,
                                          wall_seconds=previous + time.perf_counter()-start)
        self.save()

    def previous_seconds(self, name):
        return self.state["stages"].get(name, {}).get("wall_seconds", 0)

    def identity(self, name, source, upstream, model_names, implementation):
        return dict(source, stage=name, config_fingerprint=self.fingerprint, upstream_sha256=signature(upstream),
                    model_fingerprint=signature({k: self.config["model_files"][k] for k in model_names}),
                    implementation_sha256=signature(implementation))

    def documents(self, name, docs, factory, paths, model_names, upstream=None):
        start, previous = time.perf_counter(), self.previous_seconds(name)
        results, pending, errors = {}, [], 0
        implementation = [inspect.getsource(factory)]
        if name == "ner":
            implementation.extend(inspect.getsource(f) for f in (inference.e3_windows, inference.e3_source_span))
        elif name == "re":
            implementation.extend([sha(Path(relex.__file__)), inspect.getsource(inference.entities_for)])
        for doc in docs:
            did = doc["document_id"]
            dependency = upstream[did] if upstream is not None else None
            path = self.output / "checkpoints" / name / f"{did}.json"
            identity = self.identity(name, source_identity(doc), dependency, model_names, implementation)
            verify = lambda r: validate_document_result(name, doc, r, dependency, self.config)
            result = load_checkpoint(path, identity, verify)
            if result is None and name == "ner" and path.exists():
                # Старые успешные результаты уже прошли точную проверку исходных spans.
                result = load_checkpoint(path, identity | {"implementation_sha256": E3_PRE_PUNCTUATION_FIX}, verify)
            if result is None:
                pending.append((doc, dependency, path, identity))
            else:
                results[did] = result
        recovered, total = len(results), len(docs)
        with tqdm(total=total, initial=recovered, desc=name.upper(), unit="doc", dynamic_ncols=True) as bar:
            def update():
                bar.set_postfix(done=bar.n, left=total-bar.n, restored=recovered, errors=errors, refresh=True)
                self.stage(name, bar.n, total, recovered, previous, start, errors)

            update()
            if pending:
                count = min(2, len(pending))
                lock_root = DEFAULT_OUTPUT.parent if self.config.get("namespace") == NAMESPACE else self.output
                slots = [lock_root / f"{NAMESPACE}.worker-{i}.lock" for i in range(2)]
                # После аварии сначала освобождаем оба слота предыдущего запуска.
                for slot in slots:
                    with FileLock(str(slot), timeout=180):
                        pass
                verify_models(self.config, paths, model_names)
                preflight(self.config["threads"], count)
                ctx = mp.get_context("spawn")
                events = ctx.Queue()
                processes = [ctx.Process(target=worker, args=(name, factory, self.config, paths,
                                                             pending[i::count], events, slots[i])) for i in range(count)]
                tasks = {doc["document_id"]: (path, identity) for doc, _, path, identity in pending}
                try:
                    for process in processes:
                        process.start()
                    while True:
                        try:
                            did, error = events.get(timeout=.5)
                        except Empty:
                            if all(not p.is_alive() for p in processes):
                                break
                            continue
                        if error:
                            errors += 1
                            self.error(name, did, error)
                        else:
                            path, identity = tasks[did]
                            result = load_checkpoint(path, identity, lambda _: None)
                            if result is None:
                                raise ValueError("Worker reported an invalid checkpoint")
                            results[did] = result
                            bar.update(1)
                        update()
                    for process in processes:
                        if process.exitcode != 0:
                            errors += 1
                            self.error(name, None, f"Worker process exit code {process.exitcode}")
                    update()
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                        if process.pid is not None:
                            process.join()
                    events.close()
            if len(results) != total:
                raise RuntimeError(f"{name}: {len(results)}/{total} verified, see errors.jsonl")
        return {doc["document_id"]: results[doc["document_id"]] for doc in docs}

    def global_stage(self, name, calculate, inputs, implementations, model_names=()):
        start, previous = time.perf_counter(), self.previous_seconds(name)
        path = self.output / "checkpoints" / name / "corpus.json"
        identity = self.identity(name, self.corpus_identity(), inputs, model_names, implementations)
        result = load_checkpoint(path, identity, lambda _: None)
        recovered = int(result is not None)
        self.stage(name, recovered, 1, recovered, previous, start)
        if result is None:
            result = calculate()
            checkpoint(path, identity, result, time.perf_counter()-start)
        self.stage(name, 1, 1, recovered, previous, start)
        return result

    def corpus_identity(self):
        return dict(document_id="Corpus100", revision_id="sha256:" + self.config["corpus_sha256"],
                    text_sha256=self.config["corpus_sha256"])

    def embedding_blocks(self, label, names, get_embed):
        import numpy as np

        name = "embeddings_" + label
        previous, start = self.previous_seconds(name), time.perf_counter()
        batch_size, dimension = (self.config["embeddings"][key] for key in ("batch", "dimension"))
        unique = list(dict.fromkeys(names))
        blocks = [unique[i:i+batch_size] for i in range(0, len(unique), batch_size)]
        signature_code = inspect.getsource(inference.e5_worker)
        pending, vectors = [], {}

        def check(rows, block):
            values = np.asarray(rows, dtype=np.float32)
            if (values.shape != (len(block), dimension) or not np.isfinite(values).all()
                    or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-6)):
                raise ValueError("Invalid E5 block")

        for pos, block in enumerate(blocks):
            path = self.output / "checkpoints" / "embeddings" / f"{label}-{pos:05}.json"
            identity = self.identity(name, self.corpus_identity(), block, ("e5",), signature_code)
            results = load_checkpoint(path, identity, lambda rows: check(rows, block))
            if results is None:
                pending.append((block, path, identity))
            else:
                vectors.update(zip(block, results))
        restored = len(blocks)-len(pending)
        with tqdm(total=len(blocks), initial=restored, desc=name, unit="block") as bar:
            self.stage(name, restored, len(blocks), restored, previous, start)
            for block, path, identity in pending:
                since = time.perf_counter()
                results = get_embed()(block)
                check(results, block)
                checkpoint(path, identity, results, time.perf_counter()-since)
                vectors.update(zip(block, results))
                bar.update(1)
                self.stage(name, bar.n, len(blocks), restored, previous, start)
        return vectors


def preflight(threads, workers):
    import psutil

    logical = psutil.cpu_count() or 1
    available = psutil.virtual_memory().available
    if logical < workers*threads or available < (workers*3+1)*2**30:
        raise RuntimeError(f"Local preflight failed: {logical} logical CPUs, {available / 2**30:.2f} GiB available")
    print(f"CPU preflight PASS: {logical} logical CPUs, {available / 2**30:.2f} GiB available, "
          f"{workers} workers × {threads} threads", flush=True)


def log_mlflow(uri, report):
    import requests

    if not uri:
        return dict(status="DISABLED")
    api = uri.rstrip("/") + "/api/2.0/mlflow/"

    def call(method, route, **kwargs):
        response = requests.request(method, api + route, timeout=(3, 8), **kwargs)
        response.raise_for_status()
        return response.json()

    try:
        experiment = call("GET", "experiments/get-by-name", params={"experiment_name": "data_expansion"})
        experiment_id = experiment["experiment"]["experiment_id"]
        external_id = f"corpus100-{report['graph_fingerprint']}"
        existing = call("POST", "runs/search", json=dict(
            experiment_ids=[experiment_id], filter=f"tags.external_id = '{external_id}'", max_results=2))
        runs = existing.get("runs", [])
        if len(runs) > 1:
            raise ValueError("Duplicate MLflow external_id")
        if runs:
            run_id = runs[0]["info"]["run_id"]
            if runs[0]["info"]["status"] == "FINISHED":
                return dict(status="PASS", run_id=run_id, reused=True)
        else:
            run = call("POST", "runs/create", json=dict(experiment_id=experiment_id, run_name=STAGE + " Corpus100",
                       start_time=int(time.time()*1000), tags=[dict(key="external_id", value=external_id)]))
            run_id = run["run"]["info"]["run_id"]
        measured = {k: v for k, v in report["counts"].items() if isinstance(v, (int, float))}
        measured.update({f"{k}_seconds": v for k, v in report["stage_seconds"].items()})
        measured["peak_rss_bytes"] = report["peak_rss_bytes"]
        now = int(time.time()*1000)
        call("POST", "runs/log-batch", json=dict(run_id=run_id, metrics=[
            dict(key=k, value=v, timestamp=now, step=0) for k, v in measured.items()],
            params=[dict(key=key, value=report[key]) for key in
                    ("namespace", "graph_fingerprint", "corpus_sha256", "config_fingerprint")]))
        call("POST", "runs/update", json=dict(run_id=run_id, status="FINISHED", end_time=now))
        return dict(status="PASS", run_id=run_id, reused=False)
    except (requests.RequestException, KeyError, ValueError) as exc:
        return dict(status="BLOCKED", error_type=type(exc).__name__)


def build(run, docs, paths, neo4j_env, mlflow_uri):
    cfg = run.config
    documents = {doc["document_id"]: doc for doc in docs}
    ner = run.documents("ner", docs, inference.ner_worker, paths, ("ner",))
    relations = run.documents("re", docs, inference.re_worker, paths, ("re", "re_common"), upstream=ner)
    resolution = run.global_stage("entity_resolution", lambda: graph.resolve_entities(
        documents, ner, relations, cfg["er"]["threshold"]), [ner, relations],
        [inspect.getsource(graph.resolve_entities), inspect.getsource(graph.audit)], ("ner", "re"))
    embedding = cfg["embeddings"]
    embed = None

    def get_embed():
        nonlocal embed
        if embed is None:
            verify_models(cfg, paths, ("e5",))
            embed = inference.e5_worker(paths["e5"], cfg["threads"], cfg["seed"], embedding)
        return embed

    source_vectors = run.embedding_blocks("entity", [m["text_name"] for m in sorted(
        (m for group in resolution["entities"].values() for m in group),
        key=lambda m: m["source_entity"]["entity_id"])], get_embed)
    links = run.global_stage("semantic", lambda: graph.semantic_links(
        resolution, source_vectors, embedding["semantic_threshold"]), [resolution, source_vectors],
        inspect.getsource(graph.semantic_links), ("e5",))

    def concepts():
        entities = graph.profiles(documents, resolution)
        lexical, catalog, existing = graph.lexical_concepts(entities, links)
        names = [e.name for e in entities] + [c.name for c in sorted(catalog, key=lambda c: c.concept_id)]
        vectors = run.embedding_blocks("concept", names, get_embed)
        candidates = graph.hybrid_candidates(entities, catalog, existing, links, vectors,
                         embedding["hybrid_threshold"], embedding["hybrid_top_entities"], embedding["hybrid_top_concepts"])
        return graph.hybrid_concepts(entities, lexical, catalog, existing, candidates, embedding["hybrid_threshold"])

    concepts_result = run.global_stage("concepts", concepts, [resolution, links],
        [inspect.getsource(f) for f in (graph.profiles, graph.lexical_concepts, graph.hybrid_candidates,
                                        graph.hybrid_concepts, graph.evaluate)], ("ner", "e5"))
    fingerprint = signature([run.fingerprint, resolution, links, concepts_result])
    nodes, edges = graph.build_payload(NAMESPACE, fingerprint, documents, resolution, links, concepts_result)
    counts = graph.graph_statistics(nodes, edges, resolution, ner, relations)
    graph.require(counts["documents"] == 100 and counts["linked_document_pairs"] > 0, "Graph acceptance criterion failed")
    report = dict(stage=STAGE, namespace=NAMESPACE, config_fingerprint=run.fingerprint,
                  graph_fingerprint=fingerprint, corpus_sha256=cfg["corpus_sha256"], counts=counts,
                  publication_spec_sha256=sha(SPEC),
                  stage_seconds={k: v["wall_seconds"] for k, v in run.state["stages"].items()},
                  peak_rss_bytes=run.memory.peak, neo4j=dict(status="NOT RUN"))
    report_path = run.output / "results.json"
    prior = read(report_path) if report_path.exists() else {}
    if prior.get("graph_fingerprint") != fingerprint or prior.get("neo4j", {}).get("status") != "PASS":
        prior = {}
        atomic_json(report_path, report)
    started, previous = time.perf_counter(), run.previous_seconds("neo4j")
    run.stage("neo4j", 0, 1, 0, previous, started)
    driver, database = relex.connect_neo4j(neo4j_env)
    try:
        report["neo4j"] = graph.import_neo4j(driver, database, NAMESPACE, nodes, edges)
    finally:
        driver.close()
    run.stage("neo4j", 1, 1, int(report["neo4j"]["first"]["created_nodes"] == 0), previous, started)
    report["stage_seconds"]["neo4j"] = run.state["stages"]["neo4j"]["wall_seconds"]
    report["peak_rss_bytes"] = run.state["peak_rss_bytes"]
    if prior:
        # Сохраняем измерения первой публикации и отдельно фиксируем повторное чтение.
        report = prior | dict(last_resume=dict(time=stamp(), neo4j=report["neo4j"],
                                               wall_seconds=time.perf_counter()-started))
    report["mlflow"] = log_mlflow(mlflow_uri, report)
    atomic_json(report_path, report)
    run.state["status"] = "complete"
    run.save()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--neo4j-env", type=Path)
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    parser.add_argument("--e5-model", type=Path, default=ROOT.parent / "embedding-models/intfloat__e5-small-v2")
    args = parser.parse_args()
    paths = dict(ner=str(ROOT / "artifacts/ner_v2/models/gliner25"),
                 re=str(ROOT / "artifacts/models/gliner-relex-base-v1.0"),
                 re_common=str(ROOT / "artifacts/models/gliner-relex-large-v1.0"), e5=str(args.e5_model.resolve()))
    print("WARNING: LONG CORPUS100 CPU RUN. TWO PERSISTENT WORKERS PER NER/RE STAGE. "
          "NEO4J WRITES ARE SCOPED TO " + NAMESPACE.upper(), flush=True)
    lock_path = DEFAULT_OUTPUT.parent / (NAMESPACE + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=0):
        if not CORPUS.exists():
            subprocess.run([sys.executable, str(FROZEN / "prepare.py"), "restore"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(FROZEN / "prepare.py"), "verify"], cwd=ROOT, check=True)
        docs = read_documents(CORPUS)
        output = args.output.resolve()
        manifest = output / "manifest.json"
        # Редакция процедуры публикации не меняет конфигурацию уже выполненного inference.
        original_spec = read(manifest)["configuration"]["spec_sha256"] if args.resume and manifest.exists() else None
        config = configure(args.threads, original_spec)
        output.mkdir(parents=True, exist_ok=True)
        (output / "errors.jsonl").touch(exist_ok=True)
        with MemorySampler() as memory:
            run = Run(output, args.resume, config, memory)
            try:
                build(run, docs, paths, args.neo4j_env, args.mlflow_uri)
            except KeyboardInterrupt:
                run.state["status"] = "interrupted"
                run.save()
                return 130
            except Exception as exc:
                run.state["status"] = "failed"
                run.error(run.state.get("current_stage"), None, str(exc))
                raise
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
