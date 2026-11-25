"""Run the frozen Corpus100 contextual Shared Concepts benchmark."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import inspect
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

from filelock import FileLock
import numpy as np
from tqdm import tqdm

import build_corpus_graph as frozen
from build_corpus_graph import atomic_json, read, sha, signature
from corpus_graph_inference import e5_worker
from corpus_graph_payload import profiles
from cgraph.ner_v2.io import read_documents, validate_span
from relex_experiment import MemorySampler
from contextual_shared_concepts import (
    MAX_TOKENS, PREFIX, RULES, SEED, THRESHOLDS, WINDOWS, DocumentWindows,
    cosine_scores, extract_windows, filter_mask, fold, mean_embeddings, pair_count,
    pair_guard_codes, pair_indices, stats,
)

ROOT = frozen.ROOT
OUTPUT = ROOT / "artifacts/benchmark_3a5_12"
REPORT = ROOT / "reports/retrieval/benchmark_3a5_12"
PRIOR = frozen.DEFAULT_OUTPUT
DIAG = ROOT / "reports/retrieval/benchmark_3a5_11/shared_concepts_diagnostics/REJECTED_CANDIDATES.csv"
DIAG_CACHE = ROOT / "artifacts/shared_concepts_diagnostics_3a5_11/corpus100_semantic.json"
MODEL = ROOT.parent / "embedding-models/intfloat__e5-small-v2"
REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
RECIPE = dict(model="intfloat/e5-small-v2", dimension=384, prefix=PREFIX, pooling="masked_mean",
              mode="text_name", batch=8, token_limit=MAX_TOKENS)
STAGE = "3A.5.12"
CHECKPOINT_LIMIT_SECONDS = 6 * 3600
CHECKPOINT_LIMIT_RSS = 4 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_checkpoint(path, metadata, stage, **identity):
    result = frozen.load_checkpoint(path, dict(
        document_id="Corpus100", text_sha256=metadata["configuration"]["corpus_sha256"],
        revision_id="sha256:" + metadata["configuration"]["corpus_sha256"],
        config_fingerprint=metadata["fingerprint"], stage=stage, **identity), lambda _: None)
    require(result is not None, f"Invalid frozen checkpoint: {path.name} ({stage})")
    return result


def checked_entities(documents, resolution):
    by = resolution["entities"]
    result = []
    for entity in profiles(documents, resolution):
        doc = documents[entity.document_id]
        expansions = defaultdict(set)
        members = by[entity.entity_id]
        require(bool(entity.name.strip() and entity.entity_types and entity.evidence), "Missing profile fields")
        for evidence, member in zip(entity.evidence, members, strict=True):
            source = member["source_entity"]
            require(doc["document_id"] == evidence["document_id"] == member["context"]["document_id"]
                    == source["document_id"] and source["entity_id"] == member["context"]["source_entity_id"]
                    and resolution["mapping"][source["entity_id"]] == entity.entity_id
                    and (source["start"], source["end"]) == (
                        evidence["start"] + member["context"]["mention_start"],
                        evidence["start"] + member["context"]["mention_end"])
                    and evidence["start"] <= source["start"] < source["end"] <= evidence["end"],
                    "Inconsistent entity provenance")
            for keys in [("start", "end", "text"), ("mention_start", "mention_end", "mention_text")]:
                validate_span(doc["text"], *(evidence[k] for k in keys))
            mentions = source["source_mentions"]
            require(bool(mentions) and sorted(m["mention_id"] for m in mentions) == sorted(evidence["source_mention_ids"]),
                    "Missing source mentions")
            for mention in mentions:
                require((mention["document_id"], mention["revision_id"], mention["start"], mention["end"],
                         mention["surface_text"]) ==
                        (doc["document_id"], doc["revision_id"], source["start"], source["end"], source["surface_text"]),
                        "Source mention identity differs from entity")
                for item in mention["abbreviation_candidates"]:
                    for key in ("short", "long"):
                        validate_span(doc["text"], item[key+"_start"], item[key+"_end"], item[key+"_form"])
                    short, long = fold(item["short_form"]), fold(item["long_form"])
                    # Учитываем раскрытие, привязанное к форме члена профиля в источнике.
                    if fold(source["surface_text"]) in (short, long):
                        expansions[short].add(long)
        result.append(dict(entity_id=entity.entity_id, document_id=entity.document_id, name=entity.name,
                           entity_types=list(entity.entity_types), topic=doc["topic"], evidence=list(entity.evidence),
                           expansions={k: sorted(v) for k, v in sorted(expansions.items())}))
    return result


def name_reuse(entities, shared, metadata):
    lexical = sorted((r for r in shared["concepts"] if r["concept_id"].startswith("concept:2g:")),
                     key=lambda r: r["concept_id"])
    names = list(dict.fromkeys([e["name"] for e in entities] + [c["name"] for c in lexical]))
    model = signature({"e5": metadata["configuration"]["model_files"]["e5"]})
    implementation = signature(inspect.getsource(e5_worker))
    result, files = {}, {}
    for i in range(0, len(names), RECIPE["batch"]):
        batch = names[i:i+RECIPE["batch"]]
        path = PRIOR / "checkpoints/embeddings" / f"concept-{i//RECIPE['batch']:05}.json"
        files[str(path.relative_to(ROOT))] = sha(path)
        vectors = frozen_checkpoint(path, metadata, "embeddings_concept", upstream_sha256=signature(batch),
                                    model_fingerprint=model, implementation_sha256=implementation)
        array = np.asarray(vectors, dtype=np.float32)
        require(array.shape == (len(batch), RECIPE["dimension"]) and np.isfinite(array).all()
                and np.allclose(np.linalg.norm(array, axis=1), 1, rtol=0, atol=2e-6), "Invalid cached E5 vectors")
        result.update(zip(batch, array, strict=True))
    return {e["entity_id"]: result[e["name"]] for e in entities}, files


def inputs(model):
    metadata = read(PRIOR/"manifest.json")
    config = metadata["configuration"]
    require(signature(config) == metadata["fingerprint"], "Frozen configuration checksum mismatch")
    require(all(config["embeddings"][k] == v for k, v in RECIPE.items()), "E5 recipe mismatch")
    require(config["device"] == "cpu" and config["seed"] == SEED, "Frozen device or seed mismatch")
    files = config["model_files"]["e5"]
    for name, expected in files.items():
        require(sha(model/name) == expected, f"E5 checksum mismatch for {name}")
    revision_file = model/".cache/huggingface/download/model.safetensors.metadata"
    require(revision_file.read_text(encoding="utf-8").splitlines()[0] == REVISION, "E5 revision mismatch")
    paths = [frozen.CORPUS, frozen.FROZEN/"document_lock.json", PRIOR/"manifest.json", PRIOR/"results.json",
             PRIOR/"checkpoints/entity_resolution/corpus.json", PRIOR/"checkpoints/semantic/corpus.json",
             PRIOR/"checkpoints/concepts/corpus.json", DIAG, DIAG_CACHE,
             DIAG_CACHE.with_name("input_manifest.json"),
             ROOT/"contracts/runtime_ontology.json",
             ROOT/"reports/retrieval/benchmark_3a5_11/shared_concepts_diagnostics/REPORT.md",
             ROOT/"reports/retrieval/benchmark_3a5_11/shared_concepts_diagnostics/CONCEPTS.md"]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    require(sha(frozen.CORPUS) == config["corpus_sha256"], "Frozen corpus checksum mismatch")
    documents = {d["document_id"]: d for d in read_documents(frozen.CORPUS)}
    locks = read(frozen.FROZEN/"document_lock.json")
    require({d["document_id"] for d in locks} == documents.keys()
            and all(all(documents[d["document_id"]][k] == d[k] for k in
                        ("revision_id", "text_sha256", "topic", "source_url", "license")) for d in locks)
            and sorted(Counter(d["topic"] for d in documents.values()).values()) == [20]*5, "Document lock mismatch")
    resolution = frozen_checkpoint(paths[4], metadata, "entity_resolution")
    semantics = frozen_checkpoint(paths[5], metadata, "semantic")
    shared = frozen_checkpoint(paths[6], metadata, "concepts")
    baseline = read(paths[3])
    require(signature([metadata["fingerprint"], resolution, semantics, shared]) == baseline["graph_fingerprint"]
            == read(paths[9])["graph_fingerprint"], "Frozen graph fingerprint mismatch")
    entities = checked_entities(documents, resolution)
    names, name_files = name_reuse(entities, shared, metadata)
    hashes.update(name_files)
    with DIAG.open(encoding="utf-8", newline="") as stream:
        diagnostic = list(csv.DictReader(stream))
    summary = read(DIAG_CACHE)
    require(len(diagnostic) == len(summary["examples"]) == 20
            and all(a["candidate_id"] == b["candidate_id"] for a, b in zip(diagnostic, summary["examples"])),
            "Diagnostics do not match the frozen replay")
    sizes = Counter(e["document_id"] for e in entities)
    total = pair_count(len(entities))
    cross = total - sum(map(pair_count, sizes.values()))
    require((len(documents), len(entities), total, cross) == (100, 4506, 10149765, 10042851),
            "Frozen pair counts differ from Corpus100 contract")
    return dict(entities=entities, documents=documents, baseline=baseline, shared=shared, diagnostic=diagnostic,
                names=names, hashes=hashes, model_files=files, threads=config["threads"],
                counts=dict(documents=100, entities=4506, all_pairs=total, cross_document_pairs=cross),
                baseline_summary={k: summary[k] for k in ("threshold_ee", "threshold_ec", "semantic")})


def atomic_array(path, values):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def checked_output(path, identity, files):
    if not path.exists():
        return None
    try:
        value = read(path)
        checksum = value.pop("checksum")
        valid = checksum == signature(value) and all(
            (path.parent/name).is_file() and sha(path.parent/name) == value["files"][name] for name in files)
    except (ValueError, TypeError, KeyError, FileNotFoundError):
        valid = False
    if not valid:
        print(f"Checkpoint integrity failed, restoring {path.name} from upstream", flush=True)
        return None
    require(value["fingerprint"] == identity, f"Incompatible checkpoint {path.name}")
    return value


def complete(path, identity, files, **result):
    value = dict(result, fingerprint=identity, files={name: sha(path.parent/name) for name in files})
    atomic_json(path, dict(value, checksum=signature(value)))
    return value


class EmbeddingCache:
    def __init__(self, connection, model, threads, sampler, started):
        self.db = connection
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors "
                        "(input TEXT PRIMARY KEY, vector BLOB NOT NULL, source_window TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS batches (source_window TEXT NOT NULL, seconds REAL NOT NULL)")
        self.model = model
        self.threads = threads
        self.sampler = sampler
        self.started = started
        self.embed = None

    def run(self, texts, tokenizer, window):
        cached, pending = {}, []
        for value in texts:
            row = self.db.execute("SELECT vector FROM vectors WHERE input = ?", (value,)).fetchone()
            if row is None:
                pending.append(value)
            else:
                cached[value] = np.frombuffer(row[0], dtype="<f4")
        pending.sort(key=lambda value: len(tokenizer.encode(PREFIX+value).ids))
        batch_size, dimension = RECIPE["batch"], RECIPE["dimension"]
        for i in tqdm(range(0, len(pending), batch_size), desc=f"{window} · E5", unit="batch"):
            require(self.sampler.peak <= CHECKPOINT_LIMIT_RSS and
                    time.perf_counter()-self.started <= CHECKPOINT_LIMIT_SECONDS, "Stage resource budget exceeded")
            start = time.perf_counter()
            if self.embed is None:
                self.embed = e5_worker(self.model, self.threads, SEED, RECIPE)
            batch = pending[i:i+batch_size]
            values = np.asarray(self.embed(batch), dtype="<f4")
            require(values.shape == (len(batch), dimension) and np.isfinite(values).all()
                    and np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=2e-6), "Invalid E5 output")
            with self.db:
                self.db.executemany("INSERT INTO vectors VALUES (?, ?, ?)",
                                    ((key, value.tobytes(), window) for key, value in zip(batch, values, strict=True)))
                self.db.execute("INSERT INTO batches VALUES (?, ?)", (window, time.perf_counter()-start))
            cached.update(zip(batch, values, strict=True))
        encoded = self.db.execute("SELECT count(*) FROM vectors WHERE source_window = ?", (window,)).fetchone()[0]
        seconds = self.db.execute("SELECT coalesce(sum(seconds), 0) FROM batches WHERE source_window = ?",
                                  (window,)).fetchone()[0]
        return cached, dict(unique_inputs=len(texts), cache_hits=len(texts)-encoded, encoded=encoded,
                            inference_seconds=seconds)


def document_metrics(entities, left, right):
    doc_names = sorted({e["document_id"] for e in entities})
    topics = sorted({e["topic"] for e in entities})
    doc_index = {d: i for i, d in enumerate(doc_names)}
    topic_index = {t: i for i, t in enumerate(topics)}
    doc = np.array([doc_index[e["document_id"]] for e in entities])
    topic = np.array([topic_index[e["topic"]] for e in entities])
    by_doc = np.zeros(len(doc_names), dtype=np.int32)
    by_doc[doc] = topic
    entity_totals = np.bincount(topic, minlength=len(topics))
    doc_totals = np.bincount(by_doc, minlength=len(topics))

    def measure(mask):
        a, b = left[mask], right[mask]
        ids = np.flatnonzero(np.bincount(np.concatenate((a, b)), minlength=len(doc)))
        docs = np.unique(doc[ids])
        da, db = doc[a], doc[b]
        doc_pairs = np.unique(np.minimum(da, db)*len(doc_names) + np.maximum(da, db)).size
        ta, tb = topic[a], topic[b]
        matrix = np.bincount(np.minimum(ta, tb)*len(topics) + np.maximum(ta, tb),
                             minlength=len(topics)**2).reshape(len(topics), -1)
        entity_counts = np.bincount(topic[ids], minlength=len(topics))
        doc_counts = np.bincount(by_doc[docs], minlength=len(topics))
        return dict(entities=len(ids), entity_share=len(ids)/len(entities), documents=len(docs),
                    document_share=len(docs)/len(doc_names), document_pairs=doc_pairs, topics={
                        name: dict(entities=int(entity_counts[i]), entity_share=float(entity_counts[i]/entity_totals[i]),
                                   documents=int(doc_counts[i]), document_share=float(doc_counts[i]/doc_totals[i]),
                                   within_topic_pairs=int(matrix[i, i]))
                        for i, name in enumerate(topics)}, cross_topics={
                            f"{a} | {b}": int(matrix[i, j]) for i, a in enumerate(topics)
                            for j, b in enumerate(topics) if i < j})

    return measure


def threshold_sweep(window, name_scores, context_scores, name_domain, guard, measure, sampler):
    candidate_context = context_scores[name_domain]
    candidate_name = name_scores[name_domain]
    ctx_count = {t: int(np.count_nonzero(context_scores >= t)) for t in THRESHOLDS}
    name_count = {t: int(np.count_nonzero(candidate_name >= t)) for t in THRESHOLDS}
    grid = product(THRESHOLDS, repeat=2)
    rows = []
    for tn, tc in tqdm(grid, total=len(THRESHOLDS)**2, desc=f"{window} · thresholds", unit="config"):
        since = time.perf_counter()
        both = filter_mask(candidate_name, candidate_context, tn, tc)
        remaining = both.copy()
        counts, matched = {}, {}
        for i, rule in enumerate(RULES):
            fails = (guard & (1 << i)) != 0
            matched[rule] = int(np.count_nonzero(both & fails))
            counts[rule] = int(np.count_nonzero(remaining & fails))
            remaining &= ~fails
        rows.append(dict(window=window, threshold_name=tn, threshold_context=tc, cross_pairs=len(name_scores),
                         passes_name=name_count[tn], passes_context=ctx_count[tc],
                         pure_pairs=int(both.sum()), guarded_pairs=int(remaining.sum()), excluded=counts,
                         rule_hits=matched, pure=measure(both), guarded=measure(remaining),
                         sweep_row_seconds=time.perf_counter()-since, process_peak_rss_bytes=sampler.peak))
    return rows


def execute(args, data, output, sampler, started):
    from contextlib import closing
    from tokenizers import Tokenizer
    from threadpoolctl import threadpool_limits

    entities = data["entities"]
    if args.smoke:
        counts = Counter()
        smoke_entities = []
        for entity in entities:
            topic = entity["topic"]
            if counts[topic] < 10:
                smoke_entities.append(entity)
                counts[topic] += 1
        entities = smoke_entities
    ids = [e["entity_id"] for e in entities]
    code = [Path(__file__), ROOT/"scripts/contextual_shared_concepts.py",
            ROOT/"scripts/corpus_graph_inference.py", ROOT/"src/cgraph/semantic_concepts.py"]
    identity = dict(stage=STAGE, smoke=args.smoke, inputs_sha256=data["hashes"], model_revision=REVISION,
                    model_files_sha256=data["model_files"], embedding=RECIPE | dict(
                        normalization="individual_l2", context_aggregation="mean_float64", dtype="float32",
                        threads=data["threads"], device="cpu", workers=1, context_template="{name}. {window}"),
                    pair_cosine=dict(dtype="float64", block=128, top_k=None),
                    windows=WINDOWS, thresholds=list(THRESHOLDS), rules=list(RULES), seed=SEED,
                    entity_ids_sha256=signature(ids), versions={p: version(p) for p in
                        ("torch", "numpy", "tokenizers", "transformers", "tqdm", "threadpoolctl", "matplotlib", "psutil")},
                    python=sys.version.split()[0], code_sha256={p.name: sha(p) for p in code})
    fingerprint = signature(identity)
    state = output/"identity.json"
    if state.exists():
        require(args.resume and read(state) == identity, "Existing output requires compatible --resume")
    else:
        atomic_json(output/"entities.json", entities)
        atomic_json(output/"git_source.json", dict(head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        atomic_json(state, identity)
    require(read(output/"entities.json") == entities, "Stored entity IDs or evidence differ")

    tokenizer = Tokenizer.from_file(str(args.model/"tokenizer.json"))
    geometry = {k: DocumentWindows(doc["text"]) for k, doc in data["documents"].items()}
    left, right = pair_indices([e["document_id"] for e in entities])
    name_values = np.asarray([data["names"][i] for i in ids])
    names_files = ("name_vectors.npy", "name_scores.npy")
    with threadpool_limits(limits=data["threads"], user_api="blas"):
        name_checkpoint = checked_output(output/"name.json", fingerprint, names_files)
        if name_checkpoint is None:
            since = time.perf_counter()
            name_scores = cosine_scores(name_values, left, right, "Name")
            atomic_array(output/"name_vectors.npy", name_values)
            atomic_array(output/"name_scores.npy", name_scores)
            name_checkpoint = complete(output/"name.json", fingerprint, names_files, distribution=stats(name_scores),
                                       seconds=time.perf_counter()-since, reused_entity_ids=len(ids),
                                       unique_inputs=len({e["name"] for e in entities}))
        else:
            name_scores = np.load(output/"name_scores.npy", mmap_mode="r", allow_pickle=False)
        name_domain = name_scores >= THRESHOLDS[0]
        guard = pair_guard_codes(entities, left, right, name_domain)
        measure = document_metrics(entities, left[name_domain], right[name_domain])

        with closing(sqlite3.connect(output/"embeddings.sqlite3")) as db:
            cache = EmbeddingCache(db, args.model, data["threads"], sampler, started)
            for window in args.windows:
                selected = (f"{window}_windows.json", f"{window}_vectors.npy", f"{window}_scores.npy")
                check = output/f"{window}.json"
                if checked_output(check, fingerprint, selected):
                    print(f"{window}: verified, restored", flush=True)
                    continue
                since = time.perf_counter()
                if (output/selected[0]).exists():
                    prepared = checked_output(output/f"{window}_extract.json", fingerprint, selected[:1])
                else:
                    prepared = None
                if prepared is None:
                    contexts, truncated = extract_windows(entities, data["documents"], geometry, tokenizer, window)
                    atomic_json(output/selected[0], contexts)
                    prepared = complete(output/f"{window}_extract.json", fingerprint, selected[:1],
                                        truncated_windows=truncated, extraction_seconds=time.perf_counter()-since)
                else:
                    contexts = read(output/selected[0])
                unique = dict.fromkeys(c["input"] for row in contexts for c in row["contexts"])
                vectors, usage = cache.run(unique, tokenizer, window)
                aggregation = time.perf_counter()
                means = np.asarray([mean_embeddings([vectors[c["input"]] for c in row["contexts"]])
                                    for row in tqdm(contexts, desc=f"{window} · aggregate", unit="entity")])
                atomic_array(output/selected[1], means)
                aggregation_seconds = time.perf_counter()-aggregation
                scoring = time.perf_counter()
                scores = cosine_scores(means, left, right, window)
                atomic_array(output/selected[2], scores)
                rows = threshold_sweep(window, name_scores, scores, name_domain, guard, measure, sampler)
                distribution = stats(scores)
                score_seconds = time.perf_counter()-scoring
                complete(check, fingerprint, selected, distribution=distribution, rows=rows, **usage,
                         window_occurrences=sum(len(c["provenance"]) for r in contexts for c in r["contexts"]),
                         entity_unique_windows=sum(len(r["contexts"]) for r in contexts),
                         truncated_windows=prepared["truncated_windows"], extraction_seconds=prepared["extraction_seconds"],
                         aggregation_seconds=aggregation_seconds, cosine_sweep_seconds=score_seconds,
                         seconds=prepared["extraction_seconds"]+usage["inference_seconds"]+aggregation_seconds+score_seconds,
                         peak_rss_bytes=sampler.peak)
                del scores, contexts, vectors, means
    return identity, name_checkpoint, entities, left, right


def main():
    parser = ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=list(WINDOWS))
    parser.add_argument("--smoke-id", default="smoke", help="Choose a separate smoke checkpoint directory")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    if not re.fullmatch(r"smoke(?:[-_][a-zA-Z0-9]+)*", args.smoke_id):
        parser.error("invalid --smoke-id")
    args.windows = [window for window in WINDOWS if window in args.windows]
    started = time.perf_counter()
    if not args.preflight:
        print("WARNING: LONG-RUNNING CORPUS100 EXPERIMENT. ONE LOCAL E5 CPU INFERENCE SESSION.", flush=True)
    with MemorySampler() as sampler:
        data = inputs(args.model)
        print(json.dumps(dict(preflight="PASS", **data["counts"], model_revision=REVISION,
                              verified_name_ids=len(data["names"]), diagnostics=len(data["diagnostic"]))), flush=True)
        if args.preflight:
            return
        output = OUTPUT/args.smoke_id if args.smoke else OUTPUT/"corpus100"
        output.mkdir(parents=True, exist_ok=True)
        with FileLock(output/"run.lock", timeout=0):
            identity, name, entities, left, right = execute(args, data, output, sampler, started)
            from contextual_shared_concepts_report import publish

            publish(args, data, identity, name, entities, left, right, output, REPORT, sampler, started)


if __name__ == "__main__":
    main()
