"""Проверка и чтение замороженного пула B из Stage 3A.5.7."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LEGACY = HERE.parent / "benchmark_3a5_2"
DEFAULT_SOURCE = LEGACY / "integrated_runs/20261122T191609-112996Z"
DEFAULT_NAMES = ROOT / ".cache/semantic_concept_sweep/representations.json"
NAMES_SHA256 = "5f045f6bdc413544d473999d932c45f988d9de82a4c8c1f6a97ef8a245176390"
ORACLE_BUDGETS = (20, 50, 100, 200)
RECOVERY = (
    "Restore the original Stage 3A.5.7 cache, or explicitly rebuild B into a new directory: "
    r"& .\.venv\Scripts\python.exe .\reports\retrieval\benchmark_3a5_2\run_integrated_benchmark.py "
    r"--variants B --output-dir $PoolDir --neo4j-env $Neo4jEnv. "
    "Then pass --source-run $PoolDir. See README for snapshot and checksum requirements."
)

sys.path[:0] = [str(LEGACY), str(ROOT / "src")]

from cgraph.evidence_extraction import extract_evidence
from cgraph.path_reranking import _evidence_error, rerank_paths
from cgraph.retrieval_types import TARGET_NAMESPACE
from integrated_benchmark_reporting import encode_json
from run_integrated_benchmark import DATASET_HASHES, VARIANTS, digest, load_questions, read_json, read_record
from validate_draft import LOCK, SOURCE, documents, jsonl


def file_sha(path):
    checksum = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


@dataclass
class Pool:
    question: object
    raw: dict
    bundles: dict
    seeds: list
    baseline: list
    names: dict
    source_files: dict
    order: list | None = None
    c_scores: dict | None = None
    preselection_seconds: float | None = None
    baseline_seconds: float | None = None


def stage_value(directory, row, name, key):
    trace = row["timings"][name]
    expected = f"cache/{name}/{key}.json"
    if trace["file"] != expected or trace["key"] != key:
        raise ValueError(f"{row['key']} has unexpected {name} cache reference")
    path = directory / expected
    record = read_record(path, key)
    if record is None:
        status = "missing" if not path.is_file() else "invalid or corrupt"
        raise ValueError(f"{row['key']} {name} cache is {status}: {path}. {RECOVERY}")
    return record["value"], expected


def verify_bundles(raw, bundles, corpus):
    paths = raw["paths"]
    if len(bundles) != len(paths) or any(
        bundle["path_id"] != path["path_id"] for path, bundle in zip(paths, bundles, strict=True)
    ):
        raise ValueError("Missing, extra or misordered evidence bundles")
    graph_docs = {}
    for bundle in bundles:
        for doc in bundle["documents"]:
            doc_id = doc["document_id"]
            source = corpus[doc_id]
            if any(doc.get(key) != source.get(key)
                   for key in ("text", "revision_id", "text_sha256", "source_url", "title", "doi")):
                raise ValueError(f"Evidence document metadata differs from frozen source: {doc_id}")
            if graph_docs.setdefault(doc_id, doc) != doc:
                raise ValueError(f"Conflicting evidence document properties: {doc_id}")
    regenerated = extract_evidence(paths, graph_docs, namespace=raw["namespace"],
                                   graph_fingerprint=raw["graph_fingerprint"])
    if regenerated != bundles:
        raise ValueError("Evidence differs from source-exact extraction")
    return {bundle["path_id"]: bundle for bundle in bundles}


def load_pools(directory, names_path, smoke=False, notify=print):
    anchor = read_json(HERE / "FROZEN_B.json")
    if not paths_exist(directory, ("config.json", "manifest.json", "per_question.jsonl", "results.json")):
        raise ValueError(f"Incomplete Stage 3A.5.7 run at {directory}. {RECOVERY}")
    config = read_json(directory / "config.json")
    manifest = read_json(directory / "manifest.json")
    identity = config["identity"]
    graph = identity["graph"]
    if (config["stage"] != "3A.5.7" or config["schema_version"] != 1
            or config["variants"]["B"] != VARIANTS["B"] or "B" not in config["selection"]["variants"]
            or manifest["status"] != "COMPLETE" or manifest["fingerprint"] != digest(config)
            or read_json(directory / "results.json")["manifest"] != manifest
            or identity["dataset_sha256"] != DATASET_HASHES or identity["corpus_sha256"] != file_sha(SOURCE)
            or identity["document_lock_sha256"] != file_sha(LOCK)
            or identity["models"] != anchor["models"] or graph["namespace"] != TARGET_NAMESPACE
            or any(graph[key] != anchor[key] for key in ("snapshot_sha256", "graph_fingerprint"))):
        raise ValueError("Source run differs from the frozen Stage 3A.5.7 variant B protocol")
    corpus = documents()
    questions, references = load_questions(corpus)
    if config["question_keys"] != [q.key for q in questions]:
        raise ValueError("Source must contain all 35 questions in frozen order")
    if not names_path.is_file() or file_sha(names_path) != NAMES_SHA256:
        raise ValueError(f"Restore frozen entity representations at {names_path} or use --entity-names")
    names = {f"{TARGET_NAMESPACE}:{r['entity_id']}": (r["document_id"], r["text_name"])
             for r in read_json(names_path)["entities"]}
    rows = [row for row in jsonl(directory / "per_question.jsonl") if row["variant"] == "B"]
    if [row["key"] for row in rows] != [q.key for q in questions]:
        raise ValueError("Source run must have exactly one completed B row per frozen key")
    if smoke:
        selected = {"A/3A52-A01", "C/3A52-C03"}
        pairs = [(q, row) for q, row in zip(questions, rows, strict=True) if q.key in selected]
    else:
        pairs = zip(questions, rows, strict=True)
    variant = VARIANTS["B"]
    fingerprint = manifest["fingerprint"]
    pools = []
    for q, row in pairs:
        if row["question"] != q.text or row["status"] != "COMPLETE":
            raise ValueError(f"Unexpected question or status in B row {q.key}")
        seeds = row["linking"]["seeds"]
        physical = dict((s["node_key"], s["document_id"]) for s in seeds)
        query_hash = digest(q.text)
        bfs_key = digest([fingerprint, "bfs", [query_hash, list(physical.items()), variant["bfs"], False]])
        raw, bfs_file = stage_value(directory, row, "bfs", bfs_key)
        evidence_key = digest([fingerprint, "evidence", [
            raw["namespace"], raw["graph_fingerprint"], digest(raw["paths"])]])
        bundles, evidence_file = stage_value(directory, row, "evidence", evidence_key)
        rerank_key = digest([fingerprint, "rerank", [query_hash, bfs_key, evidence_key, seeds, "C", 5]])
        ranking, rerank_file = stage_value(directory, row, "rerank", rerank_key)
        baseline = ranking["paths"]
        signature = digest([q.text, seeds, raw["paths"], bundles, baseline])
        if (signature != anchor["questions"][q.key] or len(raw["paths"]) != row["candidate_count"]
                or raw["graph_fingerprint"] != anchor["graph_fingerprint"] or raw["namespace"] != TARGET_NAMESPACE
                or [path["path_id"] for path in baseline] != [path["path_id"] for path in row["top_5"]]):
            raise ValueError(f"{q.key} differs from the original frozen B inputs or baseline")
        validated = verify_bundles(raw, bundles, corpus)
        pools.append(Pool(q, raw, validated, seeds, baseline, names,
                          {"bfs": bfs_file, "evidence": evidence_file, "rerank": rerank_file}))
    notify(f"Verified B cache | {len(pools)} questions | {sum(len(p.raw['paths']) for p in pools)} raw paths")
    return pools, references, {
        "stage_3a5_7": fingerprint, "variant": "B", "snapshot_sha256": anchor["snapshot_sha256"],
        "graph_fingerprint": anchor["graph_fingerprint"], "protocol": variant,
        "corpus_sha256": identity["corpus_sha256"], "dataset_sha256": DATASET_HASHES,
        "entity_names_sha256": NAMES_SHA256,
        "questions": {p.question.key: {"frozen_sha256": anchor["questions"][p.question.key],
                                        "retrieval_sha256": digest(p.raw)}
                      for p in pools},
    }


def paths_exist(directory, files):
    return all((directory / file).is_file() for file in files)


def prepare(pool):
    started = perf_counter()
    question = pool.question.text
    baseline = rerank_paths(question, pool.raw, pool.bundles.values(), pool.seeds, policy="C")["paths"]
    pool.baseline_seconds = perf_counter() - started
    if baseline != pool.baseline:
        raise ValueError(f"{pool.question.key}: current C differs from frozen C")
    # Сохраняем префикс production C. Остаток упорядочиваем по исходному base_score C.
    scores = {item["path_id"]: item["base_score"] for item in baseline}
    prefix = list(scores)
    for path in pool.raw["paths"]:
        pid = path["path_id"]
        bundle = pool.bundles[pid]
        if pid in scores or _evidence_error(path, bundle, pool.raw["graph_fingerprint"]):
            continue
        singleton = rerank_paths(question, {**pool.raw, "paths": [path]}, [bundle],
                                 pool.seeds, policy="C", top_k=1)["paths"]
        if singleton:
            scores[pid] = singleton[0]["base_score"]
    pool.order = prefix + sorted(scores.keys() - set(prefix), key=lambda pid: (-scores[pid], pid))
    pool.c_scores = scores
    pool.preselection_seconds = perf_counter() - started


def render(path, bundle, names, representation):
    facts, documents_by_id, transitions = [], {}, []
    for i, item in enumerate(bundle["edges"], 1):
        edge = item["edge"]
        if edge["type"] != "RELEX_REL":
            transitions.append(f"Step {i}: {edge['type']} (navigation only; no factual assertion).")
            continue
        doc = edge["document_id"]
        head, tail = (names[edge[key]] for key in ("head_key", "tail_key"))
        if head[0] != doc or tail[0] != doc:
            raise ValueError(f"{path['path_id']}: entity name belongs to another document")
        direction = " (traversed in reverse)" if edge["traversed_reverse"] else ""
        fact = f"Step {i}{direction} [document {doc}]: {head[1]} | {edge['relation_type']} | {tail[1]}"
        facts.append(fact)
        doc_facts, windows = documents_by_id.setdefault(doc, ([], {}))
        doc_facts.append(fact)
        for source in item["source_relations"]:
            for w in source["windows"]:
                locator = (w["start"], w["end"], w["window_id"])
                windows[locator] = w["text"]
    evidence = {doc: [f"Evidence from {doc} [{start}:{end}]\n{text}"
                      for (start, end, _), text in sorted(windows.items())]
                for doc, (_, windows) in sorted(documents_by_id.items())}
    if representation == "document_aware":
        groups = ["\n".join([f"Document {doc}", *doc_facts, *evidence[doc]])
                  for doc, (doc_facts, _) in sorted(documents_by_id.items())]
        return "\n".join(transitions + groups)
    return "\n".join(transitions + facts + [text for windows in evidence.values() for text in windows])
