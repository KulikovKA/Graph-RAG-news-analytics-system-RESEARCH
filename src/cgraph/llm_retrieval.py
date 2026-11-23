"""Входы генерации из замороженного профиля C."""

from collections.abc import Callable
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "reports/retrieval/benchmark_3a5_9"
PROFILE_SHA = "4c4194a58c1d1e14282253d38880e834584ed5a0c1cce4cd36e3ee3724575867"
MATRIX_SHA = "e969774274026dc6ec010feadfc504d666ea708949a26de716d7e71daf5f8425"
REPLAY_INDEX_SHA = "f0a386c19076049d693018810f758b809965ad858a2bf91a9173322f3ef718e5"
SOURCE = ROOT / "reports/ner_v2/research_run2/input_abstracts.jsonl"
LOCK = ROOT / "reports/entity_resolution/relex_base_t035/dynamic_concept_graph/document_lock.json"
EDGE_FIELDS = ("key", "type", "relation_type", "document_id", "head_key", "tail_key", "traversed_reverse")
WINDOW_FIELDS = ("document_id", "revision_id", "text_sha256", "start", "end", "window_id")


class RetrievalError(ValueError):
    pass


def checked_bytes(path: Path, digest: str) -> bytes:
    content = path.read_bytes()
    if sha256(content).hexdigest() != digest:
        raise RetrievalError("Frozen input checksum mismatch: " + path.name)
    return content


def check_python_source(path: Path, expected_sha: str) -> None:
    # SHA профиля снят с Windows checkout. Git может сохранить тот же Python-файл с LF.
    original = path.read_bytes()
    lf = original.replace(b"\r\n", b"\n")
    candidates = original, lf, lf.replace(b"\n", b"\r\n")
    if all(sha256(content).hexdigest() != expected_sha for content in candidates):
        raise RetrievalError("Frozen Python source checksum mismatch: " + path.name)


def load_profile() -> dict:
    return json.loads(checked_bytes(STAGE / "PROFILE.json", PROFILE_SHA))


def load_corpus(profile: dict) -> dict[str, dict]:
    locked = json.loads(checked_bytes(LOCK, profile["document_lock_sha256"]))
    corpus = {doc["document_id"]: doc for doc in map(json.loads, checked_bytes(
        SOURCE, profile["corpus_sha256"]).splitlines())}
    if len(locked) != 30 or len(corpus) != 30 or set(corpus) != {doc["document_id"] for doc in locked}:
        raise RetrievalError("Frozen corpus membership mismatch")
    for metadata in locked:
        doc = corpus[metadata["document_id"]]
        digest = sha256(doc["text"].encode("utf-8")).hexdigest()
        if (doc["revision_id"] != "sha256:" + digest or doc["text_sha256"] != digest
                or any(doc[k] != metadata[k] for k in ("revision_id", "text_sha256", "title", "source_url"))):
            raise RetrievalError("Frozen document revision mismatch")
    return corpus


def frozen_replay(question: str, profile: dict) -> dict:
    directory = STAGE / "results/matrix"
    manifest = json.loads(checked_bytes(directory / "manifest.json", MATRIX_SHA))
    for key in ("graph_fingerprint", "snapshot_sha256"):
        if manifest[key] != profile[key]:
            raise RetrievalError("Replay graph identity mismatch")
    question_hash = sha256(json.dumps(question, ensure_ascii=False).encode("utf-8")).hexdigest()
    index = json.loads(checked_bytes(STAGE.parent / "benchmark_3a5_10/REPLAY_INDEX.json", REPLAY_INDEX_SHA))
    rows = [row for row in index if row["question_hash"] == question_hash
            and re.fullmatch(r"([ABC])/3A52-\1[0-9]{2}", row["key"])]
    if len(rows) != 1:
        raise RetrievalError("Replay requires an exact archived approved A/B/C question")
    row = rows[0]
    registry = json.loads(checked_bytes(directory / "evidence.json", manifest["files"]["evidence.json"]))
    return {
        "archive": {"key": row["key"], "manifest_sha256": MATRIX_SHA, "question_sha256": question_hash,
                    "replay_index_sha256": REPLAY_INDEX_SHA,
                    "historical_seconds": row["seconds"], "historical_upstream_seconds": row["upstream_seconds"]},
        "paths": [{"path_id": key, **registry[key]} for key in row["top"]],
    }


def project_ranked(ranked: list[dict]) -> list[dict]:
    paths = []
    for item in ranked:
        path = item["path"]
        facts = []
        for entry in item["evidence"]["edges"]:
            edge = entry["edge"]
            if edge["type"] != "RELEX_REL":
                continue
            if entry["status"] != "valid":
                raise RetrievalError("Selected path has invalid evidence")
            for source in entry["source_relations"]:
                facts.append({"document_id": edge["document_id"], "edge_key": edge["key"],
                              "source_triple_id": source["source_triple_id"],
                              "windows": [{k: window[k] for k in WINDOW_FIELDS}
                                          for window in source["windows"]]})
        paths.append({"path_id": path["path_id"], "document_ids": path["document_ids"],
                      "edges": [{k: edge[k] for k in EDGE_FIELDS} for edge in path["edges"]],
                      "factual_document_ids": sorted({fact["document_id"] for fact in facts}), "facts": facts})
    return paths


def retrieve_c(question: str, profile: dict, driver, database, timings: dict,
               *, analyze: Callable, link: Callable) -> dict:
    from .evidence_extraction import EvidenceExtractor
    from .graph_retrieval import GraphRetriever, TraversalLimits
    from .path_reranking import rerank_paths

    def measure(name, call, *args, **kwargs):
        started = perf_counter()
        try:
            return call(*args, **kwargs)
        finally:
            timings[name] = perf_counter() - started

    production = profile["production"]
    namespace = profile["namespace"]
    limits = TraversalLimits(**production["bfs"])
    analysis = measure("query_analysis", analyze, question)
    seeds = measure("entity_linking", link, analysis, production["linking"]["top_k"])[:limits.max_seeds]
    raw = measure("graph_retrieval", GraphRetriever(driver, database=database, namespace=namespace).retrieve,
                  seeds, limits, seed_fair=production["seed_fair"])
    if raw["graph_fingerprint"] != profile["graph_fingerprint"]:
        raise RetrievalError("Neo4j graph fingerprint mismatch")
    evidence = measure("evidence_extraction",
                       EvidenceExtractor(driver, database=database, namespace=namespace).extract, raw)
    ranked = measure("reranking", rerank_paths, question, raw, evidence, seeds,
                     policy=production["reranking"], top_k=production["rerank_top_k"])
    return {"paths": project_ranked(ranked["paths"]), "diagnostics": ranked["diagnostics"],
            "truncated": ranked["truncated"], "reason": ranked["reason"]}


class LiveRetrievalRuntime:
    """Владеет ресурсами одной последовательной сессии с профилем C."""

    def __init__(self, profile, driver, database, analyzer, linker, read_snapshot, set_seed):
        self.profile = profile
        self._driver, self._database = driver, database
        self._analyzer, self._linker = analyzer, linker
        self._read_snapshot, self._set_seed = read_snapshot, set_seed

    def __enter__(self):
        if self._driver is None:
            raise RetrievalError("Live runtime is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        driver = self._driver
        self._driver = self._analyzer = self._linker = None
        if driver is not None:
            driver.close()

    def retrieve(self, question: str, timings: dict) -> dict:
        if self._driver is None:
            raise RetrievalError("Live runtime is closed")

        started = perf_counter()
        before = self._read_snapshot(self._driver, self._database)
        if any(before[k] != self.profile[k] for k in ("graph_fingerprint", "snapshot_sha256")):
            raise RetrievalError("Live graph snapshot mismatch")
        timings["retrieval_preflight"] = perf_counter() - started
        self._set_seed(self.profile["seed"])
        result = retrieve_c(question, self.profile, self._driver, self._database, timings,
                            analyze=self._analyzer.analyze, link=self._linker.link_seeds)

        started = perf_counter()
        after = self._read_snapshot(self._driver, self._database)
        timings["retrieval_postflight"] = perf_counter() - started
        if before != after:
            raise RetrievalError("Live graph changed during retrieval")
        result["graph_verified"] = {k: after[k] for k in ("graph_fingerprint", "snapshot_sha256")}
        return result


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def open_live_runtime(profile: dict, *, neo4j_env: Path | None, entity_cache: Path,
                      gliner_path: Path) -> LiveRetrievalRuntime:
    search_path = sys.path[:]
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from build_relex_graphs import connect_neo4j
        from smoke_retrieval_3a1 import verify
    finally:
        sys.path[:] = search_path

    import torch
    from neo4j import READ_ACCESS
    from .entity_linking import EntityLinker, load_entity_cache
    from .query_analysis import E3_HASHES, MODEL_ID, E3Extractor, QueryAnalyzer

    for name, digest in profile["source_sha256"].items():
        check_python_source(ROOT / name, digest)
    versions = {"python": platform.python_version()}
    versions.update({name: version(name) for name in profile["versions"] if name != "python"})
    if versions != profile["versions"]:
        raise RetrievalError("Installed retrieval versions differ from PROFILE.json")
    gliner = profile["models"]["GLiNER"]
    if gliner["sha256"] != E3_HASHES or gliner["id"] != MODEL_ID:
        raise RetrievalError("GLiNER identity mismatch")
    cache = load_entity_cache(entity_cache)
    identity = profile["models"]["E5"]
    model_root = Path(cache.model["root"])
    hashes = {Path(row["path"]).relative_to(model_root).as_posix(): row["sha256"]
              for row in cache.model["files"]}
    if (cache.fingerprint != identity["entity_cache_fingerprint"]
            or cache.model["model"] != identity["id"] or hashes != identity["sha256"]):
        raise RetrievalError("E5 identity mismatch")

    production = profile["production"]
    analyzer = QueryAnalyzer(E3Extractor(gliner_path), threshold=production["gliner_threshold"])
    linker = EntityLinker(cache, namespace=profile["namespace"], **production["linking"])
    torch.set_num_threads(profile["cpu_threads"])

    def read_snapshot(driver, database):
        facts = {}
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            verify(session, facts)
        return facts

    driver, database = connect_neo4j(neo4j_env)
    return LiveRetrievalRuntime(profile, driver, database, analyzer, linker, read_snapshot, torch.manual_seed)


def live_retrieval(question: str, profile: dict, timings: dict, *, neo4j_env: Path | None,
                   entity_cache: Path, gliner_path: Path) -> dict:
    started = perf_counter()
    with open_live_runtime(profile, neo4j_env=neo4j_env, entity_cache=entity_cache, gliner_path=gliner_path) as runtime:
        setup = perf_counter() - started
        result = runtime.retrieve(question, timings)
        timings["retrieval_preflight"] += setup
        return result
