"""Compare A and pinned C on the historical holdout and single-document controls."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
sys.path.insert(0, str(BENCHMARK))

from run_block_c import audit, compact_paths
from run_document_aware import BASELINE, BASELINE_SHA256, GOLD, measure, path_output, replay_baseline, summarize
from validate_draft import documents, jsonl, require
from cgraph.evidence_extraction import EvidenceExtractor, _DOCUMENT_QUERY
from cgraph.graph_retrieval import GraphRetriever, TraversalLimits, _ADJACENCY_QUERY, _FINGERPRINT_QUERY, _NODE_QUERY
from cgraph.path_reranking import rerank_paths_a as rerank_paths
from document_aware_reranking import score_candidates, select_paths


SOURCE_COMMIT = "cd38cb4131109366adee4525bbbee01b714e9e0d"
POLICY_COMMIT = "123f7f4237a7ddc5cb6506f402d6044c4889d446"
POLICY_SHA256 = "ec077c482c3a4c15ee5a627f8efda4208b07cdd0a77b069325578603fbcd131b"
HOLDOUT_SHA256 = "cfddb313dfe97d989021a76f83136ff3f4dcb504450b8d3a6127cdd60622265f"
HOLDOUT = HERE / "holdout_original_c.jsonl"


def read_pinned(path, digest):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, f"Unexpected SHA-256: {path.name}")
    return raw


class FrozenSnapshot:
    """Read-only responses for the four production Cypher queries in this experiment."""

    def __init__(self, nodes, edges, fingerprint):
        self.nodes = nodes
        self.fingerprint = fingerprint
        self.documents = {n["properties"]["document_id"]: n["properties"]
                          for n in nodes.values() if n["label"] == "RELEXDocument"}
        self.adjacency = defaultdict(list)
        for _, e in sorted(edges.items()):
            if e["type"] == "RELEX_MENTIONS":
                continue
            h, t = nodes[e["head"]], nodes[e["tail"]]
            row = {"head": h["properties"], "head_labels": [h["label"]],
                   "tail": t["properties"], "tail_labels": [t["label"]],
                   "kind": e["type"], "properties": e["properties"]}
            for key in {e["head"], e["tail"]}:
                self.adjacency[key].append(row)

    def session(self, **options):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return None

    def execute_read(self, callback):
        return callback(self)

    def run(self, query, *, namespace, **parameters):
        require(namespace == audit.TARGET_NAMESPACE, "Unexpected namespace")
        if query == _FINGERPRINT_QUERY:
            return [{"fingerprint": self.fingerprint}]
        if query == _NODE_QUERY:
            node = self.nodes.get(parameters["key"])
            return [{"node": node["properties"], "labels": [node["label"]]}] if node else []
        if query == _ADJACENCY_QUERY:
            return self.adjacency[parameters["key"]][:parameters["limit"]]
        if query == _DOCUMENT_QUERY:
            return [{"document": self.documents[d]} for d in parameters["document_ids"] if d in self.documents]
        raise ValueError("Unsupported query in frozen snapshot")


def load_snapshot(corpus):
    stage = audit.stage
    frozen = json.loads((BENCHMARK.parents[2] / ".cache/semantic_concept_sweep/baseline.json").read_bytes())
    source_nodes, source_edges = stage.graph.unpack_snapshot(frozen)
    manifest, delta = stage.load_delta()
    nodes, edges, fingerprint = stage.build(source_nodes, source_edges, manifest, delta)
    require(fingerprint == audit.expected_fingerprint(), "Frozen graph fingerprint differs")
    require(stage.graph.snapshot_hash(nodes, edges) == audit.EXPECTED_SNAPSHOT_SHA256, "Frozen graph hash differs")
    snapshot = FrozenSnapshot(nodes, edges, fingerprint)
    require(snapshot.documents.keys() == corpus.keys(), "Frozen document membership differs")
    for key, doc in snapshot.documents.items():
        require(all(doc[f] == corpus[key][f] for f in ("text", "revision_id", "text_sha256")),
                "Frozen graph document differs")
    return snapshot


def compare(question, retrieval, bundles, seeds):
    start = perf_counter()
    a = rerank_paths(question, retrieval, bundles, seeds)
    a_seconds = perf_counter() - start
    start = perf_counter()
    candidates = score_candidates(question, retrieval, bundles, seeds)
    require(len(candidates) == a["ranking_diagnostics"]["eligible_paths"], "A/C eligibility differs")
    c, profiles = select_paths(candidates, "C")
    c_seconds = perf_counter() - start
    a_top, c_top = [path_output(p) for p in a["paths"]], [path_output(p) for p in c]
    reordered = {**retrieval, "paths": retrieval["paths"][::-1]}
    a_repeat = rerank_paths(question, reordered, bundles[::-1], seeds)
    c_repeat, profiles_repeat = select_paths(candidates[::-1], "C")
    require([path_output(p) for p in a_repeat["paths"]] == a_top, "A output changed on permutation")
    require([path_output(p) for p in c_repeat] == c_top and profiles_repeat == profiles,
            "C output changed on permutation")
    return {"A": {"top_5": a_top, "reranking_seconds": a_seconds},
            "C": {"top_5": c_top, "reranking_seconds": c_seconds, "document_profiles": profiles}}


def retrieval_row(key, question, retrieval, bundles, seeds):
    return {"question_id": key, "question": question, "candidate_count": len(retrieval["paths"]),
            "traversal_diagnostics": retrieval["diagnostics"], "deterministic_on_permutation": True,
            "policies": compare(question, retrieval, bundles, seeds)}


def evaluate(rows, references, groups):
    for row in rows:
        gold = references[row["question_id"]]
        # Индексы для метрики привязаны к исходным строкам и не изменяют holdout.
        gold = {**gold, "evidence": [{**e, "evidence_id": e.get("evidence_id", f"E{i:02d}")}
                                    for i, e in enumerate(gold["evidence"], 1)]}
        row.update(gold_documents=sorted(gold["document_contributions"]), gold_evidence_count=len(gold["evidence"]))
        for result in row["policies"].values():
            result["metrics"] = measure(result["top_5"], gold)
            factual = {d for p in result["top_5"] for d in p["factual_document_ids"]}
            result["out_of_reference_factual_documents"] = sorted(factual - set(row["gold_documents"]))
    return {group: {policy: {**summarize(items, policy),
                            "out_of_reference_factual_doc_occurrences": sum(
                                len(r["policies"][policy]["out_of_reference_factual_documents"]) for r in items),
                            "median_reranking_seconds": median(r["policies"][policy]["reranking_seconds"] for r in items)}
                    for policy in ("A", "C")} for group, items in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    policy_code = (BENCHMARK / "document_aware_reranking.py").read_bytes()
    original_import = policy_code.replace(b"rerank_paths_a as rerank_paths", b"rerank_paths", 1)
    require(hashlib.sha256(original_import).hexdigest() == POLICY_SHA256, "Historical policy code differs")
    raw = read_pinned(HOLDOUT, HOLDOUT_SHA256)
    baseline = json.loads(read_pinned(BASELINE, BASELINE_SHA256))
    read_pinned(GOLD, baseline["benchmark_input"]["sha256"])
    corpus = documents()
    snapshot = load_snapshot(corpus)
    limits = TraversalLimits()
    require(asdict(limits) == baseline["settings"]["traversal"], "Traversal configuration differs")
    retriever, extractor = GraphRetriever(snapshot), EvidenceExtractor(snapshot)
    node_properties = {key: n["properties"] for key, n in snapshot.nodes.items()}
    approved, new = [], []
    for saved in baseline["questions"]:
        seeds = saved["entity_linking"]["traversal_seeds"]
        retrieval = retriever.retrieve(seeds, limits)
        bundles = extractor.extract(retrieval)
        require({k: v for k, v in retrieval.items() if k != "paths"} == saved["traversal"],
                f"{saved['question_id']}: live traversal diagnostics differ")
        paths, facts = compact_paths(retrieval, bundles, node_properties)
        require(list(paths) == list(saved["candidate_paths"]) and paths == saved["candidate_paths"]
                and facts == saved["facts"], f"{saved['question_id']}: live candidates or provenance differ")
        replay_baseline(saved, retrieval, bundles, seeds)
        approved.append(retrieval_row(saved["question_id"], saved["question"], retrieval, bundles, seeds))
    print("PASS: 2,000 ordered live candidates, evidence, diagnostics and 50 A ranks reproduced", flush=True)

    if not args.parity_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from tqdm import tqdm
        from cgraph.entity_linking import EntityLinker, load_entity_cache
        from cgraph.query_analysis import QueryAnalyzer

        analyzer = QueryAnalyzer()
        settings = baseline["settings"]
        cache = load_entity_cache()
        require(analyzer.threshold == settings["query_threshold"] and cache.model == settings["e5"]
                and cache.fingerprint == baseline["preflight"]["entity_cache_fingerprint"], "Model configuration differs")
        linker = EntityLinker(cache, min_score=settings["experimental_link_min_score"])
        torch.manual_seed(settings["torch_seed"])
        torch.set_num_threads(settings["cpu_threads"])
        historical = [(r["question_id"], r["question"]) for r in map(json.loads, raw.splitlines())]
        controls = [(r["question_id"], r["question"]) for r in jsonl(GOLD) if r["category"] in {"A", "B"}]
        require(len(historical) == 10 and len(controls) == 15, "Unexpected evaluation scope")
        print("WARNING: SEQUENTIAL LOCAL GLINER AND E5 INFERENCE FOR 25 QUESTIONS.", flush=True)
        for key, question in tqdm(historical + controls, desc="Holdout and controls", unit="question"):
            times = {}
            start = perf_counter()
            query = analyzer.analyze(question)
            times["query_analysis"] = perf_counter() - start
            start = perf_counter()
            link = linker.link(query, top_k=settings["link_top_k"])
            times["entity_linking"] = perf_counter() - start
            seeds = link["seeds"][:limits.max_seeds]
            start = perf_counter()
            retrieval = retriever.retrieve(seeds, limits)
            times["snapshot_traversal"] = perf_counter() - start
            start = perf_counter()
            bundles = extractor.extract(retrieval)
            times["evidence_extraction"] = perf_counter() - start
            row = retrieval_row(key, question, retrieval, bundles, seeds)
            row["shared_seconds"] = times
            new.append(row)

    # Эталонные документы и окна передаются только расчёту метрик после отбора.
    gold = {r["question_id"]: r for r in jsonl(GOLD)}
    refs = {**gold, **{r["question_id"]: r for r in map(json.loads, raw.splitlines())}}
    summary = evaluate(approved, gold, {"approved_c_replay": approved})
    if new:
        summary.update(evaluate(new, refs, {"holdout": new[:10], "single_document": new[10:],
                                           "controls_A": new[10:18], "controls_B": new[18:]}))
    used_keys = {f["edge_key"] for r in approved + new for mode in r["policies"].values()
                 for p in mode["top_5"] for f in p["facts"]}
    # Компактный справочник исходного направления для ручной оценки фактов.
    fact_names = {}
    for rows in snapshot.adjacency.values():
        for e in rows:
            prop = e["properties"]
            if prop["key"] in used_keys:
                fact_names[prop["key"]] = [e["head"]["name"], prop["relation_type"], e["tail"]["name"]]
    report = {"mode": "frozen_snapshot", "source_commit": SOURCE_COMMIT, "holdout_sha256": HOLDOUT_SHA256,
              "policy_commit": POLICY_COMMIT, "policy_sha256": POLICY_SHA256,
              "graph_fingerprint": snapshot.fingerprint, "snapshot_sha256": audit.EXPECTED_SNAPSHOT_SHA256,
              "settings": baseline["settings"],
              "live_archive_parity": {"candidate_paths": 2000, "a_rank_positions": 50},
              "summary": summary, "approved_replay": approved, "evaluation": new, "fact_names": fact_names}
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(report, out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
