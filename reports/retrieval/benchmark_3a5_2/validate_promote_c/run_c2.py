"""Compare A/C/C2 on the unchanged Stage 3A.5.3 candidate pools."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import run_validation as previous
from cgraph.path_reranking import DEFAULT_RERANKING_POLICY, rerank_paths
from run_document_aware import summarize


HERE = Path(__file__).resolve().parent
POLICIES = ("A", "C", "C2")


def compact_ranked(ranking):
    return [{**previous.path_output(p), "metrics": p["metrics"]} for p in ranking["paths"]]


def without_metrics(paths):
    return [{k: v for k, v in p.items() if k != "metrics"} for p in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-production", type=Path, help="Verify default on approved C and archived seeds")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    baseline = json.loads(previous.read_pinned(previous.BASELINE, previous.BASELINE_SHA256))
    raw = previous.read_pinned(previous.HOLDOUT, previous.HOLDOUT_SHA256)
    previous.read_pinned(previous.GOLD, baseline["benchmark_input"]["sha256"])
    snapshot = previous.load_snapshot(previous.documents())
    limits = previous.TraversalLimits()
    previous.require(previous.asdict(limits) == baseline["settings"]["traversal"], "Traversal limits differ")
    nodes = {k: n["properties"] for k, n in snapshot.nodes.items()}
    retriever, extractor = previous.GraphRetriever(snapshot), previous.EvidenceExtractor(snapshot)
    approved, evaluation = [], []
    fact_names = {}
    past = json.loads((HERE / "validation_results.json").read_bytes())

    def run(key, question, seeds, saved, production=None):
        retrieval = retriever.retrieve(seeds, limits)
        bundles = extractor.extract(retrieval)
        previous.require(len(retrieval["paths"]) == saved["candidate_count"]
                         and retrieval["diagnostics"] == saved["traversal_diagnostics"],
                         f"{key}: candidate pool differs")
        ordered = {**retrieval, "paths": retrieval["paths"][::-1]}
        policies = {}
        for policy in ("production", "fallback_A") if production is not None else POLICIES:
            selection = {"policy": "A"} if policy == "fallback_A" else ({} if policy == "production"
                                                                       else {"policy": policy})
            context = {"entity_nodes": nodes} if policy == "C2" else {}
            start = perf_counter()
            result = rerank_paths(question, retrieval, bundles, seeds, **selection, **context)
            elapsed = perf_counter() - start
            top = compact_ranked(result)
            repeated = rerank_paths(question, ordered, bundles[::-1], seeds, **selection, **context)
            previous.require(result == repeated, f"{key}: {policy} changed under permutation")
            if production is not None:
                previous.require(top == production[policy], f"{key}: {policy} differs from offline")
            elif policy in ("A", "C"):
                previous.require(without_metrics(top) == saved["policies"][policy]["top_5"],
                                 f"{key}: {policy} differs from Stage 3A.5.3")
            policies[policy] = {"top_5": top, "reranking_seconds": elapsed}
        return {"question_id": key, "question": question, "candidate_count": len(retrieval["paths"]),
                "paths_sha256": hashlib.sha256("\n".join(p["path_id"] for p in retrieval["paths"]).encode()).hexdigest(),
                "traversal_diagnostics": retrieval["diagnostics"], "policies": policies}

    offline = None
    if args.verify_production:
        offline = json.loads(args.verify_production.read_bytes())
        previous.require(DEFAULT_RERANKING_POLICY in ("C", "C2"), "Production still defaults to A")
    for i, (source, saved) in enumerate(zip(baseline["questions"], past["approved_replay"], strict=True)):
        production = None
        if offline:
            saved_policies = offline["approved"][i]["policies"]
            winner = DEFAULT_RERANKING_POLICY
            production = {"production": saved_policies[winner]["top_5"], "fallback_A": saved_policies["A"]["top_5"]}
        row = run(source["question_id"], source["question"], source["entity_linking"]["traversal_seeds"], saved, production)
        previous.require(row["paths_sha256"] == hashlib.sha256("\n".join(source["candidate_paths"]).encode()).hexdigest(),
                         f"{source['question_id']}: archived path IDs differ")
        approved.append(row)
    if not offline:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        from tqdm import tqdm
        from cgraph.query_analysis import QueryAnalyzer
        from cgraph.entity_linking import EntityLinker, load_entity_cache

        settings = baseline["settings"]
        analyzer, cache = QueryAnalyzer(), load_entity_cache()
        previous.require(analyzer.threshold == settings["query_threshold"] and cache.model == settings["e5"]
                         and cache.fingerprint == baseline["preflight"]["entity_cache_fingerprint"], "Model settings differ")
        linker = EntityLinker(cache, min_score=settings["experimental_link_min_score"])
        torch.manual_seed(settings["torch_seed"])
        torch.set_num_threads(settings["cpu_threads"])
        historical = [(r["question_id"], r["question"]) for r in map(json.loads, raw.splitlines())]
        controls = [(r["question_id"], r["question"]) for r in previous.jsonl(previous.GOLD) if r["category"] in {"A", "B"}]
        print("WARNING: SEQUENTIAL LOCAL GLINER AND E5 FOR THE 25 VALIDATION QUERIES.", flush=True)
        for (key, question), saved in tqdm(zip(historical + controls, past["evaluation"], strict=True), total=25,
                                          desc="A/C/C2 comparison", unit="question"):
            previous.require((key, question) == (saved["question_id"], saved["question"]), "Question identity differs")
            query = analyzer.analyze(question)
            link = linker.link(query, top_k=settings["link_top_k"])
            evaluation.append(run(key, question, link["seeds"][:limits.max_seeds], saved))

    # Эталоны участвуют только в оценке после формирования выдач.
    gold = {r["question_id"]: r for r in previous.jsonl(previous.GOLD)}
    refs = {**gold, **{r["question_id"]: r for r in map(json.loads, raw.splitlines())}}
    modes = ("production", "fallback_A") if offline else POLICIES
    for rows, references in ((approved, gold), (evaluation, refs)):
        for row in rows:
            r = references[row["question_id"]]
            expected = {**r, "evidence": [{**e, "evidence_id": e.get("evidence_id", f"E{i:02d}")}
                                          for i, e in enumerate(r["evidence"], 1)]}
            row.update(gold_documents=sorted(r["document_contributions"]), gold_evidence_count=len(r["evidence"]))
            for result in row["policies"].values():
                result["metrics"] = previous.measure(result["top_5"], expected)
                for path in result["top_5"]:
                    for f in path["facts"]:
                        fact_names[f["edge_key"]] = None
    groups = {"approved": approved}
    if evaluation:
        groups.update(holdout=evaluation[:10], controls_A=evaluation[10:18], controls_B=evaluation[18:])
    summary = {}
    for name, rows in groups.items():
        summary[name] = {}
        for mode in modes:
            policy = {**summarize(rows, mode), "median_reranking_seconds":
                      median(r["policies"][mode]["reranking_seconds"] for r in rows)}
            off_ref = 0
            for row in rows:
                paths = row["policies"][mode]["top_5"]
                docs = {d for p in paths for d in p["factual_document_ids"]}
                off_ref += len(docs - set(row["gold_documents"]))
            policy["out_of_reference_factual_documents"] = off_ref
            summary[name][mode] = policy
    for rows in snapshot.adjacency.values():
        for e in rows:
            key = e["properties"]["key"]
            if key in fact_names:
                fact_names[key] = [e["head"]["name"], e["properties"]["relation_type"], e["tail"]["name"]]
    previous.require(all(fact_names.values()), "Missing fact endpoints")
    report = {"snapshot_sha256": previous.audit.EXPECTED_SNAPSHOT_SHA256, "fingerprint": snapshot.fingerprint,
              "default_policy": DEFAULT_RERANKING_POLICY if offline else None,
              "production_parity": bool(offline), "summary": summary, "approved": approved,
              "evaluation": evaluation, "fact_names": fact_names}
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(report, out, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        out.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
