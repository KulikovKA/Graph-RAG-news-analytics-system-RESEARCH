"""Stage 3A.5.6 on a verified read-only Neo4j snapshot."""

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from neo4j import READ_ACCESS
from tqdm import tqdm

from run_block_c import HERE, audit, compact_paths
from cgraph.evidence_extraction import extract_evidence
from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
from cgraph.path_reranking import rerank_paths


class ResearchLimits(TraversalLimits):
    def __post_init__(self):
        if type(self.max_hops) is int and 6 <= self.max_hops <= 15:
            TraversalLimits(**{**asdict(self), "max_hops": 5})
        else:
            super().__post_init__()


class Snapshot:
    def __init__(self, session):
        graph = audit.stage.graph
        self.records = {r["key"]: r for r in session.run(graph.READ_NODES, ns=audit.TARGET_NAMESPACE)}
        self.relationships = session.run(graph.READ_EDGES, ns=audit.TARGET_NAMESPACE).data()
        self.nodes = {k: r["properties"] for k, r in self.records.items()}
        self.documents = {n["document_id"]: n for k, n in self.nodes.items()
                          if "RELEXDocument" in self.records[k]["labels"]}
        self.adjacency = defaultdict(list)
        for r in self.relationships:
            if r["type"] not in {"RELEX_REL", "SIMILAR_TO", "HAS_SHARED_CONCEPT"}:
                continue
            h, t = self.records[r["head"]], self.records[r["tail"]]
            row = dict(head=h["properties"], head_labels=h["labels"], tail=t["properties"],
                       tail_labels=t["labels"], kind=r["type"], properties=r["properties"])
            for key in {r["head"], r["tail"]}:
                self.adjacency[key].append(row)
        for rows in self.adjacency.values():
            rows.sort(key=lambda row: row["properties"]["key"])

    def run(self, query, **params):
        if "AS fingerprint" in query:
            return [{"fingerprint": next(iter(self.nodes.values()))["fingerprint"]}]
        key = params["key"]
        if "AS node" in query:
            self.seed_reads.append(key)
            return [{"node": self.nodes[key], "labels": self.records[key]["labels"]}]
        self.expanded.append(key)
        rows = self.adjacency[key][:params["limit"]]
        self.examined_edges.update(r["properties"]["key"] for r in rows)
        self.examined_nodes.update(n["key"] for r in rows for n in (r["head"], r["tail"]))
        return rows


def evaluate(snapshot, row, limits, fair=False, cutoff=.85, reverse=False):
    seeds = [{k: v for k, v in seed.items() if k != "rank"}
             for seed in row["entity_linking"]["candidates"] if seed["score"] >= cutoff][:limits.max_seeds]
    if cutoff == .85:
        assert seeds == row["entity_linking"]["seeds"][:limits.max_seeds]
    if reverse:
        seeds.reverse()
    snapshot.seed_reads, snapshot.expanded = [], []
    snapshot.examined_nodes, snapshot.examined_edges = set(), set()
    unique = {}
    for seed in seeds:
        unique.setdefault(seed["node_key"], seed)
    start = perf_counter()
    result = GraphRetriever(None)._retrieve(snapshot, list(unique.values()), limits, seed_fair=fair)
    traversal = perf_counter() - start
    evidence = extract_evidence(result["paths"], snapshot.documents,
                                graph_fingerprint=result["graph_fingerprint"])
    ranking = rerank_paths(row["question"], result, evidence, seeds)
    paths, _ = compact_paths(result, evidence, snapshot.nodes)
    gold = set(row["gold_docs"])
    pool = {d for p in paths.values() for d in p["document_ids"]}
    factual = {d for p in paths.values() for d in p["factual_documents"]}
    top = {d for r in ranking["paths"] for d in paths[r["path_id"]]["document_ids"]}
    per_seed = Counter(p["seed_key"] for p in paths.values())
    return dict(question_id=row["question_id"], gold_candidate_docs=sorted(pool & gold),
                factual_gold_candidate_docs=sorted(factual & gold), gold_top5_docs=sorted(top & gold),
                candidate_recall=len(pool & gold) / len(gold),
                factual_candidate_recall=len(factual & gold) / len(gold), top5_recall=len(top & gold) / len(gold),
                paths=len(paths), factual_paths=sum(bool(p["factual_documents"]) for p in paths.values()),
                nonfactual_paths=sum(not p["factual_documents"] for p in paths.values()),
                gold_factual_paths=sum(bool(gold.intersection(p["factual_documents"])) for p in paths.values()),
                outside_gold_paths=sum(not gold.intersection(p["document_ids"]) for p in paths.values()),
                cross_doc_factual=sum(p["cross_doc_factual"] for p in paths.values()),
                transitions=sum(len(p["transitions"]) for p in paths.values()),
                seconds=traversal, total_seconds=perf_counter() - start, truncated=result["truncated"],
                **result["diagnostics"], seed_reads=snapshot.seed_reads,
                root_adjacency_observed=sorted(set(snapshot.seed_reads) & set(snapshot.expanded)),
                paths_by_seed={k: per_seed[k] for k in snapshot.seed_reads},
                unique_nodes_examined=len(snapshot.examined_nodes | set(snapshot.seed_reads)),
                unique_edges_examined=len(snapshot.examined_edges),
                unique_path_nodes=len({n for p in paths.values() for n in p["node_keys"]}),
                unique_path_edges=len({e["key"] for p in paths.values() for e in p["edges"]}),
                gold_factual_path_ids=[k for k, p in paths.items() if gold.intersection(p["factual_documents"])],
                path_ids=list(paths), top5_path_ids=[r["path_id"] for r in ranking["paths"]])


def path_digest(ids):
    return hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()


def validate_bolt(driver, database, rows, snapshot_report):
    print("WARNING: SEQUENTIAL READ-ONLY BOLT LATENCY CHECK.", flush=True)
    retriever = GraphRetriever(driver, database=database)
    output = {"A-200": [], "B-200": [], "A-100": []}
    cases = (("A-200", "A-200-original", 200, False),
             ("B-200", "B-200-original", 200, True),
             ("A-100", "max_candidates=100", 100, False))
    for index, row in enumerate(tqdm(rows, desc="Bolt")):
        seeds = row["entity_linking"]["seeds"][:32]
        for mode, case, cap, fair in cases:
            expected = snapshot_report["runs"][case]["questions"][index]
            start = perf_counter()
            result = retriever.retrieve(seeds, TraversalLimits(max_candidates=cap), seed_fair=fair)
            seconds = perf_counter() - start
            ids = [p["path_id"] for p in result["paths"]]
            assert path_digest(ids) == expected["path_ids_sha256"], row["question_id"]
            assert all(result["diagnostics"][key] == expected[key]
                       for key in result["diagnostics"]), row["question_id"]
            output[mode].append(dict(question_id=row["question_id"], seconds=seconds,
                                     diagnostics=result["diagnostics"]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bolt-reference", type=Path, help="Verify snapshot parity and time A/B at 200 and A at 100 over Bolt")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    frozen_linking = (HERE / "block_c_retrieval.json").read_bytes()
    old = json.loads(frozen_linking)
    rows = old["questions"]
    base = ResearchLimits(max_candidates=200)
    report = {"snapshot": {}, "runs": {}, "baseline_settings": asdict(base),
              "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "linking_sha256": hashlib.sha256(frozen_linking).hexdigest()}
    print("WARNING: READ-ONLY NEO4J SNAPSHOT VERIFICATION AND SEQUENTIAL BFS SWEEP.", flush=True)
    driver, database = audit.stage.graph.base.core.connect_neo4j()
    with driver, driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        audit.verify(session, report["snapshot"])
        if args.bolt_reference:
            reference = json.loads(args.bolt_reference.read_bytes())
            assert report["snapshot"]["snapshot_sha256"] == reference["snapshot"]["snapshot_sha256"]
            timings = validate_bolt(driver, database, rows, reference)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(timings, output, indent=2)
            return
        snapshot = Snapshot(session)

    cache = {}

    def run(name, limits, fair=False, cutoff=.85, reverse=False):
        key = (limits, fair, cutoff, reverse)
        if key not in cache:
            cache[key] = [evaluate(snapshot, row, limits, fair, cutoff, reverse)
                     for row in tqdm(rows, desc=name, leave=False)]
        questions = cache[key]
        summary = {field: mean(q[field] for q in questions) for field in
                   ("candidate_recall", "factual_candidate_recall", "top5_recall", "truncated")}
        summary.update({field: sum(q[field] for q in questions) for field in
                        ("paths", "factual_paths", "gold_factual_paths", "outside_gold_paths",
                         "cross_doc_factual", "nodes_examined", "relationships_examined")})
        summary.update(gold_docs=sum(len(q["gold_candidate_docs"]) for q in questions),
                       median_seconds=median(q["seconds"] for q in questions),
                       median_total_seconds=median(q["total_seconds"] for q in questions),
                       limits_hit=dict(Counter(hit for q in questions for hit in q["limits_hit"])))
        report["runs"][name] = dict(limits=asdict(limits), fair=fair, cutoff=cutoff, reverse=reverse,
                                    summary=summary, questions=questions)
        return summary

    run("baseline", base)
    for q, prior in zip(report["runs"]["baseline"]["questions"], rows):
        assert q["path_ids"] == list(prior["candidate_paths"]), (q["question_id"], next(((i, a, b) for i, (a, b) in enumerate(zip(q["path_ids"], prior["candidate_paths"])) if a != b), None))
        assert q["top5_path_ids"] == [p["path_id"] for p in prior["top_5"]], q["question_id"]
    sweep = {"max_candidates": [100, 200, 300, 500, 800, 1000], "max_neighbors": [20, 50, 100],
             "max_relationships": [500, 1000, 2000], "max_seeds": [8, 16, 32], "max_hops": range(2, 16)}
    for field, values in sweep.items():
        for value in values:
            if value != getattr(base, field):
                run(f"{field}={value}", replace(base, **{field: value}))

    def quality(summary):
        return (summary["factual_candidate_recall"], summary["candidate_recall"], summary["top5_recall"],
                -summary["relationships_examined"], -summary["paths"])

    options = [r for name, r in report["runs"].items()
               if name.startswith("max_candidates=") and r["limits"]["max_candidates"] > 200]
    best = ResearchLimits(**max(options, key=lambda r: quality(r["summary"]))["limits"])
    report["comparison_budget"] = best.max_candidates
    for limits in (base, best):
        for fair in (False, True):
            for reverse in (False, True):
                run(f"{'B' if fair else 'A'}-{limits.max_candidates}-{'reverse' if reverse else 'original'}",
                    limits, fair, reverse=reverse)
    shortlist = [r for r in report["runs"].values() if r["limits"]["max_hops"] <= 5 and not r["reverse"]]
    chosen = max(shortlist, key=lambda r: quality(r["summary"]))
    for cutoff in (.75, .80, .85, .90, .95):
        run(f"threshold-{cutoff}-baseline", base, cutoff=cutoff)
        run(f"threshold-{cutoff}-selected", ResearchLimits(**chosen["limits"]), chosen["fair"], cutoff=cutoff)
    report["selected"] = {k: chosen[k] for k in ("limits", "fair", "summary")}
    key = next(k for k, n in snapshot.nodes.items() if n.get("document_id") == "W2084709926"
               and n.get("name", "").casefold() == "coherent fiber optic bundle")
    report["c09"] = dict(seed_key=key, incident=[{**r, "properties": {k: v for k, v in r["properties"].items()
                                                          if k in {"resolution_decision_2f2", "is_self_loop"}}}
                                    for r in snapshot.relationships if key in (r["head"], r["tail"])],
                         document_factual_edges=[r["key"] for r in snapshot.relationships
                                                 if r["type"] == "RELEX_REL"
                                                 and r["properties"].get("document_id") == "W2084709926"])
    baseline_questions = report["runs"]["baseline"]["questions"]
    larger_questions = report["runs"][f"max_candidates={best.max_candidates}"]["questions"]
    report["gold_factual_paths_added_at_comparison_budget"] = {
        a["question_id"]: len(set(b["gold_factual_path_ids"]) - set(a["gold_factual_path_ids"]))
        for a, b in zip(baseline_questions, larger_questions)}
    for run in report["runs"].values():
        for question in run["questions"]:
            if "path_ids" in question:
                question["path_ids_sha256"] = path_digest(question.pop("path_ids"))
                question["gold_factual_path_ids_sha256"] = path_digest(question.pop("gold_factual_path_ids"))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=1, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
