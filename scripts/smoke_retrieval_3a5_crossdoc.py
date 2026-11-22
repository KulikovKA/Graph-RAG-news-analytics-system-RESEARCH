"""Run targeted and real question-driven read-only smoke tests on frozen Stage 2G.3."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

from smoke_retrieval_3a5 import ROOT, audit, failure
from cgraph.evidence_extraction import EvidenceExtractor
from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
from cgraph.path_reranking import rerank_paths_a as rerank_paths
from cgraph.retrieval_types import SUSPICIOUS_CELL_PAIRS, TARGET_NAMESPACE as NS


SEMANTIC = "(b:RELEXEntity)-[s:SIMILAR_TO]->(c:RELEXEntity)"
CONCEPT = "(b:RELEXEntity)-[m:HAS_SHARED_CONCEPT]->(k:RELEXConcept)<-[n:HAS_SHARED_CONCEPT]-(c:RELEXEntity)"
BRIDGES = {
    "semantic": (SEMANTIC, "[b,c]", "[s]", "s.resolution_decision_2f2 = 'RELATED'"),
    "concept": (CONCEPT, "[b,k,c]", "[m,n]", """
        all(link IN [m,n] WHERE coalesce(link.requires_review,false) = false
            AND NOT toUpper(coalesce(link.review_status,'')) IN ['REVIEW','REJECT','SUSPICIOUS']
            AND link.factual = false AND link.identity = false AND link.fact_inheritance_allowed = false)
        AND all(pair IN [[b,m],[c,n]] WHERE
            (pair[1].document_id IS NOT NULL OR pair[1].document_ids IS NOT NULL)
            AND coalesce(pair[1].document_id, pair[0].document_id) = pair[0].document_id
            AND pair[0].document_id IN coalesce(pair[1].document_ids,[pair[0].document_id])
            AND all(doc IN coalesce(pair[1].document_ids,[]) WHERE doc IS NOT NULL AND doc <> '')
            AND NOT [pair[0].entity_id,k.concept_id] IN $blocked)
    """),
}
NAMES = """
MATCH (n {namespace:$ns}) WHERE n.key IN $keys
RETURN n.key AS key, n.name AS name, n.document_id AS document_id
"""
LIMITS = TraversalLimits()
RETRY_LIMITS = TraversalLimits(max_neighbors=100, max_candidates=1000,
                               max_relationships=5000, timeout_seconds=20)
QUOTE_LIMIT = 220


def discover(tx, kind, mixed):
    """Независимый Cypher выбирает один маршрут с минимальной степенью старта."""
    pattern, nodes, edges, trust = BRIDGES[kind]
    facts = "true"
    if mixed:
        pattern = "(a:RELEXEntity)-[f:RELEX_REL]->" + pattern + "-[g:RELEX_REL]->(d:RELEXEntity)"
        nodes, edges = f"[a]+{nodes}+[d]", f"[f]+{edges}+[g]"
        facts = "f.document_id = b.document_id AND a.document_id = b.document_id AND g.document_id = c.document_id AND d.document_id = c.document_id"
    query = f"""
    MATCH {pattern}
    WHERE b.document_id <> c.document_id AND ({trust}) AND ({facts})
    WITH {nodes} AS nodes, {edges} AS edges, b.name AS bridge_start_name
    WHERE all(node IN nodes WHERE node.namespace = $ns AND single(other IN nodes WHERE other = node))
          AND all(edge IN edges WHERE edge.namespace = $ns)
    WITH nodes[0] AS seed, nodes, edges, bridge_start_name
    WHERE seed.name IS NOT NULL AND trim(seed.name) <> ''
    RETURN [node IN nodes | node.key] AS nodes, [edge IN edges | edge.key] AS edges,
           seed.name AS seed_name, seed.entity_id AS entity_id, seed.document_id AS document_id,
           bridge_start_name,
           COUNT {{ (seed)-[:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-() }} AS start_degree
    ORDER BY start_degree, nodes, edges LIMIT 1
    """
    rows = tx.run(query, ns=NS, blocked=sorted(map(list, SUSPICIOUS_CELL_PAIRS))).data()
    return rows[0] if rows else None


def mixed_facts(path):
    docs = {e["document_id"] for e in path["edges"] if e["type"] == "RELEX_REL"}
    return len(docs) >= 2 and any(e["type"] != "RELEX_REL" for e in path["edges"])


def describe(path, bundle, names, provenance):
    for item in bundle["edges"]:
        edge = item["edge"]
        key = edge["key"]
        if key in provenance:
            continue
        props = edge["properties"]
        windows = []
        for source in item["source_relations"]:
            for w in source["windows"]:
                if w["status"] != "valid":
                    continue
                windows.append({**{k: w.get(k) for k in (
                    "document_id", "revision_id", "text_sha256", "source_url", "title", "window_id", "start", "end")},
                    "source_triple_id": source["source_triple_id"],
                    "source_head_id": source["source"].get("head_id"),
                    "source_tail_id": source["source"].get("tail_id"),
                    "excerpt": w["text"][:QUOTE_LIMIT],
                    "excerpt_end": w["start"] + min(QUOTE_LIMIT, len(w["text"]))})
        provenance[key] = {
            **{k: edge[k] for k in ("type", "relation_type", "head_key", "tail_key", "document_id")},
            "non_factual": item["non_factual"], "evidence_status": item["status"], "windows": windows,
            "properties_sha256": hashlib.sha256(json.dumps(props, sort_keys=True, ensure_ascii=False,
                                                            allow_nan=False).encode()).hexdigest(),
            "properties": {k: v for k, v in props.items() if k in {
                "original_head_id", "original_tail_id", "original_left_canonical_id", "original_right_canonical_id",
                "canonical_triple_id", "triple_id", "source_triple_ids", "semantic_link_id", "similarity",
                "left_canonical_id", "right_canonical_id", "source_entity_id", "target_concept_id",
                "source_membership_id", "membership_id", "resolution_decision_2f2", "review_status",
                "requires_review", "document_id", "document_ids", "left_document_id", "right_document_id",
                "factual", "identity", "fact_inheritance_allowed", "directed", "undirected", "source_stage"}},
        }
    return {"path_id": path["path_id"], "document_ids": path["document_ids"],
            "nodes": [names[key] for key in path["node_keys"]],
            "edges": [{"key": e["key"], "traversed_reverse": e["traversed_reverse"]} for e in path["edges"]],
            "evidence_status": bundle["status"], "evidence_diagnostics": bundle["diagnostics"],
            "reasons": bundle["reasons"]}


def run_paths(driver, database, seeds, question, limits, index, witness=None):
    times = {}
    started = perf_counter()
    retrieval = GraphRetriever(driver, database=database).retrieve(seeds, limits)
    times["graph_retrieval"] = perf_counter() - started
    started = perf_counter()
    evidence = EvidenceExtractor(driver, database=database).extract(retrieval)
    times["evidence_extraction"] = perf_counter() - started
    started = perf_counter()
    ranked = rerank_paths(question, retrieval, evidence, seeds)
    times["reranking"] = perf_counter() - started
    json.dumps((retrieval, evidence, ranked), allow_nan=False)

    bundles = {b["path_id"]: b for b in evidence}
    cross = [p for p in retrieval["paths"] if len(p["document_ids"]) > 1]
    top = {p["path_id"]: p for p in ranked["paths"]}
    exclusions = {p["path_id"]: p["reason"] for p in ranked["excluded"]}
    witness_path = next((p for p in cross if witness and p["node_keys"] == witness["nodes"]
                         and [e["key"] for e in p["edges"]] == witness["edges"]), None)
    wanted = {p["path_id"]: p for p in cross}
    wanted.update((p["path_id"], p["path"]) for p in ranked["paths"])
    keys = sorted({k for p in wanted.values() for k in p["node_keys"]})
    from neo4j import Query, READ_ACCESS
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        names = {r["key"]: dict(r) for r in session.run(Query(NAMES, timeout=10), ns=NS, keys=keys)}
    index["nodes"].update(names)

    for path_id, path in wanted.items():
        details = describe(path, bundles[path_id], names, index["edges"])
        details["nodes"] = path["node_keys"]
        index["paths"][path_id] = details

    def disposition(path_id):
        if path_id in top:
            return "top_5"
        return exclusions.get(path_id, ranked["reason"] or "unaccounted")

    evidence_counts = Counter(bundles[p["path_id"]]["status"] for p in cross)
    eligibility = {"top_5", "outside_top_k", "duplicate_factual_sources"}
    mixed_paths = [p for p in cross if mixed_facts(p)]
    return {
        "seconds": {k: round(v, 6) for k, v in times.items()}, "limits": asdict(limits),
        "graph_fingerprint": retrieval["graph_fingerprint"], "path_count": len(retrieval["paths"]),
        "cross_count": len(cross), "cross_evidence": dict(evidence_counts),
        "cross_eligible": sum(disposition(p["path_id"]) in eligibility for p in cross),
        "cross_top_5": sum(p["path_id"] in top for p in cross),
        "mixed_count": len(mixed_paths),
        "mixed_valid": sum(bundles[p["path_id"]]["status"] == "valid" for p in mixed_paths),
        "mixed_top_5": sum(p["path_id"] in top for p in mixed_paths),
        "cross_paths": [{"path_id": p["path_id"], "reranking": disposition(p["path_id"])} for p in cross],
        "truncated": retrieval["truncated"], "diagnostics": retrieval["diagnostics"],
        "top_5": [{k: p[k] for k in ("path_id", "score", "score_breakdown")} for p in ranked["paths"]],
        "witness_path_id": witness_path["path_id"] if witness_path else None,
        "witness_reranking": disposition(witness_path["path_id"]) if witness_path else None,
    }


def witness_status(case, run, paths):
    path_id = run["witness_path_id"]
    if not path_id:
        return "BLOCKED" if run["truncated"] else "FAIL"
    if case.startswith("C"):
        return "PASS" if (paths[path_id]["evidence_status"] == "valid" and
                          run["witness_reranking"] in {"top_5", "outside_top_k", "duplicate_factual_sources"}) else "FAIL"
    return "PASS" if (paths[path_id]["evidence_status"] == "non_factual" and
                      run["witness_reranking"] == "no_factual_evidence") else "FAIL"


def verify(driver, database):
    from neo4j import READ_ACCESS, unit_of_work

    @unit_of_work(timeout=20)
    def read(tx):
        facts = {}
        audit.verify(tx, facts)
        return facts

    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        return session.execute_read(read)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neo4j-env", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question", action="append", default=[])
    args = parser.parse_args(argv)
    if len(args.question) > 3 or any(not q.strip() for q in args.question):
        parser.error("Supply at most three nonempty questions")

    started = perf_counter()
    report = {"status": "FAIL", "namespace": NS, "started_at": datetime.now(timezone.utc).isoformat(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "targeted": {}, "e2e": [], "integrity": "BLOCKED", "smoke_only_min_score": -1.0,
              "index": {"nodes": {}, "edges": {}, "paths": {}}}
    driver = None
    phase = "preflight"
    try:
        if args.neo4j_env is None and not os.getenv("NEO4J_PASSWORD"):
            report["blocker"] = "missing_neo4j_credentials"
            return 2
        driver, database = audit.stage.graph.base.core.connect_neo4j(args.neo4j_env)
        t = perf_counter()
        report["before"] = verify(driver, database)
        report["preflight_seconds"] = round(perf_counter() - t, 6)
        print("Frozen preflight PASS", flush=True)

        from neo4j import READ_ACCESS, unit_of_work

        @unit_of_work(timeout=20)
        def find(tx):
            return {case: discover(tx, kind, mixed) for case, kind, mixed in (
                ("A", "semantic", False), ("B", "concept", False),
                ("C_semantic", "semantic", True), ("C_concept", "concept", True))}

        phase = "discovery"
        t = perf_counter()
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            witnesses = session.execute_read(find)
        report["discovery_seconds"] = round(perf_counter() - t, 6)
        retry_used = False
        for case, witness in witnesses.items():
            phase = f"targeted_{case}"
            entry = report["targeted"][case] = {"witness": witness, "runs": []}
            if witness is None:
                entry["status"] = "NOT_AVAILABLE"
                continue
            seed = {k: witness[k] for k in ("entity_id", "document_id")}
            seed.update(node_key=witness["nodes"][0], mention_id=None, match_method="diagnostic", score=0.0)
            question = f"What is reported about {witness['seed_name']}?"
            entry["question"] = question
            run = run_paths(driver, database, [seed], question, LIMITS, report["index"], witness)
            entry["runs"].append(run)
            if not run["witness_path_id"] and run["truncated"] and not retry_used:
                retry_used = True
                run = run_paths(driver, database, [seed], question, RETRY_LIMITS, report["index"], witness)
                entry["runs"].append(run)
            entry["status"] = witness_status(case, run, report["index"]["paths"])
            print(f"{phase}: {entry['status']}, {run['cross_count']} cross-document paths", flush=True)

        phase = "model_setup"
        from cgraph.entity_linking import E5Encoder, EntityLinker, load_entity_cache
        from cgraph.query_analysis import E3Extractor, QueryAnalyzer
        import torch
        t = perf_counter()
        cache = load_entity_cache(ROOT / ".cache/semantic_concept_sweep")
        encoder = E5Encoder(cache.model)
        linker = EntityLinker(cache, encoder, min_score=-1.0)
        report["cache"] = {"entities": len(cache.entities), "fingerprint": cache.fingerprint}
        report["model_setup_seconds"] = round(perf_counter() - t, 6)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        analyzer = QueryAnalyzer(E3Extractor(ROOT / "artifacts/ner_v2/models/gliner25"))
        auto_questions = [f"What is reported about {witnesses[c][field]}?"
                          for c, field in (("C_semantic", "bridge_start_name"), ("C_concept", "bridge_start_name"),
                                           ("C_semantic", "seed_name"), ("A", "seed_name"), ("B", "seed_name"))
                          if witnesses[c] and witnesses[c][field]]
        questions = args.question or list(dict.fromkeys(auto_questions))[:3]
        if questions:
            print("WARNING: RUNNING REAL GLINER AND E5 CPU INFERENCE FOR UP TO THREE QUESTIONS.", flush=True)
        for question in questions:
            entry = {"question": question, "runs": [], "seconds": {}}
            report["e2e"].append(entry)
            phase = "query_analysis"
            t = perf_counter()
            analysis = analyzer.analyze(question)
            entry["seconds"][phase] = round(perf_counter() - t, 6)
            entry["analysis"] = analysis
            phase = "entity_linking"
            t = perf_counter()
            linking = linker.link(analysis, top_k=2)
            entry["seconds"][phase] = round(perf_counter() - t, 6)
            entry["linking"] = linking
            entry["fallback_used"] = not analysis["mentions"]
            if not linking["seeds"]:
                continue
            phase = "e2e_graph_evidence_reranking"
            run = run_paths(driver, database, linking["seeds"], question, LIMITS, report["index"])
            entry["runs"].append(run)
            if not run["cross_eligible"] and run["truncated"] and not retry_used:
                retry_used = True
                run = run_paths(driver, database, linking["seeds"], question, RETRY_LIMITS, report["index"])
                entry["runs"].append(run)
            print(f"E2E: {run['cross_count']} cross-document paths, {run['cross_top_5']} in Top-5", flush=True)
            if run["cross_top_5"]:
                break
    except Exception as exc:
        report["failure"] = {"stage": phase, **failure(exc)}
    finally:
        if driver is not None:
            try:
                t = perf_counter()
                report["after"] = verify(driver, database)
                report["postflight_seconds"] = round(perf_counter() - t, 6)
                if "before" in report:
                    report["integrity"] = "PASS" if report["before"] == report["after"] else "FAIL"
            except Exception as exc:
                report["integrity"] = "FAIL"
                report["postflight_failure"] = failure(exc)
            finally:
                driver.close()
        e2e_runs = [r for q in report["e2e"] for r in q["runs"]]
        targeted_runs = [r for case in report["targeted"].values() for r in case["runs"]]
        statuses = {k: report["targeted"].get(k, {}).get("status", "BLOCKED") for k in ("A", "B", "C_semantic", "C_concept")}
        c = [statuses[k] for k in ("C_semantic", "C_concept")]
        statuses["C"] = next(s for s in ("PASS", "FAIL", "BLOCKED", "NOT_AVAILABLE") if s in c)
        statuses["E2E"] = ("PASS" if any(r["cross_count"] for r in e2e_runs) else
                           "BLOCKED" if not e2e_runs or any(r["truncated"] for r in e2e_runs) else "FAIL")
        statuses["Reranker"] = ("PASS" if any(r["cross_top_5"] for r in e2e_runs + targeted_runs) else
                                "BLOCKED" if not e2e_runs and not targeted_runs else "FAIL")
        statuses["Integrity"] = report["integrity"]
        report["scenarios"] = statuses
        required = [statuses[k] for k in ("A", "B", "C", "E2E", "Reranker", "Integrity")]
        if all(s == "PASS" for s in required) and "failure" not in report:
            report["status"] = "PASS"
        elif "PASS" in required and report["integrity"] != "FAIL":
            report["status"] = "PARTIAL"
        report["total_seconds"] = round(perf_counter() - started, 6)
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload + "\n")
        print(json.dumps({"status": report["status"], "scenarios": statuses, "failure": report.get("failure")}))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
