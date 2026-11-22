"""Run the frozen Stage 3A.5.2 Block C retrieval baseline against live Neo4j."""

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter

from validate_draft import HERE, documents, jsonl, require


ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from smoke_retrieval_3a5 import audit, failure

QUESTIONS = HERE / "questions_draft.jsonl"


def compact_paths(retrieval, bundles, nodes):
    evidence_by_path = {b["path_id"]: b for b in bundles}
    paths, facts = {}, {}
    for path in retrieval["paths"]:
        path_id = path["path_id"]
        bundle = evidence_by_path[path_id]
        transitions = []
        path_nodes = path["node_keys"]
        documents = [nodes[key].get("document_id") for key in path_nodes]
        document_positions = [(i, d) for i, d in enumerate(documents) if d]
        for (i, left), (j, right) in zip(document_positions, document_positions[1:]):
            if left != right:
                transitions.append({"from_document": left, "to_document": right,
                                    "nodes": path_nodes[i:j+1],
                                    "edge_types": [e["type"] for e in path["edges"][i:j]]})
        evidence_documents = set()
        for entry in bundle["edges"]:
            edge = entry["edge"]
            if edge["type"] != "RELEX_REL":
                continue
            facts[edge["key"]] = {
                **{k: edge[k] for k in ("head_key", "tail_key", "relation_type", "document_id")},
                **{k: entry[k] for k in ("status", "reasons", "source_relations")},
            }
            if bundle["navigation_valid"] and entry["status"] == "valid":
                evidence_documents.add(edge["document_id"])
        paths[path_id] = {**path, "hops": len(path["edges"]), "transitions": transitions,
                          "edges": [{k: e[k] for k in ("key", "type", "relation_type", "head_key",
                                                      "tail_key", "traversed_reverse", "document_id")}
                                    for e in path["edges"]],
                          "evidence_status": bundle["status"], "evidence_reasons": bundle["reasons"],
                          "factual_documents": sorted(evidence_documents),
                          "cross_doc_factual": bool(transitions) and len(evidence_documents) >= 2}
    return paths, facts


def measure(result, gold):
    required = set(gold["document_contributions"])
    result["gold_docs"] = sorted(required)
    result["gold_evidence_ids"] = [e["evidence_id"] for e in gold["evidence"]]
    result["reference_set_non_exhaustive"] = gold["reference_set_non_exhaustive"]
    if result["status"] != "PASS":
        return
    paths = result["candidate_paths"]
    top = [paths[ranked["path_id"]] for ranked in result["top_5"]]
    top_docs = {d for p in top for d in p["document_ids"]}
    factual_docs = {d for p in top for d in p["factual_documents"]}
    candidate_docs = {d for p in paths.values() for d in p["document_ids"]}
    factual_candidates = {d for p in paths.values() for d in p["factual_documents"]}
    windows = {}
    for path in top:
        for edge in path["edges"]:
            fact = result["facts"].get(edge["key"])
            if fact is None or fact["status"] != "valid":
                continue
            for source in fact["source_relations"]:
                for window in source["windows"]:
                    windows[edge["key"], source["source_triple_id"], window["window_id"]] = window
    overlaps = {}
    for e in gold["evidence"]:
        matches = []
        for (key, triple_id, window_id), w in windows.items():
            start, end = max(w["start"], e["start"]), min(w["end"], e["end"])
            if w["document_id"] == e["document_id"] and w["revision_id"] == e["revision_id"] and start < end:
                matches.append(dict(edge_key=key, triple_id=triple_id, window_id=window_id,
                                    intersection=[start, end], contains_gold_span=(start, end) == (e["start"], e["end"])))
        if matches:
            overlaps[e["evidence_id"]] = matches
    result.update(candidate_documents=sorted(candidate_docs), candidate_factual_documents=sorted(factual_candidates),
                  candidate_cross_doc_factual=[i for i, p in paths.items() if p["cross_doc_factual"]],
                  retrieved_docs_at_5=sorted(top_docs), factual_docs_at_5=sorted(factual_docs),
                  cross_doc_factual_at_5=[p["path_id"] for p in top if p["cross_doc_factual"]],
                  window_overlap_at_5=overlaps,
                  metrics={"GoldDocRecall@5": len(top_docs & required)/len(required),
                           "AllRequiredDocs@5": required <= top_docs,
                           "EvidenceWindowOverlap@5": len(overlaps)/len(gold["evidence"]),
                           "EvidenceCoverage@5": None,
                           "GoldDocRecall@10": None, "AllRequiredDocs@10": None,
                           "EvidenceCoverage@10": None})


def run_question(row, analyzer, linker, retriever, extractor, rerank_paths, limits, nodes):
    key, question = row
    result = {"question_id": key, "question": question, "status": "FAILED", "seconds": {}}
    stage = None

    def timed(name, call, *args):
        nonlocal stage
        stage = name
        start = perf_counter()
        try:
            return call(*args)
        finally:
            result["seconds"][name] = round(perf_counter() - start, 6)

    try:
        query = result["query_analysis"] = timed("query_analysis", analyzer.analyze, question)
        link = result["entity_linking"] = timed("entity_linking", linker.link, query)
        # Traversal принимает не более max_seeds; порядок линкера сохранён.
        seeds = link["seeds"][:limits.max_seeds]
        link["traversal_seeds"] = seeds
        link["seed_cap_excluded"] = link["seeds"][limits.max_seeds:]
        accepted = {(s["mention_id"], s["node_key"]) for s in link["seeds"]}
        link["cutoff_excluded"] = [c for c in link["candidates"]
                                   if (c["mention_id"], c["node_key"]) not in accepted]
        retrieval = timed("graph_retrieval", retriever.retrieve, seeds)
        result["traversal"] = {k: v for k, v in retrieval.items() if k != "paths"}
        evidence = timed("evidence_extraction", extractor.extract, retrieval)
        ranking = timed("reranking", rerank_paths, question, retrieval, evidence, seeds)
        result["candidate_paths"], result["facts"] = compact_paths(retrieval, evidence, nodes)
        result["top_5"] = [{k: v for k, v in p.items() if k not in ("path", "evidence")}
                           for p in ranking["paths"]]
        result["top_10"] = None
        result["reranking"] = {k: v for k, v in ranking.items() if k != "paths"}
        result["status"] = "PASS"
    except Exception as exc:
        result.update(error_stage=stage, **failure(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-score", type=float, required=True,
                        help="Explicit experimental E5 query-linking policy. No production cutoff exists.")
    parser.add_argument("--output", type=Path, required=True, help="New JSON path")
    parser.add_argument("--reranking-policy", choices=("A", "C", "C2"),
                        help="Override the default ranking policy for this run")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    started = datetime.now(timezone.utc)
    questions = [(r["question_id"], r["question"]) for r in jsonl(QUESTIONS) if r["category"] == "C"]
    report = {"started_at_utc": started.isoformat(), "status": "BLOCKED", "questions": [],
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "namespace": audit.TARGET_NAMESPACE, "preflight": {},
              "unsupported": {"top_10": "rerank_paths contract caps top_k at 5"}}
    driver = None
    phase = "preflight"
    try:
        from neo4j import READ_ACCESS
        from tqdm import tqdm
        import torch
        from cgraph.query_analysis import MODEL_ID, E3_HASHES, QueryAnalyzer
        from cgraph.entity_linking import _sha, load_entity_cache, EntityLinker
        from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
        from cgraph.evidence_extraction import EvidenceExtractor
        from cgraph.path_reranking import DEFAULT_RERANKING_POLICY, rerank_paths

        limits = TraversalLimits()
        analyzer = QueryAnalyzer()
        source = documents()
        cache = load_entity_cache()
        preflight = report["preflight"]
        preflight.update(frozen_corpus_documents=len(source), entity_cache_fingerprint=cache.fingerprint)
        for directory, files in ((analyzer.extractor.model_path, E3_HASHES),
                                 (Path(cache.model["root"]), {f["path"]: f["sha256"] for f in cache.model["files"]})):
            for name, sha in files.items():
                require(_sha(directory / name) == sha, f"model_checksum:{Path(name).name}")
        preflight["model_integrity"] = "PASS"
        linker = EntityLinker(cache, min_score=args.min_score)
        report["settings"] = {"gliner": MODEL_ID, "query_threshold": analyzer.threshold, "e5": cache.model,
                              "link_top_k": 5, "experimental_link_min_score": args.min_score,
                              "traversal": asdict(limits), "rerank_top_k": 5,
                              "torch": torch.__version__, "python": sys.version.split()[0],
                              "torch_seed": 42, "cpu_threads": 4, "answer_generation": False}
        report["settings"]["reranking_policy"] = args.reranking_policy or DEFAULT_RERANKING_POLICY
        require(bool(os.getenv("NEO4J_PASSWORD")), "missing_process_neo4j_credentials")
        driver, database = audit.stage.graph.base.core.connect_neo4j()
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            audit.verify(session, preflight)
            records = session.run(audit.stage.graph.READ_NODES, ns=audit.TARGET_NAMESPACE).data()
        props = [r["properties"] for r in records]
        db_docs = {n["document_id"]: n for n, r in zip(props, records) if "RELEXDocument" in r["labels"]}
        require(db_docs.keys() == source.keys(), "live_document_ids")
        for key, doc in db_docs.items():
            require(all(doc[f] == source[key][f] for f in ("revision_id", "text_sha256", "text")), "live_document_content")
        preflight.update(status="PASS", live_documents=len(db_docs))
        nodes = {n["key"]: {k: n[k] for k in ("entity_id", "concept_id", "document_id", "name") if k in n}
                 for n in props}
        report["nodes"] = nodes
        torch.manual_seed(42)
        torch.set_num_threads(4)
        retriever = GraphRetriever(driver, database=database)
        extractor = EvidenceExtractor(driver, database=database)
        kwargs = {"policy": args.reranking_policy}
        if args.reranking_policy == "C2":
            kwargs["entity_nodes"] = {n["key"]: n for n in props}
        rank = partial(rerank_paths, **kwargs)
        phase = "retrieval"
        print("WARNING: SEQUENTIAL REAL GLINER/E5 CPU INFERENCE AND LIVE NEO4J RETRIEVAL.", flush=True)
        for row in tqdm(questions, desc="Block C", unit="question"):
            report["questions"].append(run_question(row, analyzer, linker, retriever, extractor, rank, limits, nodes))
        phase = "measurement"
        gold = {r["question_id"]: r for r in jsonl(QUESTIONS) if r["category"] == "C"}
        for result in report["questions"]:
            measure(result, gold[result["question_id"]])
        phase = "postflight"
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            report["postflight"] = {}
            audit.verify(session, report["postflight"])
        report["postflight"]["status"] = "PASS"
        report["status"] = "COMPLETE" if all(r["status"] == "PASS" for r in report["questions"]) else "PARTIAL"
    except Exception as exc:
        report.update(blocked_at=phase, **failure(exc))
        if isinstance(exc, ValueError) and phase == "preflight" and exc.__traceback__.tb_next:
            report["preflight"]["reason"] = str(exc)
    finally:
        if driver:
            driver.close()
    report["completed_questions"] = Counter(r["status"] for r in report["questions"])["PASS"]
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        output.write("\n")
    print(report["status"], report["completed_questions"], "/", len(questions))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
