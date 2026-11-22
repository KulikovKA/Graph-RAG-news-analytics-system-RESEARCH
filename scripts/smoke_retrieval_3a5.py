"""Read-only Stage 3A.5.1 real pipeline smoke; not a retrieval quality benchmark.

Run from the repository root with --question and optionally --neo4j-env.
min_score=-1.0 is smoke_only; component defaults and frozen assets are unchanged.
Exit codes: 0 PASS; 2 preflight failed; 3 pipeline/postflight failed.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import smoke_retrieval_3a1 as audit


STAGES = ("query_analysis", "entity_linking", "graph_retrieval", "evidence_extraction", "reranking")
QUOTE_LIMIT = 320


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def failure(exc):
    # Текст ошибок драйвера и значения переменных окружения не выводятся.
    result = {"error_type": type(exc).__name__}
    if isinstance(exc, audit.stage.graph.base.core.Neo4jConnectionError):
        result["reason"] = str(exc)
    code = getattr(exc, "code", None)
    if isinstance(code, str) and re.fullmatch(r"Neo\.[A-Za-z.]+", code):
        result["neo4j_code"] = code
    frames = traceback.extract_tb(exc.__traceback__)
    if frames:
        last = frames[-1]
        result["location"] = f"{Path(last.filename).name}:{last.lineno}:{last.name}"
        if (last.name == "require" and Path(last.filename).name in
                {"smoke_retrieval_3a1.py", "smoke_retrieval_3a5.py"}):
            result["invariant"] = str(exc)
    return result


def require(condition, name):
    if not condition:
        raise ValueError(name)


def ranked_summary(item):
    path = item["path"]
    fragments = []
    for e in item["evidence"]["edges"]:
        for source in e["source_relations"]:
            for window in source["windows"]:
                if window["status"] == "valid":
                    fragments.append({**{k: window[k] for k in (
                        "document_id", "revision_id", "text_sha256", "window_id", "start", "end",
                        "source_url", "title")}, "source_triple_id": source["source_triple_id"],
                        "edge_key": e["edge"]["key"], "text": window["text"][:QUOTE_LIMIT],
                        "excerpt_start": window["start"],
                        "excerpt_end": window["start"] + min(len(window["text"]), QUOTE_LIMIT),
                        "text_truncated": len(window["text"]) > QUOTE_LIMIT})
                    break
            if fragments:
                break
        if fragments:
            break
    require(bool(fragments), "ranked_path_without_validated_fact")
    return {"path_id": path["path_id"], "score": item["score"], "score_breakdown": item["score_breakdown"],
            "node_keys": path["node_keys"], "document_ids": path["document_ids"],
            "edges": [{k: e[k] for k in ("key", "type", "relation_type", "head_key", "tail_key",
                                       "traversed_reverse", "document_id")} for e in path["edges"]],
            "validated_excerpt": fragments[0]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--neo4j-env", type=Path)
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("--question must be nonempty")
    started = perf_counter()
    report = {"status": "FAIL", "namespace": audit.TARGET_NAMESPACE, "question": args.question,
              "smoke_only": {"min_score": -1.0, "link_top_k_per_mention": 2, "rerank_top_k": 5},
              "stages": {s: {"stage": s, "status": "SKIP", "seconds": 0.0, "count": 0} for s in STAGES},
              "read_only": True, "kg_unchanged": None, "json_serializable": False}
    driver, database, before, cache = None, None, {}, None
    preflight_start = perf_counter()
    current = "preflight"

    def run_stage(name, operation, summarize):
        nonlocal current
        current = name
        stage_start = perf_counter()
        stage = report["stages"][name]
        try:
            value = operation()
            count, details = summarize(value)
            stage.update(count=count, details=details)
            require(count > 0 or name == "query_analysis", name + "_no_usable_results")
            stage["status"] = "PASS"
            return value
        except Exception as exc:
            stage.update(status="FAIL", **failure(exc))
            raise
        finally:
            stage["seconds"] = round(perf_counter() - stage_start, 6)
            emit(stage)

    try:
        # Доступность проверяется до inference; соединение использует helper Stage 3A.1.
        from neo4j import READ_ACCESS
        from cgraph.query_analysis import E3_HASHES, E3Extractor, QueryAnalyzer
        from cgraph.entity_linking import E5Encoder, EntityLinker, load_entity_cache
        from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
        from cgraph.evidence_extraction import EvidenceExtractor
        from cgraph.path_reranking import rerank_paths_a as rerank_paths
        import gliner2
        import torch
        require(all((ROOT / "artifacts/ner_v2/models/gliner25" / name).is_file() for name in E3_HASHES),
                "missing_gliner_assets")
        cache = load_entity_cache(ROOT / ".cache/semantic_concept_sweep")
        require(all(Path(f["path"]).is_file() for f in cache.model["files"]), "missing_e5_assets")
        report["assets"] = {"cache_entities": len(cache.entities), "cache_fingerprint": cache.fingerprint,
                            "gliner_available": True, "e5_available": True,
                            "torch_version": torch.__version__}
        if args.neo4j_env is not None:
            require(args.neo4j_env.is_file(), "neo4j_env_file_missing")
        require(args.neo4j_env is not None or bool(os.environ.get("NEO4J_PASSWORD")), "missing_neo4j_credentials")
        driver, database = audit.stage.graph.base.core.connect_neo4j(args.neo4j_env)
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            audit.verify(session, before)
        report["preflight"] = {"status": "PASS", "seconds": round(perf_counter() - preflight_start, 6),
                               "verified": before}
        emit({"preflight": report["preflight"]})
        print("WARNING: REAL GLINER AND E5 CPU INFERENCE; MODEL LOADING MAY TAKE MINUTES.", flush=True)
        torch.manual_seed(42)
        torch.set_num_threads(4)
        analyzer = QueryAnalyzer(E3Extractor(ROOT / "artifacts/ner_v2/models/gliner25"))

        def query_summary(value):
            require(analyzer.extractor._model is not None, "gliner_inference_not_executed")
            return len(value["mentions"]), {"question": args.question, "inference_executed": True,
                                           "model_identifier": value["model_identifier"],
                                           "mentions": value["mentions"], "warnings": value["warnings"]}

        analysis = run_stage("query_analysis", lambda: analyzer.analyze(args.question), query_summary)
        # GLiNER освобождается до загрузки E5; тяжёлый inference выполняется последовательно.
        del analyzer
        encoder = E5Encoder(cache.model)
        linker = EntityLinker(cache, encoder, min_score=-1.0)

        def link_summary(value):
            require(encoder._model is not None, "e5_query_embedding_not_computed")
            return len(value["seeds"]), {"query_embedding_computed": True, "smoke_only": True,
                                        "cutoff": value["cutoff"], "candidates": value["candidates"],
                                        "seed_count": len(value["seeds"])}

        linking = run_stage("entity_linking", lambda: linker.link(analysis, top_k=2), link_summary)
        limits = TraversalLimits(max_hops=3, max_neighbors=20, max_candidates=100, max_relationships=300)
        report["traversal_limits"] = asdict(limits)

        def graph_summary(value):
            return len(value["paths"]), {
                "edge_type_occurrences": dict(Counter(e["type"] for p in value["paths"] for e in p["edges"])),
                "cross_document_paths": sum(len(p["document_ids"]) > 1 for p in value["paths"]),
                "truncated": value["truncated"], "diagnostics": value["diagnostics"]}

        retrieval = run_stage("graph_retrieval", lambda: GraphRetriever(driver, database=database).retrieve(
            linking["seeds"], limits), graph_summary)

        def evidence_summary(value):
            confirmed = {e["edge"]["key"] for b in value for e in b["edges"]
                         if not e["non_factual"] and e["status"] == "valid"}
            counts = Counter(b["status"] for b in value)
            return len(confirmed), {"path_bundles": len(value),
                                "validation_status_counts": {s: counts[s] for s in (
                                    "valid", "partial", "invalid", "non_factual")},
                                "confirmed_unique_factual_edges": len(confirmed),
                                "rejection_reasons": dict(Counter(r for b in value for r in b["reasons"]))}

        evidence = run_stage("evidence_extraction", lambda: EvidenceExtractor(driver, database=database).extract(
            retrieval), evidence_summary)

        def rerank_summary(value):
            return len(value["paths"]), {"top_paths": [ranked_summary(p) for p in value["paths"]],
                                        "truncated": value["truncated"], "diagnostics": value["diagnostics"],
                                        "excluded_reasons": dict(Counter(p["reason"] for p in value["excluded"]))}

        ranked = run_stage("reranking", lambda: rerank_paths(args.question, retrieval, evidence,
                                                            linking["seeds"]), rerank_summary)
        current = "serialization"
        # Проверяется полный результат, а не только сокращённое представление для консоли.
        json.dumps(dict(analysis=analysis, linking=linking, retrieval=retrieval, evidence=evidence, ranked=ranked),
                   ensure_ascii=False, allow_nan=False)
        report["json_serializable"] = True
        report["status"] = "PASS"
    except Exception as exc:
        report["failure"] = {"stage": current, **failure(exc)}
        if current == "preflight":
            report["preflight"] = {"status": "FAIL", "seconds": round(perf_counter() - preflight_start, 6)}
    finally:
        if driver is not None:
            after_start = perf_counter()
            try:
                from neo4j import READ_ACCESS
                after = {}
                with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
                    audit.verify(session, after)
                report["kg_unchanged"] = bool(before) and all(before[k] == after[k] for k in (
                    "snapshot_sha256", "graph_fingerprint", "node_counts", "relationship_counts"))
                require(report["kg_unchanged"], "kg_snapshot_changed_or_missing_baseline")
                report["postflight"] = {"status": "PASS", "verified": after}
            except Exception as exc:
                report["status"] = "FAIL"
                report["postflight"] = {"status": "FAIL", **failure(exc)}
            report["postflight"]["seconds"] = round(perf_counter() - after_start, 6)
            driver.close()
        report["total_seconds"] = round(perf_counter() - started, 6)
        emit(report)
    return 0 if report["status"] == "PASS" else (2 if current == "preflight" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
