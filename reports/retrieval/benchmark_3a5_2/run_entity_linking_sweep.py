"""Measure linking policies on pinned questions and a read-only Stage 2G.3 snapshot."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

import numpy as np
import psutil
from tqdm import tqdm

from validate_draft import HERE, documents, jsonl, require, validate

ROOT = HERE.parents[2]
VALIDATION = HERE / "validate_promote_c"
sys.path[:0] = [str(ROOT / "src"), str(VALIDATION)]

from run_block_c import compact_paths
from run_document_aware import BASELINE, BASELINE_SHA256, GOLD, measure, path_output, replay_baseline, summarize
from run_validation import HOLDOUT, HOLDOUT_SHA256, FrozenSnapshot, audit, read_pinned
from cgraph.entity_linking import EntityLinker, E5Encoder, _expansion_searches, _sha, _text_search, load_entity_cache
from cgraph.evidence_extraction import EvidenceExtractor
from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
from cgraph.path_reranking import rerank_paths
from cgraph.query_analysis import E3Extractor, QueryAnalyzer


THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
TOP_K = (5, 10, 20, 50)
POLICIES = ("baseline", "conditional", "expansion")
BASE = (0.85, 5, "baseline")
SOURCE_COMMIT = "8fa1399"
FIXED_FILES = {
    "graph_retrieval.py": "aabd88516d03204a677ef2de168b955d441ff4ed654c96724053b10c225a70c1",
    "path_reranking.py": "a80f49ccbe07fc09801499c1603c49e6b29a440467f8e86d2c1ea5bc2c2ef27c",
    "evidence_extraction.py": "cd9acffe1bd9464d809640414298b366ec2046b17294f8f29edb1e957687a138",
}
PROCESS = psutil.Process()


def snapshot(cache_dir, corpus):
    stage = audit.stage
    source = json.loads((cache_dir / "baseline.json").read_bytes())
    source_nodes, source_edges = stage.graph.unpack_snapshot(source)
    manifest, delta = stage.load_delta()
    nodes, edges, fingerprint = stage.build(source_nodes, source_edges, manifest, delta)
    require(fingerprint == audit.expected_fingerprint(), "Frozen graph fingerprint differs")
    require(stage.graph.snapshot_hash(nodes, edges) == audit.EXPECTED_SNAPSHOT_SHA256, "Frozen graph hash differs")
    result = FrozenSnapshot(nodes, edges, fingerprint)
    require(result.documents.keys() == corpus.keys(), "Frozen document membership differs")
    for key, doc in result.documents.items():
        require(all(doc[f] == corpus[key][f] for f in ("text", "revision_id", "text_sha256")),
                "Frozen graph document differs")
    return result


def question_rows(archive, corpus):
    read_pinned(GOLD, archive["benchmark_input"]["sha256"])
    read_pinned(HOLDOUT, HOLDOUT_SHA256)
    approved = [r for r in jsonl(GOLD) if r["category"] in "ABC"]
    holdout = list(jsonl(HOLDOUT))
    require(Counter(r["category"] for r in approved) == {"A": 8, "B": 7, "C": 10} and len(holdout) == 10,
            "Unexpected question scope")
    for r in approved:
        validate(r, corpus)
    for r in holdout:
        for e in r["evidence"]:
            doc = corpus[e["document_id"]]
            require(doc["text"][e["start"]:e["end"]] == e["quote"]
                    and doc["revision_id"] == e["revision_id"], "Holdout source differs")
    return {**{f'{r["category"]}/{r["question_id"]}': r for r in approved},
            **{f'holdout/{r["question_id"]}': r for r in holdout}}


class CachedLinker(EntityLinker):
    """Use the production policy against frozen per-text score rows."""

    def __init__(self, cache, scores, **kwargs):
        super().__init__(cache, **kwargs)
        self.scores = scores
        self.requested = set()

    def _score(self, texts):
        self.requested.update(texts)
        return np.stack([self.scores[t] for t in texts])


def memory():
    info = PROCESS.memory_info()
    return {"rss_bytes": info.rss, "process_peak_rss_bytes": getattr(info, "peak_wset", None)}


def prepare(cache, model_path, questions, score_dir, settings):
    inputs = {"questions": questions, "embedding_cache": cache.fingerprint,
              "query_threshold": settings["query_threshold"], "e3": settings["gliner"],
              "linker_sha256": _sha(ROOT / "src/cgraph/entity_linking.py")}
    metadata = score_dir / "index.json"
    matrix = score_dir / "cosine.npy"
    if metadata.exists():
        data = json.loads(metadata.read_bytes())
        require(data["inputs"] == inputs and data["scores_sha256"] == _sha(matrix), "Cosine cache inputs differ")
        values = np.load(matrix, allow_pickle=False)
        require(values.shape == (len(data["texts"]), len(cache.entities)), "Cosine cache shape differs")
        return data, dict(zip(data["texts"], values))

    score_dir.mkdir(parents=True, exist_ok=True)
    print("ВНИМАНИЕ: ПОСЛЕДОВАТЕЛЬНЫЙ CPU-ПРОГОН GLINER И E5.", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    torch.manual_seed(settings["torch_seed"])
    torch.set_num_threads(settings["cpu_threads"])
    analyzer = QueryAnalyzer(E3Extractor(model_path), threshold=settings["query_threshold"])
    analyses, ner_seconds, texts = {}, {}, {}
    for question in tqdm(dict.fromkeys(questions.values()), desc="GLiNER", unit="question"):
        start = perf_counter()
        analysis = analyzer.analyze(question)
        ner_seconds[question] = perf_counter() - start
        analyses[question] = analysis
        for search in [*EntityLinker._primary_searches(analysis), _text_search(question),
                       *_expansion_searches(question, set())]:
            texts.setdefault(search.text)
    memory_ner = memory()
    del analyzer

    encoder = E5Encoder(cache.model)
    scorer = EntityLinker(cache, encoder)
    start = perf_counter()
    encoder._load()
    model_load = perf_counter() - start
    values, e5_seconds = [], {}
    for text in tqdm(texts, desc="E5 cosine cache", unit="text"):
        start = perf_counter()
        values.append(scorer._score([text])[0])
        e5_seconds[text] = perf_counter() - start
    memory_e5 = memory()
    del scorer, encoder
    values = np.stack(values)
    with matrix.open("xb") as out:
        np.save(out, values, allow_pickle=False)
    data = {"inputs": inputs, "analyses": analyses, "ner_seconds": ner_seconds, "texts": list(texts),
            "e5_seconds": e5_seconds, "e5_load_seconds": model_load, "torch": torch.__version__,
            "memory_ner": memory_ner, "memory_e5": memory_e5, "scores_sha256": _sha(matrix)}
    write_json(metadata, data)
    return data, dict(zip(texts, values))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        out.write("\n")


def link_questions(config, questions, cache, scores, inference, max_seeds):
    cutoff, top_k, policy = config
    linker = CachedLinker(cache, scores, min_score=cutoff, top_k=top_k, fallback=policy)
    results = {}
    for key, text in questions.items():
        linker.requested.clear()
        start = perf_counter()
        link = linker.link(inference["analyses"][text])
        selection_seconds = perf_counter() - start
        results[key] = {
            "candidates": link["candidates"], "seeds_before_cap": link["seeds"],
            "seeds": link["seeds"][:max_seeds], "unresolved_mentions": link["unresolved_mentions"],
            "selection_seconds": selection_seconds,
            "ner_seconds": inference["ner_seconds"][text],
            "e5_seconds": sum(inference["e5_seconds"][t] for t in sorted(linker.requested)),
            "scored_texts": sorted(linker.requested),
        }
    return results


def unique(rows):
    return {r["node_key"]: r for r in reversed(rows)}


def linking_metrics(result, gold, off_topic):
    candidates = unique(result["candidates"])
    passed = unique([c for c in result["candidates"] if c["score"] >= gold["cutoff"]])
    seeds = unique(result["seeds"])
    all_seeds = unique(result["seeds_before_cap"])
    docs = lambda items: {s["document_id"] for s in items.values()}
    required = set(gold["document_contributions"])
    missing_docs = lambda items: sorted(required - docs(items))
    return {"gold_total": len(required), "candidate_hits": len(required & docs(candidates)),
            "threshold_candidate_hits": len(required & docs(passed)),
            "uncapped_seed_hits": len(required & docs(all_seeds)), "seed_hits": len(required & docs(seeds)),
            "missing_candidate_docs": missing_docs(candidates), "missing_threshold_docs": missing_docs(passed),
            "missing_uncapped_seed_docs": missing_docs(all_seeds), "missing_seed_docs": missing_docs(seeds),
            "candidates": len(candidates), "candidate_docs": len(docs(candidates)),
            "uncapped_seed_rows": len(result["seeds_before_cap"]), "seed_rows": len(result["seeds"]),
            "seeds": len(seeds), "seed_docs": len(docs(seeds)),
            "outside_reference_candidates": sum(s["document_id"] not in required for s in candidates.values()),
            "outside_reference_seeds": sum(s["document_id"] not in required for s in seeds.values()),
            "audited_off_topic_candidates": sum(s["document_id"] in off_topic for s in candidates.values()),
            "audited_off_topic_seeds": sum(s["document_id"] in off_topic for s in seeds.values()),
            "fallback_seeds": sum(s["match_method"] == "fallback" for s in seeds.values()),
            "scored_texts": result["scored_texts"],
            "seed_precision": "NOT_EVALUATED", "without_seeds": not bool(seeds),
            "ner_seconds": result["ner_seconds"], "e5_seconds": result["e5_seconds"],
            "linking_seconds": result["selection_seconds"] + result["e5_seconds"]}


def groups(items):
    return {group: [row for key, row in items.items() if group == "all" or key.startswith(group + "/")]
            for group in ("all", "A", "B", "C", "holdout")}


def aggregate_links(items):
    def aggregate(rows):
        total = sum(row["gold_total"] for row in rows)
        recall = {k.replace("_hits", "_recall"): sum(r[k] for r in rows) / total for k in
                  ("candidate_hits", "threshold_candidate_hits", "uncapped_seed_hits", "seed_hits")}
        sums = {k: sum(r[k] for r in rows) for k in
                ("candidates", "candidate_docs", "seeds", "seed_rows", "seed_docs", "fallback_seeds",
                 "outside_reference_candidates", "outside_reference_seeds", "audited_off_topic_candidates",
                 "audited_off_topic_seeds")}
        return {**recall, **sums, "gold_total": total, "seed_precision": "NOT_EVALUATED",
                "queries_without_seeds": sum(r["without_seeds"] for r in rows),
                "fraction_without_seeds": mean(r["without_seeds"] for r in rows),
                "median_linking_seconds": median(r["linking_seconds"] for r in rows)}
    return {group: aggregate(rows) for group, rows in groups(items).items()}


def diagnostics(questions, refs, analyses, cache, scores):
    linker = CachedLinker(cache, scores)
    rows = {}
    for key, question in questions.items():
        analysis = analyses[question]
        primary = linker._primary_searches(analysis)
        searches = [*primary, _text_search(question), *_expansion_searches(question, set())]
        gold_docs = set(refs[key]["document_contributions"])
        out = []
        for search in dict.fromkeys(searches):
            ranking = linker._rank(search, scores[search.text], len(cache.entities))
            cosine_order = sorted(ranking, key=lambda c: (-c["score"], c["entity_id"]))
            cosine_ranks = {c["entity_id"]: i for i, c in enumerate(cosine_order, 1)}
            def compact(c):
                return [c["entity_id"], c["document_id"], c["rank"], cosine_ranks[c["entity_id"]], c["score"]]
            out.append({"search": asdict(search), "top_50": [compact(c) for c in ranking[:50]],
                        "gold_document_entities": [compact(c) for c in ranking if c["document_id"] in gold_docs]})
        rows[key] = {"query_analysis": analysis, "gold_documents": sorted(gold_docs), "searches": out}
    return rows


def archive_parity(archive, graph, retriever, extractor, limits, fresh):
    nodes = {key: n["properties"] for key, n in graph.nodes.items()}
    for saved in archive["questions"]:
        key = saved["question_id"]
        seeds = saved["entity_linking"]["traversal_seeds"]
        generated = fresh[f"C/{key}"]["seeds"]
        require(len(generated) == len(seeds) and all(
            {k: v for k, v in left.items() if k != "score"} == {k: v for k, v in right.items() if k != "score"}
            and abs(left["score"] - right["score"]) < 2e-6 for left, right in zip(generated, seeds)),
            f"{key}: newly inferred baseline seeds differ from live archive")
        retrieval = retriever.retrieve(seeds, limits)
        bundles = extractor.extract(retrieval)
        require({k: v for k, v in retrieval.items() if k != "paths"} == saved["traversal"],
                f"{key}: live traversal diagnostics differ")
        paths, facts = compact_paths(retrieval, bundles, nodes)
        require(list(paths) == list(saved["candidate_paths"]) and paths == saved["candidate_paths"]
                and facts == saved["facts"], f"{key}: live candidates or provenance differ")
        replay_baseline(saved, retrieval, bundles, seeds)


def evaluate_retrieval(linked, refs, questions, retriever, extractor, limits, off_topic):
    evaluated = {}
    for key, row in linked.items():
        seeds = row["seeds"]
        start = perf_counter()
        retrieval = retriever.retrieve(seeds, limits)
        traversed = perf_counter()
        bundles = extractor.extract(retrieval)
        extracted = perf_counter()
        ranking = rerank_paths(questions[key], retrieval, bundles, seeds)
        ranked = perf_counter()
        paths = [path_output(p) for p in ranking["paths"]]
        gold = refs[key]
        required = set(gold["document_contributions"])
        gold = {**gold, "evidence": [{**e, "evidence_id": e.get("evidence_id", f"E{i:02d}")}
                                    for i, e in enumerate(gold["evidence"], 1)]}
        top_docs = {d for p in paths for d in p["document_ids"]}
        candidate_docs = {d for p in retrieval["paths"] for d in p["document_ids"]}
        fallback_spans = {}
        for c in row["candidates"]:
            if c["match_method"] == "fallback" and "query_span" in c:
                fallback_spans.setdefault((c["node_key"], c["score"]), c["query_span"])
        fallback_provenance = {s["node_key"]: fallback_spans[(s["node_key"], s["score"])]
                               for s in seeds if s["match_method"] == "fallback"
                               and (s["node_key"], s["score"]) in fallback_spans}
        evaluated[key] = {
            "gold_documents": sorted(required), "gold_evidence_count": len(gold["evidence"]),
            "result": {"metrics": measure(paths, gold), "top_5": paths,
                       "candidate_documents": sorted(candidate_docs), "candidate_paths": len(retrieval["paths"]),
                       "diagnostics": retrieval["diagnostics"], "unique_seeds": list(unique(seeds).values()),
                       "fallback_seed_query_spans": fallback_provenance,
                       "seed_cap_excluded": row["seeds_before_cap"][limits.max_seeds:],
                       "outside_reference_candidate_docs": sorted(candidate_docs - required),
                       "outside_reference_top_5_docs": sorted(top_docs - required),
                       "audited_off_topic_candidate_docs": sorted(candidate_docs & off_topic.get(key, set())),
                       "audited_off_topic_top_5_docs": sorted(top_docs & off_topic.get(key, set())),
                       "seconds": {"ner": row["ner_seconds"], "e5": row["e5_seconds"],
                                   "selection": row["selection_seconds"], "bfs": traversed - start,
                                   "evidence": extracted - traversed, "reranker_a": ranked - extracted,
                                   "total": row["ner_seconds"] + row["e5_seconds"] + row["selection_seconds"] +
                                   ranked - start}},
        }
    return evaluated


def aggregate_retrieval(items):
    result = {}
    for group, rows in groups(items).items():
        summary = summarize([{**r, "policies": {"A": r["result"]}} for r in rows], "A")
        runs = [r["result"] for r in rows]
        summary["bfs_limit_fractions"] = {
            limit: mean(limit in r["diagnostics"]["limits_hit"] for r in runs)
            for limit in ("max_candidates", "max_relationships", "max_neighbors", "max_hops")}
        summary["bfs_any_limit_fraction"] = mean(bool(r["diagnostics"]["limits_hit"]) for r in runs)
        summary["median_seconds"] = {name: median(r["seconds"][name] for r in runs) for name in runs[0]["seconds"]}
        summary["document_occurrences"] = {name: sum(len(r[name]) for r in runs) for name in
                                          ("outside_reference_candidate_docs", "outside_reference_top_5_docs",
                                           "audited_off_topic_candidate_docs", "audited_off_topic_top_5_docs")}
        result[group] = summary
    return result


def audited_documents(path):
    result = {}
    if path:
        for key, docs in json.loads(path.read_bytes()).items():
            result[key] = set(docs)
    historical = json.loads((VALIDATION / "semantic_audit.json").read_bytes())
    for group, record in historical["groups"].items():
        for row in record["questions"]:
            question_id = row["question_id"]
            scope = group if group == "holdout" else question_id.split("-")[1][0]
            reviewed = row.get("reviewed_irrelevant_factual_documents", {})
            docs = {d["document_id"] for items in reviewed.values() for d in items}
            result.setdefault(f"{scope}/{question_id}", set()).update(docs)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=ROOT)
    parser.add_argument("--score-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path)
    parser.add_argument("--e2e", nargs=3, metavar=("SCORE", "TOP_K", "POLICY"), action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    for name, digest in FIXED_FILES.items():
        read_pinned(ROOT / "src/cgraph" / name, digest)
    archive = json.loads(read_pinned(BASELINE, BASELINE_SHA256))
    corpus = documents()
    refs = question_rows(archive, corpus)
    questions = {key: row["question"] for key, row in refs.items()}
    off_topic = audited_documents(args.manual_audit)
    assets = args.assets_root.resolve()
    cache_dir = assets / ".cache/semantic_concept_sweep"
    cache = load_entity_cache(cache_dir)
    settings = archive["settings"]
    limits = TraversalLimits()
    require(asdict(limits) == settings["traversal"] and cache.model == settings["e5"]
            and cache.fingerprint == archive["preflight"]["entity_cache_fingerprint"], "Frozen settings differ")
    inference, scores = prepare(cache, assets / "artifacts/ner_v2/models/gliner25",
                                questions, args.score_cache, settings)
    production = EntityLinker.for_retrieval(cache)
    e2e_configs = list(dict.fromkeys([BASE, *((float(s), int(k), p) for s, k, p in args.e2e)]))
    configs = list(dict.fromkeys([*((s, k, "baseline") for s in THRESHOLDS for k in TOP_K),
                                 *((BASE[0], BASE[1], p) for p in POLICIES), *e2e_configs]))
    result = {"source_commit": SOURCE_COMMIT, "fixed_files_sha256": FIXED_FILES, "settings": settings,
              "inputs": {"gold_sha256": _sha(GOLD), "holdout_sha256": _sha(HOLDOUT),
                         "archive_sha256": BASELINE_SHA256, "linker_sha256": inference["inputs"]["linker_sha256"]},
              "scope": list(questions), "score_cache": {"unique_texts": len(scores),
                  "entity_count": len(cache.entities), "full_cosine_scores": len(scores) * len(cache.entities),
                  "array_sha256": inference["scores_sha256"], "array_bytes": sum(row.nbytes for row in scores.values()),
                  "e5_load_seconds": inference["e5_load_seconds"], "memory_ner": inference["memory_ner"],
                  "memory_e5": inference["memory_e5"], "torch": inference["torch"], "python": sys.version},
              "production_profile": {name: getattr(production, name) for name in ("min_score", "top_k", "fallback")},
              "audited_off_topic_documents_by_question": {key: sorted(v) for key, v in off_topic.items()},
              "diagnostic_columns": ["entity_id", "document_id", "linker_rank", "cosine_rank", "cosine"],
              "entity_names": {row["entity_id"]: row["text_name"] for row in cache.entities},
              "sweep": [], "retrieval": []}
    linked = {}
    for config in tqdm(configs, desc="Offline linking sweep", unit="config"):
        options = dict(zip(("min_score", "top_k", "fallback"), config))
        rows = link_questions(config, questions, cache, scores, inference, limits.max_seeds)
        metrics = {key: linking_metrics(row, {**refs[key], "cutoff": config[0]}, off_topic.get(key, set()))
                   for key, row in rows.items()}
        result["sweep"].append({"config": options, "summary": aggregate_links(metrics), "questions": metrics})
        if config in e2e_configs:
            linked[config] = rows
    result["diagnostics"] = diagnostics(questions, refs, inference["analyses"], cache, scores)
    result["memory_after_linking"] = memory()

    print("ВНИМАНИЕ: ПРОВЕРКА СНИМКА И ПРОГОН КОНФИГУРАЦИЙ ЧЕРЕЗ BFS И RERANKER A.", flush=True)
    graph = snapshot(cache_dir, corpus)
    retriever, extractor = GraphRetriever(graph), EvidenceExtractor(graph)
    archive_parity(archive, graph, retriever, extractor, limits, linked[BASE])
    result["graph"] = {"mode": "frozen_snapshot", "fingerprint": graph.fingerprint,
                       "snapshot_sha256": audit.EXPECTED_SNAPSHOT_SHA256,
                       "live_archive_parity": {"ordered_paths": 2000, "a_positions": 50}}
    for config in tqdm(e2e_configs, desc="Fixed BFS and reranker A", unit="config"):
        rows = evaluate_retrieval(linked[config], refs, questions, retriever, extractor, limits, off_topic)
        result["retrieval"].append({"config": dict(zip(("min_score", "top_k", "fallback"), config)),
                                    "summary": aggregate_retrieval(rows), "questions": rows})
    result["memory_after_retrieval"] = memory()
    write_json(args.output, result)


if __name__ == "__main__":
    main()
