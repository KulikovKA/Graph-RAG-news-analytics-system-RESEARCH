"""Experimental offline selection over paths scored by the original reranker."""

from collections import defaultdict
from math import fsum

from cgraph.path_reranking import DIVERSITY_WEIGHT, _features, _overlap, _terms, rerank_paths_a as rerank_paths


BASE_SCORE_MARGIN = 1.0
REPEATED_DOCUMENT_PENALTY = 0.65
SECOND_PATH_WEIGHT = 0.15
DOCUMENT_FIRST_SLOTS = 3
TOP_K = 5


def score_candidates(question, retrieval, evidence, seeds):
    """Use the original gate and scoring on singletons, where diversity is zero."""
    bundles = {b["path_id"]: b for b in evidence}
    query_terms = _terms(question)
    candidates = []
    for path in sorted(retrieval["paths"], key=lambda p: p["path_id"]):
        bundle = bundles.get(path["path_id"])
        result = rerank_paths(question, {**retrieval, "paths": [path]}, [bundle] if bundle else [],
                              seeds, top_k=1)
        if not result["paths"]:
            continue
        [candidate] = result["paths"]
        features = _features(bundle)
        sources_by_doc, terms_by_doc = defaultdict(set), defaultdict(set)
        for item in bundle["edges"]:
            if item["edge"]["type"] != "RELEX_REL":
                continue
            for source in item["source_relations"]:
                for window in source["windows"]:
                    doc = window["document_id"]
                    sources_by_doc[doc].add((doc, source["source_triple_id"]))
                    terms_by_doc[doc].update(_terms(window["text"]))
        candidate.update(features=features, sources_by_doc=dict(sources_by_doc),
                         source_ids=set().union(*sources_by_doc.values()),
                         document_coverage={d: len(query_terms & terms) / len(query_terms)
                                            for d, terms in terms_by_doc.items()})
        candidates.append(candidate)
    return candidates


def document_profiles(qualified):
    grouped = defaultdict(list)
    for candidate in qualified:
        for document, coverage in candidate["document_coverage"].items():
            if coverage > 0:
                grouped[document].append(candidate)

    profiles = {}
    for document, candidates in grouped.items():
        paths = sorted(candidates, key=lambda c: (-c["base_score"], c["path_id"]))
        first = paths[0]
        first_sources = first["sources_by_doc"][document]
        second = next((c for c in paths[1:] if first_sources.isdisjoint(c["sources_by_doc"][document])), None)
        # Второй путь даёт ограниченную прибавку только за непересекающиеся source triples.
        bonus = SECOND_PATH_WEIGHT * second["document_coverage"][document] if second else 0.0
        profiles[document] = {"best_path_id": first["path_id"], "best_path_base_score": first["base_score"],
                              "second_independent_path_id": second["path_id"] if second else None,
                              "second_independent_path_bonus": bonus,
                              "score": round(fsum((first["base_score"], bonus)), 12)}
    return profiles


def score_for_slot(candidate, selected, seen_documents, repeat_penalty):
    overlap = max((_overlap(candidate["features"], p["features"]) for p in selected), default=0.0)
    breakdown = {**candidate["score_breakdown"], "diversity": -DIVERSITY_WEIGHT * overlap}
    if repeat_penalty:
        documents = candidate["features"]["documents"]
        breakdown["repeated_document"] = -repeat_penalty * len(documents & seen_documents) / len(documents)
    return round(fsum(breakdown.values()), 12), breakdown


def select_paths(candidates, policy, *, top_k=TOP_K):
    """B penalizes repeated documents. C gives up to three slots to strong document leaders."""
    if policy not in {"B", "C"} or type(top_k) is not int or not 1 <= top_k <= TOP_K:
        raise ValueError("Expected policy B or C and top_k in [1, 5]")
    best_base = max((c["base_score"] for c in candidates), default=0.0)
    qualified = [c for c in candidates if c["base_score"] >= best_base - BASE_SCORE_MARGIN
                 and c["metrics"]["question_term_coverage"] > 0]
    qualified_ids = {c["path_id"] for c in qualified}
    profiles = document_profiles(qualified) if policy == "C" else {}
    document_order = sorted(profiles, key=lambda d: (-profiles[d]["score"], d))
    selected, seen_documents, seen_sources = [], set(), set()
    seen_fact_sets = set()
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        if policy == "C":
            # Путь без нового исходного триплета не увеличивает полезность выдачи.
            remaining = [c for c in remaining if c["source_ids"] - seen_sources]
        else:
            remaining = [c for c in remaining if frozenset(c["features"]["facts"]) not in seen_fact_sets]
        if not remaining:
            break
        strong = [c for c in remaining if c["path_id"] in qualified_ids]
        pool = strong or remaining
        phase = "quality_selection" if strong else "original_score_fallback"
        leader = None
        if policy == "C" and len(selected) < DOCUMENT_FIRST_SLOTS:
            for document in document_order:
                if document in seen_documents:
                    continue
                group = [c for c in strong if c["document_coverage"].get(document, 0) > 0]
                if group:
                    pool, leader, phase = group, document, "document_first"
                    break
        repeat_penalty = REPEATED_DOCUMENT_PENALTY if policy == "B" and strong else 0.0
        scored = [(c, *score_for_slot(c, selected, seen_documents, repeat_penalty)) for c in pool]
        candidate, score, breakdown = min(scored, key=lambda row: (-row[1], row[0]["path_id"]))
        documents = candidate["features"]["documents"]
        selected.append({**candidate, "score": score, "score_breakdown": breakdown, "rank": len(selected) + 1,
                         "selection_phase": phase, "primary_document": leader,
                         "new_factual_documents": sorted(documents - seen_documents),
                         "new_source_triples": len(candidate["source_ids"] - seen_sources)})
        seen_documents.update(documents)
        seen_sources.update(candidate["source_ids"])
        seen_fact_sets.add(frozenset(candidate["features"]["facts"]))
        remaining = [c for c in remaining if c["path_id"] != candidate["path_id"]]
    return selected, {d: profiles[d] for d in document_order}
