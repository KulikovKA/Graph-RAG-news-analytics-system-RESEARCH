"""Stage 3A.4 deterministic, evidence-gated ranking. Scores are not probabilities.

Ranking uses verified quotations and, for C2, optional snapshot entity names.
Opaque keys and semantic similarities never supply factual relevance. Seed
scores express linking relevance only. All factual edges must validate fully.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from typing import Iterable, Literal, Mapping

from .retrieval_types import RetrievalResult, Seed, validate_graph_path, validate_seed


# Покрытие терминов доминирует; factual-бонус ограничен, а не растёт с числом рёбер.
RELEVANCE_WEIGHT = 3.0
FACTUAL_WEIGHT = 1.0
SEED_WEIGHT = 0.5
# Небольшие штрафы за каждый лишний hop и каждый нефактический переход.
EXTRA_HOP_PENALTY = 0.15
NON_FACTUAL_PENALTY = 0.2
REPETITION_PENALTY = 0.5
# Пересечение рёбер важнее окон и документов; независимые факты не удаляются.
DIVERSITY_WEIGHT = 0.75
OVERLAP_WEIGHTS = (0.5, 0.3, 0.2)
_STOP_WORDS = frozenset("""
    a an the and or of to in on at by for from with as is are was were be been
    do does did how what which who when where why can could would should it its
    this that these those about into between than then through using used
    и или в во на к по из с со от для как что какие какой какая это ли чем
""".split())


def _terms(text):
    return {t for t in re.findall(r"[^\W_]+", text.casefold()) if len(t) > 1 and t not in _STOP_WORDS}


def _evidence_error(path, bundle, fingerprint):
    if bundle is None:
        return "missing_evidence"
    edges = bundle.get("edges", [])
    if (bundle.get("path_id") != path["path_id"] or len(edges) != len(path["edges"])
            or any(e.get("edge") != original for e, original in zip(edges, path["edges"]))):
        return "evidence_path_mismatch"
    if (any(e["properties"].get("fingerprint") != fingerprint for e in path["edges"])
            or any(d.get("fingerprint") != fingerprint for d in bundle.get("documents", []))):
        return "evidence_graph_fingerprint_mismatch"
    if not bundle.get("navigation_valid"):
        return "untrusted_navigation"
    facts = [e for e in edges if e["edge"]["type"] == "RELEX_REL"]
    if not facts:
        return "no_factual_evidence"
    for e in facts:
        if e.get("non_factual") is not False or e.get("status") != "valid":
            return "incomplete_factual_evidence"
        sources = e.get("source_relations", [])
        if not sources or any(s.get("status") != "valid" or not s.get("windows")
                              or any(w.get("status") != "valid" for w in s["windows"]) for s in sources):
            return "incomplete_factual_evidence"
    if any(e.get("status") != "non_factual" or e.get("non_factual") is not True
           for e in edges if e["edge"]["type"] != "RELEX_REL"):
        return "invalid_non_factual_transition"
    return None


def _features(bundle):
    fact_ids, edge_ids, documents, windows, terms = set(), set(), set(), set(), set()
    for item in bundle["edges"]:
        edge = item["edge"]
        # Метаданные похожести и концептов не участвуют в текстовом evidence.
        if edge["type"] != "RELEX_REL":
            continue
        edge_ids.add(edge["key"])
        for source in item["source_relations"]:
            source_windows = []
            for w in source["windows"]:
                identity = (w["document_id"], w["revision_id"], w["text_sha256"],
                            w["window_id"], w["start"], w["end"])
                source_windows.append(identity)
                windows.add(identity)
                documents.add(w["document_id"])
                terms.update(_terms(w["text"]))
            fact_ids.add((edge["key"], edge["relation_type"], edge["head_key"], edge["tail_key"],
                          source["source_triple_id"], tuple(sorted(source_windows))))
    return {"facts": fact_ids, "edges": edge_ids, "documents": documents, "windows": windows, "terms": terms}


def _overlap(a, b):
    return math.fsum(weight * len(a[field] & b[field]) / len(a[field] | b[field])
                     for weight, field in zip(OVERLAP_WEIGHTS, ("edges", "windows", "documents"))
                     if a[field] or b[field])


RerankingPolicy = Literal["A", "C", "C2"]
DEFAULT_RERANKING_POLICY: RerankingPolicy = "C"


def rerank_paths(question: str, retrieval: RetrievalResult, evidence: Iterable[dict],
                 seeds: Iterable[Seed] = (), *, top_k: int = 5, policy: RerankingPolicy | None = None,
                 entity_nodes: Mapping[str, dict] | None = None) -> dict:
    """Return up to five evidence-backed paths, plus exclusions and diagnostics.

    Evidence must come from evidence_extraction for this exact retrieval.
    C prioritizes relevant factual documents. A is the original path-only
    ranking and C2 is an experimental endpoint-aware policy.
    Retrieval warnings are copied unchanged. Entity context is optional and
    must use the physical snapshot node key and properties. No IO or models.
    """
    policy = DEFAULT_RERANKING_POLICY if policy is None else policy
    if policy not in ("A", "C", "C2"):
        raise ValueError("policy must be A, C or C2")
    if not isinstance(question, str):
        raise TypeError("Question must be a string")
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise ValueError("top_k must be an integer between one and five")
    result = {"namespace": retrieval["namespace"], "graph_fingerprint": retrieval["graph_fingerprint"],
              "paths": [], "truncated": retrieval["truncated"], "diagnostics": deepcopy(retrieval["diagnostics"]),
              "excluded": [], "reason": None,
              "ranking_diagnostics": {"input_paths": len(retrieval["paths"]), "eligible_paths": 0,
                                      "duplicate_inputs": 0, "selected_paths": 0}}
    query_terms = _terms(question)
    if not query_terms:
        result["reason"] = "empty_question_terms"
        return result
    seed_scores = {}
    for seed in seeds:
        validate_seed(seed, retrieval["namespace"])
        previous = seed_scores.get(seed["node_key"])
        if previous is not None and previous["document_id"] != seed["document_id"]:
            raise ValueError("Conflicting seed documents")
        if previous is None or seed["score"] > previous["score"]:
            seed_scores[seed["node_key"]] = seed
    bundles = {}
    for bundle in evidence:
        identity = bundle["path_id"]
        if identity in bundles and bundles[identity] != bundle:
            raise ValueError("Conflicting evidence for path_id")
        bundles[identity] = bundle
    seen, candidates = {}, []
    for path in sorted(retrieval["paths"], key=lambda p: p["path_id"]):
        validate_graph_path(path, retrieval["namespace"])
        identity = path["path_id"]
        if identity in seen:
            if seen[identity] != path:
                raise ValueError("Conflicting GraphPath for path_id")
            result["ranking_diagnostics"]["duplicate_inputs"] += 1
            continue
        seen[identity] = path
        bundle = bundles.get(identity)
        error = _evidence_error(path, bundle, retrieval["graph_fingerprint"])
        if error:
            result["excluded"].append({"path_id": identity, "reason": error,
                                       "evidence_reasons": deepcopy(bundle.get("reasons", [])) if bundle else []})
            continue
        features = _features(bundle)
        matched = sorted(query_terms & features["terms"])
        coverage = len(matched) / len(query_terms)
        factual_count = sum(e["type"] == "RELEX_REL" for e in path["edges"])
        non_factual_count = len(path["edges"]) - factual_count
        seed = seed_scores.get(path["seed_key"])
        if seed is not None and seed["document_id"] not in path["document_ids"]:
            raise ValueError("Seed document absent from path")
        seed_score = seed["score"] if seed is not None else None
        # Диапазон нужен только для ограниченного linking-бонуса, не оценки истинности.
        bounded_seed_score = min(1.0, max(-1.0, seed_score)) if seed_score is not None else 0.0
        repetition = 1 - len(features["windows"]) / sum(
            len(s["windows"]) for e in bundle["edges"] for s in e["source_relations"])
        breakdown = {"relevance": RELEVANCE_WEIGHT * coverage,
                     "validated_factual": FACTUAL_WEIGHT,
                     "seed_linking": SEED_WEIGHT * bounded_seed_score,
                     "extra_hops": -EXTRA_HOP_PENALTY * max(0, len(path["edges"]) - 1),
                     "non_factual_transitions": -NON_FACTUAL_PENALTY * non_factual_count,
                     "repeated_windows": -REPETITION_PENALTY * repetition,
                     "diversity": 0.0}
        candidates.append({"path_id": identity, "path": path, "evidence": bundle,
                           "base_score": round(math.fsum(breakdown.values()), 12),
                           "score": 0.0, "score_breakdown": breakdown,
                           "metrics": {"question_term_coverage": coverage, "matched_terms": matched,
                                       "validated_factual_edges": factual_count,
                                       "non_factual_edges": non_factual_count, "seed_score": seed_score},
                           "features": features})
    result["ranking_diagnostics"]["eligible_paths"] = len(candidates)
    if policy == "A":
        selected = _select_a(candidates, top_k, result["excluded"])
    else:
        from .document_ranking import select_documents

        selected = select_documents(question, candidates, top_k, result["excluded"], policy,
                                    retrieval["graph_fingerprint"], entity_nodes)
    paths = []
    for rank, candidate in enumerate(selected, 1):
        item = {k: candidate[k] for k in ("path_id", "path", "evidence", "base_score", "score",
                                         "score_breakdown", "metrics")}
        for key in ("selection_phase", "primary_document", "new_factual_documents", "new_source_triples"):
            if key in candidate:
                item[key] = candidate[key]
        paths.append(deepcopy({**item, "rank": rank}))
    result["paths"] = paths
    result["excluded"].sort(key=lambda p: (p["path_id"], p["reason"]))
    result["ranking_diagnostics"]["selected_paths"] = len(selected)
    if not selected:
        result["reason"] = "no_usable_factual_paths" if retrieval["paths"] else "empty_input"
    return result


def rerank_paths_a(question: str, retrieval: RetrievalResult, evidence: Iterable[dict],
                   seeds: Iterable[Seed] = (), *, top_k: int = 5) -> dict:
    """Run original Stage 3A.4 policy A explicitly."""
    return rerank_paths(question, retrieval, evidence, seeds, top_k=top_k, policy="A")


def _select_a(candidates, top_k, excluded):
    selected = []
    while candidates and len(selected) < top_k:
        for candidate in candidates:
            overlap = max((_overlap(candidate["features"], p["features"]) for p in selected), default=0.0)
            candidate["score_breakdown"]["diversity"] = -DIVERSITY_WEIGHT * overlap
            candidate["score"] = round(math.fsum(candidate["score_breakdown"].values()), 12)
        candidates.sort(key=lambda p: (-p["score"], p["path_id"]))
        best = candidates.pop(0)
        selected.append(best)
        remaining = []
        for candidate in candidates:
            if candidate["features"]["facts"] == best["features"]["facts"]:
                excluded.append({"path_id": candidate["path_id"], "reason": "duplicate_factual_sources",
                                 "duplicate_of": best["path_id"], "evidence_reasons": []})
            else:
                remaining.append(candidate)
        candidates = remaining
    for candidate in candidates:
        excluded.append({"path_id": candidate["path_id"], "reason": "outside_top_k", "evidence_reasons": []})
    return selected
