"""Document-aware selection for provenance-gated ranking candidates."""

from collections import defaultdict
from math import fsum
import re

from .path_reranking import DIVERSITY_WEIGHT, RELEVANCE_WEIGHT, _overlap, _terms


_C_MARGIN = 1.0
_C_SECOND_WEIGHT = 0.15
_C_DOCUMENT_SLOTS = 3
_C2_ENTITY_WEIGHT = 1.8
_C2_PREDICATE_WEIGHT = 0.35
_C2_LOCAL_WEIGHT = 1.1
_C2_WINDOW_WEIGHT = 0.45
_C2_DIVERSITY_WEIGHT = 0.20
_C2_NOVELTY = 0.30
_C2_MARGIN = 0.70
_C2_DOCUMENT_MIN = 0.40
_GENERIC = frozenset("study paper research researcher author describe report different role method approach".split())
_PREDICATE_CUES = (
    ({"result", "outcome", "effect", "gain", "improve", "increase", "decrease", "reduce", "risk",
      "complication", "performance", "limit", "cause", "reason", "because"},
     {"HAS_RESULT", "CAUSES", "DEPENDS_ON", "LIMITS", "INCREASES", "DECREASES", "REDUCES"}),
    ({"use", "capture", "measure", "map", "instrument", "test", "monitor", "evaluate", "estimate",
      "appl", "method", "detect", "automat", "tool", "technique"},
     {"USES", "APPLIED_TO", "MEASURES", "PERFORMS", "EVALUATED_ON", "DERIVED_FROM"}),
    ({"different", "compare", "comparison", "similar", "versus"}, {"COMPARED_WITH"}),
    ({"dominant", "dominate", "highest", "higher", "where", "location", "located"},
     {"LOCATED_IN", "HAS_ATTRIBUTE", "HAS_MEMBER", "MEASURES"}),
)
_SENTENCE = re.compile(r".+?(?:[.!?](?=\s|$)|\n|$)", re.DOTALL)


def _stem(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ing", "ed")):
        return word[:-3] if word.endswith("ing") else word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _fact_terms(text):
    return {_stem(t) for t in _terms(text)}


def _entity_terms(nodes, edge, field, fingerprint):
    key = edge[field]
    node = nodes.get(key, {})
    if (node.get("key") != key or node.get("fingerprint") != fingerprint
            or node.get("document_id") != edge["document_id"]
            or node.get("namespace") != edge["properties"].get("namespace")
            or key != f"{node.get('namespace')}:{node.get('entity_id')}"):
        return set()
    name = node.get("name")
    if not isinstance(name, str) or name == key or name == node.get("entity_id"):
        return set()
    return _fact_terms(name)


def _localized_evidence(item, query, head, tail, window_cache):
    best = (0.0, None)
    for source in item["source_relations"]:
        for w in source["windows"]:
            identity = (w["document_id"], w["revision_id"], w["window_id"], w["start"], w["end"])
            if identity not in window_cache:
                spans = []
                for match in _SENTENCE.finditer(w["text"]):
                    text = match.group().strip()
                    if text:
                        start = match.start() + len(match.group()) - len(match.group().lstrip())
                        spans.append((start, text, _fact_terms(text)))
                window_cache[identity] = spans
            for start, text, terms in window_cache[identity]:
                anchors = int(bool(head and head <= terms)) + int(bool(tail and tail <= terms))
                if not anchors:
                    continue
                value = min(1.0, 2 * len(query & terms) / len(query)) * (1.0 if anchors == 2 else 0.35)
                locator = (identity, start, source["source_triple_id"])
                if value > best[0] or (value and value == best[0] and locator < best[1][0]):
                    best = (value, (locator, {"document_id": w["document_id"], "window_id": w["window_id"],
                                             "source_triple_id": source["source_triple_id"],
                                             "revision_id": w["revision_id"], "text_sha256": w["text_sha256"],
                                             "start": w["start"] + start, "end": w["start"] + start + len(text),
                                             "text": text, "anchor_count": anchors}))
    return best[0], best[1][1] if best[1] else None


def _fact_relevance(item, query, predicates, nodes, fingerprint, window_cache):
    edge = item["edge"]
    head = _entity_terms(nodes, edge, "head_key", fingerprint)
    tail = _entity_terms(nodes, edge, "tail_key", fingerprint)
    matches = query & (head | tail)
    entity = min(1.0, 2 * len(matches) / len(query))
    local, span = _localized_evidence(item, query, head, tail, window_cache)
    predicate = float(bool(matches) and edge["relation_type"] in predicates)
    components = {"entity_relevance": _C2_ENTITY_WEIGHT * entity,
                  "predicate_relevance": _C2_PREDICATE_WEIGHT * predicate,
                  "local_evidence_relevance": _C2_LOCAL_WEIGHT * local}
    return {"document_id": edge["document_id"], "edge_key": edge["key"],
            "components": components, "score": fsum(components.values()),
            "matched_entity_terms": sorted(matches), "local_span": span}


def _add_document_features(candidates, question, policy, fingerprint, nodes):
    original_query = _terms(question)
    terms = _fact_terms(question)
    query = terms - {_stem(t) for t in _GENERIC} or terms
    predicates = {p for cues, relations in _PREDICATE_CUES if terms & {_stem(c) for c in cues} for p in relations}
    edges, windows = {}, {}
    for c in candidates:
        sources, words, doc_scores = defaultdict(set), defaultdict(set), defaultdict(float)
        edge_features = []
        for item in c["evidence"]["edges"]:
            edge = item["edge"]
            if edge["type"] != "RELEX_REL":
                continue
            doc = edge["document_id"]
            for source in item["source_relations"]:
                sources[doc].add((doc, source["source_triple_id"]))
                if policy == "C":
                    for window in source["windows"]:
                        words[doc].update(_terms(window["text"]))
            if policy == "C2":
                if edge["key"] not in edges:
                    edges[edge["key"]] = _fact_relevance(item, query, predicates, nodes, fingerprint, windows)
                f = edges[edge["key"]]
                edge_features.append(f)
                if f["matched_entity_terms"] and f["local_span"]:
                    doc_scores[doc] = max(doc_scores[doc], f["score"])
        c["sources_by_doc"] = sources
        c["source_ids"] = set().union(*sources.values())
        if policy == "C":
            c["document_coverage"] = {d: len(original_query & w) / len(original_query) for d, w in words.items()}
            continue
        best = min(edge_features, key=lambda f: (-f["score"], f["edge_key"]))
        c["document_coverage"] = dict(doc_scores)
        c["score_breakdown"]["relevance"] *= _C2_WINDOW_WEIGHT / RELEVANCE_WEIGHT
        c["score_breakdown"].update(best["components"])
        c["base_score"] = round(fsum(c["score_breakdown"].values()), 12)
        c["metrics"]["fact_relevance"] = {"best_edge_key": best["edge_key"], "edges": edge_features}


def _document_order(candidates):
    grouped = defaultdict(list)
    for c in candidates:
        for d, coverage in c["document_coverage"].items():
            if coverage > 0:
                grouped[d].append(c)
    scores = {}
    for doc, paths in grouped.items():
        paths.sort(key=lambda c: (-c["base_score"], c["path_id"]))
        first = paths[0]
        first_sources = first["sources_by_doc"][doc]
        second = next((c for c in paths[1:] if first_sources.isdisjoint(c["sources_by_doc"][doc])), None)
        bonus = _C_SECOND_WEIGHT * second["document_coverage"][doc] if second else 0.0
        scores[doc] = round(fsum((first["base_score"], bonus)), 12)
    return sorted(scores, key=lambda d: (-scores[d], d))


def select_documents(question, candidates, top_k, excluded, policy, fingerprint, nodes):
    _add_document_features(candidates, question, policy, fingerprint, nodes or {})
    best_base = max((c["base_score"] for c in candidates), default=0.0)
    qualified = [c for c in candidates if c["base_score"] >= best_base - _C_MARGIN
                 and c["metrics"]["question_term_coverage"] > 0]
    qualified_ids = {c["path_id"] for c in qualified}
    order = _document_order(qualified) if policy == "C" else []
    selected, seen_docs, seen_sources = [], set(), set()
    remaining = candidates
    while remaining and len(selected) < top_k:
        fresh = [c for c in remaining if c["source_ids"] - seen_sources]
        for c in remaining:
            if not c["source_ids"] - seen_sources:
                excluded.append({"path_id": c["path_id"], "reason": "duplicate_factual_sources",
                                 "evidence_reasons": []})
        remaining = fresh
        if not remaining:
            break
        pool = remaining
        leader = None
        phase = "fact_relevance" if policy == "C2" else "original_score_fallback"
        if policy == "C":
            strong = [c for c in remaining if c["path_id"] in qualified_ids]
            if strong:
                pool, phase = strong, "quality_selection"
            if len(selected) < _C_DOCUMENT_SLOTS:
                for doc in order:
                    if doc not in seen_docs:
                        group = [c for c in strong if c["document_coverage"].get(doc, 0) > 0]
                        if group:
                            pool, leader, phase = group, doc, "document_first"
                            break
        for c in pool:
            overlap = max((_overlap(c["features"], p["features"]) for p in selected), default=0.0)
            breakdown = c["score_breakdown"]
            breakdown["diversity"] = -(DIVERSITY_WEIGHT if policy == "C" else _C2_DIVERSITY_WEIGHT) * overlap
            if policy == "C2":
                useful = any(d not in seen_docs and value >= _C2_DOCUMENT_MIN
                             for d, value in c["document_coverage"].items())
                # Разнообразие влияет лишь на сопоставимые пути с локальным фактическим сигналом.
                breakdown["document_novelty"] = (_C2_NOVELTY if selected and useful
                                                 and c["base_score"] >= best_base - _C2_MARGIN else 0.0)
            c["score"] = round(fsum(breakdown.values()), 12)
        best = min(pool, key=lambda c: (-c["score"], c["path_id"]))
        docs = best["features"]["documents"]
        best.update(selection_phase=phase, primary_document=leader,
                    new_factual_documents=sorted(docs - seen_docs),
                    new_source_triples=len(best["source_ids"] - seen_sources))
        selected.append(best)
        seen_docs.update(docs)
        seen_sources.update(best["source_ids"])
        remaining = [c for c in remaining if c["path_id"] != best["path_id"]]
    excluded.extend({"path_id": c["path_id"], "reason": "outside_top_k", "evidence_reasons": []} for c in remaining)
    return selected
