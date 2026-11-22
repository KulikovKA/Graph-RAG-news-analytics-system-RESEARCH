from copy import deepcopy
import json

import pytest

from cgraph.evidence_extraction import extract_evidence
from cgraph.path_reranking import rerank_paths_a as rerank_paths
from cgraph.retrieval_types import TARGET_NAMESPACE as NS
from test_evidence_extraction import change_source, document, edge, non_factual, path


def candidate(text="RNA uses ATP.", identity="r1", document_id="d1", reverse=False):
    doc = document(text, document_id)
    e = edge(doc, identity, reverse=reverse)
    change_source(e, lambda s: s[0]["evidence"][0].update(start=0, end=len(text), text=text))
    return path([e]), doc


def retrieval(paths, truncated=False):
    return dict(namespace=NS, graph_fingerprint="fp", paths=paths, truncated=truncated,
                diagnostics=dict(nodes_examined=12, relationships_examined=8,
                                 limits_hit=["max_candidates"] if truncated else []))


def rank(paths, docs, question="What uses ATP?", seeds=(), top_k=5):
    r = retrieval(paths)
    return rerank_paths(question, r, extract_evidence(paths, docs, graph_fingerprint="fp"), seeds, top_k=top_k)


def seed(p, score=0.8, doc_id="d1"):
    return dict(node_key=p["seed_key"], entity_id=p["seed_key"][len(NS) + 1:], document_id=doc_id,
                mention_id=None, score=score, match_method="synthetic")


def test_relevant_validated_path_wins_and_breakdown_sums():
    relevant, doc = candidate()
    unrelated, other = candidate("RNA binds proteins.", "r2", "d2")
    result = rank([unrelated, relevant], {"d1": doc, "d2": other}, seeds=[seed(relevant)])
    first, second = result["paths"]
    assert first["path"] == relevant and first["score"] > second["score"]
    assert first["metrics"]["question_term_coverage"] == 1
    assert first["metrics"]["seed_score"] == 0.8
    assert first["score_breakdown"]["seed_linking"] == pytest.approx(0.4)
    for item in result["paths"]:
        assert item["score"] == pytest.approx(sum(item["score_breakdown"].values()))
    json.dumps(result, allow_nan=False)


def test_invalid_evidence_cannot_be_compensated_by_seed_or_similarity():
    p, doc = candidate()
    e = p["edges"][0]
    e["properties"]["similarity"] = 1.0
    change_source(e, lambda s: s[0]["evidence"][0].update(text="false quote"))
    result = rank([p], {"d1": doc}, seeds=[seed(p, score=1e100)])
    assert result["paths"] == [] and result["reason"] == "no_usable_factual_paths"
    assert result["excluded"][0]["reason"] == "incomplete_factual_evidence"
    assert any("text_mismatch" in r for r in result["excluded"][0]["evidence_reasons"])


def test_invalid_window_terms_do_not_contribute_to_any_score():
    good, doc = candidate("RNA uses ATP.")
    bad, other = candidate("RNA binds proteins.", "r2", "d2")
    change_source(bad["edges"][0], lambda s: s[0]["evidence"].append(
        dict(document_id="d2", window_id="fake", start=0, end=3, text="ATP ATP ATP")))
    result = rank([bad, good], {"d1": doc, "d2": other})
    assert [r["path_id"] for r in result["paths"]] == [good["path_id"]]
    assert result["excluded"][0]["reason"] == "incomplete_factual_evidence"


def test_longer_path_does_not_gain_unbounded_factual_bonus():
    short, doc = candidate()
    next_edge = edge(doc, "r2")
    next_edge.update(head_key=NS + ":b", tail_key=NS + ":c")
    next_edge["properties"].update(head_id="b", tail_id="c")
    change_source(next_edge, lambda s: s[0]["evidence"][0].update(start=0, end=len(doc["text"]), text=doc["text"]))
    long = path(short["edges"] + [next_edge])
    result = rank([long, short], {"d1": doc})
    assert result["paths"][0]["path"] == short
    assert result["paths"][1]["score_breakdown"]["extra_hops"] == -0.15
    assert result["paths"][0]["score_breakdown"]["validated_factual"] == result["paths"][1]["score_breakdown"]["validated_factual"]
    assert result["paths"][1]["score_breakdown"]["repeated_windows"] < 0


def test_semantic_and_concept_transitions_are_navigation_only():
    short, doc = candidate()
    semantic = non_factual("SIMILAR_TO", "b", "c", "s", resolution_decision_2f2="RELATED", similarity=1.0,
                           evidence_json='{"text":"ATP fact"}')
    long = path(short["edges"] + [semantic])
    result = rank([short, long], {"d1": doc})
    assert len(result["paths"]) == 1 and result["paths"][0]["path"] == short
    assert result["excluded"][0]["reason"] == "duplicate_factual_sources"
    single_result = rank([long], {"d1": doc})["paths"][0]
    assert single_result["metrics"]["validated_factual_edges"] == 1
    assert single_result["metrics"]["non_factual_edges"] == 1
    assert single_result["score_breakdown"]["non_factual_transitions"] == -0.2
    m1 = non_factual("HAS_SHARED_CONCEPT", "a", "concept:c", "m1", document_ids=["d1"])
    m2 = non_factual("HAS_SHARED_CONCEPT", "b", "concept:c", "m2", reverse=True, document_ids=["d2"])
    only_navigation = rank([path([semantic]), path([m1, m2])], {})
    assert only_navigation["paths"] == []
    assert all(e["reason"] == "no_factual_evidence" for e in only_navigation["excluded"])


def test_semantic_metadata_cannot_supply_relevance_terms():
    p, doc = candidate("RNA binds proteins.")
    semantic = non_factual("SIMILAR_TO", "b", "c", "s", resolution_decision_2f2="RELATED",
                           similarity=1.0, left_name="ATP", right_name="ATP", evidence_json='{"text":"ATP"}')
    result = rank([path(p["edges"] + [semantic])], {"d1": doc}, question="ATP?")
    assert result["paths"][0]["metrics"]["question_term_coverage"] == 0


def test_cross_document_concept_bridge_preserves_only_the_actual_fact_source():
    factual, doc = candidate(document_id="d2")
    bridge = [non_factual("HAS_SHARED_CONCEPT", "a", "concept:c", "m1", document_ids=["d1"],
                          factual=False, fact_inheritance_allowed=False),
              non_factual("HAS_SHARED_CONCEPT", "d2:a", "concept:c", "m2", reverse=True,
                          document_ids=["d2"], factual=False, fact_inheritance_allowed=False)]
    p = path(bridge + factual["edges"])
    p["document_ids"] = ["d1", "d2"]
    item = rank([p], {"d2": doc}, seeds=[seed(p, doc_id="d1")])["paths"][0]
    assert item["metrics"]["validated_factual_edges"] == 1
    assert item["metrics"]["non_factual_edges"] == 2
    assert item["score_breakdown"]["extra_hops"] == -0.3
    assert item["score_breakdown"]["non_factual_transitions"] == -0.4
    assert [d["document_id"] for d in item["evidence"]["documents"]] == ["d2"]
    fact = item["evidence"]["edges"][2]
    assert fact["edge"]["head_key"] == NS + ":d2:a"
    assert fact["source_relations"][0]["windows"][0]["document_id"] == "d2"
    assert all(e["source_relations"] == [] for e in item["evidence"]["edges"][:2])


def test_reverse_traversal_keeps_original_fact_and_predicate_in_ranked_result():
    p, doc = candidate(reverse=True)
    item = rank([p], {"d1": doc})["paths"][0]
    fact = item["path"]["edges"][0]
    assert fact["traversed_reverse"] is True
    assert fact["relation_type"] == "USES"
    assert fact["head_key"] == NS + ":a" and fact["tail_key"] == NS + ":b"
    assert item["evidence"]["edges"][0]["edge"] == fact


def test_same_fact_reversed_or_with_navigation_does_not_fill_top_five():
    p, doc = candidate()
    reverse, _ = candidate(reverse=True)
    paths = [p, reverse]
    for i in range(6):
        s = non_factual("SIMILAR_TO", "b", "c" + str(i), "s" + str(i), resolution_decision_2f2="RELATED")
        paths.append(path(p["edges"] + [s]))
    result = rank(paths + [p] * 5, {"d1": doc})
    assert len(result["paths"]) == 1
    assert result["ranking_diagnostics"]["duplicate_inputs"] == 5
    assert len(result["excluded"]) == 7
    assert all(e["reason"] == "duplicate_factual_sources" for e in result["excluded"])


def test_parallel_facts_with_same_endpoints_document_and_window_survive():
    p1, doc = candidate(identity="fact1")
    p2, _ = candidate(identity="fact2")
    result = rank([p1, p2], {"d1": doc})
    assert len(result["paths"]) == 2
    assert {p["path_id"] for p in result["paths"]} == {p1["path_id"], p2["path_id"]}
    assert result["paths"][1]["score_breakdown"]["diversity"] < 0


def test_diversity_prefers_new_document_over_shared_window():
    p1, doc = candidate(identity="fact1")
    p2, _ = candidate(identity="fact2")
    p3, other = candidate(identity="fact3", document_id="d3")
    result = rank([p1, p2, p3], {"d1": doc, "d3": other})
    assert {p["path"]["edges"][0]["document_id"] for p in result["paths"][:2]} == {"d1", "d3"}


def test_determinism_tie_break_top_k_and_no_input_mutation():
    pairs = [candidate(identity="r" + str(i), document_id="d" + str(i)) for i in range(7)]
    paths = [p for p, _ in pairs]
    docs = {d["document_id"]: d for _, d in pairs}
    r = retrieval(paths, truncated=True)
    bundles = extract_evidence(paths, docs, graph_fingerprint="fp")
    original = deepcopy((r, bundles, docs))
    result = rerank_paths("What uses ATP?", r, bundles)
    permuted = deepcopy(r)
    permuted["paths"].reverse()
    assert result == rerank_paths("What uses ATP?", permuted, list(reversed(bundles)))
    assert [p["path_id"] for p in result["paths"]] == sorted(p["path_id"] for p in paths)[:5]
    assert (r, bundles, docs) == original
    assert len(result["paths"]) == 5
    assert len(rank(paths, docs, top_k=2)["paths"]) == 2


def test_fewer_than_five_and_empty_input():
    p, doc = candidate()
    assert len(rank([p], {"d1": doc})["paths"]) == 1
    assert rank([], {})["reason"] == "empty_input"
    assert rank([], {})["paths"] == []
    assert rank([p], {"d1": doc}, question="  ")["reason"] == "empty_question_terms"


def test_truncation_and_original_diagnostics_are_forwarded_even_on_abstention():
    r = retrieval([], truncated=True)
    original = deepcopy(r)
    result = rerank_paths("ATP?", r, [])
    assert result["truncated"] is True
    assert result["diagnostics"] == r["diagnostics"]
    assert result["graph_fingerprint"] == r["graph_fingerprint"]
    result["diagnostics"]["limits_hit"].append("changed")
    assert r == original


def test_seed_score_is_optional_bounded_and_uses_separate_contract():
    p, doc = candidate()
    assert "score" not in p
    assert rank([p], {"d1": doc})["paths"][0]["score_breakdown"]["seed_linking"] == 0
    seeds = [seed(p, -1), seed(p, 1e100)]
    result = rank([p], {"d1": doc}, seeds=seeds)
    assert result == rank([p], {"d1": doc}, seeds=list(reversed(seeds)))
    assert result["paths"][0]["score_breakdown"]["seed_linking"] == 0.5
    with pytest.raises(ValueError, match="finite"):
        rank([p], {"d1": doc}, seeds=[seed(p, float("nan"))])


def test_wrong_or_missing_evidence_cannot_attach_to_path():
    p, doc = candidate()
    r = retrieval([p])
    assert rerank_paths("ATP", r, [])["excluded"][0]["reason"] == "missing_evidence"
    bundles = extract_evidence([p], {"d1": doc}, graph_fingerprint="fp")
    bundles[0]["edges"][0]["edge"]["properties"]["source_json"] = "changed"
    result = rerank_paths("ATP", r, bundles)
    assert result["excluded"][0]["reason"] == "evidence_path_mismatch"


def test_untrusted_semantic_transition_cannot_be_rescued_by_valid_fact():
    p, doc = candidate()
    bad = non_factual("SIMILAR_TO", "b", "c", "s", resolution_decision_2f2="REJECT", similarity=1.0)
    result = rank([path(p["edges"] + [bad])], {"d1": doc})
    assert result["paths"] == []
    assert result["excluded"][0]["reason"] == "untrusted_navigation"


def test_evidence_from_other_graph_revision_is_not_reused():
    p, doc = candidate()
    doc["fingerprint"] = "different-snapshot"
    bundles = extract_evidence([p], {"d1": doc})
    assert bundles[0]["status"] == "valid"
    result = rerank_paths("ATP", retrieval([p]), bundles)
    assert result["paths"] == []
    assert result["excluded"][0]["reason"] == "evidence_graph_fingerprint_mismatch"


@pytest.mark.parametrize("top_k", [0, 6, -1, 1.0, True])
def test_top_k_validation(top_k):
    with pytest.raises(ValueError, match="top_k"):
        rerank_paths("ATP", retrieval([]), [], top_k=top_k)
