"""Compatibility and scoring checks for the production policy switch."""

from copy import deepcopy

import pytest

from cgraph.evidence_extraction import extract_evidence
from cgraph.path_reranking import DEFAULT_RERANKING_POLICY, rerank_paths, rerank_paths_a
from test_evidence_extraction import NS, change_source, document, edge, non_factual, path
from test_path_reranking import retrieval


def example(text, name, doc_id, head, tail, predicate="USES"):
    doc = document(text, doc_id)
    fact = edge(doc, name)
    fact["relation_type"] = fact["properties"]["relation_type"] = predicate
    change_source(fact, lambda sources: sources[0].update(relation_type=predicate))
    change_source(fact, lambda sources: sources[0]["evidence"][0].update(start=2, end=len(text), text=text[2:]))
    nodes = {}
    for field, entity in (("head_key", head), ("tail_key", tail)):
        key = fact[field]
        nodes[key] = {"key": key, "entity_id": key[len(NS) + 1:], "namespace": NS,
                      "document_id": doc_id, "fingerprint": "fp", "name": entity}
    return path([fact]), doc, nodes


def ranked(question, examples, policy=None, *, top_k=5):
    paths, docs, maps = zip(*examples, strict=True)
    nodes = {key: v for one in maps for key, v in one.items()}
    bundles = extract_evidence(paths, {d["document_id"]: d for d in docs}, graph_fingerprint="fp")
    return rerank_paths(question, retrieval(paths), bundles, top_k=top_k, policy=policy, entity_nodes=nodes)


def test_production_uses_c_and_explicit_a_is_original():
    topic = "Optical pulse propagation?"
    rows = [example("β Optical pulse propagation.", f"strong-{i}", "d1", "optical", "pulse") for i in range(5)]
    rows.append(example("β Optical pulse.", "second", "d2", "optical", "pulse"))
    assert DEFAULT_RERANKING_POLICY == "C"
    default = ranked(topic, rows, top_k=2)
    assert default == ranked(topic, rows, "C", top_k=2)
    assert [p["path"]["document_ids"] for p in default["paths"]] == [["d1"], ["d2"]]
    assert [p["path"]["document_ids"] for p in ranked(topic, rows, "A", top_k=2)["paths"]] == [["d1"], ["d1"]]
    paths, docs, _ = zip(*rows, strict=True)
    evidence = extract_evidence(paths, {d["document_id"]: d for d in docs}, graph_fingerprint="fp")
    assert rerank_paths_a(topic, retrieval(paths), evidence) == rerank_paths(topic, retrieval(paths), evidence, policy="A")
    with pytest.raises(ValueError, match="policy"):
        rerank_paths(topic, retrieval(paths), evidence, policy="unknown")


def test_c2_localizes_relevant_fact_and_leaves_original_provenance_intact():
    wanted = example("β Mammographic density depends on the reader. Tumors vary.", "relevant", "d1",
                     "Mammographic density", "reader", "DEPENDS_ON")
    distraction = example("β Mammographic density depends on the reader. Tumors vary.", "distractor", "d2",
                          "Tumors", "size", "HAS_ATTRIBUTE")
    inputs = [distraction, wanted]
    before = deepcopy(inputs)
    result = ranked("Which mammographic density measurement depends on the reader?", inputs, "C2")
    first, second = result["paths"]
    f = first["metrics"]["fact_relevance"]["edges"][0]
    w = first["evidence"]["edges"][0]["source_relations"][0]["windows"][0]
    span = f["local_span"]
    assert span["text"] == "Mammographic density depends on the reader."
    assert wanted[1]["text"][span["start"]:span["end"]] == span["text"]
    assert span["revision_id"] == w["revision_id"] and span["window_id"] == w["window_id"]
    assert span["source_triple_id"] == "t:relevant" and w["end"] > span["end"]
    assert first["path"] == wanted[0] and inputs == before
    assert second["score_breakdown"]["entity_relevance"] == second["score_breakdown"]["document_novelty"] == 0
    assert ranked("Which mammographic density measurement depends on the reader?", inputs[::-1], "C2") == result


def test_predicate_boost_is_query_specific():
    result = example("β Treatment led to recovery.", "result", "d1", "Treatment", "recovery", "HAS_RESULT")
    kind = example("β Treatment led to recovery.", "kind", "d2", "Treatment", "recovery", "IS_A")
    ranked_results = ranked("What was the treatment outcome?", [result, kind], "C2")
    first = ranked_results["paths"][0]
    assert first["path"] == result[0] and first["score_breakdown"]["predicate_relevance"] > 0
    unrelated_intent = ranked("What type of treatment?", [result], "C2")
    assert unrelated_intent["paths"][0]["score_breakdown"]["predicate_relevance"] == 0


def test_context_from_wrong_fingerprint_is_ignored():
    row = example("β Treatment led to recovery.", "result", "d1", "Treatment", "recovery", "HAS_RESULT")
    for n in row[2].values():
        n["fingerprint"] = "other"
    [item] = ranked("What was the treatment outcome?", [row], "C2")["paths"]
    assert item["score_breakdown"]["entity_relevance"] == item["score_breakdown"]["predicate_relevance"] == 0
    assert item["metrics"]["fact_relevance"]["edges"][0]["local_span"] is None


@pytest.mark.parametrize("policy", ["A", "C", "C2"])
def test_untrusted_bridge_cannot_rescue_or_inherit_fact(policy):
    original, doc, nodes = example("β Treatment led to recovery.", "result", "d1",
                                   "Treatment", "recovery", "HAS_RESULT")
    bridge = non_factual("SIMILAR_TO", "b", "other", "bridge", resolution_decision_2f2="REJECT")
    mixed = path(original["edges"] + [bridge])
    mixed["document_ids"] = ["d1", "d2"]
    r = ranked("What was the treatment outcome?", [(mixed, doc, nodes)], policy)
    assert r["paths"] == [] and r["excluded"][0]["reason"] == "untrusted_navigation"
    bridge["properties"]["resolution_decision_2f2"] = "RELATED"
    [item] = ranked("What was the treatment outcome?", [(mixed, doc, nodes)], policy)["paths"]
    assert item["path"]["edges"][0]["head_key"] == original["edges"][0]["head_key"]
    assert item["path"]["edges"][0]["tail_key"] == original["edges"][0]["tail_key"]
    assert item["evidence"]["edges"][1]["source_relations"] == []
    if policy != "A":
        assert item["new_factual_documents"] == ["d1"]
