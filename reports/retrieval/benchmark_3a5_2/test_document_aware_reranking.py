"""Synthetic cases for the offline selection policies."""

import json
import sys

import pytest

from run_document_aware import HERE, check_fact, path_output
from cgraph.evidence_extraction import extract_evidence
from document_aware_reranking import score_candidates, select_paths

sys.path.insert(0, str(HERE.parents[2] / "tests"))
from test_evidence_extraction import NS, change_source, document, edge, non_factual, path
from test_path_reranking import retrieval


QUESTION = "Optical pulse propagation?"


def fact(doc, name, source_id=None):
    e = edge(doc, name)
    text = doc["text"]
    change_source(e, lambda sources: sources[0]["evidence"][0].update(start=2, end=len(text), text=text[2:]))
    if source_id is not None:
        change_source(e, lambda sources: sources[0].update(triple_id=source_id))
        e["properties"]["source_triple_ids"] = [source_id]
    return e


def candidates(paths, docs):
    extraction = extract_evidence(paths, {d["document_id"]: d for d in docs}, graph_fingerprint="fp")
    return score_candidates(QUESTION, retrieval(paths), extraction, ())


def selected_paths(paths, docs, mode, top_k=5):
    selected, _ = select_paths(candidates(paths, docs), mode, top_k=top_k)
    return [path_output(p) for p in selected]


@pytest.mark.parametrize("mode", ["B", "C"])
def test_two_relevant_documents_survive_redundant_routes(mode):
    first = document("β Optical pulse propagation.", "d1")
    second = document("β Optical pulse.", "d2")
    first_paths = [path([fact(first, f"first-{i}")]) for i in range(7)]
    repeated = path([fact(first, "first-0"), non_factual("SIMILAR_TO", "b", "c", "repeat",
                                                        resolution_decision_2f2="RELATED")])
    second_path = path([fact(second, "second")])
    result = selected_paths([*first_paths, repeated, second_path], [first, second], mode, top_k=2)
    assert [p["factual_document_ids"] for p in result] == [["d1"], ["d2"]]
    assert all(p["score"] == pytest.approx(sum(p["score_breakdown"].values())) for p in result)


@pytest.mark.parametrize("mode", ["B", "C"])
def test_document_novelty_cannot_displace_strong_paths_with_unrelated_text(mode):
    strong = document("β Optical pulse propagation.", "d1")
    weak = document("β Bread and butter.", "d2")
    paths = [path([fact(strong, f"strong-{i}")]) for i in range(3)] + [path([fact(weak, "weak")])]
    result = selected_paths(paths, [strong, weak], mode, top_k=3)
    assert len(result) == 3
    assert all(p["factual_document_ids"] == ["d1"] for p in result)


def test_same_source_under_different_graph_edges_adds_no_document_bonus_or_second_path():
    doc = document("β Optical pulse propagation.")
    paths = [path([fact(doc, name, source_id="t:common")]) for name in ["route-a", "route-b"]]
    results, profiles = select_paths(candidates(paths, [doc]), "C")
    assert len(results) == 1
    assert results[0]["new_source_triples"] == 1
    assert profiles["d1"]["second_independent_path_id"] is None
    assert profiles["d1"]["second_independent_path_bonus"] == 0
    paths.append(path([fact(doc, "independent")]))
    results, profiles = select_paths(candidates(paths, [doc]), "C")
    assert len(results) == 2
    assert profiles["d1"]["second_independent_path_bonus"] > 0


@pytest.mark.parametrize("mode", ["B", "C"])
def test_navigation_only_document_does_not_get_factual_credit(mode):
    doc = document("β Optical pulse propagation.")
    fact_edge = fact(doc, "optics")
    bridge = non_factual("SIMILAR_TO", "b", "other", "bridge", resolution_decision_2f2="RELATED",
                         evidence_json='{"text":"Optical pulse propagation."}')
    mixed = path([fact_edge, bridge])
    mixed["document_ids"] = ["d1", "d2"]
    no_fact = path([bridge])
    no_fact["document_ids"] = ["d1", "d2"]
    result = selected_paths([mixed, no_fact], [doc], mode)
    assert len(result) == 1
    assert result[0]["document_ids"] == ["d1", "d2"]
    assert result[0]["factual_document_ids"] == ["d1"]
    assert result[0]["new_factual_documents"] == ["d1"]
    bridge["properties"]["resolution_decision_2f2"] = "REJECT"
    assert selected_paths([mixed], [doc], mode) == []


@pytest.mark.parametrize("mode", ["B", "C"])
def test_preserves_provenance_and_rejects_modified_quote(mode):
    doc = document("β Optical pulse propagation.")
    e = fact(doc, "optics")
    [result] = selected_paths([path([e])], [doc], mode)
    [original] = result["facts"]
    [window] = original["windows"]
    assert original["document_id"] == window["document_id"] == "d1"
    assert original["source_triple_id"] == "t:optics"
    assert (window["start"], window["end"]) == (2, len(doc["text"]))
    assert window["text"] == doc["text"][2:]
    bundle = extract_evidence([path([e])], {"d1": doc}, namespace=NS, graph_fingerprint="fp")[0]
    item = bundle["edges"][0]
    archived_fact = {"document_id": "d1", "relation_type": "USES", **{
        k: item[k] for k in ("source_relations", "status", "reasons")}}
    check_fact(archived_fact, {"d1": doc})
    archived_fact["source_relations"][0]["windows"][0]["start"] += 1
    with pytest.raises(ValueError, match="differs from corpus"):
        check_fact(archived_fact, {"d1": doc})
    change_source(e, lambda sources: sources[0]["evidence"][0].update(text="Wrong quote"))
    assert selected_paths([path([e])], [doc], mode) == []


@pytest.mark.parametrize("mode", ["B", "C"])
def test_selection_and_breakdown_are_deterministic_on_permuted_inputs(mode):
    docs = [document("β Optical pulse propagation.", d) for d in ("d1", "d2", "d3")]
    paths = [path([fact(doc, f"{doc['document_id']}-{i}")]) for doc in docs for i in range(3)]
    first, profiles = select_paths(candidates(paths, docs), mode)
    second, shuffled = select_paths(list(reversed(candidates(paths[::-1], docs[::-1]))), mode)
    assert [path_output(p) for p in first] == [path_output(p) for p in second]
    assert profiles == shuffled
    json.dumps([path_output(p) for p in first], allow_nan=False)
