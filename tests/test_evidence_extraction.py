"""Synthetic fixtures follow the inspected 2G snapshot/importer schema."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from cgraph.evidence_extraction import EvidenceExtractor, extract_evidence
from cgraph.retrieval_types import TARGET_NAMESPACE as NS, graph_path_id


def document(text="β RNA uses ATP. RNA binds proteins.", identity="d1"):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return dict(document_id=identity, key=NS + ":" + identity, namespace=NS,
                fingerprint="fp", text=text, revision_id="sha256:" + digest, text_sha256=digest,
                source_url="https://example.org/" + identity, title="RNA study",
                source_reference_json='{"provider":"synthetic"}', doi="doi:synthetic")


def edge(doc, identity="r1", *, reverse=False):
    head, tail = (("a", "b") if doc["document_id"] == "d1"
                  else (doc["document_id"] + ":a", doc["document_id"] + ":b"))
    window = dict(window_id="w1", document_id=doc["document_id"], start=2, end=15,
                  text=doc["text"][2:15], entity_ids=["old-a", "old-b"])
    source = dict(triple_id="t:" + identity, document_id=doc["document_id"], head_id="old-a",
                  tail_id="old-b", relation_type="USES", confidence=0.7, evidence=[window])
    props = dict(key=NS + ":" + identity, namespace=NS, fingerprint="fp", head_id=head, tail_id=tail,
                 document_id=doc["document_id"], relation_type="USES", source_relation_count=1,
                 source_triple_ids=[source["triple_id"]], source_relations_json=json.dumps([source]),
                 source_json='{"original":"retained"}')
    return dict(key=props["key"], type="RELEX_REL", relation_type="USES", head_key=NS + ":" + head,
                tail_key=NS + ":" + tail, traversed_reverse=reverse, document_id=doc["document_id"], properties=props)


def path(edges):
    nodes = []
    for e in edges:
        head, tail = e["head_key"], e["tail_key"]
        if e["traversed_reverse"]:
            head, tail = tail, head
        if not nodes:
            nodes.append(head)
        assert nodes[-1] == head
        nodes.append(tail)
    return dict(path_id=graph_path_id(nodes[0], nodes, edges), seed_key=nodes[0], node_keys=nodes,
                edges=edges, document_ids=sorted({e["document_id"] for e in edges if e["document_id"]}))


def decode(e, doc):
    return extract_evidence([path([e])], {doc["document_id"]: doc}, graph_fingerprint="fp")[0]


def change_source(e, update):
    sources = json.loads(e["properties"]["source_relations_json"])
    update(sources)
    e["properties"]["source_relations_json"] = json.dumps(sources)


def test_exact_evidence_provenance_reverse_and_input_immutability():
    doc = document()
    e = edge(doc, reverse=True)
    p = path([e])
    original = deepcopy((p, doc))
    result = extract_evidence([p], {"d1": doc}, graph_fingerprint="fp")[0]
    assert result["status"] == "valid" and result["reasons"] == []
    item = result["edges"][0]
    assert item["edge"] == e and item["edge"]["traversed_reverse"] is True
    assert item["edge"]["relation_type"] == "USES"
    assert item["edge"]["head_key"] == NS + ":a"
    source = item["source_relations"][0]
    assert source["source"]["head_id"] == "old-a"
    w = source["windows"][0]
    assert w["text"] == doc["text"][w["start"]:w["end"]] == "RNA uses ATP."
    for field in ("revision_id", "text_sha256", "source_url", "title", "source_reference_json", "doi"):
        assert w[field] == doc[field]
    assert result["documents"] == [doc]
    assert (p, doc) == original
    json.dumps(result, allow_nan=False)
    w["source"]["entity_ids"].append("changed")
    assert (p, doc) == original


def test_multiple_sources_and_windows_keep_associations():
    doc = document()
    e = edge(doc)
    sources = json.loads(e["properties"]["source_relations_json"])
    sources[0]["evidence"].append(dict(window_id="w2", document_id="d1", start=16,
                                       end=len(doc["text"]), text=doc["text"][16:]))
    sources.append(deepcopy(sources[0]))
    sources[1]["triple_id"] = "t:other"
    e["properties"].update(source_relations_json=json.dumps(sources), source_triple_ids=["t:r1", "t:other"],
                           source_relation_count=2)
    result = decode(e, doc)
    assert result["status"] == "valid"
    assert [s["source_triple_id"] for s in result["edges"][0]["source_relations"]] == ["t:r1", "t:other"]
    assert result["diagnostics"]["valid_windows"] == 4


@pytest.mark.parametrize("updates,reason", [
    ({"start": -1}, "invalid_offsets"), ({"end": 1000}, "invalid_offsets"),
    ({"start": 15, "end": 2}, "invalid_offsets"), ({"start": True}, "invalid_offsets"),
    ({"start": 2.0}, "invalid_offsets"), ({"text": "different"}, "text_mismatch"),
    ({"document_id": None}, "missing_window_document_id"),
    ({"document_id": "missing"}, "window_document_mismatch"),
    ({"window_id": ""}, "missing_window_id"),
    ({"revision_id": "wrong"}, "window_revision_id_mismatch"),
])
def test_invalid_windows(updates, reason):
    doc = document()
    e = edge(doc)
    change_source(e, lambda sources: sources[0]["evidence"][0].update(updates))
    result = decode(e, doc)
    assert result["status"] == "invalid"
    assert any(reason in r for r in result["reasons"])
    assert result["diagnostics"]["validated_factual_edges"] == 0


@pytest.mark.parametrize("raw", ["{broken", "null", "{}", "[]", "[null]", "[1]", '[{"evidence": null}]'])
def test_corrupt_or_missing_sources(raw):
    doc = document()
    e = edge(doc)
    e["properties"]["source_relations_json"] = raw
    assert decode(e, doc)["status"] == "invalid"


@pytest.mark.parametrize("field,value,reason", [
    ("document_id", "other", "document_id_mismatch"),
    ("text_sha256", "bad", "document_hash_mismatch"),
    ("revision_id", None, "missing_revision_id"),
    ("revision_id", "sha256:bad", "document_revision_mismatch"),
    ("namespace", "other", "document_namespace_or_key_mismatch"),
    ("fingerprint", "other", "document_fingerprint_mismatch"),
    ("text", None, "missing_document_text"),
])
def test_document_integrity(field, value, reason):
    doc = document()
    e = edge(doc)
    doc[field] = value
    result = extract_evidence([path([e])], {"d1": doc}, graph_fingerprint="fp")[0]
    assert result["status"] == "invalid"
    assert any(reason in r for r in result["reasons"])


def test_missing_document_and_wrong_existing_document_are_not_substituted():
    doc = document()
    e = edge(doc)
    p = path([e])
    result = extract_evidence([p], {})[0]
    assert result["status"] == "invalid"
    assert any("missing_document" in r for r in result["reasons"])
    other = document(identity="other")
    change_source(e, lambda s: s[0]["evidence"][0].update(document_id="other"))
    result = extract_evidence([path([e])], {"d1": doc, "other": other})[0]
    assert result["status"] == "invalid"
    assert result["documents"] == [doc]


@pytest.mark.parametrize("mutation,reason", [
    (lambda s: s[0].update(document_id="other"), "source_document_mismatch"),
    (lambda s: s[0].update(relation_type="CAUSES"), "source_predicate_mismatch"),
    (lambda s: s[0].update(triple_id="other"), "source_triple_ids_mismatch"),
    (lambda s: s[0].update(evidence=[]), "missing_evidence"),
    (lambda s: s[0].update(head_id=None), "missing_source_endpoints"),
])
def test_source_integrity(mutation, reason):
    doc = document()
    e = edge(doc)
    change_source(e, mutation)
    result = decode(e, doc)
    assert result["status"] == "invalid"
    assert any(reason in r for r in result["reasons"])


def test_partial_windows_remain_visible_but_edge_not_confirmed():
    doc = document()
    e = edge(doc)
    change_source(e, lambda s: s[0]["evidence"].append({"document_id": "d1"}))
    result = decode(e, doc)
    assert result["status"] == "partial"
    assert result["diagnostics"]["valid_windows"] == 1
    assert result["diagnostics"]["validated_factual_edges"] == 0


def test_parallel_facts_are_separate():
    doc = document()
    paths = [path([edge(doc, identity)]) for identity in ("r1", "r2")]
    result = extract_evidence(paths, {"d1": doc})
    assert len(result) == 2
    assert result[0]["path_id"] != result[1]["path_id"]
    assert [r["edges"][0]["source_relations"][0]["source_triple_id"] for r in result] == ["t:r1", "t:r2"]


def non_factual(kind, head, tail, identity, reverse=False, **metadata):
    return dict(key=NS + ":" + identity, type=kind, head_key=NS + ":" + head,
                tail_key=NS + ":" + tail, relation_type=None, document_id=None,
                traversed_reverse=reverse,
                properties=dict(key=NS + ":" + identity, namespace=NS, fingerprint="fp", **metadata))


def test_semantic_and_concept_transitions_never_create_factual_evidence():
    semantic = non_factual("SIMILAR_TO", "a", "b", "s", resolution_decision_2f2="RELATED", similarity=0.99)
    memberships = [non_factual("HAS_SHARED_CONCEPT", "a", "concept:c", "m1", document_ids=["d1"],
                              factual=False, fact_inheritance_allowed=False, review_status="ACCEPT"),
                   non_factual("HAS_SHARED_CONCEPT", "b", "concept:c", "m2", reverse=True, document_ids=["d2"])]
    for p in (path([semantic]), path(memberships)):
        result = extract_evidence([p], {})[0]
        assert result["status"] == "non_factual" and result["navigation_valid"]
        assert result["documents"] == []
        for e, original in zip(result["edges"], p["edges"]):
            assert e["non_factual"] and e["source_relations"] == [] and e["edge"] == original


def test_untrusted_transition_remains_flagged():
    s = non_factual("SIMILAR_TO", "a", "b", "s", resolution_decision_2f2="REVIEW")
    result = extract_evidence([path([s])], {})[0]
    assert not result["navigation_valid"]
    assert "untrusted_semantic_transition" in result["reasons"]
    m = non_factual("HAS_SHARED_CONCEPT", "a", "concept:c", "m", requires_review=True)
    result = extract_evidence([path([m])], {})[0]
    assert not result["navigation_valid"]
    assert "untrusted_concept_transition" in result["reasons"]


class DocumentDriver:
    def __init__(self, rows):
        self.rows, self.calls, self.options = rows, [], None

    def session(self, **options):
        self.options = options
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute_read(self, callback):
        return callback(self)

    def run(self, query, **params):
        self.calls.append((query, params))
        return [{"document": d} for d in self.rows]


def test_batch_read_once_and_duplicate_document_refused():
    doc, other = document(), document(identity="d2")
    paths = [path([edge(doc, "r1")]), path([edge(other, "r2")])]
    retrieval = dict(namespace=NS, graph_fingerprint="fp", paths=paths, truncated=False, diagnostics={})
    driver = DocumentDriver([doc, other])
    result = EvidenceExtractor(driver, database="neo4j").extract(retrieval)
    assert all(r["status"] == "valid" for r in result)
    assert len(driver.calls) == 1 and driver.calls[0][1]["document_ids"] == ["d1", "d2"]
    assert "MATCH (d:RELEXDocument" in driver.calls[0][0]
    assert driver.options == {"database": "neo4j"}
    with pytest.raises(ValueError, match="Ambiguous"):
        EvidenceExtractor(DocumentDriver([doc, doc])).extract(retrieval)


def test_empty_batch_needs_no_driver_or_neo4j_read():
    result = dict(namespace=NS, graph_fingerprint="fp", paths=[], truncated=False, diagnostics={})
    assert EvidenceExtractor(None).extract(result) == []


def test_existing_local_snapshot_format_when_available():
    """Проверка реального артефакта отдельно от синтетических fixtures и метрик."""
    snapshot_path = Path(__file__).resolve().parents[1] / ".cache/semantic_concept_sweep/baseline.json"
    if not snapshot_path.is_file():
        pytest.skip("Local frozen snapshot not available")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    docs = {n["properties"]["document_id"]: n["properties"] for n in snapshot["nodes"]
            if "RELEXDocument" in n["labels"]}
    raw = next(r for r in snapshot["relationships"] if r["type"] == "RELEX_REL"
               and len(r["properties"]["source_triple_ids"]) > 1 and r["head"] != r["tail"])
    props = raw["properties"]
    e = dict(key=raw["key"], type="RELEX_REL", relation_type=props["relation_type"],
             head_key=raw["head"], tail_key=raw["tail"], traversed_reverse=False,
             document_id=props["document_id"], properties=props)
    result = extract_evidence([path([e])], docs, namespace=props["namespace"],
                              graph_fingerprint=props["fingerprint"])[0]
    assert result["status"] == "valid" and result["navigation_valid"]
    assert result["edges"][0]["edge"]["properties"] == props
    sources = result["edges"][0]["source_relations"]
    assert [s["source_triple_id"] for s in sources] == props["source_triple_ids"]
    assert result["diagnostics"]["valid_windows"] == sum(len(s["evidence"]) for s in json.loads(props["source_relations_json"]))
