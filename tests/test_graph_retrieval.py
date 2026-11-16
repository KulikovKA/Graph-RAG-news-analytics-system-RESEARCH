import pytest

from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
from cgraph.retrieval_types import SUSPICIOUS_CELL_PAIRS, TARGET_NAMESPACE as NS


def node(identity, doc="d1", concept=False):
    p = {"key": NS + ":" + identity, "namespace": NS, "fingerprint": "kg-fingerprint"}
    p["concept_id" if concept else "entity_id"] = identity
    if not concept:
        p["document_id"] = doc
    return p, ["RELEXConcept" if concept else "RELEXEntity"]


def edge(a, b, kind, name, **extra):
    h, hl = a
    t, tl = b
    p = {"key": NS + ":" + name, "namespace": NS, "fingerprint": "kg-fingerprint", **extra}
    if kind == "RELEX_REL":
        p.update(head_id=h["key"][len(NS)+1:], tail_id=t["key"][len(NS)+1:], relation_type="CAUSES", document_id="d1", evidence_json='{"text":"original"}')
    if kind == "HAS_SHARED_CONCEPT":
        p.setdefault("document_id", h["document_id"])
    return {"head": h, "tail": t, "head_labels": hl, "tail_labels": tl, "kind": kind, "properties": p}


class FakeDriver:
    def __init__(self, nodes, edges):
        self.nodes, self.edges, self.calls = nodes, edges, []

    def session(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute_read(self, callback):
        return callback(self)

    def run(self, query, **params):
        self.calls.append((query, params))
        if "AS fingerprint" in query:
            return [{"fingerprint": "kg-fingerprint"}]
        if "AS node" in query:
            return [{"node": p, "labels": labels} for p, labels in self.nodes if p["key"] == params["key"]]
        rows = [e for e in self.edges if params["key"] in (e["head"]["key"], e["tail"]["key"])]
        return sorted(rows, key=lambda e: e["properties"]["key"])[:params["limit"]]


def seed(n):
    return {"node_key": n[0]["key"], "entity_id": n[0]["key"][len(NS)+1:],
            "document_id": n[0]["document_id"], "mention_id": None, "score": 1.0, "match_method": "exact"}


def test_directions_parallel_facts_concept_semantic_and_cycles():
    a, b, c, x = node("a"), node("b"), node("c", "d2"), node("concept:x", concept=True)
    edges = [edge(a, b, "RELEX_REL", "r1"), edge(a, b, "RELEX_REL", "r2"),
             edge(b, c, "SIMILAR_TO", "s", resolution_decision_2f2="RELATED"),
             edge(a, x, "HAS_SHARED_CONCEPT", "m1"), edge(c, x, "HAS_SHARED_CONCEPT", "m2"),
             edge(a, a, "RELEX_REL", "loop")]
    result = GraphRetriever(FakeDriver([a, b, c, x], edges)).retrieve([seed(b)])
    assert result["graph_fingerprint"] == "kg-fingerprint"
    one = [p for p in result["paths"] if len(p["edges"]) == 1 and p["edges"][0]["type"] == "RELEX_REL"]
    assert len(one) == 2
    assert all(p["edges"][0]["traversed_reverse"] and p["edges"][0]["head_key"] == a[0]["key"] for p in one)
    assert one[0]["edges"][0]["properties"]["evidence_json"] == '{"text":"original"}'
    assert any([e["type"] for e in p["edges"]].count("HAS_SHARED_CONCEPT") == 2 for p in result["paths"])
    for p in result["paths"]:
        types = [e["type"] for e in p["edges"]]
        assert len(p["node_keys"]) == len(set(p["node_keys"]))
        assert types.count("SIMILAR_TO") <= 1 and types.count("HAS_SHARED_CONCEPT") in (0, 2)
        assert len(types) <= 5
        assert all(e["relation_type"] is None for e in p["edges"] if e["type"] != "RELEX_REL")


def test_trust_and_document_policy():
    eid, cid = next(iter(SUSPICIOUS_CELL_PAIRS))
    a, b, c, x = node(eid), node("b", "d2"), node("c"), node(cid, concept=True)
    edges = [edge(a, x, "HAS_SHARED_CONCEPT", "bad"), edge(c, x, "HAS_SHARED_CONCEPT", "ok"),
             edge(a, c, "SIMILAR_TO", "review", resolution_decision_2f2="REVIEW"),
             edge(a, c, "SIMILAR_TO", "reject", resolution_decision_2f2="REJECT"),
             edge(a, b, "RELEX_REL", "cross"), edge(a, x, "HAS_SHARED_CONCEPT", "flag", requires_review=True),
             edge(a, x, "HAS_SHARED_CONCEPT", "status", review_status="SUSPICIOUS")]
    assert GraphRetriever(FakeDriver([a, b, c, x], edges)).retrieve([seed(a)])["paths"] == []


def test_limits_and_validation():
    a, b, c = node("a"), node("b"), node("c")
    driver = FakeDriver([a, b, c], [edge(a, b, "RELEX_REL", "r1"), edge(a, c, "RELEX_REL", "r2")])
    result = GraphRetriever(driver).retrieve([seed(a)], {"max_neighbors": 1, "max_relationships": 1})
    assert result["truncated"] and result["diagnostics"]["relationships_examined"] == 1
    assert all(p["limit"] <= 1 for q, p in driver.calls if "AS kind" in q)
    with pytest.raises(ValueError):
        TraversalLimits(max_hops=6)
    with pytest.raises(ValueError):
        GraphRetriever(driver).retrieve([dict(seed(a), document_id="wrong")])
    b[0]["fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        GraphRetriever(driver).retrieve([seed(a)])


@pytest.mark.parametrize("pair", sorted(SUSPICIOUS_CELL_PAIRS))
def test_each_suspicious_cell_pair_without_flags(pair):
    a, x, b = node(pair[0]), node(pair[1], concept=True), node("b")
    driver = FakeDriver([a, x, b], [edge(a, x, "HAS_SHARED_CONCEPT", "m1"), edge(b, x, "HAS_SHARED_CONCEPT", "m2")])
    assert GraphRetriever(driver).retrieve([seed(a)])["paths"] == []


def test_seed_bound_deduplication_and_timeout():
    a = node("a")
    driver = FakeDriver([a], [])
    original = driver.execute_read
    def checked(callback):
        assert callback.timeout == 2.0
        return original(callback)
    driver.execute_read = checked
    GraphRetriever(driver).retrieve([seed(a), seed(a)], {"timeout_seconds": 2.0})
    assert sum("AS node" in q for q, _ in driver.calls) == 1
    with pytest.raises(ValueError, match="Too many seeds"):
        GraphRetriever(driver).retrieve([seed(a)] * 3, {"max_seeds": 2})


def test_candidate_cap_and_namespace_guard():
    a, b, c = node("a"), node("b"), node("c")
    driver = FakeDriver([a, b, c], [edge(a, b, "RELEX_REL", "r1"), edge(a, c, "RELEX_REL", "r2")])
    result = GraphRetriever(driver).retrieve([seed(a)], {"max_candidates": 1})
    assert len(result["paths"]) == 1 and "max_candidates" in result["diagnostics"]["limits_hit"]
    b[0]["namespace"] = "other"
    with pytest.raises(ValueError, match="namespace"):
        GraphRetriever(driver).retrieve([seed(a)])


def test_depth_limit_marks_unexpanded_frontier():
    a, b = node("a"), node("b")
    result = GraphRetriever(FakeDriver([a, b], [edge(a, b, "RELEX_REL", "r")])).retrieve([seed(a)], {"max_hops": 1})
    assert result["truncated"] and "max_hops" in result["diagnostics"]["limits_hit"]


def test_node_identity_must_match_key():
    a = node("a")
    a[0]["entity_id"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        GraphRetriever(FakeDriver([a], [])).retrieve([seed(a)])


def test_query_linker_seeds_feed_retrieval_and_preserve_original_provenance():
    import json
    import numpy as np

    from cgraph.entity_linking import EntityCache, EntityLinker
    from cgraph.query_analysis import QueryAnalyzer
    from cgraph.retrieval_types import validate_graph_path, validate_seed

    class NER:
        def predict(self, text, threshold):
            return [dict(start=0, end=3, text="RNA", score=0.8,
                         label="physical, biological or digital object")]

    class Encoder:
        def encode(self, texts):
            assert texts == ["RNA"]
            return np.array([[1, 0]], dtype=np.float32)

    cache = EntityCache(
        [dict(entity_id="b", document_id="d1", text_name="RNA"),
         dict(entity_id="c", document_id="d2", text_name="RNA")],
        np.array([[1, 0], [1, 0]], dtype=np.float32), "synthetic", {},
    )
    analysis = QueryAnalyzer(NER()).analyze("RNA?")
    linked = EntityLinker(cache, Encoder(), min_score=0.5).link(analysis, top_k=2)
    assert {s["document_id"] for s in linked["seeds"]} == {"d1", "d2"}
    for linked_seed in linked["seeds"]:
        validate_seed(linked_seed, NS)

    a, b, c = node("a"), node("b"), node("c", "d2")
    fact = edge(a, b, "RELEX_REL", "fact")
    fact["properties"].update(
        source_triple_ids=["original-triple"],
        source_relations_json=json.dumps([{
            "triple_id": "original-triple", "document_id": "d1",
            "evidence": [{"window_id": "w1", "document_id": "d1",
                          "start": 0, "end": 3, "text": "RNA"}],
        }]),
    )
    result = GraphRetriever(FakeDriver([a, b, c], [fact])).retrieve(linked["seeds"])
    assert len(result["paths"]) == 1
    path = result["paths"][0]
    validate_graph_path(path, NS)
    assert path["seed_key"] == NS + ":b"
    assert path["document_ids"] == ["d1"]
    assert path["edges"][0]["traversed_reverse"] is True
    assert path["edges"][0]["head_key"] == a[0]["key"]
    assert path["edges"][0]["properties"] == fact["properties"]
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize("kind", ["SIMILAR_TO", "HAS_SHARED_CONCEPT"])
def test_second_semantic_or_concept_transition_is_not_expanded(kind):
    a, b, c = node("a"), node("b"), node("c")
    nodes = [a, b, c]
    if kind == "SIMILAR_TO":
        edges = [edge(a, b, kind, "s1", resolution_decision_2f2="RELATED"),
                 edge(b, c, kind, "s2", resolution_decision_2f2="RELATED")]
    else:
        x, y = node("concept:x", concept=True), node("concept:y", concept=True)
        nodes += [x, y]
        edges = [edge(a, x, kind, "m1"), edge(b, x, kind, "m2"),
                 edge(b, y, kind, "m3"), edge(c, y, kind, "m4")]
    result = GraphRetriever(FakeDriver(nodes, edges)).retrieve([seed(a)])
    assert any(p["node_keys"][-1] == b[0]["key"] for p in result["paths"])
    assert all(c[0]["key"] not in p["node_keys"] for p in result["paths"])


def test_six_edge_chain_stops_at_five_physical_hops():
    nodes = [node(str(i)) for i in range(7)]
    edges = [edge(nodes[i], nodes[i + 1], "RELEX_REL", f"r{i}") for i in range(6)]
    result = GraphRetriever(FakeDriver(nodes, edges)).retrieve([seed(nodes[0])])
    assert max(len(p["edges"]) for p in result["paths"]) == 5
    assert all(nodes[-1][0]["key"] not in p["node_keys"] for p in result["paths"])
    assert "max_hops" in result["diagnostics"]["limits_hit"]


@pytest.mark.parametrize("flags", [
    {"requires_review": True}, {"review_status": "REVIEW"},
    {"review_status": "REJECT"}, {"review_status": "SUSPICIOUS"},
])
def test_unsafe_membership_flags_block_pairs_outside_cell_denylist(flags):
    a, b, x = node("a"), node("b"), node("concept:x", concept=True)
    edges = [edge(a, x, "HAS_SHARED_CONCEPT", "m1", **flags),
             edge(b, x, "HAS_SHARED_CONCEPT", "m2")]
    retriever = GraphRetriever(FakeDriver([a, b, x], edges))
    assert retriever.retrieve([seed(a)])["paths"] == []
    assert retriever.retrieve([seed(b)])["paths"] == []
