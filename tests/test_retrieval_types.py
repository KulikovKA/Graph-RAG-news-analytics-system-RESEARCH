"""Stage 3A.1 synthetic contracts; no models or database."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cgraph.retrieval_types import (
    GraphEdge, GraphPath, RetrievalResult, Seed, TARGET_NAMESPACE,
    MAX_PHYSICAL_HOPS, MAX_SEMANTIC_HOPS, MAX_CONCEPT_BRIDGES,
    SUSPICIOUS_CELL_PAIRS, graph_path_id, validate_graph_path, validate_seed,
)

NS = TARGET_NAMESPACE


def records(reverse=False):
    seed: Seed = dict(node_key=NS + ":b" if reverse else NS + ":a",
                      entity_id="b" if reverse else "a", document_id="doc1",
                      mention_id=None, score=.8, match_method="vector")
    edge: GraphEdge = dict(key=NS + ":rel1", type="RELEX_REL", relation_type="USES",
                           head_key=NS + ":a", tail_key=NS + ":b", traversed_reverse=reverse,
                           document_id="doc1", properties=dict(key=NS + ":rel1", namespace=NS,
                           head_id="a", tail_id="b", relation_type="USES", document_id="doc1",
                           original_head_id="old_a", original_tail_id="old_b",
                           source_triple_ids=["triple1"], source_relations_json='[{"triple_id":"triple1"}]'))
    nodes = [NS + ":b", NS + ":a"] if reverse else [NS + ":a", NS + ":b"]
    path: GraphPath = dict(path_id=graph_path_id(seed["node_key"], nodes, [edge]),
                           seed_key=seed["node_key"], node_keys=nodes, edges=[edge], document_ids=["doc1"])
    result: RetrievalResult = dict(namespace=NS, graph_fingerprint="fingerprint-not-snapshot-sha",
                                   paths=[path], truncated=False,
                                   diagnostics=dict(nodes_examined=2, relationships_examined=1, limits_hit=[]))
    return seed, edge, path, result


@pytest.mark.parametrize("reverse", [False, True])
def test_records_preserve_physical_fact_and_provenance(reverse):
    seed, edge, path, result = records(reverse)
    before = deepcopy(result)
    validate_seed(seed, NS)
    validate_graph_path(path, NS)
    assert result == before  # validation cannot rewrite a reversed fact
    assert edge["head_key"] == NS + ":a" and edge["tail_key"] == NS + ":b"
    assert edge["relation_type"] == "USES"
    assert edge["properties"]["original_head_id"] == "old_a"
    assert edge["properties"]["source_triple_ids"] == ["triple1"]
    assert path["document_ids"] == [seed["document_id"]]
    assert json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False)) == result
    assert json.loads(json.dumps(seed)) == seed


def test_path_identity_is_stable_and_sensitive_to_edge_identity_and_order():
    _, edge, path, _ = records()
    assert graph_path_id(path["seed_key"], path["node_keys"], deepcopy(path["edges"])) == path["path_id"]
    edge["key"] = NS + ":parallel_relation"
    assert graph_path_id(path["seed_key"], path["node_keys"], [edge]) != path["path_id"]
    assert records(True)[2]["path_id"] != records(False)[2]["path_id"]


@pytest.mark.parametrize("change", [
    lambda p: p["node_keys"].append(NS + ":extra"),
    lambda p: p["edges"][0].update(tail_key=NS + ":disconnected"),
    lambda p: p["edges"][0].update(traversed_reverse=True),
    lambda p: p["edges"][0].update(head_key="other:a"),
    lambda p: p["edges"][0].update(key="other:rel1"),
    lambda p: p["edges"][0]["properties"].update(namespace="other"),
    lambda p: p["edges"][0].update(type="RELEX_MENTIONS"),
    lambda p: p["edges"][0].update(relation_type=None),
    lambda p: p["edges"][0].update(relation_type="USED_FOR"),
    lambda p: p["edges"][0]["properties"].update(head_id="different"),
    lambda p: p.update(seed_key=NS + ":wrong"),
    lambda p: p.update(document_ids=[]),
    lambda p: p.update(document_ids=["doc1", "doc1"]),
    lambda p: p.update(path_id="unstable"),
])
def test_invalid_path_invariants(change):
    path = records()[2]
    change(path)
    with pytest.raises(ValueError):
        validate_graph_path(path, NS)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_invalid_seed_score(score):
    seed = records()[0]
    seed["score"] = score
    with pytest.raises(ValueError):
        validate_seed(seed, NS)


def test_seed_identity_and_zero_edge_intermediate():
    seed = records()[0]
    seed["entity_id"] = "different"
    with pytest.raises(ValueError):
        validate_seed(seed, NS)
    nodes = [NS + ":a"]
    path = dict(seed_key=nodes[0], node_keys=nodes, edges=[], document_ids=["doc1"],
                path_id=graph_path_id(nodes[0], nodes, []))
    validate_graph_path(path, NS)


def test_nonfactual_edge_preserves_trust_properties_without_claiming_validation():
    _, edge, path, _ = records()
    edge.update(type="SIMILAR_TO", relation_type=None, document_id=None)
    edge["properties"].update(resolution_decision_2f2="REVIEW", similarity=.96)
    validate_graph_path(path, NS)  # structurally valid; future trust filter must reject
    assert edge["properties"]["resolution_decision_2f2"] == "REVIEW"
    assert (MAX_PHYSICAL_HOPS, MAX_SEMANTIC_HOPS, MAX_CONCEPT_BRIDGES) == (5, 1, 1)
    assert len(SUSPICIOUS_CELL_PAIRS) == 2


def test_multihop_concept_bridge_and_stable_document_order():
    _, edge, _, _ = records()
    concept = NS + ":concept:c"
    first = deepcopy(edge)
    first.update(type="HAS_SHARED_CONCEPT", relation_type=None, tail_key=concept, document_id=None)
    second = deepcopy(first)
    second.update(key=NS + ":membership2", head_key=NS + ":b", traversed_reverse=True)
    second["properties"]["key"] = second["key"]
    nodes = [NS + ":a", concept, NS + ":b"]
    path = dict(seed_key=nodes[0], node_keys=nodes, edges=[first, second], document_ids=["doc1", "doc2"],
                path_id=graph_path_id(nodes[0], nodes, [first, second]))
    validate_graph_path(path, NS)
    path["document_ids"].reverse()
    with pytest.raises(ValueError, match="sorted and unique"):
        validate_graph_path(path, NS)


def test_contract_import_does_not_require_driver_or_model_dependencies():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'neo4j', 'torch', 'numpy', 'gliner', 'gliner2', 'transformers'}:
            raise AssertionError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
import cgraph.retrieval_types
"""
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    subprocess.run([sys.executable, "-B", "-S", "-c", code], env=env, check=True, capture_output=True, text=True)
