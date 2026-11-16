"""Stage 3A.1 wire contracts. Standard library only; no retrieval implementation.

TypedDicts provide static typing, not runtime validation. The explicit validators
below check seed/path invariants, not every possible nested provenance field.
RELEX_REL direction and predicate always describe the physical, original fact;
traversed_reverse describes navigation only. Properties retain original IDs,
JSON evidence strings and trust fields without flattening or rewriting them.

Future traversal policy: <=5 physical hops, <=1 SIMILAR_TO, <=1 concept bridge
(two HAS_SHARED_CONCEPT edges). Trusted SIMILAR_TO requires RELATED in
resolution_decision_2f2 and distinct endpoints. Exclude SUSPICIOUS_CELL_PAIRS,
requires_review=true and REVIEW/REJECT/SUSPICIOUS memberships. RELEX_MENTIONS
is document/evidence lookup only, never a path edge. RELEX_REL stays within its
source document; conceptual/semantic transitions never inherit factual claims.
Budgets may be lowered by callers; no separate configuration system is needed.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Literal, TypeAlias, TypedDict


TARGET_NAMESPACE = "relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v2_sem_name_t0830"
MAX_PHYSICAL_HOPS = 5
MAX_SEMANTIC_HOPS = 1
MAX_CONCEPT_BRIDGES = 1
SUSPICIOUS_CELL_PAIRS = frozenset({
    ("e:231c4582bed80098121ab6d8da48133468e9eb8717b057ffa9bfd461baa3b66d", "concept:2g1:7ae66a4459e9d0bec5010222"),
    ("e:b97ece67b0a4af2e49316cdea76c641bd0ac8d808822305d4e7abcc4bef49e24", "concept:2g1:7ae66a4459e9d0bec5010222"),
})
JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
EdgeType = Literal["RELEX_REL", "SIMILAR_TO", "HAS_SHARED_CONCEPT"]


class Seed(TypedDict):
    node_key: str
    entity_id: str
    document_id: str
    mention_id: str | None
    score: float
    match_method: str  # extensible: exact/vector/fallback, without changing schema


class GraphEdge(TypedDict):
    key: str
    type: EdgeType
    relation_type: str | None  # factual predicate only; None for nonfactual edges
    head_key: str
    tail_key: str
    traversed_reverse: bool
    document_id: str | None  # semantic/concept edges need not have one owner
    properties: dict[str, JSONValue]  # complete original Neo4j properties


class GraphPath(TypedDict):
    path_id: str
    seed_key: str
    node_keys: list[str]
    edges: list[GraphEdge]
    document_ids: list[str]  # sorted unique IDs, including seed/node documents


class RetrievalDiagnostics(TypedDict):
    nodes_examined: int
    relationships_examined: int
    limits_hit: list[str]


class RetrievalResult(TypedDict):
    namespace: str
    graph_fingerprint: str  # Neo4j fingerprint, NOT snapshot SHA-256
    paths: list[GraphPath]
    truncated: bool
    diagnostics: RetrievalDiagnostics


def _key(value: str, namespace: str) -> None:
    if not isinstance(value, str) or not value.startswith(namespace + ":") or value == namespace + ":":
        raise ValueError("Key outside requested namespace")


def validate_seed(seed: Seed, namespace: str) -> None:
    """Check identity mapping and finite linking score; cosine may be negative."""
    _key(seed["node_key"], namespace)
    if seed["node_key"] != namespace + ":" + seed["entity_id"] or not seed["document_id"]:
        raise ValueError("Seed identity/document mismatch")
    if isinstance(seed["score"], bool) or not math.isfinite(seed["score"]):
        raise ValueError("Seed score must be finite")


def graph_path_id(seed_key: str, node_keys: list[str], edges: list[GraphEdge]) -> str:
    """Stable identity includes ordered physical edge IDs and traversal direction."""
    payload = [seed_key, node_keys, [[e["key"], e["traversed_reverse"]] for e in edges]]
    return "path:" + hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def validate_graph_path(path: GraphPath, namespace: str) -> None:
    """Check structure/direction/namespace. This is NOT a trust-policy evaluator.

    Zero-edge seed paths are valid intermediate records, not factual answers.
    Candidate budgets and trust filtering belong to the future retriever.
    """
    nodes, edges = path["node_keys"], path["edges"]
    if not nodes or nodes[0] != path["seed_key"] or len(edges) != len(nodes) - 1:
        raise ValueError("Path length/seed mismatch")
    for key in nodes:
        _key(key, namespace)
    for index, edge in enumerate(edges):
        for field in ("key", "head_key", "tail_key"):
            _key(edge[field], namespace)
        if type(edge["traversed_reverse"]) is not bool:
            raise ValueError("Traversal direction must be boolean")
        endpoints = (edge["head_key"], edge["tail_key"])
        if edge["traversed_reverse"]:
            endpoints = endpoints[::-1]
        if endpoints != (nodes[index], nodes[index + 1]):
            raise ValueError("Disconnected edge or incorrect traversal direction")
        if edge["type"] not in ("RELEX_REL", "SIMILAR_TO", "HAS_SHARED_CONCEPT"):
            raise ValueError("Unsupported path edge type")
        if edge["type"] == "RELEX_REL":
            if not edge["relation_type"] or not edge["document_id"]:
                raise ValueError("Fact requires predicate and source document")
        elif edge["relation_type"] is not None:
            raise ValueError("Nonfactual edge cannot carry a factual predicate")
        props = edge["properties"]
        if props.get("key") != edge["key"] or props.get("namespace") != namespace:
            raise ValueError("Original property key/namespace mismatch")
        if edge["type"] == "RELEX_REL":
            if (props.get("relation_type") != edge["relation_type"] or
                    props.get("document_id") != edge["document_id"] or
                    edge["head_key"] != namespace + ":" + str(props.get("head_id")) or
                    edge["tail_key"] != namespace + ":" + str(props.get("tail_id"))):
                raise ValueError("Fact direction/predicate differs from original properties")
        if edge["document_id"] and edge["document_id"] not in path["document_ids"]:
            raise ValueError("Missing source document in path")
    if path["document_ids"] != sorted(set(path["document_ids"])):
        raise ValueError("Document IDs must be sorted and unique")
    if path["path_id"] != graph_path_id(path["seed_key"], nodes, edges):
        raise ValueError("Non-deterministic path ID")
