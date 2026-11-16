from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Iterable
import math

from .retrieval_types import (MAX_PHYSICAL_HOPS, SUSPICIOUS_CELL_PAIRS, TARGET_NAMESPACE,
                              GraphEdge, GraphPath, RetrievalResult, Seed, graph_path_id,
                              validate_graph_path, validate_seed)


@dataclass(frozen=True)
class TraversalLimits:
    max_hops: int = 5
    max_neighbors: int = 50
    max_candidates: int = 200
    max_relationships: int = 1000
    max_seeds: int = 32
    timeout_seconds: float = 10.0

    def __post_init__(self):
        for name in ("max_hops", "max_neighbors", "max_candidates", "max_relationships", "max_seeds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        if self.max_hops > MAX_PHYSICAL_HOPS:
            raise ValueError("max_hops exceeds the physical hop contract")


_NODE_QUERY = """
MATCH (n {namespace: $namespace, key: $key})
RETURN properties(n) AS node, labels(n) AS labels LIMIT 2
"""
_FINGERPRINT_QUERY = """
MATCH (n {namespace: $namespace})
RETURN n.fingerprint AS fingerprint LIMIT 1
"""
_ADJACENCY_QUERY = """
MATCH (n {namespace: $namespace, key: $key})-[r:RELEX_REL|SIMILAR_TO|HAS_SHARED_CONCEPT]-(other)
WHERE r.namespace = $namespace AND other.namespace = $namespace
RETURN properties(startNode(r)) AS head, labels(startNode(r)) AS head_labels,
       properties(endNode(r)) AS tail, labels(endNode(r)) AS tail_labels,
       type(r) AS kind, properties(r) AS properties
ORDER BY r.key, other.key
LIMIT $limit
"""


class GraphRetriever:
    """Read-only BFS; relationship budget counts returned adjacency rows, without lookahead.

    A full page or a frontier at max_hops is conservatively marked truncated. Seed reads (up to two rows per
    seed) and one fingerprint row are separate from the relationship budget.
    At most max_seeds input seeds are accepted, then deduplicated by node key.
    A server-side transaction timeout applies to the whole retrieval.
    Neo4j may inspect more relationships internally for ordering and filtering.
    """

    def __init__(self, driver, *, database=None, namespace=TARGET_NAMESPACE):
        if namespace != TARGET_NAMESPACE:
            raise ValueError("Unsupported namespace")
        self.driver, self.database, self.namespace = driver, database, namespace

    def retrieve(self, seeds: Iterable[Seed], limits: TraversalLimits | dict | None = None) -> RetrievalResult:
        if limits is None:
            limits = TraversalLimits()
        elif isinstance(limits, dict):
            limits = TraversalLimits(**limits)
        if not isinstance(limits, TraversalLimits):
            raise TypeError("limits must be TraversalLimits or a dict")
        seeds = list(islice(seeds, limits.max_seeds + 1))
        if len(seeds) > limits.max_seeds:
            raise ValueError("Too many seeds")
        unique = {}
        for seed in seeds:
            validate_seed(seed, self.namespace)
            if seed["node_key"] in unique and unique[seed["node_key"]]["document_id"] != seed["document_id"]:
                raise ValueError("Conflicting seed documents")
            unique.setdefault(seed["node_key"], seed)
        seeds = list(unique.values())
        options = {"database": self.database} if self.database is not None else {}
        from neo4j import unit_of_work

        @unit_of_work(timeout=limits.timeout_seconds)
        def read(tx):
            return self._retrieve(tx, seeds, limits)

        with self.driver.session(**options) as session:
            return session.execute_read(read)

    def _retrieve(self, tx, seeds: list[Seed], limits: TraversalLimits) -> RetrievalResult:
        ns = self.namespace
        records = list(tx.run(_FINGERPRINT_QUERY, namespace=ns))
        fingerprint = records[0]["fingerprint"] if records else None
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Namespace has no graph fingerprint")
        diagnostics = {"nodes_examined": 0, "relationships_examined": 0, "limits_hit": []}
        hits: set[str] = set()
        paths: list[GraphPath] = []
        queue = deque()
        for seed in seeds:
            records = list(tx.run(_NODE_QUERY, namespace=ns, key=seed["node_key"]))
            diagnostics["nodes_examined"] += len(records)
            if len(records) != 1:
                raise ValueError("Seed node missing or ambiguous")
            node, labels = records[0]["node"], records[0]["labels"]
            self._check_node(node, labels, fingerprint)
            if node["key"] != seed["node_key"] or "RELEXEntity" not in labels or node.get("document_id") != seed["document_id"]:
                raise ValueError("Seed node/document mismatch")
            queue.append((seed["node_key"], [seed["node_key"]], [], {seed["document_id"]}, 0, 0))

        while queue:
            seed_key, nodes, edges, docs, semantic, memberships = queue.popleft()
            if len(edges) >= limits.max_hops:
                hits.add("max_hops")
                continue
            remaining = limits.max_relationships - diagnostics["relationships_examined"]
            if remaining <= 0:
                hits.add("max_relationships")
                break
            cap = min(limits.max_neighbors, remaining)
            rows = list(tx.run(_ADJACENCY_QUERY, namespace=ns, key=nodes[-1], limit=cap))
            diagnostics["relationships_examined"] += len(rows)
            diagnostics["nodes_examined"] += len(rows)
            if len(rows) == cap:
                hits.add("max_neighbors" if cap == limits.max_neighbors else "max_relationships")
            for row in rows:
                head, tail, props = row["head"], row["tail"], row["properties"]
                self._check_node(head, row["head_labels"], fingerprint)
                self._check_node(tail, row["tail_labels"], fingerprint)
                if props.get("namespace") != ns or props.get("fingerprint") != fingerprint:
                    raise ValueError("Relationship namespace/fingerprint mismatch")
                hk, tk = head["key"], tail["key"]
                if nodes[-1] not in (hk, tk):
                    raise ValueError("Disconnected adjacency result")
                reverse = nodes[-1] == tk
                next_key = hk if reverse else tk
                if hk == tk or next_key in nodes or props.get("key") in {e["key"] for e in edges}:
                    continue
                kind = row["kind"]
                if kind not in ("RELEX_REL", "SIMILAR_TO", "HAS_SHARED_CONCEPT"):
                    raise ValueError("Unexpected relationship type")
                hlabels, tlabels = row["head_labels"], row["tail_labels"]
                if kind == "HAS_SHARED_CONCEPT":
                    if "RELEXEntity" not in hlabels or "RELEXConcept" not in tlabels:
                        continue
                    pair = (hk[len(ns) + 1:], tk[len(ns) + 1:])
                    if (pair in SUSPICIOUS_CELL_PAIRS or props.get("requires_review") is True or
                            str(props.get("review_status", "")).upper() in {"REVIEW", "REJECT", "SUSPICIOUS"}):
                        continue
                    if memberships >= 2 or (memberships == 0 and reverse) or (memberships == 1 and not reverse):
                        continue
                    if props.get("document_id") != head.get("document_id"):
                        continue
                else:
                    if "RELEXEntity" not in hlabels or "RELEXEntity" not in tlabels or memberships == 1:
                        continue
                    if kind == "SIMILAR_TO" and (semantic or props.get("resolution_decision_2f2") != "RELATED"):
                        continue
                    if kind == "RELEX_REL" and (not props.get("document_id") or
                            head.get("document_id") != props.get("document_id") or
                            tail.get("document_id") != props.get("document_id")):
                        continue
                edge: GraphEdge = {"key": props.get("key"), "type": kind,
                        "relation_type": props.get("relation_type") if kind == "RELEX_REL" else None,
                        "head_key": hk, "tail_key": tk, "traversed_reverse": reverse,
                        "document_id": props.get("document_id") if kind == "RELEX_REL" else None,
                        "properties": dict(props)}
                new_nodes, new_edges = nodes + [next_key], edges + [edge]
                new_docs = docs | {d for d in (head.get("document_id"), tail.get("document_id")) if d}
                count = memberships + (kind == "HAS_SHARED_CONCEPT")
                path: GraphPath = {"path_id": graph_path_id(seed_key, new_nodes, new_edges), "seed_key": seed_key,
                        "node_keys": new_nodes, "edges": new_edges, "document_ids": sorted(new_docs)}
                validate_graph_path(path, ns)
                if count != 1:
                    if len(paths) >= limits.max_candidates:
                        hits.add("max_candidates")
                        break
                    paths.append(path)
                queue.append((seed_key, new_nodes, new_edges, new_docs,
                              semantic + (kind == "SIMILAR_TO"), count))
            if "max_candidates" in hits:
                break
        diagnostics["limits_hit"] = sorted(hits)
        return {"namespace": ns, "graph_fingerprint": fingerprint, "paths": paths,
                "truncated": bool(hits), "diagnostics": diagnostics}

    def _check_node(self, node, labels, fingerprint):
        if (node.get("namespace") != self.namespace or node.get("fingerprint") != fingerprint or
                not isinstance(node.get("key"), str) or not node["key"].startswith(self.namespace + ":") or
                not ({"RELEXEntity", "RELEXConcept"} & set(labels))):
            raise ValueError("Node namespace, label or fingerprint mismatch")
        node_labels = set(labels) & {"RELEXEntity", "RELEXConcept"}
        if len(node_labels) != 1:
            raise ValueError("Ambiguous node label")
        identity = node.get("entity_id" if "RELEXEntity" in node_labels else "concept_id")
        if not isinstance(identity, str) or not identity or node["key"] != self.namespace + ":" + identity:
            raise ValueError("Node identity/key mismatch")
