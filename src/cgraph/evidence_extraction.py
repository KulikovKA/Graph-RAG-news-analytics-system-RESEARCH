"""Stage 3A.4: source-exact evidence decoding; no inference or graph writes.

``extract_evidence`` accepts RELEXDocument properties keyed by document_id.
``EvidenceExtractor.extract`` reads those properties in one Neo4j transaction.
Validation confirms provenance and exact quotations, not scientific truth.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from typing import Iterable, Mapping

from .retrieval_types import (GraphPath, RetrievalResult, TARGET_NAMESPACE, MAX_PHYSICAL_HOPS,
                              SUSPICIOUS_CELL_PAIRS, validate_graph_path)


_DOCUMENT_QUERY = """
MATCH (d:RELEXDocument {namespace: $namespace})
WHERE d.document_id IN $document_ids
RETURN properties(d) AS document
ORDER BY d.document_id, d.key
"""


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _document_reasons(document, document_id, namespace, fingerprint):
    if not isinstance(document, dict):
        return ["missing_document"]
    reasons = []
    if document.get("document_id") != document_id:
        reasons.append("document_id_mismatch")
    if document.get("namespace") != namespace or document.get("key") != namespace + ":" + document_id:
        reasons.append("document_namespace_or_key_mismatch")
    if fingerprint is not None and document.get("fingerprint") != fingerprint:
        reasons.append("document_fingerprint_mismatch")
    text = document.get("text")
    if not isinstance(text, str):
        reasons.append("missing_document_text")
    else:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if document.get("text_sha256") != digest:
            reasons.append("document_hash_mismatch")
        revision = document.get("revision_id")
        if not _nonempty(revision):
            reasons.append("missing_revision_id")
        elif revision.startswith("sha256:") and revision != "sha256:" + digest:
            reasons.append("document_revision_mismatch")
    return reasons


def _window(raw, source, edge, documents, document_errors, source_errors):
    reasons = list(source_errors)
    item = {"source": deepcopy(raw), "status": "invalid", "reasons": reasons}
    if not isinstance(raw, dict):
        reasons.append("invalid_window")
        return item
    for field in ("document_id", "window_id", "start", "end", "text"):
        item[field] = raw.get(field)
    doc_id = raw.get("document_id")
    if not _nonempty(doc_id):
        reasons.append("missing_window_document_id")
        return item
    if doc_id != source.get("document_id") or doc_id != edge["document_id"]:
        reasons.append("window_document_mismatch")
        return item
    document = documents.get(doc_id)
    reasons.extend(document_errors.get(doc_id, ["missing_document"]))
    if not _nonempty(raw.get("window_id")):
        reasons.append("missing_window_id")
    if isinstance(document, dict):
        for field in ("revision_id", "text_sha256", "source_url", "title", "doi", "source_reference_json"):
            item[field] = deepcopy(document.get(field))
        for field in ("revision_id", "text_sha256"):
            if field in raw and raw[field] != document.get(field):
                reasons.append("window_" + field + "_mismatch")
        start, end, text = raw.get("start"), raw.get("end"), document.get("text")
        if (type(start) is not int or type(end) is not int or not isinstance(text, str)
                or not 0 <= start < end <= len(text)):
            reasons.append("invalid_offsets")
        elif not isinstance(raw.get("text"), str) or text[start:end] != raw["text"]:
            reasons.append("text_mismatch")
    item["status"] = "valid" if not reasons else "invalid"
    return item


def _navigation_reasons(path, namespace):
    """Защита от ослабления политики 3A.3 при передаче путей вручную."""
    reasons = []
    edges = path["edges"]
    if (len(edges) > MAX_PHYSICAL_HOPS or len(set(path["node_keys"])) != len(path["node_keys"])
            or len({e["key"] for e in edges}) != len(edges)):
        reasons.append("invalid_navigation_structure")
    semantic = [e for e in edges if e["type"] == "SIMILAR_TO"]
    if len(semantic) > 1 or any(e["properties"].get("resolution_decision_2f2") != "RELATED" for e in semantic):
        reasons.append("untrusted_semantic_transition")
    memberships = [(i, e) for i, e in enumerate(edges) if e["type"] == "HAS_SHARED_CONCEPT"]
    if memberships and (len(memberships) != 2 or memberships[1][0] != memberships[0][0] + 1
                        or memberships[0][1]["traversed_reverse"]
                        or not memberships[1][1]["traversed_reverse"]):
        reasons.append("invalid_concept_bridge")
    for _, edge in memberships:
        props = edge["properties"]
        pair = (edge["head_key"][len(namespace) + 1:], edge["tail_key"][len(namespace) + 1:])
        if (pair in SUSPICIOUS_CELL_PAIRS or props.get("requires_review") is True
                or str(props.get("review_status", "")).upper() in {"REVIEW", "REJECT", "SUSPICIOUS"}):
            reasons.append("untrusted_concept_transition")
    return sorted(set(reasons))


def _fact(edge, documents, document_errors):
    result = {"edge": deepcopy(edge), "non_factual": False, "status": "invalid",
              "source_relations": [], "reasons": []}
    reasons = result["reasons"]
    props = edge["properties"]
    try:
        sources = json.loads(props.get("source_relations_json", ""))
    except (ValueError, TypeError):
        reasons.append("invalid_source_relations_json")
        return result
    if not isinstance(sources, list) or not sources:
        reasons.append("missing_source_relations")
        return result
    ids = props.get("source_triple_ids")
    ids_valid = isinstance(ids, list) and bool(ids) and all(_nonempty(i) for i in ids)
    source_ids = [s.get("triple_id") for s in sources if isinstance(s, dict)]
    mapping_valid = (ids_valid and all(_nonempty(i) for i in source_ids)
                     and len(source_ids) == len(sources)
                     and Counter(ids) == Counter(source_ids)
                     and len(set(ids)) == len(ids))
    if not mapping_valid:
        reasons.append("source_triple_ids_mismatch")
    if "source_relation_count" in props and props["source_relation_count"] != len(sources):
        reasons.append("source_relation_count_mismatch")
    for source in sources:
        source_errors = list(reasons)
        record = {"source": deepcopy(source), "source_triple_id": None,
                  "status": "invalid", "windows": [], "reasons": source_errors}
        result["source_relations"].append(record)
        if not isinstance(source, dict):
            source_errors.append("invalid_source_relation")
            continue
        record["source_triple_id"] = source.get("triple_id")
        if source.get("document_id") != edge["document_id"]:
            source_errors.append("source_document_mismatch")
        if source.get("relation_type") != edge["relation_type"]:
            source_errors.append("source_predicate_mismatch")
        # Исходные head/tail сохраняются, но не сравниваются с переназначенными 2F.3 IDs.
        if not _nonempty(source.get("head_id")) or not _nonempty(source.get("tail_id")):
            source_errors.append("missing_source_endpoints")
        windows = source.get("evidence")
        if not isinstance(windows, list) or not windows:
            source_errors.append("missing_evidence")
            continue
        record["windows"] = [_window(w, source, edge, documents, document_errors, source_errors) for w in windows]
        record["status"] = _status([w["status"] for w in record["windows"]])
    result["status"] = _status([s["status"] for s in result["source_relations"]])
    return result


def _status(statuses):
    if statuses and all(s == "valid" for s in statuses):
        return "valid"
    return "partial" if any(s in {"valid", "partial"} for s in statuses) else "invalid"


def extract_evidence(paths: Iterable[GraphPath], documents: Mapping[str, dict], *,
                     namespace: str = TARGET_NAMESPACE, graph_fingerprint: str | None = None) -> list[dict]:
    """Decode every source separately, retaining raw properties and invalid evidence.

    Status is valid only when every source/window validates. Partial records are
    inspectable but cannot support an entire ranked path. Document hashes are
    checked once per batch. Documents must include original namespace/key fields.
    """
    paths = list(paths)
    for path in paths:
        validate_graph_path(path, namespace)
    required = sorted({e["document_id"] for p in paths for e in p["edges"] if e["type"] == "RELEX_REL"})
    errors = {d: _document_reasons(documents.get(d), d, namespace, graph_fingerprint) for d in required}
    bundles = []
    for path in paths:
        edges, reasons = [], _navigation_reasons(path, namespace)
        navigation_valid = not reasons
        for index, edge in enumerate(path["edges"]):
            if edge["type"] == "RELEX_REL":
                item = _fact(edge, documents, errors)
            else:
                item = {"edge": deepcopy(edge), "non_factual": True, "status": "non_factual",
                        "source_relations": [], "reasons": []}
            if graph_fingerprint is not None and edge["properties"].get("fingerprint") != graph_fingerprint:
                item["status"] = "invalid"
                item["reasons"].append("edge_fingerprint_mismatch")
            edges.append(item)
            reasons.extend(f"edge[{index}]:{r}" for r in item["reasons"])
            for si, source in enumerate(item["source_relations"]):
                reasons.extend(f"edge[{index}].source[{si}]:{r}" for r in source["reasons"])
                for wi, window in enumerate(source["windows"]):
                    reasons.extend(f"edge[{index}].source[{si}].window[{wi}]:{r}" for r in window["reasons"])
        facts = [e for e in edges if not e["non_factual"]]
        windows = [w for e in facts for s in e["source_relations"] for w in s["windows"]]
        docs = sorted({e["edge"]["document_id"] for e in facts})
        bundles.append({"path_id": path["path_id"], "edges": edges,
                        "documents": [deepcopy(documents[d]) for d in docs if isinstance(documents.get(d), dict)],
                        "status": _status([e["status"] for e in facts]) if facts else "non_factual",
                        "navigation_valid": navigation_valid,
                        "reasons": sorted(set(reasons)),
                        "diagnostics": {"factual_edges": len(facts),
                                        "validated_factual_edges": sum(e["status"] == "valid" for e in facts),
                                        "non_factual_edges": len(edges) - len(facts),
                                        "windows": len(windows),
                                        "valid_windows": sum(w["status"] == "valid" for w in windows)}})
    return bundles


class EvidenceExtractor:
    """Optional Neo4j adapter: one batched MATCH through execute_read, with timeout."""

    def __init__(self, driver, *, database=None, namespace=TARGET_NAMESPACE):
        if namespace != TARGET_NAMESPACE:
            raise ValueError("Unsupported namespace")
        self.driver, self.database, self.namespace = driver, database, namespace

    def extract(self, retrieval: RetrievalResult) -> list[dict]:
        if retrieval["namespace"] != self.namespace:
            raise ValueError("Retrieval namespace mismatch")
        paths = retrieval["paths"]
        for path in paths:
            validate_graph_path(path, self.namespace)
        document_ids = sorted({e["document_id"] for p in paths for e in p["edges"] if e["type"] == "RELEX_REL"})
        documents = {}
        if document_ids:
            from neo4j import unit_of_work

            @unit_of_work(timeout=10.0)
            def read(tx):
                return [dict(r["document"]) for r in tx.run(
                    _DOCUMENT_QUERY, namespace=self.namespace, document_ids=document_ids)]

            options = {"database": self.database} if self.database is not None else {}
            with self.driver.session(**options) as session:
                for document in session.execute_read(read):
                    doc_id = document["document_id"]
                    if doc_id in documents:
                        raise ValueError("Ambiguous RELEXDocument identity")
                    documents[doc_id] = document
        return extract_evidence(paths, documents, namespace=self.namespace,
                                graph_fingerprint=retrieval["graph_fingerprint"])
