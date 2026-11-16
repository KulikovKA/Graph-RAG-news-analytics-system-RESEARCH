"""Read-only Stage 3A.1 verification; no models, imports into Neo4j or files.

Run: .venv/Scripts/python.exe -B scripts/smoke_retrieval_3a1.py
Optional: --neo4j-env <existing-file>. Credentials are handled by the existing
connection helper and never included in this script's output.
Exit codes: 0 passed; 2 connection/configuration unavailable; 3 invariant failure;
4 verification could not finish. READ_ACCESS routes reads, but does not replace
server-side read-only privileges. The connection helper itself runs RETURN 1.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgraph.retrieval_types import SUSPICIOUS_CELL_PAIRS, TARGET_NAMESPACE
import import_hybrid_concept_2g3 as stage


EXPECTED_SNAPSHOT_SHA256 = "ba06b63dfba69c4a930101426934c717b733257daf91309b6bb7ecb99ebcc1dc"
EXPECTED_NODES = {"RELEXDocument": 30, "RELEXEntity": 811, "RELEXConcept": 36}
EXPECTED_EDGES = {"RELEX_REL": 1307, "SIMILAR_TO": 492, "HAS_SHARED_CONCEPT": 138, "RELEX_MENTIONS": 811}
CROSS_NAMESPACE = """MATCH (a)-[r]->(b)
WHERE (a.namespace=$ns OR b.namespace=$ns OR r.namespace=$ns)
AND (coalesce(a.namespace,'')<>$ns OR coalesce(b.namespace,'')<>$ns OR coalesce(r.namespace,'')<>$ns)
RETURN count(r) AS count"""


def expected_fingerprint():
    """Reproduce the importer's signature from small, hash-checked artifacts.

    No load_delta/build/main call: those read large decisions or prepare imports.
    Snapshot SHA and fingerprint are distinct checks.
    """
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    manifest = read(stage.REPORT / "manifest.json")
    source_manifest = read(stage.graph.OUT / "manifest.json")
    source_verification = read(stage.graph.OUT / "verification.json")
    require(manifest["source_namespace"] == stage.SOURCE, "local_source_namespace")
    require(manifest["source_sha256"] == source_verification["target_sha256"], "local_source_sha")
    require(manifest["source_fingerprint"] == source_manifest["fingerprint"], "local_source_fingerprint")
    path = stage.REPORT / "new_memberships.jsonl"
    require(stage.graph.base.sha(path) == manifest["artifact_files_sha256"][path.name], "local_membership_hash")
    pairs = {(r["source_entity_id"], r["target_concept_id"])
             for r in stage.graph.base.core.read_jsonl(path)
             if (r.get("embedding_mode"), r.get("threshold"), r.get("status")) == ("text_name", "0.830", "ACCEPT")}
    require(len(pairs) == 8, "local_delta_count")
    # Exact signature in import_hybrid_concept_2g3.build, using its digest helper.
    signature = dict(stage="2G.3", source_sha256=manifest["source_sha256"],
                     source_fingerprint=manifest["source_fingerprint"], target=stage.TARGET,
                     mode="text_name", threshold="0.830", resolver="Hybrid C",
                     delta_sha256=stage.graph.digest(sorted(pairs)))
    return stage.graph.digest(signature)


def require(condition, name):
    if not condition:
        raise ValueError(name)


def check_evidence(props, documents):
    require(bool(props.get("source_triple_ids")), "source_triple_ids")
    sources = json.loads(props["source_relations_json"])
    require(bool(sources), "source_relations")
    windows = 0
    for source in sources:
        require(source["triple_id"] in props["source_triple_ids"], "source_triple_mapping")
        require(source["document_id"] == props["document_id"], "source_document_mapping")
        require(bool(source.get("evidence")), "source_evidence")
        for evidence in source["evidence"]:
            doc = documents[evidence["document_id"]]
            start, end = evidence["start"], evidence["end"]
            require(evidence["document_id"] == source["document_id"], "evidence_document")
            require(0 <= start < end <= len(doc["text"]), "evidence_offsets")
            require(doc["text"][start:end] == evidence["text"], "evidence_exact_slice")
            require(bool(evidence["window_id"]) and bool(doc["revision_id"]) and bool(doc["text_sha256"]), "evidence_provenance")
            windows += 1
    return windows


def verify(session, facts):
    ns = TARGET_NAMESPACE
    raw_nodes = session.run(stage.graph.READ_NODES, ns=ns).data()
    raw_edges = session.run(stage.graph.READ_EDGES, ns=ns).data()
    facts["node_counts"] = dict(Counter(label for row in raw_nodes for label in row["labels"]))
    facts["relationship_counts"] = dict(Counter(row["type"] for row in raw_edges))
    require(len(raw_nodes) == 877 and facts["node_counts"] == EXPECTED_NODES, "node_counts")
    require(len(raw_edges) == 2748 and facts["relationship_counts"] == EXPECTED_EDGES, "relationship_counts")
    nodes, edges = stage.graph.unpack_snapshot({"nodes": raw_nodes, "relationships": raw_edges})
    records = [r["properties"] for r in [*nodes.values(), *edges.values()]]
    require(all(p.get("namespace") == ns and p.get("key", "").startswith(ns + ":") for p in records), "namespace_keys")
    require(all(p["key"] == k for k, r in [*nodes.items(), *edges.items()] for p in [r["properties"]]), "property_keys")
    fingerprints = {p.get("fingerprint") for p in records}
    require(len(fingerprints) == 1 and all(isinstance(f, str) and f for f in fingerprints), "unique_fingerprint")
    fingerprint = next(iter(fingerprints))
    facts["graph_fingerprint"] = fingerprint
    expected = expected_fingerprint()
    facts["expected_fingerprint"] = expected
    require(fingerprint == expected, "expected_fingerprint")
    facts["snapshot_sha256"] = stage.graph.snapshot_hash(nodes, edges)
    require(facts["snapshot_sha256"] == EXPECTED_SNAPSHOT_SHA256, "snapshot_sha256")
    directions = {"RELEX_REL": ("RELEXEntity", "RELEXEntity"), "SIMILAR_TO": ("RELEXEntity", "RELEXEntity"),
                  "HAS_SHARED_CONCEPT": ("RELEXEntity", "RELEXConcept"), "RELEX_MENTIONS": ("RELEXDocument", "RELEXEntity")}
    for edge in edges.values():
        require(edge["head"] in nodes and edge["tail"] in nodes, "relationship_endpoints")
        h, t, p = nodes[edge["head"]], nodes[edge["tail"]], edge["properties"]
        require((h["label"], t["label"]) == directions[edge["type"]], "relationship_direction")
        if edge["type"] == "RELEX_REL":
            require(h["properties"]["document_id"] == t["properties"]["document_id"] == p["document_id"], "within_document_fact")
            require(bool(p.get("relation_type")), "factual_predicate")
        elif edge["type"] == "SIMILAR_TO":
            require(p.get("directed") is False and p.get("undirected") is True, "semantic_direction_flags")
        elif edge["type"] == "HAS_SHARED_CONCEPT":
            require(all(p.get(k) is False for k in ("factual", "identity", "fact_inheritance_allowed")), "concept_safety_flags")
        elif edge["type"] == "RELEX_MENTIONS":
            require(h["properties"]["document_id"] == t["properties"]["document_id"] == p["document_id"], "document_membership")
    facts["directions_checked"] = list(directions)
    semantic = [r for r in edges.values() if r["type"] == "SIMILAR_TO"]
    facts["resolution_decision_2f2"] = dict(Counter(r["properties"].get("resolution_decision_2f2", "MISSING") for r in semantic))
    facts["semantic_self_loops"] = sum(r["head"] == r["tail"] for r in semantic)
    facts["related_count"] = sum(r["properties"].get("resolution_decision_2f2") == "RELATED" for r in semantic)
    facts["trusted_related_nonloops"] = sum(r["properties"].get("resolution_decision_2f2") == "RELATED" and r["head"] != r["tail"] for r in semantic)
    memberships = [r for r in edges.values() if r["type"] == "HAS_SHARED_CONCEPT"]
    suspicious = [r for r in memberships if r["properties"].get("review_status") == "SUSPICIOUS" or r["properties"].get("requires_review") is True]
    pairs = {(nodes[r["head"]]["properties"]["entity_id"], nodes[r["tail"]]["properties"]["concept_id"]) for r in suspicious}
    require(len(suspicious) == 2 and pairs == SUSPICIOUS_CELL_PAIRS, "suspicious_cell_pairs")
    require(all(r["properties"].get("review_status") == "SUSPICIOUS" and r["properties"].get("requires_review") is True for r in suspicious), "suspicious_cell_flags")
    facts["suspicious_cell_memberships"] = 2
    documents = {r["properties"]["document_id"]: r["properties"] for r in nodes.values() if r["label"] == "RELEXDocument"}
    sample = sorted((r for r in edges.values() if r["type"] == "RELEX_REL"), key=lambda r: r["properties"]["key"])[:5]
    facts["sample_evidence_windows"] = sum(check_evidence(r["properties"], documents) for r in sample)
    facts["sample_factual_edges"] = len(sample)
    cross = session.run(CROSS_NAMESPACE, ns=ns).single()["count"]
    require(cross == 0, "cross_namespace_relationships")
    facts["cross_namespace_relationships"] = cross


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neo4j-env", type=Path)
    args = parser.parse_args(argv)
    report = {"namespace": TARGET_NAMESPACE, "status": "not_verified", "verified": {},
              "scope": "Counts, directions, fingerprint, snapshot hash, trust metadata, five factual evidence samples and cross-namespace guard"}
    try:
        driver, database = stage.graph.base.core.connect_neo4j(args.neo4j_env)
    except Exception as exc:
        report.update(reason="Connection/configuration unavailable; no live assertions verified", error_type=type(exc).__name__)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    try:
        from neo4j import READ_ACCESS
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            verify(session, report["verified"])
        report["status"] = "passed"
        code = 0
    except ValueError as exc:
        # Our invariant names are fixed strings; never print driver exception text.
        reason = str(exc)
        report.update(status="failed", reason=reason if reason.replace("_", "").isalnum() else "invalid_graph_or_local_artifact")
        code = 3
    except Exception as exc:
        report.update(status="incomplete", reason="Verification interrupted; remaining checks not verified", error_type=type(exc).__name__)
        code = 4
    finally:
        driver.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
