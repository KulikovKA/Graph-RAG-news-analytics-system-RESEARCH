"""Идемпотентная публикация одного изолированного поколения графа RESEARCH."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
import re
import subprocess
from urllib.parse import urlsplit

from neo4j import GraphDatabase, READ_ACCESS
from tqdm import tqdm

import corpus_graph_payload as graph
from shared_concepts_v2_pipeline import RESEARCH_GRAPH_NAMESPACE as NAMESPACE


@dataclass(frozen=True)
class Target:
    uri: str
    database: str
    container: str

    def record(self):
        return asdict(self)


def inspect_target(target):
    uri = urlsplit(target.uri)
    if (uri.scheme != "bolt" or uri.hostname != "127.0.0.1" or not uri.port
            or uri.username or uri.password or uri.query or uri.fragment or uri.path
            or not re.fullmatch(r"[a-z][a-z0-9.-]{0,62}", target.database) or target.database == "system"
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", target.container)):
        raise ValueError("Use an explicitly named local Docker container and its user database over 127.0.0.1 Bolt")
    container, = json.loads(subprocess.check_output(["docker", "inspect", "--type", "container", target.container]))
    labels = container["Config"].get("Labels") or {}
    data_mounts = [mount for mount in container["Mounts"] if mount["Destination"] == "/data"]
    if (uri.port == 17687 or container["Name"] == "/article-analysis-neo4j-1"
            or labels.get("com.docker.compose.project") == "article-analysis"
            or any("article-analysis_neo4j_data" in str(mount) for mount in data_mounts)):
        raise ValueError("MAIN/shared local Neo4j is excluded. Use an isolated RESEARCH container")
    if (container["Name"] != "/" + target.container or not container["State"]["Running"]
            or not container["Config"]["Image"].split("/")[-1].startswith("neo4j:")):
        raise ValueError("Expected a running Neo4j container")
    ports = container["NetworkSettings"]["Ports"].get("7687/tcp") or []
    if {"HostIp": "127.0.0.1", "HostPort": str(uri.port)} not in ports:
        raise ValueError("Bolt does not map directly to the specified RESEARCH container on loopback")
    return container["Id"]


def read_credentials(env_file):
    values = {}
    if env_file:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            match = re.fullmatch(r"\s*(NEO4J_(?:USER|PASSWORD))\s*=\s*(.*?)\s*", line)
            if match:
                values[match[1]] = match[2].strip("\"'")
    user = os.environ.get("NEO4J_USER", values.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", values.get("NEO4J_PASSWORD"))
    if not password:
        raise ValueError("Set NEO4J_PASSWORD or use --neo4j-env for the isolated RESEARCH database")
    return user, password


def connect(target, env_file):
    inspect_target(target)
    driver = GraphDatabase.driver(target.uri, auth=read_credentials(env_file), connection_timeout=10)
    try:
        with driver.session(database=target.database, default_access_mode=READ_ACCESS) as session:
            data = session.run("RETURN 1").consume().database
            if data != target.database:
                raise ValueError("Neo4j connection resolved to a different database")
        return driver
    except BaseException:
        driver.close()
        raise


def count_payload(nodes, edges):
    return dict(nodes=dict(Counter(row["label"] for row in nodes.values())),
                relationships=dict(Counter(row["type"] for row in edges.values())))


def counts(tx):
    return dict(
        nodes={record["labels"][0] if len(record["labels"]) == 1 else tuple(record["labels"]): record["count"]
               for record in tx.run("MATCH (n) RETURN labels(n) AS labels, count(n) AS count")},
        relationships={record["kind"]: record["count"] for record in tx.run(
            "MATCH ()-[r]->() RETURN type(r) AS kind, count(r) AS count")},
    )


def isolated(tx):
    snapshot = graph.outside_snapshot(tx, NAMESPACE)
    if snapshot["nodes"] or snapshot["relationships"]:
        raise ValueError("RESEARCH graph publication requires a database empty outside the V2 namespace")


def verify(tx, payload):
    isolated(tx)
    actual = counts(tx)
    if actual != payload["expected"]:
        raise ValueError(f"Neo4j aggregate mismatch: expected {payload['expected']}; found {actual}")
    checked, _ = graph.verify_neo4j(tx, NAMESPACE, payload["nodes"], payload["edges"])
    return dict(counts=actual, readback=checked)


def bootstrap(driver, database, payload):
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        session.execute_read(isolated)
    result = graph.import_neo4j(driver, database, NAMESPACE, payload["nodes"], payload["edges"])
    with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
        return result | session.execute_read(verify, payload)


def changed_payload(previous, target):
    remove_nodes, remove_edges = {}, {}
    put_nodes, put_edges = {}, {}
    anchor = previous["anchor"]
    if target["anchor"] != anchor:
        raise ValueError("The frozen generation anchor cannot move")
    for key, row in previous["nodes"].items():
        new = target["nodes"].get(key)
        if key == anchor:
            generation_fields = {"scv2_generation", "scv2_catalog_sha256"}
            stable = lambda item: {name: value for name, value in item["properties"].items()
                                   if name not in generation_fields}
            if new is None or row["label"] != new["label"] or stable(row) != stable(new):
                raise ValueError("Append cannot change the frozen anchor document")
        elif new is not None and new["label"] != row["label"]:
            raise ValueError(f"Append cannot change an existing node label: {key}")
        elif row["label"] != "RELEXConcept" and new != row:
            raise ValueError(f"Append cannot change existing document or entity: {key}")
        if new is None:
            remove_nodes[key] = row
    for key, row in previous["edges"].items():
        new = target["edges"].get(key)
        if new is not None and new["type"] != row["type"]:
            raise ValueError(f"Append cannot change an existing relationship type: {key}")
        if row["type"] != "HAS_SHARED_CONCEPT" and new != row:
            raise ValueError(f"Append cannot change existing facts, mentions or SIMILAR_TO: {key}")
        if new is None or (new["head"], new["tail"]) != (row["head"], row["tail"]):
            remove_edges[key] = row
    put_nodes.update({key: row for key, row in target["nodes"].items() if previous["nodes"].get(key) != row})
    put_edges.update({key: row for key, row in target["edges"].items() if previous["edges"].get(key) != row})
    return remove_nodes, remove_edges, put_nodes, put_edges


def run_batches(tx, query, rows, bar):
    for start in range(0, len(rows), 100):
        batch = rows[start:start+100]
        matched = tx.run(query, rows=batch, ns=NAMESPACE).single(strict=True)["n"]
        if matched != len(batch):
            raise ValueError("Neo4j mutation did not match its planned keys")
        bar.update(len(batch))


def publish_append(driver, database, previous, target):
    remove_nodes, remove_edges, put_nodes, put_edges = changed_payload(previous, target)
    size = sum(map(len, (remove_nodes, remove_edges, put_nodes, put_edges)))

    def transaction(tx):
        marker = tx.run(
            "MATCH (n:RELEXDocument {key:$key, namespace:$ns}) RETURN n.scv2_generation AS generation",
            key=previous["anchor"], ns=NAMESPACE).single(strict=True)["generation"]
        if marker == target["generation"]:
            return dict(status="PASS", mutations=0, **verify(tx, target))
        if marker != previous["generation"]:
            raise ValueError("Neo4j is not at the parent generation")
        verify(tx, previous)
        with tqdm(total=size, desc="Neo4j atomic append", unit="object") as bar:
            run_batches(tx, "UNWIND $rows AS key MATCH ()-[r:HAS_SHARED_CONCEPT {key:key, namespace:$ns}]->() "
                        "DELETE r RETURN count(r) AS n", sorted(remove_edges), bar)
            run_batches(tx, "UNWIND $rows AS key MATCH (n:RELEXConcept {key:key, namespace:$ns}) "
                        "DELETE n RETURN count(n) AS n", sorted(remove_nodes), bar)
            for kind in graph.LABELS:
                run_batches(tx, f"UNWIND $rows AS row MERGE (n:{kind} {{key:row.properties.key,namespace:$ns}}) "
                            "SET n = row.properties RETURN count(n) AS n",
                            [row for row in put_nodes.values() if row["label"] == kind], bar)
            for kind, (head, tail) in graph.EDGE_LABELS.items():
                run_batches(tx, f"UNWIND $rows AS row MATCH (a:{head} {{key:row.head,namespace:$ns}}),"
                            f"(b:{tail} {{key:row.tail,namespace:$ns}}) "
                            f"MERGE (a)-[r:{kind} {{key:row.properties.key,namespace:$ns}}]->(b) "
                            "SET r = row.properties RETURN count(r) AS n",
                            [row for row in put_edges.values() if row["type"] == kind], bar)
        return dict(status="PASS", mutations=size, **verify(tx, target))

    with driver.session(database=database) as session:
        result = session.execute_write(transaction)
        if session.execute_read(verify, target) != {key: result[key] for key in ("counts", "readback")}:
            raise ValueError("Neo4j read-back changed after append commit")
        return result
