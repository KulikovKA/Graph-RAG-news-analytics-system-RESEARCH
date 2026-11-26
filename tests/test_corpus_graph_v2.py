from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from threading import Barrier, Lock, get_ident
from types import SimpleNamespace

import pytest

from cgraph.shared_concepts_v2 import Entity, digest
from test_shared_concepts_v2 import METADATA, entity, vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import corpus_graph_payload as graph
import corpus_graph_v2 as v2
import corpus_graph_v2_inputs as inputs
import corpus_graph_v2_neo4j as db
import shared_concepts_v2_pipeline as pipeline


class Record(dict):
    def data(self):
        return dict(self)


class Result(list):
    def __init__(self, rows, nodes=0, edges=0):
        super().__init__(Record(row) for row in rows)
        self.counters = SimpleNamespace(nodes_created=nodes, relationships_created=edges)

    def single(self, strict):
        assert strict and len(self) == 1
        return self[0]

    def consume(self):
        return self


class MemoryNeo4j:
    """Синтетический адаптер протокола транзакций, без драйвера и сети."""

    def __init__(self):
        self.nodes, self.edges = {}, {}
        self.writes = 0

    def session(self, database, **_):
        assert database == "neo4j"
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def execute_read(self, command, *args):
        return command(self, *args)

    def execute_write(self, command, *args):
        before = deepcopy((self.nodes, self.edges))
        try:
            return command(self, *args)
        except BaseException:
            self.nodes, self.edges = before
            raise

    def run(self, query, **kwargs):
        ns = kwargs.get("ns")
        if query == graph.READ_NODES:
            return Result(dict(key=key, labels=[row["label"]], properties=row["properties"])
                          for key, row in sorted(self.nodes.items()) if row["properties"]["namespace"] == ns)
        if query == graph.READ_EDGES:
            return Result(row | dict(key=key) for key, row in sorted(self.edges.items())
                          if row["properties"]["namespace"] == ns)
        if query.startswith("MATCH (n) RETURN labels"):
            return Result(dict(labels=[kind], count=n)
                          for kind, n in Counter(row["label"] for row in self.nodes.values()).items())
        if query.startswith("MATCH ()-[r]->() RETURN type"):
            return Result(dict(kind=kind, count=n)
                          for kind, n in Counter(row["type"] for row in self.edges.values()).items())
        if query.startswith("MATCH (n:RELEXDocument {key:$key"):
            return Result([dict(generation=self.nodes[kwargs["key"]]["properties"]["scv2_generation"])])
        if "RETURN count(r) AS n" in query and "coalesce(a.namespace" in query:
            return Result([dict(n=0)])
        if "elementId(n) AS id" in query:
            return Result(dict(id=key, labels=[row["label"]], properties=row["properties"])
                          for key, row in self.nodes.items() if row["properties"]["namespace"] != ns)
        if "elementId(r) AS id" in query:
            return Result([])
        rows = kwargs["rows"]
        self.writes += 1
        if "DELETE r" in query:
            assert all(self.edges[key]["type"] == "HAS_SHARED_CONCEPT" for key in rows)
            for key in rows:
                del self.edges[key]
            return Result([dict(n=len(rows))])
        if "DELETE n" in query:
            assert all(self.nodes[key]["label"] == "RELEXConcept" for key in rows)
            assert all(not {row["head"], row["tail"]}.intersection(rows) for row in self.edges.values())
            for key in rows:
                del self.nodes[key]
            return Result([dict(n=len(rows))])
        if "MERGE (n:" in query:
            made = sum(row["properties"]["key"] not in self.nodes for row in rows)
            for row in rows:
                self.nodes[row["properties"]["key"]] = deepcopy(row)
            return Result([dict(n=len(rows))], nodes=made)
        assert "MERGE (a)-[r:" in query
        made = sum(row["properties"]["key"] not in self.edges for row in rows)
        for row in rows:
            assert row["head"] in self.nodes and row["tail"] in self.nodes
            self.edges[row["properties"]["key"]] = deepcopy(row)
        return Result([dict(n=len(rows))], edges=made)


def sources(keys, *, frozen=False):
    entities = [entity(key, doc=f"doc-{key}", name="Shared term") for key in keys]
    docs, members, mapping = {}, {}, {}
    for e in entities:
        key, did, name = e.entity_id, e.document_id, e.name
        docs[did] = dict(document_id=did, revision_id="r1", text=name)
        raw = dict(entity_id=key, document_id=did, start=0, end=len(name), surface_text=name,
                   entity_types=["METHOD"], normalized_forms=[name], confidence=.99)
        member = dict(source_entity=raw, context=dict(document_id=did, start=0, end=len(name), text=name))
        members[key] = [member]
        mapping[key] = key
    metadata = METADATA | dict(research_graph=pipeline.RESEARCH_GRAPH_NAMESPACE)
    if frozen:
        metadata["corpus"] = "Corpus100"
    return inputs.Inputs(dict(documents=docs, resolution=dict(entities=members, mapping=mapping, relations=[])),
                         entities, metadata)


def test_graph_bootstrap_append_and_crash_resume(tmp_path, monkeypatch):
    target = db.Target("bolt://127.0.0.1:27687", "neo4j", "isolated")
    server = MemoryNeo4j()
    base_inputs = sources("ab", frozen=True)
    options = dict(threads=1, max_seconds=30, max_rss=2**30, checkpoint_every=1)
    config = inputs.approved_config()
    monkeypatch.setattr(v2, "frozen_inputs", lambda _: base_inputs)
    monkeypatch.setattr(pipeline, "historical_cases", lambda _: [])
    monkeypatch.setattr(pipeline, "EXPECTED_EE", dict(entities=2, cross_document_pairs=1, raw_pairs=1,
                         guarded_pairs=1, entities_with_neighbors=2, documents_with_neighbors=2))
    monkeypatch.setattr(v2, "BOOTSTRAP_NODES", dict(RELEXDocument=2, RELEXEntity=2, RELEXConcept=1))
    monkeypatch.setattr(v2, "BOOTSTRAP_EDGES", dict(RELEX_MENTIONS=2, HAS_SHARED_CONCEPT=2))
    monkeypatch.setattr(v2, "BOOTSTRAP_MEMBERS_SHA256", digest([["a", "b"]]))
    previous_output = tmp_path/v2.NAMESPACE
    previous_output.mkdir()
    old_pointer = b"historical publication state"
    (previous_output/"CURRENT.json").write_bytes(old_pointer)
    output = tmp_path/v2.OUTPUT.name
    assert output != previous_output
    write_checked = v2.write_checked

    with monkeypatch.context() as patch:
        def stop_before_pointer(path, value):
            if path.name == "CURRENT.json":
                raise InterruptedError("after server commit, before CURRENT")
            write_checked(path, value)

        patch.setattr(v2, "write_checked", stop_before_pointer)
        with pytest.raises(InterruptedError):
            v2.run_graph(server, target, output, config, options)
    assert not (output/"CURRENT.json").exists() and (output/"PENDING.json").is_file()
    old_writes = server.writes
    result = v2.run_graph(server, target, output, config, options)
    assert result["first"]["created_nodes"] == result["first"]["created_relationships"] == 0
    assert server.writes == old_writes
    current = v2.load_checked(output/"CURRENT.json")
    base_payload = v2.saved_payload(output/"generations"/current["request"], current)
    concept_key, = [k for k, n in server.nodes.items() if n["label"] == "RELEXConcept"]
    assert server.nodes[concept_key]["properties"]["version"] == 1
    assert len(server.nodes[concept_key]["properties"]["context_centroid"]) == 384

    monkeypatch.setattr(v2, "delta_identity", lambda _: dict(base_generation=current["generation"],
                        input_sha256={"delta": "fixture"}))
    monkeypatch.setattr(v2, "delta_inputs", lambda *args: sources("abc"))
    delta = tmp_path/"delta"
    with monkeypatch.context() as patch:
        patch.setattr(v2, "write_checked", stop_before_pointer)
        with pytest.raises(InterruptedError):
            v2.run_graph(server, target, output, config, options, delta=delta)
    assert v2.load_checked(output/"CURRENT.json")["generation"] == current["generation"]
    old_writes = server.writes
    result = v2.run_graph(server, target, output, config, options, delta=delta)
    assert result["mutations"] == 0 and server.writes == old_writes
    assert result["counts"] == dict(nodes=dict(RELEXDocument=3, RELEXEntity=3, RELEXConcept=1),
                                    relationships=dict(RELEX_MENTIONS=3, HAS_SHARED_CONCEPT=3))
    assert server.nodes[concept_key]["properties"]["version"] == 2
    assert len(server.nodes[concept_key]["properties"]["member_ids"]) == 3
    assert all(row["properties"]["factual"] is False and row["properties"]["evidence_json"]
               for row in server.edges.values() if row["type"] == "HAS_SHARED_CONCEPT")
    for key, row in base_payload["edges"].items():
        if row["type"] == "RELEX_MENTIONS":
            assert server.edges[key] == row
    assert v2.run_graph(server, target, output, config, options, delta=delta)["mutations"] == 0
    assert (previous_output/"CURRENT.json").read_bytes() == old_pointer


def test_publication_rejects_other_namespaces_before_build(tmp_path, monkeypatch):
    target = db.Target("bolt://127.0.0.1:27687", "neo4j", "isolated")
    server = MemoryNeo4j()
    server.nodes["protected"] = dict(label="ScientificWork", properties={"namespace": "article-analysis-domain-v1"})
    monkeypatch.setattr(v2, "frozen_inputs", lambda _: pytest.fail("Should stop before build"))
    with pytest.raises(ValueError, match="empty outside"):
        v2.run_graph(server, target, tmp_path/"graph", inputs.approved_config(), {})
    assert server.writes == 0


def test_publication_refuses_legacy_similarities_before_build_or_write(tmp_path, monkeypatch):
    target = db.Target("bolt://127.0.0.1:27687", "neo4j", "isolated")
    server = MemoryNeo4j()
    forbidden = dict(type="SIMILAR_TO", head="a", tail="b", properties={"namespace": v2.NAMESPACE})
    server.edges["historical-edge"] = forbidden
    monkeypatch.setattr(v2, "frozen_inputs", lambda _: pytest.fail("Should stop before build"))
    with pytest.raises(ValueError, match="outside the current V2 profile"):
        v2.run_graph(server, target, tmp_path/"graph", inputs.approved_config(), {})
    server.edges.clear()
    generated = dict(nodes={}, edges={"unexpected": forbidden})
    with pytest.raises(ValueError, match="Unexpected relationship type"):
        db.bootstrap(server, "neo4j", generated)
    assert server.writes == 0


def test_initial_import_uses_four_sessions_and_resumes_after_partial_wave():
    size = 400
    ns = v2.NAMESPACE
    identity = lambda item: f"{ns}:{item}"
    nodes, edges = {}, {}
    for index in range(size):
        doc, entity_key, mention = map(identity, (f"doc-{index:04}", f"ent-{index:04}", f"m-{index:04}"))
        for key, label in ((doc, "RELEXDocument"), (entity_key, "RELEXEntity")):
            nodes[key] = dict(label=label, properties=dict(key=key, namespace=ns, fingerprint="fixture"))
        edges[mention] = dict(type="RELEX_MENTIONS", head=doc, tail=entity_key,
                              properties=dict(key=mention, namespace=ns, fingerprint="fixture"))
    payload = dict(nodes=nodes, edges=edges, expected=db.count_payload(nodes, edges))
    store = MemoryNeo4j()
    gate, mutation = Barrier(4, timeout=10), Lock()
    active_threads = set()
    sessions = []
    interrupt_first_wave = True

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute_read(self, command, *args):
            return store.execute_read(command, *args)

        def execute_write(self, command, query, batch):
            if interrupt_first_wave and "MERGE (n:RELEXDocument" in query:
                with mutation:
                    active_threads.add(get_ident())
                    sessions.append(self)
                gate.wait()
                if batch[0]["properties"]["key"] == identity("doc-0300"):
                    raise InterruptedError("one batch failed")
            with mutation:
                return store.execute_write(command, query, batch)

    class Driver:
        def session(self, database, **_):
            assert database == "neo4j"
            return Session()

    driver = Driver()
    with pytest.raises(InterruptedError, match="one batch failed"):
        db.bootstrap(driver, "neo4j", payload)
    assert len(active_threads) == len({id(session) for session in sessions}) == 4
    assert len(store.nodes) == 300 and not store.edges
    interrupt_first_wave = False
    resumed = db.bootstrap(driver, "neo4j", payload)
    assert resumed["first"]["created_nodes"] == 500
    assert resumed["first"]["created_relationships"] == size
    assert resumed["counts"] == payload["expected"]
    assert db.bootstrap(driver, "neo4j", payload)["first"]["created_nodes"] == 0


def test_relationship_waves_do_not_share_endpoints_between_workers():
    rows = ([dict(head="hub", tail=f"n-{i}") for i in range(21)]
            + [dict(head=f"n-{i}", tail=f"extra-{i}") for i in range(35)]
            + [dict(head="loop", tail="loop")])
    waves = list(graph.import_waves(rows, 4, relationships=True, batch_size=3))
    assert any(len(wave) == 4 for wave in waves)
    assert sorted(id(row) for wave in waves for batch in wave for row in batch) == sorted(map(id, rows))
    for wave in waves:
        seen = set()
        for batch in wave:
            assert 0 < len(batch) <= 3
            keys = {row[side] for row in batch for side in ("head", "tail")}
            assert seen.isdisjoint(keys)
            seen.update(keys)


def test_target_requires_verified_loopback_mapping_and_excludes_main(monkeypatch):
    inspect = dict(Id="fixture", Name="/research", Config=dict(Labels={"com.docker.compose.project": "research"},
                   Image="neo4j:2026.08.1-community"), Mounts=[], State={"Running": True},
                   NetworkSettings={"Ports": {"7687/tcp": [{"HostIp": "127.0.0.1", "HostPort": "27687"}]}})
    monkeypatch.setattr(db.subprocess, "check_output", lambda _: json.dumps([inspect]).encode())
    assert db.inspect_target(db.Target("bolt://127.0.0.1:27687", "neo4j", "research")) == "fixture"
    with pytest.raises(ValueError, match="MAIN"):
        db.inspect_target(db.Target("bolt://127.0.0.1:17687", "neo4j", "research"))
    with pytest.raises(ValueError, match="user database"):
        db.inspect_target(db.Target("bolt://127.0.0.1:27687", "system", "research"))
    with pytest.raises(ValueError, match="does not map"):
        db.inspect_target(db.Target("bolt://127.0.0.1:27688", "neo4j", "research"))
    inspect["Config"]["Labels"]["com.docker.compose.project"] = "article-analysis"
    with pytest.raises(ValueError, match="MAIN"):
        db.inspect_target(db.Target("bolt://127.0.0.1:27687", "neo4j", "research"))


def test_shared_target_requires_opt_in_database_and_verified_proxy(monkeypatch):
    def labels(service):
        return {"com.docker.compose.project": "article-analysis", "com.docker.compose.service": service}

    containers = {
        db.SHARED_CONTAINER: dict(
            Id="shared-id", Name="/"+db.SHARED_CONTAINER, State={"Running": True},
            Mounts=[dict(Destination="/data", Name=db.SHARED_VOLUME)],
            Config=dict(Labels=labels("neo4j"), Image="neo4j:2026.08.1-community"),
            NetworkSettings=dict(Ports={}, Networks={db.SHARED_NETWORK: dict(NetworkID="fixture", Aliases=["neo4j"])}),
        ),
        db.SHARED_PROXY: dict(
            Name="/"+db.SHARED_PROXY, State={"Running": True},
            Config=dict(Labels=labels("neo4j-bolt-proxy"),
                        Cmd=["TCP-LISTEN:17687,fork,reuseaddr", "TCP:neo4j:7687"]),
            NetworkSettings=dict(Ports={"17687/tcp": [{"HostIp": "127.0.0.1", "HostPort": "17687"}]},
                                 Networks={db.SHARED_NETWORK: dict(NetworkID="fixture")}),
        ),
    }
    monkeypatch.setattr(db.subprocess, "check_output", lambda cmd: json.dumps([containers[cmd[-1]]]).encode())
    requested = db.Target(db.SHARED_URI, "neo4j", db.SHARED_CONTAINER)
    with pytest.raises(ValueError, match="--allow-shared-neo4j"):
        db.inspect_target(requested)
    enabled = db.Target(db.SHARED_URI, "neo4j", db.SHARED_CONTAINER, True)
    assert db.inspect_target(enabled) == "shared-id"
    assert enabled.record() == requested.record()
    with pytest.raises(ValueError, match="explicitly approved"):
        db.inspect_target(db.Target(db.SHARED_URI, "other", db.SHARED_CONTAINER, True))
    containers[db.SHARED_PROXY]["Config"]["Cmd"][1] = "TCP:other:7687"
    with pytest.raises(ValueError, match="differs from"):
        db.inspect_target(enabled)


def test_append_rejects_existing_fact_changes():
    relation = dict(type="RELEX_REL", head="a", tail="b", properties=dict(factual=True))
    previous = dict(anchor="doc", nodes={}, edges={"fact": relation})
    changed = deepcopy(previous)
    changed["edges"]["fact"]["properties"]["factual"] = False
    with pytest.raises(ValueError, match="existing facts"):
        db.changed_payload(previous, changed)


def test_approved_configuration_cannot_be_retuned(tmp_path):
    modified = json.loads(inputs.APPROVED.read_text())
    modified["ec"]["context"] = .86
    path = tmp_path / "config.json"
    path.write_text(json.dumps(modified), encoding="utf-8")
    with pytest.raises(ValueError, match="approved EC"):
        inputs.approved_config(path)


@pytest.mark.parametrize("legacy_manifest", (False, True))
def test_delta_from_existing_er_without_semantic_checkpoint(tmp_path, monkeypatch, legacy_manifest):
    previous = sources("a")
    delta = sources("b")
    source = delta.source
    doc = source["documents"]["doc-b"]
    text = doc["text"]
    checksum = sha256(text.encode()).hexdigest()
    doc.update(title=text, source_url="https://example.invalid/b", discipline="fixture",
               contract_version="ner-v2-source-1", year=2026, field={"id": "fixture"}, text_sha256=checksum,
               source_reference={"source_url": "https://example.invalid/b", "text_sha256": checksum})
    member = source["resolution"]["entities"]["b"][0]
    member["context"].update(mention_start=0, mention_end=len(text), source_entity_id="b")
    member["source_mention_ids"] = ["mention-b"]
    mention = {key: member["source_entity"][key] for key in ("document_id", "start", "end", "surface_text")}
    member["source_entity"]["source_mentions"] = [mention | dict(
        revision_id="r1", mention_id="mention-b", abbreviation_candidates=[])]
    (tmp_path/"documents.jsonl").write_text(json.dumps(doc)+"\n", encoding="utf-8")
    inputs.old.checkpoint(tmp_path/"entity_resolution.json", {"stage": "entity_resolution"},
                          source["resolution"], 0)
    profile, = pipeline.resolved_profiles({"doc-b": doc}, tmp_path/"entity_resolution.json")
    ref = profile["evidence"][0] | dict(text_sha256=checksum, window_start=0, window_end=len(text),
                                        input_start=0, input_end=len(text))
    embedded = Entity.create(profile, [dict(text=text, input=profile["name"]+". "+text, provenance=[ref])],
                             (vector(0), vector(0, .5)), digest(METADATA["recipe"]))
    pipeline.write_entity_bundle(tmp_path/"entities.json", [embedded])
    files = {name: v2.sha(tmp_path/name) for name in inputs.DELTA_FILES | {"entities.json"}}
    if legacy_manifest:
        files["semantic.json"] = "unused former checksum"
    v2.write_checked(tmp_path/"manifest.json", dict(base_generation="seed", files_sha256=files))
    monkeypatch.setattr(inputs, "vectorize", lambda *_: pytest.fail("Inference is unnecessary"))
    monkeypatch.setattr(graph, "semantic_links", lambda *_: pytest.fail("Similarity calculation is forbidden"))
    identity = inputs.delta_identity(tmp_path)
    assert "semantic.json" not in identity["input_sha256"]
    prepared = inputs.delta_inputs(tmp_path, identity, previous, {"generation": "seed"}, None,
                                   tmp_path/"cache", 1, 30, 2**30)
    assert list(prepared.source["documents"]) == ["doc-a", "doc-b"]
    assert set(prepared.source) == {"documents", "resolution"}
    assert prepared.entities[-1].fingerprint == embedded.fingerprint
    with pytest.raises(ValueError, match="different parent"):
        inputs.delta_inputs(tmp_path, identity, previous, {"generation": "other"}, None,
                            tmp_path/"cache", 1, 30, 2**30)


def test_frozen_graph_inputs_do_not_read_historical_semantic_artifacts(monkeypatch):
    previous = pipeline.previous
    required = (
        previous.frozen.CORPUS, previous.frozen.FROZEN / "document_lock.json",
        previous.PRIOR / "manifest.json", previous.PRIOR / "checkpoints/entity_resolution/corpus.json",
    )
    archived = (previous.PRIOR / "checkpoints/semantic/corpus.json", previous.PRIOR / "checkpoints/concepts/corpus.json")
    input_hashes = {p.relative_to(pipeline.ROOT).as_posix(): str(i) for i, p in enumerate(required + archived)}
    shared = dict(smoke=False, model_revision=pipeline.REVISION, seed=pipeline.contextual.SEED,
                  entity_ids_sha256="fixture", model_files_sha256={}, pair_cosine={}, inputs_sha256=input_hashes,
                  windows={"S1": 1}, embedding=pipeline.RECIPE)
    historical = shared | dict(force_recompute=False, rules=pipeline.contextual.RULES)
    identity = shared | dict(stage="3A.5.12")
    manifest = dict(stage="3A.5.13", status="complete", fingerprint=pipeline.signature(historical), identity=historical)
    monkeypatch.setattr(pipeline, "read", lambda path: manifest if path == pipeline.FROZEN_MANIFEST else identity)
    visited = []

    def file_sha(path):
        assert path not in archived
        visited.append(path)
        return input_hashes[path.relative_to(pipeline.ROOT).as_posix()]

    class StopAfterInputChecks(Exception):
        pass

    def recipe(_):
        raise StopAfterInputChecks

    monkeypatch.setattr(pipeline, "sha", file_sha)
    monkeypatch.setattr(pipeline, "vector_recipe", recipe)
    with pytest.raises(StopAfterInputChecks):
        pipeline.frozen_corpus(graph_inputs_only=True)
    assert set(visited) == set(required)
