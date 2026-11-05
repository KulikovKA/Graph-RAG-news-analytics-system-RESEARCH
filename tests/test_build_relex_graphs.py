"""Original E3 offsets, directed prediction conversion, isolation and idempotence."""
import importlib.util
import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("build_relex_graphs", Path(__file__).parents[1] / "scripts/build_relex_graphs.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


@pytest.fixture
def sample():
    entities = [
        {"entity_id": "e:a", "document_id": "D", "start": 4, "end": 9,
         "surface_text": "Alice", "entity_types": ["ACTOR"]},
        {"entity_id": "e:b", "document_id": "D", "start": 15, "end": 19,
         "surface_text": "data", "entity_types": ["DATA_RESOURCE"]},
    ]
    window = {"window_id": "w:1", "document_id": "D", "start": 4, "end": 19,
              "text": "Alice uses data", "entity_ids": ["e:a", "e:b"]}
    raw = {"head": {"start": 0, "end": 5, "text": "Alice"},
           "tail": {"start": 11, "end": 15, "text": "data"}, "relation": "uses", "score": .8}
    return entities, window, raw


def test_original_e3_offsets_and_multitypes_preserved():
    documents, entities = build.load_inputs()
    assert len(documents) == 30
    assert len(entities) == len({e["entity_id"] for e in entities}) == 1211
    assert sum(len(e["source_mentions"]) for e in entities) == 1327
    texts = {d["document_id"]: d["text"] for d in documents}
    for entity in entities:
        assert texts[entity["document_id"]][entity["start"]:entity["end"]] == entity["surface_text"]
        assert set(entity["entity_types"]) == {m["ner_label_hint"] for m in entity["source_mentions"]}


def test_offsets_direction_and_deduplication(sample):
    entities, window, raw = sample
    relation = build.convert_prediction(raw, window, entities)
    assert (relation["head_id"], relation["tail_id"], relation["relation_type"]) == ("e:a", "e:b", "USES")
    duplicate = {**relation, "confidence": .9, "evidence": [{**window, "window_id": "w:2"}]}
    rows = build.deduplicate([relation, duplicate, relation])
    assert len(rows) == 1
    assert rows[0]["confidence"] == .9
    assert len(rows[0]["evidence"]) == 2
    reversed_raw = {**raw, "head": raw["tail"], "tail": raw["head"]}
    assert build.convert_prediction(reversed_raw, window, entities)["triple_id"] != relation["triple_id"]


@pytest.mark.parametrize("failure", ["offset", "text", "document", "label"])
def test_reject_non_e3_endpoints(sample, failure):
    entities, window, raw = sample
    if failure == "offset":
        raw["head"]["start"] = 1
    elif failure == "text":
        raw["tail"]["text"] = "DATA"
    elif failure == "document":
        entities[1]["document_id"] = "OTHER"
    else:
        raw["relation"] = "invented label"
    with pytest.raises(ValueError):
        build.convert_prediction(raw, window, entities)


def test_graph_isolation_and_stable_directed_edge(sample):
    entities, window, raw = sample
    relation = build.convert_prediction(raw, window, entities)
    base = build.edge_row("relex_base", relation, "base", "run")
    large = build.edge_row("relex_large", relation, "large", "run")
    assert base == build.edge_row("relex_base", relation, "base", "run")
    assert base["head_key"] == "relex_base:e:a"
    assert base["tail_key"] == "relex_base:e:b"
    assert base["properties"]["key"] != large["properties"]["key"]
    assert base["properties"]["window_text"] == "Alice uses data"
    assert base["properties"]["relation_type"] == "USES"
    assert base["properties"]["confidence"] == .8
    assert base["properties"]["namespace"] == "relex_base"
    with pytest.raises(ValueError):
        build.edge_row("article-analysis-domain-v1", relation, "base", "run")
    assert "namespace: $namespace" in build.ENTITY_QUERY
    assert build.EDGE_QUERY.count("namespace: $namespace") == 3
    assert "h.document_id = t.document_id" in build.EDGE_QUERY


def test_adaptive_overlapping_windows_preserve_entities():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    document = {"document_id": "D", "text": text}
    entities = [{"entity_id": "a", "start": 6, "end": 16},
                {"entity_id": "b", "start": 31, "end": 39}]
    windows = build.make_windows(document, entities, lambda text: len(text) <= 32)
    assert len(windows) > 1
    assert {e for w in windows for e in w["entity_ids"]} == {"a", "b"}
    assert any(a["end"] > b["start"] for a, b in zip(windows, windows[1:]))
    for window in windows:
        assert window["text"] == text[window["start"]:window["end"]]
        for entity in entities:
            assert not entity["start"] < window["start"] < entity["end"]
            assert not entity["start"] < window["end"] < entity["end"]


def test_suspicious_candidates_retained(sample):
    entities, window, raw = sample
    raw["relation"] = "has attribute"
    relation = build.convert_prediction(raw, window, entities)
    assert relation["suspicious_type_hint"] is True
    assert len(build.deduplicate([relation])) == 1


@pytest.mark.parametrize("architecture", ["token_level", "markerV0"])
def test_supplied_embeddings_equal_vendor_span_layer(architecture):
    """Compare arbitrary E3-order projections with actual trained architecture's grid API."""
    import torch
    from gliner.modeling.span_rep import SpanRepLayer
    torch.manual_seed(13)
    words = torch.randn(1, 5, 8)
    layer = SpanRepLayer(8, 3, architecture, dropout=0).eval()
    grid = torch.tensor([[[i, min(i + width, 4)] for i in range(5) for width in range(3)]])
    selected = torch.tensor([[2, 4], [0, 0], [1, 2]])  # Deliberately different from grid order.
    actual = build.explicit_span_embeddings(layer.span_rep_layer, words, selected)
    if architecture == "markerV0":
        vendor = layer(words, grid).reshape(1, 15, 8)[:, [8, 0, 4], :]
    else:
        vendor = layer(words, selected.unsqueeze(0))
    torch.testing.assert_close(actual, vendor)


@pytest.mark.parametrize("worker_count", [1, 2, 7])
def test_workers_partition_documents_without_duplicates(worker_count):
    documents = [{"document_id": f"D{i}"} for i in range(30)]
    shards = [build.assigned_documents(documents, i, worker_count) for i in range(worker_count)]
    identifiers = [d["document_id"] for shard in shards for d in shard]
    assert len(identifiers) == len(set(identifiers)) == 30
    assert set(identifiers) == {d["document_id"] for d in documents}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_four_resume_workers_use_disjoint_checkpoints_without_inference(tmp_path, monkeypatch):
    documents = [{"document_id": f"D{i}"} for i in range(5)]
    build.atomic_json(tmp_path / "prepared_inputs.json", {"documents": documents, "entities": [], "windows": []})
    config = {"output": str(tmp_path), "workers_per_model": 2, "threads": 2,
              "resume": True, "fingerprint": "run"}
    # A resumed worker must not even import GLiNER when its documents are complete.
    import sys
    monkeypatch.setitem(sys.modules, "gliner", None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "test-original")
    for variant in ("base", "large"):
        for document in documents:
            build.atomic_json(tmp_path / "checkpoints" / variant / f"{document['document_id']}.json",
                              {"fingerprint": "run", **document})
    barrier_calls = []
    barrier = SimpleNamespace(wait=lambda timeout: barrier_calls.append(timeout), abort=lambda: None)
    events = queue.SimpleQueue()
    for variant in ("base", "large"):
        for index in range(2):
            build.worker(variant, index, config, barrier, events)
    progress = {v: [] for v in ("base", "large")}
    completed = set()
    while not events.empty():
        worker_id, event, value = events.get()
        if event == "progress":
            progress[worker_id.split(".")[0]].append(value)
        elif event == "done":
            completed.add(worker_id)
    assert len(barrier_calls) == len(completed) == 4
    for variant in progress:
        assert sorted(progress[variant]) == [d["document_id"] for d in documents]
        timings = [json.loads((tmp_path / f"{variant}.worker{i + 1}.timing.json").read_text()) for i in range(2)]
        assert build.aggregate_worker_timings(timings)["resumed_documents"] == 5
        assert build.aggregate_worker_timings(timings)["processed_documents"] == 0
        assert all((tmp_path / f"{variant}.worker{i + 1}.log").exists() for i in range(2))


def test_parallel_model_timing_uses_longest_worker():
    timings = [
        {"worker_wall_seconds": 8, "inference_wall_seconds": 5, "processed_documents": 15, "resumed_documents": 0},
        {"worker_wall_seconds": 10, "inference_wall_seconds": 6, "processed_documents": 13, "resumed_documents": 2},
    ]
    combined = build.aggregate_worker_timings(timings)
    assert combined["worker_wall_seconds"] == 10
    assert combined["inference_wall_seconds"] == 6
    assert combined["processed_documents"] == 28
    assert combined["resumed_documents"] == 2


@pytest.mark.parametrize("code,expected", [
    ("Neo.ClientError.Security.Unauthorized", "rejected"),
    ("Neo.ClientError.Security.AuthenticationRateLimit", "temporarily blocked"),
])
def test_auth_errors_close_driver_and_explain_environment_priority(monkeypatch, tmp_path, code, expected):
    from neo4j import GraphDatabase

    class ConnectionFailure(Exception):
        pass

    error = ConnectionFailure("sensitive server details")
    error.code = code
    attempts, closed, credentials = [], [], []

    def verify():
        attempts.append(True)
        raise error

    fake_driver = SimpleNamespace(verify_connectivity=verify, close=lambda: closed.append(True))

    def make_driver(uri, *, auth, connection_timeout):
        credentials.append(auth)
        return fake_driver

    monkeypatch.setattr(GraphDatabase, "driver", make_driver)
    monkeypatch.setenv("NEO4J_PASSWORD", "process-only-secret")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    env_file = tmp_path / ".env"
    env_file.write_text("NEO4J_PASSWORD=file-only-secret\n", encoding="utf-8")
    with pytest.raises(build.Neo4jConnectionError) as caught:
        build.connect_neo4j(env_file)
    message = str(caught.value)
    assert expected in message and "override --neo4j-env" in message
    assert "process-only-secret" not in message and "file-only-secret" not in message
    assert "sensitive server details" not in message
    assert credentials == [("neo4j", "process-only-secret")]
    assert attempts == closed == [True]  # One attempt, no credential fallback or retry loop.
