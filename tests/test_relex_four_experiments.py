"""Four experiment entrypoints; mocked predictions only, no weights or Neo4j."""
import importlib
import json
from pathlib import Path
import queue
import subprocess
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("relex_experiment")


@pytest.fixture
def source():
    document = {"document_id": "D", "text": "Alice uses data", "title": "Example", "revision_id": "r:1",
                "source_url": "https://openalex.org/D", "source_reference": {"provider": "openalex"}}
    window = {"window_id": "w:1", "document_id": "D", "start": 0, "end": 15,
              "text": document["text"], "entity_ids": []}
    native = [{"start": 0, "end": 5, "text": "Alice", "label": "actor", "score": .8},
              {"start": 11, "end": 15, "text": "data", "label": "data resource", "score": .9}]
    relation = {"head": {"start": 0, "end": 5, "text": "Alice"},
                "tail": {"start": 11, "end": 15, "text": "data"}, "relation": "uses", "score": .85}
    return document, window, native, relation


def test_namespaces_are_disjoint(experiment):
    namespaces = {experiment.experiment_namespace(m, v) for m in ("e3", "joint") for v in ("base", "large")}
    assert len(namespaces) == 4
    assert namespaces == experiment.core.NAMESPACES
    assert len({experiment.core.graph_key(ns, "same-document-span") for ns in namespaces}) == 4


def test_threshold_namespace_isolated_and_routes_to_large(experiment):
    namespace = experiment.resolve_namespace("e3", "large", "relex_large_t035")
    assert experiment.core.namespace_variant(namespace) == "large"
    assert experiment.core.graph_key(namespace, "node") != experiment.core.graph_key("relex_large", "node")


@pytest.mark.parametrize("namespace", ["article-analysis-domain-v1", "relex_base_t035", "relex_joint_large_t035", "relex_large;DELETE", "relex_large_" + "x" * 64])
def test_invalid_or_other_experiment_namespace_rejected(experiment, namespace):
    with pytest.raises(ValueError):
        experiment.resolve_namespace("e3", "large", namespace)


def test_joint_preserves_native_types_and_window_offsets(experiment, source):
    document, window, native, relation = source
    native.append({**native[0], "label": "object", "score": .75})
    raw = [{"window": window, "entities": native, "predictions": [relation]}]
    entities = experiment.joint_entities(document, raw, ["actor", "data resource", "object"],
                                         ["ACTOR", "DATA_RESOURCE", "OBJECT"])
    assert len(entities) == 2
    assert entities[0]["entity_types"] == ["ACTOR", "OBJECT"]
    assert len(entities[0]["source_mentions"]) == 2
    mapped = experiment.mapped_relations(raw, entities)
    assert mapped[0]["head_id"] == entities[0]["entity_id"]
    assert mapped[0]["tail_id"] == entities[1]["entity_id"]
    assert mapped[0]["relation_type"] == "USES"
    native[0]["text"] = "wrong"
    with pytest.raises(ValueError):
        experiment.joint_entities(document, raw, ["actor", "data resource", "object"], ["ACTOR", "DATA_RESOURCE", "OBJECT"])


def test_native_joint_api_does_not_force_spans(experiment, source):
    _, window, native, relation = source
    seen = {}

    class FakeModel:
        def inference(self, **kwargs):
            seen.update(kwargs)
            return [native], [[relation]]

    entities, relations = experiment.infer_joint_window(FakeModel(), window, ["actor", "data resource"], .3, .7)
    assert entities == native and relations == [relation]
    assert "input_spans" not in seen
    assert seen["return_relations"] is True and seen["flat_ner"] is False
    assert seen["threshold"] == .3 and seen["relation_threshold"] == .7


@pytest.mark.parametrize("script", ["build_e3_relex_base.py", "build_e3_relex_large.py",
                                    "build_joint_relex_base.py", "build_joint_relex_large.py"])
def test_each_entrypoint_exposes_cli_without_running_models(script):
    result = subprocess.run([sys.executable, str(SCRIPTS / script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout and "--threads" in result.stdout and "--workers" in result.stdout


@pytest.mark.parametrize("mode,variant", [(m, v) for m in ("e3", "joint") for v in ("base", "large")])
@pytest.mark.parametrize("use_suffix", [False, True])
def test_each_experiment_writes_and_resumes_without_model_reload(experiment, source, tmp_path, monkeypatch, mode, variant, use_suffix):
    document, window, native, relation = source
    documents = [document, {**document, "document_id": "D2"}]
    windows = [window, {**window, "document_id": "D2", "window_id": "w:2"}]
    labels = [code.lower().replace("_", " ") for code in json.loads(experiment.core.ONTOLOGY.read_text(encoding="utf8"))["entity_types"]]
    codes = list(json.loads(experiment.core.ONTOLOGY.read_text(encoding="utf8"))["entity_types"])
    input_entities = []
    for doc, win in zip(documents, windows):
        members = experiment.joint_entities(doc, [{"window": win, "entities": native}], labels, codes)
        input_entities.extend(members)
        win["entity_ids"] = [e["entity_id"] for e in members]
    e3_calls = []

    def load_e3(limit):
        e3_calls.append(True)
        return documents, input_entities

    monkeypatch.setattr(experiment.core, "load_inputs", load_e3)
    monkeypatch.setattr(experiment.core, "load_documents", lambda limit: documents)
    monkeypatch.setattr(experiment.core, "plan_windows", lambda *args: (windows, labels, 512))
    monkeypatch.setattr(experiment, "model_fingerprints", lambda paths: {v: {} for v in paths})
    monkeypatch.setattr(experiment.core, "predict_window", lambda *args: ([relation], {"spans": 2}))
    loads = []

    class FakeModel:
        def inference(self, **kwargs):
            return [native], [[relation]]

    def load_model(path, threads):
        loads.append(str(path))
        return FakeModel()

    monkeypatch.setattr(experiment, "load_model", load_model)
    shards = []

    def execute_fake_workers(mode, variant, config, documents):
        assert config["workers"] == 2
        events = queue.SimpleQueue()
        barrier = SimpleNamespace(wait=lambda timeout: None, abort=lambda: None)
        for index in range(config["workers"]):
            shards.append(index)
            experiment.process_shard(mode, variant, index, config, barrier, events)
        return [json.loads((tmp_path / f"worker{i + 1}.timing.json").read_text()) for i in range(2)]

    monkeypatch.setattr(experiment, "execute_workers", execute_fake_workers)
    monkeypatch.setattr(experiment.core, "connect_neo4j", lambda *_: pytest.fail("No Neo4j connection allowed"))
    args = ["--inference-only", "--output", str(tmp_path)]
    expected_namespace = experiment.experiment_namespace(mode, variant)
    if use_suffix:
        expected_namespace += "_t035"
        args += ["--namespace", expected_namespace, "--relation-threshold", "0.35"]
    experiment.run(mode, variant, args)
    assert len(loads) == 2 and shards == [0, 1]
    assert bool(e3_calls) == (mode == "e3")
    config = json.loads((tmp_path / "experiment_config.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert config["namespace"] == expected_namespace
    assert summary["metrics"]["unique_triples"] == 2
    assert summary["metrics"]["entities"] == 4
    assert summary["metrics"]["workers"] == 2
    assert summary["neo4j"]["status"] == "NOT_IMPORTED"
    assert len((tmp_path / "relation_distribution.csv").read_text().splitlines()) == 21
    if mode == "joint":
        assert all("mentions_035" not in path for path in config["source_sha256"])
    experiment.run(mode, variant, [*args, "--resume"])
    assert len(loads) == 2 and shards == [0, 1]  # Resume starts no workers when all documents are complete.
    assert json.loads((tmp_path / "summary.json").read_text())["metrics"]["resumed_documents"] == 2
