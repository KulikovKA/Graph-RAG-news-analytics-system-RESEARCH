import csv
from collections import Counter
from io import StringIO
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cgraph.entity_linking import EntityCache, EntityLinker
from cgraph.evidence_extraction import extract_evidence
from cgraph.path_reranking import rerank_paths
from cgraph.retrieval_types import TARGET_NAMESPACE
from test_path_reranking import candidate, retrieval


BENCHMARK = Path(__file__).resolve().parents[1] / "reports/retrieval/benchmark_3a5_2"
sys.path.insert(0, str(BENCHMARK))
import run_integrated_benchmark as runner
from integrated_benchmark_reporting import write_reports


def test_frozen_dataset_and_variant_configs():
    questions, references = runner.load_questions(runner.documents())
    assert Counter(q.group for q in questions) == {"A": 8, "B": 7, "C": 10, "HOLDOUT": 10}
    assert len(references) == len({q.key for q in questions}) == 35
    assert all(f"{g}/3A52-C03" in references for g in ("C", "HOLDOUT"))
    assert all(set(vars(q)) == {"group", "question_id", "text"} for q in questions)
    assert [(v["reranking"], v["linking"]["top_k"], v["bfs"]["max_candidates"])
            for v in runner.VARIANTS.values()] == [("A", 5, 200), ("C", 5, 200), ("C", 10, 200), ("C", 10, 100)]
    for config in runner.VARIANTS.values():
        assert config["linking"]["min_score"] == 0.85 and config["linking"]["fallback"] == "baseline"
        assert config["gliner_threshold"] == 0.15 and config["rerank_top_k"] == 5
        assert config["seed_fair"] is False
        assert {k: v for k, v in config["bfs"].items() if k != "max_candidates"} == {
            "max_hops": 5, "max_neighbors": 50, "max_relationships": 1000,
            "max_seeds": 32, "timeout_seconds": 10.0,
        }


@pytest.fixture
def synthetic(tmp_path):
    p, doc = candidate()
    other, second = candidate("RNA interacts with ATP.", "r2", "d2")
    documents = {"d1": doc, "d2": second}
    nodes = {node: {"document_id": document_id} for path, document_id in ((p, "d1"), (other, "d2"))
             for node in path["node_keys"]}
    gold = {
        "document_contributions": {"d1": ["gold"]}, "expected_answer": "gold only",
        "evidence": [{"evidence_id": "s1", "document_id": "d1",
                      "revision_id": doc["revision_id"], "start": 0, "end": 3}],
    }
    entities = [{"entity_id": "a", "document_id": "d1", "text_name": "RNA"}] + [
        {"entity_id": f"other-{i}", "document_id": "d2", "text_name": f"Unmatched {i}"} for i in range(12)
    ]
    cache = EntityCache(entities, np.array([[1., 0.]] + [[0., 1.]] * 12, dtype=np.float32), "cache", {})
    calls = []
    question = runner.Question("C", "3A52-C03", "What uses RNA and ATP?")

    def analyze(text):
        assert type(text) is str and text == question.text
        calls.append("ner")
        return {"text": text, "mentions": [
            {"mention_id": "m", "surface_text": "RNA", "start": 10, "end": 13},
        ]}

    def encode(texts):
        calls.append(("e5", tuple(texts)))
        return np.array([[1., 0.]] * len(texts), dtype=np.float32)

    def retrieve(seeds, limits, *, seed_fair):
        assert seeds[0]["node_key"] == f"{TARGET_NAMESPACE}:a"
        assert not seed_fair
        calls.append(("bfs", limits.max_candidates))
        return retrieval([p, other], truncated=limits.max_candidates == 100)

    def extract(result):
        calls.append("evidence")
        return extract_evidence(result["paths"], documents, graph_fingerprint="fp")

    def rank(text, result, evidence, seeds, **config):
        assert text == question.text
        calls.append(("rerank", config["policy"]))
        return rerank_paths(text, result, evidence, seeds, **config)

    messages = []
    progress = runner.Progress(False, messages.append, [question], runner.VARIANTS, {})
    stages = runner.StageCache(tmp_path, "test-fingerprint", progress)
    pipeline = runner.Pipeline(stages, cache, SimpleNamespace(encode=encode), SimpleNamespace(analyze=analyze),
                               SimpleNamespace(retrieve=retrieve), SimpleNamespace(extract=extract), rank, "fp")
    return SimpleNamespace(question=question, nodes=nodes, pipeline=pipeline, gold=gold,
                           calls=calls, progress=progress, messages=messages)


def test_production_linking_shared_score_traversal_and_evidence(synthetic):
    pipeline, question = synthetic.pipeline, synthetic.question
    original_results = []
    for variant in runner.VARIANTS:
        retrieval_result, bundles, ranking, timings, trace = pipeline.evaluate(question.text, variant)
        fresh = EntityLinker(pipeline.linker.cache, pipeline.linker.encoder,
                             **runner.VARIANTS[variant]["linking"])
        query = {"text": question.text,
                 "mentions": [{"mention_id": "m", "surface_text": "RNA", "start": 10, "end": 13}]}
        assert trace["seeds"] == fresh.link(query)["seeds"]
        synthetic.calls.pop()  # Убираем отдельный эталонный вызов E5 из счётчика runner.
        original_results.append(timings)
    a, b, c, d = original_results
    assert synthetic.calls.count("ner") == 1
    assert [call for call in synthetic.calls if call[0] == "e5"] == [("e5", ("RNA",))]
    assert [call for call in synthetic.calls if call[0] == "bfs"] == [("bfs", 200), ("bfs", 100)]
    assert synthetic.calls.count("evidence") == 1
    assert a["bfs"]["key"] == b["bfs"]["key"] == c["bfs"]["key"] != d["bfs"]["key"]
    assert {entry["evidence"]["key"] for entry in original_results} == {a["evidence"]["key"]}
    assert a["e5"]["seconds"] == b["e5"]["seconds"] == c["e5"]["seconds"] == d["e5"]["seconds"]
    assert a["e5"]["origin"] == "computed" and all(r["e5"]["origin"] != "computed" for r in (b, c, d))
    assert a["rerank"]["key"] != b["rerank"]["key"]
    assert b["rerank"]["key"] == c["rerank"]["key"]


def test_resume_errors_and_report(tmp_path, synthetic):
    question, pipe = synthetic.question, synthetic.pipeline
    observed_errors = []
    failed = pipe.evaluate

    def interrupt_a(text, variant):
        if variant == "B":
            raise RuntimeError("private driver info")
        return failed(text, variant)

    pipe.evaluate = interrupt_a
    saved = {}
    args = (tmp_path, "test-fingerprint", [question], ["A", "B"], {question.key: synthetic.gold}, synthetic.nodes)
    runner.execute(*args, pipe, synthetic.progress, saved, lambda *error: observed_errors.append(error))
    assert list(saved) == [(question.key, "A")]
    assert observed_errors[0][-1] == "B" and "private" not in runner.diagnostic(observed_errors[0][1])
    restored = runner.restore_checkpoints(tmp_path, "test-fingerprint", [question], ["A", "B"], synthetic.messages.append)
    assert restored == saved
    assert runner.restore_checkpoints(tmp_path, "different", [question], ["A", "B"], synthetic.messages.append) == {}
    pipe.evaluate = failed
    before = len(synthetic.calls)
    runner.execute(*args, pipe, synthetic.progress, restored, pytest.fail)
    assert synthetic.calls[before:] == [("rerank", "C")]
    assert len(restored) == 2
    a = restored[question.key, "A"]
    assert a["metrics"]["GoldDocRecall@5"] == 1
    assert a["metrics"]["outside_gold_count"] == 1
    assert a["metrics"]["EvidenceWindowOverlap@5"] == 1
    assert a["metrics"]["SemanticEvidenceCoverage@5"] == "NOT_EVALUATED"
    assert a["top_5"][0]["facts"][0]["source_triple_id"]
    rerank_path = tmp_path / a["timings"]["rerank"]["file"]
    assert runner.read_json(rerank_path)["record"]["value"]["paths"][0]["evidence"]["edges"][0]["source_relations"]
    manifest = {"status": "COMPLETE", "completed_questions": 1, "expected_questions": 1,
                "completed_evaluations": 2, "expected_evaluations": 2, "started_at_utc": "2026-11-22",
                "attempts": []}
    config = {"selection": {"variants": ["A", "B"]}, "variants": runner.VARIANTS,
              "identity": {"git_commit": "test", "versions": {"python": "3.11", "torch": "2", "neo4j": "6"},
                           "graph": {"namespace": TARGET_NAMESPACE, "graph_fingerprint": "fp"},
                           "models": {"GLiNER": {"id": "e3"}, "E5": {"id": "e5"}}}}
    write_reports(tmp_path, config, manifest, [question], restored)
    with (tmp_path / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {(r["group"], r["variant"]) for r in rows} == {("ALL", "A"), ("ALL", "B"), ("C", "A"), ("C", "B")}
    assert all(float(r["gold_recall_micro"]) == 1.0 for r in rows)
    result = runner.read_json(tmp_path / "results.json")
    assert result["questions"][0]["best"] == ["A", "B"]
    assert result["paired_effects"][0]["comparison"] == "A → B"
    assert "C/3A52-C03" in (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert len((tmp_path / "per_question.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    rerank_path.write_text("corrupted", encoding="utf-8")
    assert list(runner.restore_checkpoints(
        tmp_path, "test-fingerprint", [question], ["A", "B"], synthetic.messages.append)) == [(question.key, "B")]


def test_tqdm_restores_counts_and_renders_variants(synthetic, monkeypatch):
    stream = StringIO()
    monkeypatch.setattr(sys, "stderr", stream)
    q = synthetic.question
    holdout = runner.Question("HOLDOUT", q.question_id, q.text)
    progress = runner.Progress(True, synthetic.messages.append, [q, holdout], ["A", "B"],
                               {(q.key, "A"): {"variant": "A"}})
    assert [bar.n for bar in progress.bars] == [1, 0, 0]
    progress.question(holdout)
    progress.variant("A")
    assert progress.bars[2].n == 1
    progress.variant("B")
    assert progress.bars[2].n == 0
    progress.stage("Evidence Extraction")
    progress.close()
    assert all(word in stream.getvalue() for word in ("Overall benchmark", "Questions", "Variant B", "restored=1",
                                                      "HOLDOUT/C03", "Evidence Extraction"))


def test_run_lock_and_atomic_json(tmp_path):
    with runner.run_lock(tmp_path):
        with pytest.raises(runner.BenchmarkError, match="locked"):
            with runner.run_lock(tmp_path):
                pytest.fail("second writer acquired the lock")
    with runner.run_lock(tmp_path):
        path = tmp_path / "atomic.json"
        runner.write_json(path, {"ok": 1})
        with pytest.raises(ValueError):
            runner.write_json(path, {"ok": float("nan")})
        assert json.loads(path.read_bytes()) == {"ok": 1}


def test_model_logging_keeps_library_bars_off_the_console(tmp_path, capsys):
    from transformers.utils import logging as transformers_logging

    previous = transformers_logging.is_progress_bar_enabled()
    log = tmp_path / "execution.log"
    with runner.model_output(log):
        assert not transformers_logging.is_progress_bar_enabled()
        logging.getLogger("transformers").warning("test model load")
    assert transformers_logging.is_progress_bar_enabled() == previous
    assert "test model load" in log.read_text(encoding="utf-8")
    assert capsys.readouterr().err == ""


def test_run_interrupts_and_resumes_without_recalculating(tmp_path, synthetic, monkeypatch):
    question = synthetic.question
    evaluation = synthetic.pipeline.evaluate
    args = SimpleNamespace(resume=False, no_progress=True)
    graph = {"namespace": TARGET_NAMESPACE, "graph_fingerprint": "fp"}
    identity = {"graph": graph, "git_commit": "test", "versions": {"python": "3.11", "torch": "2", "neo4j": "6"},
                "models": {"GLiNER": {"id": "E3"}, "E5": {"id": "E5"}}}
    config = {"selection": {"variants": ["A", "B"]}, "variants": runner.VARIANTS, "identity": identity}

    def interrupt(text, variant):
        if variant == "B":
            raise KeyboardInterrupt
        return evaluation(text, variant)

    def make_pipeline(stages, *_):
        synthetic.pipeline.stages = stages
        return synthetic.pipeline

    monkeypatch.setattr(runner, "Pipeline", make_pipeline)
    monkeypatch.setattr(runner, "verify_graph", lambda *_: (graph, {}))
    params = (tmp_path, config, None, None, synthetic.nodes, object(), synthetic.pipeline.linker.cache,
              [question], {question.key: synthetic.gold}, synthetic.messages.append)
    synthetic.pipeline.evaluate = interrupt
    assert runner.run_benchmark(args, *params) == 130
    manifest = runner.read_json(tmp_path / "manifest.json")
    assert manifest["status"] == "INTERRUPTED" and manifest["completed_evaluations"] == 1
    assert json.loads((tmp_path / "errors.jsonl").read_bytes())["error_type"] == "KeyboardInterrupt"

    synthetic.pipeline.evaluate = evaluation
    args.resume = True
    before = len(synthetic.calls)
    assert runner.run_benchmark(args, *params) == 0
    assert synthetic.calls[before:] == [("rerank", "C")]
    manifest = runner.read_json(tmp_path / "manifest.json")
    assert manifest["status"] == "COMPLETE" and manifest["completed_questions"] == 1
    assert manifest["attempts"][1]["restored_evaluations"] == 1
    with pytest.raises(runner.BenchmarkError, match="experiment is complete"):
        runner.run_benchmark(args, *params)


def test_baseline_a_replays_one_archived_question():
    from run_document_aware import BASELINE, replay_baseline, restore_question

    archive = runner.read_json(BASELINE)
    saved = next(row for row in archive["questions"] if row["question_id"] == "3A52-C05")
    graph_fingerprint = archive["preflight"]["graph_fingerprint"]
    stored_retrieval, bundles = restore_question(saved, runner.documents(), graph_fingerprint)
    seeds = saved["entity_linking"]["traversal_seeds"]
    replayed = replay_baseline(saved, stored_retrieval, bundles, seeds)
    settings = runner.VARIANTS["A"]
    assert archive["settings"]["traversal"] == settings["bfs"]
    assert archive["settings"]["experimental_link_min_score"] == settings["linking"]["min_score"]
    assert archive["settings"]["link_top_k"] == settings["linking"]["top_k"]
    assert rerank_paths(saved["question"], stored_retrieval, bundles, seeds,
                        policy=settings["reranking"], top_k=settings["rerank_top_k"]) == replayed
