from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest

import run_neural_reranker_benchmark as runner
from frozen_pool import Pool, ROOT, prepare, render, stage_value, verify_bundles
from model_adapters import causal_prompt, inspect_model, model_path, tokenize_pairs
from neural_reporting import candidate_oracle, evaluate, top_details
from run_integrated_benchmark import Question

sys.path.insert(0, str(ROOT / "tests"))

from cgraph.evidence_extraction import extract_evidence
from cgraph.path_reranking import rerank_paths
from test_evidence_extraction import change_source, non_factual, path
from test_path_reranking import candidate, retrieval


def fixture(question_id="3A52-C03", group="C"):
    good, first = candidate()
    other, second = candidate("RNA bonds proteins.", "r2", "d2")
    bad, third = candidate("RNA uses NAD.", "r3", "d3")
    change_source(bad["edges"][0], lambda sources: sources[0]["evidence"][0].update(text="forged"))
    raw = retrieval([bad, other, good])
    corpus = {doc["document_id"]: doc for doc in (first, second, third)}
    bundles = extract_evidence(raw["paths"], corpus, graph_fingerprint="fp")
    q = Question(group, question_id, "What uses ATP?")
    baseline = rerank_paths(q.text, raw, bundles, policy="C")["paths"]
    names = {edge[e]: (edge["document_id"], e) for p in raw["paths"] for edge in p["edges"]
             for e in ("head_key", "tail_key")}
    pool = Pool(q, raw, verify_bundles(raw, bundles, corpus), [], baseline, names, {})
    return pool, corpus, good, other, bad


def reference(corpus, doc):
    return {"document_contributions": {doc: ["synthetic"]},
            "evidence": [{"document_id": doc, "revision_id": corpus[doc]["revision_id"],
                          "start": 0, "end": 2, "evidence_id": "E01"}]}


def test_deterministic_preselection_and_gold_isolation():
    pool, corpus, good, other, bad = fixture()
    prepare(pool)
    order = list(pool.order)
    pool.raw["paths"].reverse()
    prepare(pool)
    assert pool.order == order
    assert bad["path_id"] not in order
    scores = {good["path_id"]: -1.0, other["path_id"]: 1000.0}
    ranked, details = runner.neural_rankings(pool, scores, order)
    assert ranked["neural_only"][0]["path_id"] == other["path_id"]
    assert evaluate(top_details(ranked["neural_only"][:1]), reference(corpus, "d1"))["GoldDocRecall@5"] == 0
    assert evaluate(top_details(ranked["neural_only"][:1]), reference(corpus, "d2"))["GoldDocRecall@5"] == 1
    transformed = {pid: score * 20 + 50 for pid, score in scores.items()}
    assert runner.neural_rankings(pool, transformed, order)[1]["hybrid_C_neural"] == details["hybrid_C_neural"]
    assert candidate_oracle(pool, reference(corpus, "d3"))["200"]["absent_factual_gold"] == ["d3"]
    with pytest.raises(ValueError, match="invalid evidence"):
        runner.neural_rankings(pool, {**scores, bad["path_id"]: 1e50}, [*order, bad["path_id"]])
    assert runner.rank_percentiles({"z": 2, "a": 2, "b": 1}) == {"a": .75, "z": .75, "b": 0}


def test_forged_cached_provenance_and_missing_raw_cache(tmp_path):
    pool, corpus, good, _, _ = fixture()
    bundles = deepcopy(list(pool.bundles.values()))
    item = next(b for b in bundles if b["path_id"] == good["path_id"])
    item["edges"][0]["source_relations"][0]["windows"][0]["text"] = "forged"
    with pytest.raises(ValueError, match="source-exact"):
        verify_bundles(pool.raw, bundles, corpus)
    row = {"key": pool.question.key, "timings": {"bfs": {"key": "pin", "file": "cache/bfs/pin.json"}}}
    with pytest.raises(ValueError, match="cache is missing:.*pin.json.*--variants B"):
        stage_value(tmp_path, row, "bfs", "pin")


def test_off_gold_sources_exclude_navigation_only_documents():
    pool, corpus, *_ = fixture()
    top = top_details(pool.baseline[:1])
    top[0]["document_ids"] = [*top[0]["document_ids"], "navigation-doc"]
    metrics = evaluate(top, reference(corpus, "d1"))
    assert "navigation-doc" in metrics["off_gold_documents"]
    assert "navigation-doc" not in metrics["off_gold_sources"]


def test_render_keeps_document_identity():
    pool, _, good, other, _ = fixture()
    left, right = good["edges"][0], other["edges"][0]
    nav = non_factual("SIMILAR_TO", "b", "d2:a", "n1", resolution_decision_2f2="RELATED")
    joined = path([left, nav, right])
    bundle = {"edges": [pool.bundles[good["path_id"]]["edges"][0], {"edge": nav},
                        pool.bundles[other["path_id"]]["edges"][0]]}
    fact = render(joined, bundle, pool.names, "fact_focused")
    grouped = render(joined, bundle, pool.names, "document_aware")
    assert "SIMILAR_TO (navigation only" in fact
    assert "Evidence from d1 [0:13]\nRNA uses ATP." in fact
    assert "Evidence from d2 [0:19]\nRNA bonds proteins." in grouped
    assert grouped.index("RNA uses ATP.") < grouped.index("Document d2")


@pytest.mark.parametrize("adapter,labels", [("qwen", ("yes", "no")), ("mixedbread", ("1", "0"))])
def test_causal_truncation_preserves_question_and_suffix(adapter, labels):
    class CharTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(map(ord, t)) for t in texts],
                    "offset_mapping": [[(i, i + 1) for i in range(len(t))] for t in texts]}

        def pad(self, batch, **kwargs):
            return batch

    prefix, suffix, actual = causal_prompt(adapter, "Which result?")
    limit = len(prefix + suffix) + 3
    result, counts = tokenize_pairs(CharTokenizer(), "Which result?", ["abcdef"], limit, adapter)
    assert actual == labels
    assert "".join(map(chr, result["input_ids"][0])) == prefix + "abc" + suffix
    assert counts[0]["original_tokens"] == limit + 3
    assert counts[0]["truncated_tokens"] == 3


def test_missing_model_and_corrupt_checkpoint(tmp_path):
    override = tmp_path / "missing"
    assert model_path("qwen3-0.6b", tmp_path, {"qwen3-0.6b": str(override)}) == override
    model = inspect_model("qwen3-0.6b", override)
    assert model["status"] == "UNAVAILABLE" and "config.json" in model["error"]
    runner.checkpoint(tmp_path, "a", "baseline", "C", "03", value={"one": 1})
    assert runner.checkpoint(tmp_path, "a", "baseline", "C", "03") == {"one": 1}
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        runner.checkpoint(tmp_path, "b", "baseline", "C", "03")


def test_interruption_resume_and_missing_model_isolation(monkeypatch, tmp_path):
    questions = [fixture(group=group) for group in ("C", "HOLDOUT")]
    pools = [p for p, *_ in questions]
    refs = {pool.question.key: reference(corpus, "d1") for pool, corpus, *_ in questions}
    calls = []

    class FakeAdapter:
        def __init__(self, *args):
            pass

        def score(self, question, passages):
            calls.append(passages)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return list(range(len(passages))), [{"original_tokens": 5, "input_tokens": 5,
                                                "truncated_tokens": 0}] * len(passages), .01, .02

    monkeypatch.setattr(runner, "Adapter", FakeAdapter)
    settings = {**runner.DEFAULTS, "models": ["minilm-l6", "qwen3-0.6b"], "warmup_batches": 0}
    models = {"minilm-l6": {"status": "READY", "path": str(tmp_path)},
              "qwen3-0.6b": {"status": "UNAVAILABLE", "error": "Missing model fixture"}}
    config = {"settings": settings, "tiers": {"minilm-l6": "Small", "qwen3-0.6b": "Large"},
              "question_keys": list(refs)}
    args = SimpleNamespace(resume=False, no_progress=True)
    assert runner.run(tmp_path, config, models, pools, refs, args) == 130
    saved = tmp_path / "scores/minilm-l6/fact_focused/C/3A52-C03.json"
    original = saved.read_bytes()
    args.resume = True
    assert runner.run(tmp_path, config, models, pools, refs, args) == 3
    assert len(calls) == 3
    assert saved.read_bytes() == original
    assert (tmp_path / "scores/minilm-l6/fact_focused/HOLDOUT/3A52-C03.json").is_file()
    assert "Missing model fixture" in (tmp_path / "errors.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Resume fingerprint mismatch"):
        runner.verify_resume(tmp_path, {**config, "settings": {**settings, "threads": 8}}, models)
