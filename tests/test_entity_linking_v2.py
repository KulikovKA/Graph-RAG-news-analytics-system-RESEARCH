from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cgraph.entity_linking_v2 import (
    NAMESPACE, EntityLinkerV2, LinkingPolicy, Similarities, merge_mentions, rank_mentions,
)
from cgraph.shared_concepts_v2 import DIMENSION, Entity, digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import corpus_graph_v2 as graph
import link_corpus100 as corpus
import study_entity_linking_v2 as study
from shared_concepts_v2_pipeline import load_checked, write_checked


def vector(x, y=0):
    values = np.zeros(DIMENSION)
    values[:2] = x, y
    return values


def mention(text, surface):
    start = text.index(surface)
    return dict(mention_id=f"m:{start}", surface_text=surface, start=start, end=start + len(surface))


def query(text, *surfaces):
    return dict(text=text, mentions=[mention(text, surface) for surface in surfaces])


def test_query_context_matches_sentence_radius_and_exact_offsets():
    class Tokenizer:
        def encode(self, text):
            assert text.startswith("query: ")
            return SimpleNamespace(ids=[1] * len(text.split()))

    contexts = corpus.QueryContexts(Tokenizer())
    assert contexts(query("Where is Alpha used?", "Alpha")) == ["Alpha. Where is Alpha used?"]
    questions = "First sentence. Second sentence. Where is Alpha used? Following text. Last sentence."
    assert contexts(query(questions, "Alpha")) == ["Alpha. Second sentence. Where is Alpha used? Following text."]
    bad = query(questions, "Alpha")
    bad["mentions"][0]["start"] += 1
    with pytest.raises(ValueError, match="Offsets"):
        contexts(bad)


def test_score_thresholds_top_k_and_ties():
    scores = Similarities(np.array([[.75, .90, .74, .85, .75], [.875, .80, .70, .40, .80]]),
                          np.array([[.85, .50, 1., .75, .65], [.875, .90, 1., 1., .80]]))
    c1 = rank_mentions(scores, LinkingPolicy(.75, .5, 3))
    assert [[i for i, _ in row] for row in c1] == [[0, 3, 1], [0, 1, 4]]
    c2 = rank_mentions(scores, LinkingPolicy(.75, .5, 2, .80))
    assert [[i for i, _ in row] for row in c2] == [[0, 3], [0, 1]]
    assert [i for i, _, _ in merge_mentions(c2)] == [0, 1, 3]
    assert merge_mentions(c2)[0][1] == 1


def test_linear_combination_uses_name_filter_but_no_context_filter():
    scores = Similarities(np.array([[.99, .74]]), np.array([[0., 1.]]))
    assert rank_mentions(scores, LinkingPolicy(.75, .95, 5, .90))[0][0][0] == 0
    assert rank_mentions(scores, LinkingPolicy(.75, .05, 5, .90)) == [[]]
    with pytest.raises(ValueError, match="alpha"):
        LinkingPolicy(.75, float("nan"), 5)


def test_runtime_returns_provenanced_seeds_and_uses_two_vectors_per_mention():
    calls = []

    class Encoder:
        def encode(self, texts):
            calls.append(texts)
            return np.stack([vector(1), vector(0, 1)])

    entities = [
        SimpleNamespace(entity_id=key, document_id=doc, vectors=np.stack([vector(1), vector(0, 1)]))
        for key, doc in (("e:z", "d2"), ("e:a", "d1"))
    ]
    linker = EntityLinkerV2(entities, {}, lambda _: ["Alpha. What about Alpha?"], encoder=Encoder())
    outcome = linker.link(query("What about Alpha?", "Alpha"), LinkingPolicy(.95, .5, 1))
    assert calls == [["Alpha", "Alpha. What about Alpha?"]]
    assert outcome["seeds"] == [dict(node_key=NAMESPACE + ":e:a", entity_id="e:a", document_id="d1",
                                     mention_id="m:11", score=1., match_method="vector")]
    assert outcome["candidates"][0]["s_context"] == 1.
    assert linker.link(query(""), LinkingPolicy(.95, .5, 1))["abstained"]
    assert len(calls) == 1


def test_generation_follows_current_and_accepts_appended_vectors(tmp_path):
    recipe = dict(model=corpus.MODEL, revision=corpus.REVISION, tokenizer_revision=corpus.REVISION,
                  window=corpus.WINDOW, embedding=corpus.RECIPE,
                  model_files_sha256={key: "hash" for key in ("tokenizer.json", "model.safetensors", "config.json")})

    def publish(request, generation, count):
        entities = [Entity.create(
            dict(entity_id=f"e:{i}", document_id=f"d{i}", name=f"Entity {i}", entity_types=["OBJECT"],
                 evidence=[dict(document_id=f"d{i}")]),
            [dict(text="Context.", provenance=[{}])], np.stack([vector(1), vector(0, 1)]), digest(recipe),
        ) for i in range(count)]
        folder = tmp_path / "generations" / request
        (folder / "catalog").mkdir(parents=True)
        source = dict(documents={e.document_id: {} for e in entities},
                      resolution=dict(entities={e.entity_id: [] for e in entities}))
        graph.prepare_inputs(folder, request, lambda: graph.Inputs(
            source, entities, dict(recipe=recipe, research_graph=NAMESPACE)))
        write_checked(folder / "catalog/catalog.json", dict(
            generation=generation, status="complete", fingerprints={e.entity_id: e.fingerprint for e in entities}))
        write_checked(folder / "verification.json", {"status": "PASS"})
        write_checked(tmp_path / "CURRENT.json", dict(request=request, generation=generation, payload_sha256="payload"))
        return folder

    publish("a" * 64, "base", 1)
    assert corpus.load_generation(tmp_path)[1]["entity_count"] == 1
    appended = publish("b" * 64, "delta", 2)
    loaded, snapshot = corpus.load_generation(tmp_path)
    assert (snapshot["generation"], snapshot["entity_count"]) == ("delta", 2)
    assert [e.entity_id for e in loaded.entities] == ["e:0", "e:1"]
    write_checked(appended / "catalog/catalog.json", dict(generation="base", status="complete"))
    with pytest.raises(ValueError, match="published Corpus100"):
        corpus.load_generation(tmp_path)


def test_grid_covers_unique_inclusive_combinations():
    policies = study.grid()
    assert study.WORKERS == 3
    assert len(set(policies)) == 2280
    assert Counter(p.variant for p in policies) == {"C1": 380, "C2": 1900}
    assert {p.threshold_name for p in policies} == {.75, .80, .85, .90, .95}
    assert {p.threshold_score for p in policies} == {None, .75, .80, .85, .90, .95}
    assert {p.alpha for p in policies} == {n / 100 for n in range(5, 100, 5)}
    assert {p.top_k for p in policies} == {5, 10, 15, 20}


def test_questions_require_isolated_texts_and_source_evidence(tmp_path):
    rows = [
        dict(id=split, split=split, question=question, targets=[
            dict(entity_id="e:1", document_id="d1", evidence=dict(revision_id="r1", start=0, end=5, text="Alpha"))
        ]) for split, question in (("calibration", "What is Alpha?"), ("holdout", "Why Alpha?"))
    ]
    docs = {"d1": dict(text="Alpha text.", revision_id="r1")}
    entities = [SimpleNamespace(entity_id="e:1", document_id="d1")]
    path = tmp_path / "questions.jsonl"

    def read():
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return study.read_questions(path, docs, entities)

    assert read()["holdout"][0]["id"] == "holdout"
    rows[1]["targets"][0]["evidence"]["end"] = 6
    with pytest.raises(ValueError, match="Offsets"):
        read()
    rows[1]["question"] = "  what IS   Alpha? "
    with pytest.raises(ValueError, match="duplicate"):
        read()


def test_smoke_resume_after_calibration_with_three_worker_grid(tmp_path):
    out = tmp_path / "run"
    calls, cosine_calls = [], []
    holdout_fails = True
    entities = (SimpleNamespace(entity_id="e:a", document_id="d:a"),
                SimpleNamespace(entity_id="e:b", document_id="d:b"))

    class Analyzer:
        def analyze(self, text):
            calls.append(text)
            if text == "Holdout?":
                assert (out / "selection.json").is_file()
                if holdout_fails:
                    raise KeyboardInterrupt
            return query(text, text[:-1])

    scores = Similarities(np.array([[.90, .80]]), np.array([[.70, .80]]))

    def similarities(analysis):
        cosine_calls.append(analysis["text"])
        return scores

    linker = SimpleNamespace(entities=entities, similarities=similarities)
    splits = {split: [dict(id=split, question=text, targets=[dict(entity_id="e:a", document_id="d:a")])]
              for split, text in (("calibration", "Calibrate?"), ("holdout", "Holdout?"))}
    c1 = LinkingPolicy(.75, .5, 5)
    c2 = LinkingPolicy(.75, .5, 5, .95)
    policies = [c1, c2, LinkingPolicy(.95, .5, 5), LinkingPolicy(.95, .6, 5)]
    manifest = dict(graph=dict(generation="smoke", entity_count=2))
    with pytest.raises(KeyboardInterrupt):
        study.run(out, manifest, splits, linker, Analyzer(), policies, resume=False)
    checkpoints = list((out / "calibration").glob("*.json"))
    assert len(checkpoints) == 3
    sizes = {p: p.stat().st_mtime_ns for p in checkpoints}
    holdout_fails = False
    study.run(out, manifest, splits, linker, Analyzer(), policies, resume=True)
    study.run(out, manifest, splits, linker, Analyzer(), policies, resume=True)
    assert sizes == {p: p.stat().st_mtime_ns for p in checkpoints}
    assert calls == ["Calibrate?", "Holdout?", "Holdout?"]
    assert cosine_calls == ["Calibrate?", "Holdout?"]
    assert load_checked(out / "selection.json")["policies"] == [asdict(c1), asdict(c2)]
    report = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert Counter(row["split"] for row in report["rows"]) == {"calibration": 4, "holdout": 2}
    assert all(row["precision_at_k"] is None for row in report["rows"])
    losses = json.loads((out / "diagnostics.json").read_text(encoding="utf-8"))
    assert losses["holdout_cases"]["questions_with_loss"] == 1
    assert losses["holdout_cases"]["cases"][0]["lost_entities"][0]["s_name"] == .90
    assert (out / "results.csv").is_file()
    assert (out / "REPORT.md").is_file()
    with pytest.raises(ValueError, match="new directory"):
        study.run(out, manifest | {"changed": True}, splits, linker, Analyzer(), policies, resume=True)


def test_metrics_for_complete_labels_and_zero_seeds():
    entities = (SimpleNamespace(entity_id="e:wrong", document_id="d:other"),
                SimpleNamespace(entity_id="e:target", document_id="d:gold"))
    gold = dict(precision_complete=True, targets=[dict(entity_id="e:target", document_id="d:gold")])
    meta = dict(inference_ms=10.)
    scores = Similarities(np.array([[.95, .80]]), np.array([[.95, .80]]))
    row = study.evaluate(LinkingPolicy(.75, .5, 5), [(gold, meta, scores)], entities)
    assert row["entity_recall_at_k"] == row["document_recall_at_k"] == 1
    assert row["mrr_at_k"] == .5
    assert row["precision_at_k"] == .2
    assert row["seeds_total"] == 2
    assert row["latency_p95_ms"] >= 10.
    empty = Similarities(np.empty((0, 2)), np.empty((0, 2)))
    row = study.evaluate(LinkingPolicy(.75, .5, 5), [(gold, meta, empty)], entities)
    assert row["seeds_total"] == row["mrr_at_k"] == row["precision_at_k"] == 0
    assert row["no_seeds_rate"] == 1
