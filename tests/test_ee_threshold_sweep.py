from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ee_threshold_sweep as sweep
from contextual_shared_concepts import pair_guard_codes, pair_indices
from ee_threshold_sweep_report import configuration, neighborhood


def test_exact_grid():
    grid = sweep.grid(False)
    assert len(grid) == len(set(grid)) == 528
    assert set(w for w, _, _ in grid) == {"S0", "S1", "S2"}
    assert all(sum(w == window for w, _, _ in grid) == 176 for window in sweep.WINDOWS)
    assert grid[0] == ("S0", .85, .75)
    assert grid[-1] == ("S2", .95, .90)


def test_and_guards_and_coverage():
    entities = [dict(entity_id=str(i), name=str(i), document_id=doc, topic=topic,
                     entity_types=[kind], expansions={}, evidence=[])
                for i, (doc, topic, kind) in enumerate(
                    [("a", "X", "METHOD"), ("a", "X", "METHOD"), ("b", "X", "METHOD"),
                     ("c", "Y", "RESULT"), ("d", "Y", "METHOD")])]
    left, right = pair_indices([e["document_id"] for e in entities])
    positions = list(zip(left, right, strict=True))
    names, contexts = np.full(len(left), .84), np.full(len(left), .74)
    for pair, values in {(0, 2): (.85, .75), (0, 3): (.90, .80), (0, 4): (.85, .74),
                          (1, 2): (.84, .90), (2, 4): (.86, .76)}.items():
        names[positions.index(pair)], contexts[positions.index(pair)] = values
    domain = names >= .85
    guard = pair_guard_codes(entities, left, right, domain)
    a, b = left[domain], right[domain]
    measure = sweep.previous.document_metrics(entities, a, b)
    old = dict(n="1", candidate_id="case", kind="entity_entity", score=".90", reason="diagnostic")
    cases = [(positions.index((0, 3)), 0, 3, old, None, 1)]
    sampler = SimpleNamespace(process=psutil.Process(), peak=1)
    out = configuration("S0", .85, .75, dict(name=names, S0=contexts), names[domain], contexts[domain],
                        a, b, guard, measure, cases, entities, sampler)
    metrics = out["metrics"]
    assert (metrics["pure_pairs"], metrics["guarded_pairs"]) == (3, 2)
    assert metrics["pure_within_topic_pairs"] == metrics["guarded_within_topic_pairs"] == 1
    assert (metrics["pure_cross_topic_pairs"], metrics["guarded_cross_topic_pairs"]) == (2, 1)
    assert metrics["guarded_entity_share"] == .6
    assert metrics["guarded_document_share"] == .75
    assert metrics["guarded_topics"]["Y"]["document_share"] == .5
    assert out["neighborhoods"][1]["degree_histogram"] == {"0": 2, "1": 2, "2": 1}
    assert out["neighborhoods"][1]["closed_neighborhood_histogram"] == {"1": 2, "2": 2, "3": 1}
    assert out["neighborhoods"][1]["isolated_entities"] == 2
    assert out["diagnostics"][0]["pure_pass"]
    assert not out["diagnostics"][0]["guarded_pass"]
    assert neighborhood(np.zeros(5, dtype=int))["closed_neighborhood_histogram"] == {"1": 5}


def test_configuration_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "configuration.json"
    result = neighborhood(np.array([2, 10, 11]))
    saved = sweep.complete(path, "run-id", (), neighborhoods=[result])
    assert sweep.checked_output(path, "run-id", ()) == saved


def test_spawn_workers_score_and_restore_separate_checkpoints(tmp_path):
    vectors = np.eye(2, sweep.RECIPE["dimension"], dtype=np.float64)
    for window in sweep.WINDOWS:
        name = f"{window}_vectors.npy"
        sweep.atomic_array(tmp_path / name, vectors)
        sweep.complete(tmp_path / f"{window}_vectors.json", "run-id", (name,),
                       usage={}, vector_seconds=0, peak_rss_bytes=0)
    data = dict(entities=[dict(document_id="a"), dict(document_id="b")],
                documents={}, counts=dict(cross_document_pairs=1))
    args = SimpleNamespace(model=tmp_path, smoke=False, force_recompute=True)
    results = sweep.prepare_windows(tmp_path, "run-id", None, data, args, 1)
    for window, result in results.items():
        score = f"{window}_scores.npy"
        assert sweep.array_file(tmp_path / score, (1,), np.float64).tolist() == [0.]
        restored = sweep.prepare(window, tmp_path, "run-id", None, data, tmp_path, 1, True)
        assert result == restored
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        sweep.prepare("S0", tmp_path, "wrong-run", None, data, tmp_path, 1, True)
    (tmp_path / "S0_scores.npy").write_bytes(b"corrupt")
    repaired = sweep.prepare("S0", tmp_path, "run-id", None, data, tmp_path, 1, True)
    assert repaired["files"] == results["S0"]["files"]
