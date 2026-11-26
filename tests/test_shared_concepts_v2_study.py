from itertools import product
from pathlib import Path
import sys

import pytest

from cgraph.shared_concepts_v2 import digest
from test_shared_concepts_v2 import METADATA, entity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import study_shared_concepts_v2 as study
from shared_concepts_v2_pipeline import load_checked, sha
from shared_concepts_v2_study_metrics import score_histogram, shortlist


def ranking_row(point, covered=2):
    return dict(point=point, metrics=dict(covered_entities=covered, covered_disciplines=1, covered_documents=2,
                by_topic={"fixture": dict(entities=covered, total_entities=5)}, memberships=3, concepts=1))


def test_stage_plans_grid_and_rank():
    a = study.stage_a(False)
    assert len(a) == 121
    assert a[0] == (90, 85, 98, 95, 10)
    assert a[-1] == (100, 95, 98, 95, 10)
    ranked_a = [ranking_row(key, (key[0]+key[1]) % 5) for key in a]
    b = study.stage_b(ranked_a, False)
    assert b == study.stage_b(list(reversed(ranked_a)), False)
    assert len(b) == 3 * 99
    for ec in {point[:2] for point in b}:
        assert {p[2:4] for p in b if p[:2] == ec} == set(product(range(92, 101), range(90, 101)))
    c = study.stage_c([ranking_row(key) for key in b], False)
    assert len(c) == 5 * 11
    for pair in {point[:4] for point in c}:
        assert {p[4] for p in c if p[:4] == pair} == set(range(5, 16))
    ranked_c = [ranking_row(key, 3 if key[-1] == 10 else 2) for key in c]
    centers = [tuple(row["point"]) for row in shortlist(ranked_c, 5, 4)]
    d = study.stage_d(ranked_c, False)
    assert d == study.stage_d(ranked_c[::-1], False)
    assert set(centers).issubset(d) and len(d) <= 55
    assert all(any(sum(abs(i-j) for i, j in zip(point, center, strict=True)) <= 1 for center in centers) for point in d)
    assert all(all(v in axis for v, axis in zip(point, study.RANGES, strict=True)) for point in d)


def test_three_workers_resume_and_serial_equivalence(tmp_path):
    rows = [entity(str(i), i * 15) for i in range(5)]
    pairs = [(str(i), str(i+1), .966, .966) for i in range(4)]
    inputs = study.StudyInput(rows, METADATA, dict(key="fixture", edges=pairs, metrics={"guarded_pairs": len(pairs)},
                                                   guarded_cosine=score_histogram([r[2:] for r in pairs])))
    target = tmp_path / "interrupted"
    first = []

    def interrupt(row):
        first.append(tuple(row["point"]))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        study.run_study(inputs, target, "parallel", smoke=True, workers=3, on_saved=interrupt)
    assert not (target / "complete.json").exists()
    completed = study.trial_path(target, first[0]) / "metrics.json"
    preserved = sha(completed)
    resumed = study.run_study(inputs, target, "parallel", smoke=True, workers=3)
    assert sha(completed) == preserved
    serial = study.run_study(inputs, tmp_path / "serial", "serial", smoke=True, workers=1)
    assert resumed["stages"] == serial["stages"]
    assert resumed["shortlist"] == serial["shortlist"]
    earlier = {tuple(p) for stage in ("A", "B", "C") for p in resumed["stages"][stage]}
    assert {tuple(p) for p in resumed["stages"]["D"]} - earlier
    assert {tuple(row["point"]): row["metrics"] for row in resumed["results"]} == {
        tuple(row["point"]): row["metrics"] for row in serial["results"]
    }
    assert len(resumed["results"]) >= 7
    assert all(sum(row["metrics"]["cosine"]["cc"][0]) == sum(row["metrics"]["cc"].get(k, 0)
                                                             for k in ("partial", "disjoint"))
               for row in resumed["results"])
    marker = load_checked(target / "complete.json")
    for path in ("results.csv", "ec_heatmap.png", "cc_heatmaps.png", "delta.png", "REPORT.md"):
        assert sha(target / path) == marker["files_sha256"][path]
    details = load_checked(target / "details/MANIFEST.json")
    assert details["points"][study.point_id(study.BASELINE)]["baseline"]
    assert not list((target / "trials").glob("*/catalog.json.gz"))
    assert digest(study.run_study(inputs, target, "parallel", smoke=True)) == digest(resumed)
