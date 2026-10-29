"""Meaningful boundary/leakage checks; artificial fixtures excluded from DEV metrics."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from cgraph.common import read_json
from cgraph.diagnostic_ner import merge_mentions
from cgraph.labels import gold_entities


def test_union_preserves_unicode_nested_conflicting_kinds_and_originals():
    chunk = dict(chunk_id="c", text="🙂 NO₂ sensor")
    source = [dict(start=2, end=5, surface_text="NO₂", kind="MATERIAL", score=.6, mention_id="old"),
              dict(start=2, end=5, surface_text="NO₂", kind="ANALYTE", score=.7),
              dict(start=2, end=12, surface_text="NO₂ sensor", kind="DEVICE", score=.9)]
    other = [dict(source[0], score=.8)]
    merged = merge_mentions(chunk, [source, other])
    assert len(merged) == 3
    assert {m["kind"] for m in merged} == {"MATERIAL", "ANALYTE", "DEVICE"}
    assert next(m for m in merged if m["kind"] == "MATERIAL")["score"] == .8
    assert source[0]["mention_id"] == "old" and source[0]["score"] == .6
    assert len({m["mention_id"] for m in merged}) == 3


def test_every_bio_class_has_frozen_mapping_without_forced_unsupported_kinds():
    cfg = read_json("config/c_diag_002.json")
    for spec in cfg["scientific_ner"].values():
        classes = {v[2:] for v in spec["id2label"].values() if v != "O"}
        assert classes == set(spec["mapping"])
    assert cfg["scientific_ner"]["chem"]["mapping"] == {"CHEM": "MATERIAL"}
    assert cfg["scientific_ner"]["battery"]["mapping"]["FAILURE_MODE"] is None


def test_gold_projection_cannot_export_relations_qualifiers_or_gold_roles():
    mention = dict(chunk_id="c", mention_id="m", start=0, end=1, surface_text="X", kind="MATERIAL",
                   contextual_role="GOLD_ANSWER", qualifiers=["GOLD_ANSWER"], relation="GOLD_ANSWER")
    projected = gold_entities(dict(mentions=[mention], assertions=["GOLD_ANSWER"]), dict(chunk_id="c"))
    assert set(projected[0]) == {"mention_id", "start", "end", "surface_text", "kind", "contextual_role", "score"}
    assert projected[0]["contextual_role"] == "UNKNOWN" and projected[0]["score"] is None
    assert "GOLD_ANSWER" not in str(projected)


@pytest.mark.parametrize("gold_mode,path", [(False, "data/labels/D001.jsonl"),
    (True, "data/labels/D001.jsonl"), (False, "artifacts/c_diag_002/diagnostic_entities.json"),
    (True, "data/HOLDOUT/D999.json"), (False, "data/CALIBRATION/D999.json")])
def test_guard_rejects_gold_relations_and_closed_splits_before_open(gold_mode, path):
    script = f'''import runpy
ns = runpy.run_path("scripts/run_c_diag_002.py", run_name="test_guard")
ns["install_guard"]({gold_mode!r})
try:
    open({path!r})
except PermissionError:
    print("DENIED")
else:
    raise AssertionError("Leakage guard did not reject access")
'''
    output = subprocess.check_output([sys.executable, "-c", script], text=True)
    assert "DENIED" in output


def test_new_inference_has_no_training_or_evaluation_calls():
    for filename in ("src/cgraph/diagnostic_ner.py", "scripts/run_c_diag_002.py"):
        tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                method = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert method not in {"fit", "fit_transform", "train", "backward", "step", "load_labels", "evaluate"}
