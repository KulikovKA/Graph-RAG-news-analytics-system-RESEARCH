"""Проверка привязки исторического импорта к опубликованным данным."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from project_tracking.schema import validate

SCRIPT = Path(__file__).with_name("import_stage_3a59_tracking.py")
SPEC = importlib.util.spec_from_file_location("historical_tracking", SCRIPT)
assert SPEC and SPEC.loader
historical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(historical)


def fixture_files():
    sha = "a" * 64
    commit = "b" * 40
    profile = {
        "base_git": commit,
        "snapshot_sha256": sha,
        "graph_fingerprint": sha,
        "corpus_sha256": sha,
        "profile_id": "profile",
        "seed": 42,
        "models": {"E5": {"id": "intfloat/e5-small-v2", "sha256": {"model.safetensors": sha}}},
        "production": {
            "bfs": {"max_hops": 5, "max_candidates": 100},
            "rerank_top_k": 5,
            "reranking": "C",
        },
    }
    summary = {
        "production-fifo-100-0": {
            "HOLDOUT": {
                "replay_p50": 0.071,
                "replay_p95": 0.67,
                "rss_high_water_bytes": 423268352,
                "questions": 10,
            }
        }
    }
    encoded = json.dumps(summary).encode()
    summary_hash = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "source_git": commit,
        "snapshot_sha256": sha,
        "graph_fingerprint": sha,
        "files": {"summary.json": summary_hash},
    }
    files = {
        "PROFILE.json": json.dumps(profile).encode(),
        "results/confirmation/manifest.json": json.dumps(manifest).encode(),
        "results/confirmation/summary.json": encoded,
    }
    return files, summary_hash


def test_historical_metrics_and_split(monkeypatch):
    files, expected = fixture_files()
    monkeypatch.setattr(historical, "EXPECTED_SUMMARY", expected)
    monkeypatch.setattr(historical, "read_git_file", lambda _repository, name: files[name])
    params, metrics, tags = historical.build_event(Path("."))
    validate("retrieval_quality", params, metrics, tags)
    assert params["split"] == "HOLDOUT"
    assert metrics["latency_p50_ms"] == 71
    assert metrics["questions_count"] == 10
    assert tags["imported"] == "true"
    assert "mrr" not in metrics


def test_modified_summary_rejected(monkeypatch):
    files, expected = fixture_files()
    files["results/confirmation/summary.json"] += b" "
    monkeypatch.setattr(historical, "EXPECTED_SUMMARY", expected)
    monkeypatch.setattr(historical, "read_git_file", lambda _repository, name: files[name])
    with pytest.raises(ValueError, match="сумма"):
        historical.build_event(Path("."))
