"""Проверка происхождения агрегатов смешанного smoke."""

import hashlib
import json
from pathlib import Path

import import_stage_3a510_tracking as historical
import pytest
from project_tracking.schema import validate


def fixture_files():
    profile = {
        "snapshot_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "graph_fingerprint": "c" * 64,
        "profile_id": "production-C",
    }
    report = (
        "Всего **10 попыток Chat Completions**, включая timeout. "
        "Семь ответов вернули usage **9607 input, 762 output, 10369 total**. "
        "Оценка **$0.00189825**."
    )
    files = {
        historical.PROFILE_PATH: json.dumps(profile).encode(),
        historical.SPEC_PATH: b"spec fixture",
        historical.REPORT_PATH: report.encode("utf-8"),
        historical.RESULTS_PATH: json.dumps(
            {
                "retrieval_profile": "production-C",
                "completion_attempts": 10,
                "usage_reported_attempts": 7,
                "reported_usage": {
                    "input_tokens": 9607,
                    "output_tokens": 762,
                    "total_tokens": 10369,
                },
            }
        ).encode(),
    }
    return files, {path: hashlib.sha256(data).hexdigest() for path, data in files.items()}


def test_imported_usage(monkeypatch):
    files, hashes = fixture_files()
    monkeypatch.setattr(historical, "EXPECTED_HASHES", hashes)
    monkeypatch.setattr(historical, "read_git_file", lambda _repository, path: files[path])
    params, metrics, tags = historical.build_event(Path("."))
    validate("retrieval_quality", params, metrics, tags)
    assert params["split"] == "SMOKE"
    assert metrics == {
        "input_tokens": 9607,
        "output_tokens": 762,
        "completion_attempts": 10,
        "usage_reported_attempts": 7,
        "estimated_groq_cost_usd": 0.00189825,
    }
    assert tags["imported"] == "true"
    assert tags["unavailable_fields"] == "groq_cost_usd"


def test_tampering_is_rejected(monkeypatch):
    files, hashes = fixture_files()
    files[historical.REPORT_PATH] += b" "
    monkeypatch.setattr(historical, "EXPECTED_HASHES", hashes)
    monkeypatch.setattr(historical, "read_git_file", lambda _repository, path: files[path])
    with pytest.raises(ValueError, match="Контрольная сумма"):
        historical.build_event(Path("."))
