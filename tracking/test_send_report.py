"""Контракт исследовательского адаптера для агрегированных отчётов."""

import send_report


def report():
    return {
        "external_id": "fixture-1",
        "params": {
            "git_commit": "a" * 40,
            "config_hash": "b" * 64,
            "snapshot_hash": "c" * 64,
            "retrieval_profile": "fixture",
            "split": "DEV",
        },
        "metrics": {"questions_count": 2},
        "tags": {"evaluation_source": "fixture", "annotation_status": "unverified"},
    }


def test_retrieval_delivery(monkeypatch):
    calls = []
    monkeypatch.setattr(
        send_report, "record", lambda *args, **kwargs: calls.append(args) or "queued"
    )
    assert send_report.send("retrieval_quality", report()) == "queued"
    assert calls[0][0] == "retrieval_quality"


def test_private_payload_and_unverified_gold_are_rejected(monkeypatch):
    monkeypatch.setattr(send_report, "record", lambda *args, **kwargs: "delivered")
    value = report()
    value["params"]["raw_query"] = "private"
    assert send_report.send("retrieval_quality", value) == "error"
    value = report()
    value["metrics"]["mrr"] = 0.9
    assert send_report.send("retrieval_quality", value) == "error"


def test_judge_mock_and_tracking_error(monkeypatch):
    value = {
        "external_id": "judge-fixture",
        "params": {
            "judge_model": "mock-judge",
            "generator_model": "mock-generator",
            "rubric_version": "v1",
            "prompt_version": "v1",
            "dataset_hash": "d" * 64,
            "split": "DEV",
        },
        "metrics": {"questions_count": 1, "groundedness": 0.8},
        "tags": {"evaluation_source": "fixture", "annotation_status": "unverified"},
    }
    monkeypatch.setattr(send_report, "record", lambda *args, **kwargs: "delivered")
    assert send_report.send("llm_as_judge", value) == "delivered"
    monkeypatch.setattr(send_report, "record", lambda *args, **kwargs: 1 / 0)
    assert send_report.send("llm_as_judge", value) == "error"
