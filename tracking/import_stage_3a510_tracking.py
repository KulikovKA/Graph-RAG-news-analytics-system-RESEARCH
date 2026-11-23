"""Импорт опубликованных агрегатов смешанного Groq smoke Stage 3A.5.10."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from project_tracking.client import record

REPORT_COMMIT = "85b69302f6d3c0b4a5d3630b46194f7646ddafda"
REPORT_PATH = "reports/retrieval/benchmark_3a5_10/SMOKE_RESULTS.md"
RESULTS_PATH = "reports/retrieval/benchmark_3a5_10/RESULTS.json"
SPEC_PATH = "reports/retrieval/benchmark_3a5_10/SPEC.md"
PROFILE_PATH = "reports/retrieval/benchmark_3a5_9/PROFILE.json"
EXPECTED_HASHES = {
    RESULTS_PATH: "b31431db32cdba1084f69249480c741903ac6501d2b9c8d95ce8bfa0e943cc0e",
    REPORT_PATH: "b058a8d38abab4dc0ce99ada16c9237ba8f4eb2d66675c664a96528d2e177cbc",
    SPEC_PATH: "f37ea80bd9638044a9d1ea330b57431590a70aa8298dd8fbbc58a93accfbf278",
    PROFILE_PATH: "4c4194a58c1d1e14282253d38880e834584ed5a0c1cce4cd36e3ee3724575867",
}


def read_git_file(repository: Path, path: str) -> bytes:
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repository), "show", f"{REPORT_COMMIT}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def build_event(repository: Path) -> tuple[dict, dict, dict]:
    files = {path: read_git_file(repository, path) for path in EXPECTED_HASHES}
    for path, data in files.items():
        if hashlib.sha256(data).hexdigest() != EXPECTED_HASHES[path]:
            raise ValueError("Контрольная сумма опубликованного документа не совпадает")
    report = files[REPORT_PATH].decode("utf-8")
    result = json.loads(files[RESULTS_PATH])
    usage = result["reported_usage"]
    if usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]:
        raise ValueError("Сумма токенов в отчёте не совпадает")
    estimate = re.search(r"\*\*\$(0\.\d+)\*\*", report)
    if estimate is None:
        raise ValueError("В отчёте отсутствует опубликованная оценка стоимости")
    profile = json.loads(files[PROFILE_PATH])
    if result["retrieval_profile"] != profile["profile_id"]:
        raise ValueError("Профиль в отчёте и манифесте не совпадает")
    params = {
        "schema_version": 1,
        "git_commit": REPORT_COMMIT,
        "config_hash": EXPECTED_HASHES[SPEC_PATH],
        "snapshot_hash": profile["snapshot_sha256"],
        "corpus_hash": profile["corpus_sha256"],
        "graph_fingerprint": profile["graph_fingerprint"],
        "retrieval_profile": profile["profile_id"],
        "split": "SMOKE",
    }
    metrics = {
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "completion_attempts": result["completion_attempts"],
        "usage_reported_attempts": result["usage_reported_attempts"],
        "estimated_groq_cost_usd": float(estimate.group(1)),
    }
    tags = {
        "imported": "true",
        "evaluation_source": "operational",
        "annotation_status": "none",
        "measurement_scope": "groq_smoke",
        "unavailable_fields": "groq_cost_usd",
    }
    return params, metrics, tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research", type=Path, help="Путь к Git-репозиторию RESEARCH")
    args = parser.parse_args()
    params, metrics, tags = build_event(args.research)
    result = record(
        "retrieval_quality",
        f"{REPORT_COMMIT}:3a510:groq-smoke",
        params,
        metrics,
        tags,
    )
    print(f"Результат отправки: {result}")
    return 0 if result == "delivered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
