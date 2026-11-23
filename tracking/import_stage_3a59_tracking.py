"""Импорт описательного исторического замера из закреплённого коммита RESEARCH."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from project_tracking.client import record

REPORT_COMMIT = "bf541d34e34c557bd24846648a3c78f3303f78fe"
STAGE = "reports/retrieval/benchmark_3a5_9"
EXPECTED_SUMMARY = "00be2fab47124ea38f7cedeb0c11e02e240493951854326a1c654a08552ad3a0"


def read_git_file(repository: Path, name: str) -> bytes:
    """Читать объект Git, не обращаясь к изменяемой рабочей копии."""
    return subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-C",
            str(repository),
            "show",
            f"{REPORT_COMMIT}:{STAGE}/{name}",
        ],
        check=True,
        capture_output=True,
    ).stdout


def build_event(repository: Path) -> tuple[dict, dict, dict]:
    profile_data = read_git_file(repository, "PROFILE.json")
    manifest_data = read_git_file(repository, "results/confirmation/manifest.json")
    summary_data = read_git_file(repository, "results/confirmation/summary.json")
    manifest = json.loads(manifest_data)
    profile = json.loads(profile_data)
    summary = json.loads(summary_data)
    if hashlib.sha256(summary_data).hexdigest() != EXPECTED_SUMMARY:
        raise ValueError("Контрольная сумма опубликованной сводки не совпадает")
    if manifest["files"]["summary.json"] != EXPECTED_SUMMARY:
        raise ValueError("Контрольная сумма сводки в манифесте не совпадает")
    if any(
        manifest[key] != profile[profile_key]
        for key, profile_key in (
            ("source_git", "base_git"),
            ("snapshot_sha256", "snapshot_sha256"),
            ("graph_fingerprint", "graph_fingerprint"),
        )
    ):
        raise ValueError("Версии в манифесте и профиле не совпадают")
    result = summary["production-fifo-100-0"]["HOLDOUT"]
    config = profile["production"]
    params = {
        "schema_version": 1,
        "git_commit": manifest["source_git"],
        "config_hash": hashlib.sha256(profile_data).hexdigest(),
        "corpus_hash": profile["corpus_sha256"],
        "graph_fingerprint": manifest["graph_fingerprint"],
        "snapshot_hash": manifest["snapshot_sha256"],
        "retrieval_profile": profile["profile_id"],
        "embedding_model": profile["models"]["E5"]["id"],
        "embedding_revision": profile["models"]["E5"]["sha256"]["model.safetensors"],
        "bfs_depth": config["bfs"]["max_hops"],
        "bfs_limit": config["bfs"]["max_candidates"],
        "reranker": config["reranking"],
        "seed": profile["seed"],
        "top_k": config["rerank_top_k"],
        "split": "HOLDOUT",
    }
    metrics = {
        "latency_p50_ms": result["replay_p50"] * 1000,
        "latency_p95_ms": result["replay_p95"] * 1000,
        "peak_rss_bytes": result["rss_high_water_bytes"],
        "questions_count": result["questions"],
    }
    tags = {
        "imported": "true",
        "evaluation_source": "gold",
        "measurement_scope": "snapshot_replay",
        "annotation_status": "unverified",
    }
    return params, metrics, tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("research", type=Path, help="Путь к Git-репозиторию RESEARCH")
    args = parser.parse_args()
    params, metrics, tags = build_event(args.research)
    result = record(
        "retrieval_quality",
        f"{REPORT_COMMIT}:3a59:confirmation:production:HOLDOUT",
        params,
        metrics,
        tags,
    )
    print(f"Результат отправки: {result}")
    return 0 if result == "delivered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
