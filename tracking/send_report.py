"""Передать доверенный агрегированный отчёт в общий MLflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from project_tracking.client import record
from project_tracking.schema import validate

LOGGER = logging.getLogger(__name__)
EXPERIMENTS = {"retrieval_quality", "llm_as_judge"}
REPORT_FIELDS = {"external_id", "params", "metrics", "tags", "status", "parent_run_id"}


def send(experiment: str, report: object) -> str:
    """Записать проверенное событие, сохранив независимость исследовательского кода."""
    try:
        if experiment not in EXPERIMENTS or not isinstance(report, dict):
            raise ValueError("invalid experiment or report")
        if (
            report.keys() - REPORT_FIELDS
            or not {"external_id", "params", "metrics", "tags"} <= report.keys()
        ):
            raise ValueError("invalid report fields")
        params = report["params"]
        metrics = report["metrics"]
        tags = report["tags"]
        validate(experiment, params, metrics, tags)
        return record(
            experiment,
            report["external_id"],
            params,
            metrics,
            tags,
            status=report.get("status", "FINISHED"),
            parent_run_id=report.get("parent_run_id"),
        )
    except Exception:
        LOGGER.warning("Не удалось отправить агрегированный отчёт в tracking")
        return "error"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        LOGGER.warning("Не удалось прочитать отчёт")
        return 1
    result = send(args.experiment, report)
    print(f"Результат отправки: {result}")
    return 0 if result == "delivered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
