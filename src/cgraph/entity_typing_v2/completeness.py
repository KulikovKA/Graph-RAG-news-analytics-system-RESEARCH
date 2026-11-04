"""Offline, per-ID accounting; coverage with error rows is not a successful run."""
from collections import Counter

from .dataset import load_frozen
from .runner import MODE_FILES
from .storage import file_hash, now, read_jsonl, write_csv, write_json


def check_completeness(output, validated=None):
    from .analysis import predictions, route
    manifest, candidates, ontology = load_frozen(output)
    ids = set(manifest["frozen"]["candidate_ids"])
    errors = []
    def safe_rows(filename):
        try:
            return read_jsonl(output / filename)
        except (ValueError, OSError) as exc:
            errors.append(f"Unreadable {filename}: {exc}")
            return []
    if manifest["frozen"]["dataset"] == "full" and len(ids) != 1211:
        errors.append("Full dataset must contain exactly 1211 unique candidate IDs")
    if validated is None:
        try:
            validated = predictions(output, manifest, candidates, ontology)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            validated = {}
            errors.append("Saved prediction validation failed: " + str(exc))
    policy = manifest.get("hybrid_policy", {})
    thresholds = policy.get("routing_thresholds", [.5, .55, .6, .65])
    variants = {f"{mode}:threshold={threshold:.2f}:risk={risk}": (mode, threshold, risk)
                for mode in MODE_FILES for threshold in thresholds for risk in (False, True)}
    if set(policy.get("expected_variants", variants)) != set(variants):
        errors.append("Hybrid policy variant list mismatch")
    for name, expected_hash in policy.get("source_hashes", {}).items():
        if not (output / name).exists() or file_hash(output / name) != expected_hash:
            errors.append("Stale hybrid source: " + name)
    matrix, methods = [], {}

    def account(name, rows, accepted):
        counts = Counter(r.get("candidate_id") for r in rows)
        by_id = {r.get("candidate_id"): r for r in rows}
        duplicates = sorted(str(cid) for cid, count in counts.items() if count > 1)
        unexpected = sorted(str(cid) for cid in counts.keys() - ids)
        missing, invalid, complete, pending = [], [], 0, 0
        for cid in sorted(ids):
            row = by_id.get(cid)
            if row is None:
                state, reason = "MISSING", "No result row"
                missing.append(cid)
            elif counts[cid] != 1:
                state, reason = "ERROR", "Duplicate candidate ID"
                invalid.append(cid)
            elif cid in accepted and row.get("processing_status", "COMPLETE") == "COMPLETE":
                state, reason = "COMPLETE", ""
                complete += 1
            elif (row.get("processing_status", row.get("typing_status")) in ("NOT_PROCESSED", "ERROR") and
                  row.get("entity_type") is None):
                state, reason = "NOT_PROCESSED", row.get("error_reason", "No validated decision")
                pending += 1
            else:
                state, reason = "ERROR", "Unvalidated result or inconsistent status/type"
                invalid.append(cid)
            matrix.append({"candidate_id": cid, "method": name, "status": state, "error_reason": reason})
        covered = not missing and not unexpected and not duplicates and not invalid
        methods[name] = {"rows": len(rows), "complete": complete, "not_processed": pending,
                         "missing_candidate_ids": missing, "invalid_candidate_ids": invalid,
                         "duplicate_candidate_ids": duplicates, "unexpected_candidate_ids": unexpected,
                         "all_ids_accounted_for": covered, "all_predictions_complete": covered and complete == len(ids)}

    for name, filename in {"T1": "T1_gliner.jsonl", "T21": MODE_FILES["no-thinking"], "T22": MODE_FILES["thinking"]}.items():
        account(name, safe_rows(filename), validated.get(name, {}))
    hybrid_rows = safe_rows("T3_hybrid.jsonl")
    if hybrid_rows and not policy:
        errors.append("Hybrid results lack policy/source provenance")
    unknown_variants = sorted({str(r.get("variant")) for r in hybrid_rows if r.get("variant") not in variants})
    if unknown_variants:
        errors.append("Unexpected hybrid variants: " + ", ".join(unknown_variants))
    for variant, (mode, threshold, risk) in variants.items():
        rows = [r for r in hybrid_rows if r.get("variant") == variant]
        qwen = validated.get("T22" if mode == "thinking" else "T21", {})
        accepted = set()
        for row in rows:
            cid = row.get("candidate_id")
            g = validated.get("T1", {}).get(cid)
            if not g:
                continue
            source, reason = route(g, threshold, risk, policy.get("risk_margin", .15))
            expected = g if source == "gliner" else qwen.get(cid)
            if (expected and row.get("processing_status", "COMPLETE") == "COMPLETE" and
                    row.get("route") == source and row.get("route_reason") == reason and
                    row.get("entity_type") == expected["entity_type"] and row.get("typing_status") == expected["typing_status"]):
                accepted.add(cid)
        account("T3:" + variant, rows, accepted)
    complete = not errors and all(m["all_predictions_complete"] for m in methods.values())
    report = {"checked_at": now(), "dataset": manifest["frozen"]["dataset"], "candidate_count": len(ids),
              "status": "COMPLETE" if complete else "PARTIAL", "success": complete,
              "all_ids_accounted_for": not errors and all(m["all_ids_accounted_for"] for m in methods.values()),
              "methods": methods, "errors": errors,
              "note": "Explicit NOT_PROCESSED rows preserve coverage but do not make an experiment successful"}
    write_json(output / "completeness.json", report)
    write_csv(output / "completeness.csv", matrix, ["candidate_id", "method", "status", "error_reason"])
    manifest["completion"] = report
    write_json(output / "manifest.json", manifest)
    return report
