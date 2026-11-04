"""Offline routing, descriptive comparisons, runtime accounting, later human evaluation."""
from collections import Counter, defaultdict
from itertools import combinations

from tqdm import tqdm

from .dataset import gliner_rows, load_frozen
from .prompting import batch_record, plan_batches
from .runner import MODE_FILES, _recover
from .storage import (atomic_text, digest, now, read_json, read_jsonl,
                      write_csv, write_json, write_jsonl)


def predictions(output, manifest, candidates, ontology):
    ids = {c["candidate_id"] for c in candidates}
    result = {}
    t1 = read_jsonl(output / "T1_gliner.jsonl")
    if t1:
        if t1 != gliner_rows(candidates):
            raise ValueError("T1 differs from frozen saved hints")
        result["T1"] = {r["candidate_id"]: r for r in t1}
    for mode, filename in MODE_FILES.items():
        rows = read_jsonl(output / filename)
        if not rows:
            continue
        if len(rows) != len(ids) or {r["candidate_id"] for r in rows} != ids:
            raise ValueError("Qwen output has duplicate, missing, or unexpected IDs")
        if not manifest.get("runtime"):
            raise ValueError("Qwen output lacks frozen runtime provenance")
        run_digest = digest([manifest["frozen"], manifest["runtime"]])
        completed, _ = _recover(output, mode, run_digest, candidates, ontology["entity_types"])
        result["T22" if mode == "thinking" else "T21"] = completed
    return result


def route(gliner, threshold, risk_aware=False, risk_margin=.15):
    margin = gliner["score_margin"]
    if risk_aware and margin is not None and margin < risk_margin:
        return "qwen", "close_alternative_hints"
    if gliner["top1_score"] >= threshold:
        return "gliner", "top1_score_at_or_above_routing_threshold"
    return "qwen", "top1_score_below_routing_threshold"


def _planning_config(config, runtime):
    """Repack with the exact batching/output settings of the frozen full-Qwen plan."""
    result = dict(config)
    for key in ("batch_size", "output_tokens_per_candidate", "thinking_reserve_tokens",
                "prompt_overhead_tokens", "num_predict_thinking"):
        result[key] = runtime[key]
    result["num_predict"] = runtime["options"]["num_predict"]
    return result


def hybrid(output, config):
    manifest, candidates, ontology = load_frozen(output)
    methods = predictions(output, manifest, candidates, ontology)
    if not all(method in methods for method in ("T1", "T21", "T22")):
        raise ValueError("Hybrid requires saved T1, T21 and T22 (partial Qwen rows remain NOT_PROCESSED)")
    planning = _planning_config(config, manifest["runtime"])
    ctx = manifest["runtime"]["num_ctx"]
    baseline = plan_batches(candidates, ontology, planning, ctx)
    full_input = sum(batch_record(b, ontology, planning)["input_token_upper_bound"] for b in baseline)
    results, estimates = [], []
    variants = [(mode, threshold, risk) for mode in MODE_FILES for threshold in config["routing_thresholds"] for risk in (False, True)]
    for mode, threshold, risk in tqdm(variants, desc="Offline hybrid", unit="policy"):
        qwen = methods["T22" if mode == "thinking" else "T21"]
        variant = f"{mode}:threshold={threshold:.2f}:risk={risk}"
        routed = []
        for c in candidates:
            cid = c["candidate_id"]
            g, q = methods["T1"][cid], qwen.get(cid)
            source, reason = route(g, threshold, risk, config["risk_margin"])
            if source == "qwen":
                routed.append(c)
            decision = g if source == "gliner" else q
            results.append({"variant": variant, "candidate_id": cid, "mode": mode,
                            "routing_threshold": threshold, "detection_threshold": .35,
                            "risk_aware": risk, "risk_margin": config["risk_margin"],
                            "route": source, "route_reason": reason,
                            "gliner_entity_type": g["entity_type"], "gliner_score": g["top1_score"],
                            "gliner_margin": g["score_margin"], "qwen_entity_type": q["entity_type"] if q else None,
                            "qwen_typing_status": q["typing_status"] if q else "NOT_PROCESSED",
                            "entity_type": decision["entity_type"] if decision else None,
                            "typing_status": decision["typing_status"] if decision else "NOT_PROCESSED"})
        repacked = plan_batches(routed, ontology, planning, ctx) if routed else []
        hybrid_input = sum(batch_record(b, ontology, planning)["input_token_upper_bound"] for b in repacked)
        estimates.append({"variant": variant, "routed_candidates": len(routed),
                          "baseline_calls": len(baseline), "repacked_calls": len(repacked),
                          "potential_calls_saved_percent": 100 * (1 - len(repacked) / len(baseline)),
                          "baseline_input_token_upper_bound": full_input,
                          "repacked_input_token_upper_bound": hybrid_input,
                          "potential_input_tokens_saved_percent": 100 * (1 - hybrid_input / full_input),
                          "estimate_kind": "planned, repacked document batches, conservative byte token bounds",
                          "measured_hybrid_inference_latency_seconds": None})
    write_jsonl(output / "T3_hybrid.jsonl", results)
    write_json(output / "T3_plan_estimates.json", estimates)
    return results


def _ratio(n, d):
    return n / d if d else None


def comparison(left, right):
    common = sorted(left.keys() & right.keys())
    conflicts = Counter((left[c]["entity_type"], right[c]["entity_type"]) for c in common
                        if left[c]["entity_type"] != right[c]["entity_type"])
    differences = sum(conflicts.values())
    return {"common_valid_candidates": len(common), "same_type_count": len(common) - differences,
            "agreement": _ratio(len(common) - differences, len(common)), "disagreement": _ratio(differences, len(common)),
            "conflicting_types": {a + " -> " + b: n for (a, b), n in sorted(conflicts.items())}}


def runtime_metrics(output):
    metrics = {}
    for mode in MODE_FILES:
        records = [read_json(p) for p in sorted((output / "attempts" / mode).glob("*.json"))]
        sessions = [read_json(p) for p in (output / "sessions").glob("*.json") if read_json(p)["mode"] == mode]
        responses = [r["response"] for r in records if r.get("response")]
        inputs = sum(r.get("prompt_eval_count", 0) for r in responses)
        outputs = sum(r.get("eval_count", 0) for r in responses)
        prompt_s = sum(r.get("prompt_eval_duration", 0) for r in responses) / 1e9
        decode_s = sum(r.get("eval_duration", 0) for r in responses) / 1e9
        wall = sum(r.get("wall_seconds", 0) for r in records)
        successful = {cid for r in records for cid in r.get("parsed", {})}
        metrics[mode] = {"attempt_count": len(records), "retries": sum(r["retry_depth"] > 0 for r in records),
                         "transport_error_count": sum(any(e["kind"] == "transport" for e in r.get("errors", [])) for r in records),
                         "input_tokens": inputs, "output_tokens_including_runtime_thinking": outputs,
                         "thinking_tokens": sum(r["thinking_token_count"] for r in records if r.get("thinking_token_count") is not None)
                            if records and all(r.get("thinking_token_count") is not None for r in records if r.get("response")) and responses else None,
                         "prompt_eval_seconds": prompt_s, "decode_seconds": decode_s,
                         "actual_inference_seconds": prompt_s + decode_s,
                         "chat_model_loading_seconds": sum(r.get("load_duration", 0) for r in responses) / 1e9,
                         "preflight_loading_seconds": sum(s["allocation"]["load_only_response"].get("load_duration", 0) for s in sessions) / 1e9,
                         "input_tokens_per_second": _ratio(inputs, prompt_s), "output_tokens_per_second": _ratio(outputs, decode_s),
                         "attempt_wall_seconds": wall, "mean_wall_seconds_per_completed_candidate": _ratio(wall, len(successful)),
                         "batch_latency_seconds": [r.get("wall_seconds") for r in records],
                         "client_peak_rss_bytes": max((s["memory"]["client_peak_rss_bytes"] for s in sessions), default=None),
                         "ollama_processes_peak_rss_bytes": max((s["memory"]["ollama_processes_peak_rss_bytes"] for s in sessions), default=None),
                         "active_qwen_session_wall_seconds": sum(s["wall_seconds"] for s in sessions),
                         "memory_note": "sampled process RSS; Ollama subprocess sums may count shared pages more than once"}
    return metrics


def summarize(output):
    manifest, candidates, ontology = load_frozen(output)
    methods = predictions(output, manifest, candidates, ontology)
    all_ids = {c["candidate_id"] for c in candidates}
    labels = list(ontology["entity_types"]) + ["UNKNOWN", "NON_ENTITY"]
    methods = {name: methods.get(name, {}) for name in ("T1", "T21", "T22")}
    metrics = {"created_at": now(), "candidate_count": len(candidates),
               "evaluation_policy": "descriptive agreement only; no accuracy/precision/recall/F1 without independent human labels",
               "methods": {}, "comparisons": {}, "hybrid": {}, "runtime": runtime_metrics(output)}
    for name, rows in methods.items():
        counts = Counter(r["entity_type"] for r in rows.values())
        metrics["methods"][name] = {"complete": len(rows), "missing_candidate_ids": sorted(all_ids - rows.keys()),
                                    "distribution": {label: counts[label] for label in labels}}
    disagreements = []
    by_id = {c["candidate_id"]: c for c in candidates}
    for a, b in combinations(methods, 2):
        metrics["comparisons"][a + "_vs_" + b] = comparison(methods[a], methods[b])
        for cid in sorted(methods[a].keys() & methods[b].keys()):
            if methods[a][cid]["entity_type"] != methods[b][cid]["entity_type"]:
                c = by_id[cid]
                disagreements.append({"candidate_id": cid, "document_id": c["document_id"], "surface_text": c["surface_text"],
                                      "context": c["context"], "method_a": a, "method_b": b,
                                      "type_a": methods[a][cid]["entity_type"], "type_b": methods[b][cid]["entity_type"]})
    groups = defaultdict(list)
    for row in read_jsonl(output / "T3_hybrid.jsonl"):
        groups[row["variant"]].append(row)
    estimates = {e["variant"]: e for e in read_json(output / "T3_plan_estimates.json")} if (output / "T3_plan_estimates.json").exists() else {}
    for name, rows in groups.items():
        if len(rows) != len(all_ids) or {r["candidate_id"] for r in rows} != all_ids:
            raise ValueError("Hybrid candidate IDs do not match frozen manifest")
        valid = {r["candidate_id"]: r for r in rows if r["entity_type"] is not None}
        counts = Counter(r["entity_type"] for r in valid.values())
        qwen_method = "T22" if rows[0]["mode"] == "thinking" else "T21"
        metrics["hybrid"][name] = {**estimates.get(name, {}),
            "accepted_gliner": sum(r["route"] == "gliner" for r in rows),
            "routed_qwen": sum(r["route"] == "qwen" for r in rows),
            "missing_candidate_ids": sorted(all_ids - valid.keys()),
            "replaced_types": sum(r["entity_type"] is not None and r["entity_type"] != r["gliner_entity_type"] for r in rows),
            "distribution": {label: counts[label] for label in labels},
            "vs_gliner": comparison(valid, methods["T1"]), "vs_qwen": comparison(valid, methods[qwen_method])}
    failures = read_jsonl(output / "failure_log.jsonl")
    metrics["error_counts"] = dict(Counter(e["kind"] for e in failures))
    metrics["json_error_count"] = sum(e["kind"] in ("invalid_json", "json_schema", "record_schema") for e in failures)
    sessions = [read_json(p) for p in (output / "sessions").glob("*.json")]
    commands = [read_json(p) for p in (output / "command_sessions").glob("*.json")]
    metrics["total_active_experiment_wall_seconds"] = sum(s["wall_seconds"] for s in commands or sessions)
    metrics["wall_time_source"] = "CLI command sessions" if commands else "direct Qwen sessions"
    metrics["wall_time_note"] = "sum of measured command sessions; excludes pauses between launches; offline hybrid latency is not inference latency"
    write_json(output / "metrics.json", metrics)
    flat = []
    def flatten(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(prefix + "." + key if prefix else key, item)
        else:
            import json
            flat.append({"metric": prefix, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value})
    flatten("", metrics)
    write_csv(output / "metrics.csv", flat, ["metric", "value"])
    write_csv(output / "disagreements.csv", disagreements,
              ["candidate_id", "document_id", "surface_text", "context", "method_a", "method_b", "type_a", "type_b"])
    text = ["# ENTITY-TYPING-V2-001", "", f"Frozen dataset: {manifest['frozen']['dataset']}; candidates: {len(candidates)}.",
            "", "Agreement is descriptive and does not establish correctness. Qwen confidence estimates are subjective; GLiNER scores are used only for routing.",
            "", "| Method | Complete | Missing |", "|---|---:|---:|"]
    text.extend(f"| {name} | {len(rows)} | {len(all_ids - rows.keys())} |" for name, rows in methods.items())
    text.extend(["", "Comparisons (denominator: common valid candidate IDs):", ""])
    for name, value in metrics["comparisons"].items():
        text.append(f"- {name}: agreement={value['agreement']}, n={value['common_valid_candidates']}.")
    text.extend(["", f"Hybrid policies saved: {len(groups)}. Savings are planned estimates from repacked batches and byte token bounds, not measured hybrid latency.",
                 "", "Runtime counts include all attempts/retries. Separate thinking token counts may be unavailable. See metrics.json for timings, missing IDs, conflicts, distributions and RSS.",
                 "", "Independent human labels are required for precision, recall and F1; use the evaluate command after annotation."])
    atomic_text(output / "experiment_summary.md", "\n".join(text) + "\n")
    return metrics


def evaluate(output, gold_path):
    """Candidate typing evaluation on explicitly human-reviewed IDs, not NER recall."""
    manifest, candidates, ontology = load_frozen(output)
    rows = read_jsonl(gold_path)
    ids = {c["candidate_id"] for c in candidates}
    labels = list(ontology["entity_types"])
    if not rows or len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Gold must be nonempty with unique candidate IDs")
    for row in rows:
        if (row["candidate_id"] not in ids or row.get("review_status") != "HUMAN_REVIEWED" or
                row.get("entity_type") not in labels + ["UNKNOWN", "NON_ENTITY"]):
            raise ValueError("Gold requires frozen IDs, allowed types and review_status=HUMAN_REVIEWED")
    gold = {r["candidate_id"]: r["entity_type"] for r in rows}
    methods = predictions(output, manifest, candidates, ontology)
    hybrids = defaultdict(dict)
    for row in read_jsonl(output / "T3_hybrid.jsonl"):
        if row["entity_type"] is not None:
            hybrids["T3:" + row["variant"]][row["candidate_id"]] = row
    methods.update(hybrids)
    result = {"gold_sha256": digest(rows), "reviewed_candidates": len(gold),
              "scope": "typing on reviewed frozen candidates only; not corpus mention detection recall", "methods": {}}
    for name, predicted in methods.items():
        common = set(gold) & predicted.keys()
        per_type = {}
        for label in labels:
            tp = sum(gold[c] == label and predicted[c]["entity_type"] == label for c in common)
            fp = sum(gold[c] != label and predicted[c]["entity_type"] == label for c in common)
            # Abstention/NOT_PROCESSED is a false negative for a reviewed positive.
            fn = sum(gold[c] == label and (c not in predicted or predicted[c]["entity_type"] != label) for c in gold)
            per_type[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": _ratio(tp, tp + fp),
                               "recall": _ratio(tp, tp + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}
        tp, fp, fn = (sum(p[k] for p in per_type.values()) for k in ("tp", "fp", "fn"))
        result["methods"][name] = {"complete_reviewed": len(common), "missing_reviewed": sorted(set(gold) - common),
                                   "per_type": per_type, "micro_precision": _ratio(tp, tp + fp),
                                   "micro_recall": _ratio(tp, tp + fn), "micro_f1": _ratio(2 * tp, 2 * tp + fp + fn)}
    write_json(output / "human_evaluation.json", result)
    return result
