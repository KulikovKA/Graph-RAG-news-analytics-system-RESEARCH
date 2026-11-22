"""Метрики и отчёты. Ссылочная разметка используется только в этом слое."""

from collections import Counter
import csv
from io import StringIO
import math
from statistics import mean

from frozen_pool import ORACLE_BUDGETS, encode_json
from integrated_benchmark_reporting import atomic_text, format_table as table_lines, write_json
from run_document_aware import measure, path_output

BASELINE = "heuristic_C"
POLICIES = ("neural_only", "hybrid_C_neural")
SEMANTICS = "NOT_EVALUATED"


def top_details(items):
    return [{**path_output(item), "edges": [
        {key: edge[key] for key in ("key", "type", "relation_type", "document_id", "head_key", "tail_key",
                                    "traversed_reverse")} for edge in item["path"]["edges"]
    ]} for item in items]


def candidate_oracle(pool, reference):
    required = set(reference["document_contributions"])
    paths = {p["path_id"]: p for p in pool.raw["paths"]}
    oracle = {}
    for budget in ORACLE_BUDGETS:
        ids = pool.order[:budget]
        docs = {d for pid in ids for d in paths[pid]["document_ids"]}
        factual = {e["edge"]["document_id"] for pid in ids for e in pool.bundles[pid]["edges"]
                   if e["edge"]["type"] == "RELEX_REL"}
        oracle[str(budget)] = {"paths": len(ids), "gold_hits": sorted(docs & required),
                               "factual_gold_hits": sorted(factual & required),
                               "recall": len(docs & required) / len(required),
                               "factual_recall": len(factual & required) / len(required),
                               "absent_factual_gold": sorted(required - factual)}
    return oracle


def evaluate(top, reference):
    metrics = measure(top, reference)
    required = set(reference["document_contributions"])
    documents = {d for item in top for d in item["document_ids"]}
    factual_documents = {d for item in top for d in item["factual_document_ids"]}
    seen = set()
    gain = reciprocal = 0.0
    for rank, item in enumerate(top, 1):
        hits = required.intersection(item["factual_document_ids"]) - seen
        if hits:
            gain += len(hits) / math.log2(1 + rank)
            if reciprocal == 0.0:
                reciprocal = 1 / rank
            seen.update(hits)
    windows = Counter((w["document_id"], w["revision_id"], w["text_sha256"], w["start"], w["end"])
                      for item in top for fact in item["facts"] for w in fact["windows"])
    transitions = Counter(edge["type"] for item in top for edge in item["edges"])
    factual = transitions["RELEX_REL"]
    total = transitions.total()
    return {**metrics, "nDCG@5_factual_doc_first_hit": gain / len(required),
            "MRR@5_factual_doc_first_hit": reciprocal, "unique_documents": len(documents),
            "duplicate_evidence_windows": windows.total() - len(windows),
            "repeated_evidence_fraction": (windows.total() - len(windows)) / max(1, windows.total()),
            "transition_counts": dict(transitions), "factual_transition_share": factual / total if total else None,
            "non_factual_transition_share": (total - factual) / total if total else None,
            "off_gold_documents": sorted(documents - required),
            "off_gold_sources": sorted(factual_documents - required)}


def percentile(values, fraction):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    return ordered[lower] + (ordered[math.ceil(rank)] - ordered[lower]) * (rank - lower)


def format_table(headings, rows):
    return "\n".join(table_lines(headings, rows))


def distribution(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": min(values), "p05": percentile(values, .05),
            "p50": percentile(values, .5), "p95": percentile(values, .95), "max": max(values),
            "mean": mean(values)}


def quality(rows):
    measures = [row["metrics"] for row in rows]
    gold_total = sum(len(row["gold_documents"]) for row in rows)
    evidence_total = sum(row["gold_evidence_count"] for row in rows)
    transitions = sum(sum(m["transition_counts"].values()) for m in measures)
    factual = sum(m["transition_counts"].get("RELEX_REL", 0) for m in measures)
    return {
        "GoldDocRecall@5_micro": sum(len(m["gold_documents_in_top_5"]) for m in measures) / gold_total,
        "GoldDocRecall@5_macro": mean(m["GoldDocRecall@5"] for m in measures),
        "FactualGoldDocRecall@5_micro": sum(len(m["factual_gold_documents_in_top_5"]) for m in measures) / gold_total,
        "FactualGoldDocRecall@5_macro": mean(m["FactualGoldDocRecall@5"] for m in measures),
        "EvidenceWindowOverlap@5_micro": sum(len(m["overlapping_evidence_ids"]) for m in measures) / evidence_total,
        **{key: mean(m[key] for m in measures) for key in (
            "AllRequiredDocs@5", "EvidenceWindowOverlap@5", "nDCG@5_factual_doc_first_hit",
            "MRR@5_factual_doc_first_hit", "unique_documents", "unique_factual_documents",
            "duplicate_evidence_windows", "repeated_evidence_fraction", "duplicate_source_triples")},
        "factual_transition_share": factual / transitions if transitions else None,
        "non_factual_transition_share": (transitions - factual) / transitions if transitions else None,
        "mean_off_gold_documents": mean(len(m["off_gold_documents"]) for m in measures),
        "mean_off_gold_sources": mean(len(m["off_gold_sources"]) for m in measures),
    }


def cost(rows, model_attempts):
    times = [row["timing"] for row in rows]
    inference = sum(t.get("inference_seconds", 0) for t in times)
    tokens = sum(t.get("original_tokens", 0) for t in times)
    return {
        "selection_seconds_total": sum(t["selection_seconds"] for t in times),
        "preselection_seconds_total": sum(t["preselection_seconds"] for t in times),
        "render_seconds_total": sum(t.get("render_seconds", 0) for t in times),
        "tokenization_seconds_total": sum(t.get("tokenization_seconds", 0) for t in times),
        "inference_seconds_total": inference,
        "latency_p50_seconds": percentile([t["rerank_seconds"] for t in times], .5),
        "latency_p95_seconds": percentile([t["rerank_seconds"] for t in times], .95),
        "pairs_per_inference_second": sum(t.get("pairs", 0) for t in times) / inference if inference else None,
        "peak_process_rss_mib": max((a["peak_process_rss_mib"] for a in model_attempts), default=None),
        "model_load_seconds_total": sum(a.get("model_load_seconds", 0) for a in model_attempts),
        "warmup_seconds_total": sum(a.get("warmup_seconds", 0) for a in model_attempts),
        "original_tokens": tokens, "truncated_tokens": sum(t.get("truncated_tokens", 0) for t in times),
        "truncation_fraction": sum(t.get("truncated_tokens", 0) for t in times) / tokens if tokens else None,
    }


def compare(before, after, budget):
    gold = set(before["gold_documents"])
    docs_before = set(before["metrics"]["factual_gold_documents_in_top_5"])
    docs_after = set(after["metrics"]["factual_gold_documents_in_top_5"])
    a, b = ([n["path_id"] for n in row["top_5"]] for row in (before, after))
    oracle = before["candidate_oracle"]
    return {"key": before["key"], "model": after["model"], "policy": after["policy"],
            "representation": after["representation"], "baseline_top_5": a, "top_5": b,
            "paths_added": [pid for pid in b if pid not in a],
            "paths_removed": [pid for pid in a if pid not in b],
            "gold_promoted": sorted(docs_after - docs_before), "gold_lost": sorted(docs_before - docs_after),
            "gold_rank_changes": {d: [first_rank(before, d), first_rank(after, d)] for d in sorted(gold)},
            "absent_from_neural_pool": oracle[str(budget)]["absent_factual_gold"],
            "absent_from_full_eligible_pool": oracle["200"]["absent_factual_gold"],
            "absent_from_raw_bfs": sorted(gold - set(before["raw_bfs_documents"])),
            "off_gold_sources": after["metrics"]["off_gold_sources"],
            "factual_recall_delta": after["metrics"]["FactualGoldDocRecall@5"] -
            before["metrics"]["FactualGoldDocRecall@5"]}


def first_rank(row, doc):
    return next((item["rank"] for item in row["top_5"] if doc in item["factual_document_ids"]), None)


def report(directory, config, manifest, baseline, outcomes):
    questions = config["question_keys"]
    settings = config["settings"]
    rows = [baseline[key] for key in questions if key in baseline]
    ordered_blocks = ((model, rep, key) for model in settings["models"]
                      for rep in settings["representation"] for key in questions)
    model_rows = [row for block in ordered_blocks if block in outcomes for row in outcomes[block]["evaluations"]]
    groups = {"ALL": set(config["question_keys"]),
              "APPROVED": {key for key in config["question_keys"] if not key.startswith("HOLDOUT/")}}
    groups.update({g: {key for key in config["question_keys"] if key.startswith(g + "/")}
                   for g in ("A", "B", "C", "HOLDOUT")})
    summaries, deltas, oracle = [], [], []
    for group, keys in groups.items():
        if not keys:
            continue
        c_rows = [row for row in rows if row["key"] in keys]
        if c_rows:
            summaries.append({"group": group, "model": BASELINE, "tier": "Baseline", "policy": BASELINE,
                              "representation": None, "completed": len(c_rows), "expected": len(keys),
                              **quality(c_rows), **cost(c_rows, [])})
            total = sum(len(row["gold_documents"]) for row in c_rows)
            for budget in ORACLE_BUDGETS:
                refs = [row["candidate_oracle"][str(budget)] for row in c_rows]
                oracle.append({"group": group, "budget": budget, "questions": len(c_rows),
                               "recall_micro": sum(len(r["gold_hits"]) for r in refs) / total,
                               "factual_recall_micro": sum(len(r["factual_gold_hits"]) for r in refs) / total,
                               "recall_macro": mean(r["recall"] for r in refs),
                               "factual_recall_macro": mean(r["factual_recall"] for r in refs)})
        for model in config["settings"]["models"]:
            for representation in config["settings"]["representation"]:
                for policy in POLICIES:
                    tag = {"group": group, "model": model, "tier": config["tiers"][model],
                           "policy": policy, "representation": representation}
                    active = [r for r in model_rows if r["key"] in keys and r["model"] == model
                              and r["representation"] == representation and r["policy"] == policy]
                    attempts = [a for a in manifest["model_attempts"] if a["model"] == model]
                    measured = {**quality(active), **cost(active, attempts)} if active else {}
                    summaries.append({**tag, "completed": len(active), "expected": len(keys), **measured})
                    if active:
                        after = quality(active)
                        before = quality([baseline[r["key"]] for r in active])
                        deltas.append({**tag, "paired_questions": len(active), "baseline": before,
                                       "after": after, "delta": {k: after[k] - v if v is not None and after[k] is not None
                                                                else None for k, v in before.items()}})
    changes = [compare(baseline[row["key"]], row, config["settings"]["candidate_budget"]) for row in model_rows]
    for row in rows:
        row_key = row["key"]
        if row_key.rsplit("/", 1)[-1] in {"3A52-C03", "3A52-C05", "3A52-C07", "3A52-C09"}:
            row["audit"] = {"factual_gold": row["metrics"]["factual_gold_documents_in_top_5"],
                            "gold_first_rank": {d: first_rank(row, d) for d in row["gold_documents"]},
                            "absent_full": row["candidate_oracle"]["200"]["absent_factual_gold"]}
    scores = {model: {rep: distribution([pair["neural_score"] for (m, r, _), out in outcomes.items()
                                         if m == model and r == rep for pair in out["pairs"]])
                      for rep in config["settings"]["representation"]} for model in config["settings"]["models"]}
    result = {"manifest": manifest, "summaries": summaries, "paired_deltas": deltas, "candidate_oracle": oracle,
              "SemanticEvidenceCoverage@5": SEMANTICS, "top_5_changes": changes, "score_distributions": scores}
    write_json(directory / "results.json", result)
    atomic_text(directory / "per_question.jsonl", "".join(encode_json(row) + "\n" for row in rows + model_rows))
    delta_fields = {(r["group"], r["model"], r["policy"], r["representation"]): r for r in deltas}
    flat = []
    for row in summaries:
        d = delta_fields.get(tuple(row[k] for k in ("group", "model", "policy", "representation")), {})
        flat.append({**row, **{"paired_delta_" + k: v for k, v in d.get("delta", {}).items()}})
    columns = list(dict.fromkeys(k for row in flat for k in row))
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(flat)
    atomic_text(directory / "comparison.csv", buffer.getvalue(), encoding="utf-8-sig")
    atomic_text(directory / "comparison.md", markdown(config, result, rows))


def markdown(config, result, baselines):
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    all_rows = [r for r in result["summaries"] if r["group"] == "ALL"]
    delta = {(r["model"], r["representation"], r["policy"]): r for r in result["paired_deltas"]
             if r["group"] == "ALL"}
    lines = ["# Stage 3A.5.8 — Neural & Hybrid Graph-RAG Reranking", "",
             f"Status **{result['manifest']['status']}**. Frozen B, budget {config['settings']['candidate_budget']}. "
             "Latency covers reranking only. Model load and warmup are separate.", "",
             format_table(["Model", "Size", "Representation", "Policy", "N", "Recall μ", "Recall macro",
                           "Factual μ", "Δ factual μ paired", "All docs", "Overlap μ", "p50 s", "p95 s",
                           "Pairs/s inference", "Peak RSS MiB"], [
                 [r["model"], r["tier"], r["representation"] or "—", r["policy"],
                  f"{r['completed']}/{r['expected']}", *[
                      number(r.get(k)) for k in ("GoldDocRecall@5_micro", "GoldDocRecall@5_macro",
                                                 "FactualGoldDocRecall@5_micro")],
                  number(delta.get((r["model"], r["representation"], r["policy"]), {}).get(
                      "delta", {}).get("FactualGoldDocRecall@5_micro")),
                  *[number(r.get(k)) for k in ("AllRequiredDocs@5", "EvidenceWindowOverlap@5_micro",
                                              "latency_p50_seconds", "latency_p95_seconds",
                                              "pairs_per_inference_second", "peak_process_rss_mib")]]
                 for r in sorted(all_rows, key=lambda r: ("Baseline Small Base Large".split().index(r["tier"]),
                                                          r["model"]))]), "",
             "## Candidate oracle", "",
             format_table(["Group", "Budget", "Questions", "Gold μ", "Factual Gold μ"], [
                 [r["group"], r["budget"], r["questions"], number(r["recall_micro"]), number(r["factual_recall_micro"])]
                 for r in result["candidate_oracle"]]), "",
             "## C03, C05, C07 and C09", "",
             "Approved C and historical HOLDOUT are shown under their respective keys. Rank changes refer to "
             "the first selected path with validated factual evidence for the document.", ""]
    for row in baselines:
        if "audit" in row:
            lines += [f"- **{row['key']}** — C factual hits {row['audit']['factual_gold']}, missing from full eligible "
                      f"pool {row['audit']['absent_full']}, off-Gold sources {row['metrics']['off_gold_sources']}."]
    audits = {r["key"] for r in baselines if "audit" in r}
    lines += ["", format_table(["Question", "Model / repr / policy", "Promoted", "Lost", "Absent at budget",
                               "Gold ranks C → new", "+/− paths"], [
        [r["key"], f"{r['model']} / {r['representation']} / {r['policy']}", ", ".join(r["gold_promoted"]),
         ", ".join(r["gold_lost"]), ", ".join(r["absent_from_neural_pool"]),
         ", ".join(f"{d} {a or '—'}→{b or '—'}" for d, (a, b) in r["gold_rank_changes"].items()),
         f"+{len(r['paths_added'])}/−{len(r['paths_removed'])}"]
        for r in result["top_5_changes"] if r["key"] in audits]), "",
        "## Top-5 changes", "",
        "Full ordered paths and source windows are in per_question.jsonl. Every paired change, off-Gold source "
        "and the Gold gaps at budget, full eligible pool and raw BFS are recorded in results.json.", ""]
    changes = sorted((r for r in result["top_5_changes"] if r["baseline_top_5"] != r["top_5"]),
                     key=lambda r: -abs(r["factual_recall_delta"]))
    for r in changes[:8]:
        show = lambda paths: ", ".join(pid.split(":")[-1][:10] for pid in paths) or "—"
        lines.append(f"- {r['key']} · {r['model']} · {r['representation']} · {r['policy']} — "
                     f"ordered path IDs {show(r['baseline_top_5'])} → {show(r['top_5'])}. "
                     f"Factual Gold added {r['gold_promoted']}, lost {r['gold_lost']}.")
    lines += ["", "## Interpretation and quality/speed", "",
              "Document retrieval metrics use the frozen reference set. Off-Gold sources may be relevant. "
              "Semantic Evidence Coverage is **NOT_EVALUATED**. Graph provenance and overlap establish source "
              "identity and offsets. Semantic correctness of extracted relations needs independent factual labels. "
              "Historical HOLDOUT can overlap in content with approved questions.", ""]
    approved = [r for r in result["summaries"] if r["group"] == "APPROVED"]
    baseline = next((r for r in approved if r["model"] == BASELINE and r["completed"] == 25), None)
    contenders = [] if baseline is None else [
        r for r in approved if r["model"] != BASELINE and r["completed"] == 25
        and r["FactualGoldDocRecall@5_micro"] > baseline["FactualGoldDocRecall@5_micro"]
        and r["AllRequiredDocs@5"] >= baseline["AllRequiredDocs@5"]
        and r["latency_p95_seconds"] <= 10 and (r["peak_process_rss_mib"] or math.inf) <= 8192]
    if contenders and result["manifest"]["status"] == "COMPLETE":
        best = min(contenders, key=lambda r: (r["latency_p95_seconds"], -r["FactualGoldDocRecall@5_micro"]))
        lines.append(f"Candidate for follow-up audit: **{best['model']} / {best['representation']} / "
                     f"{best['policy']}**. It is the fastest eligible configuration by p95.")
    elif baseline and result["manifest"]["status"] == "COMPLETE":
        lines.append("Retain C as the quality/speed reference. The completed configurations did not meet all "
                     "predeclared follow-up criteria.")
    else:
        lines.append("A quality/speed recommendation requires a completed full run.")
    lines += ["", "Follow-up criteria on the 25 approved questions are higher factual micro recall, "
              "AllRequiredDocs at least equal to C, reranking p95 ≤ 10 s and peak process RSS ≤ 8 GiB. "
              "The report makes no production promotion or statistical-significance decision.", ""]
    return "\n".join(lines)
