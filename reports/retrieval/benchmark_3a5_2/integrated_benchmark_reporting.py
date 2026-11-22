"""Metrics and artifacts for the integrated retrieval benchmark."""

from collections import Counter
import csv
from io import StringIO
import json
import os
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile


EFFECTS = (
    ("A", "B", "Document-aware reranking A → C"),
    ("B", "C", "Entity Linking top_k 5 → 10"),
    ("C", "D", "FIFO BFS max_candidates 200 → 100"),
    ("A", "D", "Joint effect"),
)
FOCUS_IDS = {"3A52-C03", "3A52-C05", "3A52-C07", "3A52-C09"}
MAIN_FIELDS = {
    "gold_recall_micro": "Gold Recall micro", "gold_recall_macro": "Gold Recall macro",
    "factual_recall_micro": "Factual Recall", "all_required_rate": "AllRequired",
    "window_overlap_micro": "Window Overlap", "mean_unique_factual_docs": "Unique Docs",
    "mean_standalone_seconds": "Latency (s)",
}
AGGREGATE_MEANS = {
    "gold_recall_macro": "GoldDocRecall@5", "factual_recall_macro": "FactualGoldDocRecall@5",
    "all_required_rate": "AllRequiredDocs@5", "window_overlap_macro": "EvidenceWindowOverlap@5",
    "candidate_recall_macro": "GoldDocCandidateRecall",
    "factual_candidate_recall_macro": "FactualGoldDocCandidateRecall",
    "mean_unique_factual_docs": "unique_factual_documents", "mean_duplicate_triples": "duplicate_source_triples",
    "mean_outside_gold_docs": "outside_gold_count", "mean_outside_gold_factual_docs": "outside_gold_factual_count",
    "mean_top_cross_document_factual_paths": "top_cross_document_factual_paths",
    "mean_top_navigation_transitions": "top_navigation_transitions",
    "mean_candidate_cross_document_factual_paths": "candidate_cross_document_factual_paths",
    "mean_candidate_navigation_transitions": "candidate_navigation_transitions",
}
ROW_MEANS = ("candidate_count", "nodes_examined", "relationships_examined", "wall_seconds",
             "standalone_seconds", "computed_stage_seconds")
STAGE_NAMES = ("ner", "e5", "linking", "bfs", "evidence", "rerank")
BFS_LIMIT_NAMES = ("max_hops", "max_neighbors", "max_candidates", "max_relationships")


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path, text, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding=encoding, newline="", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, encode_json(value) + "\n")


def evaluate_metrics(question, variant, retrieval, bundles, ranking, gold, nodes, timings, linking_trace):
    from run_block_c import compact_paths
    from run_document_aware import measure, path_output

    paths, _ = compact_paths(retrieval, bundles, nodes)
    selected = [paths[item["path_id"]] for item in ranking["paths"]]
    top = [path_output(item) for item in ranking["paths"]]
    for item, path in zip(top, selected, strict=True):
        item["transitions"] = path["transitions"]
        item["node_keys"] = path["node_keys"]
        item["edge_keys"] = [edge["key"] for edge in path["edges"]]
        for source in item["facts"]:
            for window in source["windows"]:
                # Полный текст и свойства сохранены в артефакте стадии rerank.
                del window["text"]
    metrics = measure(top, gold)
    required = set(gold["document_contributions"])
    all_docs = {doc for path in paths.values() for doc in path["document_ids"]}
    all_factual = {doc for path in paths.values() for doc in path["factual_documents"]}
    top_docs = {doc for path in selected for doc in path["document_ids"]}
    factual_docs = {doc for path in selected for doc in path["factual_documents"]}
    candidate_hits = required & all_docs
    factual_candidate_hits = required & all_factual

    def navigation_count(pool):
        return sum(bool({"HAS_SHARED_CONCEPT", "SIMILAR_TO"} & set(t["edge_types"]))
                   for path in pool for t in path["transitions"])

    metrics.update(
        GoldDocCandidateRecall=len(candidate_hits) / len(required),
        FactualGoldDocCandidateRecall=len(factual_candidate_hits) / len(required),
        candidate_gold_hits=sorted(candidate_hits), factual_candidate_gold_hits=sorted(factual_candidate_hits),
        outside_gold_count=len(top_docs - required), outside_gold_factual_count=len(factual_docs - required),
        top_cross_document_factual_paths=sum(path["cross_doc_factual"] for path in selected),
        candidate_cross_document_factual_paths=sum(path["cross_doc_factual"] for path in paths.values()),
        top_navigation_transitions=navigation_count(selected),
        candidate_navigation_transitions=navigation_count(paths.values()),
    )
    active = [stage for stage in timings.values() if stage is not None]
    return {
        "key": question.key, "dataset_group": question.group, "question_id": question.question_id,
        "question": question.text, "variant": variant, "status": "COMPLETE",
        "gold_count": len(required), "gold_docs": sorted(required), "gold_evidence_count": len(gold["evidence"]),
        "reference_set_non_exhaustive": gold.get("reference_set_non_exhaustive"),
        "top_docs": sorted(top_docs), "top_factual_docs": sorted(factual_docs),
        "outside_gold_docs": sorted(top_docs - required), "candidate_documents": sorted(all_docs),
        "candidate_factual_documents": sorted(all_factual), "candidate_count": len(retrieval["paths"]),
        "top_5": top, "ranking_diagnostics": ranking["ranking_diagnostics"], "ranking_reason": ranking["reason"],
        "truncated": retrieval["truncated"], **retrieval["diagnostics"], "metrics": metrics, "timings": timings,
        "linking": linking_trace,
        "standalone_seconds": sum(stage["seconds"] for stage in active),
        "computed_stage_seconds": sum(stage["seconds"] for stage in active if stage["origin"] == "computed"),
    }


def summarize(rows):
    if not rows:
        return {}
    metrics = [row["metrics"] for row in rows]
    gold_total = sum(r["gold_count"] for r in rows)
    evidence_total = sum(r["gold_evidence_count"] for r in rows)
    gold_hits = sum(len(m["gold_documents_in_top_5"]) for m in metrics)
    factual_hits = sum(len(m["factual_gold_documents_in_top_5"]) for m in metrics)
    overlaps = sum(len(m["overlapping_evidence_ids"]) for m in metrics)
    counter = Counter(limit for row in rows for limit in row["limits_hit"])
    return {
        "gold_hits": gold_hits, "gold_total": gold_total, "gold_recall_micro": gold_hits / gold_total,
        "factual_gold_hits": factual_hits, "factual_recall_micro": factual_hits / gold_total,
        "overlapping_evidence_spans": overlaps, "gold_evidence_spans": evidence_total,
        "window_overlap_micro": overlaps / evidence_total,
        "candidate_recall_micro": sum(len(m["candidate_gold_hits"]) for m in metrics) / gold_total,
        "factual_candidate_recall_micro": sum(len(m["factual_candidate_gold_hits"]) for m in metrics) / gold_total,
        **{field: mean(m[key] for m in metrics) for field, key in AGGREGATE_MEANS.items()},
        **{f"mean_{field}": mean(row[field] for row in rows) for field in ROW_MEANS},
        **{f"mean_{name}_seconds": mean((row["timings"][name] or {}).get("seconds", 0.0) for row in rows)
           for name in STAGE_NAMES},
        **{f"mean_{key}": mean((row["timings"]["e5"] or {}).get(key, 0.0) for row in rows)
           for key in ("encoding_seconds", "cosine_and_validation_seconds")},
        "truncation_rate": mean(row["truncated"] for row in rows),
        **{f"{limit}_hit_rate": counter[limit] / len(rows) for limit in BFS_LIMIT_NAMES},
    }


def variant_deltas(before, after):
    old_rank = {p["path_id"]: p["rank"] for p in before["top_5"]}
    new_rank = {p["path_id"]: p["rank"] for p in after["top_5"]}
    old_gold = set(before["metrics"]["gold_documents_in_top_5"])
    new_gold = set(after["metrics"]["gold_documents_in_top_5"])
    old_factual = set(before["metrics"]["factual_gold_documents_in_top_5"])
    new_factual = set(after["metrics"]["factual_gold_documents_in_top_5"])
    return {
        "gold_gained": sorted(new_gold - old_gold), "gold_lost": sorted(old_gold - new_gold),
        "factual_gold_gained": sorted(new_factual - old_factual), "factual_gold_lost": sorted(old_factual - new_factual),
        "documents_added": sorted(set(after["top_docs"]) - set(before["top_docs"])),
        "documents_removed": sorted(set(before["top_docs"]) - set(after["top_docs"])),
        "rank_changes": [{"path_id": key, "before": old_rank.get(key), "after": new_rank.get(key)}
                         for key in dict.fromkeys([*new_rank, *old_rank]) if old_rank.get(key) != new_rank.get(key)],
    }


def question_comparison(questions, saved, variants):
    ordered_metrics = ("GoldDocRecall@5", "FactualGoldDocRecall@5", "EvidenceWindowOverlap@5")
    output = []
    for question in questions:
        rows = {v: saved[question.key, v] for v in variants if (question.key, v) in saved}
        scores = {v: tuple(row["metrics"][name] for name in ordered_metrics) for v, row in rows.items()}
        best = [v for v in scores if scores[v] == max(scores.values())]
        changes = {f"{left}_to_{right}": variant_deltas(rows[left], rows[right])
                   for left, right, _ in EFFECTS if left in rows and right in rows}
        output.append({"key": question.key, "best": best, "changes": changes, "scores": scores})
    return output


def format_table(headings, rows):
    return ["| " + " | ".join(headings) + " |",
            "| " + " | ".join("---" for _ in headings) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows)]


def markdown(config, manifest, summaries, effects, comparisons, saved):
    identity = config["identity"]
    graph = identity["graph"]
    version = identity["versions"]
    lines = [
        "# Stage 3A.5.7 — Integrated retrieval benchmark", "",
        f"**{manifest['status']}** · {manifest['completed_questions']}/{manifest['expected_questions']} questions "
        f"· {manifest['completed_evaluations']}/{manifest['expected_evaluations']} evaluations", "",
        f"- Started {manifest['started_at_utc']} UTC",
        f"- Git commit `{identity['git_commit']}`",
        f"- Python {version['python']}, PyTorch {version['torch']}, Neo4j driver {version['neo4j']}",
        f"- Namespace `{graph['namespace']}`",
        f"- Graph fingerprint `{graph['graph_fingerprint']}`",
        f"- Models `{identity['models']['GLiNER']['id']}` and `{identity['models']['E5']['id']}` "
        "(weight hashes in [config.json](config.json))",
        "- Semantic Evidence Coverage = NOT_EVALUATED", "",
        "## Policies", "",
        *format_table(("Variant", "Reranker", "Cutoff", "Link top_k", "Fallback", "BFS candidates"),
                      ((v, c["reranking"], c["linking"]["min_score"], c["linking"]["top_k"],
                        c["linking"]["fallback"], c["bfs"]["max_candidates"]) for v, c in config["variants"].items())),
        "", "All policies use GLiNER 0.15, FIFO BFS (5 hops, 50 neighbors, 1,000 relationships, 32 seeds, "
        "10-second transaction timeout), and five ranked paths.", "",
        "## Quality and latency", "",
        *format_table(("Group", "Variant", "Completed", *MAIN_FIELDS.values()),
                      ((s["group"], s["variant"], f"{s['completed']}/{s['expected']}",
                        *(f"{s[k]:.3f}" if k in s else "—" for k in MAIN_FIELDS)) for s in summaries)),
        "", "Latency is the sum of the measured pipeline stages for each policy with shared stage costs "
        "included. It includes observed first-use model loading. See the CSV for actual wall time and "
        "additional stage computation. Partial summaries use completed evaluations.", "",
        "## Component comparisons", "",
        *format_table(("Group", "Change", "Component", "Paired q", "Δ Gold micro", "Δ Factual micro",
                       "Δ Candidate micro", "Δ Window micro", "Δ Unique Docs", "Δ latency (s)"),
                      ((e["group"], e["comparison"], e["component"], e["questions"],
                        *(f"{e[k]:+.3f}" for k in ("gold_recall_micro", "factual_recall_micro",
                                                   "candidate_recall_micro", "window_overlap_micro",
                                                   "mean_unique_factual_docs", "mean_standalone_seconds")))
                       for e in effects)),
        "", "Deltas use the same questions on both sides. Adjacent comparisons hold the other settings fixed. "
        "A → D measures the joint effect.", "",
        "## Questions", "",
        *format_table(("Question", "A", "B", "C", "D", "Best"),
                      ((r["key"], *(f"{r['scores'][v][0]:.3f}" if v in r["scores"] else "—" for v in "ABCD"),
                        ", ".join(r["best"]) or "—") for r in comparisons)),
        "", "Cells show GoldDocRecall@5. Best uses Gold recall, factual recall and then window overlap, "
        "with ties listed. [Per-question data](per_question.jsonl) has document IDs, ranked paths and source "
        "windows for every completed variant. [Results](results.json) has all pairwise document and rank changes.",
        "", "## C03, C05, C07 and C09", "",
    ]
    for comparison in comparisons:
        key = comparison["key"]
        if key.split("/")[1] not in FOCUS_IDS:
            continue
        lines += [f"### {key}", ""]
        for variant in comparison["scores"]:
            row = saved[key, variant]
            metrics = row["metrics"]
            source_docs = ", ".join(row["top_factual_docs"]) or "—"
            ranked = ", ".join(f"{p['rank']}. `{p['path_id'][:12]}`" for p in row["top_5"]) or "—"
            lines.append(f"- **{variant}** · Gold found {', '.join(metrics['gold_documents_in_top_5']) or '—'} "
                         f"· missing {', '.join(metrics['missing_gold_documents']) or '—'} · "
                         f"candidate Gold {', '.join(metrics['candidate_gold_hits']) or '—'} · "
                         f"fact sources {source_docs} · paths {ranked}. [Evidence]({row['timings']['rerank']['file']})")
        for label, change in comparison["changes"].items():
            gained = ", ".join(change["gold_gained"]) or "—"
            lost = ", ".join(change["gold_lost"]) or "—"
            new_docs = ", ".join(change["documents_added"]) or "—"
            lines.append(f"- {label.replace('_to_', ' → ')} · Gold gained {gained} · Gold lost {lost} "
                         f"· new docs {new_docs} · {len(change['rank_changes'])} path rank changes")
        lines.append("")
    lines += ["## Interpretation", "",
              "The dataset covers 35 questions over a frozen corpus. The historical C set shares IDs with "
              "approved C, and some question texts recur. Results are specific to the pinned snapshot and query set.",
              "Top-5 counts paths. Factual documents require validated source provenance. Window overlap uses "
              "the same document and revision with intersecting offsets. Documents outside the reference set "
              "can be relevant. Navigation transitions and validated factual cross-document paths have separate counts.",
              "BFS truncation follows the retrieval diagnostics. Timing depends on the host, Neo4j load, query order "
              "and caching. Execution attempts and errors are recorded in the manifest and errors.jsonl.", ""]
    joint_effect = next((e for e in effects if e["group"] == "ALL" and e["comparison"] == "A → D"), None)
    if joint_effect:
        lines += [f"Across {joint_effect['questions']} paired questions, D changed Gold recall by "
                  f"{100 * joint_effect['gold_recall_micro']:+.1f} percentage points, factual recall by "
                  f"{100 * joint_effect['factual_recall_micro']:+.1f} points and mean standalone pipeline "
                  f"latency by {joint_effect['mean_standalone_seconds']:+.3f} s relative to A.", ""]
    return "\n".join(lines)


def write_reports(directory, config, manifest, questions, saved):
    variants = config["selection"]["variants"]
    summaries, effects = [], []
    for group in ("ALL", *dict.fromkeys(q.group for q in questions)):
        keys = [q.key for q in questions if group in ("ALL", q.group)]
        for variant in variants:
            rows = [saved[key, variant] for key in keys if (key, variant) in saved]
            summaries.append({"group": group, "variant": variant, "completed": len(rows),
                              "expected": len(keys), **summarize(rows)})
        for left, right, component in EFFECTS:
            paired = [key for key in keys if (key, left) in saved and (key, right) in saved]
            if paired:
                before = summarize([saved[key, left] for key in paired])
                after = summarize([saved[key, right] for key in paired])
                effects.append({"group": group, "comparison": f"{left} → {right}", "component": component,
                                "questions": len(paired), **{k: after[k] - before[k] for k in after}})
    comparisons = question_comparison(questions, saved, variants)
    rows = [saved[q.key, v] for q in questions for v in variants if (q.key, v) in saved]
    unique = {}
    for row in rows:
        for name, stage in row["timings"].items():
            if stage:
                unique.setdefault((name, stage["key"]), stage["seconds"])
    pipeline_cost = {name: sum(value for (stage, _), value in unique.items() if stage == name)
                     for name in STAGE_NAMES}
    write_json(directory / "results.json", {
        "manifest": manifest, "summary": summaries, "paired_effects": effects, "questions": comparisons,
        "SemanticEvidenceCoverage@5": "NOT_EVALUATED",
        "unique_completed_pipeline_stage_seconds": pipeline_cost,
        "observed_attempt_wall_seconds": sum(a["wall_seconds"] for a in manifest["attempts"]),
    })
    atomic_text(directory / "per_question.jsonl", "".join(encode_json(row) + "\n" for row in rows))
    columns = list(dict.fromkeys([
        "group", "variant", "completed", "expected", *MAIN_FIELDS,
        *(k for row in summaries for k in row),
    ]))
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(summaries)
    atomic_text(directory / "comparison.csv", buffer.getvalue(), encoding="utf-8-sig")
    atomic_text(directory / "comparison.md", markdown(config, manifest, summaries, effects, comparisons, saved))
