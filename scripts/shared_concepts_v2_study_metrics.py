"""Структурные метрики и визуализация поэтапного исследования Shared Concepts V2."""

from collections import Counter, defaultdict
import csv
from fractions import Fraction
from itertools import combinations
import os

import numpy as np

from cgraph.shared_concepts_v2 import COSINE_BINS, digest
from shared_concepts_v2_pipeline import summaries, write_checked

PARAMETERS = ("ec_name", "ec_context", "cc_name", "cc_context", "delta")


def score_histogram(scores):
    values = np.asarray(scores).reshape(-1, 2)
    return [np.histogram(values[:, channel], bins=COSINE_BINS)[0].tolist() for channel in range(2)]


def measure(catalog, topics):
    result = summaries(catalog)
    totals = Counter(topics.values())
    total_documents = defaultdict(set)
    for key, topic in topics.items():
        total_documents[topic].add(catalog.entities[key].document_id)
    coverage = {
        topic: result["by_topic"].get(topic, dict(entities=0, documents=0, memberships=0)) |
        dict(total_entities=totals[topic], total_documents=len(total_documents[topic]))
        for topic in sorted(totals)
    }
    concepts = list(catalog.concepts.values())
    member_sets = sorted(c.members for c in concepts)
    entity_pairs = {
        (a, b) for members in member_sets for a, b in combinations(members, 2)
        if catalog.entities[a].document_id != catalog.entities[b].document_id
    }
    ec_events = [event for event in catalog.events if event["operation"] in ("initialize", "merge")]
    ec_outcomes = Counter("PASS" if event["reason"] == "NESTING" else event["reason"] for event in ec_events)
    rounds = [row for event in ec_events for row in event["rounds"]]
    ec_round_counts = Counter(str(len(event["rounds"])) for event in ec_events)
    paired = result["concept_sizes"].get("2", 0)
    return result | dict(
        by_topic=coverage, covered_disciplines=sum(row["documents"] > 0 for row in coverage.values()),
        cross_document_entity_pairs=len(entity_pairs), catalog_members_sha256=digest(member_sets),
        pair_concepts=paired, pair_concept_share=paired / len(concepts) if concepts else None,
        ec=dict(attempts=len(ec_events), outcomes=dict(ec_outcomes), rounds=dict(ec_round_counts)),
        cosine=dict(ec_rounds=score_histogram([score for row in rounds for score in row["scores"]]),
                    ec_final=score_histogram([score for concept in concepts for score in concept.scores]),
                    cc=catalog.cc_histogram.tolist()),
    )


def structural_rank(row):
    m = row["metrics"]
    weakest = min((Fraction(x["entities"], x["total_entities"]) for x in m["by_topic"].values()), default=0)
    overlap = Fraction(m["memberships"], m["covered_entities"]) if m["covered_entities"] else 0
    return (-m["covered_disciplines"], -m["covered_documents"], -weakest,
            -m["covered_entities"], overlap, m["concepts"], tuple(row["point"]))


def shortlist(rows, limit, distinct_dimensions=5):
    chosen = {}
    for row in sorted(rows, key=structural_rank):
        chosen.setdefault(tuple(row["point"][:distinct_dimensions]), row)
        if len(chosen) == limit:
            break
    return list(chosen.values())


def flatten(row, stages):
    m = row["metrics"]
    stages = "/".join(stage for stage, points in stages.items() if tuple(row["point"]) in points)
    return (
        {name: number / 100 for name, number in zip(PARAMETERS, row["point"], strict=True)} |
        dict(stages=stages, concepts=m["concepts"], memberships=m["memberships"],
             entities=m["covered_entities"], documents=m["covered_documents"], disciplines=m["covered_disciplines"],
             document_pairs=m["document_pairs"], cross_document_entity_pairs=m["cross_document_entity_pairs"],
             pair_concepts=m["pair_concepts"], pair_concept_share=m["pair_concept_share"],
             strict_inclusions=m["strict_inclusions"], ec_pass=m["ec"]["outcomes"].get("PASS", 0),
             ec_reject=m["ec"]["attempts"] - m["ec"]["outcomes"].get("PASS", 0),
             cc_pass=m["cc"].get("eligible", 0), cc_reject=m["cc"].get("threshold_reject", 0),
             cc_nested=m["cc"].get("nested", 0),
             merge_pass=m["operations"]["merge"]["outcomes"].get("PASS", 0),
             merge_reject=m["operations"]["merge"]["attempts"] - m["operations"]["merge"]["outcomes"].get("PASS", 0),
             ec_excluded=m["operations"]["initialize"]["accepted_exclusions"] +
                         m["operations"]["merge"]["accepted_exclusions"],
             ec_rounds_max=max(map(int, m["ec"]["rounds"]), default=0),
             ec_exclusion_budget=m["ec"]["outcomes"].get("EXCLUSION_BUDGET", 0),
             ec_min_members_or_docs=m["ec"]["outcomes"].get("MIN_MEMBERS_OR_DOCS", 0),
             ec_invalid_sum=m["ec"]["outcomes"].get("INVALID_SUM", 0),
             merge_nesting_reject=m["operations"]["merge"]["outcomes"].get("NESTING", 0),
             elapsed_seconds=row["elapsed_seconds"], peak_rss_bytes=row["peak_rss_bytes"],
             catalog_members_sha256=m["catalog_members_sha256"])
    )


def heatmaps(plt, panels, x_index, y_index, figure_path):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.7), squeeze=False,
                             layout="constrained")
    covered = [r["metrics"]["covered_entities"] for _, rows in panels for r in rows]
    image = None
    for axis, (title, rows) in zip(axes[0], panels, strict=True):
        xs = sorted({r["point"][x_index] for r in rows})
        ys = sorted({r["point"][y_index] for r in rows})
        values = {(r["point"][x_index], r["point"][y_index]): r["metrics"]["covered_entities"] for r in rows}
        image = axis.imshow([[values.get((x, y), np.nan) for x in xs] for y in ys], origin="lower",
                            vmin=min(covered), vmax=max(max(covered), min(covered) + 1), aspect="auto", cmap="viridis")
        axis.set(xticks=range(len(xs)), yticks=range(len(ys)), xticklabels=[f"{x/100:.2f}" for x in xs],
                 yticklabels=[f"{y/100:.2f}" for y in ys], xlabel=PARAMETERS[x_index], ylabel=PARAMETERS[y_index],
                 title=title)
        axis.tick_params(axis="x", labelrotation=45)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Covered entities")
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)


def plot_delta(plt, rows, path):
    series = defaultdict(list)
    for row in rows:
        series[tuple(row["point"][:4])].append(row)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, layout="constrained")
    for anchor, values in sorted(series.items()):
        values.sort(key=lambda r: r["point"][4])
        delta = [r["point"][4] / 100 for r in values]
        label = "/".join(f"{v/100:.2f}" for v in anchor)
        axes[0].plot(delta, [r["metrics"]["covered_entities"] for r in values], "o-", label=label)
        axes[1].plot(delta, [sum(r["metrics"]["operations"][phase]["accepted_exclusions"]
                                for phase in ("initialize", "merge")) for r in values], "o-", label=label)
    axes[0].set_ylabel("Covered entities")
    axes[0].legend(title="EC name/context, CC name/context")
    axes[1].set(xlabel="delta", ylabel="Excluded memberships (accepted operations)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def atomic_text(path, writer):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_reports(output, rows, stages, final, ee, smoke):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = {tuple(row["point"]): row for row in rows}
    stage_sets = {stage: set(points) for stage, points in stages.items()}
    flattened = [flatten(row, stage_sets) for row in rows]
    write_checked(output / "results.json", dict(
        mode="smoke" if smoke else "study", ee=ee, cosine_bin_edges=COSINE_BINS.tolist(),
        cosine_channels=["name", "context"], stages=stages, results=rows,
        shortlist=[row["point"] for row in final]))

    def csv_rows(stream):
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)

    atomic_text(output / "results.csv", csv_rows)
    heatmaps(plt, [("A · EC", [index[key] for key in stages["A"]])], 0, 1, output / "ec_heatmap.png")
    panels = defaultdict(list)
    for key in stages["B"]:
        panels[key[:2]].append(index[key])
    heatmaps(plt, [(f"EC {a/100:.2f} / {b/100:.2f}", values) for (a, b), values in sorted(panels.items())],
             2, 3, output / "cc_heatmaps.png")
    plot_delta(plt, [index[key] for key in stages["C"]], output / "delta.png")

    lines = [
        "# Shared Concepts V2 · Corpus100",
        "", f"Режим **{'smoke' if smoke else 'поэтапный эксперимент'}**. "
        f"Измерено {len(rows)} уникальных конфигураций. После EE-защит прошло {ee['guarded_pairs']} пар.",
        "", "| Этап | Точек, включая переиспользованные |", "|---|---:|",
        *(f"| {stage} | {len(points)} |" for stage, points in stages.items()),
        "", "Ранжирование использует убывающие охват дисциплин, документов, минимальную долю покрытых сущностей "
        "в дисциплине и общий охват сущностей. Далее идут по возрастанию среднее число memberships "
        "на покрытую сущность, число концептов и численные параметры.",
        "", "Пять перспективных настроек выбираются среди измеренных точек с разными сочетаниями EC/CC. "
        "Структурные измерения служат навигацией для дальнейшей оценки на разметке.",
        "", "| EC name/context | CC name/context | δ | Концепты | Memberships | Сущности | Документы |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in final:
        a, b, c, d, delta = [v / 100 for v in row["point"]]
        m = row["metrics"]
        lines.append(f"| {a:.2f}/{b:.2f} | {c:.2f}/{d:.2f} | {delta:.2f} | {m['concepts']} | "
                     f"{m['memberships']} | {m['covered_entities']} | {m['covered_documents']} |")
    lines += ["", "[CSV](results.csv) · [JSON](results.json) · [EC](ec_heatmap.png) · "
              "[CC](cc_heatmaps.png) · [δ](delta.png) · [Подробные артефакты](details/MANIFEST.json)", ""]
    atomic_text(output / "REPORT.md", lambda stream: stream.write("\n".join(lines)))
