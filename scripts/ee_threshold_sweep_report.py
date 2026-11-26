"""Metrics and local publication for the Stage 3A.5.13 EE sweep."""
from time import perf_counter

import numpy as np

from build_corpus_graph import atomic_json, sha, signature
from contextual_shared_concepts import RULES, filter_mask, guard_reasons
from contextual_shared_concepts_report import atomic_csv, pair_evidence


def neighborhood(degrees):
    quantiles = np.percentile(degrees, [25, 50, 75, 90, 95, 99])
    histogram = {str(i): int(count) for i, count in enumerate(np.bincount(degrees)) if count}
    return dict(isolated_entities=histogram.get("0", 0), mean=float(degrees.mean()),
                std=float(degrees.std()), maximum=int(degrees.max()), degree_histogram=histogram,
                closed_neighborhood_histogram={str(int(degree) + 1): count for degree, count in histogram.items()},
                **dict(zip(("q1", "median", "q3", "p90", "p95", "p99"), quantiles.tolist(), strict=True)))


def configuration(window, tn, tc, scores, names, contexts, left, right, guard, measure, cases, entities, sampler):
    started = perf_counter()
    keys = dict(window=window, threshold_name=tn, threshold_context=tc)
    pure = filter_mask(names, contexts, tn, tc)
    guarded = pure & (guard == 0)
    metrics, neighborhoods = {}, []
    for phase, mask in (("pure", pure), ("guarded", guarded)):
        coverage = measure(mask)
        coverage["within_topic_pairs"] = sum(t["within_topic_pairs"] for t in coverage["topics"].values())
        coverage["cross_topic_pairs"] = sum(coverage["cross_topics"].values())
        degree = np.bincount(np.concatenate((left[mask], right[mask])), minlength=len(entities))
        neighborhoods.append(dict(keys, phase=phase, **neighborhood(degree)))
        metrics.update({f"{phase}_{key}": value for key, value in coverage.items()})
        metrics[f"{phase}_pairs"] = int(mask.sum())
        metrics[f"{phase}_isolated_entities"] = neighborhoods[-1]["isolated_entities"]

    diagnostics = []
    for pos, i, j, old, concept, matches in cases:
        name, context = float(scores["name"][pos]), float(scores[window][pos])
        a, b = entities[i], entities[j]
        rules = guard_reasons(a, b)
        before = bool(filter_mask(name, context, tn, tc))
        diagnostics.append(dict(keys, index=int(old["n"]), old_candidate_id=old["candidate_id"],
                                old_kind=old["kind"], old_cosine=float(old["score"]), old_reason=old["reason"],
                                old_target_concept_id=concept["concept_id"] if concept else "",
                                matching_ref_count=matches, name_cosine=name, context_cosine=context,
                                guard_reasons=rules, pure_pass=before, guarded_pass=before and not rules,
                                **{f"{side}_{key}": value for side, entity in (("a", a), ("b", b))
                                   for key, value in pair_evidence(entity).items()}))
    metrics["guard_rule_hits"] = {rule: int(np.count_nonzero(pure & ((guard & (1 << i)) != 0)))
                                 for i, rule in enumerate(RULES)}
    return dict(metrics=dict(keys, **metrics, sweep_seconds=perf_counter() - started,
                             process_rss_bytes=sampler.process.memory_info().rss,
                             session_peak_rss_bytes=sampler.peak),
                neighborhoods=neighborhoods, diagnostics=diagnostics)


def figures(folder, metrics, neighborhoods, windows, anchor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(exist_ok=True)
    paths = []

    def save(fig, filename):
        path = folder / filename
        tmp = path.with_suffix(".tmp")
        fig.savefig(tmp, format="png", dpi=140)
        plt.close(fig)
        tmp.replace(path)
        paths.append(path)

    fields = ("pairs", "entity_share", "document_share")
    headings = ("EE pairs", "Entity coverage", "Document coverage")
    for window in windows:
        rows = [r for r in metrics if r["window"] == window]
        indexed = {(r["threshold_name"], r["threshold_context"]): r for r in rows}
        names = sorted({n for n, _ in indexed})
        contexts = sorted({c for _, c in indexed})
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for line, phase in zip(axes, ("pure", "guarded"), strict=True):
            for ax, metric, heading in zip(line, fields, headings, strict=True):
                values = [[indexed.get((n, c), {}).get(f"{phase}_{metric}", np.nan) for n in names]
                          for c in contexts]
                image = ax.imshow(values, origin="lower", aspect="auto",
                                  vmin=0, vmax=1 if metric.endswith("share") else None)
                ax.set(xticks=range(len(names)), xticklabels=[f"{n:.2f}" for n in names],
                       yticks=range(len(contexts)), yticklabels=[f"{c:.2f}" for c in contexts],
                       xlabel="Name cosine threshold", ylabel="Context cosine threshold",
                       title=f"{window} · {heading} · {phase}")
                ax.tick_params(axis="x", labelrotation=45)
                fig.colorbar(image, ax=ax)
        save(fig, f"{window}_heatmaps.png")

        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for line, axis, fixed, threshold in zip(axes, ("name", "context"), ("context", "name"), anchor, strict=True):
            subset = sorted((r for r in rows if r[f"threshold_{fixed}"] == threshold),
                            key=lambda r: r[f"threshold_{axis}"])
            for ax, metric, heading in zip(line, fields, headings, strict=True):
                for phase in ("pure", "guarded"):
                    ax.plot([r[f"threshold_{axis}"] for r in subset],
                            [r[f"{phase}_{metric}"] for r in subset], marker=".", label=phase)
                ax.set(xlabel=f"{axis.title()} cosine threshold", title=f"{heading} · {fixed} = {threshold:.2f}")
                ax.legend()
        fig.suptitle(window)
        save(fig, f"{window}_threshold_curves.png")

        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for row in neighborhoods:
            if row["window"] == window and (row["threshold_name"], row["threshold_context"]) == anchor:
                points = sorted((int(key), value) for key, value in row["degree_histogram"].items())
                ax.scatter(*zip(*points, strict=True), s=12, label=row["phase"])
        ax.set(xlabel="EE neighbors", ylabel="Entities (log scale)", yscale="log",
               title=f"{window} · Name / context {anchor[0]:.2f} / {anchor[1]:.2f}")
        ax.legend()
        save(fig, f"{window}_degree_distribution.png")
    return paths


def publish(folder, identity, prepared, rows, anchor, started, sampler, source_commit):
    metrics = [row["metrics"] for row in rows]
    diagnostics = [case for row in rows for case in row["diagnostics"]]
    neighborhoods = [summary for row in rows for summary in row["neighborhoods"]]
    fields = ("pure_pairs", "guarded_pairs", "pure_entity_share", "guarded_entity_share",
              "pure_document_share", "guarded_document_share", "pure_isolated_entities", "guarded_isolated_entities")
    comparison = {}
    for row in metrics:
        tn, tc = key = row["threshold_name"], row["threshold_context"]
        comparison.setdefault(key, dict(threshold_name=tn, threshold_context=tc)).update(
            {f"{row['window']}_{field}": row[field] for field in fields})
    tables = dict(THRESHOLD_SWEEP=metrics, WINDOW_COMPARISON=list(comparison.values()),
                  DIAGNOSTIC_CASES=diagnostics, NEIGHBORHOOD_STATISTICS=neighborhoods)
    files = []
    for name, table in tables.items():
        path = folder / f"{name}.csv"
        atomic_csv(path, table)
        files.append(path)
    files.extend(figures(folder / "figures", metrics, neighborhoods, identity["windows"], anchor))

    report = [
        "# Stage 3A.5.13 · Fine-grained EE threshold sweep", "",
        f"Corpus100. Завершено конфигураций — **{len(rows)}**. Режим — "
        + ("проверочный перебор" if identity["smoke"] else "полный перебор") + ".", "",
        "Пара из разных документов проходит при одновременном достижении обоих порогов. "
        "Столбцы `pure` показывают результаты до защит, `guarded` — после защит Stage 3A.5.12.", "",
        "Покрытие — доля всех канонических сущностей или документов с хотя бы одной EE-парой. "
        "Для тематических долей знаменатель ограничен тематикой. Межтематическая пара учитывается один раз. "
        "Статистика степени включает изолированные сущности, стандартное отклонение рассчитано по всей "
        "совокупности, квантили используют линейную интерполяцию. Размер закрытого окружения — "
        "число непосредственных соседей плюс сама сущность, включая одиночные окружения.", "",
        "Диагностика повторяет канонические EE-сопоставления 20 исторических случаев, "
        "включая исторический референт для записей EC. Диагностика описывает прохождение фильтра.", "",
        "| Окно | Пар до защит | Пар после защит | Сущностей после защит | Документов после защит |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        if (row["threshold_name"], row["threshold_context"]) == anchor:
            report.append("| {window} | {pure_pairs} | {guarded_pairs} | {guarded_entities} | "
                          "{guarded_documents} |".format(**row))
    report.extend(["", f"В таблице пороги name/context — {anchor[0]:.2f}/{anchor[1]:.2f}. "
                   "Графики зависимости показывают сечения по каждому порогу через эту точку. "
                   "Распределения на графиках соответствуют этой точке, а CSV содержит всю сетку.", "",
                   "Время строки измеряется при вычислении метрик этой конфигурации. RSS строки — "
                   "текущий RSS процесса и накопленный в этой сессии максимум замеров по дереву процессов. "
                   "При возобновлении строки сохраняют исходные измерения.", "",
                   "[Перебор](THRESHOLD_SWEEP.csv) · [Окна](WINDOW_COMPARISON.csv) · "
                   "[Диагностика](DIAGNOSTIC_CASES.csv) · [Окружения](NEIGHBORHOOD_STATISTICS.csv) · "
                   "[Манифест](MANIFEST.json)", ""])
    path = folder / "REPORT.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(report), encoding="utf-8")
    tmp.replace(path)
    files.append(path)
    atomic_json(folder / "MANIFEST.json", dict(stage=identity["stage"], identity=identity,
                fingerprint=signature(identity), source_commit=source_commit, status="complete",
                configuration_count=len(rows), diagnostic_rows=len(diagnostics),
                vector_and_score_artifacts=prepared, session_seconds=perf_counter() - started,
                session_peak_rss_bytes=sampler.peak,
                files_sha256={p.relative_to(folder).as_posix(): sha(p) for p in files}))
