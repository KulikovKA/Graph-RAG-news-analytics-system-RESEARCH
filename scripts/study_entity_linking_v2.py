"""Перебор политик Entity Linking V2 на заранее размеченных вопросах."""
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

from filelock import FileLock
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import contextual_shared_concepts as contextual
import link_corpus100 as corpus
from build_corpus_graph import ROOT, atomic_json, sha
from shared_concepts_v2_pipeline import load_checked, write_checked
from cgraph.entity_linking_v2 import LinkingPolicy, Similarities, merge_mentions, rank_mentions
from cgraph.ner_v2.io import validate_span
from cgraph.query_analysis import E3_HASHES, MODEL_ID, QueryAnalyzer
from cgraph.shared_concepts_v2 import digest

THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
ALPHAS = tuple(i / 100 for i in range(5, 100, 5))
TOP_K = (5, 10, 15, 20)
WORKERS = 3


def grid():
    return [LinkingPolicy(name, alpha, top_k, score)
            for name, alpha, score, top_k in product(THRESHOLDS, ALPHAS, (None, *THRESHOLDS), TOP_K)]


def read_questions(path, documents, entities):
    by_id = {e.entity_id: e.document_id for e in entities}
    splits = {"calibration": [], "holdout": []}
    seen_ids, seen_texts = set(), set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        key, text, split = (row[k] for k in ("id", "question", "split"))
        if not isinstance(text, str) or not isinstance(split, str):
            raise ValueError("Question and split must be strings")
        normalized = " ".join(text.casefold().split())
        if (not isinstance(key, str) or not key.strip() or not normalized or key in seen_ids
                or normalized in seen_texts or split not in splits):
            raise ValueError("Invalid or duplicate question ID/text/split")
        seen_ids.add(key)
        seen_texts.add(normalized)
        target_ids = set()
        for target in row["targets"]:
            entity_id, document_id = target["entity_id"], target["document_id"]
            if entity_id in target_ids or by_id.get(entity_id) != document_id or document_id not in documents:
                raise ValueError(f"Invalid target entity/document for {key}: {entity_id}")
            target_ids.add(entity_id)
            source, ref = documents[document_id], target["evidence"]
            if not ref["revision_id"] or ref["revision_id"] != source["revision_id"]:
                raise ValueError(f"Evidence revision mismatch for {key}: {document_id}")
            validate_span(source["text"], ref["start"], ref["end"], ref["text"])
        if not target_ids or type(row.get("precision_complete", False)) is not bool:
            raise ValueError(f"Missing targets or invalid precision_complete for {key}")
        splits[split].append(row)
    if not all(splits.values()):
        raise ValueError("Both calibration and holdout must contain labeled questions")
    return splits


def prepare(rows, linker, analyzer, cache, split):
    cache.mkdir(exist_ok=True)
    prepared = []
    for row in tqdm(rows, desc=f"{split}: GLiNER / E5 / cosine cache", unit="question"):
        path = cache / (digest(row["question"]) + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                meta = json.loads(saved["meta"].item())
                similarities = Similarities(saved["name"], saved["context"])
            if (meta["query"]["text"] != row["question"]
                    or similarities.name.shape != (len(meta["query"]["mentions"]), len(linker.entities))):
                raise ValueError(f"Cache does not match question or graph generation: {path}")
        else:
            started = perf_counter()
            query = analyzer.analyze(row["question"])
            similarities = linker.similarities(query)
            meta = dict(query=query, inference_ms=(perf_counter() - started) * 1000)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, meta=json.dumps(meta, ensure_ascii=False),
                                    name=similarities.name, context=similarities.context)
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(path)
        prepared.append((row, meta, similarities))
    return prepared


def evaluate(policy, prepared, entities):
    rows = []
    for question, cached, similarities in prepared:
        started = perf_counter()
        ranks = merge_mentions(rank_mentions(similarities, policy))
        rank_ms = (perf_counter() - started) * 1000
        targets = {t["entity_id"] for t in question["targets"]}
        target_docs = {t["document_id"] for t in question["targets"]}
        found = [entities[i] for i, _, _ in ranks]
        ids = [e.entity_id for e in found]
        docs = {e.document_id for e in found}
        first = next((i for i, entity in enumerate(ids[:policy.top_k], 1) if entity in targets), None)
        rows.append(dict(
            entity_recall_at_k=len(targets.intersection(ids)) / len(targets),
            document_recall_at_k=len(target_docs & docs) / len(target_docs),
            mrr_at_k=1 / first if first else 0.0,
            precision_at_k=(sum(entity in targets for entity in ids[:policy.top_k]) / policy.top_k
                            if question.get("precision_complete") else None),
            seeds=len(ids), inference_ms=cached["inference_ms"], rank_ms=rank_ms,
        ))
    latency = [row["inference_ms"] + row["rank_ms"] for row in rows]
    return dict(variant=policy.variant, **asdict(policy), questions=len(rows),
                entity_recall_at_k=mean(r["entity_recall_at_k"] for r in rows),
                document_recall_at_k=mean(r["document_recall_at_k"] for r in rows),
                mrr_at_k=mean(r["mrr_at_k"] for r in rows),
                precision_at_k=(mean(r["precision_at_k"] for r in rows)
                                if all(r["precision_at_k"] is not None for r in rows) else None),
                seeds_total=sum(r["seeds"] for r in rows),
                seeds_mean=mean(r["seeds"] for r in rows),
                no_seeds_rate=mean(r["seeds"] == 0 for r in rows),
                latency_mean_ms=mean(latency), latency_p95_ms=float(np.percentile(latency, 95)),
                cached_rank_mean_ms=mean(r["rank_ms"] for r in rows))


def parameters(result):
    return LinkingPolicy(*(result[key] for key in ("threshold_name", "alpha", "top_k", "threshold_score")))


def evaluate_group(policies, prepared, entities, checkpoint):
    result = [evaluate(policy, prepared, entities) for policy in policies]
    write_checked(checkpoint, {"results": result})
    return result


def sweep(policies, prepared, entities, folder):
    folder.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for policy in policies:
        groups[policy.threshold_name, policy.alpha].append(policy)
    results = {}
    with tqdm(total=len(policies), desc=f"{folder.name}: policies", unit="config") as progress:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            pending = {}
            try:
                for key, group in groups.items():
                    checkpoint = folder / (digest([asdict(policy) for policy in group]) + ".json")
                    if checkpoint.exists():
                        rows = load_checked(checkpoint)["results"]
                        if [parameters(row) for row in rows] != group:
                            raise ValueError(f"Policy checkpoint mismatch: {checkpoint}")
                        results[key] = rows
                        progress.update(len(rows))
                    else:
                        pending[executor.submit(evaluate_group, group, prepared, entities, checkpoint)] = key
                for future in as_completed(pending):
                    rows = future.result()
                    results[pending[future]] = rows
                    progress.update(len(rows))
            except BaseException:
                executor.shutdown(cancel_futures=True)
                raise
    return [row for key in groups for row in results[key]]


def choose(calibration):
    # Качество, затем размер набора seeds. Нестабильный latency не влияет на выбор.
    def order(row):
        return (-row["entity_recall_at_k"], -row["document_recall_at_k"], -row["mrr_at_k"],
                row["seeds_mean"], row["top_k"], -row["threshold_name"],
                -(row["threshold_score"] or 0), row["alpha"])

    selected = [parameters(min((r for r in calibration if r["variant"] == variant), key=order))
                for variant in ("C1", "C2")]
    matched = replace(selected[1], threshold_score=None)
    return selected if matched in selected else [*selected, matched]


def loss_cases(policy, prepared, entities):
    before = replace(policy, threshold_score=None)
    cases = []
    for question, cached, scores in prepared:
        baseline = merge_mentions(rank_mentions(scores, before))
        kept = {i for i, _, _ in merge_mentions(rank_mentions(scores, policy))}
        found_docs = {entities[i].document_id for i in kept}
        targets = {t["entity_id"]: t for t in question["targets"]}
        target_docs = {t["document_id"] for t in targets.values()}
        lost_docs = sorted(({entities[i].document_id for i, _, _ in baseline} & target_docs) - found_docs)
        lost = []
        for i, mention, score in baseline:
            entity = entities[i]
            if i not in kept and entity.entity_id in targets:
                lost.append(dict(target=targets[entity.entity_id], mention=cached["query"]["mentions"][mention],
                                 s_name=float(scores.name[mention, i]), s_context=float(scores.context[mention, i]),
                                 score=score, margin=score - policy.threshold_score))
        if lost or lost_docs:
            cases.append(dict(id=question["id"], question=question["question"],
                              lost_entities=lost, lost_documents=lost_docs,
                              document_evidence=[t for t in targets.values() if t["document_id"] in lost_docs]))
    return dict(policy=asdict(policy), questions_with_loss=len(cases), cases=cases)


def diagnostics(calibration, selected, calibration_data, holdout_data, entities):
    control = {parameters(row): row for row in calibration if row["variant"] == "C1"}
    paired = []
    for threshold in sorted({r["threshold_score"] for r in calibration if r["variant"] == "C2"}):
        pairs = [(control[replace(parameters(row), threshold_score=None)], row)
                 for row in calibration if row["threshold_score"] == threshold]
        paired.append(dict(
            threshold_score=threshold, pairs=len(pairs),
            pairs_with_entity_recall_loss=sum(a["entity_recall_at_k"] > b["entity_recall_at_k"] for a, b in pairs),
            pairs_with_document_recall_loss=sum(a["document_recall_at_k"] > b["document_recall_at_k"] for a, b in pairs),
            mean_seeds_removed=mean(a["seeds_mean"] - b["seeds_mean"] for a, b in pairs),
        ))
    probes = list(dict.fromkeys([selected[1], *(replace(selected[1], threshold_score=r["threshold_score"])
                                                for r in paired)]))
    return dict(calibration_paired=paired, calibration_cases=[loss_cases(p, calibration_data, entities) for p in probes],
                holdout_cases=loss_cases(selected[1], holdout_data, entities))


def save_reports(folder, identity, calibration, holdout, selected, losses):
    rows = [dict(split=split, **r) for split, records in
            (("calibration", calibration), ("holdout", holdout)) for r in records]
    tmp = folder / "results.csv.tmp"
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(folder / "results.csv")
    atomic_json(folder / "results.json", dict(identity=identity, selected=[asdict(p) for p in selected], rows=rows))
    atomic_json(folder / "diagnostics.json", losses)
    report = [
        "# Entity Linking V2", "",
        f"Поколение `{identity['graph']['generation']}`, сущностей {identity['graph']['entity_count']}.",
        f"Calibration: {calibration[0]['questions']} вопросов и {len(calibration)} конфигураций.",
        f"Holdout: {holdout[0]['questions']} вопросов. Повторного подбора на holdout нет.", "",
        "| Профиль на holdout | Name | Alpha | Score | K | Entity R | Doc R | MRR | Seeds / вопрос | Без seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in zip(("C1", "C2", "C1 при тех же Name, Alpha и K"), holdout):
        report.append(f"| {label} | {row['threshold_name']:.2f} | {row['alpha']:.2f} | "
                      f"{row['threshold_score'] or '—'} | {row['top_k']} | {row['entity_recall_at_k']:.3f} | "
                      f"{row['document_recall_at_k']:.3f} | {row['mrr_at_k']:.3f} | "
                      f"{row['seeds_mean']:.2f} | {row['no_seeds_rate']:.3f} |")
    report += [
        "", "Первые два профиля предложены по calibration и требуют утверждения.", "",
        "| Score | Пар C1/C2 на calibration | Потеря Entity Recall | Потеря Document Recall | Seeds убрано / вопрос |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in losses["calibration_paired"]:
        report.append(f"| {row['threshold_score']:.2f} | {row['pairs']} | {row['pairs_with_entity_recall_loss']} | "
                      f"{row['pairs_with_document_recall_loss']} | {row['mean_seeds_removed']:.3f} |")
    report += [
        "", "Примеры с исходными фрагментами сохранены в [diagnostics.json](diagnostics.json).",
        f"На holdout у выбранного C2 вопросов с потерей размеченной сущности или документа: "
        f"{losses['holdout_cases']['questions_with_loss']}.", "",
        "Recall усредняется по вопросам и учитывает объединение seeds из Top-K каждого упоминания. "
        "Сущности объединяются по максимальному score, при равенстве сортируются по ID. "
        "MRR и Precision используют первые K объединённого списка. Для Precision знаменатель K. "
        "Precision считается только при полноте оценок во всём оцениваемом сплите. "
        "Частичная разметка и выбор исходных документов ограничивают переносимость результатов. "
        "Document Recall использует документы-владельцы seeds.", "",
        "Latency складывает время первого GLiNER/E5/cosine и время ранжирования на кэше. "
        "Холодная загрузка моделей учитывается там, где произошла. Latency ранжирования "
        "измерена при трёх потоках перебора и зависит от хоста.", "",
        "[Метрики и данные запуска](results.json) · [Таблица CSV](results.csv)", "",
    ]
    tmp = folder / "REPORT.md.tmp"
    tmp.write_text("\n".join(report), encoding="utf-8")
    tmp.replace(folder / "REPORT.md")


def run(out, identity, splits, linker, analyzer, policies, *, resume):
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "manifest.json"
    with FileLock(out / ".lock", timeout=0):
        if state_path.exists():
            if not resume or load_checked(state_path) != identity:
                raise ValueError("Run exists or its graph, questions, code or runtime changed. Use a new directory.")
        elif resume or any(p.name != ".lock" for p in out.iterdir()):
            raise ValueError("Resume requires an existing matching manifest; a new run requires an empty directory.")
        else:
            write_checked(state_path, identity)

        with threadpool_limits(limits=1, user_api="blas"):
            calibration_data = prepare(splits["calibration"], linker, analyzer, out / "scores", "calibration")
            calibration = sweep(policies, calibration_data, linker.entities, out / "calibration")
            selected = choose(calibration)
            selection_path = out / "selection.json"
            selection = dict(policies=[asdict(policy) for policy in selected])
            if selection_path.exists():
                if load_checked(selection_path) != selection:
                    raise ValueError("Saved calibration selection differs")
            else:
                write_checked(selection_path, selection)
            holdout_data = prepare(splits["holdout"], linker, analyzer, out / "scores", "holdout")
            holdout = sweep(selected, holdout_data, linker.entities, out / "holdout")
            losses = diagnostics(calibration, selected, calibration_data, holdout_data, linker.entities)
        save_reports(out, identity, calibration, holdout, selected, losses)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="JSONL с замороженными calibration/holdout")
    parser.add_argument("--output", type=Path, required=True, help="Каталог нового запуска или его checkpoint")
    parser.add_argument("--graph-root", type=Path, default=corpus.GRAPH_ROOT)
    parser.add_argument("--e5-model", type=Path, default=corpus.DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.graph_root.resolve()):
        parser.error("--output должен находиться вне артефактов графа")
    print("ПРЕДУПРЕЖДЕНИЕ: ДЛИТЕЛЬНЫЙ ПЕРЕБОР 2280 КОНФИГУРАЦИЙ ENTITY LINKING V2.", flush=True)
    inputs, snapshot = corpus.load_generation(args.graph_root)
    questions_sha256 = sha(args.questions)
    splits = read_questions(args.questions, inputs.source["documents"], inputs.entities)
    if sha(args.questions) != questions_sha256:
        raise ValueError("Questions changed during loading")
    linker = corpus.build_linker(inputs, args.e5_model)
    del inputs
    analyzer = QueryAnalyzer()
    policies = grid()
    code = [Path(__file__), Path(corpus.__file__), Path(contextual.__file__)]
    code.extend(ROOT / "src/cgraph" / name for name in (
        "entity_linking_v2.py", "entity_linking.py", "query_analysis.py", "ner_v2/detection.py",
        "ner_v2/io.py", "semantic_concepts.py", "shared_concepts_v2.py",
    ))
    identity = dict(graph=snapshot, questions_sha256=questions_sha256, workers=WORKERS,
                    ner=dict(model=MODEL_ID, files_sha256=E3_HASHES, threshold=analyzer.threshold),
                    code_sha256={p.name: sha(p) for p in code}, python=sys.version.split()[0],
                    versions={pkg: version(pkg) for pkg in (
                        "numpy", "tokenizers", "transformers", "torch", "gliner2", "threadpoolctl"
                    )},
                    grid_sha256=digest([asdict(policy) for policy in policies]),
                    calibration_configurations=len(policies),
                    split_ids={split: [row["id"] for row in rows] for split, rows in splits.items()})
    run(args.output, identity, splits, linker, analyzer, policies, resume=args.resume)


if __name__ == "__main__":
    main()
