"""Поэтапное исследование EC, CC и delta на замороженном Corpus100."""

from argparse import ArgumentParser
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import inspect
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
from time import perf_counter

from filelock import FileLock
from threadpoolctl import threadpool_info, threadpool_limits
from tqdm import tqdm

from build_corpus_graph import sha
from cgraph.shared_concepts_v2 import Catalog, Config, EE, Thresholds, canonical, digest, unit
import contextual_shared_concepts as contextual
from relex_experiment import MemorySampler
from run_shared_concepts_v2 import OUTPUT, slug
from shared_concepts_v2_pipeline import (
    EXPECTED_EE, PREVIOUS_RUN, ROOT, ResourceBudget, ee_edges, frozen_corpus, historical_cases,
    load_checked, source_identity, write_checked,
)
from shared_concepts_v2_study_metrics import measure, score_histogram, shortlist, write_reports
from shared_concepts_v2_study_progress import COMPATIBLE_RUNNERS, WorkerProgress, WorkerRows


BASELINE = (95, 92, 98, 95, 10)
RANGES = (range(90, 101), range(85, 96), range(92, 101), range(90, 101), range(5, 16))
BLAS_THREADS = 1
SCHEMA = 1
_WORKER = None


def point_id(point):
    return "-".join(f"{number:02d}" for number in point)


def point_config(point):
    a, b, c, d, delta = point
    return Config(Thresholds(a / 100, b / 100), Thresholds(c / 100, d / 100), Fraction(delta, 100))


def stage_a(smoke):
    axes = ((95, 92), (95, 93), (96, 92)) if smoke else product(*RANGES[:2])
    return sorted((*ec, *BASELINE[2:]) for ec in axes)


def stage_b(rows, smoke):
    anchors = shortlist(rows, 1 if smoke else 3, 2)
    axes = ((98, 95), (98, 96), (99, 95)) if smoke else tuple(product(*RANGES[2:4]))
    return sorted({(*row["point"][:2], *cc, BASELINE[4]) for row in anchors for cc in axes})


def stage_c(rows, smoke):
    anchors = shortlist(rows, 1 if smoke else 5, 4)
    axes = (9, 10, 11) if smoke else RANGES[4]
    return sorted({(*row["point"][:4], delta) for row in anchors for delta in axes})


def stage_d(rows, smoke):
    centers = [tuple(row["point"]) for row in shortlist(rows, 1 if smoke else 5, 4)]
    points = set(centers)
    for center in centers:
        for index in ((1,) if smoke else range(len(RANGES))):
            for step in (-1, 1):
                value = center[index] + step
                if value in RANGES[index]:
                    points.add((*center[:index], value, *center[index + 1:]))
    return sorted(points)


@dataclass(frozen=True)
class StudyInput:
    entities: list
    metadata: dict
    ee: dict


def prepare_input(frozen, cache_dir):
    entities, metadata = frozen_corpus(frozen)
    by = {e.entity_id: e for e in entities}
    # Ключ зависит только от входа и реализации EE. Изменения EC/CC и отчётов его не инвалидируют.
    key = digest(dict(
        profiles=sorted((e.entity_id, e.fingerprint) for e in entities), recipe=metadata["recipe"],
        ee=EE, guards=contextual.RULES, code=[inspect.getsource(ee_edges), inspect.getsource(unit),
                                             sha(Path(contextual.__file__))], threads=BLAS_THREADS))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with FileLock(path.with_suffix(".lock"), timeout=300):
        if path.exists():
            record = load_checked(path)
            if record["key"] != key:
                raise ValueError("Shared EE cache identity mismatch")
        else:
            with threadpool_limits(limits=BLAS_THREADS):
                edges, metrics = ee_edges(by, set(by))
            record = dict(key=key, edges=edges, metrics=metrics,
                          guarded_cosine=score_histogram([row[2:] for row in edges]))
            assert_ee(record)
            write_checked(path, record)
    assert_ee(record)
    print(f"EE S1: {record['metrics']['guarded_pairs']} guarded pairs · shared cache {key[:12]}", flush=True)
    return StudyInput(entities, metadata, record)


def assert_ee(record):
    actual = {key: record["metrics"][key] for key in EXPECTED_EE}
    if actual != EXPECTED_EE or len(record["edges"]) != EXPECTED_EE["guarded_pairs"]:
        raise ValueError(f"Corpus100 EE regression failed: {actual}")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_archive(path, data):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", compresslevel=1, mtime=0) as zipped:
            zipped.write(canonical(data).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_archive(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def init_worker(data, output, run_id, identity, options, events):
    global _WORKER
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER = (data, output, run_id, identity, options,
               {e.entity_id: e.profile.get("topic", "") for e in data.entities}, events)
    WorkerProgress(events).report("idle")


def trial_path(output, point):
    return output / "trials" / point_id(point)


def validate_row(row, identity, point):
    if row["study"] != identity or tuple(row["point"]) != tuple(point):
        raise ValueError(f"Trial result identity mismatch: {point_id(point)}")
    return row


def run_worker(point):
    data, output, run_id, identity, options, topics, events = _WORKER
    progress = WorkerProgress(events, point)
    progress.report("opening trial")
    try:
        return work_on_trial(point, data, output, run_id, identity, options, topics, progress)
    finally:
        WorkerProgress(events).report("idle")


def work_on_trial(point, data, output, run_id, identity, options, topics, progress):
    folder = trial_path(output, point)
    folder.mkdir(parents=True, exist_ok=True)
    metrics_path = folder / "metrics.json"
    archive = folder / "catalog.json.gz"
    checkpoint = folder / "checkpoint.json.gz"
    with FileLock(folder / ".lock", timeout=300), threadpool_limits(limits=BLAS_THREADS), MemorySampler() as sampler:
        if metrics_path.is_file():
            return validate_row(load_checked(metrics_path), identity, point)
        if archive.is_file():
            # Если процесс завершился между двумя фиксациями, восстанавливаем метрики без повторного расчёта.
            row = validate_row(read_archive(archive)["row"], identity, point)
        else:
            row = evaluate(point, data, run_id, identity, options, topics, sampler, checkpoint, archive, progress)
        row = row | dict(archive_sha256=sha(archive))
        write_checked(metrics_path, row)
        checkpoint.unlink(missing_ok=True)
        return row


def evaluate(point, data, run_id, identity, options, topics, sampler, checkpoint, archive, progress):
    started = perf_counter()
    budget = ResourceBudget(sampler, started, options["max_seconds"], options["max_rss"])
    parent = mp.parent_process()

    def check():
        if parent is not None and not parent.is_alive():
            raise RuntimeError("Study coordinator exited")
        budget.check()

    config = point_config(point)
    namespace = f"research:corpus100:{run_id}:{point_id(point)}"
    catalog = Catalog(data.entities, config, namespace, digest(data.metadata["recipe"]), track_cc=True)
    total_seconds = peak_rss = attempts = 0
    started_at = utc_now()
    if checkpoint.is_file():
        progress.report("restoring checkpoint")
        saved = read_archive(checkpoint)
        validate_row(saved, identity, point)
        catalog.restore(saved["state"])
        catalog.event_seconds = saved["event_seconds"]
        total_seconds, peak_rss, attempts = saved["elapsed_seconds"], saved["peak_rss_bytes"], saved["attempts"]
        started_at = saved["started_at"]
    else:
        catalog.add_edges((a, b) for a, b, *_ in data.ee["edges"])
        catalog.initialize(catalog.entities, progress=lambda groups: progress.track(groups, "EC groups"), check=check)
        catalog.prepare_cc(progress=lambda concepts: progress.track(concepts, "CC index"), check=check)

    def save():
        progress.report(f"checkpoint @{attempts}")
        write_archive(checkpoint, dict(
            study=identity, point=point, attempts=attempts, state=catalog.snapshot(),
            event_seconds=catalog.event_seconds, elapsed_seconds=total_seconds + perf_counter() - started,
            peak_rss_bytes=max(peak_rss, sampler.peak), started_at=started_at))

    if not checkpoint.is_file():
        save()
    try:
        while catalog.queue:
            check()
            if not catalog.merge_next():
                break
            attempts += 1
            if attempts % options["checkpoint_every"] == 0:
                save()
            else:
                progress.report(f"CC tried {attempts} / queue {len(catalog.queue)}", force=False)
        check()
    except RuntimeError:
        save()
        raise
    progress.report("collecting metrics")
    metrics = measure(catalog, topics)
    row = dict(study=identity, point=point, config=config.to_dict(), metrics=metrics,
               elapsed_seconds=total_seconds + perf_counter() - started, peak_rss_bytes=max(peak_rss, sampler.peak),
               worker_pid=os.getpid(), started_at=started_at, completed_at=utc_now(),
               blas_threads=[lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"])
    progress.report("saving result")
    write_archive(archive, dict(row=row, namespace=namespace, config_hash=catalog.config_hash,
                               state=catalog.snapshot(), event_seconds=catalog.event_seconds))
    return row


def research_identity(data, run_id, smoke):
    return dict(
        schema=SCHEMA, run=run_id, smoke=smoke, baseline=BASELINE, ranges=[list(axis) for axis in RANGES],
        ee_source=data.ee["key"], input=data.metadata, input_fingerprint=digest([e.fingerprint for e in data.entities]),
        blas_threads=BLAS_THREADS, algorithm=source_identity(),
        study_code_sha256={name: sha(Path(__file__).with_name(name)) for name in (
            "study_shared_concepts_v2.py", "shared_concepts_v2_study_metrics.py")})


def persist_details(output, data, identity, rows, final):
    details = output / "details"
    details.mkdir(exist_ok=True)
    selected = {tuple(row["point"]) for row in final} | {BASELINE}
    files = {}
    member_ids = set()
    for row in rows:
        point = tuple(row["point"])
        if point not in selected:
            continue
        source = trial_path(output, point) / "catalog.json.gz"
        if sha(source) != row["archive_sha256"]:
            raise ValueError(f"Invalid detailed trial artifact: {point_id(point)}")
        # Подробное состояние уже вычислено воркером. Копируем атомарно, сохраняя оригинал до завершения прогона.
        dest = details / f"{point_id(point)}.json.gz"
        if not dest.exists() or sha(dest) != row["archive_sha256"]:
            tmp = dest.with_suffix(".tmp")
            shutil.copyfile(source, tmp)
            os.replace(tmp, dest)
        # Исторические примеры и полный provenance доступны только для публикуемых каталогов.
        artifact = read_archive(dest)
        catalog = Catalog(data.entities, point_config(point), artifact["namespace"],
                          digest(data.metadata["recipe"]), track_cc=True)
        catalog.restore(artifact["state"])
        member_ids.update(key for concept in catalog.concepts.values() for key in concept.members)
        if data.metadata.get("corpus") == "Corpus100":
            diagnostics = details / f"{point_id(point)}-diagnostics.json"
            write_checked(diagnostics, dict(point=point, study=identity, cases=historical_cases(catalog)))
        files[point_id(point)] = dict(path=dest.name, sha256=sha(dest),
                                     member_sets_sha256=row["metrics"]["catalog_members_sha256"],
                                     baseline=point == BASELINE)
    profiles = details / "profiles.json.gz"
    write_archive(profiles, dict(study=identity, profiles={
        e.entity_id: dict(profile=e.profile, fingerprint=e.fingerprint)
        for e in data.entities if e.entity_id in member_ids
    }))
    write_checked(details / "MANIFEST.json", dict(study=identity, points=files,
                                                 profiles=dict(path=profiles.name, sha256=sha(profiles))))


def prune_workspace(output, rows):
    # Удаляем только временные файлы текущего исследования после фиксации итоговых артефактов.
    for row in rows:
        folder = trial_path(output, row["point"])
        for name in ("catalog.json.gz", "catalog.json.tmp", "checkpoint.json.gz", "checkpoint.json.tmp",
                     "metrics.json.tmp"):
            (folder / name).unlink(missing_ok=True)


def resume_study_identity(saved, definition):
    current = digest(definition)
    if saved["study"] == current:
        return current
    hashes = definition["study_code_sha256"]
    legacy = COMPATIBLE_RUNNERS.get(hashes[Path(__file__).name])
    if legacy:
        previous = definition | dict(study_code_sha256=hashes | {Path(__file__).name: legacy})
        if saved["study"] == digest(previous) == digest(saved["definition"]):
            return saved["study"]
    raise ValueError("Incompatible study resume; use the original inputs or a new run ID")


def run_study(data, output, run_id, *, smoke=False, workers=3, max_seconds=1800, max_rss=2 * 1024**3,
              checkpoint_every=500, on_saved=None):
    if min(workers, max_seconds, max_rss, checkpoint_every) < 1:
        raise ValueError("Workers and resource limits must be positive")
    output.mkdir(parents=True, exist_ok=True)
    definition = research_identity(data, run_id, smoke)
    identity = digest(definition)
    manifest, complete = output / "manifest.json", output / "complete.json"
    with FileLock(output / ".study.lock", timeout=0):
        if manifest.exists():
            identity = resume_study_identity(load_checked(manifest), definition)
        else:
            write_checked(manifest, dict(study=identity, definition=definition, created_at=utc_now(),
                                        git_head=subprocess.check_output(
                                            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
        if complete.exists():
            marker = load_checked(complete)
            for filename, checksum in marker["files_sha256"].items():
                if sha(output / filename) != checksum:
                    raise ValueError(f"Completed output checksum mismatch: {filename}")
            results = load_checked(output / "results.json")
            prune_workspace(output, results["results"])
            return results
        options = dict(max_seconds=max_seconds, max_rss=max_rss, checkpoint_every=checkpoint_every)
        return execute_stages(data, output, run_id, identity, smoke, workers, options, on_saved)


def execute_stages(data, output, run_id, identity, smoke, workers, options, on_saved):
    stage_sizes = (3, 3, 3, 3) if smoke else (121, 297, 55, 55)
    rows, stages = {}, {}
    detail_limit = 2 if smoke else 5
    retained = {BASELINE}

    def register(row):
        nonlocal retained
        point = tuple(row["point"])
        rows[point] = row
        selected = {tuple(item["point"]) for item in shortlist(rows.values(), detail_limit, 4)} | {BASELINE}
        # При добавлении точек прежний проигравший не может попасть в финальный shortlist:
        # в его группе уже есть лучшая точка либо выше стоят достаточно других групп.
        for retired in (retained | {point}) - selected:
            (trial_path(output, retired) / "catalog.json.gz").unlink(missing_ok=True)
        retained = selected

    context = mp.get_context("spawn")
    with ExitStack() as stack:
        events = context.Queue()
        stack.callback(events.close)
        pool = stack.enter_context(context.Pool(workers, initializer=init_worker,
                                                initargs=(data, output, run_id, identity, options, events)))
        total = stack.enter_context(tqdm(total=sum(stage_sizes), desc=f"Corpus100 | {workers} workers",
                                         unit="point", position=0, dynamic_ncols=True))
        worker_rows = stack.enter_context(WorkerRows(events, workers))
        stage_inputs = stage_a(smoke)
        for name, next_stage in (("A", stage_b), ("B", stage_c), ("C", stage_d), ("D", None)):
            stages[name] = stage_inputs
            plan = output / f"stage_{name}.json"
            if plan.is_file():
                if load_checked(plan) != dict(study=identity, points=[list(point) for point in stage_inputs]):
                    raise ValueError(f"Stage {name} plan mismatch")
            else:
                write_checked(plan, dict(study=identity, points=stage_inputs))
            if name == "D":
                total.total = total.n + len(stage_inputs)
                total.refresh()
            with tqdm(total=len(stage_inputs), desc=f"Stage {name}", unit="point", position=1, leave=False,
                      dynamic_ncols=True) as bar:
                pending = []
                for point in stage_inputs:
                    if point not in rows:
                        path = trial_path(output, point) / "metrics.json"
                        if path.exists():
                            register(validate_row(load_checked(path), identity, point))
                        else:
                            pending.append(point)
                            continue
                    bar.update()
                    total.update()
                bar.set_postfix(cached=len(stage_inputs) - len(pending), queued=len(pending))
                results = pool.imap_unordered(run_worker, pending, chunksize=1)
                for row in worker_rows.results(results, len(pending)):
                    register(row)
                    bar.update()
                    total.update()
                    if on_saved is not None:
                        on_saved(row)
                worker_rows.collect()
            if next_stage is not None:
                stage_inputs = next_stage([rows[point] for point in stage_inputs], smoke)
    ordered = [rows[point] for point in sorted(rows)]
    final = shortlist(ordered, detail_limit, 4)
    persist_details(output, data, identity, ordered, final)
    ee = data.ee["metrics"] | dict(cache_key=data.ee["key"], guarded_cosine=data.ee["guarded_cosine"])
    write_reports(output, ordered, stages, final, ee, smoke)
    generated = ["results.json", "results.csv", "REPORT.md", "ec_heatmap.png", "cc_heatmaps.png", "delta.png"]
    generated.extend(path.relative_to(output).as_posix() for path in sorted((output / "details").iterdir()))
    write_checked(output / "complete.json", dict(study=identity, completed_at=utc_now(),
                                                files_sha256={name: sha(output / name) for name in generated}))
    prune_workspace(output, ordered)
    return load_checked(output / "results.json")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=slug, required=True, help="Идентификатор прогона и восстановления")
    parser.add_argument("--smoke", action="store_true", help="Ограниченная проверка всех четырёх этапов")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--frozen-run", type=Path, default=PREVIOUS_RUN)
    parser.add_argument("--max-seconds", type=int, default=1800, help="Лимит на сессию конфигурации")
    parser.add_argument("--max-rss-mib", type=int, default=2048, help="Лимит RSS одного воркера")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Интервал сохранения в попытках CC merge")
    args = parser.parse_args()
    if min(args.workers, args.max_seconds, args.max_rss_mib, args.checkpoint_every) < 1:
        parser.error("Workers and resource limits must be positive")
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[variable] = str(BLAS_THREADS)
    print("ПРЕДУПРЕЖДЕНИЕ: ИССЛЕДОВАНИЕ CORPUS100 МОЖЕТ ЗАНЯТЬ ПРОДОЛЖИТЕЛЬНОЕ ВРЕМЯ.", flush=True)
    try:
        data = prepare_input(args.frozen_run, OUTPUT / "study_ee_cache")
        output = OUTPUT / "studies" / args.run_id
        run_study(data, output, args.run_id, smoke=args.smoke, workers=args.workers,
                  max_seconds=args.max_seconds, max_rss=args.max_rss_mib * 1024**2,
                  checkpoint_every=args.checkpoint_every)
        print(output / "REPORT.md")
    except KeyboardInterrupt:
        print("Прогон прерван. Повторите ту же команду для восстановления.", flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    mp.freeze_support()
    main()
