"""Stage 3A.5.13 fine-grained EE threshold sweep on frozen Corpus100."""
from argparse import ArgumentParser
from contextlib import closing, nullcontext
from functools import partial
from importlib.metadata import version
from itertools import product
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from time import perf_counter

from filelock import FileLock
import numpy as np
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer
from tqdm import tqdm

import benchmark_contextual_shared_concepts as previous
import contextual_shared_concepts as contextual
from benchmark_contextual_shared_concepts import (
    atomic_array, atomic_json, checked_output, complete, read, require, sha, signature,
)
from contextual_shared_concepts_report import map_diagnostic
from ee_threshold_sweep_report import configuration, publish
from relex_experiment import MemorySampler

STAGE = "3A.5.13"
WINDOWS = {key: contextual.WINDOWS[key] for key in ("S0", "S1", "S2")}
NAME_THRESHOLDS = tuple(i / 100 for i in range(85, 96))
CONTEXT_THRESHOLDS = tuple(i / 100 for i in range(75, 91))
ANCHOR = (.90, .90)
COSINE = dict(dtype="float64", block=128, top_k=None)
OUTPUT = previous.ROOT / "artifacts/benchmark_3a5_13"
PREVIOUS = previous.OUTPUT / "corpus100"
RECIPE = previous.RECIPE | dict(normalization="individual_l2", context_aggregation="mean_float64",
                               dtype="float32", device="cpu", context_template="{name}. {window}")
INFERENCE_SLOTS = 2
inference_slot = nullcontext()


def grid(smoke):
    thresholds = ((.85, .85), ANCHOR) if smoke else tuple(product(NAME_THRESHOLDS, CONTEXT_THRESHOLDS))
    return [(window, tn, tc) for window in WINDOWS for tn, tc in thresholds]


def source_hashes():
    paths = (Path(__file__), Path(__file__).with_name("ee_threshold_sweep_report.py"),
             Path(previous.__file__), Path(contextual.__file__),
             Path(__file__).with_name("contextual_shared_concepts_report.py"),
             Path(__file__).with_name("corpus_graph_inference.py"),
             previous.ROOT / "src/cgraph/semantic_concepts.py")
    return {path.name: sha(path) for path in paths}


def previous_identity(data, ids, code):
    try:
        old = read(PREVIOUS / "identity.json")
        old_recipe = {k: v for k, v in old["embedding"].items() if k not in {"threads", "workers"}}
        hashes = {k.replace("\\", "/"): v for k, v in data["hashes"].items()}
        old_hashes = {k.replace("\\", "/"): v for k, v in old["inputs_sha256"].items()}
        if (old["stage"] == previous.STAGE and old["smoke"] is False
                and old_hashes == hashes and old["entity_ids_sha256"] == signature(ids)
                and old["model_revision"] == previous.REVISION and old["model_files_sha256"] == data["model_files"]
                and old["seed"] == previous.SEED and old_recipe == RECIPE and old["pair_cosine"] == COSINE
                and all(old["windows"][key] == value for key, value in WINDOWS.items())
                and all(old["code_sha256"][key] == code[key] for key in (
                    "benchmark_contextual_shared_concepts.py", "contextual_shared_concepts.py",
                    "corpus_graph_inference.py", "semantic_concepts.py"))
                and read(PREVIOUS / "entities.json") == data["entities"]):
            return signature(old)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def vector_file(folder, label, count):
    return array_file(folder / f"{label}_vectors.npy", (count, RECIPE["dimension"]),
                      np.float32 if label == "name" else np.float64)


def array_file(path, shape, dtype):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    require(array.shape == shape and array.dtype == dtype, f"Invalid array layout: {path}")
    return array


def cached_previous(label, prior, files):
    if not prior:
        return None
    try:
        return checked_output(PREVIOUS / f"{label}.json", prior, files)
    except ValueError:
        return None


def context_vectors(folder, label, identity, data, model, threads, sampler, started):
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    windows_file = f"{label}_windows.json"
    extraction = folder / f"{label}_extract.json"
    if checked_output(extraction, identity, (windows_file,)):
        rows = read(folder / windows_file)
    else:
        geometry = {key: contextual.DocumentWindows(doc["text"]) for key, doc in data["documents"].items()}
        rows, truncated = contextual.extract_windows(data["entities"], data["documents"], geometry,
                                                    tokenizer, label)
        atomic_json(folder / windows_file, rows)
        complete(extraction, identity, (windows_file,), truncated_windows=truncated)
    unique = dict.fromkeys(c["input"] for row in rows for c in row["contexts"])
    # У каждого окна своя база. Транзакция предыдущего этапа сохраняет каждый батч.
    with inference_slot, closing(sqlite3.connect(folder / f"{label}_embeddings.sqlite3")) as db:
        cache = previous.EmbeddingCache(db, model, threads, sampler, started)
        vectors, usage = cache.run(unique, tokenizer, label)
        del cache
    means = np.asarray([contextual.mean_embeddings([vectors[c["input"]] for c in row["contexts"]])
                        for row in tqdm(rows, desc=f"{label} aggregate", unit="entity",
                                        position=WINDOWS[label])])
    return means, usage


def name_vectors(folder, data, model, threads, force):
    entities = data["entities"]
    if not force:
        return np.asarray([data["names"][e["entity_id"]] for e in entities]), dict(source="3A.5.11")

    # Повторяем исходный порядок входов и границы батчей замороженного name-кэша.
    lexical = sorted((c for c in data["shared"]["concepts"] if c["concept_id"].startswith("concept:2g:")),
                     key=lambda c: c["concept_id"])
    names = list(dict.fromkeys([e["name"] for e in entities] + [c["name"] for c in lexical]))
    batch_dir = folder / "name_batches"
    batch_dir.mkdir(exist_ok=True)
    vectors, embed = {}, None
    size = RECIPE["batch"]
    for offset in tqdm(range(0, len(names), size), desc="Name E5", unit="batch"):
        batch = names[offset:offset + size]
        path = batch_dir / f"{offset:05}.npy"
        if path.exists():
            values = array_file(path, (len(batch), RECIPE["dimension"]), np.float32)
        else:
            if embed is None:
                embed = previous.e5_worker(model, threads, previous.SEED, previous.RECIPE)
            values = np.asarray(embed(batch), dtype=np.float32)
        require(values.shape == (len(batch), RECIPE["dimension"]) and np.isfinite(values).all()
                and np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=2e-6),
                "Invalid name embeddings")
        if not path.exists():
            atomic_array(path, values)
        vectors.update(zip(batch, values, strict=True))
    return np.asarray([vectors[e["name"]] for e in entities]), dict(encoded=len(names))


def prepare(label, folder, identity, prior, data, model, threads, force, pairs=None):
    with FileLock(folder / f"{label}.lock", timeout=0), threadpool_limits(limits=threads, user_api="blas"), \
            MemorySampler() as sampler:
        started = perf_counter()
        count = len(data["entities"])
        size = data["counts"]["cross_document_pairs"]
        vector, score = f"{label}_vectors.npy", f"{label}_scores.npy"
        checkpoint = folder / f"{label}.json"
        local = checked_output(checkpoint, identity, (vector, score))
        if local:
            return local | dict(folder=str(folder))

        cached = cached_previous(label, prior, (vector, score))
        if cached:
            vector_file(PREVIOUS, label, count)
            array_file(PREVIOUS / score, (size,), np.float64)
            return dict(folder=str(PREVIOUS), source=previous.STAGE, files={
                key: cached["files"][key] for key in (vector, score)}, fingerprint=prior,
                seconds=perf_counter() - started, peak_rss_bytes=sampler.peak)

        vector_checkpoint = folder / f"{label}_vectors.json"
        saved = checked_output(vector_checkpoint, identity, (vector,))
        if saved:
            values = vector_file(folder, label, count)
        else:
            cached = cached_previous(label, prior, (vector,))
            if cached:
                values, usage = vector_file(PREVIOUS, label, count), dict(source=previous.STAGE)
            elif label == "name":
                values, usage = name_vectors(folder, data, model, threads, force)
            else:
                values, usage = context_vectors(folder, label, identity, data, model, threads, sampler, started)
            atomic_array(folder / vector, values)
            saved = complete(vector_checkpoint, identity, (vector,), usage=usage,
                             vector_seconds=perf_counter() - started, peak_rss_bytes=sampler.peak)
        if pairs is None:
            pairs = contextual.pair_indices([e["document_id"] for e in data["entities"]])
        scores = contextual.cosine_scores(values, *pairs, label, block=COSINE["block"])
        atomic_array(folder / score, scores)
        result = complete(checkpoint, identity, (vector, score), source=STAGE, usage=saved["usage"],
                          vector_seconds=saved["vector_seconds"], seconds=perf_counter() - started,
                          peak_rss_bytes=max(saved["peak_rss_bytes"], sampler.peak))
        return result | dict(folder=str(folder))


def worker_init(lock, slots):
    global inference_slot
    inference_slot = slots
    tqdm.set_lock(lock)


def window_worker(label, *args):
    progress = partial(tqdm, position=WINDOWS[label], leave=False)
    previous.tqdm = contextual.tqdm = progress
    try:
        return label, prepare(label, *args)
    finally:
        previous.tqdm = contextual.tqdm = tqdm


def prepare_windows(folder, identity, prior, data, args, threads):
    worker_data = {key: data[key] for key in ("entities", "documents", "counts")}
    jobs = [(window, folder, identity, prior, worker_data, args.model, threads, args.force_recompute)
            for window in WINDOWS]
    if all(cached_previous(window, prior, (f"{window}_vectors.npy", f"{window}_scores.npy"))
           for window in WINDOWS):
        return dict(window_worker(*job) for job in jobs)
    require(not args.smoke, "Smoke requires all three compatible Stage 3A.5.12 score checkpoints")
    ctx = mp.get_context("spawn")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with ctx.Pool(len(WINDOWS), initializer=worker_init,
                  initargs=(ctx.RLock(), ctx.BoundedSemaphore(INFERENCE_SLOTS))) as pool:
        return dict(job.get() for job in [pool.apply_async(window_worker, job) for job in jobs])


def execute(args, folder, data, sampler, started):
    entities = data["entities"]
    code = source_hashes()
    ids = [e["entity_id"] for e in entities]
    threads = min(args.threads, data["threads"], max(1, (os.cpu_count() or 1) // len(WINDOWS)))
    configurations = grid(args.smoke)
    identity = dict(stage=STAGE, smoke=args.smoke, force_recompute=args.force_recompute,
                    inputs_sha256=data["hashes"], entity_ids_sha256=signature(ids), windows=WINDOWS,
                    configurations=[list(row) for row in configurations], rules=list(contextual.RULES),
                    counts=data["counts"], seed=previous.SEED, model_revision=previous.REVISION,
                    model_files_sha256=data["model_files"], embedding=RECIPE, pair_cosine=COSINE,
                    threads_per_worker=threads, context_workers=len(WINDOWS), inference_slots=INFERENCE_SLOTS,
                    python=sys.version.split()[0], code_sha256=code,
                    versions={p: version(p) for p in ("torch", "numpy", "tokenizers", "transformers",
                                                     "tqdm", "matplotlib", "psutil", "threadpoolctl")})
    state = folder / "identity.json"
    if state.exists():
        require(args.resume and read(state) == identity, "Existing output requires compatible --resume")
    else:
        require(not args.resume, f"No experiment to resume in {folder}")
        atomic_json(state, identity)
    fingerprint = signature(identity)
    prior = None if args.force_recompute else previous_identity(data, ids, code)
    if args.smoke:
        require(prior, "Smoke requires compatible frozen Stage 3A.5.12 artifacts")
    pairs = contextual.pair_indices([e["document_id"] for e in entities])
    left, right = pairs
    name = prepare("name", folder, fingerprint, prior, data, args.model, threads, args.force_recompute, pairs)
    prepared = {"name": name} | prepare_windows(folder, fingerprint, prior, data, args, threads)
    size = len(left)
    scores = {key: array_file(Path(info["folder"]) / f"{key}_scores.npy", (size,), np.float64)
              for key, info in prepared.items()}
    mapped = map_diagnostic(data["diagnostic"], entities, data["shared"], left, right)
    domain = scores["name"] >= NAME_THRESHOLDS[0]
    guard = contextual.pair_guard_codes(entities, left, right, domain)
    names, candidate_left, candidate_right = scores["name"][domain], left[domain], right[domain]
    measure = previous.document_metrics(entities, candidate_left, candidate_right)
    checkpoint_dir = folder / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    sweep_identity = signature([fingerprint, {key: value["files"] for key, value in prepared.items()}])
    rows = []
    current = None
    with tqdm(total=len(configurations), desc="EE threshold sweep", unit="config") as progress:
        for window, tn, tc in configurations:
            if window != current:
                current, context_domain = window, scores[window][domain]
            path = checkpoint_dir / f"{window}_{tn:.2f}_{tc:.2f}.json"
            row = checked_output(path, sweep_identity, ())
            if row is None:
                row = configuration(window, tn, tc, scores, names, context_domain,
                                    candidate_left, candidate_right, guard, measure, mapped, entities, sampler)
                row = complete(path, sweep_identity, (), **row)
            rows.append(row)
            progress.update()
    publish(folder, identity, prepared, rows, ANCHOR, started, sampler,
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=previous.ROOT, text=True).strip())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=previous.MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=2, help="Maximum CPU threads per worker")
    parser.add_argument("--force-recompute", action="store_true", help="Use a separate run and recompute E5")
    parser.add_argument("--resume", action="store_true", help="Continue with the original run options")
    parser.add_argument("--smoke", action="store_true", help="Run six configurations from frozen scores")
    args = parser.parse_args()
    if args.threads < 1 or (args.smoke and args.force_recompute):
        parser.error("Provide positive --threads and use --smoke separately from --force-recompute")
    output = (args.output or OUTPUT / ("smoke" if args.smoke else
                                       "corpus100_recomputed" if args.force_recompute else "corpus100")).resolve()
    require(output != previous.OUTPUT.resolve() and previous.OUTPUT.resolve() not in output.parents,
            "Output must be outside frozen Stage 3A.5.12 artifacts")
    output.mkdir(parents=True, exist_ok=True)
    print("WARNING: CORPUS100 EE SWEEP. CACHE MISSES CAN START LOCAL CPU INFERENCE.", flush=True)
    with FileLock(output / "run.lock", timeout=0), MemorySampler() as sampler:
        started = perf_counter()
        execute(args, output, previous.inputs(args.model), sampler, started)


if __name__ == "__main__":
    main()
