"""Stage 3A.5.8: воспроизводимый neural/hybrid reranking поверх замороженного B."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import re
import sys
from threading import Event, Thread
from time import perf_counter
import traceback

from frozen_pool import (HERE, ROOT, DEFAULT_SOURCE, DEFAULT_NAMES, ORACLE_BUDGETS, _evidence_error,
                         digest, encode_json, file_sha, load_pools, prepare, read_json, read_record, render)
from integrated_benchmark_reporting import atomic_text, write_json
from model_adapters import Adapter, DEFAULT_MODEL_ROOT, REGISTRY, environment, inspect_model, model_path
from neural_reporting import BASELINE, POLICIES, candidate_oracle, distribution, evaluate, report, top_details
from run_integrated_benchmark import model_output, run_lock

REPRESENTATIONS = ("fact_focused", "document_aware")
SCHEMA = 1
HYBRID_WEIGHT = 0.5
TORCH_SEED = 42
RANKING_RULE = "C Top-5 prefix, then descending C base_score and path_id; new source required in neural Top-5"
TRUNCATION = "Preserve full question and model framing; remove passage tokens from the right; no window reattribution"
DEFAULTS = {"models": list(REGISTRY), "model_root": str(DEFAULT_MODEL_ROOT), "model_path": {},
            "source_run": str(DEFAULT_SOURCE), "entity_names": str(DEFAULT_NAMES), "candidate_budget": 20,
            "representation": ["fact_focused"], "smoke": False, "device": "cpu", "batch_size": 4,
            "threads": 4, "max_length": 512, "warmup_batches": 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Verify frozen inputs and local model files; no inference")
    mode.add_argument("--resume", action="store_true", help="Read unchanged settings from an existing run")
    parser.add_argument("--smoke", action="store_true", default=None, help="Run A01 and approved C03")
    parser.add_argument("--models", choices=REGISTRY, nargs="+", help="Models in execution order (default all six)")
    parser.add_argument("--model-root", type=Path, help=f"Default {DEFAULT_MODEL_ROOT}")
    parser.add_argument("--model-path", action="append", metavar="ID=PATH", help="Repeat for local folder overrides")
    parser.add_argument("--source-run", type=Path, help="Complete 35-question Stage 3A.5.7 run with variant B")
    parser.add_argument("--entity-names", type=Path, help="Override the pinned representations.json file")
    parser.add_argument("--candidate-budget", type=int, choices=ORACLE_BUDGETS, help="Pairs per question at most (default 20)")
    parser.add_argument("--representation", choices=(*REPRESENTATIONS, "both"), help="Default fact_focused")
    parser.add_argument("--output-dir", type=Path, help="New run path or existing path for --resume")
    parser.add_argument("--device", help="cpu (default), cuda, or cuda:N with a compatible Torch build")
    parser.add_argument("--batch-size", type=int, help="Default 4")
    parser.add_argument("--threads", type=int, help="Torch intra-op CPU threads (default 4)")
    parser.add_argument("--max-length", type=int, help="Total sequence tokens, default 512, range 128–512")
    parser.add_argument("--warmup-batches", type=int, help="Batches per load, default 1")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output-dir")
    return args


def settings_for(args, directory):
    settings = dict(DEFAULTS)
    if args.resume:
        settings.update(read_json(directory / "config.json")["settings"])
    for key in settings.keys() - {"model_path", "representation"}:
        value = getattr(args, key)
        if value is not None:
            settings[key] = str(value.resolve()) if isinstance(value, Path) else value
    if args.representation:
        settings["representation"] = list(REPRESENTATIONS) if args.representation == "both" else [args.representation]
    if args.model_path:
        overrides = dict(settings["model_path"])
        for value in args.model_path:
            model, separator, path = value.partition("=")
            if not separator or model not in settings["models"] or not path:
                raise ValueError(f"Invalid --model-path {value!r}; use a selected ID and ID=PATH")
            overrides[model] = str(Path(path).resolve())
        settings["model_path"] = overrides
    if len(settings["models"]) != len(set(settings["models"])):
        raise ValueError("--models must contain distinct IDs")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", settings["device"]):
        raise ValueError("--device must be cpu, cuda or cuda:N")
    if min(settings["batch_size"], settings["threads"]) < 1 or settings["warmup_batches"] < 0:
        raise ValueError("Batch size and threads must be positive; warmup batches cannot be negative")
    if not 128 <= settings["max_length"] <= 512:
        raise ValueError("--max-length must be between 128 and 512")
    return settings


def code_fingerprint():
    files = [*HERE.glob("*.py"), HERE / "MODEL_REGISTRY.json", HERE / "FROZEN_B.json",
             HERE / "requirements-win-cpu.txt"]
    files.extend(HERE.parent / "benchmark_3a5_2" / name for name in (
        "run_integrated_benchmark.py", "integrated_benchmark_reporting.py", "validate_draft.py",
        "run_document_aware.py", "document_aware_reranking.py"))
    files.extend(ROOT / "src/cgraph" / name for name in (
        "__init__.py", "retrieval_types.py", "evidence_extraction.py", "path_reranking.py", "document_ranking.py"))
    return {file.relative_to(ROOT).as_posix(): file_sha(file) for file in sorted(files)
            if not file.name.startswith("test_")}


def checkpoint(directory, identity, *parts, value=None):
    target = directory.joinpath(*parts).with_suffix(".json")
    key = digest([identity, parts])
    if value is not None:
        write_json(target, {"key": key, "sha256": digest(value), "record": value})
        return value
    if target.exists():
        saved = read_record(target, key)
        if saved is None:
            raise ValueError(f"Invalid checkpoint {target}; restore it or start a new run")
        return saved
    return None


def verify_resume(directory, config, models):
    if read_json(directory / "config.json") != config or read_json(directory / "model_fingerprints.json") != models:
        raise ValueError("Resume fingerprint mismatch (settings, inputs, code, runtime or local models)")
    manifest = read_json(directory / "manifest.json")
    if manifest["fingerprint"] != digest(config):
        raise ValueError("Resume manifest fingerprint mismatch")
    return manifest


def rank_percentiles(scores):
    groups = defaultdict(list)
    for pid, score in scores.items():
        groups[score].append(pid)
    result = {}
    preceding = 0
    for score in sorted(groups, reverse=True):
        tied = groups[score]
        value = 1 - (preceding + (len(tied) - 1) / 2) / max(1, len(scores) - 1)
        result.update((pid, value) for pid in tied)
        preceding += len(tied)
    return result


def neural_rankings(pool, scores, ids):
    if set(scores) != set(ids):
        raise ValueError("Scores do not cover the frozen eligible preselection")
    c_order = {pid: i for i, pid in enumerate(ids)}
    relevances = rank_percentiles(scores)
    c_percentiles = {pid: 1 - i / max(1, len(ids) - 1) for pid, i in c_order.items()}
    hybrids = {pid: (1 - HYBRID_WEIGHT) * c_percentiles[pid] + HYBRID_WEIGHT * relevances[pid] for pid in ids}
    paths = {p["path_id"]: p for p in pool.raw["paths"]}
    rankings, details = {}, {}
    for policy, values in zip(POLICIES, (scores, hybrids), strict=True):
        order = sorted(ids, key=lambda pid: (-values[pid], pid))
        selected, seen = [], set()
        for pid in order:
            path, bundle = paths[pid], pool.bundles[pid]
            if _evidence_error(path, bundle, pool.raw["graph_fingerprint"]):
                raise ValueError(f"{pid}: invalid evidence in neural preselection")
            sources = {(item["edge"]["document_id"], source["source_triple_id"])
                       for item in bundle["edges"] if item["edge"]["type"] == "RELEX_REL"
                       for source in item["source_relations"]}
            if len(selected) == 5 or not sources - seen:
                continue
            seen.update(sources)
            selected.append({"rank": len(selected) + 1, "path_id": pid, "path": path, "evidence": bundle,
                             "base_score": pool.c_scores[pid], "score": values[pid],
                             "score_breakdown": {"neural_score": scores[pid], "C_rank_percentile": c_percentiles[pid],
                                                 "neural_rank_percentile": relevances[pid], "hybrid": hybrids[pid]}})
        positions = {item["path_id"]: item["rank"] for item in selected}
        details[policy] = {pid: {"score": values[pid], "score_rank": i,
                                 "selected_rank": positions.get(pid)} for i, pid in enumerate(order, 1)}
        rankings[policy] = selected
    return rankings, details


def base_row(pool, model, policy, representation, items, reference, timing):
    top = top_details(items)
    return {"key": pool.question.key, "group": pool.question.group, "question": pool.question.text,
            "model": model, "policy": policy, "representation": representation, "timing": timing,
            "source_cache": pool.source_files, "gold_documents": sorted(reference["document_contributions"]),
            "gold_evidence_count": len(reference["evidence"]),
            "reference_set_non_exhaustive": reference.get("reference_set_non_exhaustive"),
            "top_5": top, "metrics": evaluate(top, reference)}


def preselect(pool, reference):
    prepare(pool)
    return {
        **base_row(pool, BASELINE, BASELINE, None, pool.baseline, reference,
                   {"preselection_seconds": 0.0, "selection_seconds": pool.baseline_seconds,
                    "rerank_seconds": pool.baseline_seconds}),
        "c_preselection_seconds": pool.preselection_seconds,
        "c_preselection": [{"path_id": pid, "base_score": pool.c_scores[pid]} for pid in pool.order],
        "candidate_oracle": candidate_oracle(pool, reference),
        "raw_bfs_documents": sorted({d for p in pool.raw["paths"] for d in p["document_ids"]}),
        "raw_bfs_paths": len(pool.raw["paths"]),
        "raw_bfs_truncated": pool.raw["truncated"], "raw_bfs_diagnostics": pool.raw["diagnostics"],
    }


@contextmanager
def sample_rss(attempt):
    import psutil

    process = psutil.Process()
    stop = Event()
    attempt["peak_process_rss_mib"] = process.memory_info().rss / 2**20

    def sample():
        while not stop.wait(0.05):
            attempt["peak_process_rss_mib"] = max(attempt["peak_process_rss_mib"], process.memory_info().rss / 2**20)

    worker = Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def evaluate_question(adapter, pool, model, representation, reference, settings, bar):
    ids = pool.order[:settings["candidate_budget"]]
    paths = {p["path_id"]: p for p in pool.raw["paths"]}
    started = perf_counter()
    passages = [render(paths[pid], pool.bundles[pid], pool.names, representation) for pid in ids]
    render_seconds = perf_counter() - started
    scores, lengths = [], []
    tokenization = inference = 0.0
    size = settings["batch_size"]
    bar.reset(total=len(ids))
    bar.set_description_str(f"Pairs | {model} | {pool.question.key} | budget {settings['candidate_budget']}")
    for index in range(0, len(ids), size):
        values, counts, tokenize_sec, inference_sec = adapter.score(pool.question.text, passages[index:index + size])
        scores.extend(values)
        lengths.extend(counts)
        tokenization += tokenize_sec
        inference += inference_sec
        bar.update(len(values))
        bar.set_postfix_str(f"inference {len(scores) / max(inference, 1e-9):.1f} pairs/s")
    started = perf_counter()
    rankings, details = neural_rankings(pool, dict(zip(ids, scores, strict=True)), ids)
    selection = perf_counter() - started
    timing = {"preselection_seconds": pool.preselection_seconds, "render_seconds": render_seconds,
              "tokenization_seconds": tokenization, "inference_seconds": inference, "selection_seconds": selection,
              "rerank_seconds": pool.preselection_seconds + render_seconds + tokenization + inference + selection,
              "pairs": len(ids), "original_tokens": sum(t["original_tokens"] for t in lengths),
              "truncated_tokens": sum(t["truncated_tokens"] for t in lengths)}
    evaluations = [base_row(pool, model, policy, representation, rankings[policy], reference, timing)
                   for policy in POLICIES]
    pairs = [{"path_id": pid, "C_preselection_rank": rank, "C_base_score": pool.c_scores[pid],
              "neural_score": score, "representation_sha256": digest(passage), **counts,
              "ranking": {policy: details[policy][pid] for policy in POLICIES}}
             for rank, (pid, score, counts, passage) in enumerate(zip(ids, scores, lengths, passages, strict=True), 1)]
    return {"evaluations": evaluations, "pairs": pairs, "score_distribution": distribution(scores),
            "max_sequence_length": settings["max_length"]}


def run(directory, config, models, pools, references, args):
    from tqdm import tqdm

    fingerprint = digest(config)
    settings = config["settings"]
    tasks = [(model, rep, pool.question.key) for model in settings["models"]
             for rep in settings["representation"] for pool in pools]
    if args.resume:
        manifest = verify_resume(directory, config, models)
    else:
        write_json(directory / "config.json", config)
        write_json(directory / "model_fingerprints.json", models)
        manifest = {"fingerprint": fingerprint, "started_at": now(), "attempts": [], "model_attempts": []}
    manifest["status"] = "RUNNING"
    attempt = {"started_at": now()}
    manifest["attempts"].append(attempt)
    write_json(directory / "manifest.json", manifest)
    errors_path = directory / "errors.jsonl"
    if not errors_path.is_file():
        atomic_text(errors_path, "")
    log_path = directory / "execution.log"
    def event(message):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{now()} {message}\n")
        tqdm.write(message)

    def error(model, stage, exc, key=None):
        message = {"at": now(), "model": model, "stage": stage, "question": key,
                   "type": type(exc).__name__, "message": str(exc)}
        atomic_text(errors_path, errors_path.read_text(encoding="utf-8") + encode_json(message) + "\n")
        if exc.__traceback__:
            with log_path.open("a", encoding="utf-8") as stream:
                traceback.print_exception(exc, file=stream)
        event(f"{model} | {stage} | {key or '—'} | {message['message']}")

    def score_checkpoint(model, representation, key, value=None):
        return checkpoint(directory, fingerprint, "scores", model, representation, *key.split("/"), value=value)

    baselines, outcomes = {}, {}
    code = 3
    started = perf_counter()
    try:
        with tqdm(total=len(pools), desc="C preselection + oracle", disable=args.no_progress) as staging:
            for pool in pools:
                key = pool.question.key
                saved = checkpoint(directory, fingerprint, "baseline", *key.split("/")) if args.resume else None
                if saved is None:
                    saved = checkpoint(directory, fingerprint, "baseline", *key.split("/"),
                                       value=preselect(pool, references[key]))
                pool.c_scores = {r["path_id"]: r["base_score"] for r in saved["c_preselection"]}
                pool.order = list(pool.c_scores)
                pool.preselection_seconds = saved["c_preselection_seconds"]
                baselines[key] = saved
                staging.update()
        for task in tasks:
            saved = score_checkpoint(*task) if args.resume else None
            if saved is not None:
                outcomes[task] = saved
        event(f"{len(outcomes)}/{len(tasks)} neural blocks restored")
        with (
            tqdm(total=len(tasks), initial=len(outcomes), desc="Overall neural blocks", position=0,
                 disable=args.no_progress) as overall,
            tqdm(total=len(settings["models"]), desc="Models", position=1, disable=args.no_progress) as model_bar,
            tqdm(desc="Questions by representation", position=2, disable=args.no_progress) as questions_bar,
            tqdm(desc="Candidate pairs", position=3, disable=args.no_progress) as pairs_bar,
        ):
            for model in settings["models"]:
                model_bar.set_description_str(f"Models | {model}")
                pending = [t for t in tasks if t[0] == model and t not in outcomes]
                if pending:
                    run_model(model, pending, pools, references, config, models, manifest, outcomes, directory,
                              overall, questions_bar, pairs_bar, score_checkpoint, event, error)
                model_bar.update()
        if len(outcomes) == len(tasks):
            manifest["status"], code = "COMPLETE", 0
    except KeyboardInterrupt:
        manifest["status"], code = "INTERRUPTED", 130
        event("Interrupted. Completed question blocks are retained.")
    except Exception as exc:
        error(BASELINE, "benchmark", exc)
    finally:
        if manifest["status"] == "RUNNING":
            manifest["status"] = "PARTIAL"
        attempt.update(finished_at=now(), wall_seconds=perf_counter() - started, status=manifest["status"])
        manifest.update(completed_blocks=len(outcomes), expected_blocks=len(tasks), completed_baselines=len(baselines))
        write_json(directory / "manifest.json", manifest)
        report(directory, config, manifest, baselines, outcomes)
        event(f"{manifest['status']} | {len(outcomes)}/{len(tasks)} neural blocks | {directory}")
    return code


def run_model(model, pending, pools, references, config, models, manifest, outcomes, directory,
              overall, questions_bar, pairs_bar, save, event, error):
    import torch

    settings = config["settings"]
    if models[model]["status"] != "READY":
        error(model, "preflight", ValueError(models[model]["error"]))
        return
    by_key = {p.question.key: p for p in pools}
    attempt = {"model": model, "started_at": now(), "status": "LOADING"}
    manifest["model_attempts"].append(attempt)
    adapter = None
    try:
        with sample_rss(attempt), model_output(directory / "execution.log"):
            event(f"{model} | loading | {settings['device']} | budget {settings['candidate_budget']}")
            started = perf_counter()
            adapter = Adapter(model, Path(models[model]["path"]), settings["device"], settings["max_length"])
            attempt["model_load_seconds"] = perf_counter() - started
            started = perf_counter()
            warm_pool, rep = next(((by_key[key], rep) for _, rep, key in pending if by_key[key].order),
                                  (None, None))
            if warm_pool and settings["warmup_batches"]:
                paths = {p["path_id"]: p for p in warm_pool.raw["paths"]}
                sample = warm_pool.order[:min(settings["batch_size"], settings["candidate_budget"])]
                passages = [render(paths[pid], warm_pool.bundles[pid], warm_pool.names, rep) for pid in sample]
                for _ in range(settings["warmup_batches"]):
                    adapter.score(warm_pool.question.text, passages)
            attempt.update(warmup_seconds=perf_counter() - started, status="SCORING")
            questions_bar.reset(total=len(pending))
            for task in pending:
                _, representation, key = task
                questions_bar.set_description_str(f"Questions | {model} | {representation} | {key}")
                try:
                    result = evaluate_question(adapter, by_key[key], model, representation, references[key],
                                               settings, pairs_bar)
                    outcomes[task] = save(*task, value=result)
                    overall.update()
                    event(f"{model} | {representation} | {key} | saved {len(result['pairs'])} pairs")
                except Exception as exc:
                    error(model, f"score/{representation}", exc, key)
                    if isinstance(exc, (MemoryError, torch.OutOfMemoryError)):
                        break
                finally:
                    questions_bar.update()
                    write_json(directory / "manifest.json", manifest)
        attempt["status"] = "COMPLETE" if all(t in outcomes for t in pending) else "PARTIAL"
    except KeyboardInterrupt:
        attempt["status"] = "INTERRUPTED"
        raise
    except Exception as exc:
        error(model, attempt["status"].lower(), exc)
        attempt["status"] = "FAILED"
    finally:
        attempt["finished_at"] = now()
        del adapter
        gc.collect()
        if settings["device"].startswith("cuda"):
            torch.cuda.empty_cache()
        write_json(directory / "manifest.json", manifest)


def main(argv=None):
    args = parse_args(argv)
    directory = (args.output_dir or HERE / "runs" /
                 datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")).resolve()
    try:
        settings = settings_for(args, directory)
        runtime, problems = environment()
        print("WARNING: VERIFYING FROZEN POOLS AND HASHING LOCAL MODEL WEIGHTS MAY TAKE TIME.", flush=True)
        pools, references, pool_identity = load_pools(Path(settings["source_run"]), Path(settings["entity_names"]),
                                                      settings["smoke"])
        models = {}
        for model in settings["models"]:
            record = inspect_model(model, model_path(model, Path(settings["model_root"]), settings["model_path"]))
            models[model] = record
            print(f"{model} | {record['status']} | {record.get('error', record['path'])}", flush=True)
        if problems:
            raise ValueError("; ".join(problems) + ". Use the Stage 3A.5.8 environment from README.md.")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        import torch
        from sentence_transformers import CrossEncoder
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        if not all((CrossEncoder, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer)):
            raise ValueError("Could not import model APIs from the pinned environment")
        device = torch.device(settings["device"])
        if device.type == "cuda":
            if not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count():
                raise ValueError("Requested CUDA device is unavailable (CPU lock supports --device cpu)")
            runtime["cuda_device_name"] = torch.cuda.get_device_name(device)
            runtime["cuda_version"] = torch.version.cuda
        if args.preflight:
            ready = all(record["status"] == "READY" for record in models.values())
            print("Preflight " + ("PASS" if ready else "PARTIAL — see local model diagnostics"), flush=True)
            return 0 if ready else 2
        torch.set_num_threads(settings["threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(TORCH_SEED)
        torch.use_deterministic_algorithms(True)
        config = {"stage": "3A.5.8", "schema_version": SCHEMA, "settings": settings, "source": pool_identity,
                  "runtime": runtime, "code_sha256": code_fingerprint(), "model_fingerprint": digest(models),
                  "question_keys": [pool.question.key for pool in pools],
                  "tiers": {model: REGISTRY[model]["tier"] for model in settings["models"]},
                  "hybrid_neural_weight": HYBRID_WEIGHT, "torch_seed": TORCH_SEED,
                  "precision": "float32", "attention": "sdpa", "interop_threads": 1,
                  "ranking_rule": RANKING_RULE, "truncation": TRUNCATION}
        if not args.resume:
            directory.mkdir(parents=True, exist_ok=True)
        with run_lock(directory):
            if not args.resume and any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError("Output directory is not empty; use --resume or choose a new path")
            print("WARNING: CPU NEURAL INFERENCE CAN TAKE TIME. MODELS RUN SEQUENTIALLY.", flush=True)
            return run(directory, config, models, pools, references, args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"Stage 3A.5.8 | {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
