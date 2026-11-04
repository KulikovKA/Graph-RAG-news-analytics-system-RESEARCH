"""CLI entry point. Only the explicit qwen command may call a generation endpoint."""
import argparse
import hashlib
import json
import sys
import time
import uuid
import zipfile
from contextlib import ExitStack
from pathlib import Path

from tqdm import tqdm

from .analysis import evaluate, hybrid, summarize
from .completeness import check_completeness
from .dataset import gliner_rows, load_frozen, prepare
from .ollama import OllamaClient, hardware_info, validate_config
from .runner import dry_run, run_qwen
from .servers import ManagedServers, lock_identity, make_client, server_urls
from .storage import exclusive_lock, now, read_json, write_json, write_jsonl


def parser():
    p = argparse.ArgumentParser(description="ENTITY-TYPING-V2-001: saved E3 + local Qwen3.5-4B")
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "gliner", "qwen", "hybrid", "summarize", "export-results", "evaluate", "check-completeness", "night"):
        c = commands.add_parser(name)
        c.add_argument("--config", default="config/entity_typing_v2.json")
        c.add_argument("--dataset", choices=("pilot", "full"), help="Default: full (1211 candidates), from config")
        c.add_argument("--output", help="Run root; full is stored in its full/ subdirectory")
        c.add_argument("--dry-run", action="store_true", help="Metadata and prompt planning only; never load/generate")
        c.add_argument("--server-urls", nargs="+", help="One or two distinct Ollama endpoints")
        for flag, kind in (("num-thread", int), ("num-predict", int), ("num-predict-thinking", int),
                           ("batch-size", int), ("max-model-context", int), ("runtime-context-limit", int),
                           ("max-retries", int), ("timeout-seconds", int), ("model", str), ("base-url", str)):
            c.add_argument("--" + flag, type=kind)
        c.add_argument("--strict-max-context", action="store_true", default=None)
        if name == "qwen":
            c.add_argument("--mode", choices=("no-thinking", "thinking"), default="no-thinking")
            c.add_argument("--resume", action="store_true")
        if name in ("export-results", "night"):
            c.add_argument("--zip", dest="zip_path", help="Destination archive, outside the run directory")
        if name == "evaluate":
            c.add_argument("--gold", required=True, help="Independent HUMAN_REVIEWED JSONL")
    return p


def export_results(output, destination, root, before_manifest=None):
    if destination.resolve().is_relative_to(output.resolve()):
        raise ValueError("Place ZIP outside the run directory")
    allowed = {".json", ".jsonl", ".csv", ".md", ".log", ".txt", ".xml"}
    paths = [(p, p.relative_to(output).as_posix()) for p in sorted(output.rglob("*"))
             if p.is_file() and p.suffix in allowed and not any(part.startswith(".") for part in p.relative_to(output).parts)
             and not (output.name != "full" and p.relative_to(output).parts[0] == "full")]
    reproduction = [root / "config/entity_typing_v2.json", root / "config/entity_typing_v2_requirements.txt",
                    root / "tests/test_entity_typing_v2.py", root / "tests/test_entity_typing_v2_servers.py",
                    root / "scripts/run_entity_typing_v2.py",
                    root / "scripts/run_entity_typing_v2.ps1", root / "run_all_night.ps1",
                    root / "docs/ENTITY_TYPING_V2_README.md"]
    reproduction += sorted((root / "src/cgraph/entity_typing_v2").glob("*.py"))
    paths += [(p, "reproduction/" + p.relative_to(root).as_posix()) for p in reproduction if p.exists()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    checksums = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, name in tqdm(paths, desc="Export results", unit="file"):
            if before_manifest is not None and name == "manifest.json":
                continue
            data = path.read_bytes()
            archive.writestr(name, data)
            checksums[name] = hashlib.sha256(data).hexdigest()
        if before_manifest is not None:
            # Night orchestration commits final status/timings after copying payloads.
            before_manifest()
            data = (output / "manifest.json").read_bytes()
            archive.writestr("manifest.json", data)
            checksums["manifest.json"] = hashlib.sha256(data).hexdigest()
        archive.writestr("archive_sha256_manifest.json", json.dumps(checksums, indent=2) + "\n")
    temporary.replace(destination)
    tqdm.write(f"Exported {destination.resolve()}")


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config_path = Path(args.config)
    config = read_json(config_path if config_path.is_absolute() else root / config_path)
    args.dataset = args.dataset or config.get("default_dataset", "full")
    for key in ("num_thread", "num_predict", "num_predict_thinking", "batch_size", "max_model_context",
                "runtime_context_limit", "strict_max_context", "max_retries", "timeout_seconds", "model", "base_url", "server_urls"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    validate_config(config)
    output = Path(args.output or config["output"])
    output = output if output.is_absolute() else root / output
    if args.dataset == "full":
        output /= "full"
    started = time.monotonic()
    try:
        # A server-wide client lock prevents ON/OFF overlap across datasets/output roots.
        # It is also used for dry-run only to avoid reading concurrently changing artifacts.
        with ExitStack() as stack:
            for identity in sorted({lock_identity(url) for url in server_urls(config)}):
                lock_path = root / ".cache/entity_typing_v2" / (hashlib.sha256(identity.encode()).hexdigest()[:16] + ".lock")
                stack.enter_context(exclusive_lock(lock_path))
            stack.enter_context(exclusive_lock(output / ".run.lock"))
            if args.command in ("prepare", "night"):
                manifest, candidates = prepare(config, root, output, args.dataset)
                write_json(output / "hardware.json", hardware_info())
                tqdm.write(f"Frozen {args.dataset}: {len(candidates)} candidates -> {output}")
            else:
                manifest, candidates, _ = load_frozen(output)
            if manifest["frozen"]["dataset"] != args.dataset:
                raise ValueError("Dataset flag disagrees with frozen manifest")
            if args.dry_run:
                report = dry_run(output, config, make_client(config, dry_run=True))
                return 0 if report.get("preflight_ok", report["model_available"]) else 2
            if args.command == "night":
                from .night import run_night
                destination = Path(args.zip_path) if args.zip_path else output.parent / (args.dataset + "_results.zip")
                with ManagedServers(config, output):
                    return run_night(output, root, config, args.dataset, make_client(config), destination, export_results)
            if args.command == "gliner":
                write_jsonl(output / "T1_gliner.jsonl", tqdm(gliner_rows(candidates), desc="T1 saved GLiNER", unit="candidate"))
            elif args.command == "qwen":
                try:
                    if args.resume:
                        from .runner import completed_mode
                        complete = completed_mode(output, config, args.mode)
                    else:
                        complete = None
                    if complete is None:
                        with ManagedServers(config, output):
                            run_qwen(output, config, make_client(config), args.mode, args.resume)
                finally:
                    # Make partial-run diagnostics reviewable even after Ctrl+C/retry exhaustion.
                    write_json(output / "command_sessions" / (uuid.uuid4().hex + ".json"),
                               {"command": args.command, "finished_at": now(), "wall_seconds": time.monotonic() - started})
                    summarize(output)
            elif args.command == "hybrid":
                hybrid(output, config)
            elif args.command == "evaluate":
                evaluate(output, Path(args.gold))
            elif args.command == "check-completeness":
                report = check_completeness(output)
                tqdm.write(f"Completeness: {report['status']}; IDs accounted for={report['all_ids_accounted_for']}")
                return 0 if report["success"] else 3
            elif args.command == "export-results":
                summarize(output)
                destination = Path(args.zip_path) if args.zip_path else output.parent / (
                    "full_results.zip" if args.dataset == "full" else output.name + "_pilot_results.zip")
                export_results(output, destination, root)
            if args.command in ("gliner", "hybrid", "summarize"):
                summarize(output)
            # Record measured active command time separately from pause-inclusive elapsed dates.
            if args.command != "qwen":
                write_json(output / "command_sessions" / (uuid.uuid4().hex + ".json"),
                           {"command": args.command, "finished_at": now(), "wall_seconds": time.monotonic() - started})
        return 0
    except KeyboardInterrupt:
        tqdm.write("Stopped. Completed batch decisions are saved; continue with --resume.")
        return 130
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
