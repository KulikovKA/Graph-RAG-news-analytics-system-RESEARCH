"""Dedicated C-RESTART CLI. Deliberately has no training command."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .common import config, file_sha256, fingerprint, read_json, write_json


def extract(cfg, args):
    started, cpu = time.perf_counter(), time.process_time()
    # This command always runs offline, including library initialization.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    from .inputs import load_inputs, load_ontology
    from .validation import validate_prediction
    from .ner import ScientificNER
    from .zeroshot_re import ZeroShotRE
    from .zeroshot_pipeline import extract_document
    import psutil
    import torch

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new/empty; existing predictions are immutable")
    output.mkdir(parents=True, exist_ok=True)
    docs, ontology = load_inputs(cfg, args.subset), load_ontology(cfg)
    if set(cfg["zero_shot_re"]["capabilities"]) != set(ontology["relations"]):
        raise ValueError("Every frozen relation needs an explicit capability declaration")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    ner = ScientificNER(cfg)
    if cfg["zero_shot_re"].get("backend") == "pretrained_nli_zero_shot_re":
        from .zeroshot_nli import EntailmentRE
        re_model = EntailmentRE(cfg, ontology)
    else:
        re_model = ZeroShotRE(cfg)
    ner.model.requires_grad_(False)
    print("Pretrained models strictly loaded on CPU; starting DEV", file=sys.stderr, flush=True)
    for doc in docs:
        prediction = extract_document(doc, ner, re_model, ontology, cfg)
        validate_prediction(prediction)
        write_json(output / f"{doc['document_code']}.json", prediction)
        print(f"{doc['document_code']}: {sum(len(c['entities']) for c in prediction['chunks'])} entities, "
              f"{sum(len(c['raw_proposals']) for c in prediction['chunks'])} raw, "
              f"{sum(len(c['assertions']) for c in prediction['chunks'])} final", file=sys.stderr, flush=True)
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", None)
    method = "Windows GetProcessMemoryInfo PeakWorkingSetSize; process lifetime including imports/load"
    if peak is None:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        method = "getrusage ru_maxrss; process lifetime including imports/load"
    tracked = subprocess.check_output(["git", "ls-files", "src/cgraph", "scripts", "config/c_restart_001.json"], text=True).splitlines()
    # Include new source files before the experiment is committed, excluding the
    # example-bearing annotation module and historical fitted-classifier code.
    source_files = sorted(set(tracked) | {p.as_posix() for p in Path("src/cgraph").glob("zeroshot*.py")}
                          | {"scripts/assert_zero_shot_inference.py", "config/c_restart_001.json"})
    source_files = [p for p in source_files if Path(p).is_file() and Path(p).name not in {"labels.py", "relations.py"}]
    run = dict(architecture_id="C-RESTART-001", mode="end_to_end", subset=args.subset,
               documents=[d["document_code"] for d in docs],
               wall_ms=1000 * (time.perf_counter() - started), cpu_ms=1000 * (time.process_time() - cpu),
               peak_ram_bytes=peak, peak_ram_method=method, rss_final_bytes=mem.rss,
               ner_calls=ner.calls, encoder_forward_calls=ner.encoder_calls, re_batch_calls=re_model.calls,
               re_encoder_forward_calls=re_model.encoder_calls, model_api_calls=ner.calls + re_model.calls, llm_calls=0,
               re_score_summary=re_model.score_summary,
               ner_max_sequence_tokens=ner.max_sequence_tokens, re_max_sequence_tokens=re_model.max_sequence_tokens,
               ner_model=ner.identity, relation_model=re_model.identity,
               ner_load_ms=ner.load_ms, re_load_ms=re_model.load_ms,
               config=cfg, config_sha256=fingerprint(cfg), source_manifest_sha256=file_sha256(cfg["manifest"]),
               dependency_lock_sha256=file_sha256("uv.lock"),
               code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               code_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
               source_code_files_sha256={p: file_sha256(p) for p in source_files},
               prediction_files_sha256={d["document_code"]: file_sha256(output / f"{d['document_code']}.json") for d in docs},
               runtime={p: importlib.metadata.version(p) for p in ("torch", "transformers", "gliner", "glirel", "loguru", "numpy", "tokenizers", "psutil")},
               platform=platform.platform(), python=platform.python_version(), cpu=platform.processor(),
               logical_cpus=psutil.cpu_count(), cpu_threads=cfg["cpu_threads"],
               run_regime="One fresh process, sorted DEV; CPU float32; includes imports/load/hash verification, excludes downloads/evaluation",
               training_performed=False, gold_in_inference=False, semantic_verification="NOT_INDEPENDENTLY_MEASURED")
    write_json(output / "run.json", run)
    return {k: run[k] for k in ("documents", "wall_ms", "cpu_ms", "peak_ram_bytes", "model_api_calls", "llm_calls")}


def main():
    parser = argparse.ArgumentParser(description="Pretrained-only architecture C; DEV only; no training")
    parser.add_argument("--config", default="config/c_restart_001.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download-models")
    for command in ("extract", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--subset", choices=["all", "validation", "train"], default="all",
                       help="Legacy subsets are comparison views, never used for training")
        p.add_argument("--output", required=True)
        if command == "evaluate":
            p.add_argument("--predictions", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    if cfg.get("architecture_id") != "C-RESTART-001" or "relations" in cfg:
        raise ValueError("Dedicated zero-shot configuration required; fitted classifier settings forbidden")
    if args.command == "download-models":
        from .zeroshot_assets import download
        result = download(cfg)
    elif args.command == "extract":
        result = extract(cfg, args)
    else:
        from .zeroshot_evaluation import run_evaluation
        result = run_evaluation(cfg, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
