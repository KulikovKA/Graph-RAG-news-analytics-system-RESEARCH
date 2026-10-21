import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from .common import config, file_sha256, fingerprint, read_json, write_json


def inference_code_hashes():
    # Even provenance collection must respect the annotation-module boundary.
    return {p.as_posix(): file_sha256(p) for p in sorted(Path("src/cgraph").glob("*.py")) if p.name != "labels.py"}


def main():
    parser = argparse.ArgumentParser(description="C-001: strictly non-generative DEV-only graph extraction")
    parser.add_argument("--config", default="config/c001.json")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import-dev")
    p.add_argument("--source-root", required=True)
    sub.add_parser("fetch-dev")
    sub.add_parser("download-models")
    sub.add_parser("train")
    for name in ("extract", "diagnose-re"):
        p = sub.add_parser(name)
        p.add_argument("--subset", choices=["train", "validation", "all"], default="validation")
        p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--subset", choices=["train", "validation", "all"], default="validation")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.command == "import-dev":
        from .prepare import import_dev
        result = import_dev(cfg, args.source_root)
    elif args.command == "fetch-dev":
        from .prepare import fetch_dev
        result = fetch_dev(cfg)
    elif args.command == "download-models":
        from .model_setup import download_models
        result = download_models(cfg)
    elif args.command == "train":
        from .inputs import load_inputs, load_ontology
        from .labels import load_labels
        from .relations import train
        docs = load_inputs(cfg, "train")
        partition = read_json(cfg["partition"])
        if set(partition["train"]) & set(partition["validation"]):
            raise ValueError("Train/validation overlap")
        result = train(cfg, load_ontology(cfg), docs, load_labels(cfg, docs))
    elif args.command in ("extract", "diagnose-re"):
        result = run_extraction(cfg, args)
    else:
        result = run_evaluation(cfg, args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


def run_extraction(cfg, args):
    started, cpu = time.perf_counter(), time.process_time()
    import psutil
    from .inputs import load_inputs, load_ontology
    from .pipeline import extract_document, process_chunk
    from .relations import RelationClassifier
    from .validation import validate_prediction

    docs, ontology = load_inputs(cfg, args.subset), load_ontology(cfg)
    classifier = RelationClassifier(cfg, ontology)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new/empty; runs are immutable")
    output.mkdir(parents=True, exist_ok=True)
    mode = "end_to_end" if args.command == "extract" else "gold_entities_re_diagnostic"
    if args.command == "extract":
        # No labels module is imported in this branch or any inference dependency.
        from .ner import ScientificNER
        ner = ScientificNER(cfg)
    else:
        from .labels import gold_entities, load_labels
        labels = load_labels(cfg, docs)
        ner = None
    for doc in docs:
        if ner is not None:
            prediction = extract_document(doc, ner, classifier, ontology, cfg)
        else:
            t, ct = time.perf_counter(), time.process_time()
            before = classifier.calls
            chunks = [process_chunk(doc, c, gold_entities(labels[doc["document_code"]], c), classifier, ontology, cfg) for c in doc["chunks"]]
            prediction = dict(schema_version="cgraph-v1", mode=mode, document_code=doc["document_code"],
                              document_id=doc["document_id"], revision_id=doc["revision_id"], split="DEV", chunks=chunks,
                              execution=dict(wall_ms=(time.perf_counter() - t) * 1000, cpu_ms=(time.process_time() - ct) * 1000,
                                             ner_calls=0, re_calls=classifier.calls - before, llm_calls=0))
        validate_prediction(prediction)
        write_json(output / f"{doc['document_code']}.json", prediction)
        print(f"{doc['document_code']}: {sum(len(c['entities']) for c in prediction['chunks'])} mentions, "
              f"{sum(len(c['assertions']) for c in prediction['chunks'])} accepted", file=sys.stderr, flush=True)
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", None)
    peak_method = "Windows GetProcessMemoryInfo PeakWorkingSetSize, process lifetime incl. imports/model load"
    if peak is None:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        peak_method = "Linux getrusage ru_maxrss, process lifetime incl. imports/model load"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = None
    run = dict(mode=mode, subset=args.subset, documents=[d["document_code"] for d in docs],
               wall_ms=(time.perf_counter() - started) * 1000, cpu_ms=(time.process_time() - cpu) * 1000,
               peak_ram_bytes=peak, peak_ram_method=peak_method, rss_final_bytes=mem.rss,
               llm_calls=0, ner_calls=ner.calls if ner else 0, encoder_forward_calls=ner.encoder_calls if ner else 0,
               max_sequence_tokens=ner.max_sequence_tokens if ner else None, re_batch_calls=classifier.calls,
               ner_model=ner.identity if ner else None, ner_load_ms=ner.load_ms if ner else None,
               relation_model=classifier.metadata, config=cfg, config_sha256=fingerprint(cfg),
               source_manifest_sha256=fingerprint(read_json(cfg["manifest"])), code_commit=commit,
               source_code_files_sha256=inference_code_hashes(),
               dependency_lock_sha256=file_sha256("uv.lock"),
               code_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
               runtime={p: importlib.metadata.version(p) for p in ["torch", "gliner", "transformers", "scikit-learn", "numpy", "tokenizers", "psutil"]},
               platform=platform.platform(), python=platform.python_version(), cpu=platform.processor(),
               logical_cpus=psutil.cpu_count(), cpu_threads=cfg["cpu_threads"],
               run_regime="One fresh process; local model files; includes dependency imports after CLI dispatch and model loading; document order sorted; later documents warm")
    write_json(output / "run.json", run)
    return {k: run[k] for k in ["mode", "documents", "wall_ms", "cpu_ms", "peak_ram_bytes", "ner_calls", "encoder_forward_calls", "re_batch_calls", "llm_calls"]}


def run_evaluation(cfg, args):
    from .inputs import load_inputs, load_ontology
    from .labels import load_labels
    from .evaluation import evaluate
    from .validation import validate_chunk, validate_prediction

    docs = load_inputs(cfg, args.subset)
    labels, ontology = load_labels(cfg, docs), load_ontology(cfg)
    directory = Path(args.predictions)
    run = read_json(directory / "run.json")
    codes = {d["document_code"] for d in docs}
    if not codes <= set(run["documents"]):
        raise ValueError("Requested documents missing from prediction run")
    if args.subset == "validation" and codes & set(run["relation_model"]["train_documents"]):
        raise ValueError("Model was trained on validation documents")
    predictions = [read_json(directory / f"{d['document_code']}.json") for d in docs]
    for doc, prediction in zip(docs, predictions, strict=True):
        validate_prediction(prediction)
        if prediction["mode"] != run["mode"]:
            raise ValueError("Diagnostic and end-to-end predictions cannot be mixed")
        lookup = {c["chunk_id"]: c for c in doc["chunks"]}
        for chunk in prediction["chunks"]:
            validate_chunk(doc, lookup[chunk["chunk_id"]], chunk, ontology)
    result = evaluate(predictions, labels)
    result.update(mode=run["mode"], subset=args.subset, partition=read_json(cfg["partition"]),
                  prediction_run_sha256=file_sha256(directory / "run.json"),
                  prediction_files_sha256={d["document_code"]: file_sha256(directory / f"{d['document_code']}.json") for d in docs},
                  performance={k: run[k] for k in ["documents", "wall_ms", "cpu_ms", "peak_ram_bytes", "peak_ram_method", "llm_calls", "ner_calls", "encoder_forward_calls", "re_batch_calls"]},
                  performance_scope="Full prediction run listed above; per-document times are in performance_documents",
                  performance_documents={p["document_code"]: dict(p["execution"], stage_ms={k: sum(c["stage_ms"].get(k, 0) for c in p["chunks"]) for k in {k for c in p["chunks"] for k in c["stage_ms"]}}) for p in predictions},
                  independent_of_training_documents=args.subset == "validation",
                  inferential_status="Descriptive DEV validation only; no confirmatory holdout claim")
    if run["mode"] == "gold_entities_re_diagnostic":
        # Entity accuracy here is tautological and is not a model-quality result.
        for scope in [result["micro"], result["macro_document"], *result["documents"].values()]:
            scope.pop("entity_span", None)
            scope.pop("entity_typed", None)
    write_json(args.output, result)
    return result["micro"]


if __name__ == "__main__":
    main()
