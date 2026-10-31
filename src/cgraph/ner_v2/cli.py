"""Preparation is inference-free. Benchmark requires a separate explicit opt-in."""
import argparse
import json

from cgraph.common import read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="NER v2 preparation and future offline benchmark")
    parser.add_argument("--config", default="config/ner_v2.json")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-models", help="Download/hash pinned files; do not instantiate models")
    commands.add_parser("verify-models", help="Verify all offline model files without loading them")
    commands.add_parser("verify-tokenizers", help="Offline tokenizer/dependency check on synthetic text; no model loaded")
    corpus = commands.add_parser("prepare-corpus", help="Replay frozen snapshot, or acquire from OpenAlex")
    corpus.add_argument("--refresh", action="store_true", help="Explicitly replace the frozen sample from API")
    commands.add_parser("validate-input", help="Validate source-only JSONL, without NLP or NER")
    benchmark = commands.add_parser("benchmark", help="FUTURE STAGE ONLY: runs NER inference")
    benchmark.add_argument("--allow-inference", action="store_true", help="Required explicit inference opt-in")
    benchmark.add_argument("--output", default="artifacts/ner_v2/benchmark")
    args = parser.parse_args(argv)
    if args.command == "benchmark" and not args.allow_inference:
        parser.error("benchmark performs NER inference; explicit --allow-inference is required")
    settings = read_json(args.config)
    if args.command in ("prepare-models", "verify-models"):
        from .model_setup import prepare_models, verify_models
        operation = prepare_models if args.command == "prepare-models" else verify_models
        result = operation(settings["model_lock"], settings["model_root"])
        print(json.dumps({"status": result["status"], "verified_files": len(result["files"])}, ensure_ascii=True))
    elif args.command == "verify-tokenizers":
        from .model_setup import verify_tokenizers
        print(json.dumps(verify_tokenizers(settings), ensure_ascii=True))
    elif args.command == "prepare-corpus":
        from .openalex import prepare_corpus
        documents = prepare_corpus(settings["input"], seed=settings["seed"], refresh=args.refresh)
        print(f"Prepared {len(documents)} documents; no inference")
    elif args.command == "validate-input":
        from .io import read_documents
        print(f"Validated {len(read_documents(settings['input']))} source documents")
    else:
        from .benchmark import run
        run(settings, args.output)


if __name__ == "__main__":
    main()
