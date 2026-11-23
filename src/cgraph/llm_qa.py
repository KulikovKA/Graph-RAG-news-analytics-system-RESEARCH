"""CLI для Stage 3A.5.10, запуск из исследовательского checkout."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from .llm_context import (PROMPT_VERSION, AnswerError, ContextError, build_context, validate_answer)
from .llm_groq import GenerationSettings, GroqClient, HarmonyTokenCounter, ProviderError, SECRET_PATTERN
from .llm_retrieval import (ROOT, PROFILE_SHA, RetrievalError, frozen_replay, live_retrieval, load_corpus,
                            load_profile, open_live_runtime, peak_rss_bytes)


def run_question(question, *, replay, neo4j_env=None, entity_cache=None, gliner_path=None,
                 runtime=None, tokenizer_factory=None):
    start = perf_counter()
    timings = {}
    record = {"schema_version": "3A.5.10-v1", "mode": "frozen_replay" if replay else "full_graph_rag",
              "question": question, "status": "ERROR", "timings_seconds": timings,
              "prompt_version": PROMPT_VERSION, "usage": None, "llm": None}
    stage = "configuration"
    key = os.getenv("GROQ_API_KEY", "").strip()
    try:
        if SECRET_PATTERN.search(question) or (key and key in question):
            record["question"] = None
            raise ProviderError("secret_in_question")
        settings = GenerationSettings.from_environment()
        record["model"] = settings.model
        record["generation_settings"] = asdict(settings)
        profile = load_profile() if runtime is None else runtime.profile
        record["retrieval_identity"] = {k: profile[k] for k in (
            "profile_id", "namespace", "corpus_sha256", "document_lock_sha256",
            "graph_fingerprint", "snapshot_sha256")}
        record["retrieval_identity"]["profile_sha256"] = PROFILE_SHA
        corpus = load_corpus(profile)

        stage = "retrieval"
        moment = perf_counter()
        try:
            if replay:
                retrieval = frozen_replay(question, profile)
            elif runtime is not None:
                retrieval = runtime.retrieve(question, timings)
            else:
                retrieval = live_retrieval(question, profile, timings, neo4j_env=neo4j_env,
                                           entity_cache=entity_cache, gliner_path=gliner_path)
        finally:
            timings["retrieval"] = perf_counter() - moment
        record["retrieval"] = retrieval

        stage = "context"
        moment = perf_counter()
        try:
            tokenizer_factory = tokenizer_factory or HarmonyTokenCounter
            count = tokenizer_factory(Path(os.environ["LLM_TOKENIZER_PATH"]),
                                      os.environ["LLM_TOKENIZER_SHA256"], settings.reasoning_effort)
            context = build_context(question, retrieval["paths"], corpus, count, settings.max_input_tokens)
        finally:
            timings["context_building"] = perf_counter() - moment
        prompt_json = json.dumps(context.messages, ensure_ascii=False)
        if SECRET_PATTERN.search(prompt_json) or (key and key in prompt_json):
            raise ProviderError("secret_in_prompt")
        record["context"] = {
            "messages": context.messages, "sources": context.sources, "excluded_by_budget": [
                {k: v for k, v in source.items() if k != "text"} for source in context.excluded],
            "input_tokens_with_reserve": context.input_tokens, "framing_reserve": count.reserve,
            "tokenizer_sha256": count.sha256, "tokenizers_version": version("tokenizers"),
            "prompt_sha256": sha256(prompt_json.encode("utf-8")).hexdigest(),
        }

        stage = "generation"
        if not context.sources:
            code = "context_budget_excluded_all_windows" if context.excluded else "no_factual_windows"
            record.update(status="INSUFFICIENT_EVIDENCE", answer="Недостаточно исходных фрагментов для ответа.",
                          blocks=[], citations=[], limitations=[code], llm={"status": "SKIPPED", "attempts": []})
            timings["llm"] = 0.0
            return record
        moment = perf_counter()
        try:
            client = GroqClient(settings)
            raw, trace = client.complete(context.messages, context.input_tokens)
            record["llm"], record["usage"] = trace, trace["usage"]
        finally:
            timings["llm"] = perf_counter() - moment
        stage = "answer_validation"
        record.update(validate_answer(raw, context))
    except ProviderError as exc:
        record["error"] = {"stage": stage, "code": exc.code}
        if exc.trace:
            record["llm"], record["usage"] = exc.trace, exc.trace["usage"]
    except (AnswerError, ContextError, RetrievalError) as exc:
        record["error"] = {"stage": stage, "code": type(exc).__name__, "detail": str(exc)}
        if isinstance(exc, AnswerError):
            record["llm"]["status"] = "invalid_model_answer"
    except KeyError:
        code = "missing_tokenizer_configuration" if stage == "context" else "missing_required_field"
        record["error"] = {"stage": stage, "code": code}
    except Exception as exc:
        # Сырые исключения SDK и драйвера могут содержать адреса и секреты.
        record["error"] = {"stage": stage, "code": type(exc).__name__}
        if type(exc).__name__ == "Neo4jConnectionError":
            record["error"]["detail"] = "Check the existing Neo4j instance and NEO4J_* settings."
    finally:
        timings["end_to_end"] = perf_counter() - start
    return record


def emit_result(record, output):
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if SECRET_PATTERN.search(encoded) or (key and key in encoded):
        print("ERROR: secret_in_result", file=sys.stderr)
        return 2
    if output is not None:
        try:
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded + "\n")
        except OSError:
            print("ERROR: output_write_failed", file=sys.stderr)
            return 2
    print(encoded, flush=True)
    if record.get("error", {}).get("code") == "missing_GROQ_API_KEY":
        print("Set GROQ_API_KEY in the process environment. See .env.example.", file=sys.stderr)
    return 2 if record["status"] == "ERROR" else 0


def run_interactive(*, neo4j_env, entity_cache, gliner_path, output):
    runtime = None
    tokenizer_factory = lru_cache(maxsize=1)(HarmonyTokenCounter)
    exit_code, index = 0, 0
    with ExitStack() as lifetime:
        try:
            while True:
                if sys.stdin.isatty():
                    print("question (exit to quit)> ", file=sys.stderr, end="", flush=True)
                line = sys.stdin.readline()
                if not line or line.strip().casefold() == "exit":
                    break
                question = line.rstrip("\r\n")
                if not question.strip():
                    continue
                started = perf_counter()
                setup = None
                if runtime is None:
                    try:
                        runtime = lifetime.enter_context(open_live_runtime(
                            load_profile(), neo4j_env=neo4j_env, entity_cache=entity_cache, gliner_path=gliner_path))
                    except Exception as exc:
                        print(f"ERROR: live runtime startup failed ({type(exc).__name__}). "
                              "Check the frozen model assets and NEO4J_* settings.", file=sys.stderr)
                        return 2
                    setup = perf_counter() - started
                index += 1
                record = run_question(question, replay=False, runtime=runtime, tokenizer_factory=tokenizer_factory)
                if setup is not None:
                    timings = record["timings_seconds"]
                    timings["runtime_setup"] = setup
                    timings["end_to_end"] += setup
                    if "retrieval" in timings:
                        timings["retrieval"] += setup
                record["session_query"] = index
                record["peak_rss_bytes"] = peak_rss_bytes()
                path = None if output is None else output / f"question-{index:04d}.json"
                exit_code = max(exit_code, emit_result(record, path))
        except KeyboardInterrupt:
            return 130
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description="Answer a question with the frozen Graph-RAG C profile.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question")
    mode.add_argument("--interactive", action="store_true", help="Keep one live runtime until exit or EOF")
    parser.add_argument("--replay", action="store_true", help="Use exact approved Stage 3A.5.9 archived retrieval")
    parser.add_argument("--neo4j-env", type=Path, help="Existing file with Neo4j connection settings")
    parser.add_argument("--entity-cache", type=Path, default=ROOT / ".cache/semantic_concept_sweep")
    parser.add_argument("--gliner-path", type=Path, default=ROOT / "artifacts/ner_v2/models/gliner25")
    parser.add_argument("--output", type=Path, help="New JSON result file, or an existing directory for --interactive")
    args = parser.parse_args(argv)
    if args.interactive:
        if args.replay:
            parser.error("--interactive requires live retrieval")
        if args.output is not None and not args.output.is_dir():
            parser.error("--interactive --output must name an existing directory")
    else:
        if not args.question.strip():
            parser.error("--question must be nonempty")
        if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
            parser.error("--output must name a new file in an existing directory")
    if not args.replay:
        print("WARNING: LIVE GRAPH PREFLIGHT AND LOCAL GLINER/E5 INFERENCE MAY TAKE TIME.",
              file=sys.stderr, flush=True)
    options = {"neo4j_env": args.neo4j_env, "entity_cache": args.entity_cache, "gliner_path": args.gliner_path}
    if args.interactive:
        return run_interactive(**options, output=args.output)
    record = run_question(args.question, replay=args.replay, **options)
    return emit_result(record, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
