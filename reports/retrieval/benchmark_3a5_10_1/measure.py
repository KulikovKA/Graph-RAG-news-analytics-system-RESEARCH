"""Парное live-измерение на Neo4j с исходным кодом Stage 3A.5.10."""

import argparse
import builtins
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import types
from unittest.mock import patch

from tqdm import tqdm

from cgraph.entity_linking import E5Encoder, EntityLinker
from cgraph.llm_context import PROMPT_VERSION, build_context
from cgraph.llm_groq import GenerationSettings, HarmonyTokenCounter
from cgraph.llm_retrieval import ROOT, PROFILE_SHA, load_corpus, load_profile


BASE = "85b69302f6d3c0b4a5d3630b46194f7646ddafda"
QUESTIONS = (
    ("A01", "What limits microbial production of limonene?"),
    ("B01", "How can a detector avoid generating unnecessary region proposals when the number of people in drone images varies?"),
    ("C09", "What different roles do optical fibers play in plasma diagnostics and nonlinear optics research?"),
)


def baseline_module():
    source = subprocess.check_output(["git", "show", BASE + ":src/cgraph/llm_retrieval.py"], cwd=ROOT)
    module = types.ModuleType("cgraph._baseline_retrieval")
    module.__file__ = str(ROOT / "src/cgraph/llm_retrieval.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def peak_rss():
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def observe(stack, calls):
    def measured(label, fn):
        def wrapped(*args, **kwargs):
            started = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                calls.setdefault(label, []).append(perf_counter() - started)
        return wrapped

    original_import = builtins.__import__

    def observe_gliner(name, globals=None, locals=None, fromlist=(), level=0):
        if name != "gliner2" or (globals or {}).get("__name__") != "cgraph.query_analysis":
            return original_import(name, globals, locals, fromlist, level)
        builtins.__import__ = original_import
        gliner = measured("gliner_import", original_import)(name, globals, locals, fromlist, level)
        extractor = gliner.AutoExtractor
        stack.enter_context(patch.object(extractor, "from_pretrained",
                                        measured("gliner_load", extractor.from_pretrained)))
        return gliner

    stack.enter_context(patch.object(builtins, "__import__", observe_gliner))
    stack.enter_context(patch.object(E5Encoder, "_load", measured("e5_load", E5Encoder._load)))
    link = EntityLinker.link_seeds

    def capture(self, *args, **kwargs):
        seeds = link(self, *args, **kwargs)
        calls.setdefault("seeds", []).append(seeds)
        return seeds

    stack.enter_context(patch.object(EntityLinker, "link_seeds", capture))


def measure(args):
    settings = GenerationSettings.from_environment()
    profile = load_profile()
    corpus = load_corpus(profile)
    options = {"neo4j_env": args.neo4j_env, "entity_cache": ROOT / ".cache/semantic_concept_sweep",
               "gliner_path": ROOT / "artifacts/ner_v2/models/gliner25"}
    paths = {"baseline": ROOT / "src/cgraph/llm_retrieval.py",
             "persistent": ROOT / "src/cgraph/llm_retrieval.py"}
    baseline = baseline_module() if args.mode == "baseline" else None
    if baseline is None:
        from cgraph.llm_retrieval import open_live_runtime

    def tokenizer():
        return HarmonyTokenCounter(Path(os.environ["LLM_TOKENIZER_PATH"]), os.environ["LLM_TOKENIZER_SHA256"],
                                   settings.reasoning_effort)

    calls = {}
    rows = []
    with ExitStack() as lifetime:
        observe(lifetime, calls)
        counter = None
        for index, (case, question) in enumerate(tqdm(QUESTIONS, desc=args.mode, unit="question")):
            calls.clear()
            started = perf_counter()
            stages = {}
            if baseline is None and index == 0:
                runtime = lifetime.enter_context(open_live_runtime(profile, **options))
                stages["runtime_setup"] = perf_counter() - started
            retrieval_start = perf_counter()
            if baseline is not None:
                result = baseline.live_retrieval(question, profile, stages, **options)
            else:
                result = runtime.retrieve(question, stages)
            stages["retrieval"] = perf_counter() - retrieval_start + stages.get("runtime_setup", 0.0)
            context_start = perf_counter()
            counter = tokenizer() if counter is None or baseline is not None else counter
            context = build_context(question, result["paths"], corpus, counter, settings.max_input_tokens)
            stages["context_building"] = perf_counter() - context_start
            stages["end_to_end_without_api"] = perf_counter() - started
            memory = peak_rss()
            if memory > 8 * 1024**3:
                raise RuntimeError("Peak RSS budget exceeded")
            seeds, = calls.pop("seeds")
            rows.append({"case": case, "timings_seconds": stages, "loads_seconds": dict(calls),
                         "peak_rss_bytes": memory, "seeds": seeds[:profile["production"]["bfs"]["max_seeds"]],
                         "retrieval": result, "sources": context.sources, "messages": context.messages,
                         "input_tokens_with_reserve": context.input_tokens})
    identity = {k: profile[k] for k in ("profile_id", "corpus_sha256", "document_lock_sha256",
                                       "graph_fingerprint", "snapshot_sha256", "versions", "cpu_threads", "seed")}
    code = (subprocess.check_output(["git", "show", BASE + ":src/cgraph/llm_retrieval.py"], cwd=ROOT)
            if baseline is not None else paths[args.mode].read_bytes())
    return {"mode": args.mode, "baseline_git": BASE, "profile_sha256": PROFILE_SHA, "identity": identity,
            "code_sha256": sha256(code).hexdigest(), "psutil_version": version("psutil"),
            "prompt_version": PROMPT_VERSION, "tokenizer_sha256": counter.sha256, "api_calls": 0, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "persistent"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--neo4j-env", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist")
    print("WARNING: LIVE CPU GLINER/E5 AND NEO4J MEASUREMENT. THREE SEQUENTIAL QUESTIONS.", flush=True)
    result = measure(args)
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, ensure_ascii=False, allow_nan=False, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
