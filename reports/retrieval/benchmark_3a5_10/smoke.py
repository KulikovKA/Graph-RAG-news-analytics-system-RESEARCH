"""Последовательная демонстрация, максимум восемь отправок chat/completions."""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from cgraph.llm_groq import GenerationSettings, GroqClient, HarmonyTokenCounter, ProviderError
from cgraph.llm_qa import run_question
from cgraph.llm_retrieval import ROOT


CASES = (
    ("single", True, "What limits microbial production of limonene?"),
    ("multi_hop", True, "How can a detector avoid generating unnecessary region proposals when the number of people in drone images varies?"),
    ("cross_market", True, "How does consumer demand affect prices and pricing decisions in different markets?"),
    ("cross_optics", True, "What different roles do optical fibers play in plasma diagnostics and nonlinear optics research?"),
    ("live_parity", False, "What limits microbial production of limonene?"),
    ("live_missing_detail", False, "For object detection in drone imagery, what was the GPU power consumption in watts?"),
    ("live_empty", False, "What is the serial number of the first lunar teapot?"),
)


def write(directory, name, value):
    with (directory / f"{name}.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--neo4j-env", type=Path)
    args = parser.parse_args(argv)
    os.environ["LLM_MAX_RETRIES"] = "0"
    settings = GenerationSettings.from_environment()
    args.output.mkdir(parents=True, exist_ok=False)
    print("WARNING: SEQUENTIAL LIVE NER/E5 AND UP TO EIGHT GROQ COMPLETION REQUESTS.", flush=True)
    token_count = HarmonyTokenCounter(Path(os.environ["LLM_TOKENIZER_PATH"]),
                                      os.environ["LLM_TOKENIZER_SHA256"], settings.reasoning_effort)
    messages = [{"role": "system", "content": "Return a JSON object."},
                {"role": "user", "content": 'Return exactly {"ok":true}.'}]
    try:
        raw, trace = GroqClient(settings).complete(messages, token_count(messages))
        minimal = {"status": "PASS" if json.loads(raw) == {"ok": True} else "FAIL", "llm": trace,
                   "messages": messages, "input_tokens_with_reserve": token_count(messages),
                   "tokenizer_sha256": token_count.sha256}
    except ProviderError as error:
        minimal = {"status": "BLOCKED", "error": error.code, "llm": error.trace}
    except ValueError:
        minimal = {"status": "FAIL", "error": "invalid_smoke_json", "llm": trace}
    write(args.output, "minimal", minimal)
    if minimal["status"] != "PASS":
        return 2
    failed = False
    for name, replay, question in tqdm(CASES, desc="3A.5.10", unit="case"):
        result = run_question(question, replay=replay, neo4j_env=args.neo4j_env,
                              entity_cache=ROOT / ".cache/semantic_concept_sweep",
                              gliner_path=ROOT / "artifacts/ner_v2/models/gliner25")
        write(args.output, name, result)
        failed |= result["status"] == "ERROR"
        error = result.get("error", {}).get("code")
        if error in {"authentication_error", "unexpected_provider_model", "model_or_endpoint_not_found"}:
            break
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
