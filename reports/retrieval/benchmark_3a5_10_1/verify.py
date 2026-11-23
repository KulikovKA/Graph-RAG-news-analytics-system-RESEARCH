"""Отдельная проверка холодной и прогретой live-выдачи по сохранённым артефактам."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree

from cgraph.llm_retrieval import PROFILE_SHA, frozen_replay, load_corpus, load_profile


def require(condition, which):
    if not condition:
        raise ValueError(which)


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_context(row, profile, corpus):
    messages = row["messages"]
    prompt = json.loads(messages[1]["content"])
    paths = row["retrieval"]["paths"]
    require(paths == frozen_replay(prompt["question"], profile)["paths"], "C archive parity")
    require(prompt["allowed_citations"] == [s["citation_id"] for s in row["sources"]], "allowed citation IDs")
    require(prompt["sources"] == [{k: v for k, v in s.items() if k != "origins"} for s in row["sources"]],
            "source text and metadata in prompt")
    factual_origins = {(p["path_id"], f["source_triple_id"], w["window_id"], w["document_id"],
                        w["revision_id"], w["start"], w["end"])
                       for p in paths for f in p["facts"] for w in f["windows"]}
    covered_origins = set()
    for source in row["sources"]:
        doc = corpus[source["document_id"]]
        start, end = source["start"], source["end"]
        require(0 <= start < end <= len(doc["text"]) and source["text"] == doc["text"][start:end],
                "source exact offsets")
        require(all(source[k] == doc[k] for k in ("revision_id", "text_sha256", "title", "source_url")),
                "source frozen metadata")
        for origin in source["origins"]:
            covered_origins.add((origin["path_id"], origin["source_triple_id"], origin["window_id"],
                                source["document_id"], source["revision_id"], start, end))
    require(covered_origins == factual_origins, "source triple provenance")
    require(row["retrieval"]["graph_verified"] == {
        k: profile[k] for k in ("graph_fingerprint", "snapshot_sha256")}, "live graph identity")


def verify(baseline, persistent, cli_output=None):
    profile = load_profile()
    corpus = load_corpus(profile)
    require(baseline["mode"] == "baseline" and persistent["mode"] == "persistent", "run modes")
    for field in ("baseline_git", "identity", "prompt_version", "tokenizer_sha256", "psutil_version"):
        require(baseline[field] == persistent[field], field)
    for run in (baseline, persistent):
        require(run["profile_sha256"] == PROFILE_SHA and run["api_calls"] == 0, "profile or API budget")
    require(len(baseline["rows"]) == len(persistent["rows"]) == 3, "expected three requests")
    comparisons = []
    for index, (old, new) in enumerate(zip(baseline["rows"], persistent["rows"], strict=True)):
        for field in ("case", "seeds", "retrieval", "sources", "messages", "input_tokens_with_reserve"):
            require(old[field] == new[field], f"{new['case']} differs on {field}")
        check_context(new, profile, corpus)
        for loader in ("gliner_load", "e5_load"):
            require(len(old["loads_seconds"].get(loader, [])) == 1, "baseline loader count")
            require(len(new["loads_seconds"].get(loader, [])) == int(index == 0), "persistent loader count")
        if index:
            require(new["timings_seconds"]["end_to_end_without_api"] < old["timings_seconds"]["end_to_end_without_api"],
                    "warm query was not faster")
        comparisons.append({
            "case": new["case"], "runtime_phase": "warm" if index else "cold",
            "seed_count": len(new["seeds"]), "seeds_sha256": digest(new["seeds"]),
            "path_ids": [p["path_id"] for p in new["retrieval"]["paths"]],
            "source_count": len(new["sources"]), "sources_sha256": digest(new["sources"]),
            "messages_sha256": digest(new["messages"]),
            "baseline": {k: old[k] for k in ("timings_seconds", "loads_seconds", "peak_rss_bytes")},
            "persistent": {k: new[k] for k in ("timings_seconds", "loads_seconds", "peak_rss_bytes")},
        })
    if cli_output is not None:
        require(len(cli_output) == 2, "CLI output count")
        for index, (output, reference) in enumerate(zip(cli_output, persistent["rows"][:2], strict=True), 1):
            require(output["session_query"] == index and output["error"]["code"] == "missing_GROQ_API_KEY"
                    and output["llm"] is None and output["mode"] == "full_graph_rag", "CLI sequence and API status")
            require(output["retrieval"] == reference["retrieval"] and
                    output["context"]["sources"] == reference["sources"] and
                    output["context"]["messages"] == reference["messages"], "CLI provenance and prompt parity")
            require(("runtime_setup" in output["timings_seconds"]) == (index == 1), "CLI setup count")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("persistent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cli-output", type=Path)
    parser.add_argument("--junit", required=True, type=Path)
    args = parser.parse_args()
    baseline_bytes, persistent_bytes = args.baseline.read_bytes(), args.persistent.read_bytes()
    baseline, persistent = json.loads(baseline_bytes), json.loads(persistent_bytes)
    cli = None if args.cli_output is None else [
        json.loads(path.read_bytes()) for path in sorted(args.cli_output.glob("question-*.json"))]
    comparisons = verify(baseline, persistent, cli)
    suite = ElementTree.parse(args.junit).getroot().find("testsuite").attrib
    result = {
        "status": "PASS", "api_calls": 0, "interactive_cli_parity": "PASS" if cli is not None else "NOT RUN",
        "baseline_git": baseline["baseline_git"], "identity": persistent["identity"],
        "profile_sha256": persistent["profile_sha256"], "tokenizer_sha256": persistent["tokenizer_sha256"],
        "prompt_version": persistent["prompt_version"],
        "baseline_code_sha256": baseline["code_sha256"], "persistent_code_sha256": persistent["code_sha256"],
        "baseline_artifact_sha256": sha256(baseline_bytes).hexdigest(),
        "persistent_artifact_sha256": sha256(persistent_bytes).hexdigest(),
        "pytest": {k: suite[k] for k in ("tests", "failures", "errors", "skipped", "time")},
        "comparisons": comparisons,
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, ensure_ascii=False, allow_nan=False, indent=2)
        out.write("\n")
    print("PASS: three pairs match seeded retrieval, frozen archive, exact evidence and prompt.")


if __name__ == "__main__":
    main()
