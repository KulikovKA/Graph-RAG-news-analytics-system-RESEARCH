"""Отдельная проверка provenance и результатов smoke без дополнительных API-запросов."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree

from cgraph.llm_retrieval import ROOT


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "benchmark_3a5_9"
SOURCE = ROOT / "reports/ner_v2/research_run2/input_abstracts.jsonl"


def read(path):
    return json.loads(path.read_bytes())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify_index():
    matrix = PRIOR / "results/matrix"
    manifest = read(matrix / "manifest.json")
    content = (matrix / "questions.jsonl").read_bytes()
    require(sha256(content).hexdigest() == manifest["files"]["questions.jsonl"], "matrix checksum")
    fields = ("key", "question_hash", "top", "seconds", "upstream_seconds")
    expected = [{k: row[k] for k in fields} for row in map(json.loads, content.splitlines())
                if row["profile"] == "production-fifo-100-0" and row["key"].split("/")[0] in ("A", "B", "C")]
    require(len(expected) == 25 and read(HERE / "REPLAY_INDEX.json") == expected, "replay projection")
    return {row["question_hash"]: row for row in expected}


def inspect(record, corpus, registry, archived):
    trace = record.get("llm") or {}
    attempts = trace.get("attempts", [])
    usage = trace.get("usage")
    retrieval = record.get("retrieval", {})
    paths = retrieval.get("paths", [])
    context = record.get("context", {})
    sources = context.get("sources", [])
    included = {s["citation_id"]: s for s in sources}
    excluded = context.get("excluded_by_budget", [])
    case = {"status": record["status"], "mode": record.get("mode", "api_minimal"), "usage": usage,
            "request_id": trace.get("request_id"), "attempts": attempts, "error": record.get("error"),
            "timings_seconds": record.get("timings_seconds"), "path_ids": [p["path_id"] for p in paths],
            "citation_ids": [c["citation_id"] for c in record.get("citations", [])],
            "citations": record.get("citations", []),
            "included_documents": list(dict.fromkeys(s["document_id"] for s in sources)),
            "model": trace.get("model"), "generation_settings": trace.get("settings"),
            "tokenizer_sha256": context.get("tokenizer_sha256") or record.get("tokenizer_sha256"),
            "prompt_sha256": context.get("prompt_sha256"), "input_tokens_with_reserve": (
                context.get("input_tokens_with_reserve") or record.get("input_tokens_with_reserve")),
            "excluded_windows": len(excluded)}
    if usage:
        require(usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"], "usage sum")
        require(usage["input_tokens"] <= case["input_tokens_with_reserve"], "token margin")

    if record.get("mode") == "frozen_replay":
        key = sha256(json.dumps(record["question"], ensure_ascii=False).encode()).hexdigest()
        require(case["path_ids"] == archived[key]["top"], "archived rank order")
        require(paths == [{"path_id": path_id, **registry[path_id]} for path_id in archived[key]["top"]],
                "archived provenance")

    if context:
        message = json.loads(context["messages"][1]["content"])
        require(message["question"] == record["question"], "question bytes")
        require(message["allowed_citations"] == list(included), "prompt IDs")
        require(message["sources"] == [{k: v for k, v in s.items() if k != "origins"} for s in sources],
                "prompt evidence")
        expected = set()
        for path in paths:
            edges = {e["key"]: e for e in path["edges"]}
            for fact in path["facts"]:
                edge = edges[fact["edge_key"]]
                require(edge["type"] == "RELEX_REL", "navigation provenance")
                for window in fact["windows"]:
                    expected.add((fact["document_id"], window["revision_id"], window["start"], window["end"],
                                  path["path_id"], fact["source_triple_id"], window["window_id"], edge["key"],
                                  edge["relation_type"], edge["head_key"], edge["tail_key"],
                                  edge["traversed_reverse"]))
        found, window_keys = set(), set()
        for source in [*sources, *excluded]:
            doc = corpus[source["document_id"]]
            start, end = source["start"], source["end"]
            require(0 <= start < end <= len(doc["text"]), "corpus bounds")
            require(source.get("text", doc["text"][start:end]) == doc["text"][start:end], "corpus offsets")
            for field in ("revision_id", "text_sha256", "title", "source_url"):
                require(source[field] == doc[field], "corpus " + field)
            window_key = (source["document_id"], source["revision_id"], start, end)
            require(window_key not in window_keys, "duplicate context window")
            window_keys.add(window_key)
            for origin in source["origins"]:
                found.add((*window_key, *[origin[k] for k in ("path_id", "source_triple_id", "window_id",
                           "edge_key", "relation_type", "head_key", "tail_key", "traversed_reverse")]))
        require(expected == found, "context origin coverage")
        for citation in record.get("citations", []):
            source = included[citation["citation_id"]]
            require(citation == {k: v for k, v in source.items() if k != "text"}, "wrong resolved citation")
        require(set(case["citation_ids"]) == {cid for block in record.get("blocks", []) for cid in block["citations"]},
                "claim citation coverage")
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--earlier-minimal", action="append", type=Path, default=[])
    parser.add_argument("--final-cli", action="append", type=Path, default=[])
    parser.add_argument("--junit", type=Path, required=True)
    args = parser.parse_args()
    archived = verify_index()
    corpus = {row["document_id"]: row for row in map(json.loads, SOURCE.read_bytes().splitlines())}
    registry = read(PRIOR / "results/matrix/evidence.json")
    records = {path.stem: read(path) for path in sorted(args.smoke.glob("*.json"))}
    runs = {name: inspect(value, corpus, registry, archived) for name, value in records.items()}
    older = [inspect(read(path), corpus, registry, archived) for path in args.earlier_minimal]
    parity = None
    if "live_parity" in records and "paths" in records["live_parity"].get("retrieval", {}):
        live, replay = records["live_parity"], records["single"]
        parity = {"ranked_paths_and_provenance": live["retrieval"]["paths"] == replay["retrieval"]["paths"],
                  "identical_prompt": live["context"]["messages"] == replay["context"]["messages"]}
    final_parity = {}
    for path in args.final_cli:
        checked = read(path)
        inspect(checked, corpus, registry, archived)
        reference = records["live_parity" if checked["mode"] == "full_graph_rag" else "single"]
        require((checked.get("llm") is None and checked["error"]["code"] == "missing_GROQ_API_KEY"
                 and checked["retrieval"]["paths"] == reference["retrieval"]["paths"]
                 and checked["context"]["messages"] == reference["context"]["messages"]), "final CLI parity")
        final_parity[checked["mode"]] = "PASS"
    measured = [r["usage"] for r in [*older, *runs.values()] if r["usage"]]
    summary = {
        "retrieval_profile": "3A59-production-C", "frozen_index_rows_verified": len(archived),
        "live_replay_parity": parity, "final_cli_parity_without_api": final_parity,
        "cases": runs, "earlier_preflight_attempts": older,
        "artifact_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(args.smoke.glob("*.json"))},
        "completion_attempts": sum(len(r["attempts"]) for r in [*older, *runs.values()]),
        "usage_reported_attempts": len(measured),
        "reported_usage": {key: sum(u[key] for u in measured)
                           for key in ("input_tokens", "output_tokens", "total_tokens")},
        "junit": {k: v for k, v in ElementTree.parse(args.junit).getroot().find("testsuite").attrib.items()
                  if k in ("tests", "errors", "failures", "skipped", "time", "timestamp")},
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
