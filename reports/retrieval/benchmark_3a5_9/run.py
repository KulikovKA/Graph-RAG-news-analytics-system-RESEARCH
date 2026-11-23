"""Stage 3A.5.9, парный replay и однофакторные эксперименты."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import psutil
from tqdm import tqdm

from inputs import (ANCHOR, CachedLinker, DEFAULT_SOURCE, EXTRA, FINGERPRINT, HERE, ROOT, digest, extras,
                    file_sha, inputs, original, read_json, require)
from losses import diagnose
from neural_reporting import evaluate, percentile, quality, top_details
from profiles import direct_fact_index, diverse, lexical_index, preselect, run_pool
from cgraph.entity_linking import _expansion_searches

BASE = ("production", "fifo", 100, 0)
B = ("k5", "fifo", 200, 0)
LINK = ["dedup", "round", "adaptive", "clauses", "k5", "k20", "rescue"]
TRAVERSAL = ["fair", "productive"]
PREFIXES = (20, 50, 100)


def dump(path, value):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def label(profile):
    return "-".join(map(str, profile))


def matrix():
    return [BASE, B, *( (m, *BASE[1:]) for m in LINK ),
            *( (BASE[0], t, *BASE[2:]) for t in TRAVERSAL ),
            *( (*p[:3], n) for p in (BASE, B) for n in PREFIXES )]


def compact(item, evidence):
    facts = []
    for fact in item["facts"]:
        facts.append({**fact, "windows": [{k: v for k, v in w.items() if k not in ("source_url", "text")}
                                           for w in fact["windows"]]})
    pid = item["path_id"]
    proof = {k: item[k] for k in ("edges", "document_ids", "factual_document_ids")}
    proof["facts"] = facts
    require(evidence.setdefault(pid, proof) == proof, "Path ID collision")
    return pid


def measure(key, profile, pool, top, ref, times, evidence, query, analysis_seconds):
    ranked = top_details(top)
    metrics = evaluate(ranked, ref)
    return dict(key=key, profile=label(profile), question_hash=digest(query["text"]),
                top=[compact(p, evidence) for p in ranked], metrics=metrics,
                gold_documents=sorted(ref["document_contributions"]), gold_evidence_count=len(ref["evidence"]),
                seeds=[s["entity_id"] for s in pool["seeds"]], seed_sha256=digest(pool["seeds"]),
                seed_count_before_cap=len(pool["uncapped"]), fallback_spans=pool["fallback_spans"],
                paths=len(pool["raw"]["paths"]), raw_sha256=digest([p["path_id"] for p in pool["raw"]["paths"]]),
                diagnostics=pool["raw"]["diagnostics"], seconds=times, replay_seconds=sum(times.values()),
                upstream_seconds=analysis_seconds,
                peak_process_rss=psutil.Process().memory_info().peak_wset)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        group = row["key"].split("/")[0]
        grouped[row["profile"], group].append(row)
        if group in ("A", "B", "C"):
            grouped[row["profile"], "approved"].append(row)
    result = defaultdict(dict)
    for (profile, group), rr in grouped.items():
        result[profile][group] = {**quality(rr), "questions": len(rr),
            "hits": sum(len(r["metrics"]["factual_gold_documents_in_top_5"]) for r in rr),
            "required": sum(len(r["gold_documents"]) for r in rr),
            "all_factual": sum(len(r["metrics"]["factual_gold_documents_in_top_5"]) == len(r["gold_documents"])
                               for r in rr), "mean_edges": mean(r["diagnostics"]["relationships_examined"] for r in rr),
            "rss_high_water_bytes": max(r["peak_process_rss"] for r in rr),
            **{f"replay_p{p}": percentile([r["replay_seconds"] for r in rr], p/100) for p in (50, 95)},
            "stage_means_s": {s: mean(r["seconds"].get(s, 0) for r in rr)
                              for s in {k for r in rr for k in r["seconds"]}}}
    return result


def components(summary):
    def best(profiles):
        def priority(p):
            group = summary[label(p)]["approved"]
            return (-group["hits"], group["mean_off_gold_sources"], sum(x != y for x, y in zip(p, BASE)),
                    p[3], label(p))
        return min(profiles, key=priority)

    linking = [(m, *BASE[1:]) for m in LINK + ["document"]]
    traversal = [(BASE[0], t, *BASE[2:]) for t in TRAVERSAL]
    selection = [(*BASE[:3], n) for n in PREFIXES]
    combos = []
    for include_base in (True, False):
        pool = [BASE] if include_base else []
        combos.append((best(pool+linking)[0], best(pool+traversal)[1], 100, best(pool+selection)[3]))
    return combos, best


def execute(phase, output, prior):
    graph, cache, questions, references, archived = inputs()
    started = perf_counter()
    productive = direct_fact_index(graph)
    index = lexical_index(graph)
    index_seconds = perf_counter()-started
    extra = extras() if phase != "diagnosis" else {}
    configs = [BASE, B] if phase == "diagnosis" else matrix()
    document = ("document", *BASE[1:])
    if phase == "lexical":
        configs = [document]
    if phase in ("combined", "confirmation"):
        summary = read_json(prior / "matrix/summary.json")
        summary.update(read_json(prior / "lexical/summary.json"))
        combos, best = components(summary)
        configs = combos
        if phase == "confirmation":
            summary.update(read_json(prior / "combined/summary.json"))
            configs = [BASE, best(list(dict.fromkeys(matrix()+[document]+combos)))]
    configs = list(dict.fromkeys(configs))
    run_cases = [(q, False) for q in questions if q.group != "HOLDOUT"]
    if phase == "confirmation":
        run_cases = [(q, False) for q in questions if q.group == "HOLDOUT"]
        run_cases.extend((q, True) for q in questions if q.key in extra["paraphrases"])
    output.mkdir(parents=True, exist_ok=False)
    evidence, rows, losses = {}, [], {}
    for q, paraphrased in tqdm(run_cases, desc=phase, unit="question"):
        if paraphrased:
            query = extra["paraphrases"][q.key]["query"]
            scores = {}
            upstream = extra["paraphrases"][q.key]["seconds"]
        else:
            query, scores = original(q, archived, cache)
            upstream = sum(archived[q.key, "D"]["timings"][s]["seconds"] for s in ("ner", "e5"))
        # Только недостающие cosine-строки. Исходные строки интегрированного запуска приоритетны.
        needed = {s.text for s in _expansion_searches(query["text"], set())}
        needed.update(m["surface_text"] for m in query["mentions"])
        if not query["mentions"]:
            needed.add(query["text"])
        missing = needed - scores.keys()
        for text in missing:
            if text in extra.get("scores", {}):
                scores[text] = np.array(extra["scores"][text]["scores"])
        if paraphrased:
            terms = {m["surface_text"] for m in query["mentions"]} or {query["text"]}
            upstream += sum(extra["scores"][text]["seconds"] for text in terms)
        linker = CachedLinker(cache, scores, min_score=.85)
        grouped = defaultdict(list)
        for *prefix, budget in configs:
            grouped[tuple(prefix)].append(budget)
        for (mode, traversal, candidates), budgets in grouped.items():
            pool = run_pool(query, linker, graph, productive, mode, traversal, candidates, index)
            control = (mode, traversal, candidates, 0)
            if control in (BASE, B) and not paraphrased:
                archive = archived[q.key, "D" if control == BASE else "B"]
                require(pool["seeds"] == archive["linking"]["seeds"] and
                        [p["path_id"] for p in pool["ranked"]] == [p["path_id"] for p in archive["top_5"]]
                        and pool["raw"]["diagnostics"]["relationships_examined"] == archive["relationships_examined"],
                        f"Control differs: {q.key} {control}")
                if control == B:
                    require(digest([q.text, pool["seeds"], pool["raw"]["paths"], pool["bundles"], pool["ranked"]])
                            == ANCHOR["questions"][q.key], "Frozen B differs")
            order = None
            if phase == "diagnosis" or any(budgets) or (phase == "confirmation" and q.group == "HOLDOUT"):
                started = perf_counter()
                order = preselect(query["text"], pool)
                pre_seconds = perf_counter()-started
            for budget in budgets:
                profile = (*control[:3], budget)
                times = pool["seconds"].copy()
                top = pool["ranked"]
                if budget:
                    started = perf_counter()
                    top = diverse(query["text"], order, budget)
                    times.update(preselection=pre_seconds, selection=perf_counter()-started)
                ref = references[q.key]
                key = q.key if not paraphrased else "PARAPHRASE/" + q.key
                row = measure(key, profile, pool, top, ref, times, evidence, query, upstream)
                if mode == "clauses":
                    row["clause_e5_seconds_upper_bound"] = sum(extra["scores"][text]["seconds"]
                                                               for text in missing if text in extra["scores"])
                rows.append(row)
            if phase == "diagnosis" or (phase == "confirmation" and q.group == "HOLDOUT" and control == BASE):
                losses.setdefault(q.key, {})[label(control)] = diagnose(query, linker, graph, pool, order,
                    set(ref["document_contributions"]), 5 if mode == "k5" else 10, candidates)
        require(psutil.Process().memory_info().peak_wset < 8 * 1024**3, "RSS budget exceeded")
    with (output/"questions.jsonl").open("x", encoding="utf8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")
    dump(output/"evidence.json", evidence)
    dump(output/"summary.json", aggregate(rows))
    if losses:
        dump(output/"losses.json", losses)
    dump(output/"manifest.json", dict(profiles=configs, phase=phase, rows=len(rows), source_run=FINGERPRINT,
         extra_cache=file_sha(EXTRA) if extra else None, index_seconds=index_seconds, indexed_nodes=len(productive),
         source_git="320a85612daa414645154c64b6e340b974b0d727", anchor_sha256=file_sha(
             HERE.parent/"benchmark_3a5_8/FROZEN_B.json"), snapshot_sha256=ANCHOR["snapshot_sha256"],
         graph_fingerprint=graph.fingerprint, files={p.name:file_sha(p) for p in output.iterdir()},
         code={p.name:file_sha(p) for p in (Path(__file__), HERE/"inputs.py", HERE/"losses.py", HERE/"profiles.py")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("diagnosis", "matrix", "lexical", "combined", "confirmation"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    execute(args.phase, args.directory / args.phase, args.directory)
