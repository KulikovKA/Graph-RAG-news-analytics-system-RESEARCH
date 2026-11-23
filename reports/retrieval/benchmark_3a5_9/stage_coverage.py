"""Послепрогонный расчёт доступности обязательного источника по стадиям для каждой абляции."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from inputs import CachedLinker, extras, inputs, original, read_json, require
from cgraph.evidence_extraction import EvidenceExtractor
from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
from cgraph.path_reranking import _evidence_error, rerank_paths
from profiles import direct_fact_index, diverse, lexical_index, preselect, run_pool
from run import BASE, dump, label


def factual(paths):
    return {e["document_id"] for p in paths for e in p["edges"] if e["type"] == "RELEX_REL"}


def coverage(directory, output):
    graph, cache, queries, refs, archive = inputs()
    extra = extras()
    productive = direct_fact_index(graph)
    index = lexical_index(graph)
    expected, configs = {}, {}
    for phase in ("matrix", "lexical", "combined"):
        root = directory / phase
        for profile in read_json(root/"manifest.json")["profiles"]:
            configs.setdefault(tuple(profile[:3]), set()).add(profile[3])
        for line in (root/"questions.jsonl").read_bytes().splitlines():
            row = json.loads(line)
            expected[row["key"], row["profile"]] = row["top"]
    scores = {t: np.array(v["scores"]) for t,v in extra["scores"].items()}
    rows = []
    for q in tqdm([q for q in queries if q.group != "HOLDOUT"], desc="Stage coverage"):
        query, primary = original(q, archive, cache)
        linker = CachedLinker(cache, {**scores, **primary}, min_score=.85)
        required = refs[q.key]["document_contributions"]
        for (mode, traversal, cap), budgets in configs.items():
            pool = run_pool(query, linker, graph, productive, mode, traversal, cap, index)
            raw = pool["raw"]
            eligible = [p for p,b in zip(raw["paths"],pool["bundles"],strict=True)
                        if not _evidence_error(p,b,graph.fingerprint)]
            order = preselect(q.text, pool) if any(budgets) else []
            for n in sorted(budgets):
                prefix = [p["path"] for p in order[:n]] if n else eligible
                chosen = diverse(q.text, order, n) if n else pool["ranked"]
                name = label((mode,traversal,cap,n))
                require([p["path_id"] for p in chosen] == expected[q.key,name], "Top differs from published run")
                stages = dict(uncapped_seed_docs={s["document_id"] for s in pool["uncapped"]},
                              accepted_seed_docs={s["document_id"] for s in pool["seeds"]},
                              raw_docs={d for p in raw["paths"] for d in p["document_ids"]},
                              raw_relex=factual(raw["paths"]), eligible=factual(eligible),
                              preselection=factual(prefix), top=factual([p["path"] for p in chosen]))
                rows.append(dict(key=q.key, profile=name, required=sorted(required),
                                 counts=dict(eligible=len(eligible),preselection=len(prefix)),
                                 hits={k: sorted(v.intersection(required)) for k,v in stages.items()}))
            if (mode,traversal,cap) == BASE[:3]:
                # Разворот после cap меняет только порядок обхода. Scores реранкера те же.
                reversed_raw = GraphRetriever(graph).retrieve(reversed(pool["seeds"]), TraversalLimits())
                bundles = EvidenceExtractor(graph).extract(reversed_raw)
                top = rerank_paths(q.text,reversed_raw,bundles,pool["seeds"],policy="C")["paths"]
                rows.append(dict(key=q.key,profile="diagnostic_reverse_fifo",required=sorted(required),
                                 hits=dict(top=sorted(factual([p["path"] for p in top]).intersection(required))),
                                 edges=reversed_raw["diagnostics"]["relationships_examined"],
                                 top_ids=[p["path_id"] for p in top]))
    totals = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for k, docs in r["hits"].items():
            totals[r["profile"]][k] += len(docs)
    dump(output, dict(rows=rows, approved_hits=totals))


if __name__ == "__main__":
    coverage(*map(Path,sys.argv[1:]))
