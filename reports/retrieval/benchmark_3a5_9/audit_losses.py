"""Дополнительная оценка пропусков после фиксации финалиста, включая исторический B."""

import sys
from pathlib import Path

from tqdm import tqdm

from inputs import ANCHOR, CachedLinker, digest, inputs, original, require
from losses import diagnose
from profiles import direct_fact_index, preselect, run_pool
from run import B, BASE, dump, label


def main(output):
    graph, cache, questions, refs, archived = inputs()
    productive = direct_fact_index(graph)
    result = {}
    for q in tqdm(questions, desc="Final loss audit"):
        query, scores = original(q, archived, cache)
        linker = CachedLinker(cache, scores, min_score=.85)
        for config in (BASE, B, ("k5", "fifo", 100, 0)):
            mode, traversal, cap, _ = config
            pool = run_pool(query, linker, graph, productive, mode, traversal, cap)
            if config == B:
                require(digest([q.text, pool["seeds"], pool["raw"]["paths"], pool["bundles"], pool["ranked"]])
                        == ANCHOR["questions"][q.key], "Frozen B signature differs")
            required = set(refs[q.key]["document_contributions"])
            docs = {e["edge"]["document_id"] for p in pool["ranked"] for e in p["evidence"]["edges"]}
            if required <= docs:
                continue
            order = preselect(query["text"], pool)
            losses = diagnose(query, linker, graph, pool, order, required, 5 if mode == "k5" else 10, cap)
            for loss in losses:
                degrees = {}
                for seed in loss["uncapped"]:
                    key = seed["node_key"]
                    rows = graph.adjacency.get(key, [])
                    degrees[key] = dict(entity=linker._names[seed["entity_id"]], total=len(rows),
                        direct_fact=sum(r["kind"] == "RELEX_REL" and r["head"]["key"] != r["tail"]["key"]
                                        for r in rows),
                        physical_types=sorted({r["kind"] for r in rows}))
                loss["seed_degrees"] = degrees
                loss["seed_input"] = [s["entity_id"] for s in pool["seeds"]]
                loss["selected"] = [p["path_id"] for p in pool["ranked"]]
            result.setdefault(q.key, {})[label(config)] = losses
    dump(output, result)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
