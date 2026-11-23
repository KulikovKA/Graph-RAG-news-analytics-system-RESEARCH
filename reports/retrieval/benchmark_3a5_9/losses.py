"""Оценка после выдачи. Поиск свидетеля достижимости использует Gold только для диагностики."""

from collections import defaultdict, deque
from dataclasses import asdict, replace

from cgraph.entity_linking import _text_search
from cgraph.evidence_extraction import EvidenceExtractor
from cgraph.graph_retrieval import GraphRetriever, TraversalLimits, _ADJACENCY_QUERY
from cgraph.path_reranking import _evidence_error


def distances(adjacency, roots):
    dist = dict.fromkeys(roots, 0)
    queue = deque(roots)
    while queue:
        node = queue.popleft()
        if dist[node] == 5:
            continue
        for other in adjacency[node]:
            if other not in dist:
                dist[other] = dist[node] + 1
                queue.append(other)
    return dist


def witness(graph, seeds, target):
    # Двусторонняя нижняя оценка расстояния сохраняет все простые пути длиной до пяти.
    adjacency = defaultdict(set)
    ends = set()
    for key, rows in graph.adjacency.items():
        for row in rows:
            h, t = row["head"]["key"], row["tail"]["key"]
            if h == t or (row["kind"] == "SIMILAR_TO" and
                          row["properties"].get("resolution_decision_2f2") != "RELATED"):
                continue
            adjacency[key].add(h if t == key else t)
            if row["kind"] == "RELEX_REL" and row["properties"]["document_id"] == target:
                ends.update((h, t))
    starts = list(dict.fromkeys(s["node_key"] for s in seeds))
    near_seed, near_doc = distances(adjacency, starts), distances(adjacency, ends)

    class Filtered:
        def run(self, query, *, namespace, key, limit):
            assert query == _ADJACENCY_QUERY
            return [r for r in graph.adjacency[key] if near_seed.get(key, 6) + 1 +
                    near_doc.get(r["tail"]["key"] if r["head"]["key"] == key else r["head"]["key"], 6) <= 5]

    # Отбрасывание продолжений после первого валидного факта допустимо при поиске свидетеля.
    frontier = deque((s["node_key"], [s["node_key"]], [], {s["document_id"]}, 0, 0)
                     for s in {s["node_key"]: s for s in seeds}.values() if near_doc.get(s["node_key"], 6) <= 4)
    limits = TraversalLimits(max_neighbors=10000, max_relationships=10**12)
    iterator = GraphRetriever(graph)._paths(Filtered(), frontier, limits, graph.fingerprint,
                                             dict(nodes_examined=0, relationships_examined=0), set())
    for path in iterator:
        if path["edges"][-1]["document_id"] == target:
            raw = dict(paths=[path], namespace=GraphRetriever(graph).namespace, graph_fingerprint=graph.fingerprint)
            bundle, = EvidenceExtractor(graph).extract(raw)
            if not _evidence_error(path, bundle, graph.fingerprint):
                return dict(path_id=path["path_id"], seed=path["seed_key"], edges=[e["key"] for e in path["edges"]])
    return None


def diagnose(query, linker, graph, pool, order, required, k, budget):
    result = []
    for doc in sorted(required):
        def supplies(p):
            return any(e["edge"]["document_id"] == doc and e["status"] == "valid" for e in p["evidence"]["edges"])

        if any(supplies(p) for p in pool["ranked"]):
            continue
        ranks = []
        for s in linker._primary_searches(query) or [_text_search(query["text"])]:
            rows = linker._rank(s, linker.scores[s.text], len(linker.cache.entities))
            candidates = [c for c in rows if c["document_id"] == doc]
            if candidates:
                ranks.append(dict(span=asdict(s), best=candidates[0], best_cosine=max(c["score"] for c in candidates),
                                  above_cutoff_in_k=sum(c["score"] >= .85 and c["rank"] <= k for c in candidates)))
        first = next((i for i, p in enumerate(order, 1) if supplies(p)), None)
        raw = pool["raw"]
        doc_paths = [p for p in raw["paths"] if doc in p["document_ids"]]
        raw_facts = [p for p in doc_paths if any(e["document_id"] == doc for e in p["edges"])]
        uncapped = [s for s in pool["uncapped"] if s["document_id"] == doc]
        selected = [s for s in pool["seeds"] if s["document_id"] == doc]
        match = witness(graph, pool["seeds"], doc) if first is None else None
        cap_tests = {}
        for name in raw["diagnostics"]["limits_hit"]:
            if first is not None or name not in ("max_candidates", "max_relationships", "max_neighbors"):
                continue
            old = TraversalLimits(max_candidates=budget)
            limits = replace(old, **{name: 2 * getattr(old, name)})
            test = GraphRetriever(graph).retrieve(pool["seeds"], limits)
            bundles = EvidenceExtractor(graph).extract(test)
            cap_tests[name] = {"value": getattr(limits, name), "edges": test["diagnostics"]["relationships_examined"],
                               "witness": next((p["path_id"] for p, b in zip(test["paths"], bundles, strict=True)
                                   if not _evidence_error(p, b, graph.fingerprint) and
                                   any(e["document_id"] == doc for e in p["edges"])), None)}
        result.append(dict(document=doc, corpus_present=doc in graph.documents, mentions=ranks,
                           graph_entities=sum(n["label"] == "RELEXEntity" and n["properties"]["document_id"] == doc
                                              for n in graph.nodes.values()),
                           graph_fact_edges=len({r["properties"]["key"] for rows in graph.adjacency.values()
                               for r in rows if r["kind"] == "RELEX_REL" and r["properties"]["document_id"] == doc}),
                           uncapped=uncapped, capped=selected, raw_document_paths=len(doc_paths),
                           raw_factual_paths=len(raw_facts), eligible_paths=sum(supplies(p) for p in order),
                           eligible_first=first, eligible_example=order[first-1]["path_id"] if first else None,
                           oracle_20_50_100=[bool(first and first <= n) for n in (20, 50, 100)],
                           limits=raw["diagnostics"], double_one_limit=cap_tests, unbudgeted_witness=match))
    return result
