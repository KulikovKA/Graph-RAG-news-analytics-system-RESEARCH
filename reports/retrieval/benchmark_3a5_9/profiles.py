"""Исследовательские профили. На входе только вопрос, read-only граф и результаты поиска."""

from collections import Counter, defaultdict
import math
import re
from itertools import zip_longest
from time import perf_counter

from cgraph.entity_linking import _expansion_searches, _text_search
from cgraph.evidence_extraction import EvidenceExtractor
from cgraph.graph_retrieval import GraphRetriever, TraversalLimits
from cgraph.path_reranking import _evidence_error, _terms, rerank_paths


def direct_fact_index(graph):
    return {key for key, edges in graph.adjacency.items()
            if any(r["kind"] == "RELEX_REL" and r["head"]["key"] != r["tail"]["key"]
                   and r["properties"]["document_id"] == r["head"].get("document_id")
                   == r["tail"].get("document_id") for r in edges)}


def lexical_index(graph):
    documents = {}
    counts = Counter()
    for key, doc in graph.documents.items():
        tokens = [s for s in re.findall(r"[^\W_]+", (doc["title"]+" "+doc["text"]).casefold()) if _terms(s)]
        frequencies = Counter(tokens)
        documents[key] = (frequencies, len(tokens))
        counts.update(frequencies.keys())
    return documents, counts, sum(n for _, n in documents.values()) / len(documents)


def document_seeds(linker, text, searches, productive, index):
    documents, counts, average = index
    scores = {}
    for key, (frequencies, length) in documents.items():
        norm = 1.2 * (.25 + .75 * length / average)
        scores[key] = math.fsum(math.log1p((len(documents)-counts[t]+.5)/(counts[t]+.5))
                                * 2.2 * frequencies[t]/(frequencies[t]+norm) for t in sorted(_terms(text)))
    leaders = sorted((d for d in scores if scores[d] > 0), key=lambda d: (-scores[d], d))[:2]
    candidates = {}
    for span in searches:
        for candidate in linker._rank(span, linker.scores[span.text], len(linker.cache.entities)):
            doc = candidate["document_id"]
            if doc in leaders and candidate["node_key"] in productive:
                if doc not in candidates or candidate["score"] > candidates[doc][0]["score"]:
                    candidates[doc] = candidate, span
    seeds, spans = [], []
    for doc in leaders:
        if doc in candidates:
            c, span = candidates[doc]
            seeds.append({k:v for k,v in c.items() if k != "rank"})
            spans.append(dict(start=span.start, end=span.end, text=span.text, node_key=c["node_key"],
                              mention_id=span.mention_id, cosine=c["score"], document=doc, bm25=scores[doc]))
    return seeds, spans


def generate_seeds(linker, query, mode, productive, index=None):
    searches = linker._primary_searches(query) or ([_text_search(query["text"])] if query["text"].strip() else [])
    groups = []
    for search in searches:
        scores = linker.scores[search.text]
        k = {"k5": 5, "k20": 20}.get(mode, 10)
        ranked = linker._rank(search, scores, k)
        seeds = linker._accepted(ranked)
        if mode == "rescue" and any(s["node_key"] not in productive for s in seeds):
            alternatives = {}
            for c in linker._rank(search, scores, len(linker.cache.entities)):
                if c["score"] >= .80 and c["node_key"] in productive:
                    if c["document_id"] not in alternatives or c["score"] > alternatives[c["document_id"]]["score"]:
                        alternatives[c["document_id"]] = c
            seeds = [({k: alternatives[s["document_id"]][k] for k in s}
                      if s["node_key"] not in productive and s["document_id"] in alternatives else s)
                     for s in seeds]
        if mode == "adaptive" and not any(s["node_key"] in productive for s in seeds):
            extended = linker._accepted(linker._rank(search, scores, 20)[len(ranked):])
            seeds.extend(s for s in extended if s["node_key"] in productive)
        groups.append(seeds)
    flat = [s for group in groups for s in group]
    extra_spans = []
    if mode == "document":
        reserved, extra_spans = document_seeds(linker, query["text"], searches, productive, index)
        flat = reserved + [s for s in flat if s["node_key"] not in {r["node_key"] for r in reserved}]
    if mode == "clauses":
        covered = set()
        for s in flat:
            covered.update(_terms(linker._names[s["entity_id"]]))
        seen = {s["node_key"] for s in flat}
        for span in _expansion_searches(query["text"], covered)[:4]:
            ranked = linker._rank(span, linker.scores[span.text], 10)
            best = next((s for s in linker._accepted(ranked)
                         if s["node_key"] not in seen and s["node_key"] in productive), None)
            if best:
                flat.insert(len(extra_spans), best)
                extra_spans.append(dict(node_key=best["node_key"], start=span.start, end=span.end, text=span.text))
                seen.add(best["node_key"])
    if mode == "round":
        flat = [s for layer in zip_longest(*groups) for s in layer if s is not None]
    if mode in ("dedup", "round"):
        unique = {}
        for s in flat:
            unique.setdefault(s["node_key"], s)
        flat = list(unique.values())
    return flat, extra_spans


def run_pool(query, linker, graph, productive, mode="production", traversal="fifo", candidates=100, index=None):
    started = perf_counter()
    uncapped, spans = generate_seeds(linker, query, mode, productive, index)
    seeds = uncapped[:32]
    linked = perf_counter()
    inputs = sorted(seeds, key=lambda s: s["node_key"] not in productive) if traversal == "productive" else seeds
    raw = GraphRetriever(graph).retrieve(inputs, TraversalLimits(max_candidates=candidates),
                                         seed_fair=traversal == "fair")
    traversed = perf_counter()
    bundles = EvidenceExtractor(graph).extract(raw)
    extracted = perf_counter()
    ranked = rerank_paths(query["text"], raw, bundles, seeds, policy="C")["paths"]
    finished = perf_counter()
    return dict(seeds=seeds, uncapped=uncapped, fallback_spans=spans, raw=raw, bundles=bundles, ranked=ranked,
                seconds=dict(linking=linked-started, bfs=traversed-linked, evidence=extracted-traversed,
                             ranking=finished-extracted))


def preselect(text, pool):
    # Воспроизводим порядок prepare() этапа 3A.5.8, сохраняя записи для аудита.
    prefix = pool["ranked"]
    selected = {p["path_id"] for p in prefix}
    rest = []
    for path, bundle in zip(pool["raw"]["paths"], pool["bundles"], strict=True):
        if path["path_id"] in selected or _evidence_error(path, bundle, pool["raw"]["graph_fingerprint"]):
            continue
        rest.extend(rerank_paths(text, {**pool["raw"], "paths": [path]}, [bundle],
                                 pool["seeds"], top_k=1, policy="C")["paths"])
    return prefix + sorted(rest, key=lambda p: (-p["base_score"], p["path_id"]))


def diverse(text, order, budget):
    query = _terms(text)
    pending = []
    for item in order[:budget]:
        doc_terms = defaultdict(set)
        sources = set()
        for edge in item["evidence"]["edges"]:
            for source in edge["source_relations"]:
                doc = edge["edge"]["document_id"]
                sources.add((doc, source["source_triple_id"]))
                for w in source["windows"]:
                    doc_terms[doc].update(_terms(w["text"]))
        matching = {d for d, terms in doc_terms.items() if query & terms}
        pending.append((item, sources, matching))
    chosen, docs, used = [], set(), set()
    while pending and len(chosen) < 5:
        fresh = [p for p in pending if p[1] - used]
        if not fresh:
            break
        pick = next((p for p in fresh if p[2] - docs), fresh[0]) if chosen else fresh[0]
        chosen.append({**pick[0], "rank": len(chosen) + 1})
        used.update(pick[1])
        docs.update(d for d, _ in pick[1])
        pending = [p for p in fresh if p[0]["path_id"] != pick[0]["path_id"]]
    return chosen
