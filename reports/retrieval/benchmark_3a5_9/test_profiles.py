from copy import deepcopy
from collections import Counter

import numpy as np

from inputs import CachedLinker
from cgraph.entity_linking import EntityCache
from cgraph.retrieval_types import TARGET_NAMESPACE
from profiles import diverse, generate_seeds


def example():
    records = [dict(entity_id=f"e:{n}", document_id=f"d{n//2}", text_name=f"entity {n}") for n in range(4)]
    cache = EntityCache(records, np.ones((4, 1)), "synthetic", {})
    scores = {"first": np.array([.91, .90, .80, .6]), "second": np.array([.89, .86, .96, .88])}
    linker = CachedLinker(cache, scores, min_score=.85)
    text = "first second"
    mentions = [dict(mention_id=f"m{i}", surface_text=t, start=text.index(t), end=text.index(t)+len(t))
                for i, t in enumerate(scores)]
    return linker, dict(text=text, mentions=mentions)


def test_round_deduplicates_before_cap_and_keeps_first_span():
    linker, q = example()
    seeds, _ = generate_seeds(linker, q, "round", set())
    assert [(s["entity_id"], s["mention_id"]) for s in seeds] == [
        ("e:0", "m0"), ("e:2", "m1"), ("e:1", "m0"), ("e:3", "m1")]


def test_empty_and_rescue_keeps_document_and_original_cosine():
    linker, q = example()
    productive = {TARGET_NAMESPACE+":e:1"}
    seeds, _ = generate_seeds(linker, q, "rescue", productive)
    assert seeds[0]["entity_id"] == "e:1" and seeds[0]["score"] == .90 and seeds[0]["mention_id"] == "m0"
    assert generate_seeds(linker, dict(text="", mentions=[]), "adaptive", productive) == ([], [])
    assert diverse("", [], 20) == []


def test_diverse_first_path_fresh_sources_and_stable_order():
    def path(pid, document, triple):
        return dict(path_id=pid, rank=1, evidence=dict(edges=[dict(edge=dict(document_id=document),
                    source_relations=[dict(source_triple_id=triple, windows=[dict(text="topic")])])]))
    paths = [path("a", "D", "t1"), path("b", "D", "t1"), path("c", "D", "t2"), path("d", "E", "t3")]
    before = deepcopy(paths)
    assert [p["path_id"] for p in diverse("topic", paths, 3)] == ["a", "c"]
    assert [p["path_id"] for p in diverse("topic", paths, 4)] == ["a", "d", "c"]
    assert paths == before


def test_document_fallback_ignores_zero_overlap_and_preserves_offsets():
    linker, q = example()
    productive = {TARGET_NAMESPACE+":e:3"}
    index = ({"d0": ({}, 1), "d1": ({"second": 1}, 1)}, {"second": 1}, 1)
    docs = {k: (Counter(f), n) for k, (f, n) in index[0].items()}
    seeds, spans = generate_seeds(linker, q, "document", productive, (docs, Counter(index[1]), 1))
    assert len(spans) == 1 and seeds[0]["node_key"] in productive
    s = spans[0]
    assert q["text"][s["start"]:s["end"]] == s["text"] == "second"
