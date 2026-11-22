import numpy as np
import pytest

from cgraph.entity_linking import EntityCache, EntityLinker, E5Encoder, load_entity_cache
from cgraph.query_analysis import QueryAnalyzer


class FakeNER:
    def __init__(self, found=True):
        self.found = found

    def predict(self, text, threshold):
        return ([dict(start=0, end=3, text="RNA", label="physical, biological or digital object", score=0.8)]
                if self.found else [])


class FakeEncoder:
    def __init__(self):
        self.texts = []

    def encode(self, texts):
        self.texts.extend(texts)
        return np.array([[1, 0] for _ in texts], dtype=np.float32)


def cache():
    return EntityCache([dict(entity_id="e:b", document_id="d2", text_name="RNA"),
                        dict(entity_id="e:a", document_id="d1", text_name="RNA"),
                        dict(entity_id="e:c", document_id="d3", text_name="other")],
                       np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32), "synthetic", {})


def test_exact_keeps_contextual_ids_and_deterministic_ties():
    linker = EntityLinker(cache(), FakeEncoder(), namespace="ns", min_score=0.5)
    query = QueryAnalyzer(FakeNER()).analyze("RNA?")
    result = linker.link(query, 3)
    assert [x["entity_id"] for x in result["candidates"]] == ["e:a", "e:b", "e:c"]
    assert [x["document_id"] for x in result["seeds"]] == ["d1", "d2"]
    assert [x["match_method"] for x in result["candidates"]] == ["exact", "exact", "vector"]
    assert result["seeds"][0]["node_key"] == "ns:e:a"
    assert result["mode"] == "threshold"
    assert result["cutoff"] == 0.5 and result["abstention_reason"] is None
    assert not result["abstained"] and result["unresolved_mentions"] == []
    assert linker.link_seeds(query, 3) == result["seeds"]


def test_abstention_and_full_question_fallback():
    encoder = FakeEncoder()
    analyzer = QueryAnalyzer(FakeNER(False))
    linker = EntityLinker(cache(), encoder, namespace="ns")
    result = linker.link(analyzer.analyze("What is RNA?"), 1)
    assert encoder.texts == ["What is RNA?"]
    assert result["candidates"][0]["match_method"] == "fallback"
    assert result["candidates"][0]["mention_id"] is None
    assert result["abstained"] and result["seeds"] == []
    assert result["mode"] == "candidates_only"
    assert result["abstention_reason"] == "no_linking_policy"
    empty = linker.link(analyzer.analyze(""))
    assert empty["candidates"] == [] and empty["abstention_reason"] == "empty_query"


def test_candidates_only_does_not_promote_even_identical_vectors_without_policy():
    encoder = FakeEncoder()
    linker = EntityLinker(cache(), encoder)
    query = QueryAnalyzer(FakeNER()).analyze("RNA?")
    with pytest.raises(ValueError, match="explicit min_score policy"):
        linker.link_seeds(query)
    assert encoder.texts == []
    result = linker.link(query, 2)
    assert len(result["candidates"]) == 2
    assert all(candidate["score"] == 1.0 for candidate in result["candidates"])
    assert result["mode"] == "candidates_only" and result["cutoff"] is None
    assert result["seeds"] == [] and result["abstained"]
    assert result["unresolved_mentions"] == [query["mentions"][0]["mention_id"]]
    assert result["abstention_reason"] == "no_linking_policy"


@pytest.mark.parametrize("found", [True, False])
def test_explicit_policy_abstains_on_weak_similarity_even_for_exact_names(found):
    class WeakEncoder:
        def encode(self, texts):
            return np.array([[0, -1] for _ in texts], dtype=np.float32)

    linker = EntityLinker(cache(), WeakEncoder(), min_score=0.5)
    query = QueryAnalyzer(FakeNER(found)).analyze("RNA?")
    result = linker.link(query)
    assert result["candidates"]
    assert max(candidate["score"] for candidate in result["candidates"]) == 0.0
    assert result["mode"] == "threshold" and result["abstention_reason"] == "below_min_score"
    assert result["abstained"] and result["seeds"] == []
    assert linker.link_seeds(query) == []
    if found:
        assert result["candidates"][0]["match_method"] == "exact"
        assert result["unresolved_mentions"] == [query["mentions"][0]["mention_id"]]


def test_corrupt_cache_fails_closed(tmp_path):
    (tmp_path / "embedding_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_entity_cache(tmp_path)


def test_conditional_fallback_recovers_after_mention_fails_threshold():
    class Encoder:
        def encode(self, texts):
            return np.array([[-1, 0] if t == "RNA" else [0, 1] for t in texts], dtype=np.float32)

    query = QueryAnalyzer(FakeNER()).analyze("RNA question?")
    baseline = EntityLinker(cache(), Encoder(), min_score=0.8, top_k=1)
    assert baseline.link_seeds(query) == []
    linker = EntityLinker(cache(), Encoder(), min_score=0.8, top_k=1, fallback="conditional")
    result = linker.link(query)
    assert result["unresolved_mentions"] == [query["mentions"][0]["mention_id"]]
    assert [(seed["entity_id"], seed["match_method"]) for seed in result["seeds"]] == [("e:c", "fallback")]
    candidate = result["candidates"][-1]
    assert candidate["query_span"] == {"start": 0, "end": len(query["text"]), "text": query["text"]}
    assert "query_span" not in result["seeds"][0]


def test_expansion_keeps_literal_spans_and_prioritizes_primary_exact_seeds():
    text = "gene expression and plastid genome?"
    mentions = [{"mention_id": "vector-first", "surface_text": "gene", "start": 0, "end": 4},
                {"mention_id": "exact-later", "surface_text": "gene expression", "start": 0, "end": 15}]
    rows = [
        dict(entity_id="e:a", document_id="d1", text_name="gene expression"),
        dict(entity_id="e:b", document_id="d2", text_name="genetics"),
        dict(entity_id="e:c", document_id="d3", text_name="plastid genome"),
        dict(entity_id="e:d", document_id="d4", text_name="genome length"),
    ]
    vectors = np.array([[0.8, 0.6], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    entity_cache = EntityCache(rows, vectors, "synthetic", {})

    class Encoder:
        def encode(self, texts):
            return np.array([[0, 1] if t == "plastid genome" else [-1, 0] if t == text else [1, 0]
                             for t in texts], dtype=np.float32)

    linker = EntityLinker(entity_cache, Encoder(), min_score=0.75, top_k=2, fallback="expansion",
                          max_fallback_seeds=1)
    result = linker.link({"text": text, "mentions": mentions})
    assert [(s["entity_id"], s["match_method"]) for s in result["seeds"]] == [
        ("e:a", "exact"), ("e:b", "vector"), ("e:c", "fallback")]
    assert len({s["node_key"] for s in result["seeds"]}) == 3
    spans = [row["query_span"] for row in result["candidates"] if row["match_method"] == "fallback"]
    assert {span["text"] for span in spans} == {"plastid genome", text}
    assert all(text[span["start"]:span["end"]] == span["text"] for span in spans)
    assert result == linker.link({"text": text, "mentions": mentions})


def test_conditional_skips_when_specific_primary_covers_question():
    encoder = FakeEncoder()
    rows = [dict(entity_id="e:a", document_id="d1", text_name="plastid genome")]
    entity_cache = EntityCache(rows, np.array([[1, 0]], dtype=np.float32), "synthetic", {})
    text = "What is the plastid genome?"
    mention = {"mention_id": "qm", "surface_text": "plastid genome", "start": 12, "end": 26}
    linker = EntityLinker(entity_cache, encoder, min_score=0.85, fallback="conditional")
    assert [s["match_method"] for s in linker.link_seeds({"text": text, "mentions": [mention]})] == ["exact"]
    assert encoder.texts == ["plastid genome"]


def test_constructor_top_k_and_baseline_duplicate_order_are_reproducible():
    text = "RNA?"
    query = QueryAnalyzer(FakeNER()).analyze(text)
    query["mentions"].append({**query["mentions"][0], "mention_id": "second"})
    linker = EntityLinker(cache(), FakeEncoder(), min_score=0.8, top_k=1)
    assert [s["entity_id"] for s in linker.link_seeds(query)] == ["e:a", "e:a"]
    assert [s["entity_id"] for s in linker.link_seeds(query, 2)] == ["e:a", "e:b", "e:a", "e:b"]


def test_retrieval_profile_uses_selected_top_k_without_changing_baseline():
    query = QueryAnalyzer(FakeNER()).analyze("RNA?")
    rows = [dict(entity_id=f"e:{i:02d}", document_id=f"d{i}", text_name="RNA") for i in range(12)]
    entity_cache = EntityCache(rows, np.tile(np.array([[1, 0]], dtype=np.float32), (12, 1)), "synthetic", {})
    production = EntityLinker.for_retrieval(entity_cache, FakeEncoder(), namespace="ns")
    baseline = EntityLinker(entity_cache, FakeEncoder(), namespace="ns", min_score=0.85)
    assert (production.min_score, production.top_k, production.fallback) == (0.85, 10, "baseline")
    assert len(production.link_seeds(query)) == 10
    assert production.link_seeds(query, 5) == baseline.link_seeds(query)


@pytest.mark.parametrize("settings", [{"top_k": 0}, {"fallback": "unknown"}, {"max_fallback_seeds": -1}])
def test_invalid_linking_settings(settings):
    with pytest.raises(ValueError):
        EntityLinker(cache(), **settings)


def test_e5_prefix_padding_and_masked_pooling():
    torch = pytest.importorskip("torch")
    class Token:
        def __init__(self, ids):
            self.ids, self.attention_mask, self.type_ids = ids, [1] * len(ids), [0] * len(ids)
    class Tokenizer:
        def encode_batch(self, texts):
            assert texts == ["query: RNA", "query: a b"]
            return [Token([1, 2]), Token([3])]
    class Model:
        def __call__(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[1, 2], [3, 0]]
            assert kwargs["attention_mask"].tolist() == [[1, 1], [1, 0]]
            class Output: pass
            output = Output()
            output.last_hidden_state = torch.tensor([[[1., 0.], [0., 1.]], [[0., 2.], [100., 100.]]])
            return output
    encoder = E5Encoder({})
    encoder._tokenizer, encoder._model = Tokenizer(), Model()
    actual = encoder.encode(["RNA", "a b"])
    np.testing.assert_allclose(actual, [[2**-0.5, 2**-0.5], [0, 1]], atol=1e-6)
