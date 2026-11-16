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
