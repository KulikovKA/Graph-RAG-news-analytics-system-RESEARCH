import io
from itertools import product
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_contextual_shared_concepts as runner
from contextual_shared_concepts import (
    PREFIX, THRESHOLDS, WINDOWS, DocumentWindows, cosine_scores, extract_windows, filter_mask, fit_window,
    guard_reasons, mean_embeddings, normalized, pair_guard_codes, pair_indices, sentence_spans,
)
from contextual_shared_concepts_report import atomic_csv, pair_position


def tokenizer():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "query", ":", ".", "left", "right", "two", "words", "entity")
    result = Tokenizer(WordLevel({word: i for i, word in enumerate(tokens)}, unk_token="[UNK]"))
    result.pre_tokenizer = Whitespace()
    result.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    return result


def test_scientific_sentences_and_all_sentence_windows():
    sentences = ["The rate was 3.14 g., e.g. at noon.", "See Fig. 2 and Eq. 4.",
                 "F. Smith et al. measured cobalt.", "This is the multiword entity.",
                 'A "quoted result."', "Next result (4.2 mg).", "Conclusion."]
    text = " ".join(sentences)
    windows = DocumentWindows(text)
    mention = "multiword entity"
    start = text.index(mention)
    end = start + len(mention)
    assert [text[a:b] for a, b in sentence_spans(text)] == sentences
    for radius in range(4):
        a, b = windows.window(f"S{radius}", start, end)
        assert text[a:b] == " ".join(sentences[3-radius:4+radius])
    assert windows.window("S3", 0, 3)[0] == 0
    assert windows.window("S3", len(text)-5, len(text))[1] == len(text)
    a, b = windows.window("S0", text.index("cobalt."), text.index("This")+4)
    assert text[a:b] == " ".join(sentences[2:4])


@pytest.mark.parametrize("size", [8, 16, 32, 64, 128])
def test_word_windows_use_offsets_and_full_mention(size):
    before = [f"l{i}" for i in range(140)]
    after = [f"r{i}" for i in range(140)]
    mention = "lignocellulosic\tbiomass"
    text = " ".join(before+["same name", mention]+after)
    start = text.index(mention)
    result = DocumentWindows(text)
    a, b = result.window(f"W{size}", start, start+len(mention))
    assert text[a:b].split() == (before+["same", "name"])[-size:]+mention.split()+after[:size]
    assert result.window(f"W{size}", 0, 2)[0] == 0
    assert result.window(f"W{size}", len(text)-2, len(text))[1] == len(text)
    with pytest.raises(ValueError):
        result.window(f"W{size}", 3, len(text)+1)


def test_repeated_mention_deduplicates_model_input_only():
    text = "The entity and entity appeared again."
    doc = dict(text=text, text_sha256="text")
    first, second = text.index("entity"), text.rindex("entity")
    evidence = [dict(document_id="d", revision_id="r", source_url="url",
                     source_entity_id=f"e-{i}", source_mention_ids=[f"m-{i}"], mention_text="entity",
                     mention_start=p, mention_end=p+6) for i, p in enumerate((first, second))]
    rows, count = extract_windows(
        [dict(entity_id="e", name="entity", evidence=evidence)], {"d": doc}, {"d": DocumentWindows(text)}, tokenizer(), "S0")
    assert count == 0
    contexts = rows[0]["contexts"]
    assert len(contexts) == 1
    assert contexts[0]["input"] == "entity. " + text
    assert [(p["mention_start"], p["source_mention_ids"]) for p in contexts[0]["provenance"]] == [
        (first, ["m-0"]), (second, ["m-1"])]


@pytest.mark.parametrize("left, right", [(800, 800), (0, 700), (700, 0)])
def test_truncation_preserves_source_span_and_special_tokens(left, right):
    text = "left "*left + "two words" + " right"*right
    start, end = len("left "*left), len("left "*left+"two words")
    tok = tokenizer()
    lo, hi = fit_window(tok, text, "entity", 0, len(text), start, end)
    assert 0 <= lo <= start < end <= hi <= len(text)
    assert text[lo:hi][start-lo:end-lo] == "two words"
    assert len(tok.encode(PREFIX+"entity. "+text[lo:hi]).ids) == 512


def test_truncation_fails_if_mention_itself_is_too_long():
    text = "two words "*300
    with pytest.raises(ValueError, match="intact"):
        fit_window(tokenizer(), text, "entity", 0, len(text), 0, len(text)-1)


def test_arithmetic_mean_is_saved_without_renormalization():
    vector = mean_embeddings([[1., 0.], [0., 1.]])
    np.testing.assert_array_equal(vector, [.5, .5])
    assert np.linalg.norm(vector) == pytest.approx(2**-.5)
    vectors = np.array([vector, [.75, .1], [1., 0.]])
    left, right = pair_indices(["a", "b", "c"])
    expected = [np.dot(vectors[i], vectors[j])/np.linalg.norm(vectors[i])/np.linalg.norm(vectors[j])
                for i, j in zip(left, right)]
    np.testing.assert_allclose(cosine_scores(vectors, left, right, "test", block=1), expected, atol=1e-15)
    np.testing.assert_allclose(cosine_scores(normalized(vectors), left, right, "test"), expected, atol=1e-15)
    with pytest.raises(ValueError, match="zero"):
        mean_embeddings([[1, 0], [-1, 0]])
    with pytest.raises(ValueError):
        normalized([[np.nan, 1], [1, np.inf]])


def test_independent_thresholds_pair_identity_and_monotonicity():
    left, right = pair_indices(["a", "b", "a", "b", "c"])
    pairs = list(zip(left, right))
    assert pairs == [(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)]
    assert [pair_position(j, i, left, right) for i, j in pairs] == list(range(len(pairs)))
    with pytest.raises(ValueError):
        pair_position(0, 2, left, right)
    n, c = np.array([.96, .75, .90, .85]), np.array([.74, .97, .90, .85])
    np.testing.assert_array_equal(filter_mask(n, c, .85, .85), [False, False, True, True])
    counts = np.array([filter_mask(n, c, a, b).sum() for a, b in product(THRESHOLDS, repeat=2)]).reshape(5, 5)
    assert np.all(np.diff(counts, axis=0) <= 0)
    assert np.all(np.diff(counts, axis=1) <= 0)


def test_guards_work_without_name_matching():
    profiles = [
        dict(entity_types=["RESULT"], expansions={"abc": ["adaptive binary classifier"]}),
        dict(entity_types=["ACTOR"], expansions={"abc": ["another broader category"]}),
        dict(entity_types=["ATTRIBUTE"], expansions={}),
        dict(entity_types=["PROCESS", "TASK"], expansions={}),
    ]
    left, right = pair_indices(["a", "b", "c", "d"])
    codes = pair_guard_codes(profiles, left, right, np.ones(len(left), dtype=bool))
    assert codes.tolist() == [7, 3, 3, 2, 2, 0]
    assert not guard_reasons(profiles[2], profiles[3])
    assert not guard_reasons(
        dict(entity_types=["ATTRIBUTE"], expansions={"auc": ["area under the curve"]}),
        dict(entity_types=["ATTRIBUTE"], expansions={"auc": ["area under the roc curve"]}))


def test_checkpoint_checks_compatibility_and_corruption(tmp_path):
    data, index = tmp_path/"scores.npy", tmp_path/"S0.json"
    runner.atomic_array(data, np.arange(5.))
    finished = runner.complete(index, "run-a", [data.name], count=5)
    assert runner.checked_output(index, "run-a", [data.name]) == finished
    with pytest.raises(ValueError, match="Incompatible"):
        runner.checked_output(index, "run-b", [data.name])
    assert not runner.checked_output(tmp_path/"W128.json", "run-a", [])
    data.write_bytes(b"corrupted")
    assert runner.checked_output(index, "run-a", [data.name]) is None


def test_inference_batch_resumes_without_reencoding(monkeypatch, tmp_path):
    path = tmp_path/"embeddings.sqlite3"
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 1)
    monkeypatch.setattr(runner, "RECIPE", runner.RECIPE | {"batch": 1})
    transcript = []

    def embed(batch):
        transcript.extend(batch)
        if batch == ["right"]:
            raise RuntimeError("interrupted")
        return np.eye(1, 384, dtype=np.float32)

    monkeypatch.setattr(runner, "e5_worker", lambda *args: embed)
    db = sqlite3.connect(path)
    cache = runner.EmbeddingCache(db, tmp_path, 4, SimpleNamespace(peak=0), 1)
    with pytest.raises(RuntimeError, match="interrupted"):
        cache.run(["left", "right"], tokenizer(), "S0")
    db.close()
    db = sqlite3.connect(path)
    monkeypatch.setattr(runner, "e5_worker", lambda *args: lambda batch: np.eye(1, 384, dtype=np.float32))
    cache = runner.EmbeddingCache(db, tmp_path, 4, SimpleNamespace(peak=0), 1)
    restored, stats = cache.run(["left", "right"], tokenizer(), "S0")
    assert list(restored) == ["left", "right"]
    assert stats["encoded"] == 2 and stats["cache_hits"] == 0
    monkeypatch.setattr(runner, "e5_worker", lambda *args: pytest.fail("Unexpected model load"))
    reused, stats = cache.run(["left"], tokenizer(), "W8")
    assert stats["encoded"] == 0 and stats["cache_hits"] == 1
    np.testing.assert_array_equal(reused["left"], restored["left"])
    db.close()


def test_csv_preserves_unicode_and_reproducibility(tmp_path):
    output = tmp_path/"evidence.csv"
    rows = [{"name": "μg/mL", "details": {"span": "γ", "offset": [1, 2]}}]
    atomic_csv(output, rows)
    first = output.read_bytes()
    atomic_csv(output, rows)
    assert output.read_bytes() == first
    assert "μg/mL" in io.StringIO(output.read_text(encoding="utf-8")).read()
    assert len(WINDOWS) == 9
