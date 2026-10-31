"""Preparation checks only: synthetic spans and linguistic NLP, never GLiNER inference."""
import json
import sys
from pathlib import Path

import pytest
from jsonschema import validate

from cgraph.common import read_json, sha256
from cgraph.ner import windows
from cgraph.ner_v2.cli import main
from cgraph.ner_v2.detection import label_descriptions, postprocess
from cgraph.ner_v2.io import read_documents, save_mentions, validate_document, validate_mention, validate_span, write_jsonl
from cgraph.ner_v2.model_setup import _matches
from cgraph.ner_v2.normalization import LinguisticNormalizer, detect_abbreviations, safe_unicode
from cgraph.ner_v2.openalex import OpenAlexClient, prepare_corpus, reconstruct_abstract, select_works

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def normalizer():
    pytest.importorskip("spacy")
    pytest.importorskip("en_core_web_sm")
    return LinguisticNormalizer()


@pytest.fixture
def document():
    text = "Cells contain cells."
    digest = sha256(text.encode())
    return {"contract_version": "ner-v2-source-1", "document_id": "synthetic", "revision_id": "revision",
            "text": text, "text_sha256": digest, "title": "Synthetic source", "year": 2026,
            "discipline": "Biology", "field": {"id": "test", "display_name": "test"},
            "source_url": "https://example.test/source", "source_reference": {
                "source_url": "https://example.test/source", "text_sha256": digest}}


def test_exact_approved_v2_inventory():
    ontology = read_json(REPO / "contracts/ner_v2/ontology.v2.json")
    assert list(ontology["entity_types"]) == ["OBJECT", "ACTOR", "CONCEPT", "METHOD", "PROCESS", "DATA_RESOURCE",
                                               "ATTRIBUTE", "RESULT", "CONDITION", "TASK", "MECHANISM"]
    assert list(ontology["relation_types"]) == ["IS_A", "HAS_PART", "HAS_MEMBER", "HAS_ATTRIBUTE", "HAS_RESULT", "USES",
        "USED_FOR", "APPLIED_TO", "PERFORMS", "PRODUCES", "DERIVED_FROM", "EVALUATED_ON", "MEASURES", "CHANGES", "CAUSES",
        "DEPENDS_ON", "EXPLAINS", "ASSOCIATED_WITH", "COMPARED_WITH", "LOCATED_IN"]
    assert len(label_descriptions(ontology)) == 11
    architectural_text = (REPO / "docs/architecture/SEQUENTIAL_HYBRID_KG_V2_DRAFT.md").read_text(encoding="utf-8")
    assert all(item["definition_ru"] in architectural_text for item in ontology["entity_types"].values())
    assert all(definition in architectural_text for definition in ontology["relation_types"].values())


def test_nfc_only_preserves_scientific_distinctions():
    assert safe_unicode("cafe\u0301\t cells") == "café cells"
    assert safe_unicode("NH₃ α β ² µ μ ℓ") == "NH₃ α β ² µ μ ℓ"
    assert safe_unicode("pH") != safe_unicode("ph")
    assert safe_unicode("NH₃") != safe_unicode("NH3")
    assert safe_unicode(safe_unicode("cafe\u0301\u00a0 cells")) == safe_unicode("cafe\u0301\u00a0 cells")


def test_normalization_plural_possessive_morphology_and_original(normalizer):
    text = "The researchers' models improved."
    start, end = text.index("researchers"), text.index(" improved")
    result = normalizer.normalize(text, start, end)
    assert result["surface_text"] == "researchers' models"
    assert result["normalized_form"] == "researcher model"
    assert any(t["morph"].get("Number") == "Plur" for t in result["tokens"])
    assert any(t["possessive"] for t in result["tokens"])
    for token in result["tokens"]:
        assert text[token["start"]:token["end"]] == token["text"]
    assert "ner" not in normalizer.nlp.pipe_names and "parser" not in normalizer.nlp.pipe_names


@pytest.mark.parametrize("phrase, expected", [("patient's cells", "patient cell"), ("patient’s cells", "patient cell"),
                                               ("cells' proteins", "cell protein"), ("cells’ proteins", "cell protein")])
def test_ascii_and_unicode_possessives(normalizer, phrase, expected):
    text = "The " + phrase + " changed."
    assert normalizer.normalize(text, 4, 4 + len(phrase))["normalized_form"] == expected


def test_technical_case_formula_unicode_repeated_and_overlapping_spans(normalizer):
    text = "🙂 NH₃ and NH₃ affect pH and p53."
    second = text.rindex("NH₃")
    result = normalizer.normalize(text, second, second + 3)
    assert result["normalized_form"] == result["surface_text"] == "NH₃"
    for term in ("pH", "p53"):
        a = text.index(term)
        assert normalizer.normalize(text, a, a + len(term))["normalized_form"] == term
    text = "The neural networks changed."
    full = normalizer.normalize(text, 4, 19)
    inner = normalizer.normalize(text, 11, 19)
    assert full["normalized_form"] == "neural network" and inner["normalized_form"] == "network"


def test_quotes_and_contractions_are_not_possessives(normalizer):
    text = "The 'cells' changed. It's stable."
    result = normalizer.normalize(text, 0, len(text))
    assert not any(t["possessive"] for t in result["tokens"])
    assert "'cell'" in result["normalized_form"]


def test_context_analysis_determinism_and_no_automatic_merge(normalizer):
    text = "The cells and cells divide."
    doc = normalizer.analyze(text)
    a, b = text.index("cells"), text.rindex("cells")
    first = normalizer.normalize(text, a, a + 5, analyzed=doc)
    last = normalizer.normalize(text, b, b + 5, analyzed=doc)
    assert first["normalized_form"] == last["normalized_form"] == "cell"
    assert first["tokens"][0]["start"] != last["tokens"][0]["start"]
    assert first == normalizer.normalize(text, a, a + 5)
    assert "canonical_entity_id" not in first
    with pytest.raises(ValueError, match="different source"):
        normalizer.normalize(text + "!", a, a + 5, analyzed=doc)


def test_partial_token_keeps_exact_surface(normalizer):
    text = "The microdevices changed."
    result = normalizer.normalize(text, 6, 12)
    assert result["surface_text"] == text[6:12]
    assert result["tokens"][0]["partial_token"]
    assert result["lemma"] == text[6:12]


def test_abbreviation_alignment_reverse_and_ambiguous_definitions():
    text = "We use principal component analysis (PCA). PCA (posterior communicating artery) matters."
    definitions = detect_abbreviations(text)
    assert {d["long_form"] for d in definitions} == {"principal component analysis", "posterior communicating artery"}
    assert {d["short_form"] for d in definitions} == {"PCA"}
    for d in definitions:
        assert text[d["short_start"]:d["short_end"]] == d["short_form"]
        assert text[d["long_start"]:d["long_end"]] == d["long_form"]
    assert detect_abbreviations(text) == definitions
    assert not detect_abbreviations("a random phrase (XYZ); sample (95).")


def test_abbreviation_is_candidate_not_replacement(normalizer):
    text = "Principal component analysis (PCA) helps. PCA works."
    start = text.rindex("PCA")
    result = normalizer.normalize(text, start, start + 3)
    assert result["normalized_form"] == "PCA"
    assert result["abbreviation_candidates"][0]["long_form"] == "Principal component analysis"


@pytest.mark.parametrize("start,end,surface", [(True, 2, "ab"), (0, False, "a"), (-1, 2, "a"),
                                                (0, 4, "abc"), (2, 2, ""), (0, 2, "bc")])
def test_invalid_offsets_rejected(start, end, surface):
    with pytest.raises(ValueError, match="exact Unicode"):
        validate_span("abc", start, end, surface)


def test_windows_preserve_codepoints_and_cover_text():
    text = "🙂 NH₃ α first second third fourth last"
    spans = list(windows(text, 4, 2))
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert text[spans[1][0]:spans[1][1]] == "α first second third"
    assert spans == list(windows(text, 4, 2))


def test_span_postprocessing_keeps_nested_multilabel_and_second_occurrence():
    text = "neural network; neural"
    def prediction(a, b, label, score):
        return {"start": a, "end": b, "text": text[a:b], "label": label, "score": score}
    raw = [(0, [prediction(0, 14, "object", .7), prediction(0, 6, "concept", .6),
                prediction(0, 14, "method", .5), prediction(16, 22, "concept", .9)]),
           (0, [prediction(0, 14, "object", .8)])]
    output = postprocess(text, raw, {"object": "OBJECT", "concept": "CONCEPT", "method": "METHOD"})
    assert len(output) == 4
    assert next(m for m in output if m["ner_label_hint"] == "OBJECT")["score"] == .8
    assert any(m["start"] == 16 for m in output)
    assert output == postprocess(text, list(reversed(raw)), {"object": "OBJECT", "concept": "CONCEPT", "method": "METHOD"})
    with pytest.raises(ValueError):
        postprocess(text, [(0, [prediction(0, 6, "concept", float('nan'))])], {"concept": "CONCEPT"})


def test_jsonl_validation_roundtrip_duplicate_hash_and_bad_types(tmp_path, document):
    path = tmp_path / "source.jsonl"
    write_jsonl(path, [document])
    original = path.read_bytes()
    assert read_documents(path) == [document]
    write_jsonl(path, read_documents(path))
    assert path.read_bytes() == original
    for alteration in ({"text": "changed"}, {"year": True}, {"document_id": None}, {"source_reference": {}},
                       {"revision_id": 42}, {"contract_version": "v1"}):
        with pytest.raises(ValueError):
            validate_document({**document, **alteration})
    write_jsonl(path, [document, document])
    with pytest.raises(ValueError, match="Duplicate"):
        read_documents(path)
    for raw in ('', '\n', '{bad json}\n', '[]\n'):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError):
            read_documents(path)


def test_mention_contract_jsonl_schema_and_rejected_corruptions(tmp_path, document):
    mention = {"contract_version": "ner-v2-mentions-1", "document_id": document["document_id"],
               "revision_id": document["revision_id"], "mention_id": "test:m1", "surface_text": "cells", "start": 14, "end": 19,
               "normalized_form": "cell", "lemma": "cell", "ner_label_hint": "OBJECT", "score": .7,
               "model_identifier": "synthetic-contract-fixture", "source_reference": document["source_reference"]}
    validate_mention(mention, document, ["OBJECT"])
    validate(mention, read_json(REPO / "contracts/ner_v2/mention.schema.json"))
    validate(document, read_json(REPO / "contracts/ner_v2/source.schema.json"))
    save_mentions(tmp_path / "mentions.jsonl", [mention], [document], ["OBJECT"])
    for alteration in ({"start": 0}, {"score": float("nan")}, {"score": True}, {"score": 1.1}, {"ner_label_hint": "FINAL"},
                       {"revision_id": "wrong"}, {"source_reference": {}}):
        with pytest.raises(ValueError):
            validate_mention({**mention, **alteration}, document, ["OBJECT"])


@pytest.mark.parametrize("index", [None, {}, {"a": [True]}, {"a": [-1]}, {"a": [1]}, {"a": [0], "b": [0]},
                                    {"a": [0, 2]}, {"a": []}, {"": [0]}, {"a": [20000]}])
def test_malformed_abstract_index_rejected(index):
    with pytest.raises(ValueError):
        reconstruct_abstract(index)


def test_abstract_reconstruction_restores_repetition_and_unicode():
    assert reconstruct_abstract({"cells": [0, 2], "🙂": [1]}) == "cells 🙂 cells"


def test_frozen_corpus_replay_balance_provenance_and_determinism(tmp_path, monkeypatch):
    snapshot = REPO / "contracts/ner_v2/openalex_snapshot.json"
    monkeypatch.chdir(tmp_path)
    def no_network(*args, **kwargs):
        raise AssertionError("Frozen replay must be offline")
    monkeypatch.setattr(OpenAlexClient, "get", no_network)
    first = prepare_corpus("first.jsonl", snapshot=snapshot)
    second = prepare_corpus("second.jsonl", snapshot=snapshot)
    assert first == second and Path("first.jsonl").read_bytes() == Path("second.jsonl").read_bytes()
    assert len(first) == len({d["document_id"] for d in first}) == 12
    assert len({d["discipline"] for d in first}) == 6
    for discipline in {d["discipline"] for d in first}:
        docs = [d for d in first if d["discipline"] == discipline]
        assert len(docs) == len({d["primary_topic"]["id"] for d in docs}) == 2
    for document in first:
        validate(document, read_json(REPO / "contracts/ner_v2/source.schema.json"))
        assert document["language"] == "en" and document["text"].strip()
        assert document["revision_id"] == "sha256:" + document["text_sha256"]
    saved = read_json(snapshot)
    assert saved["seed"] == 42
    history = next(f for f in saved["fields"] if f["discipline"] == "History")
    assert history["subfield"]["display_name"] == "History"
    assert history["filter"].startswith("primary_topic.subfield.id:")
    with pytest.raises(ValueError, match="Seed differs"):
        prepare_corpus("other.jsonl", snapshot=snapshot, seed=123)


def test_seeded_selection_is_independent_of_api_order():
    saved = read_json(REPO / "contracts/ner_v2/openalex_snapshot.json")
    candidates = [x["work"] for x in saved["selected"]]
    first = select_works(candidates, set(), 42)
    assert first == select_works(list(reversed(candidates)), set(), 42)
    assert len(first) == 2


def test_api_key_never_enters_failure_or_request_log(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "synthetic-secret-do-not-log")
    monkeypatch.setattr("cgraph.ner_v2.openalex.time.sleep", lambda _: None)
    class Session:
        def get(self, url, params, timeout):
            assert params["api_key"] == "synthetic-secret-do-not-log"
            class Response:
                status_code = 429
                headers = {"Retry-After": "0"}
            return Response()
    client = OpenAlexClient(Session())
    with pytest.raises(RuntimeError) as failure:
        client.get("works", {"sample": 30, "seed": 42})
    assert len(client.requests) == 3
    assert "synthetic-secret" not in str(failure.value) + json.dumps(client.requests)


def test_file_integrity_check_rejects_same_size_corruption(tmp_path):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"abcd")
    spec = {"size": 4, "sha256": sha256(b"abcd")}
    assert _matches(path, spec)
    path.write_bytes(b"abce")
    assert not _matches(path, spec)


def test_cli_help_validation_and_inference_gate_do_not_import_gliner(capsys):
    before = set(sys.modules)
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as blocked:
        main(["benchmark"])
    assert blocked.value.code == 2
    assert "--allow-inference" in capsys.readouterr().err
    assert not any(m.startswith("gliner") for m in set(sys.modules) - before)
