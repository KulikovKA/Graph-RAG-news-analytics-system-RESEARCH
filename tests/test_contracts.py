import copy
import json
from pathlib import Path

import pytest

from cgraph.candidates import generate_pairs, marked_text
from cgraph.common import config, read_json, sha256
from cgraph.evaluation import prf
from cgraph.inputs import DEV_CODES, load_inputs, load_ontology, validate_document
from cgraph.ner import windows
from cgraph.rules import abbreviation_map, evidence, link, measurements, qualify
from cgraph.validation import validate_evidence


@pytest.fixture
def cfg():
    return config()


@pytest.fixture
def ontology(cfg):
    return load_ontology(cfg)


def mention(mid, surface, text, kind, start=None):
    start = text.index(surface) if start is None else start
    return dict(mention_id=mid, surface_text=surface, start=start, end=start + len(surface), kind=kind)


def source(text):
    return dict(document_id="document", revision_id="revision"), dict(chunk_id="chunk", text=text, text_sha256=sha256(text.encode()))


def test_exact_frozen_ontology_and_example_free_runtime(cfg, ontology):
    manifest = read_json(cfg["manifest"])
    assert sha256(Path("contracts/ontology.v1.0.json").read_bytes()) == manifest["ontology_sha256"]
    assert len(ontology["entities"]) == 12 and len(ontology["relations"]) == 23
    text = Path(cfg["ontology"]).read_text(encoding="utf-8")
    assert '"positive_example"' not in text and '"negative_example"' not in text
    assert not any(code in text for code in DEV_CODES)
    original = read_json("contracts/ontology.v1.0.json")
    for relation, rule in ontology["relations"].items():
        for field in ("head_kinds", "tail_kinds", "required_qualifiers", "optional_qualifiers", "supported_polarities", "supported_modalities"):
            assert rule[field] == original["relations"][relation][field]


def test_partition_document_and_cluster_disjoint(cfg):
    partition = read_json(cfg["partition"])
    assert set(partition["train"]) | set(partition["validation"]) == DEV_CODES
    assert not set(partition["train"]) & set(partition["validation"])
    manifest = read_json(cfg["manifest"])["documents"]
    assert not {manifest[k]["cluster"] for k in partition["train"]} & {manifest[k]["cluster"] for k in partition["validation"]}


def test_tampered_source_and_identity_rejected(cfg):
    # Uses only the public source input, never labels.
    doc = load_inputs(cfg, "train")[0]
    spec = read_json(cfg["manifest"])["documents"][doc["document_code"]]
    validate_document(doc, spec)
    for change in (lambda d: d.update(split="HOLDOUT"), lambda d: d.update(revision_id="other"),
                   lambda d: d["chunks"][0].update(text=d["chunks"][0]["text"] + "!")):
        altered = copy.deepcopy(doc)
        change(altered)
        with pytest.raises(ValueError):
            validate_document(altered, spec)


def test_unicode_repeated_offsets_are_not_first_match():
    doc, chunk = source("α🙂 NH₃; NH₃ at 25 °C.")
    start = chunk["text"].rindex("NH₃")
    e = evidence(doc, chunk, start, start + 3)
    validate_evidence(doc, chunk, e)
    assert start == 8 and e["text"] == "NH₃"
    e["start"] -= 1
    with pytest.raises(ValueError):
        validate_evidence(doc, chunk, e)


def test_nested_entities_survive_candidate_generation(cfg, ontology):
    text = "A graphene sensor has graphene."
    _, chunk = source(text)
    entities = [mention("device", "graphene sensor", text, "DEVICE"), mention("inner", "graphene", text, "MATERIAL"),
                mention("last", "graphene", text, "MATERIAL", text.rindex("graphene"))]
    pairs, dropped = generate_pairs(chunk, entities, ontology, cfg)
    assert any(p["head"]["mention_id"] == "device" and p["tail"]["mention_id"] == "inner" for p in pairs)
    assert len(entities) == 3 and dropped == 0


def test_direction_markers_and_budget(cfg, ontology):
    text = "Graphene contains zinc oxide."
    _, chunk = source(text)
    entities = [mention("a", "Graphene", text, "MATERIAL"), mention("b", "zinc oxide", text, "MATERIAL")]
    pairs, _ = generate_pairs(chunk, entities, ontology, cfg)
    assert len(pairs) == 2
    assert marked_text(chunk, pairs[0]) != marked_text(chunk, pairs[1])
    limited = copy.deepcopy(cfg)
    limited["relations"]["max_pairs_per_chunk"] = 1
    pairs, dropped = generate_pairs(chunk, entities, ontology, limited)
    assert len(pairs) == 1 and dropped == 1


@pytest.mark.parametrize("text,polarity", [("The sensor detects NH3.", "POSITIVE"), ("The sensor does not detect NH3.", "NEGATIVE"),
                                            ("The sensor not only detects NH3.", "POSITIVE")])
def test_simple_negation_and_pseudonegation(text, polarity, cfg, ontology):
    doc, chunk = source(text)
    entities = [mention("s", "sensor", text, "DEVICE"), mention("g", "NH3", text, "ANALYTE")]
    pairs, _ = generate_pairs(chunk, entities, ontology, cfg)
    fields, reason = qualify(doc, chunk, pairs[0], "DETECTS", "a", ontology)
    assert reason is None and fields["polarity"] == polarity


@pytest.mark.parametrize("text,reason", [("The sensor does not detect NH3 but detects NO2.", "NEGATION_SCOPE"),
                                        ("The sensor detects NH3 at 25 °C.", "QUALIFIER_ATTACHMENT"),
                                        ("The sensor and NH3 are discussed.", "NO_EXPLICIT_PREDICATE")])
def test_scope_and_cooccurrence_abstain(text, reason, cfg, ontology):
    doc, chunk = source(text)
    entities = [mention("s", "sensor", text, "DEVICE"), mention("g", "NH3", text, "ANALYTE")]
    pairs, _ = generate_pairs(chunk, entities, ontology, cfg)
    fields, actual = qualify(doc, chunk, pairs[0], "DETECTS", "a", ontology)
    assert fields is None and reason in actual


def test_numeric_parsing_is_not_attachment():
    doc, chunk = source("At 25 °C, response was 20% to 43.5 ppb NH3 over 84 days.")
    parsed = measurements(doc, chunk)
    assert [(v["raw_value"], v["unit"]) for v in parsed] == [(25, "°C"), (20, "%"), (43.5, "ppb"), (84, "days")]
    for value in parsed:
        validate_evidence(doc, chunk, value["evidence"])
        assert value["attachment"] == "UNKNOWN"


def test_canonicalization_does_not_merge_chemistry_or_devices():
    text = "graphene graphene oxide sensor sensor"
    entities = [mention("g", "graphene", text, "MATERIAL"), mention("go", "graphene oxide", text, "MATERIAL", 9),
                mention("s1", "sensor", text, "DEVICE"), mention("s2", "sensor", text, "DEVICE", text.rindex("sensor"))]
    concepts, decisions = link({"revision_id": "r"}, [{"chunk_id": "c", "source_text": text, "entities": entities}])
    assert len(concepts) == 4 and not decisions
    assert abbreviation_map("reduced graphene oxide (rGO)") == {}  # no fuzzy chemistry aliases
    assert abbreviation_map("carbon nanotubes (CN)") == {}  # mismatched case/initials
    assert abbreviation_map("Carbon Nanotubes (CN)") == {"CN": "Carbon Nanotubes"}


def test_multiset_metrics_and_empty_policy():
    score = prf(["edge"], ["edge", "edge"])
    assert score["tp"] == 1 and score["fp"] == 1
    assert prf(["edge"], [])["precision"] is None
    assert prf(["edge"], [])["f1"] == 0


def test_windows_preserve_every_unicode_word():
    text = "α NH₃ 🙂 " * 100
    spans = list(windows(text, 20, 5))
    for i, char in enumerate(text):
        if not char.isspace():
            assert any(a <= i < b for a, b in spans)
    assert any(a < spans[i-1][1] for i, (a, _) in enumerate(spans) if i)


def test_schema_rejects_non_dev_and_nonzero_llm_calls():
    from jsonschema import ValidationError
    from cgraph.validation import validate_prediction
    pred = dict(schema_version="cgraph-v1", mode="gold_entities_re_diagnostic", document_code="D001", document_id="d", revision_id="r", split="DEV",
                chunks=[dict(chunk_id="c", source_text_sha256="a" * 64, entities=[], candidate_pairs=[], raw_proposals=[], assertions=[], stage_ms={})],
                execution=dict(wall_ms=0, cpu_ms=0, llm_calls=0))
    validate_prediction(pred)
    for field, value in [("document_code", "D093"), ("split", "CALIBRATION")]:
        invalid = copy.deepcopy(pred)
        invalid[field] = value
        with pytest.raises(ValidationError):
            validate_prediction(invalid)
    pred["execution"]["llm_calls"] = 1
    with pytest.raises(ValidationError):
        validate_prediction(pred)


def test_unknown_qualifiers_never_receive_full_qualified_credit():
    from cgraph.evaluation import document_metrics
    text = "The sensor detects NH3."
    entities = [dict(mention("s", "sensor", text, "DEVICE"), chunk_id="c"), dict(mention("a", "NH3", text, "ANALYTE"), chunk_id="c")]
    edge = dict(head_mention_id="s", tail_mention_id="a", relation_type="DETECTS", polarity="POSITIVE", modality="ASSERTED", qualifiers=[])
    gold = dict(mentions=entities, assertions=[dict(edge, support_status="SUPPORTED")])
    prediction = dict(chunks=[dict(chunk_id="c", entities=entities, assertions=[dict(edge, qualifier_status="UNKNOWN")],
                                  raw_proposals=[edge], candidate_pairs=[dict(head_mention_id="s", tail_mention_id="a", decision_status="ACCEPTED_PROVISIONAL")])])
    metrics = document_metrics(prediction, gold)
    assert metrics["relation_end_to_end"]["tp"] == 1
    assert metrics["qualified_relation"]["tp"] == 0 and metrics["qualified_relation"]["fn"] == 1


def test_evidence_with_wrong_revision_is_rejected():
    doc, chunk = source("NH₃")
    span = evidence(doc, chunk, 0, 3)
    span["revision_id"] = "other"
    with pytest.raises(ValueError):
        validate_evidence(doc, chunk, span)
