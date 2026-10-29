"""Artificial minimal examples test invariants, never substitute DEV predictions."""
import ast
from pathlib import Path

import pytest

from cgraph.common import config, sha256
from cgraph.inputs import load_ontology
from cgraph.zeroshot_re import candidate_pairs, tokenize_exact
from cgraph.zeroshot_rules import linguistic_negations, qualify


def source(text):
    return dict(document_id="d", revision_id="r"), dict(chunk_id="c", text=text, text_sha256=sha256(text.encode()),
              section="abstract", language="en", section_start=0, section_end=len(text))


def mention(text, surface, kind, mid):
    a = text.index(surface)
    return dict(mention_id=mid, start=a, end=a + len(surface), surface_text=surface, kind=kind)


@pytest.mark.parametrize("text,count", [("NO x sensors detect NO2 and NO₂.", 0), ("No sensor detects NH3.", 1),
    ("NO does not respond to NH3.", 1), ("not only NH3", 0), ("without NOx", 1)])
def test_chemical_not_linguistic_negation(text, count):
    assert len(linguistic_negations(text)) == count


def test_pairs_unbounded_directed_adjacent_and_before_typing_gate():
    text = "A sensor " + "far " * 100 + "detects NH3. Graphene is discussed."
    _, chunk = source(text)
    entities = [mention(text, "sensor", "DEVICE", "s"), mention(text, "NH3", "ANALYTE", "a"),
                mention(text, "Graphene", "MATERIAL", "m")]
    pairs = candidate_pairs(chunk, entities)
    assert len(pairs) == 6
    assert {(p["head"]["mention_id"], p["tail"]["mention_id"]) for p in pairs} >= {("s", "a"), ("a", "s"), ("a", "m")}
    assert len(candidate_pairs(chunk, entities, 1)) == 2


def test_nested_multitype_and_unicode_token_offsets():
    text = "🙂 A NOx-sensor detects NH₃; NH₃."
    entities = [mention(text, "NOx-sensor", "DEVICE", "s"), mention(text, "NOx", "MATERIAL", "n"),
                mention(text, "NOx", "ANALYTE", "n2"), mention(text, "NH₃", "ANALYTE", "a")]
    tokens, offsets, spans = tokenize_exact(text, entities, 0, len(text))
    assert all(text[a:b] == token for token, (a, b) in zip(tokens, offsets))
    assert len(spans) == 3  # identical spans sent to RE only once
    for (a, b), (ca, cb) in spans.items():
        assert offsets[a][0] == ca and offsets[b - 1][1] == cb
    assert all((p["head"]["start"], p["head"]["end"]) != (p["tail"]["start"], p["tail"]["end"])
               for p in candidate_pairs(dict(text=text), entities))


@pytest.mark.parametrize("text,polarity", [("The NO x sensor detects NH3.", "POSITIVE"),
    ("The sensor does not detect NH3.", "NEGATIVE"), ("The sensor detects NH3 at 25 °C.", "POSITIVE")])
def test_scope_accepts_chemical_and_optional_temperature(text, polarity):
    doc, chunk = source(text)
    pair = dict(head=mention(text, "sensor", "DEVICE", "s"), tail=mention(text, "NH3", "ANALYTE", "a"), start=0, end=len(text))
    fields, reason = qualify(doc, chunk, pair, "DETECTS", "assertion", load_ontology(config()))
    assert reason is None and fields["polarity"] == polarity
    assert fields["qualifier_status"] == "UNKNOWN"
    if "25" in text:
        assert fields["qualifiers"][0]["unit"] == "°C"
        assert fields["qualifiers"][0]["applies_to_assertion_id"] == "assertion"


def test_negation_scopes_do_not_leak_across_but():
    text = "The sensor does not detect NH3 but detects NO2."
    doc, chunk = source(text)
    pair = dict(head=mention(text, "sensor", "DEVICE", "s"), tail=mention(text, "NO2", "ANALYTE", "a"), start=0, end=len(text))
    fields, reason = qualify(doc, chunk, pair, "DETECTS", "a", load_ontology(config()))
    assert fields is None and "CROSS_CLAUSE" in reason


def test_required_direction_is_physical_not_evaluative():
    text = "Heating reduces resistance."
    doc, chunk = source(text)
    pair = dict(head=mention(text, "Heating", "PROCESS", "h"), tail=mention(text, "resistance", "PROPERTY", "r"), start=0, end=len(text))
    fields, reason = qualify(doc, chunk, pair, "CHANGES_PARAMETER", "a", load_ontology(config()))
    assert reason is None and fields["qualifiers"][0]["raw_value"] == "decrease"
    chunk["text"] = text.replace("reduces", "changes")
    fields, reason = qualify(doc, chunk, pair, "CHANGES_PARAMETER", "a", load_ontology(config()))
    assert fields is None and "REQUIRED_DIRECTION" in reason


def test_zero_shot_config_cannot_claim_verified_support():
    cfg = config("config/c_restart_001.json")
    assert "relations" not in cfg
    ontology = load_ontology(cfg)
    caps = cfg["zero_shot_re"]["capabilities"]
    assert set(caps) == set(ontology["relations"])
    assert caps["MEETS_REQUIREMENT"]["status"] == "UNSUPPORTED"
    assert all(v["scientific_semantics_validated"] is False for v in caps.values())


def test_inference_code_has_no_training_or_annotation_dependency():
    for path in Path("src/cgraph").glob("zeroshot*.py"):
        if path.name == "zeroshot_evaluation.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {"labels", "relations", "sklearn"}
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"fit", "fit_transform", "backward", "get_optimizer"}


def test_stage_funnel_separates_wrong_type_from_missing_relation():
    from cgraph.zeroshot_evaluation import loss_audit
    text = "The sensor detects NH3."
    entities = [dict(mention(text, "sensor", "DEVICE", "s"), chunk_id="c"), dict(mention(text, "NH3", "ANALYTE", "a"), chunk_id="c")]
    edge = dict(head_mention_id="s", tail_mention_id="a", relation_type="DETECTS", support_status="SUPPORTED")
    gold = dict(mentions=entities, assertions=[edge])
    pred = dict(document_code="D001", chunks=[dict(chunk_id="c", entities=entities,
                candidate_pairs=[dict(head_mention_id="s", tail_mention_id="a", model_served=True)],
                raw_proposals=[], assertions=[])])
    funnel = loss_audit([pred], {"D001": gold})
    assert funnel["correct_assertions_lost"]["raw_re"] == 1
    assert funnel["correct_assertions_remaining"]["model_served"] == 1
    pred["chunks"][0]["entities"] = [dict(m, kind="OTHER") if m["mention_id"] == "a" else m for m in entities]
    funnel = loss_audit([pred], {"D001": gold})
    assert funnel["correct_assertions_remaining"]["untyped_endpoints"] == 1
    assert funnel["correct_assertions_lost"]["typed_endpoints"] == 1


def test_raw_model_edges_survive_optional_qualifiers_and_are_domain_checked():
    from cgraph.zeroshot_pipeline import process_chunk
    text = "The sensor detects NH3 at 25 °C."
    doc, chunk = source(text)
    entities = [mention(text, "sensor", "DEVICE", "s"), mention(text, "NH3", "ANALYTE", "a")]

    class FixedModelOutput:
        """Unit fixture only; never used in the DEV experiment."""
        def predict(self, chunk, entities, pairs):
            h, t = entities
            hs, ts = (h["start"], h["end"]), (t["start"], t["end"])
            return {(hs, ts, "DETECTS"): dict(score=0.8, start=0, end=len(text)),
                    (ts, hs, "DETECTS"): dict(score=0.7, start=0, end=len(text))}, {("s", "a"), ("a", "s")}, []

    cfg = config("config/c_restart_001.json")
    output = process_chunk(doc, chunk, entities, FixedModelOutput(), load_ontology(cfg), cfg)
    assert len(output["raw_proposals"]) == 2 and len(output["assertions"]) == 1
    assert output["assertions"][0]["qualifiers"][0]["raw_text"] == "at 25 °C"
    assert output["assertions"][0]["semantic_verification"] == "NOT_INDEPENDENTLY_VERIFIED"
    assert any(p["reason"] == "ONTOLOGY_DOMAIN_RANGE" for p in output["raw_proposals"])
