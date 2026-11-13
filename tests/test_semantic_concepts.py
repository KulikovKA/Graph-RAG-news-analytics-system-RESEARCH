"""Read-only semantic sweep invariants; no model, downloads, or database."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np
import pytest

from cgraph.concepts import Concept, Entity, ConceptResolver, lexical_match
from cgraph.semantic_concepts import HybridConceptResolver, THRESHOLDS, cosine_matrix, explicit_definition, passes, pool_e5, reference_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_semantic_concepts as sweep


def e(eid, doc, name, text="Clinical patients had surgery and treatment.", types=("PROCESS",), aliases=()):
    return Entity(eid, doc, name, types, aliases, (dict(document_id=doc, text=text, source_entity_id=eid,
        start=0, end=len(text), source_mention_ids=[eid + ":m"], revision_id=doc + ":v"),))


def test_exact_decimal_thresholds_and_no_float32_boundary_rounding():
    assert len(THRESHOLDS) == 11 and THRESHOLDS[0] == "0.950" and THRESHOLDS[-1] == "1.000"
    assert all(Decimal(b) - Decimal(a) == Decimal(".005") for a, b in zip(THRESHOLDS, THRESHOLDS[1:]))
    assert not passes(np.float32(.95), "0.950")
    assert passes(1.0, "1.000")
    assert not passes(np.nextafter(1., 0.), "1.000")


def test_masked_mean_uses_attention_mask_before_l2_normalization():
    hidden = np.array([[[1., 0.], [0., 1.], [999., 999.]]], dtype=np.float32)
    pooled = pool_e5(hidden, np.array([[1, 1, 0]]))
    np.testing.assert_allclose(pooled, [[2**-.5, 2**-.5]], atol=1e-7)
    assert pooled.dtype == np.float32


def test_identical_cached_vectors_have_exact_unit_cosine_without_rounding_neighbors():
    a = np.array([[.131, .751, -.54], [.131, .751, -.54], [.131, .752, -.54]], dtype=np.float32)
    scores = cosine_matrix(a, a)
    assert scores[0, 1] == 1.0 and passes(scores[0, 1], "1.000")
    assert scores[0, 2] < 1.0 and not passes(scores[0, 2], "1.000")


def test_leave_one_document_out_removes_all_cohort_members_and_overlapping_contexts():
    entities = [e("a", "A", "patients"), e("a2", "A", "patients"), e("b", "B", "patients")]
    c = Concept("patients", "Patients", member_ids=("a", "a2", "b"))
    records = reference_records([c], {x.entity_id: x for x in entities})
    loo = next(r for r in records if r["excluded_document_id"] == "A")
    assert loo["context_source_entity_ids"] == ["b"]
    assert loo["context_document_ids"] == ["B"]
    assert all(ctx["document_id"] != "A" for ctx in loo["contexts"])


def test_single_document_catalog_falls_back_to_name_and_does_not_reuse_own_context():
    a = e("a", "A", "Surgery", "PRIVATE SELF CONFIRMATION TEXT")
    records = reference_records([Concept("s", "Surgery", member_ids=("a",))], {"a": a})
    loo = next(r for r in records if r["excluded_document_id"] == "A")
    assert loo["text_name_context"] == "Surgery" and loo["name_only_fallback"]
    assert "SELF CONFIRMATION" not in loo["text_name_context"]


def test_semantic_only_procedure_pair_can_accept_without_lexical_match():
    a = e("a", "A", "Surgical procedures")
    b = e("b", "B", "surgery")
    c = Concept("s", "surgery", member_ids=("b",))
    assert lexical_match(a, c) is None
    assert sweep.no_lexical_overlap(a, c)
    resolver = HybridConceptResolver([], [a, b])
    assert resolver.validate_semantic(a, c, b)[0] == "ACCEPT"
    assert resolver.validate_pair(a, b)[0] == "ACCEPT"
    assert not ConceptResolver([], [a, b]).resolve_batch([a, b]).memberships


def test_procedural_family_can_reconcile_method_and_process_roles():
    a = e("a", "A", "Surgical procedures", types=("METHOD",))
    b = e("b", "B", "surgery", types=("PROCESS",))
    assert HybridConceptResolver([], [a, b]).validate_pair(a, b)[0] == "ACCEPT"


def test_disjoint_names_can_accept_only_with_explicit_source_class_definition():
    a = e("a", "A", "Renal disorder", "Renal disorder is kidney disease. Clinical patients receive treatment.", ("CONDITION",))
    b = e("b", "B", "Kidney disease", types=("CONDITION",))
    c = Concept("kidney", "Kidney disease", member_ids=("b",))
    assert lexical_match(a, c) is None
    assert explicit_definition(a, c)["text"] == "Renal disorder is kidney disease"
    assert HybridConceptResolver([], [a, b]).validate_semantic(a, c, b)[0] == "ACCEPT"


def test_same_topic_and_types_without_class_evidence_remain_rejected():
    a, b = e("a", "A", "Treatment monitoring"), e("b", "B", "Surgery")
    c = Concept("s", "Surgery", member_ids=("b",))
    assert HybridConceptResolver([], [a, b]).validate_semantic(a, c, b)[0] == "REJECT"


def test_breast_cancer_source_context_can_recover_lexical_false_negative():
    a = e("a", "A", "breast cancer", "Mammographic density is a risk factor for breast cancer.", ("CONDITION",))
    b = e("b", "B", "cancer", "Patients with esophageal cancer received clinical treatment.", ("CONDITION",))
    c = Concept("cancer", "Cancer", member_ids=("b",))
    assert not ConceptResolver([c], [a, b], [("b", "cancer")]).resolve_batch([a]).memberships
    assert HybridConceptResolver([c], [a, b]).validate_semantic(a, c, b)[0] == "ACCEPT"


@pytest.mark.parametrize("pair", [("Limonene", "limonene synthase gene"), ("Total hip arthroplasty", "Total knee arthroplasty")])
def test_high_cosine_cannot_equate_distinct_scientific_objects(pair):
    text = "Clinical patients receive treatment. Gene expression and genes produced limonene."
    a, b = (e(i, d, n, text, ("CONCEPT", "OBJECT")) for i, d, n in zip(("a", "b"), ("A", "B"), pair))
    assert HybridConceptResolver([], [a, b]).validate_pair(a, b)[0] == "REJECT"


def test_semantic_discovery_scans_all_cross_doc_pairs_without_existing_similar(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    rows = [dict(entity_id="a", document_id="A", text_name="Surgical procedures"),
            dict(entity_id="b", document_id="B", text_name="surgery")]
    for mode in sweep.MODES:
        np.save(tmp_path / (mode + "_ee.npy"), np.array([[1, .98], [.98, 1.]]))
        np.save(tmp_path / (mode + "_ec.npy"), np.empty((2, 0)))
    result = sweep.discover(rows, [], [], set())
    assert len(result) == 2
    assert all(r["kind"] == "entity_entity" and not r["existing_similar_to"] for r in result)


def test_virtual_graph_is_additive_and_preserves_parallel_facts_and_loops():
    ns = sweep.NS
    nodes = {ns + ":a": dict(label="RELEXEntity", properties=dict(document_id="A"))}
    edges = {ns + ":" + rid: dict(type="RELEX_REL", head=ns + ":a", tail=ns + ":a", properties=dict(source_json=rid)) for rid in ("f1", "f2")}
    original = deepcopy((nodes, edges))
    accepted = [dict(source_entity_id="a", target_concept_id="concept", target_concept_name="Surgery", document_id="A")]
    vn, ve = sweep.virtual_graph(nodes, edges, accepted)
    assert (nodes, edges) == original
    assert len(ve) == 3 and all(ve[k] == r for k, r in edges.items())
    member = next(r for r in ve.values() if r["type"] == "HAS_SHARED_CONCEPT")
    assert member["properties"]["identity"] is member["properties"]["factual"] is member["properties"]["fact_inheritance_allowed"] is False


def test_entity_concept_discovery_uses_document_excluded_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    rows = [dict(entity_id="a", document_id="A", text_name="Surgery"), dict(entity_id="b", document_id="B", text_name="surgery")]
    refs = [dict(concept_id="c", excluded_document_id=None, reference_id="global", context_document_ids=["A"], name_only_fallback=False),
            dict(concept_id="c", excluded_document_id="A", reference_id="loo", context_document_ids=[], name_only_fallback=True)]
    for mode in sweep.MODES:
        np.save(tmp_path / (mode + "_ee.npy"), np.eye(2))
        np.save(tmp_path / (mode + "_ec.npy"), np.array([[.999, .8], [.96, .8]]))
    candidates = sweep.discover(rows, refs, [Concept("c", "Surgery")], set())
    assert len(candidates) == 2
    assert all(r["source_entity_id"] == "b" and r["reference_id"] == "global" for r in candidates)


def test_semantic_pairs_do_not_create_transitive_class_or_identity_links():
    a, b = e("a", "A", "surgery"), e("b", "B", "Surgical procedures")
    c = e("c", "C", "Medical records", types=("DATA_RESOURCE",))
    candidates = [dict(candidate_id=i, kind="entity_entity", source_entity_id=h, reference_entity_id=t,
                       embedding_mode="text_name", cosine_similarity=.99, qualifying_thresholds=["0.950"])
                  for i, h, t in (("ab", "a", "b"), ("bc", "b", "c"))]
    decisions, _ = sweep.evaluate(candidates, [a, b, c], [], set())
    assert {r["source_entity_id"] for r in decisions if r["status"] == "ACCEPT"} == {"a", "b"}
    assert all(r["status"] == "REJECT" for r in decisions if r["source_entity_id"] == "b" and r["candidate_id"] == "bc")


def test_extended_thresholds_and_vector_cutoffs_match_exact_decimal_gates():
    assert len(sweep.EXTENDED_THRESHOLDS) == 51
    assert sweep.EXTENDED_THRESHOLDS[0] == "0.700" and sweep.EXTENDED_THRESHOLDS[-1] == "0.950"
    for threshold in sweep.EXTENDED_THRESHOLDS:
        value = float(threshold)
        values = np.array([np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf)])
        assert list(values >= sweep.binary_cutoff(threshold)) == [passes(v, threshold) for v in values]


def test_topk_accounts_for_pruned_pairs_and_preserves_all_high_threshold_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "CACHE", tmp_path)
    rows = [dict(entity_id=i, document_id=i.upper(), text_name=i) for i in "abcd"]
    matrix = np.array([[1, .99, .98, .8], [.99, 1, .96, .71], [.98, .96, 1, .75], [.8, .71, .75, 1]])
    for mode in sweep.MODES:
        np.save(tmp_path / (mode + "_ee.npy"), matrix)
        np.save(tmp_path / (mode + "_ec.npy"), np.empty((4, 0)))
    distribution = []
    selected = sweep.discover(rows, [], [], set(), thresholds=("0.700", "0.950"),
        topk_entity=1, topk_concept=1, preserve_above="0.950", distribution=distribution)
    for r in distribution:
        assert r["full_candidates"] == r["reviewed_candidates"] + r["pruned_candidates"]
        if r["threshold"] == "0.700":
            assert r["full_candidates"] == 6 and r["pruned_candidates"] == 2
        else:
            assert r["full_candidates"] == r["reviewed_candidates"] == 3
    assert len(selected) == 8


def test_memoization_and_compact_provenance_preserve_frozen_scientific_decisions():
    a, b = e("a", "A", "Surgical procedures"), e("b", "B", "surgery")
    candidates = [dict(candidate_id=mode, kind="entity_entity", source_entity_id="a", reference_entity_id="b",
        embedding_mode=mode, cosine_similarity=.96, qualifying_thresholds=["0.700", "0.950"]) for mode in sweep.MODES]
    plain, plain_proposals = sweep.evaluate(candidates, [a, b], [], set())
    statistics = {}
    compact, compact_proposals = sweep.evaluate(candidates, [a, b], [], set(), compact=True, memoize=True, statistics=statistics)
    fields = ("candidate_id", "source_entity_id", "target_concept_id", "status", "reason")
    assert [tuple(r[k] for k in fields) for r in plain] == [tuple(r[k] for k in fields) for r in compact]
    assert [r["status"] for r in plain_proposals] == [r["status"] for r in compact_proposals]
    assert statistics["pair_evaluations"] == 1 and statistics["pair_calls"] == 2
    assert statistics["semantic_evaluations"] < statistics["semantic_calls"]
    assert "source_contexts_refs" in compact[0]["evidence"] and "source_contexts" not in compact[0]["evidence"]


def test_hybrid_extended_graph_keeps_validator_only_delta_when_semantic_gate_is_empty():
    ns = sweep.NS
    nodes = {ns + ":a": dict(label="RELEXEntity", properties=dict(document_id="A", name="a")),
             ns + ":b": dict(label="RELEXEntity", properties=dict(document_id="B", name="b")),
             ns + ":c": dict(label="RELEXConcept", properties=dict(concept_id="c", name="Surgery"))}
    edges = {ns + ":m": dict(type="HAS_SHARED_CONCEPT", head=ns + ":a", tail=ns + ":c", properties={})}
    delta = [dict(source_entity_id="b", target_concept_id="c", target_concept_name="Surgery", document_id="B",
                  semantic_only_membership=False, decision_id="B", status="ACCEPT", cosine_similarity=None)]
    result, accepted, _, _ = sweep.threshold_sweep(nodes, edges, [], [], [Concept("c", "Surgery")],
        thresholds=("0.950",), lexical_delta=delta, retain_evidence=False)
    assert len(result) == 2 and all(r["new_memberships"] == 1 for r in result)
    assert all(r["new_entity_concept_entity_bridges"] == 1 for r in result)
    assert {r["source_entity_id"] for r in accepted} == {"b"}


def test_compressed_decision_artifacts_are_deterministic_and_lossless(tmp_path):
    import hashlib
    rows = [dict(status="ACCEPT", name="хирургия", source_entity_id="a")]
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    sweep.gzip_jsonl(a, rows); sweep.gzip_jsonl(b, rows)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    assert sweep.read_gzip_jsonl(a) == rows


def test_extended_sweep_preserves_stage_2g1_validator_and_dictionaries_exactly():
    import hashlib
    import json
    root = Path(__file__).resolve().parents[1]
    previous = json.loads((root / "reports/entity_resolution/relex_base_t035/semantic_concept_sweep/manifest.json").read_text(encoding="utf-8"))
    current = hashlib.sha256((root / "src/cgraph/semantic_concepts.py").read_bytes()).hexdigest()
    assert current == previous["code_sha256"]["src/cgraph/semantic_concepts.py"]
    assert sweep.SEMANTIC_POLICY == previous["policy"]


def test_extended_semantic_only_analysis_does_not_call_regular_plurals_semantic_discoveries():
    a = e("a", "A", "peripheral blood mononuclear cells", types=("OBJECT",))
    b = e("b", "B", "cell", types=("OBJECT",))
    row = dict(source_entity_id="a", reference_entity_id="b", target_concept_id="new-cell", target_concept_name="cell",
               semantic_only_membership=True, status="ACCEPT", reason="frozen")
    sweep.refine_extended_semantic_flags([row], [a, b], [])
    assert not row["semantic_only_membership"] and not row["semantic_only_candidate_pair"]
    assert row["raw_token_semantic_only_2g1"] and row["status"] == "ACCEPT"


def test_posthoc_broad_hub_audit_flags_do_not_modify_scientific_decisions_or_keep_orphan_proposals():
    a = e("a", "A", "cell", types=("OBJECT",))
    b = e("b", "B", "peripheral blood mononuclear cells", types=("OBJECT",))
    rows = [dict(source_entity_id=i, target_concept_id="new-cell", target_concept_name="cell", document_id=d,
                 reference_entity_id=ref, status="ACCEPT", reason="frozen", semantic_only_membership=False)
            for i, d, ref in (("a", "A", "b"), ("b", "B", "a"))]
    original = deepcopy(rows)
    audit = sweep.extended_audit([], rows, [a, b], [], set(), [])
    assert all(r["audit_verdict"] == "SUSPICIOUS" for r in audit)
    assert rows == original
    assert sweep.audit_supported_delta(rows, {("a", "new-cell")}, set()) == []
