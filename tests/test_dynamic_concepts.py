"""Scientific-category semantics and frozen-layer safety; no database/models."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest

from cgraph.concepts import Concept, ConceptResolver, Entity, assess, class_form

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("dynamic_graph", SCRIPTS / "build_dynamic_concept_graph.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def entity(eid, doc, name="Fish", text="Fish populations inhabit aquatic marine ecosystems.", types=("OBJECT",), aliases=()):
    return Entity(eid, doc, name, types, aliases, (dict(document_id=doc, text=text,
        source_entity_id=eid, source_mention_ids=[eid + ":m"], revision_id="revision:" + doc),))


def test_two_independent_fish_contexts_share_one_concept_and_have_a_graph_path():
    a, b = entity("fish_a", "A"), entity("fish_b", "B")
    before = (asdict(a), asdict(b))
    result = ConceptResolver([], [a, b]).resolve_batch([a, b])
    assert len(result.concepts) == 1
    assert len(result.memberships) == 2
    assert {m["entity_id"] for m in result.memberships} == {"fish_a", "fish_b"}
    assert len({m["concept_id"] for m in result.memberships}) == 1
    assert before == (asdict(a), asdict(b))
    concept_id = result.concepts[0]["concept_id"]
    adjacency = {a.entity_id: {concept_id}, b.entity_id: {concept_id}, concept_id: {a.entity_id, b.entity_id}}
    assert b.entity_id in adjacency[next(iter(adjacency[a.entity_id]))]


def test_populations_remain_distinct_and_no_facts_can_be_inherited():
    a = entity("population_a", "A", "Fish populations", "Fish populations sampled in marine aquatic habitats had 10 deaths.")
    b = entity("population_b", "B", "Fish population", "Fish populations sampled in aquatic marine habitats had 0 deaths.")
    local_facts = [{"head": "population_a", "predicate": "deaths", "value": 10, "document": "A"},
                   {"head": "population_b", "predicate": "deaths", "value": 0, "document": "B"}]
    original = deepcopy(local_facts)
    result = ConceptResolver([], [a, b]).resolve_batch([a, b])
    assert len(result.memberships) == 2
    assert local_facts == original
    for m in result.memberships:
        assert m["factual"] is m["identity"] is m["fact_inheritance_allowed"] is False
        assert m["evidence"]["source_contexts"][0]["document_id"] == m["document_id"]


def test_same_string_different_meaning_does_not_create_concept():
    a = entity("a", "A", "Cell", "Cell populations in biological tissues.")
    b = entity("b", "B", "Cell", "Spreadsheet cell in computer software.")
    result = ConceptResolver([], [a, b]).resolve_batch([a, b], [("a", "b")])
    assert not result.memberships and not result.concepts
    assert all(d["status"] == "REJECT" for d in result.decisions)


@pytest.mark.parametrize("name", ["Study", "Results", "Resolution", "Reliability"])
def test_generic_or_polysemous_names_cannot_become_universal_hubs(name):
    a, b = entity("a", "A", name), entity("b", "B", name)
    result = ConceptResolver([], [a, b]).resolve_batch([a, b])
    assert not result.concepts and not result.memberships
    assert result.decisions and {d["status"] for d in result.decisions} == {"REJECT"}


def test_two_experiment_results_are_not_a_shared_scientific_class():
    a = entity("a", "A", "Improved clinical outcome", "Clinical patients improved after surgery.", ("RESULT",))
    b = entity("b", "B", "Improved clinical outcome", "Clinical patients improved after surgery.", ("RESULT",))
    result = ConceptResolver([], [a, b]).resolve_batch([a, b])
    assert not result.memberships and not result.concepts
    assert {d["reason"] for d in result.decisions} == {"experiment_specific_result_or_measurement"}


def test_same_name_and_type_without_context_is_rejected():
    a = Entity("a", "A", "Fish", ("OBJECT",))
    b = Entity("b", "B", "Fish", ("OBJECT",))
    result = ConceptResolver([], [a, b]).resolve_batch([a, b])
    assert not result.memberships
    assert all(d["status"] == "REJECT" for d in result.decisions)


def test_explicit_abbreviation_creates_one_meaningful_concept_with_stable_ids():
    a = entity("a", "A", "CNV", "Genetic genomic variation in genes.", ("CONCEPT",), ("Copy number variation",))
    b = entity("b", "B", "Copy number variation", "Genomic genetic variation in genes.", ("CONCEPT",), ("CNV",))
    resolver = ConceptResolver([], [a, b])
    result = resolver.resolve_batch([a, b])
    assert len(result.concepts) == 1 and len(result.memberships) == 2
    assert result.concepts[0]["name"] == "Copy number variation"
    assert result == resolver.resolve_batch([b, a])


def test_conflicting_abbreviation_expansions_are_rejected_even_in_same_domain():
    a = entity("a", "A", "CV", "Clinical patient disease treatment.", ("CONCEPT",), ("Cardiovascular volume",))
    b = entity("b", "B", "CV", "Clinical patient disease treatment.", ("CONCEPT",), ("Clinical variability",))
    result = ConceptResolver([], [a, b]).resolve_batch([a, b])
    assert not result.memberships and not result.concepts


def test_candidate_discovery_does_not_require_similar_to_and_reuses_catalog_id():
    a, b = entity("a", "A"), entity("b", "B")
    catalog = [Concept("existing:fish", "Fish", member_ids=("a",))]
    result = ConceptResolver(catalog, [a], [("a", "existing:fish")]).resolve_batch([b])
    assert not result.concepts
    assert len(result.memberships) == 1
    assert result.memberships[0]["concept_id"] == "existing:fish"
    assert result.decisions[0]["candidate_sources"] == ["catalog_or_repeated_name_alias", "name_token_retrieval"]


def test_incremental_novel_class_links_only_new_batch_and_affected_reference():
    a, b = entity("a", "A"), entity("b", "B")
    irrelevant = entity("unrelated", "C", "Marine habitat")
    result = ConceptResolver([], [a, irrelevant]).resolve_batch([b])
    assert len(result.concepts) == 1
    assert {m["entity_id"] for m in result.memberships} == {"a", "b"}


def test_similarity_and_shared_topic_alone_do_not_create_membership():
    a = entity("a", "A", "Fish")
    b = entity("b", "B", "Marine habitat")
    result = ConceptResolver([Concept("fish", "Fish", member_ids=("a",))], [a, b], [("a", "fish")]).resolve_batch([b], [("a", "b")])
    assert not result.memberships
    assert result.decisions[0]["status"] == "REJECT"
    assert "existing_SIMILAR_TO" in result.decisions[0]["candidate_sources"]


def test_relation_to_class_is_not_class_membership():
    a = entity("a", "A", "Green banks", "Green banking institutions provide financial market lending.", ("ACTOR",))
    b = entity("b", "B", "Co-investment of green banks", "Green banking institutions provide financial market lending.", ("ACTOR",))
    status, reason, _ = assess(b, Concept("green", "Green banks", member_ids=("a",)), a)
    assert status == "REJECT"


def test_single_document_does_not_generate_a_concept_for_each_entity():
    entities = [entity("a", "A"), entity("b", "A")]
    assert not ConceptResolver([], entities).resolve_batch(entities).concepts


def test_duplicate_catalog_aliases_are_refused():
    with pytest.raises(ValueError, match="Duplicate concept"):
        ConceptResolver([Concept("one", "Fish"), Concept("two", "Fishes", ("Fish",))], [])


def test_namespace_copy_preserves_parallel_facts_loops_and_provenance():
    ns = run.SOURCE_NS
    p = dict(namespace=ns, fingerprint="approved", document_id="A", source_json='{"provenance":"unchanged"}')
    nodes = {ns + ":a": dict(label="RELEXEntity", properties=dict(p, key=ns + ":a", entity_id="a"))}
    edges = {ns + ":" + rid: dict(type="RELEX_REL", head=ns + ":a", tail=ns + ":a",
             properties=dict(p, key=ns + ":" + rid, triple_id=rid)) for rid in ("fact1", "fact2")}
    before = deepcopy((nodes, edges))
    new_nodes, new_edges = run.copy_baseline(nodes, edges, "new")
    assert (nodes, edges) == before
    assert len(new_edges) == 2
    for key, r in new_edges.items():
        assert r["head"] == r["tail"] == run.TARGET_NS + ":a"
        assert r["properties"]["source_json"] == p["source_json"]
        assert r["properties"]["stage_2g_baseline_fingerprint"] == "approved"


def test_corrupt_or_foreign_baseline_is_refused_before_import():
    with pytest.raises(ValueError, match="Full baseline hash"):
        run.validate_baseline({}, {})


def test_arbitrary_length_factual_path_preserves_original_direction_and_document():
    nodes = {k: dict(label="RELEXEntity", properties={"document_id": "A"}) for k in ("a", "b", "c", "d")}
    edges = {rid: dict(type="RELEX_REL", head=h, tail=t,
                      properties=dict(document_id="A", relation_type="rel", source_triple_ids=[rid]))
             for rid, h, t in (("r1", "b", "a"), ("r2", "b", "c"), ("r3", "d", "c"))}
    path = run.factual_route("a", nodes, edges)
    assert len(path) == 3
    assert path[0]["from_key"] == "a" and path[0]["head"] == "b"
    assert all(r["document_id"] == "A" for r in path)
