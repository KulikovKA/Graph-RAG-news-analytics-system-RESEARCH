import json
from fractions import Fraction
from math import cos, radians, sin
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from cgraph.shared_concepts_v2 import Catalog, Config, Entity, Thresholds, aligned_cosine, digest, nesting_pairs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import shared_concepts_v2_pipeline as pipeline
from run_shared_concepts_v2 import read_configuration
from sweep_shared_concepts_v2 import configurations, sweep


METADATA = {"recipe": {"source": "synthetic"}}
RECIPE = digest(METADATA["recipe"])


def vector(angle, norm=1):
    data = np.zeros(384)
    data[:2] = np.array([cos(radians(angle)), sin(radians(angle))]) * norm
    return data


def entity(key, angle=0, context=None, *, doc=None, norm=.5, types=("METHOD",), name=None, expansions=None,
           revision="r1"):
    document = doc or f"doc-{key}"
    surface = name or key
    mention = dict(document_id=document, revision_id=revision, mention_start=0, mention_end=len(surface),
                   mention_text=surface, source_entity_id=key, source_mention_ids=[f"mention-{key}"])
    profile = dict(entity_id=key, name=surface, document_id=document, entity_types=list(types),
                   evidence=[mention], expansions=expansions or {}, topic="fixture")
    windows = [dict(text=surface, provenance=[mention])]
    return Entity.create(profile, windows, (vector(angle), vector(angle if context is None else context, norm)), RECIPE)


def config(ec=(.8, .8), cc=(.99, .99), delta="0"):
    return Config(Thresholds(*ec), Thresholds(*cc), Fraction(delta))


def add_concepts(catalog, **groups):
    for key, members in groups.items():
        trial = catalog.stabilize(members)
        assert trial.reason == "PASS"
        catalog.put(catalog.concept(key, 1, trial, (), ()))


def merge_all(catalog):
    catalog.prepare_cc()
    while catalog.merge_next():
        pass


def test_cosine_and_two_independent_channels():
    assert aligned_cosine([vector(0), vector(90, .25)], [vector(60), vector(90, .5)]) == pytest.approx([.5, 1])
    assert aligned_cosine(vector(0), -vector(0)) == -1
    assert Thresholds(.5, 1).passes([.5, 1])
    assert not Thresholds(.5, 1).passes([1, .99])
    assert not Thresholds(1, .5).passes([.99, 1])
    assert np.linalg.norm(entity("a", norm=.25).vectors[1]) == .25


@pytest.mark.parametrize("vectors", [
    np.zeros((2, 384)), np.full((2, 384), np.nan), np.full((2, 384), np.inf), np.ones((2, 383)),
])
def test_invalid_vectors(vectors):
    row = entity("a")
    with pytest.raises(ValueError):
        Entity.create(row.profile, json.loads(row.windows_json), vectors, RECIPE)


def test_profiles_reject_multiple_documents_and_empty_windows():
    row = entity("a")
    windows = json.loads(row.windows_json)
    with pytest.raises(ValueError, match="multiple documents"):
        Entity.create(row.profile | {"document_ids": ["doc-a", "doc-b"]}, windows, row.vectors, RECIPE)
    with pytest.raises(ValueError):
        Entity.create(row.profile, [], row.vectors, RECIPE)


@pytest.mark.parametrize("changes", [{"ee": {}}, {"delta": 1}, {"delta": -.1}, {"cc": {"name": 1.01, "context": 0}}])
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        Config.from_dict(config().to_dict() | changes)


def test_ee_block_direct_guards_and_delta_pairs():
    rows = [entity("a"), entity("b", doc="doc-a"), entity("c", 19), entity("d", 25),
            entity("r", types=("RESULT",)), entity("e", context=80), entity("i", types=("OBJECT",)),
            entity("p", expansions={"xyz": ["first word"]}), entity("q", expansions={"xyz": ["second token"]})]
    by = {e.entity_id: e for e in rows}
    edges, summary = pipeline.ee_edges(by, set(by), block=2)
    repeat, other_summary = pipeline.ee_edges(by, set(by), block=128)
    assert [(a, b) for a, b, *_ in edges] == [(a, b) for a, b, *_ in repeat]
    assert summary == other_summary
    assert summary["cross_document_pairs"] == len(rows) * (len(rows) - 1) // 2 - 1
    expected = set()
    for a, b, n, c in edges:
        assert (n, c) == pytest.approx(aligned_cosine(by[a].vectors, by[b].vectors), abs=1e-15)
        assert by[a].document_id != by[b].document_id
        expected.add((a, b))
    assert ("a", "c") in expected
    assert {("a", "b"), ("a", "d"), ("a", "e"), ("a", "i"), ("a", "r"), ("p", "q")}.isdisjoint(expected)
    assert all(summary["guard_hits"][key] for key in pipeline.contextual.RULES)
    incremental, counts = pipeline.ee_edges(by, {"d", "e"})
    assert {(a, b) for a, b, *_ in incremental} == {pair for pair in expected if {"d", "e"}.intersection(pair)}
    assert counts["cross_document_pairs"] == 2 * 7 + 1


def test_neighborhoods_direct_nontransitive_exact_dedup_and_overlap():
    catalog = Catalog([entity(x) for x in "abcdef"], config(), "test")
    catalog.add_edges([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    assert catalog.neighborhoods(catalog.entities) == {
        ("a", "b", "d"): ["a"], ("a", "b", "c"): ["b"], ("b", "c", "d"): ["c"], ("a", "c", "d"): ["d"],
    }
    catalog.add_edges([("e", "f")])
    catalog.initialize(catalog.entities)
    assert len(catalog.concepts) == 5
    assert len(catalog.by_entity["a"]) == 3
    pair = catalog.concepts[catalog.by_members[("e", "f")]]
    assert pair.seeds == ("e", "f")
    np.testing.assert_array_equal(pair.centroids, (vector(0), vector(0, .5)))


def test_ec_exact_fraction_and_cumulative_rechecks():
    rows = [entity(f"core-{i:02}") for i in range(17)] + [entity(str(x), context=x) for x in (13, 14, 15)]
    catalog = Catalog(rows, config(ec=(1, cos(radians(12))), delta=".10"), "test")
    result = catalog.stabilize(catalog.entities)
    assert result.reason == "EXCLUSION_BUDGET"
    assert [r["bad"] for r in result.rounds] == [("15",), ("14",), ("13",)]
    assert result.excluded == ("14", "15")
    assert result.rounds[-1]["cumulative_fraction"] == .15
    assert catalog.concepts == {}
    for delta, outcome in (("0", "EXCLUSION_BUDGET"), ("0.10", "PASS")):
        one = Catalog(rows[:9] + [entity("out", context=90)], config(delta=delta), "test")
        trial = one.stabilize(one.entities)
        assert trial.reason == outcome
        if outcome == "PASS":
            assert trial.excluded == ("out",)
            assert len(trial.rounds) == 2
            np.testing.assert_allclose(trial.sums, one.sums(trial.members), atol=1e-15)


def test_numeric_delta_keeps_input_decimal_precision(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"ec":{"name":0.8,"context":0.8},"cc":{"name":1,"context":1},'
                    '"delta":0.0999999999999999999}', encoding="utf-8")
    params = Config.from_dict(read_configuration(path))
    assert params.delta < Fraction(1, 10)
    catalog = Catalog([entity(str(j), context=90 if j == 9 else 0) for j in range(10)], params, "test")
    assert catalog.stabilize(catalog.entities).reason == "EXCLUSION_BUDGET"


def test_ec_simultaneous_prune_cannot_drop_document_or_use_zero_sum():
    rows = [entity("a", doc="d"), entity("b", doc="d"), entity("out", context=90)]
    catalog = Catalog(rows, config(delta=".67"), "test")
    trial = catalog.stabilize(catalog.entities)
    assert trial.reason == "MIN_MEMBERS_OR_DOCS"
    assert trial.excluded == ()
    assert trial.rounds[0]["bad"] == ("out",)
    catalog = Catalog([entity("a"), entity("b")], config(), "test")
    assert catalog.stabilize("ab", np.zeros((2, 384))).reason == "INVALID_SUM"


def test_ec_post_prune_exact_dedup_preserves_all_seed_lineage():
    catalog = Catalog([entity("a"), entity("b"), entity("x", context=90)], config(delta=".34"), "test")
    catalog.add_edges([("a", "b"), ("b", "x")])
    catalog.initialize(catalog.entities)
    assert len(catalog.concepts) == 1
    concept = next(iter(catalog.concepts.values()))
    assert concept.members == ("a", "b")
    assert concept.seeds == ("a", "b")
    assert len(concept.events) == 2


def test_cc_jaccard_priority_stable_ties_and_no_double_count():
    rows = [entity(k, i * 2, norm=.35 + i / 10) for i, k in enumerate("abcdefg")]
    catalog = Catalog(rows, config(cc=(.9, .9)), "test")
    add_concepts(catalog, a="ab", b="bc", c="cd", z="efg")
    catalog.prepare_cc()
    assert catalog.queue[0][:4] == (Fraction(-1, 3), (("a", "b"), ("b", "c")), ("a", 1), ("b", 1))
    assert catalog.merge_next()
    merged = next(c for c in catalog.concepts.values() if c.members == ("a", "b", "c"))
    np.testing.assert_allclose(merged.sums, catalog.sums("abc"))
    assert catalog.events[0]["jaccard"] == "1/3"
    assert catalog.cc_counts["eligible"] > 6
    while catalog.merge_next():
        pass
    final, = catalog.concepts.values()
    assert final.members == tuple("abcdefg")
    assert any(e["jaccard"] == "0" for e in catalog.events)
    np.testing.assert_allclose(final.centroids, catalog.sums("abcdefg") / 7)


def test_cc_equal_jaccard_is_independent_of_namespace_and_label():
    rows = [entity(key, 17 * index) for index, key in enumerate("abcd")]
    old_choices = set()
    config_hashes = set()
    for index in range(12):
        namespace = f"research:corpus100:namespace-{index % 3}:label-{index}"
        catalog = Catalog(rows, config(ec=(.93 if index % 2 else .94, .94), cc=(.8, .8)), namespace)
        config_hashes.add(catalog.config_hash)
        groups = {"ab": catalog.fresh_id(["initialize", tuple("ab")]),
                  "bc": catalog.fresh_id(["initialize", tuple("bc")]),
                  "cd": catalog.fresh_id(["initialize", tuple("cd")])}
        id_pair_ab_bc = sorted((groups["ab"], groups["bc"]))
        id_pair_bc_cd = sorted((groups["bc"], groups["cd"]))
        old_choices.add(id_pair_ab_bc < id_pair_bc_cd)
        add_concepts(catalog, **{cid: members for members, cid in reversed(list(groups.items()))})
        merge_all(catalog)
        assert sorted(c.members for c in catalog.concepts.values()) == [tuple("abc"), tuple("cd")]
        merged = catalog.concepts[catalog.by_members[tuple("abc")]]
        expected_id = catalog.fresh_id(["merge", *((key, 1) for key in id_pair_ab_bc), tuple("abc")])
        assert merged.concept_id == expected_id
    assert old_choices == {True, False}
    assert len(config_hashes) == 2


def test_cc_two_thresholds_and_failed_merge_keeps_catalog():
    rows = [entity("a", -25, 0), entity("b", 25, 0), entity("c", 25, 60), entity("d", 25, 60)]
    catalog = Catalog(rows, config(ec=(.9, .8), cc=(.5, .75)), "test")
    add_concepts(catalog, a="ab", b="cd")
    merge_all(catalog)
    assert catalog.cc_counts["threshold_reject"] == 1
    assert catalog.cc_counts["name_pass"] == 1 and catalog.cc_counts["context_pass"] == 0
    catalog = Catalog(rows, config(ec=(.9, .8), cc=(.5, .4)), "test")
    add_concepts(catalog, a="ab", b="cd")
    before = dict(catalog.concepts)
    merge_all(catalog)
    assert catalog.events[-1]["reason"] == "EXCLUSION_BUDGET"
    assert catalog.concepts == before


def test_cc_merge_applies_delta_and_updates_both_centroids():
    rows = [entity("a", context=-32), entity("b", context=22)] + [entity(k, context=20) for k in "cdefghij"]
    catalog = Catalog(rows, config(ec=(1, .88), cc=(1, .9), delta=".10"), "test")
    add_concepts(catalog, a="ab", b="cdefghij")
    merge_all(catalog)
    concept, = catalog.concepts.values()
    assert concept.members == tuple("bcdefghij")
    assert catalog.events[-1]["excluded"] == ("a",)
    assert catalog.events[-1]["rounds"][0]["bad"] == ("a",)
    np.testing.assert_allclose(concept.sums, catalog.sums(concept.members), atol=1e-15)


def test_nesting_dependency_is_rechecked_after_third_group_changes():
    catalog = Catalog([entity(k) for k in "1234567"], config(), "test")
    add_concepts(catalog, a="12", b="34", x="125", y="3467")
    assert nesting_pairs(list(catalog.concepts.values())) == {("a", "x"), ("b", "y")}
    merge_all(catalog)
    original = (("a", 1), ("b", 1))
    assert [e["reason"] for e in catalog.events if e["parents"] == original] == ["NESTING", "PASS"]
    assert {c.members for c in catalog.concepts.values()} == {tuple("1234"), tuple("1234567")}


def test_existing_nesting_append_both_directions_and_exact_collision():
    catalog = Catalog([entity(k) for k in "abcde"], config(), "test")
    add_concepts(catalog, inner="ab", outer="abc")
    before = dict(catalog.concepts)
    assert catalog.append("inner", "c") == "NESTING"
    assert catalog.append("inner", "e") == "NESTING"
    assert catalog.concepts == before
    assert catalog.append("outer", "d") == "PASS"
    assert catalog.append("inner", "c") == "PASS"
    assert catalog.append("inner", "c") == "NOOP"
    assert catalog.concepts["inner"].version == 2
    assert nesting_pairs(list(catalog.concepts.values())) == {("inner", "outer")}


def test_recreated_neighborhood_does_not_overwrite_expanded_concept():
    catalog = Catalog([entity(k) for k in "abc"], config(), "test")
    catalog.add_edges([("a", "b")])
    catalog.initialize({"a", "b"})
    key, = catalog.concepts
    assert catalog.append(key, "c") == "PASS"
    catalog.initialize({"a", "b"})
    assert catalog.concepts[key].members == ("a", "b", "c")
    assert catalog.by_members[("a", "b")] != key
    assert len(catalog.by_entity["a"]) == 2


def test_merge_exact_collision_cannot_retire_protected_concepts():
    catalog = Catalog([entity(k) for k in "abc"], config(), "test")
    add_concepts(catalog, one="ab", two="bc", superset="abc")
    before = dict(catalog.concepts)
    merge_all(catalog)
    assert catalog.concepts == before
    assert catalog.events[-1]["reason"] == "NESTING"
    assert catalog.cc_counts["nested"] == 2


def test_append_pre_and_post_ec_and_rollback_without_delta():
    rows = [entity("a", -20), entity("b", 20), entity("p", 45), entity("q", 25)]
    catalog = Catalog(rows, config(ec=(.9, .9), delta=".5"), "test")
    add_concepts(catalog, existing="ab")
    before = catalog.concepts["existing"]
    assert catalog.append("existing", "p") == "PRE_EC"
    assert catalog.append("existing", "q") == "POST_EC"
    assert catalog.concepts["existing"] is before
    assert catalog.events[-1]["scores"][0][0] < .9


def test_centroid_naming_tie_frequency_then_stable_id():
    catalog = Catalog([entity("a", name="First"), entity("z", name="Frequent"),
                       entity("b", name="Frequent")], config(), "test")
    add_concepts(catalog, group="azb")
    assert catalog.concepts["group"].representative == "b"
    assert catalog.concepts["group"].name == "Frequent"


def test_incremental_file_pipeline_mixed_ee_ec_and_multiple_membership(tmp_path):
    old = [entity(k, 0 if k in "ab" else 50) for k in "abcd"]
    base = pipeline.run_catalog(old, config(), "test", tmp_path / "base", metadata=METADATA)
    e = entity("e", 25)
    reversed_new = [entity("f", 90), entity("g", 90), e]
    folder = tmp_path / "updated"
    result = pipeline.run_catalog(old + reversed_new, config(), "test", folder, metadata=METADATA, base=base)
    assert result["ee"]["guarded_pairs"] == 1
    concepts = result["state"]["concepts"]
    assert {tuple(c["members"]) for c in concepts} == {("a", "b", "e"), ("c", "d", "e"), ("f", "g")}
    assert {c["concept_id"] for c in concepts if "e" in c["members"]} == {
        c["concept_id"] for c in base["state"]["concepts"]
    }
    assert result["statistics"]["memberships_per_entity"]["2"] == 1
    memberships = pipeline.load_checked(folder / "memberships.json")["memberships"]
    assert all(m["relation"] == "HAS_SHARED_CONCEPT" and not m["factual"] and m["evidence"] for m in memberships)
    assert all(m["generation"] == result["generation"] for m in memberships)
    resumed = pipeline.run_catalog(list(reversed(old + reversed_new)), config(), "test", folder,
                                   metadata=METADATA, base=base)
    assert digest(resumed) == digest(result)
    with pytest.raises(ValueError, match="Existing canonical profile changed"):
        pipeline.run_catalog([entity("a", 1)] + old[1:], config(), "test", tmp_path / "invalid",
                             metadata=METADATA, base=base)
    assert not (tmp_path / "invalid" / "catalog.json").exists()
    with pytest.raises(ValueError, match="Multiple source revisions"):
        pipeline.run_catalog(old + [entity("new", doc="doc-a", revision="r2")], config(), "test",
                             tmp_path / "revision", metadata=METADATA, base=base)


def test_append_into_empty_independent_catalog_and_duplicate_batch(tmp_path):
    base = pipeline.run_catalog([], config(), "test", tmp_path / "empty", metadata=METADATA)
    rows = [entity("a"), entity("b")]
    current = pipeline.run_catalog(rows, config(), "test", tmp_path / "first", metadata=METADATA, base=base)
    assert current["statistics"]["concepts"] == 1
    no_op = pipeline.run_catalog(rows, config(), "test", tmp_path / "same", metadata=METADATA, base=current)
    assert digest(no_op["state"]) == digest(current["state"])
    assert base["identity"]["config_hash"] == no_op["identity"]["config_hash"]


def test_incremental_new_neighborhood_uses_all_old_neighbors(tmp_path):
    old = [entity("a", 0), entity("b", 17), entity("c", 34)]
    params = config(cc=(1, 1))
    base = pipeline.run_catalog(old, params, "test", tmp_path / "base", metadata=METADATA)
    value = pipeline.run_catalog(old + [entity("new", 18)], params, "test", tmp_path / "next",
                                 metadata=METADATA, base=base)
    events = value["state"]["events"][len(base["state"]["events"]):]
    assert any(set(e["source"]) == {"a", "b", "c", "new"} and "b" in e["seeds"]
               for e in events if e["operation"] == "initialize")


def test_interrupted_merge_resumes_without_repeating_attempts(tmp_path, monkeypatch):
    rows = [entity(str(i), i * 17) for i in range(8)]
    params = config(ec=(0, 0), cc=(0, 0))
    reference = pipeline.run_catalog(rows, params, "test", tmp_path / "clean", metadata=METADATA)
    original = Catalog.merge_next

    def interrupt(self):
        if any(event["operation"] == "merge" for event in self.events):
            raise InterruptedError("checkpoint test")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(Catalog, "merge_next", interrupt)
        with pytest.raises(InterruptedError):
            pipeline.run_catalog(rows, params, "test", tmp_path / "interrupt", metadata=METADATA, checkpoint_every=1)
    checkpoint = pipeline.load_checked(tmp_path / "interrupt" / "checkpoint.json")
    assert checkpoint["attempts"] == 1
    assert not (tmp_path / "interrupt" / "catalog.json").exists()
    resumed = pipeline.run_catalog(rows, params, "test", tmp_path / "interrupt", metadata=METADATA)
    assert digest(reference["state"]) == digest(resumed["state"])
    assert reference["memberships_sha256"] == resumed["memberships_sha256"]


def test_experiment_reuses_ee_and_atomic_configuration_results(tmp_path, monkeypatch):
    result = tmp_path / "experiment"
    params = config()
    inputs = tmp_path / "config.json"
    inputs.write_text(json.dumps([dict(label="one", **params.to_dict()),
                                  dict(label="two", **config(delta=".1").to_dict())]), encoding="utf8")
    trials = configurations(inputs)
    entities = [entity("a"), entity("b")]
    monkeypatch.setattr(pipeline, "EXPECTED_EE", dict(entities=2, cross_document_pairs=1, raw_pairs=1,
                        guarded_pairs=1, entities_with_neighbors=2, documents_with_neighbors=2))
    initial = sweep(entities, METADATA, trials, result, "test")
    assert len(list((result / "ee_cache").glob("*.json"))) == 1
    assert pipeline.load_checked(result / "results.json")["complete"]
    monkeypatch.setattr(pipeline, "ee_edges", lambda *_: pytest.fail("EE recomputed during resume"))
    assert digest(sweep(entities, METADATA, trials, result, "test")) == digest(initial)
    with pytest.raises(ValueError, match="configuration list"):
        sweep(entities, METADATA, trials[:1], result, "test")


def test_exact_input_cache_reports_actual_session_hits(tmp_path, monkeypatch):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    model_files = {"tokenizer.json": pipeline.sha(model_dir / "tokenizer.json")}
    monkeypatch.setattr(pipeline, "read", lambda _path: {"identity": {"model_files_sha256": model_files}})
    loaded = []

    def worker(*_):
        loaded.append(True)
        return lambda texts: [vector(0) for _ in texts]

    monkeypatch.setattr(pipeline.previous, "e5_worker", worker)
    docs = {}
    profiles = []
    for key in "ab":
        row = entity(key, name="example").profile
        doc_id = row["document_id"]
        docs[doc_id] = dict(document_id=doc_id, revision_id="r1", source_url=f"https://example.org/{key}",
                            text="example signal.", text_sha256="fixture")
        row["evidence"][0]["source_url"] = docs[doc_id]["source_url"]
        profiles.append(row)
    budget = pipeline.ResourceBudget(SimpleNamespace(peak=1), perf_counter(), 10, 2**30)
    first, initial = pipeline.vectorize(profiles, docs, model_dir, tmp_path / "cache", 1, budget)
    again, later = pipeline.vectorize(profiles[::-1], docs, model_dir, tmp_path / "cache", 1, budget)
    assert {e.fingerprint for e in first} == {e.fingerprint for e in again}
    assert len(loaded) == 1
    assert initial["cache"]["encoded"] == 2
    assert later["cache"] == dict(unique_inputs=2, cache_hits=2, encoded=0, inference_seconds=0)
