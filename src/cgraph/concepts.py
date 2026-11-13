"""Deterministic, evidence-gated concept resolution over existing local entities.

No extraction, model calls, entity merges, or factual inference. A membership
means a shared scientific category; it never permits inheritance of local facts.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata

VERSION = "dynamic-concepts-v1"
STATUSES = ("ACCEPT", "REJECT")
POLICY = {
    "version": VERSION,
    "statuses": list(STATUSES),
    "uncertainty": "REJECT: insufficient evidence (no automatic edge)",
    "new_concept_min_documents": 2,
    "acceptance": "lexical class/explicit alias + source span + compatible contextual sense",
    "similarity_only_acceptance": False,
    "factual": False, "identity": False, "fact_inheritance_allowed": False,
}
GENERIC = set("study studies result results paper papers author authors method methods "
              "data information system systems model models process processes phenomenon "
              "quality performance improvement development evaluation testing test tests "
              "cooperation reliability resolution outcome outcomes parameters parameter "
              "values value thickness structure order formation change maintenance "
              "survival failure expression production".split())
STOP = set("the a an and or of in on to for with by from is are was were this that these "
           "those as at it its be been can may we our their has have using used use "
           "study studies results result different based new between effect effects "
           "analysis also two one other more than after before during which into".split())
# Sense witnesses are inspectable lexical rules, not learned predictions. They
# disambiguate lexical candidates; a common domain alone never creates a link.
SENSES = {
    "clinical": set("patient patients clinical surgery surgical disease diseases treatment "
                    "treated cancer glaucoma radiotherapy medical preoperative postoperative "
                    "complications mortality hemoglobin anemia knee eyes months hospital".split()),
    "genetic": set("gene genes genomic genome genomes genetics genetic expression "
                   "transcriptomics synthase overexpression ilvb ilvh acpp1 hla lipolytica".split()),
    "biological": set("fish fishes population populations aquatic marine species tissue tissues "
                      "cells cell biological biology diatom diatoms neurons astrocytes".split()),
    "economic": set("bank banks banking market markets financial economic economics income "
                    "grain farmers lending inflation exchange monetary price demand".split()),
    "physical": set("telescope lunar polarization brightness temperature magnetic optical "
                    "photons wavelength radiation regolith moon xray reflectivity".split()),
    "engineering": set("device devices power lightning shielding filtering cables generator "
                       "voltage electric electromagnetic station supply protection".split()),
    "computing": set("network networks packets algorithm algorithms wireless nodes wsn "
                     "computer computing spreadsheet software detection gpu runtime".split()),
    "historical": set("history historical archival interviews oral historian institutions "
                      "papal pope popes medals".split()),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def normalize(text):
    """Existing exact-audit NFC/case/space normalization, plus punctuation tokens."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def tokens(text):
    return set(re.findall(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", normalize(text))) - STOP


def class_form(text):
    """Only regular inflection and population syntax, never arbitrary head stripping."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalize(text))
    if words and words[-1] in {"population", "populations"}:
        words.pop()
    def singular(w):
        if len(w) > 4 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "is", "us")):
            return w[:-1]
        return w
    return " ".join(singular(w) for w in words)


@dataclass(frozen=True)
class Entity:
    entity_id: str
    document_id: str
    name: str
    entity_types: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    evidence: tuple[dict, ...] = ()

    @property
    def forms(self):
        return {class_form(s) for s in (self.name, *self.aliases) if class_form(s)}


@dataclass(frozen=True)
class Concept:
    concept_id: str
    name: str
    aliases: tuple[str, ...] = ()
    member_ids: tuple[str, ...] = ()
    existing: bool = True

    @property
    def forms(self):
        return {class_form(s) for s in (self.name, *self.aliases) if class_form(s)}


@dataclass
class Resolution:
    decisions: list[dict] = field(default_factory=list)
    concepts: list[dict] = field(default_factory=list)
    memberships: list[dict] = field(default_factory=list)


def witnesses(entity):
    context = tokens(" ".join(e["text"] for e in entity.evidence))
    return {sense: sorted(context & words) for sense, words in SENSES.items()
            if len(context & words) >= 2}


def lexical_match(entity, concept):
    if entity.forms & concept.forms:
        return "exact_or_explicit_alias"
    # Subtypes may reference a named class (breast cancer -> cancer). Do not
    # erase qualifiers to equate two differently qualified classes.
    for ef in entity.forms:
        for cf in concept.forms:
            prefix = ef[:-(len(cf) + 1)] if cf and ef.endswith(" " + cf) else ""
            # A relation to a class is not membership in that class: e.g.
            # 'co-investment of green banks' is not itself a green bank.
            if (prefix and not set(prefix.split()) & {"of", "with", "for", "from", "by", "to", "in"}
                    and cf not in GENERIC):
                return "explicit_named_class"
    return None


def assess(entity, concept, reference):
    lexical = lexical_match(entity, concept)
    left, right = witnesses(entity), witnesses(reference)
    shared_senses = sorted(set(left) & set(right))
    evidence = {
        "lexical_rule": lexical,
        "entity_forms": sorted(entity.forms), "concept_forms": sorted(concept.forms),
        "entity_types": list(entity.entity_types), "reference_types": list(reference.entity_types),
        "entity_sense_witnesses": left, "reference_sense_witnesses": right,
        "shared_senses": shared_senses,
        "source_contexts": list(entity.evidence),
        "reference_contexts": list(reference.evidence),
        "reference_entity_id": reference.entity_id,
        "reference_document_id": reference.document_id,
    }
    if class_form(concept.name) in GENERIC:
        return "REJECT", "generic_or_polysemous_category", evidence
    if not lexical:
        return "REJECT", "similarity_or_topic_without_lexical_class_evidence", evidence
    shared = entity.forms & reference.forms
    short = {class_form(a) for a in (*entity.aliases, entity.name, *reference.aliases, reference.name)
             if a.isupper() and len(a) <= 5}
    left_long = {f for f in entity.forms if len(f.split()) > 1}
    right_long = {f for f in reference.forms if len(f.split()) > 1}
    if shared and shared <= short and left_long and right_long and not left_long & right_long:
        return "REJECT", "conflicting_explicit_abbreviation_expansions", evidence
    if not entity.evidence or not reference.evidence:
        return "REJECT", "insufficient_source_evidence", evidence
    # A result is an observation, not a reusable category. Even identical values
    # in independent studies do not establish a common scientific concept.
    if "RESULT" in entity.entity_types or "RESULT" in reference.entity_types:
        return "REJECT", "experiment_specific_result_or_measurement", evidence
    if any(re.search(r"\d", f) for f in concept.forms) and not concept.existing:
        return "REJECT", "numeric_or_instance_specific_proposal", evidence
    if not shared_senses:
        return "REJECT", "insufficient_or_incompatible_contextual_sense", evidence
    # Context may reconcile noisy TASK/ATTRIBUTE labels, but an entity labelled
    # as an agent cannot become a measurement solely through a shared word.
    lt, rt = set(entity.entity_types), set(reference.entity_types)
    if not lt & rt and not (lt | rt <= {"TASK", "PROCESS", "ATTRIBUTE", "CONCEPT"}):
        return "REJECT", "incompatible_abstraction_or_entity_types", evidence
    return "ACCEPT", "source_grounded_lexical_class_and_contextual_sense", evidence


class ConceptResolver:
    """Resolve a supplied batch against an immutable, versioned catalog.

    The caller owns ingestion and persistence. Re-running with the same inputs
    yields identical IDs and decisions. A new class may also link its affected
    supplied references; there is no corpus access or NER/RE path here.
    """

    def __init__(self, catalog, reference_entities, existing_memberships=()):
        self.catalog = sorted(catalog, key=lambda c: c.concept_id)
        self.entities = {e.entity_id: e for e in reference_entities}
        self.existing = set(existing_memberships)
        index = defaultdict(set)
        for c in self.catalog:
            for f in c.forms:
                index[f].add(c.concept_id)
        if any(len(ids) > 1 for ids in index.values()):
            raise ValueError("Duplicate concept lexical identity in catalog")

    def resolve_batch(self, entities, similar_pairs=(), progress=lambda rows: rows):
        batch = sorted(entities, key=lambda e: e.entity_id)
        by_id = dict(self.entities)
        by_id.update({e.entity_id: e for e in batch})
        forms = defaultdict(set)
        for e in batch:
            for f in e.forms:
                forms[f].add(e.entity_id)
        # Proposals are data-derived repeated names/explicit abbreviations.
        # A single new document can propose a class already seen in references.
        for e in self.entities.values():
            for f in e.forms:
                if f in forms:
                    forms[f].add(e.entity_id)
        catalog = list(self.catalog)
        occupied = {f for c in catalog for f in c.forms}
        for form, ids in sorted(forms.items()):
            if form in occupied or len({by_id[i].document_id for i in ids}) < 2:
                continue
            # If an existing catalog class already represents these subtypes,
            # keep its stable ID instead of manufacturing a narrower duplicate.
            if any(form.endswith(" " + f) for f in occupied if f):
                continue
            # Name comes from the shortest explicit surface, not a generated term.
            equivalent = {f for f, other_ids in forms.items() if other_ids == ids}
            names = {a for i in sorted(ids) for a in (by_id[i].name, *by_id[i].aliases)
                     if class_form(a) in equivalent}
            name = min(names, key=lambda n: (-len(n.split()), len(n), normalize(n), n))
            catalog.append(Concept("concept:2g:" + digest(class_form(name))[:24], name,
                                   aliases=tuple(sorted(names)), member_ids=tuple(sorted(ids)), existing=False))
            occupied.update(equivalent)

        # Similarity and lexical-token sources only nominate candidates. Never
        # promote connected components/transitive similarity into concepts.
        neighbors = defaultdict(set)
        for a, b in similar_pairs:
            if a in by_id and b in by_id:
                neighbors[a].add(b); neighbors[b].add(a)
        result = Resolution()
        accepted = defaultdict(set)
        batch_ids = {e.entity_id for e in batch}
        affected_ids = batch_ids | {i for c in catalog if not c.existing for i in c.member_ids}
        for entity in progress([by_id[i] for i in sorted(affected_ids)]):
            for concept in sorted(catalog, key=lambda c: c.concept_id):
                if concept.existing and entity.entity_id not in batch_ids:
                    continue
                if (entity.entity_id, concept.concept_id) in self.existing:
                    continue
                refs = [by_id[i] for i in concept.member_ids if i in by_id and i != entity.entity_id]
                sources = []
                if lexical_match(entity, concept):
                    sources.append("catalog_or_repeated_name_alias")
                if any(r.entity_id in neighbors[entity.entity_id] for r in refs):
                    sources.append("existing_SIMILAR_TO")
                if tokens(entity.name) & tokens(concept.name) - GENERIC:
                    sources.append("name_token_retrieval")
                if not sources:
                    continue
                checks = [(*assess(entity, concept, r), r.entity_id) for r in refs]
                checks.sort(key=lambda c: (c[0] != "ACCEPT",
                                           by_id[c[3]].document_id == entity.document_id, c[3]))
                if checks:
                    status, reason, evidence, _ = checks[0]
                else:
                    status, reason, evidence = "REJECT", "insufficient_reference_evidence", {}
                decision_id = "decision:2g:" + digest([entity.entity_id, concept.concept_id, VERSION])[:24]
                row = dict(decision_id=decision_id, entity_id=entity.entity_id,
                           entity_name=entity.name, document_id=entity.document_id, concept_id=concept.concept_id,
                           concept_name=concept.name, proposed_new_concept=not concept.existing,
                           status=status, reason=reason, candidate_sources=sources, evidence=evidence)
                result.decisions.append(row)
                if status == "ACCEPT":
                    accepted[concept.concept_id].add(entity.entity_id)

        for concept in sorted(catalog, key=lambda c: c.concept_id):
            ids = accepted[concept.concept_id]
            if not concept.existing and len({by_id[i].document_id for i in ids}) < 2:
                for row in result.decisions:
                    if row["concept_id"] == concept.concept_id and row["status"] == "ACCEPT":
                        row.update(status="REJECT", reason="fewer_than_two_supported_documents")
                continue
            if not concept.existing and ids:
                result.concepts.append(dict(concept_id=concept.concept_id, name=concept.name,
                    normalized_name=class_form(concept.name), aliases=list(concept.aliases), resolver_version=VERSION,
                    source_document_ids=sorted({by_id[i].document_id for i in ids}),
                    contextual_target_ids=sorted(ids), factual=False, identity=False,
                    fact_inheritance_allowed=False))
            for row in result.decisions:
                if row["concept_id"] == concept.concept_id and row["status"] == "ACCEPT":
                    result.memberships.append(dict(
                        membership_id="membership:2g:" + digest([row["entity_id"], concept.concept_id])[:24],
                        entity_id=row["entity_id"], concept_id=concept.concept_id,
                        document_id=row["document_id"], decision_id=row["decision_id"],
                        resolver_version=VERSION, factual=False, identity=False,
                        fact_inheritance_allowed=False, evidence=row["evidence"]))
        return result
