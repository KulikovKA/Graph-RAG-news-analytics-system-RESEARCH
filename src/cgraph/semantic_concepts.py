"""Stage 2G.1 semantic candidate validation; the lexical resolver is unchanged."""
from __future__ import annotations

from decimal import Decimal
import re

from .concepts import Concept, ConceptResolver, GENERIC, assess, class_form, lexical_match, tokens, witnesses

VERSION = "hybrid-concepts-2g1-v1"
THRESHOLDS = tuple(format(Decimal(i) / Decimal(1000), ".3f") for i in range(950, 1001, 5))
MODES = ("text_name", "text_name_context")
SEMANTIC_POLICY = {
    "version": VERSION, "thresholds": list(THRESHOLDS), "cosine_is_evidence": False,
    "clinical_extra_witnesses": ["mammographic", "mammography", "breast", "esophageal", "carcinoma", "tumor", "tumour", "malignancy"],
    "source_bound_paraphrase_families": {"surgical_intervention": ["surgery", "surgical procedure"]},
    "new_concept_min_documents": 2, "no_transitive_similarity_clustering": True,
    "uncertainty": "REJECT", "factual": False, "identity": False, "fact_inheritance_allowed": False,
}


def semantic_witnesses(entity):
    result = witnesses(entity)
    ctx = tokens(" ".join(e["text"] for e in entity.evidence))
    clinical = ctx & set(SEMANTIC_POLICY["clinical_extra_witnesses"] + ["cancer", "patient", "patients", "clinical", "disease"])
    if len(clinical) >= 2:
        result["clinical"] = sorted(set(result.get("clinical", [])) | clinical)
    return result


def passes(score, threshold):
    """Compare the exact stored binary score to a requested decimal threshold."""
    return Decimal.from_float(float(score)) >= Decimal(threshold)


def pool_e5(hidden, attention_mask):
    import numpy as np
    mask = attention_mask.astype(np.float32)[..., None]
    pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
    return np.ascontiguousarray(pooled / np.linalg.norm(pooled, axis=1, keepdims=True), dtype=np.float32)


def cosine_matrix(left, right):
    """Float64 cosine; exactly identical stored vectors have cosine exactly 1.

    This is an algebraic identity, not tolerance-based rounding of nearby
    scores. Nonidentical vectors retain their full computed precision.
    """
    import numpy as np
    a, b = left.astype(np.float64), right.astype(np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    scores = np.clip(a @ b.T, -1, 1)
    index = {}
    for j, vector in enumerate(right):
        index.setdefault(vector.tobytes(), []).append(j)
    for i, vector in enumerate(left):
        for j in index.get(vector.tobytes(), []):
            scores[i, j] = 1.0
    return scores


def reference_records(catalog, entities, max_contexts=8):
    """Name/raw member contexts, leaving out the entire query document.

    No synthesized scientific description. Each context carries source IDs and
    offsets. A concept confined to the excluded document falls back to its name,
    with an explicit flag, rather than reusing the tested entity's evidence.
    """
    rows = []
    for c in sorted(catalog, key=lambda c: c.concept_id):
        members = [entities[i] for i in c.member_ids]
        for excluded in [None, *sorted({e.document_id for e in members})]:
            buckets = {}
            for e in members:
                if e.document_id == excluded:
                    continue
                for ctx in e.evidence:
                    key = (ctx["document_id"], ctx.get("start"), ctx.get("end"), ctx["text"])
                    buckets[key] = dict(ctx, contextual_entity_id=e.entity_id)
            by_doc = {}
            for key, ctx in sorted(buckets.items()):
                by_doc.setdefault(key[0], []).append(ctx)
            balanced = []
            for rank in range(max_contexts):
                for doc in sorted(by_doc):
                    if rank < len(by_doc[doc]) and len(balanced) < max_contexts:
                        balanced.append(by_doc[doc][rank])
            context = "\n".join(x["text"] for x in balanced)
            rows.append(dict(reference_id=c.concept_id + "|exclude:" + str(excluded),
                concept_id=c.concept_id, name=c.name, excluded_document_id=excluded,
                context_source_entity_ids=sorted({x["contextual_entity_id"] for x in balanced}),
                context_document_ids=sorted({x["document_id"] for x in balanced}),
                contexts=balanced, name_only_fallback=not balanced,
                text_name=c.name,
                text_name_context=f"Entity: {c.name}\nContext: {context}" if context else c.name))
    return rows


def explicit_definition(entity, concept):
    """Require a source-text predicate, not mere co-occurrence of class words."""
    names = (entity.name, *entity.aliases)
    targets = (concept.name, *concept.aliases)
    for ctx in entity.evidence:
        text = ctx["text"]
        for name in names:
            for target in targets:
                if class_form(name) == class_form(target):
                    continue
                # The entity's explicit surface must be the grammatical subject.
                pattern = (r"(?<!\w)" + re.escape(name) +
                    r"\s*(?:,\s*)?(?:is|are|means|denotes|refers to|also known as)\s+"
                    r"(?:(?:a|an|the|type of|class of|form of)\s+)*" + re.escape(target) + r"(?!\w)")
                match = re.search(pattern, text, re.I)
                if match:
                    return dict(document_id=ctx["document_id"], source_entity_id=ctx.get("source_entity_id"),
                                context_start=ctx.get("start"), local_start=match.start(), local_end=match.end(),
                                text=match.group(), relation="explicit_definition_or_classification")
    return None


def modifiers_conflict(entity, reference):
    """Specific qualified classes are not equivalent just because heads match."""
    left, right = class_form(entity.name), class_form(reference.name)
    for a, b in (("hip", "knee"), ("b cell", "t cell"), ("systolic", "diastolic"),
                 ("breast", "esophageal"), ("nasal", "corneal")):
        if (a in left and b in right) or (b in left and a in right):
            return [a, b]
    return []


class HybridConceptResolver(ConceptResolver):
    """The existing lexical policy plus an independent semantic validation path.

    Cosine only selects candidates. This method never uses cosine as proof.
    It accepts disjoint names when the original text explicitly defines their
    common class. Unsupported semantic resemblance remains REJECT.
    """

    def validate_semantic(self, entity, concept, reference):
        status, reason, evidence = assess(entity, concept, reference)
        evidence = dict(evidence, resolver_version=VERSION,
                        class_not_identity=True, factual=False, fact_inheritance_allowed=False)
        if status == "ACCEPT":
            return status, "hybrid_supported_lexical_class", evidence
        if class_form(concept.name) in GENERIC:
            return "REJECT", "generic_or_polysemous_category", evidence
        if not entity.evidence or not reference.evidence:
            return "REJECT", "insufficient_source_evidence", evidence
        if "RESULT" in entity.entity_types or "RESULT" in reference.entity_types:
            return "REJECT", "experiment_specific_result_or_measurement", evidence
        # A biochemical product and its synthase gene are not one kind of
        # object, even when the product's name dominates both embeddings.
        gene_a = "gene" in tokens(entity.name) or "genes" in tokens(entity.name)
        gene_b = "gene" in tokens(reference.name) or "genes" in tokens(reference.name)
        if gene_a != gene_b and ("genetic" in witnesses(entity) or "genetic" in witnesses(reference)):
            if class_form(concept.name) not in {"gene", "genes", "gene expression"}:
                return "REJECT", "gene_vs_product_or_other_object", evidence
        evidence["semantic_entity_senses"] = semantic_witnesses(entity)
        evidence["semantic_reference_senses"] = semantic_witnesses(reference)
        if not (set(semantic_witnesses(entity)) & set(semantic_witnesses(reference))):
            return "REJECT", "insufficient_or_incompatible_contextual_sense", evidence
        definition = explicit_definition(entity, concept)
        evidence["source_definition"] = definition
        overlap = set(entity.entity_types) & set(reference.entity_types)
        # Existing CONCEPT vs CONDITION typing is coarse. An explicitly named
        # clinical disease class can reconcile it without changing stored NER.
        clinical_disease = (
            lexical_match(entity, concept) is not None and
            {"clinical"} <= set(semantic_witnesses(entity)) & set(semantic_witnesses(reference)) and
            (tokens(concept.name) & {"cancer", "disease", "glaucoma", "anemia"}) and
            set(entity.entity_types) | set(reference.entity_types) <= {"CONCEPT", "CONDITION"}
        )
        family = set(SEMANTIC_POLICY["source_bound_paraphrase_families"]["surgical_intervention"])
        procedure_family = (class_form(entity.name) in family and class_form(reference.name) in family and
                class_form(concept.name) in family and
                "clinical" in semantic_witnesses(entity) and "clinical" in semantic_witnesses(reference) and
                set(entity.entity_types) | set(reference.entity_types) <= {"PROCESS", "TASK", "METHOD", "CONCEPT"})
        if not overlap and not clinical_disease and not procedure_family:
            return "REJECT", "incompatible_scientific_roles_or_abstraction", evidence
        if clinical_disease:
            evidence["typing_interpretation"] = "clinical_class_reading; original NER labels unchanged"
            return "ACCEPT", "source_grounded_clinical_class_with_expanded_context_witnesses", evidence
        if definition:
            return "ACCEPT", "explicit_source_definition_without_lexical_requirement", evidence
        if procedure_family:
            evidence["semantic_family"] = "surgical_intervention"
            evidence["abstraction"] = "class of clinical procedures, not the same operation or outcome"
            return "ACCEPT", "source_grounded_procedure_paraphrase_and_scientific_role", evidence
        if lexical_match(entity, concept):
            return "REJECT", reason, evidence
        return "REJECT", "semantic_resemblance_without_source_grounded_class_relation", evidence

    def validate_pair(self, left, right):
        conflict = modifiers_conflict(left, right)
        if conflict:
            return "REJECT", "contradictory_specific_qualifiers", dict(qualifiers=conflict)
        # Name selection is source-bound; no generated description or semantic
        # union-find. A proposal is supported by this particular pair only.
        names = [left.name, right.name]
        name = min(names, key=lambda s: (len(s.split()), len(s), s.casefold()))
        concept = Concept("proposal", name, member_ids=(left.entity_id, right.entity_id), existing=False)
        checks = [self.validate_semantic(e, concept, ref) for e, ref in ((left, right), (right, left))]
        if all(s == "ACCEPT" for s, _, _ in checks):
            return "ACCEPT", "independently_supported_pair_class", dict(name=name, checks=[x[2] for x in checks])
        rejected = next(x for x in checks if x[0] != "ACCEPT")
        return "REJECT", rejected[1], dict(name=name, checks=[x[2] for x in checks])
