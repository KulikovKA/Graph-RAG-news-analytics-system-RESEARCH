"""Inference consumes source chunks and fitted parameters, never annotation files."""
import time

from .candidates import generate_pairs
from .common import fingerprint
from .rules import evidence, link, measurements, qualify
from .validation import validate_chunk


def process_chunk(doc, chunk, entities, classifier, ontology, cfg):
    timing = {}
    for entity in entities:
        entity["source_spans"] = [evidence(doc, chunk, entity["start"], entity["end"])]
    t = time.perf_counter()
    pairs, dropped = generate_pairs(chunk, entities, ontology, cfg)
    timing["candidates_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    decisions = classifier.predict(chunk, pairs)
    timing["relation_classifier_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    assertions, proposals, pair_records = [], [], []
    for i, (pair, decision) in enumerate(zip(pairs, decisions, strict=True)):
        pair_record = dict(head_mention_id=pair["head"]["mention_id"], tail_mention_id=pair["tail"]["mention_id"],
                           classifier_status=decision["status"], no_relation_score=decision["no_relation_score"],
                           decision_status=decision["status"], reasons=[], unsupported_labels=decision.get("unsupported_labels", []))
        for j, (relation, score) in enumerate(decision["predictions"]):
            assertion_id = f"{chunk['chunk_id']}:a{i}:{j}"
            proposal = dict(assertion_id=assertion_id, head_mention_id=pair["head"]["mention_id"],
                            tail_mention_id=pair["tail"]["mention_id"], relation_type=relation, score=score,
                            evidence=[evidence(doc, chunk, pair["start"], pair["end"])])
            fields, reason = qualify(doc, chunk, pair, relation, assertion_id, ontology)
            proposals.append(dict(proposal, decision_status="ABSTAIN" if reason else "ACCEPTED_PROVISIONAL", reason=reason))
            if reason:
                pair_record["reasons"].append(reason)
            else:
                assertions.append(dict(proposal, **fields, support_decision="CLASSIFIER_AND_LEXICAL_SCREEN",
                                       decision_status="ACCEPTED_PROVISIONAL"))
        if decision["status"] == "RELATION":
            pair_record["decision_status"] = "ACCEPTED_PROVISIONAL" if any(a["head_mention_id"] == pair_record["head_mention_id"] and a["tail_mention_id"] == pair_record["tail_mention_id"] for a in assertions) else "ABSTAIN"
        pair_records.append(pair_record)
    numeric = measurements(doc, chunk)
    timing["scope_qualifiers_ms"] = (time.perf_counter() - t) * 1000
    output = dict(chunk_id=chunk["chunk_id"], source_text_sha256=chunk["text_sha256"], section=chunk["section"],
                  language=chunk["language"], section_start=chunk["section_start"], section_end=chunk["section_end"],
                  source_text=chunk["text"], entities=entities, candidate_pairs=pair_records,
                  raw_proposals=proposals, assertions=assertions, measurements=numeric,
                  candidate_pairs_dropped_by_budget=dropped, stage_ms=timing)
    t = time.perf_counter()
    validate_chunk(doc, chunk, output, ontology)
    timing["validation_ms"] = (time.perf_counter() - t) * 1000
    return output


def extract_document(doc, ner, classifier, ontology, cfg):
    started, cpu = time.perf_counter(), time.process_time()
    before_ner, before_re = ner.calls, classifier.calls
    chunks = []
    for chunk in doc["chunks"]:
        t = time.perf_counter()
        entities = ner.predict(chunk)
        ner_ms = (time.perf_counter() - t) * 1000
        output = process_chunk(doc, chunk, entities, classifier, ontology, cfg)
        output["stage_ms"]["ner_ms"] = ner_ms
        chunks.append(output)
    t = time.perf_counter()
    concepts, identities = link(doc, chunks)
    linking_ms = (time.perf_counter() - t) * 1000
    return dict(schema_version="cgraph-v1", architecture_id="C-001", mode="end_to_end",
                document_code=doc["document_code"], document_id=doc["document_id"], revision_id=doc["revision_id"], split="DEV",
                ontology_version="1.0", config_sha256=fingerprint(cfg),
                extraction_status="PARTIAL" if any(c["assertions"] for c in chunks) else "ABSTAINED",
                chunks=chunks, concepts=concepts, identity_decisions=identities,
                execution=dict(wall_ms=(time.perf_counter() - started) * 1000, cpu_ms=(time.process_time() - cpu) * 1000,
                               linking_ms=linking_ms, ner_calls=ner.calls - before_ner, re_calls=classifier.calls - before_re,
                               llm_calls=0), limitations=["Partial qualifiers; unsupported attachment abstains", "No cross-sentence coreference",
                               "No independently calibrated semantic verifier", "Contextual roles beyond NER target substance: UNKNOWN"])
