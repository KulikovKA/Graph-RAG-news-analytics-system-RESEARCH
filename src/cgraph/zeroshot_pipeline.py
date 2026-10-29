"""Inference-only C: independent NER -> pretrained RE -> provisional assertions."""
import time

from .common import fingerprint
from .rules import evidence, link, measurements
from .validation import validate_chunk, validate_evidence
from .zeroshot_re import candidate_pairs
from .zeroshot_rules import qualify


def process_chunk(doc, chunk, entities, re_model, ontology, cfg):
    timing = {}
    for entity in entities:
        entity["source_spans"] = [evidence(doc, chunk, entity["start"], entity["end"])]
    started = time.perf_counter()
    pairs = candidate_pairs(chunk, entities, cfg["zero_shot_re"]["sentence_window"])
    timing["candidates_ms"] = 1000 * (time.perf_counter() - started)
    started = time.perf_counter()
    predicted, served, skipped = re_model.predict(chunk, entities, pairs)
    timing["pretrained_re_ms"] = 1000 * (time.perf_counter() - started)
    started = time.perf_counter()
    assertions, proposals, records = [], [], []
    for i, pair in enumerate(pairs):
        h, t = pair["head"], pair["tail"]
        key = h["mention_id"], t["mention_id"]
        record = dict(head_mention_id=key[0], tail_mention_id=key[1], model_served=key in served,
                      decision_status="ABSTAIN", reasons=[], unsupported_labels=["MEETS_REQUIREMENT"])
        for relation in ontology["relations"]:
            result = predicted.get(((h["start"], h["end"]), (t["start"], t["end"]), relation))
            if result is None:
                continue
            identifier = f"{chunk['chunk_id']}:zs:{i}:{relation}"
            base = dict(assertion_id=identifier, head_mention_id=key[0], tail_mention_id=key[1],
                        relation_type=relation, score=result["score"],
                        evidence=[evidence(doc, chunk, result["start"], result["end"])])
            if "nli_scores" in result:
                base.update(nli_scores=result["nli_scores"], hypothesis=result["hypothesis"])
            rule = ontology["relations"][relation]
            domain_ok = h["kind"] in rule["head_kinds"] and t["kind"] in rule["tail_kinds"]
            fields, reason = (None, "ONTOLOGY_DOMAIN_RANGE")
            if domain_ok:
                # Use the actual RE evidence context, not the first candidate context.
                fields, reason = qualify(doc, chunk, dict(pair, start=result["start"], end=result["end"]),
                                         relation, identifier, ontology)
                if fields and result.get("nli_polarity", fields["polarity"]) != fields["polarity"]:
                    fields, reason = None, "NLI_SCOPE_POLARITY_CONFLICT"
            if fields:
                for span in base["evidence"] + fields["scope_evidence"]:
                    validate_evidence(doc, chunk, span)
                # Qualifier evidence must be registered exact subspans of evidence.
                registered = {s["span_id"]: s for s in fields["scope_evidence"]}
                for q in fields["qualifiers"]:
                    for ref in q["evidence_refs"]:
                        span = registered[ref]
                        if q["raw_text"] != span["text"] or not (result["start"] <= span["start"] < span["end"] <= result["end"]):
                            raise ValueError("Qualifier evidence is not grounded")
                assertions.append(dict(base, **fields, decision_status="ACCEPTED_PROVISIONAL",
                                       support_decision="PRETRAINED_RE_AND_EXACT_EVIDENCE",
                                       semantic_verification="NOT_INDEPENDENTLY_VERIFIED"))
            proposals.append(dict(base, domain_range_pass=domain_ok, scope_pass=fields is not None,
                                  evidence_pass=fields is not None, reason=reason,
                                  decision_status="ABSTAIN" if reason else "ACCEPTED_PROVISIONAL"))
            if reason:
                record["reasons"].append(reason)
            else:
                record["decision_status"] = "ACCEPTED_PROVISIONAL"
        if not record["reasons"] and record["decision_status"] == "ABSTAIN":
            record["reasons"] = ["NO_SCORE_ABOVE_FIXED_THRESHOLD" if key in served else "RE_CONTEXT_UNSUPPORTED"]
        records.append(record)
    timing["scope_evidence_ms"] = 1000 * (time.perf_counter() - started)
    output = dict(chunk_id=chunk["chunk_id"], source_text_sha256=chunk["text_sha256"], section=chunk["section"],
                  language=chunk["language"], section_start=chunk["section_start"], section_end=chunk["section_end"],
                  source_text=chunk["text"], entities=entities, candidate_pairs=records, raw_proposals=proposals,
                  assertions=assertions, measurements=measurements(doc, chunk), skipped_re_contexts=skipped,
                  re_inference_trace=getattr(re_model, "last_trace", []),
                  candidate_pairs_dropped_by_budget=0, stage_ms=timing)
    started = time.perf_counter()
    validate_chunk(doc, chunk, output, ontology)
    timing["validation_ms"] = 1000 * (time.perf_counter() - started)
    return output


def extract_document(doc, ner, re_model, ontology, cfg):
    started, cpu = time.perf_counter(), time.process_time()
    ner_before, re_before = ner.calls, re_model.calls
    chunks = []
    for chunk in doc["chunks"]:
        t = time.perf_counter()
        entities = ner.predict(chunk)
        ner_ms = 1000 * (time.perf_counter() - t)
        output = process_chunk(doc, chunk, entities, re_model, ontology, cfg)
        output["stage_ms"]["ner_ms"] = ner_ms
        chunks.append(output)
    concepts, identities = link(doc, chunks)
    return dict(schema_version="cgraph-v1", architecture_id="C-RESTART-001", mode="end_to_end",
                document_code=doc["document_code"], document_id=doc["document_id"], revision_id=doc["revision_id"],
                split="DEV", ontology_version="1.0", config_sha256=fingerprint(cfg), chunks=chunks,
                concepts=concepts, identity_decisions=identities,
                extraction_status="PARTIAL" if any(c["assertions"] for c in chunks) else "ABSTAINED",
                execution=dict(wall_ms=1000 * (time.perf_counter() - started), cpu_ms=1000 * (time.process_time() - cpu),
                               ner_calls=ner.calls - ner_before, re_calls=re_model.calls - re_before, llm_calls=0),
                limitations=["Scientific semantics of prompted labels unvalidated", "Qualifiers incomplete",
                             "Adjacent sentences only; no coreference", "Exact evidence is not semantic proof"])
