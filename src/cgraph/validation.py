from .common import sha256


def validate_prediction(prediction):
    from jsonschema import Draft202012Validator
    from .common import read_json
    Draft202012Validator(read_json("contracts/prediction.schema.json")).validate(prediction)


def validate_chunk(doc, chunk, output, ontology):
    entities = {m["mention_id"]: m for m in output["entities"]}
    if len(entities) != len(output["entities"]):
        raise ValueError("Duplicate mention ID")
    for mention in entities.values():
        if mention["kind"] not in ontology["entities"]:
            raise ValueError("Unknown entity kind")
        a, b = mention["start"], mention["end"]
        if not 0 <= a < b <= len(chunk["text"]) or mention["surface_text"] != chunk["text"][a:b]:
            raise ValueError("Entity evidence is not an exact source span")
        for span in mention.get("source_spans", []):
            validate_evidence(doc, chunk, span)
    for pair in output["candidate_pairs"]:
        if pair["head_mention_id"] not in entities or pair["tail_mention_id"] not in entities:
            raise ValueError("Dangling candidate pair")
    for proposal in output["raw_proposals"]:
        if proposal["relation_type"] not in ontology["relations"]:
            raise ValueError("Unknown relation proposal")
        for span in proposal["evidence"]:
            validate_evidence(doc, chunk, span)
    for assertion in output["assertions"]:
        h, t = entities[assertion["head_mention_id"]], entities[assertion["tail_mention_id"]]
        rule = ontology["relations"][assertion["relation_type"]]
        if h["kind"] not in rule["head_kinds"] or t["kind"] not in rule["tail_kinds"]:
            raise ValueError("Domain/range violation")
        if assertion["polarity"] not in rule["supported_polarities"] or assertion["modality"] not in rule["supported_modalities"]:
            raise ValueError("Unsupported polarity/modality")
        if not assertion["evidence"]:
            raise ValueError("Assertion lacks evidence")
        for e in assertion["evidence"] + assertion.get("scope_evidence", []):
            validate_evidence(doc, chunk, e)
        if not any(e["start"] <= min(h["start"], t["start"]) and max(h["end"], t["end"]) <= e["end"] for e in assertion["evidence"]):
            raise ValueError("Assertion evidence does not cover its endpoints")
        qtypes = {q["qualifier_type"] for q in assertion["qualifiers"]}
        if not set(rule["required_qualifiers"]) <= qtypes:
            raise ValueError("Missing required qualifiers")
        for q in assertion["qualifiers"]:
            if q["applies_to_assertion_id"] != assertion["assertion_id"]:
                raise ValueError("Qualifier attachment mismatch")
            if q["qualifier_type"] not in set(rule["optional_qualifiers"]) | set(rule["required_qualifiers"]):
                raise ValueError("Qualifier not allowed for relation")


def validate_evidence(doc, chunk, e):
    for key in ("document_id", "revision_id"):
        if e[key] != doc[key]:
            raise ValueError("Wrong evidence document/revision")
    if e["chunk_id"] != chunk["chunk_id"] or e["source_text_sha256"] != sha256(chunk["text"].encode("utf-8")):
        raise ValueError("Wrong evidence chunk/hash")
    if not 0 <= e["start"] < e["end"] <= len(chunk["text"]) or e["text"] != chunk["text"][e["start"]:e["end"]]:
        raise ValueError("Wrong evidence quote/offsets")
