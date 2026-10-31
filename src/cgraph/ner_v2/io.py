"""Strict source-only JSONL and mention contracts, with Unicode codepoint offsets."""
import json
import math
from pathlib import Path

from cgraph.common import sha256
from . import CONTRACT_VERSION


def validate_document(document):
    if not isinstance(document, dict):
        raise ValueError("Document must be an object")
    for key in ("document_id", "text", "title", "source_url", "discipline"):
        if not isinstance(document.get(key), str) or not document[key].strip():
            raise ValueError(f"Missing/non-string document field: {key}")
    if document.get("contract_version") != "ner-v2-source-1":
        raise ValueError("Unsupported source contract")
    if not isinstance(document.get("source_reference"), dict):
        raise ValueError("Missing source reference")
    reference = document["source_reference"]
    if reference.get("source_url") != document["source_url"]:
        raise ValueError("Source URL/reference mismatch")
    if (type(document.get("year")) is not int or not 1000 <= document["year"] <= 3000 or
            not isinstance(document.get("field"), dict) or not document["field"].get("id")):
        raise ValueError("Invalid publication year/field metadata")
    if document.get("revision_id") is not None and not isinstance(document["revision_id"], str):
        raise ValueError("Invalid revision ID")
    if (sha256(document["text"].encode("utf-8")) != document.get("text_sha256") or
            reference.get("text_sha256") != document.get("text_sha256")):
        raise ValueError("Source text SHA-256 mismatch")
    return document


def read_documents(path):
    documents, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line {line_number}")
            try:
                doc = validate_document(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid source JSONL line {line_number}: {exc}") from None
            if doc["document_id"] in seen:
                raise ValueError("Duplicate document ID")
            seen.add(doc["document_id"])
            documents.append(doc)
    if not documents:
        raise ValueError("Empty source JSONL")
    return documents


def validate_span(text, start, end, surface):
    if (type(start) is not int or type(end) is not int or
            not 0 <= start < end <= len(text) or text[start:end] != surface):
        raise ValueError("Offsets must be exact Unicode codepoint source spans")


def validate_mention(mention, document, labels):
    for key in ("mention_id", "surface_text", "normalized_form", "lemma", "model_identifier"):
        if not isinstance(mention.get(key), str) or not mention[key]:
            raise ValueError(f"Invalid mention field: {key}")
    if mention.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Unsupported mention contract")
    if mention.get("document_id") != document["document_id"]:
        raise ValueError("Mention document mismatch")
    if mention.get("revision_id") != document.get("revision_id"):
        raise ValueError("Mention revision mismatch")
    if mention.get("source_reference") != document["source_reference"]:
        raise ValueError("Mention source reference mismatch")
    validate_span(document["text"], mention.get("start"), mention.get("end"), mention["surface_text"])
    score = mention.get("score")
    if (isinstance(score, bool) or not isinstance(score, (int, float)) or
            not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("Invalid mention score")
    if mention.get("ner_label_hint") not in labels:
        raise ValueError("Unknown preliminary label")
    return mention


def write_jsonl(path, records):
    """Atomic, deterministic UTF-8 JSONL; materialize before replacing the destination."""
    serialized = [json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                  for record in records]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("".join(row + "\n" for row in serialized), encoding="utf-8", newline="\n")
    temporary.replace(path)


def save_mentions(path, mentions, documents, labels):
    by_id = {d["document_id"]: d for d in documents}
    seen = set()
    for mention in mentions:
        if mention.get("document_id") not in by_id:
            raise ValueError("Mention refers to an unknown source document")
        validate_mention(mention, by_id[mention["document_id"]], labels)
        if mention["mention_id"] in seen:
            raise ValueError("Duplicate mention ID")
        seen.add(mention["mention_id"])
    write_jsonl(path, mentions)
