"""Read-only provenance and structure checks for the Stage 3A.5.2 draft."""

import hashlib
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
SOURCE = REPORTS / "ner_v2/research_run2/input_abstracts.jsonl"
LOCK = REPORTS / "entity_resolution/relex_base_t035/dynamic_concept_graph/document_lock.json"
FIXED = {
    "review_status": "DRAFT_UNREVIEWED",
    "source_of_truth": "ORIGINAL_DOCUMENT",
    "generation_method": "EVIDENCE_GROUNDED_DRAFT",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path.name}:{line_number}: {error}") from error
            require(isinstance(row, dict), f"{path.name}:{line_number}: expected object")
            yield row


def documents():
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    by_id = {row["document_id"]: row for row in locked}
    require(len(by_id) == len(locked) == 30, "Invalid document lock")
    docs = {}
    for doc in jsonl(SOURCE):
        key = doc["document_id"]
        require(key in by_id and key not in docs, f"Unexpected source ID: {key}")
        for field in ("revision_id", "text_sha256", "title", "source_url"):
            require(doc[field] == by_id[key][field], f"{key}: {field}")
        digest = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
        require(doc["text_sha256"] == digest and doc["revision_id"] == f"sha256:{digest}", key)
        docs[key] = doc
    require(docs.keys() == by_id.keys(), "Source and lock membership differ")
    return docs


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate(row, docs):
    key = row["question_id"]
    category = row["category"]
    require(category in {"A", "B", "C", "D"}, f"{key}: category")
    require(re.fullmatch(rf"3A52-{category}[0-9]{{2}}", key), f"Invalid ID: {key}")
    for field in ("question", "expected_answer", "rationale", "limitations"):
        require(nonempty(row[field]), f"{key}: {field}")
    for field, value in FIXED.items():
        require(row[field] == value, f"{key}: {field}")
    require(row["requires_human_review"] is True, f"{key}: review flag")
    require(row["difficulty"] in {"easy", "medium", "hard"}, f"{key}: difficulty")
    require(row["support_status"] in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}, f"{key}: support")
    require(row["answerability"] in {"ANSWERABLE", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"},
            f"{key}: answerability")

    claims = {}
    require(isinstance(row["evidence"], list) and row["evidence"], f"{key}: evidence")
    for evidence in row["evidence"]:
        document_id = evidence["document_id"]
        require(document_id in docs, f"{key}: unknown document {document_id}")
        doc = docs[document_id]
        for field in ("revision_id", "text_sha256", "title", "source_url", "doi"):
            require(evidence[field] == doc[field], f"{key}: {document_id} {field}")
        start, end = evidence["start"], evidence["end"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(doc["text"]),
                f"{key}: invalid offsets")
        require(doc["text"][start:end] == evidence["quote"], f"{key}: quote at {start}:{end}")
        require(nonempty(evidence["supports"]), f"{key}: missing claim")
        claims.setdefault(document_id, []).append(evidence["supports"])
    require(row["document_contributions"] == claims, f"{key}: document contributions")
    count = row["required_document_count"]
    require(type(count) is int and count == len(claims), f"{key}: document count")
    if category in {"A", "B"}:
        require(count == 1, f"{key}: single document required")
    if category == "C":
        require(count >= 2, f"{key}: different documents required")


def main():
    docs = documents()
    ids, cited = set(), set()
    spans = 0
    for row in jsonl(HERE / "questions_draft.jsonl"):
        validate(row, docs)
        key = row["question_id"]
        require(key not in ids, f"Duplicate ID: {key}")
        ids.add(key)
        cited.update(row["document_contributions"])
        spans += len(row["evidence"])
    require(ids, "Empty draft")
    print(f"PASS (structure and exact source text): {len(ids)} questions, "
          f"{spans} evidence occurrences, {len(cited)}/{len(docs)} documents cited.")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError) as error:
        sys.exit(f"FAIL: {error}")
