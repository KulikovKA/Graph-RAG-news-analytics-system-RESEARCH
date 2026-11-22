"""Read-only provenance and structure checks for the Stage 3A.5.2 benchmark."""

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[1]
SOURCE = REPORTS / "ner_v2/research_run2/input_abstracts.jsonl"
LOCK = REPORTS / "entity_resolution/relex_base_t035/dynamic_concept_graph/document_lock.json"
CATEGORY_COUNTS = {"A": 8, "B": 7, "C": 10, "D": 5}
REVIEWED_CATEGORIES = {"A", "B", "C"}
# SHA-256 утверждённых строк A из коммита e39c5ecbc7240b9c70dca7a923cc69ebfb34fc6b.
# B и исходные строки D зафиксированы по коммиту cd38cb4131109366adee4525bbbee01b714e9e0d.
BLOCK_SHA256 = {
    "A": "7a9c2340dda065d15c13dcd80666442766629214009d8515658e5701bedbacd4",
    "B": "162d8056b53b6f52d2d06d19282b94c0cd6a14109615ada525d1707831de160e",
    "D": "da70f8ccdce6017b9fbd0bba3c516ae06632432405415d939f2c7eb26b09b3a0",
}
LOCK_SHA256 = "0d472e0a6a5800bba52c20b90670cd7b40a7b2d2c9ebe64e74522fbd3168e37c"
FIXED = {
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
    lock_bytes = LOCK.read_bytes()
    require(hashlib.sha256(lock_bytes).hexdigest() == LOCK_SHA256, "Document lock SHA-256 mismatch")
    locked = json.loads(lock_bytes)
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
    human_reviewed = category in REVIEWED_CATEGORIES
    review_status = "HUMAN_REVIEWED" if human_reviewed else "DRAFT_UNREVIEWED"
    require(row["review_status"] == review_status, f"{key}: review status")
    require(row["requires_human_review"] is (not human_reviewed), f"{key}: review flag")
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
        require(nonempty(row["bridge_concept"]), f"{key}: missing bridge concept")
        require(row["cross_doc_type"] in {"shared_entity", "shared_concept", "comparative_multi_document"},
                f"{key}: cross-document type")
        require(type(row["reference_set_non_exhaustive"]) is bool, f"{key}: reference set flag")
        evidence_ids = [e["evidence_id"] for e in row["evidence"]]
        require(all(nonempty(e) for e in evidence_ids) and len(set(evidence_ids)) == len(evidence_ids),
                f"{key}: missing or duplicate evidence IDs")


def main():
    docs = documents()
    benchmark = HERE / "questions_draft.jsonl"
    ids, cited = set(), set()
    spans = reviewed = 0
    c_documents = set()
    c_ranges = defaultdict(list)
    for row in jsonl(benchmark):
        validate(row, docs)
        key = row["question_id"]
        require(key not in ids, f"Duplicate ID: {key}")
        ids.add(key)
        cited.update(row["document_contributions"])
        spans += len(row["evidence"])
        reviewed += row["review_status"] == "HUMAN_REVIEWED"
        if row["category"] == "C":
            c_documents.update(row["document_contributions"])
            for e in row["evidence"]:
                c_ranges[e["document_id"], e["start"], e["end"]].append(f"{key}/{e['evidence_id']}")
    expected_ids = [
        f"3A52-{category}{number:02d}"
        for category, count in CATEGORY_COUNTS.items()
        for number in range(1, count + 1)
    ]
    require(ids == set(expected_ids), "Benchmark question IDs differ from the original 30 IDs")
    expected_reviewed = sum(CATEGORY_COUNTS[category] for category in REVIEWED_CATEGORIES)
    require(reviewed == expected_reviewed, "Unexpected number of human-reviewed questions")
    lines = benchmark.read_bytes().splitlines(keepends=True)
    require([json.loads(line)["question_id"] for line in lines] == expected_ids, "Unexpected row order")
    offset = 0
    for category, count in CATEGORY_COUNTS.items():
        if category in BLOCK_SHA256:
            digest = hashlib.sha256(b"".join(lines[offset:offset + count])).hexdigest()
            require(digest == BLOCK_SHA256[category], f"Block {category} JSONL rows changed")
        offset += count
    c_spans = sum(len(uses) for uses in c_ranges.values())
    require((len(c_documents), c_spans, len(c_ranges)) == (17, 44, 39), "Unexpected C evidence totals")
    print(f"PASS (structure and exact source text): {len(ids)} questions, "
          f"{reviewed} human-reviewed (A-C), {len(ids) - reviewed} drafts (D), "
          f"A/B/D byte-identical, {spans} evidence occurrences.")
    print(f"Frozen corpus integrity: {len(docs)} documents and document_lock.json verified.")
    print(f"Gold coverage: {len(cited)}/{len(docs)}; uncited: {', '.join(sorted(docs.keys() - cited))}.")
    print(f"Block C: {len(c_documents)} documents, {c_spans} evidence occurrences, {len(c_ranges)} unique ranges.")
    for (document_id, start, end), uses in sorted(c_ranges.items()):
        if len(uses) > 1:
            print(f"  {document_id} [{start},{end}): {', '.join(uses)}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, TypeError, ValueError) as error:
        sys.exit(f"FAIL: {error}")
