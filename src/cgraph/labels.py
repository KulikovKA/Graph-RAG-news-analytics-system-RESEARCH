"""Explicit training/evaluation boundary. Never imported by inference modules."""
from pathlib import Path

from .common import read_json, sha256
from .inputs import DEV_CODES


def load_labels(cfg, documents):
    manifest = read_json(cfg["manifest"])
    labels = {}
    for doc in documents:
        code = doc["document_code"]
        if code not in DEV_CODES or doc["split"] != "DEV":
            raise ValueError("Labels are restricted to DEV")
        path = Path(cfg["data_dir"]) / "labels" / f"{code}.jsonl"
        if sha256(path.read_bytes()) != manifest["documents"][code]["gold_sha256"]:
            raise ValueError(f"Gold SHA-256 mismatch: {code}")
        gold = read_json(path)
        for key in ("document_code", "document_id", "revision_id", "split"):
            if gold[key] != doc[key]:
                raise ValueError(f"Label/source identity mismatch: {key}")
        if gold["ontology_hash"] != manifest["ontology_sha256"]:
            raise ValueError("Label ontology hash mismatch")
        chunks = {c["chunk_id"]: c for c in doc["chunks"]}
        if {c["chunk_id"]: c["text_sha256"] for c in gold["source_chunks"]} != {k: c["text_sha256"] for k, c in chunks.items()}:
            raise ValueError("Gold chunk inventory mismatch")
        for m in gold["mentions"]:
            if chunks[m["chunk_id"]]["text"][m["start"]:m["end"]] != m["surface_text"]:
                raise ValueError("Gold mention is not an exact Unicode slice")
        labels[code] = gold
    return labels


def gold_entities(gold, chunk):
    return [dict(mention_id=m["mention_id"], start=m["start"], end=m["end"],
                 surface_text=m["surface_text"], kind=m["kind"], contextual_role="UNKNOWN",
                 score=None) for m in gold["mentions"] if m["chunk_id"] == chunk["chunk_id"]]
