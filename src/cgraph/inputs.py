"""Source-only loader. No label paths, gold objects or annotation imports."""
from pathlib import Path

from .common import read_json, sha256

DEV_CODES = frozenset({"D001", "D003", "D004", "D010", "D017", "D024", "D026", "D035", "D038", "D060", "D078", "D092"})


def load_inputs(cfg, subset="all"):
    manifest = read_json(cfg["manifest"])
    if set(manifest["documents"]) != DEV_CODES:
        raise ValueError("DEV allowlist mismatch")
    partition = read_json(cfg["partition"])
    codes = sorted(DEV_CODES if subset == "all" else partition[subset])
    if not set(codes) <= DEV_CODES:
        raise ValueError("Non-DEV document requested")
    result = []
    for code in codes:
        spec = manifest["documents"][code]
        raw = (Path(cfg["data_dir"]) / "inputs" / f"{code}.json").read_bytes()
        if sha256(raw) != spec["input_file_sha256"]:
            raise ValueError(f"{code}: input file SHA-256 mismatch")
        import json
        doc = json.loads(raw)
        validate_document(doc, spec)
        result.append(doc)
    return result


def validate_document(doc, spec):
    if doc.get("split") != "DEV" or doc.get("document_code") not in DEV_CODES:
        raise ValueError("Only DEV is permitted")
    for key in ("document_code", "document_id", "revision_id", "revision_content_hash", "cluster"):
        if doc[key] != spec[key]:
            raise ValueError(f"Source identity mismatch: {key}")
    if len(doc["chunks"]) != len(spec["chunks"]):
        raise ValueError("Chunk inventory mismatch")
    for chunk, expected in zip(doc["chunks"], spec["chunks"], strict=True):
        for key in ("chunk_id", "section", "ordinal", "section_start", "section_end", "language", "text_sha256", "stored_chunk_hash"):
            if chunk[key] != expected[key]:
                raise ValueError(f"Chunk identity mismatch: {key}")
        if sha256(chunk["text"].encode("utf-8")) != expected["text_sha256"]:
            raise ValueError("Source text SHA-256 mismatch")
        if chunk["section_end"] - chunk["section_start"] != len(chunk["text"]):
            raise ValueError("Invalid Unicode section offsets")


def load_ontology(cfg):
    """Only a label-free projection; original ontology contains DEV gold examples."""
    ontology = read_json(cfg["ontology"])
    manifest = read_json(cfg["manifest"])
    if sha256(Path(cfg["ontology"]).read_bytes()) != manifest["runtime_ontology_sha256"]:
        raise ValueError("Runtime ontology hash mismatch")
    if len(ontology["entities"]) != 12 or len(ontology["relations"]) != 23:
        raise ValueError("Frozen ontology inventory mismatch")
    return ontology
