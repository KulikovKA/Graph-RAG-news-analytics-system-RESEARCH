from pathlib import Path

from .common import read_json, sha256, write_json
from .inputs import DEV_CODES, validate_document


def fetch_dev(cfg, destination="artifacts/source_release"):
    """Fetch exactly 14 public pinned files; never download the rest of pilot-24."""
    from urllib.request import urlopen
    manifest = read_json(cfg["manifest"])
    files = {"ontology/v1.0/source_snapshot/dev_sources.jsonl": manifest["source_snapshot_sha256"],
             "owner_review_v1/RELEASE_MANIFEST.json": manifest["release_manifest_sha256"]}
    files.update({f"annotations/gold/{code}.jsonl": manifest["documents"][code]["gold_sha256"] for code in sorted(DEV_CODES)})
    base = Path(destination) / "graph_gold_pilot24"
    for relative, expected in files.items():
        url = f"https://raw.githubusercontent.com/KulikovKA/Graph-based-RAG-for-article-analysis/{manifest['source_commit']}/graph_gold_pilot24/{relative}"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        if sha256(raw) != expected:
            raise ValueError(f"Pinned upstream hash mismatch: {relative}")
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return import_dev(cfg, destination)


def import_dev(cfg, source_root):
    """Open only the DEV snapshot, release metadata and 12 allowlisted Gold files."""
    base = Path(source_root) / "graph_gold_pilot24"
    manifest = read_json(cfg["manifest"])
    snapshot = base / "ontology/v1.0/source_snapshot/dev_sources.jsonl"
    if sha256(snapshot.read_bytes()) != manifest["source_snapshot_sha256"]:
        raise ValueError("Source DEV snapshot differs from pinned release")
    if sha256((base / "owner_review_v1/RELEASE_MANIFEST.json").read_bytes()) != manifest["release_manifest_sha256"]:
        raise ValueError("Owner release manifest differs from pinned release")
    import json
    documents = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines()]
    if len(documents) != 12 or {d["document_code"] for d in documents} != DEV_CODES:
        raise ValueError("DEV inventory mismatch")
    for doc in documents:
        code = doc["document_code"]
        spec = manifest["documents"][code]
        validate_document(doc, spec)
        dest = Path(cfg["data_dir"]) / "inputs" / f"{code}.json"
        write_json(dest, doc)
        if sha256(dest.read_bytes()) != spec["input_file_sha256"]:
            raise ValueError("Canonical input serialization hash differs")
        raw = (base / "annotations/gold" / f"{code}.jsonl").read_bytes()
        if sha256(raw) != spec["gold_sha256"]:
            raise ValueError(f"Owner-approved Gold hash differs: {code}")
        dest = Path(cfg["data_dir"]) / "labels" / f"{code}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    return {"imported_documents": sorted(DEV_CODES), "split": "DEV"}
