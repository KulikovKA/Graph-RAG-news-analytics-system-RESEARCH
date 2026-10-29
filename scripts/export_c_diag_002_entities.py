"""Explicit Gold-only diagnostic projection. Never used by ordinary inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgraph.common import file_sha256, fingerprint, read_json, write_json
from cgraph.inputs import load_inputs
from cgraph.labels import gold_entities, load_labels


if __name__ == "__main__":
    cfg = read_json("config/c_diag_002.json")
    docs = load_inputs(cfg)
    labels = load_labels(cfg, docs)
    out = Path("artifacts/c_diag_002/diagnostic_entities.json")
    if out.exists():
        raise ValueError("Gold diagnostic export is immutable")
    write_json(out, dict(purpose="GOLD ENTITIES ONLY; NOT END-TO-END", config_sha256=fingerprint(cfg),
                        manifest_sha256=file_sha256(cfg["manifest"]),
                        chunks={c["chunk_id"]: gold_entities(labels[d["document_code"]], c)
                                for d in docs for c in d["chunks"]}))
    print("Entity-only diagnostic projection written; no relations/qualifiers/answers exported")
