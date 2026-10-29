"""Project immutable predictions to small NER-only inventories outside RAM measurement."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgraph.common import file_sha256, read_json, write_json
from cgraph.inputs import load_inputs

cfg = read_json("config/c_diag_002.json")
docs = load_inputs(cfg)
for variant in ("control", "chem", "battery", "union"):
    directory = Path("artifacts/c_diag_002") / variant
    run = read_json(directory / "run.json")
    inventory = {}
    for doc in docs:
        code = doc["document_code"]
        path = directory / f"{code}.json"
        if file_sha256(path) != run["prediction_files_sha256"][code]:
            raise ValueError("Primary prediction changed")
        prediction = read_json(path)
        inventory[code] = {c["chunk_id"]: [dict(start=e["start"], end=e["end"], kind=e["kind"], surface_text=e["surface_text"])
                                         for e in c["entities"]] for c in prediction["chunks"]}
    write_json(directory / "ner_inventory.json", dict(run_sha256=file_sha256(directory / "run.json"), documents=inventory))
