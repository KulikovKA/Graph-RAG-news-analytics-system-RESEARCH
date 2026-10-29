"""Auxiliary source-only NER workload profile, never Gold evaluation.

This repeats the same frozen battery NER to explain the RE query explosion.
The measured auxiliary probe overlapped the battery RE process; it is not an
independent speed benchmark and its cost is not subtracted from that run.
"""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ns = runpy.run_path("scripts/run_c_diag_002.py", run_name="cost_probe")
ns["install_guard"](False)
import torch
from cgraph.common import read_json, write_json
from cgraph.inputs import load_inputs, load_ontology
from cgraph.diagnostic_ner import PretrainedScientificNER
from cgraph.zeroshot_re import candidate_pairs
from cgraph.candidates import allowed_labels

cfg = read_json("config/c_diag_002.json")
torch.set_num_threads(4)
torch.manual_seed(42)
model = PretrainedScientificNER(cfg["scientific_ner"]["battery"], cfg["diagnostic_protocol"]["model_files_contract"])
ontology = load_ontology(cfg)
out = {}
for doc in load_inputs(cfg):
    rows = []
    for chunk in doc["chunks"]:
        entities = model.predict(chunk)
        pairs = candidate_pairs(chunk, entities, 1)
        rows.append(dict(entities=len(entities), pairs=len(pairs),
                         compatible_hypotheses=sum(len(allowed_labels(p["head"], p["tail"], ontology)) for p in pairs)))
    out[doc["document_code"]] = rows
write_json("artifacts/c_diag_002/battery_ner_cost_probe.json", dict(
    purpose="Auxiliary source-only CPU cost probe, excluded from primary metrics/timings; overlapped battery RE run",
    documents=out))
print(out)
