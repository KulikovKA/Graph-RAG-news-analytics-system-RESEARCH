"""Isolated source-only NER cost measurement; verify identical primary inventory."""
from pathlib import Path
import argparse
import runpy
import sys
import time

START, CPU = time.perf_counter(), time.process_time()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["control", "chem", "battery", "union"], required=True)
args = parser.parse_args()
namespace = runpy.run_path("scripts/run_c_diag_002.py", run_name="ner_benchmark_guard")
namespace["install_guard"](False)
import psutil
import torch
from cgraph.common import file_sha256, read_json, write_json
from cgraph.inputs import load_inputs
from cgraph.ner import ScientificNER
from cgraph.diagnostic_ner import PretrainedScientificNER, UnionNER

cfg = read_json("config/c_diag_002.json")
directory = Path("artifacts/c_diag_002") / args.variant
inventory = read_json(directory / "ner_inventory.json")
if inventory["run_sha256"] != file_sha256(directory / "run.json"):
    raise ValueError("NER benchmark inventory belongs to another primary run")
torch.manual_seed(cfg["seed"])
torch.set_num_threads(cfg["cpu_threads"])
torch.use_deterministic_algorithms(True)
models = []
if args.variant in {"control", "union"}:
    models.append(ScientificNER(cfg))
for alias in ("chem", "battery"):
    if args.variant in {alias, "union"}:
        models.append(PretrainedScientificNER(cfg["scientific_ner"][alias], cfg["diagnostic_protocol"]["model_files_contract"]))
ner = models[0] if len(models) == 1 else UnionNER(models)
predict_wall = predict_cpu = 0
total_entities = 0


def keys(entities):
    return sorted((m["start"], m["end"], m["kind"], m["surface_text"]) for m in entities)


for doc in load_inputs(cfg):
    saved = inventory["documents"][doc["document_code"]]
    for chunk in doc["chunks"]:
        before, cpu_before = time.perf_counter(), time.process_time()
        entities = ner.predict(chunk)
        predict_wall += time.perf_counter() - before
        predict_cpu += time.process_time() - cpu_before
        if keys(entities) != keys(saved[chunk["chunk_id"]]):
            raise ValueError("NER inventory differs from immutable primary run")
        total_entities += len(entities)
mem = psutil.Process().memory_info()
result = dict(variant=args.variant, purpose="Isolated source-only NER benchmark; no RE or Gold",
              wall_s=time.perf_counter() - START, cpu_s=time.process_time() - CPU,
              predict_wall_s=predict_wall, predict_cpu_s=predict_cpu, peak_ram_bytes=mem.peak_wset,
              total_entities=total_entities, inventory_matches_primary=True,
              encoder_calls=ner.encoder_calls, load_ms=ner.load_ms,
              primary_inventory_sha256=file_sha256(directory / "ner_inventory.json"),
              script_sha256=file_sha256(__file__), protocol_sha256=file_sha256("reports/C_DIAG_002_COST_PROTOCOL.md"))
if result["peak_ram_bytes"] >= 32_000_000_000:
    raise MemoryError("32 GB NER memory budget exceeded")
write_json(Path("artifacts/c_diag_002") / f"{args.variant}_ner_cost.json", result)
print(result, flush=True)
