"""One bounded official-loader retry; no DEV or Gold input."""
from pathlib import Path
import json
import os
import shutil
import sys
import time

START, CPU = time.perf_counter(), time.process_time()
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import psutil
from glirel import GLiREL
from cgraph.common import file_sha256, read_json, write_json

torch.set_num_threads(4)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
# The canonical backbone export contains the checkpoint's encoder, not a newly
# learned head. It was produced by the earlier strict full-checkpoint loader.
# Point the official loader at a local immutable copy; full GLiREL load is strict.
previous = Path(sys.argv[1])
root = Path("artifacts/c_diag_002/official_glirel")
root.mkdir(parents=True, exist_ok=True)
for p in (previous / "backbone").rglob("*"):
    if p.is_file():
        dest = root / "backbone" / p.relative_to(previous / "backbone")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(p, dest)
cfg = read_json("artifacts/models/glirel-large-v0/glirel_config.json")
cfg.update(model_name=str((root / "backbone").resolve()), device="cpu")
write_json(root / "glirel_config.json", cfg)
if not (root / "pytorch_model.bin").exists():
    os.link("artifacts/models/glirel-large-v0/pytorch_model.bin", root / "pytorch_model.bin")
model = GLiREL.from_pretrained(str(root), local_files_only=True, strict=True, map_location="cpu").to("cpu").eval().requires_grad_(False)
tokens = ['Derren', 'Nesbitt', 'had', 'a', 'history', 'of', 'being', 'cast', 'in', '"',
          'Doctor', 'Who', '"', ',', 'having', 'played', 'villainous', 'warlord', 'Tegana',
          'in', 'the', '1964', 'First', 'Doctor', 'serial', '"', 'Marco', 'Polo', '"', '.']
labels = ['country of origin', 'licensed to broadcast to', 'father', 'followed by', 'characters']
with torch.inference_mode():
    output = model.predict_relations(tokens, labels, threshold=0.0,
                                    ner=[[26, 27, 'PERSON', 'Marco Polo'], [22, 23, 'Q2989412', 'First Doctor']], top_k=1)
scores = [x["score"] for x in output if x["label"] == "characters"]
result = dict(purpose="OFFICIAL RUNTIME CONTROL ONLY; EXCLUDED FROM DEV QUALITY", source="https://github.com/jackboyla/GLiREL#usage",
              checkpoint="jackboyla/glirel-large-v0", revision="40a523e12a8432d6da364cf2a195a28755ff04d3",
              strict_load=True, official_loader=True, expected_characters_scores=[0.9923334121704102, 0.9915636777877808],
              output=output, characters_scores=scores, pass_status=len(scores) == 2 and min(scores) > 0.5,
              checkpoint_sha256=file_sha256(root / "pytorch_model.bin"),
              local_runtime_files_sha256={p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()},
              wall_s=time.perf_counter() - START, cpu_s=time.process_time() - CPU,
              peak_ram_bytes=psutil.Process().memory_info().peak_wset,
              blocker="NONE" if len(scores) == 2 and min(scores) > 0.5 else "Published control not reproduced after strict official load; root cause unresolved; STOP GLiREL")
write_json("reports/c_diag_002/glirel_probe.json", result)
print(json.dumps(result, ensure_ascii=True), flush=True)
