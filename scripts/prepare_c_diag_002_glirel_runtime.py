"""Recreate the canonical local backbone export for the official-loader probe.

Only serialization of strictly loaded, pinned pretrained weights; no training.
The measured retry reused the identical export from C-RESTART-001.
"""
from pathlib import Path
import os
import sys

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from cgraph.common import read_json
from cgraph.zeroshot_re import ZeroShotRE

torch.set_num_threads(4)
torch.manual_seed(42)
out = Path(sys.argv[1]) / "backbone"
if out.exists():
    raise ValueError("Canonical export must be new")
model = ZeroShotRE(read_json("config/c_restart_001_glirel.json"))
model.model.token_rep_layer.bert_layer.model.save_pretrained(out)
model.model.token_rep_layer.bert_layer.tokenizer.save_pretrained(out)
print("Canonical pretrained backbone exported; run the bounded official-loader probe separately")
