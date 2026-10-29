"""Checkpoint model-card control, strictly separate from DEV evaluation."""
import os
import sys

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import torch
from cgraph.common import config, write_json
from cgraph.inputs import load_ontology
from cgraph.zeroshot_nli import EntailmentRE

cfg = config("config/c_restart_001.json")
torch.set_num_threads(cfg["cpu_threads"])
torch.manual_seed(cfg["seed"])
model = EntailmentRE(cfg, load_ontology(cfg))
features = model.tokenizer(["A man is eating pizza", "A black race car starts up in front of a crowd of people."],
                          ["A man eats something", "A man is driving down a lonely road."], padding=True, return_tensors="pt")
with torch.inference_mode():
    scores = model.model(**features).logits.softmax(-1).tolist()
result = dict(purpose="Published NLI checkpoint model-card control, excluded from DEV quality/performance",
              source="https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall", probabilities=scores,
              predicted_labels=[model.model.config.id2label[max(range(3), key=lambda i: p[i])] for p in scores],
              pass_status=scores[0][1] > 0.5 and scores[1][0] > 0.5,
              checkpoint=model.identity["checkpoint"], revision=model.identity["revision"], strict_load=True)
write_json(sys.argv[1] if len(sys.argv) > 1 else "artifacts/nli_checkpoint_probe.json", result)
print(result, flush=True)
if not result["pass_status"]:
    raise RuntimeError("Pretrained NLI runtime failed its published model-card control")
