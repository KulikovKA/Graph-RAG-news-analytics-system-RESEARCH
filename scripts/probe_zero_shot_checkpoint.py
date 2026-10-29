"""Published GLiREL example as a runtime regression probe, outside DEV metrics."""
import os
import sys

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
import torch
from cgraph.common import config, write_json
from cgraph.zeroshot_re import ZeroShotRE

torch.set_num_threads(4)
torch.manual_seed(42)
model = ZeroShotRE(config("config/c_restart_001_glirel.json"))
tokens = ['Derren', 'Nesbitt', 'had', 'a', 'history', 'of', 'being', 'cast', 'in', '"',
          'Doctor', 'Who', '"', ',', 'having', 'played', 'villainous', 'warlord', 'Tegana',
          'in', 'the', '1964', 'First', 'Doctor', 'serial', '"', 'Marco', 'Polo', '"', '.']
labels = ['country of origin', 'licensed to broadcast to', 'father', 'followed by', 'characters']
with torch.inference_mode():
    output = model.model.predict_relations(tokens, labels, threshold=0.0,
             ner=[[26, 27, 'ENTITY', 'Marco Polo'], [22, 23, 'ENTITY', 'First Doctor']], top_k=-1)
scores = [x['score'] for x in output if x['label'] == 'characters']
result = dict(purpose="Published model-card runtime probe; excluded from DEV quality/performance",
              source="https://github.com/jackboyla/GLiREL#usage", characters_scores=scores,
              pass_status=len(scores) == 2 and min(scores) > 0.5,
              strict_load=model.identity['strict_load'], checkpoint=model.identity['checkpoint'],
              revision=model.identity['revision'])
write_json(sys.argv[1] if len(sys.argv) > 1 else 'artifacts/checkpoint_probe.json', result)
print(result, flush=True)
if not result['pass_status']:
    raise RuntimeError("Published checkpoint probe failed; DEV evaluation must not be interpreted as model quality")
