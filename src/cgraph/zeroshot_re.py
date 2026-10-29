"""Separate pretrained GLiREL component. No Gold, classifier fitting or LLM."""
import argparse
import re
import time
from pathlib import Path
from unittest.mock import patch

from .candidates import sentence_spans
from .common import read_json


def contexts(text, sentence_window=2):
    spans = list(sentence_spans(text))
    # Include single sentences as well as adjacent pairs; long pairs may abstain.
    return sorted(set((spans[i][0], spans[min(i + width, len(spans)) - 1][1])
                      for width in range(1, sentence_window + 1) for i in range(len(spans))))


def candidate_pairs(chunk, entities, sentence_window=2):
    """No distance/type/budget pruning before RE; retain both directions."""
    pairs = {}
    for start, end in contexts(chunk["text"], sentence_window):
        local = [e for e in entities if start <= e["start"] < e["end"] <= end]
        for h in local:
            for t in local:
                if (h["start"], h["end"]) == (t["start"], t["end"]):
                    continue  # alternate types of one span are not two endpoints
                key = h["mention_id"], t["mention_id"]
                pairs.setdefault(key, dict(head=h, tail=t, start=start, end=end))
    return list(pairs.values())


def tokenize_exact(text, entities, start, end):
    """Split also at mention boundaries; never round a character span to tokens."""
    cuts = {start, end}
    for e in entities:
        cuts.update((e["start"], e["end"]))
    tokens, offsets = [], []
    points = sorted(cuts)
    for a, b in zip(points, points[1:]):
        for match in re.finditer(r"\w+(?:[-_]\w+)*|\S", text[a:b]):
            tokens.append(match.group())
            offsets.append((a + match.start(), a + match.end()))
    starts = {a: i for i, (a, _) in enumerate(offsets)}
    ends = {b: i for i, (_, b) in enumerate(offsets)}
    spans = {}
    for e in entities:
        if e["start"] in starts and e["end"] in ends:
            spans[(starts[e["start"]], ends[e["end"]] + 1)] = (e["start"], e["end"])
    return tokens, offsets, spans


class ZeroShotRE:
    def __init__(self, cfg):
        import torch
        from transformers import AutoConfig, AutoModel
        from glirel import GLiREL
        from .zeroshot_assets import verify

        started = time.perf_counter()
        spec = cfg["zero_shot_re"]
        hashes = verify(cfg)
        backbone = str(Path(spec["backbone"]["local_dir"]).resolve())
        config = read_json(Path(spec["local_dir"]) / "glirel_config.json")
        config["model_name"] = backbone
        config["device"] = "cpu"
        # The checkpoint contains the COMPLETE backbone. Construct its skeleton
        # locally, then require every parameter to come from that checkpoint.
        def skeleton(name, *args, **kwargs):
            if str(name) != backbone:
                raise ValueError("Unexpected backbone requested")
            backbone_config = kwargs.get("config") or AutoConfig.from_pretrained(backbone, local_files_only=True)
            return AutoModel.from_config(backbone_config)

        with patch.object(AutoModel, "from_pretrained", side_effect=skeleton):
            self.model = GLiREL(argparse.Namespace(**config))
        state = torch.load(Path(spec["local_dir"]) / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True, assign=True)
        del state
        self.model.to("cpu").eval().requires_grad_(False)
        self.calls = self.encoder_calls = self.max_sequence_tokens = 0
        self.cfg = cfg
        tokenizer = self.model.token_rep_layer.bert_layer.tokenizer
        token_probe = tokenizer("runtime compatibility")["input_ids"]
        if token_probe[0] != tokenizer.cls_token_id or token_probe[-1] != tokenizer.sep_token_id:
            raise ValueError("Tokenizer omits model special tokens; incompatible checkpoint runtime")
        self.mapping = {v["prompt"]: k for k, v in spec["capabilities"].items() if v["status"] == "EXPERIMENTAL_ZERO_SHOT"}
        self.tokens_limit = spec["max_subtokens"]
        self.score_summary = dict(pair_label_scores=0, above_fixed_threshold=0, max_score=None)
        compute_score = self.model.compute_score

        def observe_scores(batch):
            result = compute_score(batch)
            # Observe actual logits without changing predictions or searching a threshold.
            probabilities = torch.sigmoid(result[0])
            if not torch.isfinite(probabilities).all():
                raise ValueError("Nonfinite RE scores")
            maximum = float(probabilities.max())
            self.score_summary["pair_label_scores"] += probabilities.numel()
            self.score_summary["above_fixed_threshold"] += int((probabilities > spec["threshold"]).sum())
            previous = self.score_summary["max_score"]
            self.score_summary["max_score"] = maximum if previous is None else max(previous, maximum)
            return result
        self.model.compute_score = observe_scores

        def count(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            length = int(ids.shape[-1])
            if length > self.tokens_limit:
                raise ValueError("RE token limit exceeded; silent truncation forbidden")
            self.encoder_calls += 1
            self.max_sequence_tokens = max(self.max_sequence_tokens, length)
        self.model.token_rep_layer.bert_layer.model.register_forward_pre_hook(count, with_kwargs=True)
        self.load_ms = 1000 * (time.perf_counter() - started)
        self.identity = dict(checkpoint=spec["checkpoint"], revision=spec["revision"],
                             license=spec["license"], device="cpu", dtype="float32", strict_load=True,
                             parameters=sum(p.numel() for p in self.model.parameters()), files=hashes,
                             train_documents=[], training_performed=False, capabilities=spec["capabilities"])

    def predict(self, chunk, entities, pairs):
        import torch
        predictions, served, skipped = {}, set(), []
        tokenizer = self.model.token_rep_layer.bert_layer.tokenizer
        for start, end in contexts(chunk["text"], self.cfg["zero_shot_re"]["sentence_window"]):
            local = [e for e in entities if start <= e["start"] < e["end"] <= end]
            tokens, offsets, spans = tokenize_exact(chunk["text"], local, start, end)
            if len(spans) < 2:
                continue
            # Mirrors GLiREL.compute_score's prompt: one [REL] and one label
            # word per type, then [SEP]. No entity markers in markerv1.
            prompt = [v for label in self.mapping for v in ("[REL]", label)] + ["[SEP]"]
            sequence_length = len(tokenizer(prompt + tokens, is_split_into_words=True)["input_ids"])
            if sequence_length > self.tokens_limit or len(tokens) > self.model.base_config.max_len:
                skipped.append(dict(start=start, end=end, subtokens=sequence_length, reason="RE_CONTEXT_LIMIT"))
                continue
            ner = [[a, b - 1, "ENTITY", chunk["text"][ca:cb]] for (a, b), (ca, cb) in spans.items()]
            self.calls += 1
            with torch.inference_mode():
                output = self.model.predict_relations(tokens, list(self.mapping), ner=ner, flat_ner=False,
                                                     threshold=float(self.cfg["zero_shot_re"]["threshold"]), top_k=-1)
            available = set(spans.values())
            for pair in pairs:
                h, t = pair["head"], pair["tail"]
                if (h["start"], h["end"]) in available and (t["start"], t["end"]) in available:
                    served.add((h["mention_id"], t["mention_id"]))
            for item in output:
                hp, tp = tuple(item["head_pos"]), tuple(item["tail_pos"])
                if hp not in spans or tp not in spans or item["label"] not in self.mapping:
                    raise ValueError("GLiREL returned unregistered endpoints/label")
                key = spans[hp], spans[tp], self.mapping[item["label"]]
                proposal = dict(score=float(item["score"]), start=start, end=end)
                if key not in predictions or predictions[key]["score"] < proposal["score"]:
                    predictions[key] = proposal
        return predictions, served, skipped
