"""Pretrained entailment as zero-shot RE (Sainz et al., EMNLP 2021).

Uses the checkpoint's real three-class NLI head, never fits relation heads.
"""
import time
from pathlib import Path

from .candidates import allowed_labels
from .common import file_sha256, read_json
from .zeroshot_re import contexts
from .zeroshot_rules import linguistic_negations


class EntailmentRE:
    def __init__(self, cfg, ontology):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        started = time.perf_counter()
        spec = cfg["zero_shot_re"]
        hashes = {f"{spec['checkpoint']}@{spec['revision']}/{name}": file_sha256(Path(spec["local_dir"]) / name)
                  for name in spec["files"]}
        if hashes != read_json(spec["file_contract"]):
            raise ValueError("Pretrained NLI assets differ from pinned hashes")
        self.tokenizer = AutoTokenizer.from_pretrained(spec["local_dir"], local_files_only=True)
        self.model, info = AutoModelForSequenceClassification.from_pretrained(spec["local_dir"], local_files_only=True,
                                                                           output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete pretrained NLI checkpoint: {info}")
        actual_labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if actual_labels != {0: "contradiction", 1: "entailment", 2: "neutral"}:
            raise ValueError("NLI class ordering is not the verified checkpoint ordering")
        probe = self.tokenizer("runtime compatibility")["input_ids"]
        if probe[0] != self.tokenizer.cls_token_id or probe[-1] != self.tokenizer.sep_token_id:
            raise ValueError("Tokenizer lacks required model special tokens")
        self.model.to("cpu").eval().requires_grad_(False)
        self.cfg, self.ontology = cfg, ontology
        self.calls = self.encoder_calls = self.max_sequence_tokens = 0
        self.score_summary = dict(pair_label_scores=0, above_fixed_threshold=0, max_score=None)
        self.last_trace = []

        def count(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None or int(ids.shape[-1]) > spec["max_subtokens"]:
                raise ValueError("Unexpected NLI input or silent context overflow")
            self.encoder_calls += 1
            self.max_sequence_tokens = max(self.max_sequence_tokens, int(ids.shape[-1]))
        self.model.deberta.register_forward_pre_hook(count, with_kwargs=True)
        self.identity = dict(backend="pretrained_nli_zero_shot_re", checkpoint=spec["checkpoint"], revision=spec["revision"],
                             license=spec["license"], device="cpu", dtype="float32", strict_load=True,
                             parameters=sum(p.numel() for p in self.model.parameters()), files=hashes,
                             train_documents=[], training_performed=False, capabilities=spec["capabilities"],
                             actual_prediction_heads=actual_labels, relation_heads_added=0)
        self.load_ms = 1000 * (time.perf_counter() - started)

    def predict(self, chunk, entities, pairs):
        import torch

        spec = self.cfg["zero_shot_re"]
        jobs, served, skipped = [], set(), []
        self.last_trace = []
        for start, end in contexts(chunk["text"], spec["sentence_window"]):
            premise = chunk["text"][start:end]
            for pair in pairs:
                h, t = pair["head"], pair["tail"]
                if not (start <= min(h["start"], t["start"]) and max(h["end"], t["end"]) <= end):
                    continue
                for label in allowed_labels(h, t, self.ontology):
                    capability = spec["capabilities"][label]
                    if capability["status"] != "EXPERIMENTAL_ZERO_SHOT":
                        continue
                    hypothesis = capability["hypothesis"].format(head=h["surface_text"], tail=t["surface_text"])
                    variants = [("POSITIVE", hypothesis)]
                    if linguistic_negations(premise):
                        variants.append(("NEGATIVE", f"It is not true that {hypothesis}"))
                    for polarity, hypothesis in variants:
                        encoded = self.tokenizer(premise, hypothesis, truncation=False)
                        if len(encoded["input_ids"]) > spec["max_subtokens"]:
                            skipped.append(dict(start=start, end=end, reason="RE_CONTEXT_LIMIT", relation_type=label))
                            continue
                        served.add((h["mention_id"], t["mention_id"]))
                        jobs.append(dict(pair=pair, label=label, polarity=polarity, hypothesis=hypothesis,
                                         encoded=encoded, start=start, end=end))
        jobs.sort(key=lambda j: len(j["encoded"]["input_ids"]))
        predicted = {}
        for offset in range(0, len(jobs), spec["batch_size"]):
            batch = jobs[offset:offset + spec["batch_size"]]
            features = self.tokenizer.pad([j["encoded"] for j in batch], padding=True, return_tensors="pt")
            self.calls += 1
            with torch.inference_mode():
                probabilities = self.model(**features).logits.softmax(dim=-1).cpu().tolist()
            for job, scores in zip(batch, probabilities, strict=True):
                entailment = scores[1]
                self.score_summary["pair_label_scores"] += 1
                previous = self.score_summary["max_score"]
                self.score_summary["max_score"] = entailment if previous is None else max(previous, entailment)
                h, t = job["pair"]["head"], job["pair"]["tail"]
                passed = entailment > spec["threshold"]
                self.last_trace.append(dict(head_mention_id=h["mention_id"], tail_mention_id=t["mention_id"],
                    relation_type=job["label"], polarity=job["polarity"], hypothesis=job["hypothesis"],
                    start=job["start"], end=job["end"], scores=dict(zip(("contradiction", "entailment", "neutral"), scores)),
                    passed_fixed_threshold=passed))
                if passed:
                    self.score_summary["above_fixed_threshold"] += 1
                    key = (h["start"], h["end"]), (t["start"], t["end"]), job["label"]
                    if key not in predicted or predicted[key]["score"] < entailment:
                        predicted[key] = dict(score=entailment, start=job["start"], end=job["end"],
                                              nli_polarity=job["polarity"], hypothesis=job["hypothesis"],
                                              nli_scores=self.last_trace[-1]["scores"])
        return predicted, served, skipped
