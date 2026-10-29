"""CPU pretrained token-classifier adapters; no annotation or training imports."""
from pathlib import Path
import time

from .common import file_sha256, read_json
from .ner import windows


def merge_mentions(chunk, groups):
    """Exact union; preserve conflicting kinds and nested boundaries, no repairs."""
    merged = {}
    for group in groups:
        for item in group:
            key = item["start"], item["end"], item["kind"]
            if key not in merged or item["score"] > merged[key]["score"]:
                merged[key] = dict(item)
    result = []
    for i, (_, item) in enumerate(sorted(merged.items())):
        item["mention_id"] = f"{chunk['chunk_id']}:m{i}"
        result.append(item)
    return result


class PretrainedScientificNER:
    def __init__(self, spec, contract_path):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        started = time.perf_counter()
        self.spec = spec
        path = Path(spec["local_dir"])
        hashes = {f"{spec['checkpoint']}@{spec['revision']}/{name}": file_sha256(path / name)
                  for name in spec["files"]}
        contract = read_json(contract_path)
        if any(contract.get(k) != v for k, v in hashes.items()):
            raise ValueError("Scientific NER checkpoint hash mismatch")
        self.model, info = AutoModelForTokenClassification.from_pretrained(
            path, local_files_only=True, output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete pretrained NER head: {info}")
        if self.model.config.id2label != {int(k): v for k, v in spec["id2label"].items()}:
            raise ValueError("Checkpoint labels differ from preregistered labels")
        self.model.to("cpu").eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        self.pipe = pipeline("token-classification", model=self.model, tokenizer=self.tokenizer,
                             aggregation_strategy="simple", device=-1)
        self.calls = self.encoder_calls = self.max_sequence_tokens = 0
        self.source_label_counts = {}
        self.unsupported_spans = 0

        def count(module, args, kwargs):
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is None:
                raise ValueError("Cannot instrument scientific NER input IDs")
            self.encoder_calls += 1
            self.max_sequence_tokens = max(self.max_sequence_tokens, int(ids.shape[-1]))
            if ids.shape[-1] > spec["max_subtokens"]:
                raise ValueError("Scientific NER token budget exceeded")
        self.model.bert.register_forward_pre_hook(count, with_kwargs=True)
        self.identity = dict(checkpoint=spec["checkpoint"], revision=spec["revision"],
                             files=hashes, device="cpu", dtype="float32", strict_load=True,
                             missing_keys=info["missing_keys"], parameters=sum(p.numel() for p in self.model.parameters()),
                             classifier_shape=list(self.model.classifier.weight.shape),
                             class_mapping=spec["mapping"], training_performed=False)
        self.load_ms = 1000 * (time.perf_counter() - started)

    def predict(self, chunk):
        import torch

        text, output = chunk["text"], []
        for a, b in windows(text, self.spec["max_window_words"], self.spec["overlap_words"]):
            n = len(self.tokenizer(text[a:b], truncation=False)["input_ids"])
            if n > self.spec["max_subtokens"]:
                raise ValueError("Window too long; silent token truncation forbidden")
            self.calls += 1
            with torch.inference_mode():
                spans = self.pipe(text[a:b])
            for span in spans:
                label = span["entity_group"]
                self.source_label_counts[label] = self.source_label_counts.get(label, 0) + 1
                kind = self.spec["mapping"][label]
                if kind is None:
                    self.unsupported_spans += 1
                    continue
                start, end = a + int(span["start"]), a + int(span["end"])
                if not 0 <= start < end <= len(text):
                    raise ValueError("Invalid scientific NER offsets")
                output.append(dict(start=start, end=end, surface_text=text[start:end], kind=kind,
                                   contextual_role="UNKNOWN", score=float(span["score"]), source_label=label))
        return merge_mentions(chunk, [output])


class UnionNER:
    def __init__(self, models):
        self.models = models
        self.identity = dict(rule="exact span/kind union, maximum duplicate score", models=[m.identity for m in models])
        self.load_ms = sum(m.load_ms for m in models)

    @property
    def calls(self):
        return sum(m.calls for m in self.models)

    @property
    def encoder_calls(self):
        return sum(m.encoder_calls for m in self.models)

    @property
    def max_sequence_tokens(self):
        return max(m.max_sequence_tokens for m in self.models)

    def predict(self, chunk):
        return merge_mentions(chunk, [model.predict(chunk) for model in self.models])
