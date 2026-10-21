"""Pinned CPU GLiNER inference with source-preserving windows."""
import re
import time
from pathlib import Path

from .common import file_sha256


def windows(text, max_words, overlap):
    words = list(re.finditer(r"\S+", text))
    if not 0 <= overlap < max_words:
        raise ValueError("Invalid window overlap")
    for i in range(0, len(words), max_words - overlap):
        end_i = min(i + max_words, len(words))
        yield words[i].start(), words[end_i - 1].end()
        if end_i == len(words):
            break


class ScientificNER:
    def __init__(self, cfg):
        import torch
        from gliner import GLiNER
        from .model_setup import runtime_model

        self.cfg = cfg
        torch.set_num_threads(cfg["cpu_threads"])
        started = time.perf_counter()
        path = runtime_model(cfg)
        self.model = GLiNER.from_pretrained(str(path), map_location="cpu", local_files_only=True, strict=True)
        self.model.to("cpu").eval()
        self.calls = 0
        self.encoder_calls = 0
        self.max_sequence_tokens = 0

        def count_encoder(module, args, kwargs):
            self.encoder_calls += 1
            ids = kwargs.get("input_ids", args[0] if args else None)
            if ids is not None:
                self.max_sequence_tokens = max(self.max_sequence_tokens, int(ids.shape[-1]))

        encoders = [m for m in self.model.modules() if type(m).__name__ == "DebertaV2Model"]
        if len(encoders) != 1:
            raise ValueError("Cannot instrument the expected single DeBERTa encoder")
        encoders[0].register_forward_pre_hook(count_encoder, with_kwargs=True)
        self.load_ms = (time.perf_counter() - started) * 1000
        self.identity = {"checkpoint": cfg["ner"]["checkpoint"], "revision": cfg["ner"]["revision"],
                         "license": cfg["ner"]["license"], "device": "cpu", "dtype": "float32",
                         "tokenizer_checkpoint": cfg["ner"]["tokenizer_checkpoint"], "tokenizer_revision": cfg["ner"]["tokenizer_revision"],
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "model_class": type(self.model).__name__, "files": {p.name: file_sha256(p) for p in Path(path).iterdir() if p.is_file()}}
        self.identity["tokenizer_class"] = type(self.model.data_processor.transformer_tokenizer).__name__

    def predict(self, chunk):
        import torch
        text = chunk["text"]
        entities = {}
        for start, end in windows(text, self.cfg["ner"]["max_window_words"], self.cfg["ner"]["overlap_words"]):
            self.calls += 1
            with torch.inference_mode():
                output = self.model.predict_entities(text[start:end], list(self.cfg["ner"]["labels"]),
                                                      threshold=self.cfg["ner"]["threshold"], flat_ner=False, multi_label=True)
            for item in output:
                a, b = start + item["start"], start + item["end"]
                if not 0 <= a < b <= len(text) or text[a:b] != item["text"]:
                    raise ValueError("NER offsets are not exact source codepoints")
                kind = self.cfg["ner"]["labels"][item["label"]]
                key = a, b, kind
                mention = dict(start=a, end=b, surface_text=text[a:b], kind=kind,
                               contextual_role="target_substance" if kind == "ANALYTE" else "UNKNOWN", score=float(item["score"]))
                if key not in entities or entities[key]["score"] < mention["score"]:
                    entities[key] = mention
        output = []
        for i, (_, mention) in enumerate(sorted(entities.items())):
            mention["mention_id"] = f"{chunk['chunk_id']}:m{i}"
            output.append(mention)
        return output
