"""Source-exact analysis of short English questions with the frozen E3 model."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TypedDict

from cgraph.ner_v2.detection import postprocess

E3_DESCRIPTORS = {
    "OBJECT": "physical, biological or digital object", "ACTOR": "person or organization",
    "CONCEPT": "scientific concept or theory", "METHOD": "algorithm or research method",
    "PROCESS": "event or scientific process", "DATA_RESOURCE": "dataset or research corpus",
    "ATTRIBUTE": "property, parameter or metric", "RESULT": "measured or observed result",
    "CONDITION": "experimental or contextual condition", "TASK": "research task or application",
    "MECHANISM": "mechanism explaining a phenomenon",
}
E3_HASHES = {
    "config.json": "d6b515c21ae2100c12267fb42a3f02c1e99ad87580366aebedb161efd74441c7",
    "encoder_config/config.json": "d36a845b9f25dcaf1ec45a1c4bdf65ea4ac20596537e14530ec9f660a63aeca4",
    "model.safetensors": "7274094de2e0c2a37a386f55fc4e23061a954da5bd7a335e7dfe56f2743c277a",
    "tokenizer.json": "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3",
    "tokenizer_config.json": "0bf3ea0873234bd9bfdd3853c440395009ac6365a925b91654daed5396d655e1",
}
MODEL_ID = "fastino/gliner2.5-base-v1@54785d51df8d86a0d8f2eb212fec15871aad2c4d"


class QueryMention(TypedDict):
    mention_id: str
    surface_text: str
    start: int
    end: int
    label: str
    score: float


class QueryAnalysis(TypedDict):
    text: str
    mentions: list[QueryMention]
    model_identifier: str
    threshold: float
    warnings: list[str]


class E3Extractor:
    def __init__(self, model_path: str | Path = "artifacts/ner_v2/models/gliner25"):
        self.model_path = Path(model_path)
        self._model = None
        self._schema = None

    def predict(self, text: str, threshold: float) -> list[dict]:
        if self._model is None:
            for name, expected in E3_HASHES.items():
                digest = hashlib.sha256()
                with (self.model_path / name).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Frozen E3 model hash mismatch: {name}")
            import torch
            from gliner2 import AutoExtractor
            torch.set_num_threads(4)
            model = AutoExtractor.from_pretrained(str(self.model_path), map_location="cpu",
                                                  local_files_only=True, attention_backend="standard")
            model.to("cpu").float().eval()
            self._model = model
            self._schema = model.create_schema().entities(E3_DESCRIPTORS)
        if len(self._model.processor.transform_record(text, self._schema, max_len=None).input_ids) > 512:
            raise ValueError("Question exceeds the E3 prompt token budget")
        import torch
        with torch.inference_mode():
            result = self._model.extract_entities(text, E3_DESCRIPTORS, threshold=threshold,
                include_confidence=True, include_spans=True, overlap_policy="allow", max_len=None)
        return [dict(start=p["start"], end=p["end"], text=p["text"],
                     label=E3_DESCRIPTORS[label], score=p["confidence"])
                for label, items in result["entities"].items() for p in items]


class QueryAnalyzer:
    def __init__(self, extractor=None, *, threshold: float = 0.15):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("NER threshold must be finite and between zero and one")
        self.extractor = extractor if extractor is not None else E3Extractor()
        self.threshold = float(threshold)

    def analyze(self, text: str) -> QueryAnalysis:
        if not isinstance(text, str):
            raise TypeError("Question must be a string")
        if not text.strip():
            return {"text": text, "mentions": [], "model_identifier": MODEL_ID,
                    "threshold": self.threshold, "warnings": ["empty_question"]}
        raw = self.extractor.predict(text, self.threshold)
        processed = postprocess(text, [(0, raw)], {v: k for k, v in E3_DESCRIPTORS.items()})
        mentions = []
        for item in processed:
            identity = f"{text}\0{item['start']}\0{item['end']}\0{item['ner_label_hint']}"
            mentions.append({"mention_id": "qm:" + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                "surface_text": item["surface_text"], "start": item["start"], "end": item["end"],
                "label": item["ner_label_hint"], "score": item["score"]})
        return {"text": text, "mentions": mentions, "model_identifier": MODEL_ID,
                "threshold": self.threshold, "warnings": [] if mentions else ["empty_ner"]}
