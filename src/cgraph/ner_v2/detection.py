"""Future GLiNER mention detection with shared settings and source-exact postprocessing."""
import math

from cgraph.common import fingerprint
from cgraph.ner import windows
from . import CONTRACT_VERSION
from .io import validate_document, validate_span, validate_mention
from .normalization import detect_abbreviations
from .progress import progress


def label_descriptions(ontology):
    if len(ontology["entity_types"]) != 11 or len(ontology["relation_types"]) != 20:
        raise ValueError("Ontology v2 requires eleven types and twenty relations")
    return {item["ner_description_en"]: key for key, item in ontology["entity_types"].items()}


def postprocess(text, raw_windows, descriptions):
    """Deduplicate only the same span and hint; keep overlaps and alternate hints."""
    mentions = {}
    for offset, predictions in raw_windows:
        for item in predictions:
            a, b = offset + item["start"], offset + item["end"]
            validate_span(text, a, b, item["text"])
            if item["label"] not in descriptions:
                raise ValueError("Detector returned a label outside the frozen descriptions")
            hint = descriptions[item["label"]]
            if (isinstance(item["score"], bool) or not isinstance(item["score"], (int, float)) or
                    not math.isfinite(item["score"]) or not 0 <= item["score"] <= 1):
                raise ValueError("Detector score must be finite and between zero and one")
            key = a, b, hint
            candidate = {"start": a, "end": b, "surface_text": item["text"],
                         "ner_label_hint": hint, "score": float(item["score"])}
            if key not in mentions or candidate["score"] > mentions[key]["score"]:
                mentions[key] = candidate
    return [mentions[key] for key in sorted(mentions)]


def check_token_budget(model, text, descriptions, limit):
    """Tokenize the actual GLiNER prompt, without a forward pass. Refuse silent truncation."""
    words = [item[0] for item in model.data_processor.words_splitter(text)]
    inputs, _ = model.data_processor.prepare_inputs([words], list(descriptions))
    tokenizer = model.data_processor.transformer_tokenizer
    ids = tokenizer(inputs, is_split_into_words=True, truncation=False)["input_ids"][0]
    if len(ids) > limit or len(words) > model.config.max_len:
        raise ValueError("Frozen window exceeds encoder/word budget; revise the protocol for ALL models")
    return len(ids)


class MentionDetector:
    def __init__(self, model, model_identifier, settings, ontology, normalizer):
        self.model = model
        self.model_identifier = model_identifier
        self.settings = settings
        self.descriptions = label_descriptions(ontology)
        self.normalizer = normalizer

    def predict(self, document, *, progress_position=0):
        validate_document(document)
        text = document["text"]
        raw_windows = []
        spans = list(windows(text, self.settings["max_window_words"], self.settings["overlap_words"]))
        with progress(spans, desc="Windows", unit="window", position=progress_position, leave=False) as window_progress:
            for start, end in window_progress:
                window = text[start:end]
                check_token_budget(self.model, window, self.descriptions, self.settings["max_input_tokens"])
                output = self.model.predict_entities(window, list(self.descriptions),
                                                     threshold=self.settings["threshold"],
                                                     flat_ner=self.settings["flat_ner"],
                                                     multi_label=self.settings["multi_label"])
                raw_windows.append((start, output))
        analyzed = self.normalizer.analyze(text)
        abbreviations = detect_abbreviations(text)
        records = []
        for item in postprocess(text, raw_windows, self.descriptions):
            identity = [document["document_id"], document.get("revision_id"),
                        item["start"], item["end"], item["ner_label_hint"], self.model_identifier]
            record = {"contract_version": CONTRACT_VERSION, "document_id": document["document_id"],
                      "revision_id": document.get("revision_id"), "mention_id": "m:" + fingerprint(identity),
                      **item, **self.normalizer.normalize(text, item["start"], item["end"],
                                                         analyzed=analyzed, abbreviations=abbreviations),
                      "model_identifier": self.model_identifier, "source_reference": document["source_reference"]}
            validate_mention(record, document, self.descriptions.values())
            records.append(record)
        return records
