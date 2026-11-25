"""Local inference adapters for the frozen Corpus100 graph recipe."""
import os
from pathlib import Path
import random
import re
from collections import defaultdict

import build_relex_graphs as relex
from cgraph.common import fingerprint
from cgraph.ner_v2.io import CONTRACT_VERSION, validate_mention, validate_span
from relex_experiment import mapped_relations


def cpu(threads, seed):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_PROGRESS_BARS="1")
    import numpy as np
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return torch


def e3_windows(text, model, schema, recipe):
    from cgraph.ner import windows

    queue = list(windows(text, recipe["window_words"], recipe["overlap_words"]))
    i = 0
    while i < len(queue):
        start, end = queue[i]
        i += 1
        window = text[start:end]
        tokens = len(model.processor.transform_record(window, schema, max_len=None).input_ids)
        if tokens <= recipe["max_input_tokens"]:
            yield start, end
            continue
        # Повторяем разбиение E3 с перекрытием при превышении бюджета.
        words = list(re.finditer(r"\S+", window))
        if len(words) < 8:
            raise ValueError("Cannot fit the frozen E3 prompt")
        boundaries = [m.end() for m in re.finditer(r"[.!?]\s+", window)]
        mid = min(boundaries, key=lambda x: abs(x-len(window)/2)) if boundaries else len(window)//2
        pivot = min(range(1, len(words)), key=lambda x: abs(words[x].start()-mid))
        overlap = min(recipe["overlap_words"], max(1, len(words)//6))
        left, right = min(len(words)-1, pivot+overlap//2), max(1, pivot-overlap//2)
        parts = [(start, start+words[left-1].end()), (start+words[right].start(), end)]
        if any(b-a >= end-start for a, b in parts):
            raise ValueError("E3 subdivision made no progress")
        queue[i:i] = parts


def e3_source_span(window, prediction):
    start, end, surface = (prediction[k] for k in ("start", "end", "text"))
    # GLiNER2 дописывает точку к окну без финальной пунктуации.
    if (end == len(window) + 1 and not window.endswith((".", "!", "?"))
            and (window + ".")[start:end] == surface):
        end -= 1
        surface = surface[:-1]
    validate_span(window, start, end, surface)
    return dict(start=start, end=end, text=surface)


def ner_worker(config, paths):
    recipe = config["ner"]
    torch = cpu(config["threads"], config["seed"])
    from gliner2 import AutoExtractor
    from cgraph.ner_v2.detection import postprocess
    from cgraph.ner_v2.normalization import detect_abbreviations
    from cgraph.ner_v2.scientific_normalization import ScientificNormalizer

    model = AutoExtractor.from_pretrained(paths["ner"], map_location="cpu", local_files_only=True,
                                          attention_backend="standard").to("cpu").float().eval()
    normalizer = ScientificNormalizer()
    labels = recipe["labels"]
    schema = model.create_schema().entities(labels)
    descriptions = {label: code for code, label in labels.items()}

    def encoder_budget(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.shape[-1] > recipe["max_input_tokens"]:
            raise ValueError("Actual E3 encoder budget exceeded")

    encoder, = (m for m in model.modules() if type(m).__name__ == "DebertaV2Model")
    encoder.register_forward_pre_hook(encoder_budget, with_kwargs=True)

    def infer(doc, _):
        text = doc["text"]
        raw = []
        with torch.inference_mode():
            for start, end in e3_windows(text, model, schema, recipe):
                window = text[start:end]
                result = model.extract_entities(window, labels, threshold=recipe["decoder_threshold"],
                                                include_confidence=True, include_spans=True,
                                                overlap_policy="allow", max_len=None)
                raw.append((start, [dict(e3_source_span(window, p), label=labels[code], score=p["confidence"])
                                    for code, items in result["entities"].items() for p in items
                                    if p["confidence"] >= recipe["threshold"]]))
        analyzed = normalizer.analyze(text)
        definitions = detect_abbreviations(text)
        mentions = []
        for item in postprocess(text, raw, descriptions):
            start, end = item["start"], item["end"]
            identity = [doc["document_id"], doc["revision_id"], start, end, item["ner_label_hint"], "E3"]
            mentions.append({**item, **normalizer.normalize(text, start, end, analyzed=analyzed,
                                                             abbreviations=definitions),
                             "contract_version": CONTRACT_VERSION, "mention_id": "m:" + fingerprint(identity),
                             "configuration": "E3", "model_identifier": recipe["model"],
                             **{key: doc[key] for key in ("document_id", "revision_id", "source_reference")}})
        return {"mentions": mentions, "abbreviations": definitions}

    return infer


def entities_for(doc, mentions):
    groups = defaultdict(list)
    for mention in mentions:
        groups[mention["start"], mention["end"]].append(mention)
    return [dict(entity_id="e:" + relex.digest([doc["document_id"], start, end]), document_id=doc["document_id"],
                 start=start, end=end, surface_text=doc["text"][start:end],
                 entity_types=sorted({m["ner_label_hint"] for m in group}),
                 normalized_forms=sorted({m["normalized_form"] for m in group}),
                 confidence=max(m["score"] for m in group), source_mentions=group)
            for (start, end), group in sorted(groups.items())]


def re_worker(config, paths):
    recipe = config["re"]
    cpu(config["threads"], config["seed"])
    from gliner import GLiNER

    plan_paths = {kind: Path(paths[kind]) for kind in ("re", "re_common")}
    model = GLiNER.from_pretrained(paths["re"], local_files_only=True).to("cpu").eval()
    ontology = recipe["ontology"]

    def infer(doc, ner):
        entities = entities_for(doc, ner["mentions"])
        windows, labels, budget = relex.plan_windows([doc], entities, plan_paths, ontology,
                                                     recipe["token_limit"], recipe["overlap"])
        raw = []
        for window in windows:
            ids = set(window["entity_ids"])
            supplied = [e for e in entities if e["entity_id"] in ids]
            predictions, audit = relex.predict_window(model, window, supplied, labels, recipe["threshold"], budget)
            raw.append(dict(window=window, predictions=predictions, audit=audit))
        return dict(entities=entities, raw_windows=raw, relations=mapped_relations(raw, entities),
                    raw_relation_count=sum(len(w["predictions"]) for w in raw))

    return infer


def validate_ner(doc, result, recipe):
    seen = set()
    for mention in result["mentions"]:
        validate_mention(mention, doc, recipe["labels"])
        identity = [doc["document_id"], doc["revision_id"], mention["start"], mention["end"],
                    mention["ner_label_hint"], "E3"]
        key = mention["mention_id"]
        if (key != "m:" + fingerprint(identity) or key in seen or mention["score"] < recipe["threshold"]
                or mention["model_identifier"] != recipe["model"]):
            raise ValueError("Invalid E3 mention identity, score or model")
        seen.add(key)
        for token in mention["tokens"]:
            validate_span(doc["text"], token["start"], token["end"], token["text"])
    from cgraph.ner_v2.normalization import detect_abbreviations
    if result["abbreviations"] != detect_abbreviations(doc["text"]):
        raise ValueError("Abbreviation evidence differs from source")


def validate_re(doc, result, ner, recipe):
    entities = entities_for(doc, ner["mentions"])
    if result["entities"] != entities:
        raise ValueError("RE entities differ from E3")
    covered = set()
    for raw in result["raw_windows"]:
        window = raw["window"]
        start, end = window["start"], window["end"]
        validate_span(doc["text"], start, end, window["text"])
        if (window["document_id"] != doc["document_id"]
                or window["window_id"] != "w:" + relex.digest([doc["document_id"], start, end])
                or window["entity_ids"] != [e["entity_id"] for e in entities
                                            if start <= e["start"] and e["end"] <= end]):
            raise ValueError("Invalid RE window provenance")
        covered.update(window["entity_ids"])
        if any(p["score"] < recipe["threshold"] for p in raw["predictions"]):
            raise ValueError("RE score below threshold")
    if (covered != {e["entity_id"] for e in entities}
            or not result["raw_windows"]
            or result["raw_relation_count"] != sum(len(w["predictions"]) for w in result["raw_windows"])
            or result["relations"] != mapped_relations(result["raw_windows"], entities)):
        raise ValueError("Incomplete RE checkpoint")


def e5_worker(path, threads, seed, recipe):
    torch = cpu(threads, seed)
    import numpy as np
    from tokenizers import Tokenizer
    from transformers import AutoModel
    from transformers.utils import logging
    from cgraph.semantic_concepts import pool_e5

    logging.disable_progress_bar()
    tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=recipe["token_limit"])
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                      use_safetensors=True, dtype=torch.float32).cpu().eval()

    def embed(texts):
        batch = tokenizer.encode_batch([recipe["prefix"] + text for text in texts])
        inputs = {name: np.array([getattr(row, attr) for row in batch], dtype=np.int64)
                  for name, attr in (("input_ids", "ids"), ("attention_mask", "attention_mask"),
                                     ("token_type_ids", "type_ids"))}
        with torch.inference_mode():
            hidden = model(**{k: torch.from_numpy(v) for k, v in inputs.items()}).last_hidden_state.numpy()
        return pool_e5(hidden, inputs["attention_mask"]).tolist()

    return embed
