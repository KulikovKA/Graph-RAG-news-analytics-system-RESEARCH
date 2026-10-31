"""Future offline benchmark entrypoint. Never imported by preparation commands."""
import gc
import os
import random
import time
from pathlib import Path

from cgraph.common import file_sha256, fingerprint, read_json, write_json
from .detection import MentionDetector, label_descriptions
from .io import read_documents, save_mentions
from .model_setup import load_detector, verify_models
from .normalization import LinguisticNormalizer
from .progress import progress


def span_agreement(left, right):
    """Descriptive span agreement, not accuracy/F1 against nonexistent Gold."""
    a = {(m["document_id"], m["start"], m["end"]) for m in left}
    b = {(m["document_id"], m["start"], m["end"]) for m in right}
    return {"left_spans": len(a), "right_spans": len(b), "intersection": len(a & b),
            "union": len(a | b), "jaccard": len(a & b) / len(a | b) if a | b else None}


def run(settings, output):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import numpy as np
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    inventory = verify_models(settings["model_lock"], settings["model_root"])
    documents = read_documents(settings["input"])
    ontology, lock = read_json(settings["ontology"]), read_json(settings["model_lock"])
    descriptions = label_descriptions(ontology)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    normalizer = LinguisticNormalizer()
    manifest = {"status": "running", "settings": settings, "settings_fingerprint": fingerprint(settings),
                "label_descriptions": descriptions, "input_sha256": file_sha256(settings["input"]),
                "ontology_sha256": file_sha256(settings["ontology"]), "model_lock_sha256": file_sha256(settings["model_lock"]),
                "libraries": inventory["libraries"], "models": {}, "agreement": {}}
    write_json(root / "manifest.json", manifest)
    outputs = {}
    with progress(("large", "medium", "small"), desc="Models", unit="model", position=0) as models:
        for size in models:
            models.set_description(f"Models ({size}: loading)")
            repo = lock["repositories"][lock["models"][size]["checkpoint_key"]]
            started = time.perf_counter()
            model = load_detector(size, settings["model_root"])
            load_seconds = time.perf_counter() - started
            models.set_description(f"Models ({size})")
            detector = MentionDetector(model, repo["repo_id"] + "@" + repo["revision"], settings, ontology, normalizer)
            started = time.perf_counter()
            mentions = []
            with torch.inference_mode(), progress(documents, desc=f"{size}: documents", unit="doc",
                                                   position=1, leave=False) as docs:
                for document in docs:
                    docs.set_postfix_str(document["document_id"])
                    mentions.extend(detector.predict(document, progress_position=2))
            seconds = time.perf_counter() - started
            save_mentions(root / f"{size}.jsonl", mentions, documents, descriptions.values())
            manifest["models"][size] = {"model_identifier": detector.model_identifier, "load_seconds": load_seconds,
                                        "detection_and_normalization_seconds": seconds, "mentions": len(mentions),
                                        "documents_per_second": len(documents) / seconds}
            outputs[size] = mentions
            del detector, model
            gc.collect()
    for a, b in (("large", "medium"), ("large", "small"), ("medium", "small")):
        manifest["agreement"][f"{a}/{b}"] = span_agreement(outputs[a], outputs[b])
    manifest["status"] = "complete"
    write_json(root / "manifest.json", manifest)
    return manifest
