"""Проверенные входы для локального воспроизведения."""

import importlib.metadata
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmark_3a5_8"))
from frozen_pool import (DEFAULT_SOURCE, ROOT, file_sha, jsonl, documents, digest, read_json,
                         stage_value, LEGACY, SOURCE, LOCK)
from run_integrated_benchmark import DATASET_HASHES, DEPENDENCIES, VARIANTS, load_questions
from run_entity_linking_sweep import CachedLinker, snapshot
from cgraph.entity_linking import E5Encoder, EntityLinker, _expansion_searches, load_entity_cache
from cgraph.query_analysis import QueryAnalyzer
from validate_draft import require

ANCHOR = read_json(HERE.parent / "benchmark_3a5_8/FROZEN_B.json")
CACHE = ROOT / ".cache/semantic_concept_sweep"
EXTRA = ROOT / ".cache/stage_3a59_query_scores_v1.json"
FINGERPRINT = "85faa1a719f46f9ee375d7c0e07aef55ce9ef13f1afe34f581585f4971a4c897"


def inputs():
    config = read_json(DEFAULT_SOURCE / "config.json")
    identity = config["identity"]
    require(digest(config) == FINGERPRINT and config["variants"] == VARIANTS, "Integrated config changed")
    manifest = read_json(DEFAULT_SOURCE / "manifest.json")
    require(manifest["status"] == "COMPLETE" and manifest["fingerprint"] == FINGERPRINT, "Incomplete source")
    require(identity["models"] == ANCHOR["models"] and identity["dataset_sha256"] == DATASET_HASHES and
            identity["corpus_sha256"] == file_sha(SOURCE) and identity["document_lock_sha256"] == file_sha(LOCK),
            "Inputs differ")
    corpus = documents()
    graph = snapshot(CACHE, corpus)
    require(graph.fingerprint == ANCHOR["graph_fingerprint"], "Graph differs")
    questions, refs = load_questions(corpus)
    rows = {(r["key"], r["variant"]): r for r in jsonl(DEFAULT_SOURCE / "per_question.jsonl")}
    cache = load_entity_cache(CACHE)
    return graph, cache, questions, refs, rows


def original(q, rows, cache):
    key = digest(q.text)
    row = rows[q.key, "D"]
    query, _ = stage_value(DEFAULT_SOURCE, row, "ner", digest([FINGERPRINT, "ner", key]))
    texts = list(dict.fromkeys(m["surface_text"] for m in query["mentions"])) or [q.text]
    value, _ = stage_value(DEFAULT_SOURCE, row, "e5", digest([FINGERPRINT, "e5", [key, texts]]))
    scores = np.array(value["scores"], dtype=value["dtype"])
    require(query["text"] == q.text and scores.shape == (len(texts), len(cache.entities)) and
            np.isfinite(scores).all() and all(q.text[m["start"]:m["end"]] == m["surface_text"]
            for m in query["mentions"]), "Query/score alignment failed")
    return query, dict(zip(texts, scores, strict=True))


def identity():
    return dict(integrated=FINGERPRINT, models=ANCHOR["models"], paraphrases=file_sha(HERE / "paraphrases.json"),
                inputs_code=file_sha(Path(__file__)), versions={name: importlib.metadata.version(name)
                for name in DEPENDENCIES})


def prepare_extra():
    import psutil
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    graph, cache, questions, _, rows = inputs()
    texts = {}
    for q in questions:
        if q.group != "HOLDOUT":
            texts.update({s.text: None for s in _expansion_searches(q.text, set())})
    analyzer = QueryAnalyzer()
    variants = {}
    for key, text in tqdm(read_json(HERE / "paraphrases.json").items(), desc="Paraphrase GLiNER"):
        started = perf_counter()
        query = analyzer.analyze(text)
        variants[key] = dict(query=query, seconds=perf_counter()-started)
        terms = [m["surface_text"] for m in query["mentions"]] or [text]
        texts.update(dict.fromkeys(terms + [s.text for s in _expansion_searches(text, set())]))
    del analyzer
    scorer = EntityLinker(cache, E5Encoder(cache.model))
    extra = {}
    for text in tqdm(texts, desc="Query-only E5"):
        started = perf_counter()
        extra[text] = dict(scores=scorer._score([text])[0].tolist(), seconds=perf_counter()-started)
    record = dict(identity=identity(), paraphrases=variants, scores=extra,
                  peak_rss=psutil.Process().memory_info().peak_wset)
    with EXTRA.open("x", encoding="utf8") as stream:
        json.dump(dict(record=record, digest=digest(record)), stream, ensure_ascii=False, allow_nan=False)


def extras():
    envelope = read_json(EXTRA)
    record = envelope["record"]
    require(envelope["digest"] == digest(record) and record["identity"] == identity(), "Extra cache differs")
    return record


if __name__ == "__main__":
    prepare_extra()
