r"""RE-V2-001: parallel, offline RelEx on original E3 spans and existing Neo4j.

Run from PowerShell: .\.venv\Scripts\python.exe scripts\build_relex_graphs.py
See docs/RE_V2_001.md for environment variables and the API compatibility note.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/ner_v2/research_run2"
OUTPUT = ROOT / "reports/re_v2/relex_comparison"
ONTOLOGY = ROOT / "contracts/ner_v2/ontology.v2.json"
VERSION = "re-v2-001-explicit-span-head-2"
WORDS = re.compile(r"\w+(?:[-_]\w+)*|\S")  # GLiNER WhitespaceTokenSplitter
PREDICATES = {
    "IS_A": "is a", "HAS_PART": "has part", "HAS_MEMBER": "has member",
    "HAS_ATTRIBUTE": "has attribute", "HAS_RESULT": "has result", "USES": "uses",
    "USED_FOR": "used for", "APPLIED_TO": "applied to", "PERFORMS": "performs",
    "PRODUCES": "produces", "DERIVED_FROM": "derived from", "EVALUATED_ON": "evaluated on",
    "MEASURES": "measures", "CHANGES": "changes", "CAUSES": "causes",
    "DEPENDS_ON": "depends on", "EXPLAINS": "explains", "ASSOCIATED_WITH": "associated with",
    "COMPARED_WITH": "compared with", "LOCATED_IN": "located in",
}
# Review hints derived from the predicate definitions, NOT approved domain/range constraints.
# Ontology v2 deliberately has no formal domain/range matrix. Never filter using these hints.
TYPE_HINTS = {
    "HAS_ATTRIBUTE": (None, {"ATTRIBUTE"}), "HAS_RESULT": (None, {"RESULT"}),
    "USED_FOR": (None, {"TASK", "PROCESS"}),
    "PERFORMS": ({"ACTOR"}, {"PROCESS", "TASK"}),
    "EVALUATED_ON": (None, {"DATA_RESOURCE", "TASK"}),
    "MEASURES": (None, {"ATTRIBUTE"}),
    "EXPLAINS": ({"MECHANISM", "CONCEPT"}, {"PROCESS", "RESULT", "CONCEPT"}),
}
NAMESPACES = {"relex_base", "relex_large", "relex_joint_base", "relex_joint_large"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def load_documents(limit=None):
    documents = read_jsonl(SOURCE / "input_abstracts.jsonl")
    if len(documents) != 30 or len({d["document_id"] for d in documents}) != 30:
        raise ValueError("Expected exactly 30 unique original E3 documents")
    return documents[:limit] if limit else documents


def load_inputs(limit=None):
    documents = load_documents()
    mentions = read_jsonl(SOURCE / "E3/mentions_035.jsonl")
    by_doc = {d["document_id"]: d for d in documents}
    groups = defaultdict(list)
    for mention in mentions:
        doc = by_doc[mention["document_id"]]
        start, end = mention["start"], mention["end"]
        if mention["configuration"] != "E3" or not 0 <= start < end <= len(doc["text"]):
            raise ValueError("Invalid original E3 mention")
        if doc["text"][start:end] != mention["surface_text"]:
            raise ValueError(f"Offset mismatch: {mention['mention_id']}")
        if mention["revision_id"] != doc["revision_id"]:
            raise ValueError("E3 revision mismatch")
        groups[(doc["document_id"], start, end)].append(mention)
    if len(groups) != 1211:
        raise ValueError(f"Expected 1211 unique E3 spans, got {len(groups)}")
    selected = documents[:limit] if limit else documents
    selected_ids = {d["document_id"] for d in selected}
    entities = []
    for (doc_id, start, end), original in sorted(groups.items()):
        if doc_id not in selected_ids:
            continue
        entities.append({
            "entity_id": "e:" + digest([doc_id, start, end]),
            "document_id": doc_id, "start": start, "end": end,
            "surface_text": by_doc[doc_id]["text"][start:end],
            "entity_types": sorted({m["ner_label_hint"] for m in original}),
            "normalized_forms": sorted({m["normalized_form"] for m in original}),
            "confidence": max(m["score"] for m in original),
            "source_mentions": original,  # Preserve EVERY original type, score and normalization.
        })
    return selected, entities


def make_windows(document, entities, fits, overlap=0.25):
    """Largest fitting common windows; cut only outside every entity, cover all spans."""
    text = document["text"]
    boundaries = sorted({0, len(text)} | {m.start() for m in WORDS.finditer(text)})
    boundaries = [b for b in boundaries if not any(e["start"] < b < e["end"] for e in entities)]
    windows, left = [], 0
    while left < len(boundaries) - 1:
        low, high, best = left + 1, len(boundaries) - 1, None
        while low <= high:
            middle = (low + high) // 2
            if fits(text[boundaries[left]:boundaries[middle]]):
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best is None:
            raise ValueError("An indivisible E3 span exceeds the common token budget")
        start, end = boundaries[left], boundaries[best]
        members = [e["entity_id"] for e in entities if start <= e["start"] and e["end"] <= end]
        windows.append({"window_id": "w:" + digest([document["document_id"], start, end]),
                        "document_id": document["document_id"], "start": start, "end": end,
                        "text": text[start:end], "entity_ids": members})
        if best == len(boundaries) - 1:
            break
        target = end - int((end - start) * overlap)
        candidates = [i for i in range(left + 1, best + 1) if boundaries[i] <= target]
        left = candidates[-1] if candidates else left + 1
    covered = {eid for window in windows for eid in window["entity_ids"]}
    if covered != {e["entity_id"] for e in entities}:
        raise ValueError("Window plan lost original entities")
    return windows


def plan_windows(documents, entities, paths, ontology, token_limit, overlap):
    from transformers import AutoTokenizer
    tokenizers, prompts, budgets = [], [], []
    labels = [code.lower().replace("_", " ") for code in ontology["entity_types"]]
    for path in paths.values():
        config = json.loads((path / "gliner_config.json").read_text(encoding="utf-8"))
        if config["words_splitter_type"] != "whitespace" or config["span_mode"] not in {"token_level", "markerV0"}:
            raise ValueError("This adapter supports the downloaded token_level/markerV0 RelEx models")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, use_fast=False)
        prompt = []
        for label in labels:
            prompt += [config["ent_token"], label]
        prompt.append(config["sep_token"])
        for label in PREDICATES.values():
            prompt += [config["rel_token"], label]
        prompt.append(config["sep_token"])
        budgets.append(min(token_limit, config["max_len"], config["encoder_config"]["max_position_embeddings"]))
        tokenizers.append(tokenizer)
        prompts.append(prompt)

    def fits(text):
        words = WORDS.findall(text)
        return all(len(t(p + words, is_split_into_words=True, truncation=False)["input_ids"]) <= b
                   for t, p, b in zip(tokenizers, prompts, budgets))

    windows = []
    for document in documents:
        members = [e for e in entities if e["document_id"] == document["document_id"]]
        windows.extend(make_windows(document, members, fits, overlap))
    return windows, labels, min(budgets)


def explicit_span_embeddings(marker, words, indices):
    """Exact trained TokenMarker / SpanMarkerV0 endpoint projections, without grid reshape.

    SpanMarkerV0.forward requires L*max_width candidates just to reshape its
    output. The projection itself is identical for an arbitrary list of E3 spans,
    including mentions wider than max_width. No parameters are changed.
    """
    import torch
    starts = marker.project_start(words)[:, indices[:, 0], :]
    ends = marker.project_end(words)[:, indices[:, 1], :]
    return marker.out_project(torch.cat((starts, ends), dim=-1).relu())


def predict_window(model, window, members, labels, relation_threshold, token_limit):
    """Run the trained encoder, span projections and directed relation head on E3 only.

    The high-level API does not enforce input_spans. Base forward can consume
    span_idx; Large forward still filters them by NER scores and grid width.
    Use the same learned submodules, bypassing both NER selection mechanisms.
    """
    import torch
    if len(members) < 2:
        return [], {"tokens": 0, "spans": len(members), "pairs": 0}
    tokens, starts, ends = model.prepare_inputs([window["text"]])
    start_map, end_map = {v: i for i, v in enumerate(starts[0])}, {v: i for i, v in enumerate(ends[0])}
    indices = []
    for entity in members:
        a, b = entity["start"] - window["start"], entity["end"] - window["start"]
        if a not in start_map or b not in end_map:
            raise ValueError(f"E3 span not aligned to GLiNER words: {entity['entity_id']}")
        indices.append([start_map[a], end_map[b]])  # Inclusive WORD end, exclusive CHAR end.
    collator = model.data_collator_class(
        model.config, data_processor=model.data_processor, return_tokens=True,
        return_entities=True, return_id_to_classes=True, return_rel_id_to_classes=True,
        prepare_labels=False,
    )
    batch = collator(model.prepare_base_input(tokens), entity_types=labels,
                     relation_types=list(PREDICATES.values()))
    if batch["input_ids"].shape[1] > token_limit:
        raise ValueError("Actual collator token budget exceeds common window plan")
    from gliner.modeling.utils import extract_prompt_features, build_all_entity_pairs
    network = model.model
    if model.config.relations_layer != "none" or not hasattr(network, "pair_rep_layer"):
        raise ValueError("Expected v1.0 all-pairs relation head")
    with torch.inference_mode():
        encoded = network.token_rep_layer(batch["input_ids"], batch["attention_mask"])
        _, _, words, word_mask = network._extract_prompt_features_and_word_embeddings(
            encoded, batch["input_ids"], batch["attention_mask"], batch["text_lengths"], batch["words_mask"])
        if hasattr(network, "rnn"):
            words = network.rnn(words, word_mask)
        if words.shape[1] != len(tokens[0]):
            raise RuntimeError("Word embeddings were truncated")
        supplied_indices = torch.tensor(indices, dtype=torch.long)
        representations = explicit_span_embeddings(network.span_rep_layer.span_rep_layer, words, supplied_indices)
        span_mask = torch.ones((1, len(indices)), dtype=torch.bool)
        pair_indices, pair_mask, heads, tails = build_all_entity_pairs(representations, span_mask)
        rel_prompts, _ = extract_prompt_features(model.config.rel_token_index, encoded,
            batch["input_ids"], batch["attention_mask"], 1, representations.shape[-1], model.config.embed_rel_token)
        if rel_prompts.shape[1] != len(PREDICATES):
            raise RuntimeError("Relation prompt labels were truncated")
        pairs = network.pair_rep_layer(torch.cat((heads, tails), dim=-1))
        probabilities = torch.einsum("BND,BCD->BNC", pairs, rel_prompts)[0].sigmoid()
    selected = ((probabilities > relation_threshold) & pair_mask[0].bool().unsqueeze(-1)).nonzero()
    mapping = batch["rel_id_to_classes"]
    mapping = mapping[0] if isinstance(mapping, list) else mapping
    raw = []
    for pair, label in selected.tolist():
        head, tail = pair_indices[0, pair].tolist()
        raw.append({"head": {"start": members[head]["start"] - window["start"],
                              "end": members[head]["end"] - window["start"],
                              "text": members[head]["surface_text"]},
                    "tail": {"start": members[tail]["start"] - window["start"],
                              "end": members[tail]["end"] - window["start"],
                              "text": members[tail]["surface_text"]},
                    "relation": mapping[label + 1], "score": probabilities[pair, label].item()})
    return raw, {"tokens": batch["input_ids"].shape[1], "spans": len(indices),
                 "pairs": int(pair_mask.sum()), "supplied_word_spans": indices}


def convert_prediction(raw, window, entities):
    lookup = {(e["start"], e["end"]): e for e in entities if e["document_id"] == window["document_id"]}
    endpoints = []
    for name in ("head", "tail"):
        part = raw[name]
        start, end = part["start"] + window["start"], part["end"] + window["start"]
        if not window["start"] <= start < end <= window["end"]:
            raise ValueError("Relation endpoint outside source window")
        entity = lookup.get((start, end))
        if entity is None or part["text"] != entity["surface_text"]:
            raise ValueError("Relation endpoint is not an exact supplied entity span")
        endpoints.append(entity)
    reverse = {label: code for code, label in PREDICATES.items()}
    if raw["relation"] not in reverse or not math.isfinite(raw["score"]) or not 0 <= raw["score"] <= 1:
        raise ValueError("Invalid relation label/confidence")
    code = reverse[raw["relation"]]
    head, tail = endpoints
    hints = TYPE_HINTS.get(code)
    suspicious = bool(hints and any(expected is not None and not expected.intersection(entity["entity_types"])
                                   for expected, entity in zip(hints, endpoints)))
    return {"triple_id": "t:" + digest([window["document_id"], head["entity_id"], code, tail["entity_id"]]),
            "document_id": window["document_id"], "head_id": head["entity_id"], "tail_id": tail["entity_id"],
            "head_types": head["entity_types"], "tail_types": tail["entity_types"],
            "relation_type": code, "confidence": raw["score"],
            "suspicious_type_hint": suspicious, "evidence": [window]}


def deduplicate(relations):
    unique = {}
    for row in relations:
        key = row["triple_id"]
        if key not in unique:
            unique[key] = {**row, "evidence": list(row["evidence"])}
        else:
            unique[key]["confidence"] = max(unique[key]["confidence"], row["confidence"])
            existing = {w["window_id"] for w in unique[key]["evidence"]}
            unique[key]["evidence"].extend(w for w in row["evidence"] if w["window_id"] not in existing)
    return [unique[k] for k in sorted(unique)]


def assigned_documents(documents, worker_index, workers_per_model):
    """Disjoint, deterministic document shards; checkpoints stay independent of worker count."""
    if workers_per_model < 1 or not 0 <= worker_index < workers_per_model:
        raise ValueError("Invalid worker assignment")
    return documents[worker_index::workers_per_model]


def aggregate_worker_timings(timings):
    """Concurrent workers: model wall time is the longest worker, not their sum."""
    return {"worker_wall_seconds": max(t["worker_wall_seconds"] for t in timings),
            "inference_wall_seconds": max(t["inference_wall_seconds"] for t in timings),
            "processed_documents": sum(t["processed_documents"] for t in timings),
            "resumed_documents": sum(t["resumed_documents"] for t in timings),
            "workers": timings}


def worker(variant, worker_index, config, barrier, events):
    out = Path(config["output"])
    worker_id = f"{variant}.worker{worker_index + 1}"
    with (out / f"{worker_id}.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                started = time.perf_counter()
                os.environ.update(OMP_NUM_THREADS=str(config["threads"]), MKL_NUM_THREADS=str(config["threads"]),
                                  OPENBLAS_NUM_THREADS=str(config["threads"]), TOKENIZERS_PARALLELISM="false",
                                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                payload = json.loads((out / "prepared_inputs.json").read_text(encoding="utf-8"))
                documents = assigned_documents(payload["documents"], worker_index, config["workers_per_model"])
                checkpoints = out / "checkpoints" / variant
                checkpoints.mkdir(parents=True, exist_ok=True)
                pending = []
                for doc in documents:
                    path = checkpoints / f"{doc['document_id']}.json"
                    if config["resume"] and path.exists():
                        cached = json.loads(path.read_text(encoding="utf-8"))
                        if cached["fingerprint"] != config["fingerprint"] or cached["document_id"] != doc["document_id"]:
                            raise ValueError("Resume checkpoint fingerprint mismatch")
                        events.put((worker_id, "progress", doc["document_id"]))
                    else:
                        pending.append(doc)
                model = None
                if pending:
                    import torch
                    from gliner import GLiNER
                    torch.set_num_threads(config["threads"])
                    torch.set_num_interop_threads(1)
                    model = GLiNER.from_pretrained(config["models"][variant], local_files_only=True).to("cpu").eval()
                    api = {"class": type(model).__name__, "inference_signature": str(inspect.signature(model.inference)),
                           "model_forward_signature": str(inspect.signature(model.model.forward)),
                           "represent_spans_source": inspect.getsource(type(model.model).represent_spans),
                           "inference_source_sha256": digest(inspect.getsource(type(model).inference)),
                           "span_mode": model.config.span_mode,
                           "adapter": "trained encoder -> exact E3 endpoint projections -> all directed pairs -> trained relation head; NER selection bypassed"}
                    atomic_json(out / f"{worker_id}.api.json", api)
                    if worker_index == 0:
                        atomic_json(out / f"{variant}.api.json", api)
                    print(json.dumps(api, ensure_ascii=False), flush=True)
                events.put((worker_id, "ready", len(pending)))
                barrier.wait(timeout=600)
                inference_start_utc = datetime.now(timezone.utc).isoformat()
                inference_started = time.perf_counter()
                for doc in pending:
                    begin = time.perf_counter()
                    members = [e for e in payload["entities"] if e["document_id"] == doc["document_id"]]
                    entity_map = {e["entity_id"]: e for e in members}
                    raw_windows, relations = [], []
                    for window in payload["windows"]:
                        if window["document_id"] != doc["document_id"]:
                            continue
                        supplied = [entity_map[eid] for eid in window["entity_ids"]]
                        raw, audit = predict_window(model, window, supplied, config["entity_labels"],
                                                    config["relation_threshold"], config["token_limit"])
                        raw_windows.append({"window": window, "audit": audit, "predictions": raw})
                        relations.extend(convert_prediction(r, window, members) for r in raw)
                    result = {"fingerprint": config["fingerprint"], "document_id": doc["document_id"],
                              "model": config["model_ids"][variant], "raw_windows": raw_windows,
                              "relations": deduplicate(relations), "raw_relation_count": len(relations),
                              "seconds": time.perf_counter() - begin}
                    atomic_json(checkpoints / f"{doc['document_id']}.json", result)
                    print(f"{doc['document_id']} raw={len(relations)} unique={len(result['relations'])} seconds={result['seconds']:.2f}", flush=True)
                    events.put((worker_id, "progress", doc["document_id"]))
                atomic_json(out / f"{worker_id}.timing.json", {
                    "worker_id": worker_id,
                    "document_ids": [d["document_id"] for d in documents],
                    "worker_wall_seconds": time.perf_counter() - started,
                    "inference_wall_seconds": time.perf_counter() - inference_started,
                    "processed_documents": len(pending), "resumed_documents": len(documents) - len(pending),
                    "inference_start_utc": inference_start_utc,
                    "inference_end_utc": datetime.now(timezone.utc).isoformat(),
                })
                events.put((worker_id, "done", None))
            except BaseException:
                traceback.print_exc()
                barrier.abort()
                events.put((worker_id, "error", traceback.format_exc()))
                raise


def graph_key(namespace, identity):
    if namespace not in NAMESPACES:
        raise ValueError("Only dedicated RE-V2-001 namespaces are writable")
    return f"{namespace}:{identity}"


def edge_row(namespace, relation, model_id, fingerprint):
    evidence = max(relation["evidence"], key=lambda w: len(w["text"]))
    return {"head_key": graph_key(namespace, relation["head_id"]),
            "tail_key": graph_key(namespace, relation["tail_id"]),
            "properties": {"key": graph_key(namespace, relation["triple_id"]), "namespace": namespace,
                           "document_id": relation["document_id"], "relation_type": relation["relation_type"],
                           "confidence": relation["confidence"], "model": model_id, "fingerprint": fingerprint,
                           "source_url": f"https://openalex.org/{relation['document_id']}",
                           "window_text": evidence["text"], "window_start": evidence["start"], "window_end": evidence["end"],
                           "evidence_json": json.dumps(relation["evidence"], ensure_ascii=False),
                           "suspicious_type_hint": relation["suspicious_type_hint"]}}


DOCUMENT_QUERY = """
MERGE (d:RELEXDocument {key: $key, namespace: $namespace}) SET d += $properties
"""
ENTITY_QUERY = """
MATCH (d:RELEXDocument {key: $document_key, namespace: $namespace})
UNWIND $rows AS row
MERGE (e:RELEXEntity {key: row.key, namespace: $namespace}) SET e += row.properties
MERGE (d)-[m:RELEX_MENTIONS {key: row.mention_key, namespace: $namespace}]->(e)
SET m.model = $model, m.document_id = $document_id
RETURN count(e) AS n
"""
EDGE_QUERY = """
UNWIND $rows AS row
MATCH (h:RELEXEntity {key: row.head_key, namespace: $namespace})
MATCH (t:RELEXEntity {key: row.tail_key, namespace: $namespace})
WHERE h.document_id = t.document_id AND h.document_id = row.properties.document_id
MERGE (h)-[r:RELEX_REL {key: row.properties.key, namespace: $namespace}]->(t)
SET r += row.properties
RETURN count(r) AS n
"""


def import_graph(driver, database, namespace, documents, entities, predictions, config):
    model = config["model_ids"][namespace.rsplit("_", 1)[-1]]
    with driver.session(database=database) as session:
        for label in ("RELEXDocument", "RELEXEntity"):
            session.run(f"CREATE CONSTRAINT uq_{label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE").consume()
        # Claim this namespace for this exact extraction configuration; no old ontology labels are touched.
        existing = session.run("MATCH (n) WHERE n.namespace=$ns RETURN DISTINCT n.fingerprint AS f",
                               ns=namespace).data()
        if any(row["f"] != config["fingerprint"] for row in existing):
            raise ValueError(f"{namespace} contains another experiment; refusing to overwrite it")
        for document, prediction in zip(documents, predictions):
            doc_id = document["document_id"]
            members = [e for e in entities if e["document_id"] == doc_id]
            rows = []
            for e in members:
                props = {k: v for k, v in e.items() if k != "source_mentions"}
                props.update(key=graph_key(namespace, e["entity_id"]), namespace=namespace, model=model,
                             fingerprint=config["fingerprint"], source_url=document["source_url"],
                             source_mentions_json=json.dumps(e["source_mentions"], ensure_ascii=False))
                rows.append({"key": props["key"], "properties": props,
                             "mention_key": graph_key(namespace, "mention:" + e["entity_id"])})
            edges = [edge_row(namespace, r, model, config["fingerprint"]) for r in prediction["relations"]]

            def transaction(tx):
                props = {"document_id": doc_id, "title": document["title"], "text": document["text"],
                         "source_url": document["source_url"], "doi": document.get("doi"),
                         "revision_id": document["revision_id"], "namespace": namespace, "model": model,
                         "fingerprint": config["fingerprint"], "source_json": json.dumps(document, ensure_ascii=False)}
                tx.run(DOCUMENT_QUERY, key=graph_key(namespace, doc_id), namespace=namespace, properties=props).consume()
                count = tx.run(ENTITY_QUERY, document_key=graph_key(namespace, doc_id), namespace=namespace,
                               rows=rows, model=model, document_id=doc_id).single()["n"]
                if count != len(rows):
                    raise RuntimeError("Incomplete entity import")
                count = tx.run(EDGE_QUERY, namespace=namespace, rows=edges).single()["n"]
                if count != len(edges):
                    raise RuntimeError("Incomplete directed relation import")
            session.execute_write(transaction)
        counts = {}
        doc_ids = [d["document_id"] for d in documents]
        for label, name in (("RELEXDocument", "documents"), ("RELEXEntity", "entities")):
            counts[name] = session.run(f"MATCH (n:{label}) WHERE n.namespace=$ns AND n.document_id IN $ids RETURN count(n) AS n",
                                       ns=namespace, ids=doc_ids).single()["n"]
        for label, name in (("RELEX_REL", "relations"), ("RELEX_MENTIONS", "mentions")):
            counts[name] = session.run(f"MATCH ()-[r:{label}]->() WHERE r.namespace=$ns AND r.document_id IN $ids RETURN count(r) AS n",
                                       ns=namespace, ids=doc_ids).single()["n"]
        counts["cross_namespace_edges"] = session.run("""
            MATCH (h)-[r]->(t) WHERE r.namespace=$ns
            AND (h.namespace IS NULL OR t.namespace IS NULL OR h.namespace<>$ns OR t.namespace<>$ns)
            RETURN count(r) AS n""", ns=namespace).single()["n"]
        expected = {"documents": len(documents), "entities": len(entities), "mentions": len(entities),
                    "relations": sum(len(p["relations"]) for p in predictions), "cross_namespace_edges": 0}
        if counts != expected:
            raise RuntimeError(f"Neo4j verification failed: {counts} != {expected}")
        return counts


def connect_neo4j(env_file=None):
    if env_file:
        # Read only Neo4j variables into process environment; never write or log credentials.
        for line in Path(env_file).read_text(encoding="utf-8-sig").splitlines():
            match = re.fullmatch(r"\s*(NEO4J_(?:URI|USER|PASSWORD|DATABASE))\s*=\s*(.*?)\s*", line)
            if match:
                os.environ.setdefault(match[1], match[2].strip("\"'"))
    from neo4j import GraphDatabase
    uri = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:17687")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Set NEO4J_PASSWORD (or use --neo4j-env with the existing project's .env)")
    driver = GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USER", "neo4j"), password),
                                  connection_timeout=10)
    driver.verify_connectivity()
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    with driver.session(database=database) as session:
        session.run("RETURN 1").consume()
    return driver, database


def legacy_counts(driver, database):
    """Read-only guard: existing data outside our two namespaces must keep its counts."""
    with driver.session(database=database) as session:
        return {"nodes": session.run("MATCH (n) WHERE NOT coalesce(n.namespace,'') IN $ns RETURN count(n) AS n",
                    ns=sorted(NAMESPACES)).single()["n"],
                "edges": session.run("MATCH ()-[r]->() WHERE NOT coalesce(r.namespace,'') IN $ns RETURN count(r) AS n",
                    ns=sorted(NAMESPACES)).single()["n"]}


def csv_file(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(out, documents, entities, results, graph_counts, elapsed, memory):
    stats, triple_sets, all_relations = {}, {}, {}
    for variant, predictions in results.items():
        relations = [r for p in predictions for r in p["relations"]]
        all_relations[variant] = relations
        triple_sets[variant] = {r["triple_id"] for r in relations}
        involved = {r[k] for r in relations for k in ("head_id", "tail_id")}
        scores = sorted(r["confidence"] for r in relations)
        timing = json.loads((out / f"{variant}.timing.json").read_text(encoding="utf-8"))
        stats[variant] = {"model": variant, "documents": len(documents), "entities": len(entities),
                          "source_mentions": sum(len(e["source_mentions"]) for e in entities),
                          "extracted_relations": sum(p["raw_relation_count"] for p in predictions),
                          "unique_triples": len(relations), "involved_entities": len(involved),
                          "entity_coverage": len(involved) / len(entities),
                          "suspicious_type_hint_fraction": sum(r["suspicious_type_hint"] for r in relations) / max(1, len(relations)),
                          "endpoint_violations": 0, "confidence_mean": statistics.mean(scores) if scores else None,
                          "confidence_min": min(scores) if scores else None, "confidence_max": max(scores) if scores else None,
                          "document_inference_seconds": sum(p["seconds"] for p in predictions),
                          "worker_wall_seconds": timing["worker_wall_seconds"],
                          "resumed_documents": timing["resumed_documents"],
                          "peak_rss_mib": memory.get(variant, 0) / 2**20,
                          "total_run_seconds": elapsed, "peak_combined_rss_mib": memory["combined"] / 2**20}
        write_jsonl(out / f"{variant}.predictions.jsonl", predictions)
        write_jsonl(out / f"{variant}.relations.jsonl", relations)
    intersection = triple_sets["base"] & triple_sets["large"]
    union = triple_sets["base"] | triple_sets["large"]
    agreement = {"shared_triples": len(intersection), "union_triples": len(union),
                 "jaccard": len(intersection) / len(union) if union else None,
                 "base_only": len(triple_sets["base"] - triple_sets["large"]),
                 "large_only": len(triple_sets["large"] - triple_sets["base"])}
    atomic_json(out / "comparison.json", {"models": stats, "agreement": agreement, "neo4j": graph_counts})
    for variant in results:
        write_jsonl(out / f"{variant}_only.jsonl", [r for r in all_relations[variant] if r["triple_id"] not in intersection])
    csv_file(out / "metrics.csv", list(stats.values()), list(stats["base"]))
    distribution = []
    for code in PREDICATES:
        distribution.append({"relation_type": code, **{v: sum(r["relation_type"] == code for r in all_relations[v]) for v in results}})
    csv_file(out / "relation_distribution.csv", distribution, ["relation_type", "base", "large"])
    csv_file(out / "per_document.csv", [{"document_id": d["document_id"],
             **{v: len(results[v][i]["relations"]) for v in results}} for i, d in enumerate(documents)],
             ["document_id", "base", "large"])
    confidence_bins = []
    for i in range(10):
        confidence_bins.append({"lower": i / 10, "upper": (i + 1) / 10,
            **{v: sum(i / 10 <= r["confidence"] <= 1 and (r["confidence"] < (i + 1) / 10 or i == 9)
                      for r in all_relations[v]) for v in results}})
    csv_file(out / "confidence_distribution.csv", confidence_bins, ["lower", "upper", "base", "large"])
    lines = ["# RE-V2-001: original E3 -> GLiNER-RelEx -> Neo4j", "",
             "| Metric | Base | Large |", "|---|---:|---:|"]
    for metric in stats["base"]:
        if metric != "model":
            lines.append(f"| {metric} | {stats['base'][metric]} | {stats['large'][metric]} |")
    lines += ["", f"Triple agreement: `{json.dumps(agreement)}`.", "",
              f"Verified Neo4j counts: `{json.dumps(graph_counts)}`.", "",
              "Original E3 may contain several entity types per span: all source records are preserved; one node per document/start/end.",
              "Predictions use trained encoder/span projections/relation head with exact E3 endpoints; both Base token-level and Large markerV0 NER selection are bypassed.",
              "All above-threshold window predictions are preserved before overlap deduplication. Triple confidence is the maximum across windows.",
              "Direction is preserved. Exact E3 endpoints and same-document membership are validated before import.",
              "Type hints are nonbinding review candidates inferred from predicate definitions. Ontology v2 has no approved domain/range matrix; formal violation rate is unavailable.",
              "No suspicious triples are removed. Model disagreement and type hints do not establish an error.",
              "No gold relation labels were used; RE quality metrics are unavailable.",
              "RAM is sampled process RSS (parent plus all workers), every 0.25s; per-model RSS is summed across its workers at each sample. Docker/server RAM is excluded. Values are measured samples, not OS high-water marks.",
              "Resume retains per-document inference durations; worker/total wall time describe this invocation.", "",
              "Browser: http://127.0.0.1:17474 ; connect to bolt://127.0.0.1:17687.",
              "Use base.cypher / large.cypher to show isolated entities as well as directed relations.", "",
              "API references: [Base model card](https://huggingface.co/knowledgator/gliner-relex-base-v1.0), "
              "[Large model card](https://huggingface.co/knowledgator/gliner-relex-large-v1.0). "
              "The actual installed API audit is saved in base.api.json / large.api.json."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:7]))
    print(f"Shared triples: {len(intersection)}; Base only: {agreement['base_only']}; Large only: {agreement['large_only']}")
    print(f"Report: {out / 'report.md'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base-model", type=Path, default=ROOT / "artifacts/models/gliner-relex-base-v1.0")
    parser.add_argument("--large-model", type=Path, default=ROOT / "artifacts/models/gliner-relex-large-v1.0")
    parser.add_argument("--workers-per-model", type=int, default=1, help="Independent document workers per model (default: 1)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per worker (default: 4)")
    parser.add_argument("--relation-threshold", type=float, default=0.7)
    parser.add_argument("--token-limit", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--neo4j-env", type=Path, help="Read only NEO4J_* from an existing .env; never copied")
    parser.add_argument("--inference-only", action="store_true", help="Save predictions without Neo4j; later use --resume to import")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.workers_per_model < 1 or not 0 < args.relation_threshold < 1 or not 0 < args.overlap < 1:
        parser.error("Invalid worker/thread count, threshold or overlap")
    started = time.perf_counter()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    driver, database = (None, os.environ.get("NEO4J_DATABASE", "neo4j")) if args.inference_only else connect_neo4j(args.neo4j_env)
    try:
        legacy_before = legacy_counts(driver, database) if driver else None
        print("Preparing original E3, model fingerprints and common token windows...", flush=True)
        documents, entities = load_inputs(args.limit)
        ontology = json.loads(ONTOLOGY.read_text(encoding="utf-8"))
        if set(ontology["relation_types"]) != set(PREDICATES):
            raise ValueError("Ontology must match the original 20 predicates")
        paths = {"base": args.base_model.resolve(), "large": args.large_model.resolve()}
        model_files = {}
        for variant, path in paths.items():
            model_files[variant] = {}
            for name in ("gliner_config.json", "model.safetensors", "tokenizer.json", "spm.model"):
                file = path / name
                if not file.is_file():
                    raise FileNotFoundError(f"Downloaded model incomplete: {file}")
                with file.open("rb") as stream:
                    model_files[variant][name] = hashlib.file_digest(stream, "sha256").hexdigest()
        windows, labels, budget = plan_windows(documents, entities, paths, ontology, args.token_limit, args.overlap)
        # Corpus selection is omitted from graph fingerprint so smoke and full run upsert identical triples.
        signature = {"version": VERSION, "ontology": ontology, "predicates": PREDICATES,
                     "models": model_files, "relation_threshold": args.relation_threshold,
                     "token_limit": budget, "overlap": args.overlap, "entity_labels": labels,
                     "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (SOURCE / "input_abstracts.jsonl", SOURCE / "E3/mentions_035.jsonl")}}
        config = {**signature, "fingerprint": digest(signature), "output": str(out), "threads": args.threads,
                  "workers_per_model": args.workers_per_model,
                  "resume": args.resume, "models": {v: str(p) for v, p in paths.items()},
                  "model_files_sha256": model_files,
                  "model_ids": {v: f"knowledgator/gliner-relex-{v}-v1.0" for v in paths},
                  "document_ids": [d["document_id"] for d in documents], "windows": len(windows),
                  "neo4j_database": database, "type_hints": {k: [sorted(x) if x else None for x in v] for k, v in TYPE_HINTS.items()}}
        previous = out / "experiment_config.json"
        if previous.exists():
            old = json.loads(previous.read_text(encoding="utf-8"))
            if old["fingerprint"] != config["fingerprint"] or old["document_ids"] != config["document_ids"]:
                raise ValueError("Output folder belongs to different inputs/settings; choose another --output")
            if not args.resume and any((out / "checkpoints").glob("*/*.json")):
                raise ValueError("Output folder already has a run; use --resume or another --output")
        atomic_json(previous, config)
        atomic_json(out / "prepared_inputs.json", {"documents": documents, "entities": entities, "windows": windows})
        write_jsonl(out / "entities.jsonl", entities)
        write_jsonl(out / "windows.jsonl", windows)
        for variant in paths:
            ns = "relex_" + variant
            (out / f"{variant}.cypher").write_text(
                f"// Includes all entities, including isolated nodes.\n"
                f"MATCH (e:RELEXEntity {{namespace: '{ns}'}})\n"
                f"OPTIONAL MATCH (e)-[r:RELEX_REL {{namespace: '{ns}'}}]->(t:RELEXEntity {{namespace: '{ns}'}})\n"
                "RETURN e, r, t;\n\n"
                f"MATCH (d:RELEXDocument {{namespace: '{ns}'}})-[r:RELEX_MENTIONS {{namespace: '{ns}'}}]->(e:RELEXEntity {{namespace: '{ns}'}})\n"
                "RETURN d,r,e;\n", encoding="utf-8")
        import psutil
        from tqdm import tqdm
        context = mp.get_context("spawn")
        assignments = {f"{v}.worker{i + 1}": (v, i) for v in paths for i in range(args.workers_per_model)}
        events, barrier = context.Queue(), context.Barrier(len(assignments))
        workers = {wid: context.Process(target=worker, args=(v, i, config, barrier, events), name=f"relex-{wid}")
                   for wid, (v, i) in assignments.items()}
        memory = {"base": 0, "large": 0, "combined": 0, **{wid: 0 for wid in workers}}
        bars = {wid: tqdm(total=len(assigned_documents(documents, i, args.workers_per_model)),
                          desc=f"RelEx {wid}", position=position, unit="doc")
                for position, (wid, (_, i)) in enumerate(assignments.items())}
        print(f"Starting {len(workers)} workers: {args.workers_per_model} per model, {args.threads} CPU threads each", flush=True)
        for process in workers.values():
            process.start()
        done = set()
        try:
            while len(done) < len(workers):
                combined = psutil.Process().memory_info().rss
                model_rss = {v: 0 for v in paths}
                for worker_id, process in workers.items():
                    try:
                        rss = psutil.Process(process.pid).memory_info().rss
                        memory[worker_id] = max(memory[worker_id], rss)
                        model_rss[assignments[worker_id][0]] += rss
                        combined += rss
                    except psutil.NoSuchProcess:
                        pass
                for variant, rss in model_rss.items():
                    memory[variant] = max(memory[variant], rss)
                memory["combined"] = max(memory["combined"], combined)
                try:
                    worker_id, event, value = events.get(timeout=0.25)
                    if event == "progress":
                        bars[worker_id].update(1)
                    elif event == "ready":
                        tqdm.write(f"{worker_id}: ready; {value} documents to infer (waiting for peers)")
                    elif event == "done":
                        done.add(worker_id)
                    elif event == "error":
                        raise RuntimeError(f"{worker_id} failed; see {out / (worker_id + '.log')}\n{value}")
                except queue.Empty:
                    pass
                for variant, process in workers.items():
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(f"{variant} worker exited {process.exitcode}; inspect its log")
        finally:
            for process in workers.values():
                if len(done) != len(workers) and process.is_alive():
                    process.terminate()
                process.join(timeout=15)
                if process.is_alive():
                    process.kill()
            for bar in bars.values():
                bar.close()
        results, counts = {}, {}
        for variant in paths:
            timings = [json.loads((out / f"{wid}.timing.json").read_text(encoding="utf-8"))
                       for wid, (v, _) in assignments.items() if v == variant]
            atomic_json(out / f"{variant}.timing.json", aggregate_worker_timings(timings))
            predictions = [json.loads((out / "checkpoints" / variant / f"{d['document_id']}.json").read_text(encoding="utf-8")) for d in documents]
            # Revalidate even cached predictions before ANY graph write.
            for prediction in predictions:
                source_windows = [w for w in windows if w["document_id"] == prediction["document_id"]]
                if [w["window"] for w in prediction["raw_windows"]] != source_windows:
                    raise ValueError("Checkpoint source window integrity failure")
                expected = deduplicate([convert_prediction(r, w["window"], entities)
                                        for w in prediction["raw_windows"] for r in w["predictions"]])
                if expected != prediction["relations"] or prediction["fingerprint"] != config["fingerprint"]:
                    raise ValueError("Checkpoint relation integrity failure")
            results[variant] = predictions
        for variant, predictions in results.items():
            counts[variant] = (import_graph(driver, database, "relex_" + variant, documents, entities, predictions, config)
                               if driver else {"status": "NOT_IMPORTED", "reason": "inference-only invocation"})
        if driver:
            legacy_after = legacy_counts(driver, database)
            atomic_json(out / "legacy_data_verification.json", {"before": legacy_before, "after": legacy_after,
                                                               "unchanged_counts": legacy_before == legacy_after})
            if legacy_before != legacy_after:
                raise RuntimeError("Data counts outside the two dedicated namespaces changed during this run")
        atomic_json(out / "resources.json", {"peak_sampled_rss_bytes": memory, "sample_interval_seconds": 0.25})
        report(out, documents, entities, results, counts, time.perf_counter() - started, memory)
    finally:
        if driver:
            driver.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
