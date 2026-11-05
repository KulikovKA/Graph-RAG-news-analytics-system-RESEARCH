"""Shared implementation of the four independently launchable RelEx experiments."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import queue
import statistics
import threading
import time
import traceback
import unicodedata

import build_relex_graphs as core


def experiment_namespace(mode, variant):
    if mode not in {"e3", "joint"} or variant not in {"base", "large"}:
        raise ValueError("Unknown experiment")
    return f"relex_{variant}" if mode == "e3" else f"relex_joint_{variant}"


def resolve_namespace(mode, variant, requested=None):
    default = experiment_namespace(mode, variant)
    namespace = requested or default
    core.namespace_variant(namespace)
    if namespace != default and not namespace.startswith(default + "_"):
        raise ValueError(f"Namespace must be {default} or {default}_<run suffix>")
    return namespace


def infer_joint_window(model, window, labels, entity_threshold, relation_threshold):
    """Native joint NER+RE API: no E3 entities, input_spans or forced endpoints."""
    entities, relations = model.inference(
        texts=[window["text"]], labels=labels, relations=list(core.PREDICATES.values()),
        threshold=entity_threshold, relation_threshold=relation_threshold,
        flat_ner=False, multi_label=True, return_relations=True, batch_size=1,
    )
    if len(entities) != 1 or len(relations) != 1:
        raise ValueError("Unexpected native joint output batch")
    return entities[0], relations[0]


def joint_entities(document, raw_windows, labels, entity_codes):
    """Preserve native entity predictions and merge only identical document offsets."""
    label_to_code = dict(zip(labels, entity_codes))
    groups = defaultdict(list)
    for result in raw_windows:
        window = result["window"]
        if window["document_id"] != document["document_id"] or window["text"] != document["text"][window["start"]:window["end"]]:
            raise ValueError("Joint source window mismatch")
        for raw in result["entities"]:
            start, end = raw["start"] + window["start"], raw["end"] + window["start"]
            if not window["start"] <= start < end <= window["end"]:
                raise ValueError("Joint entity outside source window")
            if raw["text"] != document["text"][start:end] or raw["label"] not in label_to_code:
                raise ValueError("Invalid joint entity text/type")
            if not math.isfinite(raw["score"]) or not 0 <= raw["score"] <= 1:
                raise ValueError("Invalid joint entity confidence")
            groups[(start, end)].append({
                "configuration": "joint_relex", "document_id": document["document_id"],
                "start": start, "end": end, "surface_text": raw["text"],
                "ner_label_hint": label_to_code[raw["label"]], "score": raw["score"],
                "normalized_form": unicodedata.normalize("NFC", raw["text"]).strip().casefold(),
                "normalizer_version": "joint-nfc-casefold-v1", "window_id": window["window_id"],
                "revision_id": document["revision_id"], "source_reference": document["source_reference"],
                "raw_prediction": raw,
            })
    return [{"entity_id": "e:" + core.digest([document["document_id"], start, end]),
             "document_id": document["document_id"], "start": start, "end": end,
             "surface_text": document["text"][start:end],
             "entity_types": sorted({m["ner_label_hint"] for m in records}),
             "normalized_forms": sorted({m["normalized_form"] for m in records}),
             "confidence": max(m["score"] for m in records), "source_mentions": records}
            for (start, end), records in sorted(groups.items())]


def mapped_relations(raw_windows, entities):
    return core.deduplicate([core.convert_prediction(raw, result["window"], entities)
                             for result in raw_windows for raw in result["predictions"]])


def model_fingerprints(paths):
    fingerprints = {}
    for name, path in paths.items():
        fingerprints[name] = {}
        for filename in ("gliner_config.json", "model.safetensors", "tokenizer.json", "spm.model"):
            with (path / filename).open("rb") as stream:
                fingerprints[name][filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    return fingerprints


def load_model(path, threads):
    import torch
    from gliner import GLiNER
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return GLiNER.from_pretrained(str(path), local_files_only=True).to("cpu").eval()


def validate_checkpoint(result, document, windows, input_entities, config):
    if result["fingerprint"] != config["fingerprint"] or result["document_id"] != document["document_id"]:
        raise ValueError("Resume checkpoint experiment/document mismatch")
    if [r["window"] for r in result["raw_windows"]] != windows:
        raise ValueError("Resume source window mismatch")
    expected_entities = (joint_entities(document, result["raw_windows"], config["entity_labels"], config["entity_codes"])
                         if config["mode"] == "joint" else input_entities)
    if result["entities"] != expected_entities or result["relations"] != mapped_relations(result["raw_windows"], expected_entities):
        raise ValueError("Resume prediction integrity failure")
    if result["raw_relation_count"] != sum(len(r["predictions"]) for r in result["raw_windows"]):
        raise ValueError("Resume raw prediction count mismatch")


class MemorySampler:
    """Sample the parent and its model workers, including loading and inference."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        import psutil
        while not self.stop.wait(0.25):
            rss = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak = max(self.peak, rss)

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def process_shard(mode, variant, worker_index, config, barrier, events):
    """One model copy, one nonoverlapping shard, one log; no graph writes in workers."""
    out = Path(config["output"])
    worker_id = f"worker{worker_index + 1}"
    with (out / f"{worker_id}.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                started = time.perf_counter()
                payload = json.loads((out / "prepared_inputs.json").read_text(encoding="utf-8"))
                documents = core.assigned_documents(payload["documents"], worker_index, config["workers"])
                pending = []
                for document in documents:
                    windows = [w for w in payload["windows"] if w["document_id"] == document["document_id"]]
                    members = [e for e in payload["entities"] if e["document_id"] == document["document_id"]]
                    path = out / "checkpoints" / f"{document['document_id']}.json"
                    if config["resume"] and path.exists():
                        validate_checkpoint(json.loads(path.read_text(encoding="utf-8")), document, windows, members, config)
                        events.put((worker_id, "progress", document["document_id"]))
                    else:
                        pending.append((document, windows, members))
                model, load_seconds = None, 0
                if pending:
                    begin = time.perf_counter()
                    model = load_model(config["model_path"], config["threads"])
                    load_seconds = time.perf_counter() - begin
                    core.atomic_json(out / f"{worker_id}.api.json", {"mode": mode, "class": type(model).__name__,
                        "inference_signature": str(inspect.signature(model.inference)),
                        "entity_source": "original E3" if mode == "e3" else "native joint predictions"})
                events.put((worker_id, "ready", len(pending)))
                barrier.wait(timeout=600)
                for document, windows, members in pending:
                    begin = time.perf_counter()
                    raw_windows = []
                    for window in windows:
                        if mode == "e3":
                            supplied = [e for e in members if e["entity_id"] in window["entity_ids"]]
                            raw, audit = core.predict_window(model, window, supplied, config["entity_labels"],
                                config["relation_threshold"], config["token_limit"])
                            raw_windows.append({"window": window, "audit": audit, "predictions": raw})
                        else:
                            native_entities, raw = infer_joint_window(model, window, config["entity_labels"],
                                config["entity_threshold"], config["relation_threshold"])
                            raw_windows.append({"window": window, "entities": native_entities, "predictions": raw})
                    if mode == "joint":
                        members = joint_entities(document, raw_windows, config["entity_labels"], config["entity_codes"])
                    result = {"fingerprint": config["fingerprint"], "document_id": document["document_id"],
                        "model": config["model_ids"][variant], "mode": mode, "raw_windows": raw_windows,
                        "entities": members, "relations": mapped_relations(raw_windows, members),
                        "raw_relation_count": sum(len(w["predictions"]) for w in raw_windows),
                        "seconds": time.perf_counter() - begin}
                    core.atomic_json(out / "checkpoints" / f"{document['document_id']}.json", result)
                    print(f"{document['document_id']} entities={len(members)} triples={len(result['relations'])} seconds={result['seconds']:.2f}", flush=True)
                    events.put((worker_id, "progress", document["document_id"]))
                core.atomic_json(out / f"{worker_id}.timing.json", {"worker": worker_id, "model_load_seconds": load_seconds,
                    "wall_seconds": time.perf_counter() - started, "processed_documents": len(pending),
                    "resumed_documents": len(documents) - len(pending), "document_ids": [d["document_id"] for d in documents]})
                events.put((worker_id, "done", None))
            except BaseException:
                traceback.print_exc()
                barrier.abort()
                events.put((worker_id, "error", traceback.format_exc()))
                raise


def execute_workers(mode, variant, config, documents):
    from tqdm import tqdm
    context = core.mp.get_context("spawn")
    events, barrier = context.Queue(), context.Barrier(config["workers"])
    processes = {f"worker{i + 1}": context.Process(target=process_shard,
        args=(mode, variant, i, config, barrier, events), name=f"{mode}-{variant}-{i + 1}") for i in range(config["workers"])}
    bars = {f"worker{i + 1}": tqdm(total=len(core.assigned_documents(documents, i, config["workers"])),
        desc=f"{mode}_{variant}/{i + 1}", position=i, unit="doc") for i in range(config["workers"])}
    completed = set()
    try:
        for process in processes.values():
            process.start()  # Start every model copy before waiting for any result.
        while len(completed) < len(processes):
            try:
                worker_id, event, value = events.get(timeout=.25)
                if event == "progress":
                    bars[worker_id].update(1)
                elif event == "ready":
                    tqdm.write(f"{mode}_{variant}/{worker_id}: ready, {value} documents to infer")
                elif event == "done":
                    completed.add(worker_id)
                elif event == "error":
                    raise RuntimeError(f"{worker_id} failed: {value}")
            except queue.Empty:
                pass
            for worker_id, process in processes.items():
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"{worker_id} exited {process.exitcode}; see its log")
    finally:
        for process in processes.values():
            if process.pid is None:
                continue
            if len(completed) != len(processes) and process.is_alive():
                process.terminate()
            process.join(timeout=15)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for bar in bars.values():
            bar.close()
        events.close()
        events.join_thread()
    return [json.loads((Path(config["output"]) / f"{wid}.timing.json").read_text(encoding="utf-8")) for wid in processes]


def write_report(out, config, documents, predictions, graph_counts, elapsed, model_seconds, peak, resumed):
    entities = [e for p in predictions for e in p["entities"]]
    relations = [r for p in predictions for r in p["relations"]]
    involved = {r[k] for r in relations for k in ("head_id", "tail_id")}
    scores = [r["confidence"] for r in relations]
    metrics = {
        "experiment": f"{config['mode']}_{config['variant']}", "workers": config["workers"], "documents": len(documents),
        "entities": len(entities), "source_mentions": sum(len(e["source_mentions"]) for e in entities),
        "extracted_relations": sum(p["raw_relation_count"] for p in predictions),
        "unique_triples": len(relations), "involved_entities": len(involved),
        "entity_coverage": len(involved) / len(entities) if entities else 0,
        "endpoint_violations": 0,
        "suspicious_type_hint_fraction": sum(r["suspicious_type_hint"] for r in relations) / len(relations) if relations else 0,
        "confidence_mean": statistics.mean(scores) if scores else None,
        "document_inference_seconds": sum(p["seconds"] for p in predictions),
        "model_load_seconds": model_seconds, "total_run_seconds": elapsed,
        "peak_sampled_rss_mib": peak / 2**20, "resumed_documents": resumed,
    }
    core.write_jsonl(out / "predictions.jsonl", predictions)
    core.write_jsonl(out / "relations.jsonl", relations)
    core.write_jsonl(out / "entities.jsonl", entities)
    core.atomic_json(out / "summary.json", {"metrics": metrics, "neo4j": graph_counts})
    core.csv_file(out / "metrics.csv", [metrics], list(metrics))
    distribution = Counter(r["relation_type"] for r in relations)
    core.csv_file(out / "relation_distribution.csv", [{"relation_type": code, "count": distribution[code]} for code in core.PREDICATES], ["relation_type", "count"])
    core.csv_file(out / "per_document.csv", [{"document_id": p["document_id"], "entities": len(p["entities"]),
                   "relations": len(p["relations"]), "seconds": p["seconds"]} for p in predictions],
                   ["document_id", "entities", "relations", "seconds"])
    core.csv_file(out / "confidence_distribution.csv", [{"lower": i / 10, "upper": (i + 1) / 10,
        "count": sum(i / 10 <= score and (score < (i + 1) / 10 or i == 9) for score in scores)} for i in range(10)],
        ["lower", "upper", "count"])
    source = ("Original E3 entities are preserved; RelEx predicts only relations between those exact spans."
              if config["mode"] == "e3" else
              "Native joint RelEx predicts entities and relations. E3 mentions are never read or supplied in this experiment.")
    lines = [f"# {config['mode']}_{config['variant']}", "", source, "", "| Metric | Value |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in metrics.items()]
    lines += ["", f"Neo4j: `{json.dumps(graph_counts)}`; namespace `{config['namespace']}`.",
              "", "No gold labels or RE quality metrics. Type hints are nonbinding review candidates, never filters.",
              "Every relation endpoint is validated against this experiment's entities in the same document.",
              "All raw window predictions and evidence are retained. Deduplication uses exact document/head/type/tail.",
              "Base and Large use a common token window plan within each mode. E3 windows protect known spans; joint windows use text only.",
              "Wall time and peak RSS describe this invocation; per-document times include resumed checkpoints.",
              "RSS is sampled every 0.25 seconds for the parent and its model workers; other experiments and Docker are excluded.",
              "", "Browser: http://127.0.0.1:17474 ; Bolt: bolt://127.0.0.1:17687 ; query: graph.cypher."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def run(mode, variant, argv=None):
    """One experiment owns parallel model copies, a report folder and graph namespace."""
    namespace = experiment_namespace(mode, variant)
    parser = argparse.ArgumentParser(description=f"{mode}_{variant}: parallel copies of one RelEx model, one isolated graph")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, choices=range(1, 31), metavar="1..30")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2, help="Parallel copies of this model (default: 2)")
    parser.add_argument("--relation-threshold", type=float, default=0.7)
    parser.add_argument("--entity-threshold", type=float, default=0.3, help="NER threshold for joint mode only")
    parser.add_argument("--output", type=Path, default=core.OUTPUT / f"{mode}_{variant}")
    parser.add_argument("--namespace", help="Separate Neo4j namespace, e.g. relex_large_t035")
    parser.add_argument("--neo4j-env", type=Path)
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.workers < 1 or not 0 < args.relation_threshold < 1 or not 0 < args.entity_threshold < 1:
        parser.error("Invalid workers, threads or thresholds")
    try:
        namespace = resolve_namespace(mode, variant, args.namespace)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    driver = None
    started = time.perf_counter()
    with MemorySampler() as memory, (out / "inference.log").open("a", encoding="utf-8", buffering=1) as log:
        try:
            database = os.environ.get("NEO4J_DATABASE", "neo4j")
            if not args.inference_only:
                driver, database = core.connect_neo4j(args.neo4j_env)
            legacy_before = core.legacy_counts(driver, database) if driver else None
            if mode == "e3":
                documents, input_entities = core.load_inputs(args.limit)
            else:
                documents, input_entities = core.load_documents(args.limit), []
            ontology = json.loads(core.ONTOLOGY.read_text(encoding="utf-8"))
            if set(ontology["relation_types"]) != set(core.PREDICATES):
                raise ValueError("Expected the original 20 Ontology v2 predicates")
            paths = {v: core.ROOT / f"artifacts/models/gliner-relex-{v}-v1.0" for v in ("base", "large")}
            model_files = model_fingerprints(paths)
            windows, labels, budget = core.plan_windows(documents, input_entities, paths, ontology, 512, .25)
            sources = [core.SOURCE / "input_abstracts.jsonl"]
            if mode == "e3":
                sources.append(core.SOURCE / "E3/mentions_035.jsonl")
            signature = {"version": core.VERSION, "ontology": ontology, "predicates": core.PREDICATES,
                "models": model_files, "relation_threshold": args.relation_threshold,
                "token_limit": budget, "overlap": .25, "entity_labels": labels,
                "source_sha256": {str(p.relative_to(core.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
            if mode == "joint":
                signature.update(mode="joint", entity_threshold=args.entity_threshold, joint_adapter="native-inference-v1")
            config = {**signature, "fingerprint": core.digest(signature), "mode": mode, "variant": variant,
                "namespace": namespace, "threads": args.threads, "entity_codes": list(ontology["entity_types"]),
                "workers": args.workers, "resume": args.resume, "output": str(out),
                "entity_threshold": args.entity_threshold if mode == "joint" else None,
                "model_path": str(paths[variant]), "model_ids": {variant: f"knowledgator/gliner-relex-{variant}-v1.0"},
                "document_ids": [d["document_id"] for d in documents], "neo4j_database": database}
            config_path = out / "experiment_config.json"
            checkpoint_dir = out / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            if config_path.exists():
                old = json.loads(config_path.read_text(encoding="utf-8"))
                if old["fingerprint"] != config["fingerprint"] or old["document_ids"] != config["document_ids"]:
                    raise ValueError("Output folder contains another experiment; select another --output")
                if not args.resume and any(checkpoint_dir.glob("*.json")):
                    raise ValueError("Existing results: use --resume or another --output")
            core.atomic_json(config_path, config)
            core.write_jsonl(out / "windows.jsonl", windows)
            core.atomic_json(out / "prepared_inputs.json", {"documents": documents, "entities": input_entities, "windows": windows})
            predictions, pending = {}, []
            for document in documents:
                doc_id = document["document_id"]
                path = checkpoint_dir / f"{doc_id}.json"
                doc_windows = [w for w in windows if w["document_id"] == doc_id]
                members = [e for e in input_entities if e["document_id"] == doc_id]
                if args.resume and path.exists():
                    result = json.loads(path.read_text(encoding="utf-8"))
                    validate_checkpoint(result, document, doc_windows, members, config)
                    predictions[doc_id] = result
                else:
                    pending.append((document, doc_windows, members))
            resumed = len(predictions)
            model_seconds = 0
            if pending:
                timings = execute_workers(mode, variant, config, documents)
                model_seconds = max(t["model_load_seconds"] for t in timings)
                core.atomic_json(out / "workers_timing.json", timings)
            for document in documents:
                result = json.loads((checkpoint_dir / f"{document['document_id']}.json").read_text(encoding="utf-8"))
                validate_checkpoint(result, document, [w for w in windows if w["document_id"] == document["document_id"]],
                    [e for e in input_entities if e["document_id"] == document["document_id"]], config)
                predictions[document["document_id"]] = result
            ordered = [predictions[d["document_id"]] for d in documents]
            entities = [e for p in ordered for e in p["entities"]]
            # Save complete inference outputs before attempting graph import.
            core.write_jsonl(out / "predictions.jsonl", ordered)
            core.write_jsonl(out / "relations.jsonl", [r for p in ordered for r in p["relations"]])
            core.write_jsonl(out / "entities.jsonl", entities)
            graph_counts = (core.import_graph(driver, database, namespace, documents, entities, ordered, config)
                            if driver else {"status": "NOT_IMPORTED"})
            if driver:
                legacy_after = core.legacy_counts(driver, database)
                core.atomic_json(out / "legacy_data_verification.json", {"before": legacy_before, "after": legacy_after,
                    "unchanged_counts": legacy_before == legacy_after})
                if legacy_before != legacy_after:
                    raise RuntimeError("Data outside dedicated RelEx namespaces changed during the run")
            (out / "graph.cypher").write_text(
                f"MATCH (e:RELEXEntity {{namespace:'{namespace}'}})\n"
                f"OPTIONAL MATCH (e)-[r:RELEX_REL {{namespace:'{namespace}'}}]->(t:RELEXEntity {{namespace:'{namespace}'}})\n"
                "RETURN e,r,t;\n\n"
                f"MATCH (d:RELEXDocument {{namespace:'{namespace}'}})-[m:RELEX_MENTIONS {{namespace:'{namespace}'}}]->(e) RETURN d,m,e;\n",
                encoding="utf-8")
            metrics = write_report(out, config, documents, ordered, graph_counts, time.perf_counter() - started,
                                   model_seconds, max(memory.peak, memory.process.memory_info().rss), resumed)
            print(f"{mode}_{variant}: {metrics['entities']} entities, {metrics['unique_triples']} triples; report: {out / 'report.md'}")
        except core.Neo4jConnectionError as exc:
            print(str(exc), file=log)
            raise SystemExit(str(exc)) from None
        except BaseException:
            traceback.print_exc(file=log)
            raise
        finally:
            if driver:
                driver.close()
