"""Подготовка исторического Corpus30 и пополнение графа Shared Concepts V2."""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import multiprocessing as mp
from pathlib import Path
import runpy
from time import perf_counter

from filelock import FileLock
from tqdm import tqdm

import build_corpus_graph as upstream
import corpus_graph_v2 as v2
import corpus_graph_v2_neo4j as neo4j
from cgraph.ner_v2 import CONTRACT_VERSION
from cgraph.ner_v2.io import read_documents, write_jsonl
from cgraph.ner_v2.normalization import detect_abbreviations
from corpus_graph_v2_inputs import BOOTSTRAP_EDGES, BOOTSTRAP_NODES, delta_identity
from prepare_semantic_entity_input import member_record
from run_shared_concepts_v2 import limits, resource_options
from shared_concepts_v2_pipeline import load_checked, write_checked

ROOT = upstream.ROOT
SOURCE = upstream.relex.SOURCE / "input_abstracts.jsonl"
NER = upstream.relex.SOURCE / "E3/mentions_035.jsonl"
RE = upstream.relex.OUTPUT / "threshold035/e3_base/predictions.jsonl"
ER1 = upstream.ER_HISTORY / "conservative_resolution/canonical_mapping.jsonl"
ER2 = upstream.ER_HISTORY / "global_kg_2f3/canonical_mapping.jsonl"
WORK = ROOT / "artifacts/append_corpus30"
E5 = ROOT.parent / "embedding-models/intfloat__e5-small-v2"
LOCKED = {
    SOURCE: "684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1",
    NER: "495027052bf560b66c4a0aef1a2342c63aa40a3359566d0571b554234ddc5dbf",
    RE: "4c6672ae1a837debfc5f3fdc3dfd4e723ea44aeda085e27c1e4f1527f2ea1a0e",
    ER1: "f97e8f920cf7cced3483caca39bdf87079f2b8a9c7d5240b812c886cc766c8b3",
    ER2: "1c3f825e522279ceee8b054d3cfa734a4072872c7dddf4cf8f01eedbee68a714",
    NER.with_name("config.json"): "9cc9fa30598c79b0f3c444372c287c5f4187ab08e16a8c39a4fc8d4991b4a882",
    RE.with_name("experiment_config.json"): "6e641e60f49a71ee133015cec6a99d7bd069ae73c1de650f10e5401f208729a8",
}
MODEL_FILES = {"ner": ("ner",), "re": ("re", "re_common")}
FACTORIES = {"ner": upstream.inference.ner_worker, "re": upstream.inference.re_worker}
E3_DOCUMENTS_SHA256 = "93f5eb972c4c19eac76f123a141f19a8f2d0bd5b8147e786e58a64c03c6110af"


def locked_rows(path):
    try:
        contents = path.read_bytes()
    except FileNotFoundError:
        return None
    if sha256(contents).hexdigest() != LOCKED[path]:
        raise ValueError(f"Historical artifact checksum mismatch: {path}")
    return [json.loads(line) for line in contents.splitlines()]


def checked_documents():
    manifest = upstream.read(upstream.FROZEN / "corpus_manifest.json")
    if upstream.sha(SOURCE) != LOCKED[SOURCE] or upstream.sha(upstream.CORPUS) != manifest["text_file_sha256"]:
        raise ValueError("Frozen Corpus30 or Corpus100 source checksum differs")
    incoming, former = read_documents(SOURCE), read_documents(upstream.CORPUS)
    if (len(incoming), len(former)) != (30, 100):
        raise ValueError("Expected 30 historical and 100 frozen documents")
    rules = runpy.run_path(str(upstream.FROZEN / "prepare.py"))
    pair_keys, find_duplicate = rules["pair_keys"], rules["find_duplicate"]
    old = [pair_keys(doc) for doc in incoming]
    for doc in former:
        if duplicate := find_duplicate(doc, pair_keys(doc), old):
            raise ValueError(f"Corpus100 document {doc['document_id']} overlaps Corpus30: {duplicate}")
    return incoming, former


def parent_for_append(current, former):
    folder = v2.OUTPUT / "generations" / current["request"]
    payload = v2.saved_payload(folder, current)
    if payload["expected"] != dict(nodes=BOOTSTRAP_NODES, relationships=BOOTSTRAP_EDGES):
        raise ValueError("Corpus30 requires the published 100-document V2 baseline")
    documents = {row["properties"]["document_id"]: row["properties"] for row in payload["nodes"].values()
                 if row["label"] == "RELEXDocument"}
    if ({key: upstream.source_identity(doc) for key, doc in documents.items()}
            != {doc["document_id"]: upstream.source_identity(doc) for doc in former}):
        raise ValueError("CURRENT does not contain the frozen Corpus100 document revisions")
    return payload["generation"]


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["document_id"]].append(row)
    return grouped


def historical_sources(documents, ner_threshold):
    mentions, predictions = locked_rows(NER), locked_rows(RE)
    ids = {doc["document_id"] for doc in documents}
    saved_re = {row["document_id"]: row for row in predictions or ()}
    if predictions is not None and (len(predictions) != len(ids) or saved_re.keys() != ids):
        raise ValueError("Historical RE document coverage differs")
    if mentions is None and predictions is not None:
        # Выгрузка RE также содержит полные исходные записи E3.
        mentions = [m for row in predictions for entity in row["entities"] for m in entity["source_mentions"]]
    if mentions is None:
        paths = {did: NER.parent / "documents" / f"{did}.json" for did in ids}
        if all(path.is_file() for path in paths.values()):
            contents = {did: path.read_bytes() for did, path in paths.items()}
            if upstream.signature({did: sha256(data).hexdigest() for did, data in contents.items()}) != (
                E3_DOCUMENTS_SHA256
            ):
                raise ValueError("Historical per-document E3 checkpoint checksums differ")
            mentions = [m for data in contents.values() for m in json.loads(data)["mentions"]
                        if m["score"] >= ner_threshold]
    saved_ner = group_rows(mentions or ())
    if mentions is not None and (len(mentions) != 1327 or saved_ner.keys() != ids):
        raise ValueError("Historical E3 mention/document coverage differs")
    return saved_ner, saved_re


def recover_ner(doc, mentions, config):
    return dict(mentions=[m | dict(model_identifier=config["ner"]["model"], contract_version=CONTRACT_VERSION)
                          for m in mentions], abbreviations=detect_abbreviations(doc["text"]))


def recover_re(doc, saved, ner, config):
    entities = upstream.inference.entities_for(doc, ner["mentions"])
    # В историческом формате у упоминаний ещё не было этих двух полей.
    original = [e | dict(source_mentions=[{k: v for k, v in m.items()
                                           if k not in {"model_identifier", "contract_version"}}
                                          for m in e["source_mentions"]]) for e in entities]
    if original != saved["entities"]:
        return None
    experiment = upstream.read(RE.with_name("experiment_config.json"))
    if (saved["fingerprint"] != experiment["fingerprint"] or saved["model"] != config["re"]["model"]
            or saved["mode"] != "e3"):
        raise ValueError("Historical RE recipe differs")
    return dict(entities=entities, **{key: saved[key] for key in ("raw_windows", "relations", "raw_relation_count")})


def init_inference(stage, config, paths):
    global infer
    infer = FACTORIES[stage](config, paths)


def infer_document(doc, dependency):
    return infer(doc, dependency)


def stage_documents(stage, documents, saved, dependencies, config, paths, work, workers, recipe):
    results, pending = {}, {}

    def verify(doc, entry):
        if entry["origin"] not in ("historical", "local"):
            raise ValueError("Unknown checkpoint origin")
        dependency = dependencies.get(doc["document_id"])
        upstream.validate_document_result(stage, doc, entry["result"],
                                          dependency["result"] if dependency is not None else None, config)

    def restore(doc):
        did = doc["document_id"]
        dependency = dependencies.get(did)
        path = work / "checkpoints" / stage / f"{did}.json"
        identity = dict(upstream.source_identity(doc), stage=stage, recipe=recipe,
                        upstream_sha256=upstream.signature(dependency))
        entry = upstream.load_checkpoint(path, identity, lambda value: verify(doc, value))
        if entry is not None:
            return doc, path, identity, entry, None
        start = perf_counter()
        result = None
        if did in saved:
            result = (recover_ner(doc, saved[did], config) if stage == "ner"
                      else recover_re(doc, saved[did], dependency["result"], config))
        if result is not None:
            entry = dict(origin="historical", result=result)
            verify(doc, entry)
        return doc, path, identity, entry, perf_counter() - start

    with tqdm(total=len(documents), desc=stage.upper(), unit="doc") as bar:
        # Потоки проверяют документы параллельно, checkpoint записывает только родитель.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(restore, doc) for doc in documents]):
                doc, path, identity, entry, seconds = future.result()
                did = doc["document_id"]
                if entry is None:
                    pending[did] = (doc, path, identity)
                    continue
                if seconds is not None:
                    upstream.checkpoint(path, identity, entry, seconds)
                results[did] = entry
                bar.update(1)
        if pending:
            # Число тяжёлых процессов ограничено бюджетом RESEARCH.
            count = min(workers, len(pending), 2)
            upstream.verify_models(config, paths, MODEL_FILES[stage])
            upstream.preflight(config["threads"], count)
            start = perf_counter()
            with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context("spawn"),
                                     initializer=init_inference, initargs=(stage, config, paths)) as pool:
                tasks = {pool.submit(infer_document, doc, dependencies[did]["result"] if did in dependencies else None):
                         (doc, path, identity) for did, (doc, path, identity) in pending.items()}
                for future in as_completed(tasks):
                    doc, path, identity = tasks[future]
                    entry = dict(origin="local", result=future.result())
                    verify(doc, entry)
                    upstream.checkpoint(path, identity, entry, perf_counter() - start)
                    results[doc["document_id"]] = entry
                    bar.update(1)
    return results


def historical_resolution(doc, relations, first, second, threshold):
    source = {e["entity_id"]: e for e in relations["entities"]}
    mentions = {m["mention_id"]: (eid, m) for eid, e in source.items() for m in e["source_mentions"]}
    by_id, seen = {}, set()
    for row in first:
        eid, mid = row["entity_id"], row["mention_id"]
        original, mention = mentions[mid]
        saved = row["source_mention"] | {key: mention[key] for key in ("contract_version", "model_identifier")}
        if (eid != original or mid in seen or saved != mention or
                eid in by_id and any(by_id[eid][key] != row[key] for key in ("canonical_entity_id", "resolution"))):
            raise ValueError("Invalid historical 1B mention mapping")
        seen.add(mid)
        by_id[eid] = row
    if seen != mentions.keys():
        raise ValueError("Historical 1B mapping does not cover source mentions")
    first_map = {key: row["canonical_entity_id"] for key, row in by_id.items()}
    after_threshold = {first_map[key] for key, e in source.items() if e["confidence"] >= threshold}
    second_map = {row["source_canonical_id"]: row["target_canonical_id"] for row in second}
    if (len(second_map) != len(second) or second_map.keys() != after_threshold
            or any(first_map.get(to) != to for to in first_map.values())
            or any(second_map.get(to) != to for to in second_map.values())):
        raise ValueError("Historical ER representatives or max_member coverage differ")
    mapping = {key: second_map[to] for key, to in first_map.items() if to in after_threshold}
    entities = defaultdict(list)
    for key, target in sorted(mapping.items()):
        entities[target].append(member_record(source[key], doc, by_id[key]["resolution"]))
    return dict(mapping=mapping, entities=dict(entities),
                relations=[r for r in relations["relations"] if r["head_id"] in mapping and r["tail_id"] in mapping])


def resolve_documents(documents, ner, relations, threshold, workers):
    first, second = locked_rows(ER1), locked_rows(ER2)
    history = first is not None and second is not None
    if history:
        first, second = group_rows(first), group_rows(second)
        if first.keys() != second.keys() or first.keys() != {doc["document_id"] for doc in documents}:
            raise ValueError("Historical ER document coverage differs")

    def resolve(doc):
        did = doc["document_id"]
        mentions, predictions = ner[did], relations[did]
        if history and mentions["origin"] == predictions["origin"] == "historical":
            return historical_resolution(doc, predictions["result"], first[did], second[did], threshold), True
        return upstream.graph.resolve_entities({did: doc}, {did: mentions["result"]},
                                               {did: predictions["result"]}, threshold), False

    combined = dict(entities={}, mapping={}, relations=[])
    reused = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result, historic in tqdm(pool.map(resolve, documents), total=len(documents), desc="ER", unit="doc"):
            for key in ("entities", "mapping"):
                if combined[key].keys() & result[key].keys():
                    raise ValueError(f"Duplicate ER keys: {key}")
                combined[key].update(result[key])
            combined["relations"].extend(result["relations"])
            reused += historic
    if reused == 30 and tuple(map(len, combined.values())) != (811, 993, 1490):
        raise ValueError("Historical ER entity, source span or source triple counts differ")
    return combined, reused


def prepare_delta(folder, base, documents, workers):
    delta = folder / "delta"
    manifest = delta / "manifest.json"
    if manifest.exists():
        if delta_identity(delta)["base_generation"] != base:
            raise ValueError("Prepared delta belongs to another parent")
        return delta
    for path in (NER.with_name("config.json"), RE.with_name("experiment_config.json")):
        if upstream.sha(path) != LOCKED[path]:
            raise ValueError(f"Historical model configuration changed: {path}")
    config = upstream.configure(1)
    paths = dict(ner=ROOT / "artifacts/ner_v2/models/gliner25",
                 re=ROOT / "artifacts/models/gliner-relex-base-v1.0",
                 re_common=ROOT / "artifacts/models/gliner-relex-large-v1.0")
    recipe = upstream.signature(dict(configuration=config, archives=[*LOCKED.values(), E3_DOCUMENTS_SHA256],
                                     implementation={p.name: upstream.sha(p) for p in (
                                         Path(__file__), Path(upstream.inference.__file__),
                                         Path(upstream.graph.__file__),
                                         Path(upstream.relex.__file__))}))
    saved_ner, saved_re = historical_sources(documents, config["ner"]["threshold"])
    ner = stage_documents("ner", documents, saved_ner, {}, config, paths, folder, workers, recipe)
    relations = stage_documents("re", documents, saved_re, ner, config, paths, folder, workers, recipe)
    delta.mkdir(parents=True, exist_ok=True)
    er_path = delta / "entity_resolution.json"
    identity = dict(stage="entity_resolution", recipe=recipe, base_generation=base,
                    upstream_sha256=upstream.signature([ner, relations]))
    resolution = upstream.load_checkpoint(er_path, identity, lambda value: None)
    if resolution is None:
        start = perf_counter()
        resolution, reused = resolve_documents(documents, ner, relations, config["er"]["threshold"], workers)
        upstream.graph.build_payload(v2.NAMESPACE, recipe, {doc["document_id"]: doc for doc in documents},
                                     resolution, (), dict(concepts=[], memberships=[]))
        upstream.checkpoint(er_path, identity | {"reused_er_documents": reused}, resolution, perf_counter() - start)
    write_jsonl(delta / "documents.jsonl", documents)
    write_checked(manifest, dict(base_generation=base, upstream_recipe=recipe,
                                 frozen_historical_sha256={path.relative_to(ROOT).as_posix(): checksum
                                                           for path, checksum in LOCKED.items()},
                                 files_sha256={name: upstream.sha(delta / name)
                                               for name in ("documents.jsonl", "entity_resolution.json")}))
    return delta


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, help="Параллельные воркеры документов (по умолчанию 2)")
    parser.add_argument("--resume", action="store_true", help="Продолжить ранее начатую подготовку или публикацию")
    parser.add_argument("--e5-model", type=Path, default=E5)
    parser.add_argument("--neo4j-env", type=Path)
    parser.add_argument("--allow-shared-neo4j", action="store_true")
    resource_options(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 30:
        parser.error("--workers должен быть целым числом от 1 до 30")
    options = limits(args)
    config = v2.approved_config()
    if args.resume and not (WORK / "parent.json").exists():
        parser.error("--resume требует начатую подготовку Corpus30")
    print("ПРЕДУПРЕЖДЕНИЕ: ВЕКТОРИЗАЦИЯ E5 И ПОПОЛНЕНИЕ ГРАФА NEO4J МОГУТ ЗАНЯТЬ ПРОДОЛЖИТЕЛЬНОЕ ВРЕМЯ.",
          flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with FileLock(WORK / ".lock", timeout=0):
        current_path = v2.OUTPUT / "CURRENT.json"
        if not current_path.exists():
            raise ValueError("Publish the Corpus100 V2 baseline before appending Corpus30")
        current = load_checked(current_path)
        target = neo4j.Target(**current["target"], allow_shared_neo4j=args.allow_shared_neo4j)
        if target.record() == dict(uri=neo4j.SHARED_URI, database=neo4j.SHARED_DATABASE,
                                   container=neo4j.SHARED_CONTAINER) and not args.allow_shared_neo4j:
            parser.error("Текущий граф в общей Neo4j требует --allow-shared-neo4j")
        marker = WORK / "parent.json"
        delta = WORK / "delta"
        if (delta / "manifest.json").exists():
            if delta_identity(delta)["base_generation"] != load_checked(marker)["base_generation"]:
                raise ValueError("Prepared delta belongs to another parent")
        else:
            documents, former = checked_documents()
            parent = dict(base_generation=parent_for_append(current, former), target=target.record())
            if marker.exists() and load_checked(marker) != parent:
                raise ValueError("The pinned Corpus100 parent or Neo4j target has changed")
            if not marker.exists():
                write_checked(marker, parent)
            delta = prepare_delta(WORK, parent["base_generation"], documents, args.workers)
        if load_checked(marker)["target"] != target.record():
            raise ValueError("The pinned Neo4j target has changed")
        with neo4j.connect(target, args.neo4j_env) as driver:
            result = v2.run_graph(driver, target, v2.OUTPUT, config, options,
                                  delta=delta, model=args.e5_model.resolve())
        counts = result["counts"]
        if counts["nodes"]["RELEXDocument"] != 130 or counts["relationships"].get("SIMILAR_TO", 0):
            raise ValueError("Expected 130 documents and zero SIMILAR_TO after append")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
