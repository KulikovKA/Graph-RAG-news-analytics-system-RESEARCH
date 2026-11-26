"""Входы штатного графа V2 после зафиксированных стадий NER, RE и ER."""

from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

import benchmark_contextual_shared_concepts as historical
import build_corpus_graph as old
import build_relex_graphs as relex
from cgraph.ner_v2.io import read_documents
from cgraph.shared_concepts_v2 import Config, EE, MODEL, WINDOW, digest
from relex_experiment import MemorySampler
from run_shared_concepts_v2 import read_configuration
from shared_concepts_v2_pipeline import (
    EXPECTED_EE, PREVIOUS_RUN, RESEARCH_GRAPH_NAMESPACE, REVISION, ROOT, ResourceBudget, check_windows,
    frozen_corpus, load_checked, read_entity_bundle, resolved_profiles, vectorize,
)

APPROVED = ROOT / "config/shared_concepts_v2_approved.json"
REFERENCE = ROOT / "reports/retrieval/shared_concepts_v2_study/corpus100/results.json"
BOOTSTRAP_NODES = {"RELEXDocument": 100, "RELEXEntity": 4506, "RELEXConcept": 575}
BOOTSTRAP_EDGES = {"RELEX_MENTIONS": 4506, "RELEX_REL": 8498, "HAS_SHARED_CONCEPT": 4938}
BOOTSTRAP_MEMBERS_SHA256 = "b5929794303a94b26ade70903d6e87a08be8c36fc465b0736f1265dd08105bbf"
DELTA_FILES = {"documents.jsonl", "entity_resolution.json"}


@dataclass
class Inputs:
    source: dict
    entities: list
    metadata: dict


def approved_config(path=APPROVED):
    raw = read_configuration(path)
    if ((raw.pop("embedding_model"), raw.pop("embedding_revision"), raw.pop("window")) != (MODEL, REVISION, WINDOW)
            or tuple(map(float, (raw["ee"]["name"], raw["ee"]["context"]))) != EE
            or set(raw.pop("ee")) != {"name", "context"}):
        raise ValueError("Embedding recipe, window and EE must match approved graph V2")
    config = Config.from_dict(raw)
    if config.to_dict() != dict(ec={"name": .91, "context": .87}, cc={"name": .93, "context": .90}, delta="1/20"):
        raise ValueError("Graph V2 requires the approved EC, CC and delta")
    return config


def ready_frozen_checkpoints(documents, manifest):
    """Воспроизводит проверку идентичности checkpoint штатного Run без создания Run и inference."""
    config = manifest["configuration"]
    ner_implementation = [inspect.getsource(f) for f in (
        old.inference.ner_worker, old.inference.e3_windows, old.inference.e3_source_span)]
    re_implementation = [inspect.getsource(old.inference.re_worker), old.sha(Path(relex.__file__)),
                         inspect.getsource(old.inference.entities_for)]
    root = historical.PRIOR / "checkpoints"
    for doc_id, doc in tqdm(documents.items(), desc="Verify frozen NER / RE", unit="doc"):
        def ready(stage, dependency, models, code):
            identity = dict(old.source_identity(doc), stage=stage, config_fingerprint=manifest["fingerprint"],
                            upstream_sha256=old.signature(dependency),
                            model_fingerprint=old.signature({k: config["model_files"][k] for k in models}),
                            implementation_sha256=old.signature(code))
            path = root / stage / f"{doc_id}.json"
            verify = lambda results: old.validate_document_result(stage, doc, results, dependency, config)
            saved = old.load_checkpoint(path, identity, verify)
            if saved is None and stage == "ner":
                saved = old.load_checkpoint(path, identity | {"implementation_sha256": old.E3_PRE_PUNCTUATION_FIX},
                                            verify)
            if saved is None:
                raise ValueError(f"Missing or invalid frozen {stage} checkpoint for {doc_id}")
            return saved

        ner = ready("ner", None, ("ner",), ner_implementation)
        ready("re", ner, ("re", "re_common"), re_implementation)


def frozen_inputs(config, frozen_run=PREVIOUS_RUN):
    entities, metadata = frozen_corpus(frozen_run, graph_inputs_only=True)
    manifest = old.read(historical.PRIOR / "manifest.json")
    if old.signature(manifest["configuration"]) != manifest["fingerprint"]:
        raise ValueError("Invalid historical Corpus100 manifest")
    documents = {row["document_id"]: row for row in read_documents(old.CORPUS)}
    ready_frozen_checkpoints(documents, manifest)
    resolution_path = historical.PRIOR / "checkpoints/entity_resolution/corpus.json"
    resolution = historical.frozen_checkpoint(resolution_path, manifest, "entity_resolution")
    verify = load_checked(REFERENCE)
    parameters = config.to_dict()
    matching = [row for row in verify["results"] if row["config"] == parameters]
    actual = (len(documents), len(entities), len(resolution["entities"]), len(resolution["relations"]))
    expected = (100, 4506, 4506, 8498)
    if (actual != expected or verify["ee"]["guarded_pairs"] != EXPECTED_EE["guarded_pairs"]
            or len(matching) != 1
            or (matching[0]["metrics"]["concepts"], matching[0]["metrics"]["memberships"],
                matching[0]["metrics"]["catalog_members_sha256"]) != (575, 4938, BOOTSTRAP_MEMBERS_SHA256)):
        raise ValueError(f"Frozen inputs or approved study reference differ: {actual}")
    return Inputs(dict(documents=documents, resolution=resolution), entities,
                  metadata | dict(research_graph=RESEARCH_GRAPH_NAMESPACE, reference_sha256=old.sha(REFERENCE)))


def delta_identity(folder):
    manifest = load_checked(folder / "manifest.json")
    actual = {"manifest.json": old.sha(folder / "manifest.json")}
    # Запись в старом манифесте допустима, содержимое semantic.json графу больше не требуется.
    files = {name: checksum for name, checksum in manifest["files_sha256"].items() if name != "semantic.json"}
    if set(files) not in (DELTA_FILES, DELTA_FILES | {"entities.json"}):
        raise ValueError("Delta manifest must pin documents, ER and optional V2 vectors")
    for name, expected in files.items():
        checksum = old.sha(folder / name)
        if checksum != expected:
            raise ValueError(f"Delta file checksum mismatch: {name}")
        actual[name] = checksum
    return dict(base_generation=manifest["base_generation"], input_sha256=actual)


def delta_inputs(folder, identity, previous, base, model, cache, threads, max_seconds, max_rss):
    if identity["base_generation"] != base["generation"]:
        raise ValueError("Delta was prepared for a different parent generation")
    docs = {row["document_id"]: row for row in read_documents(folder/"documents.jsonl")}
    former = previous.source
    if former["documents"].keys() & docs.keys():
        raise ValueError("Only new document IDs can be appended")
    resolution = old.load_checkpoint(folder/"entity_resolution.json", {"stage": "entity_resolution"}, lambda _: None)
    if not isinstance(resolution, dict):
        raise ValueError("Invalid upstream ER checkpoint checksum")
    profiles = resolved_profiles(docs, folder/"entity_resolution.json")
    old_er = former["resolution"]
    for key in ("entities", "mapping"):
        if old_er[key].keys() & resolution[key].keys():
            raise ValueError(f"Delta changes existing ER {key}")
    all_docs = former["documents"] | docs
    merged = dict(entities=old_er["entities"] | resolution["entities"],
                  mapping=old_er["mapping"] | resolution["mapping"],
                  relations=old_er["relations"] + resolution["relations"])
    recipe = previous.metadata["recipe"]
    if "entities.json" in identity["input_sha256"]:
        vectors = sorted(read_entity_bundle(folder/"entities.json"), key=lambda e: e.entity_id)
        ordered = sorted(profiles, key=lambda p: p["entity_id"])
        if [e.profile for e in vectors] != ordered or any(e.recipe_hash != digest(recipe) for e in vectors):
            raise ValueError("Delta V2 vector provenance or pinned recipe differs")
        check_windows(ordered, [dict(entity_id=e.entity_id, contexts=json.loads(e.windows_json))
                                for e in vectors], docs)
    else:
        if model is None:
            raise ValueError("Provide precomputed entities.json or --e5-model for new documents")
        with MemorySampler() as sampler:
            budget = ResourceBudget(sampler, perf_counter(), max_seconds, max_rss)
            vectors, metadata = vectorize(profiles, docs, model, cache, threads, budget)
        if metadata["recipe"] != recipe:
            raise ValueError("Delta E5 recipe differs from the pinned base recipe")
    if delta_identity(folder) != identity:
        raise ValueError("Delta changed while preparing inputs")
    return Inputs(dict(documents=all_docs, resolution=merged),
                  previous.entities + vectors,
                  dict(recipe=recipe, research_graph=RESEARCH_GRAPH_NAMESPACE, delta=identity))
