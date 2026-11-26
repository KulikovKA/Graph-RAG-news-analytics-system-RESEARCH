"""Построение и инкрементальное пополнение файлового каталога Shared Concepts V2."""

from argparse import ArgumentParser
from decimal import Decimal
import json
from pathlib import Path
import re
from time import perf_counter

from filelock import FileLock

from build_corpus_graph import read, sha
from cgraph.ner_v2.io import read_documents
from cgraph.shared_concepts_v2 import Config, digest
from relex_experiment import MemorySampler
from shared_concepts_v2_pipeline import (
    FROZEN_MANIFEST, PREVIOUS_RUN, ROOT, ResourceBudget, frozen_corpus, load_checked, read_entity_bundle,
    resolved_profiles, run_catalog, vector_recipe, vectorize, write_checked, write_entity_bundle,
)

OUTPUT = ROOT / "artifacts/shared_concepts_v2"


def read_configuration(path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)


def slug(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value):
        raise ValueError("Expected an ASCII identifier using letters, digits, underscores and hyphens")
    return value


def resource_options(parser):
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-rss-mib", type=int, default=2048)
    parser.add_argument("--checkpoint-every", type=int, default=100)


def limits(args):
    options = dict(threads=args.threads, max_seconds=args.max_seconds, max_rss=args.max_rss_mib * 1024**2,
                   checkpoint_every=args.checkpoint_every)
    if min(options.values()) <= 0:
        raise ValueError("Resource limits must be positive")
    return options


def input_bundle(args, output):
    base_folder = OUTPUT / slug(args.base) if args.base else None
    if base_folder == output:
        raise ValueError("Use a new run ID for the next generation")
    base = load_checked(base_folder / "catalog.json") if base_folder else None
    if base and base["identity"]["input"].get("corpus") == "Corpus100":
        raise ValueError("Corpus100 cannot be used to seed an independent catalog")
    profile_source = args.profiles or args.resolution
    if bool(args.documents) != bool(profile_source) or bool(args.model) != bool(profile_source):
        raise ValueError("Provide profiles or ER checkpoint together with documents and the pinned local model")
    identity = dict(canonical_input=sha(profile_source) if profile_source else None,
                    format="profiles" if args.profiles else "entity_resolution" if args.resolution else None,
                    documents=sha(args.documents) if args.documents else None,
                    base=base["generation"] if base else None)
    frozen = read(FROZEN_MANIFEST)["identity"]
    if identity["documents"] == frozen["inputs_sha256"]["data/retrieval/benchmark_3a5_11/corpus100/abstracts.jsonl"]:
        raise ValueError("Use the isolated --corpus100 entry point for the frozen Corpus100 documents")
    marker, bundle = output / "input.json", output / "entities.json"
    if marker.exists():
        checkpoint = load_checked(marker)
        if checkpoint["source"] != identity or checkpoint["bundle_hash"] != sha(bundle):
            raise ValueError("Input bundle differs from the existing checkpoint")
        return read_entity_bundle(bundle), checkpoint["metadata"], base
    entities = read_entity_bundle(base_folder / "entities.json") if base else []
    if profile_source:
        docs = {row["document_id"]: row for row in read_documents(args.documents)}
        profiles = read(args.profiles) if args.profiles else resolved_profiles(docs, args.resolution)
        started = perf_counter()
        with MemorySampler() as sampler:
            budget = ResourceBudget(sampler, started, args.max_seconds, args.max_rss_mib * 1024**2)
            incoming, metadata = vectorize(profiles, docs, args.model, OUTPUT / "vector_cache", args.threads,
                                           budget)
        existing = {e.entity_id: e for e in entities}
        if len(incoming) != len({e.entity_id for e in incoming}):
            raise ValueError("Duplicate canonical entity ID in input")
        for entity in incoming:
            old = existing.get(entity.entity_id)
            if old is not None and old.fingerprint != entity.fingerprint:
                raise ValueError("Existing profile changed; use a new generation rebuilt from source")
            existing[entity.entity_id] = entity
        entities = list(existing.values())
    else:
        if base:
            metadata = dict(recipe=base["identity"]["input"]["recipe"])
        else:
            metadata = dict(recipe=vector_recipe(frozen["model_files_sha256"]))
    metadata["input_hashes"] = identity
    write_entity_bundle(bundle, entities)
    write_checked(marker, dict(source=identity, metadata=metadata, bundle_hash=sha(bundle)))
    return entities, metadata, base


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON с ec, cc и delta")
    parser.add_argument("--run-id", type=slug, required=True)
    parser.add_argument("--corpus100", action="store_true")
    parser.add_argument("--frozen-run", type=Path, default=PREVIOUS_RUN)
    parser.add_argument("--namespace", type=slug, help="Пространство независимого файлового каталога")
    parser.add_argument("--base", type=slug)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--profiles", type=Path)
    source.add_argument("--resolution", type=Path, help="Проверенный чекпойнт entity_resolution")
    parser.add_argument("--documents", type=Path)
    parser.add_argument("--model", type=Path)
    resource_options(parser)
    args = parser.parse_args()
    options = limits(args)
    config = Config.from_dict(read_configuration(args.config))
    output = OUTPUT / args.run_id
    if args.corpus100 and any((args.base, args.namespace, args.profiles, args.resolution, args.documents, args.model)):
        parser.error("--corpus100 is an isolated research run")
    if not args.corpus100 and not args.namespace:
        parser.error("--namespace is required for an independent catalog")
    print("ПРЕДУПРЕЖДЕНИЕ: SHARED CONCEPTS V2 МОЖЕТ ЗАНЯТЬ НЕСКОЛЬКО МИНУТ.", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".input.lock", timeout=0):
        if args.corpus100:
            entities, metadata = frozen_corpus(args.frozen_run)
            base = None
            namespace = f"research:corpus100:{args.run_id}"
        else:
            if (output / "catalog.json").exists() and not (output / "input.json").exists():
                raise ValueError("Run ID already reserved for another input")
            entities, metadata, base = input_bundle(args, output)
            namespace = f"local:{args.namespace}"
        result = run_catalog(entities, config, namespace, output, metadata=metadata, base=base,
                             regression=args.corpus100, **options)
    print(f"{output}: {result['statistics']['concepts']} concepts, {digest(result['state'])}")


if __name__ == "__main__":
    main()
