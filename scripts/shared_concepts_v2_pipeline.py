"""Файловый пайплайн Shared Concepts V2 и адаптеры входных артефактов."""

from collections import Counter
from contextlib import closing
import csv
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from itertools import chain
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
from time import perf_counter

from filelock import FileLock
import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import benchmark_contextual_shared_concepts as previous
import contextual_shared_concepts as contextual
from build_corpus_graph import read, sha, signature
from cgraph.ner_v2.io import validate_span
from cgraph.shared_concepts_v2 import (
    Catalog, DIMENSION, EE, Entity, MODEL, WINDOW, aligned_cosine, canonical, digest, nesting_pairs, unit,
)
from relex_experiment import MemorySampler

ROOT = previous.ROOT
REVISION = previous.REVISION
PREVIOUS_RUN = previous.OUTPUT / "corpus100"
RESEARCH_GRAPH_NAMESPACE = "relex_corpus100_shared_concepts_v2"
RESEARCH_CATALOG_NAMESPACE = f"research:corpus100:{RESEARCH_GRAPH_NAMESPACE}"
FROZEN_MANIFEST = ROOT / "reports/retrieval/benchmark_3a5_13/MANIFEST.json"
EXPECTED_EE = dict(entities=4506, cross_document_pairs=10042851, raw_pairs=8173, guarded_pairs=5655,
                   entities_with_neighbors=1605, documents_with_neighbors=100)
RECIPE = previous.RECIPE | dict(normalization="individual_l2", context_aggregation="mean_float64",
                                dtype="float32", device="cpu", context_template="{name}. {window}")


class ResourceBudget:
    def __init__(self, sampler, started, max_seconds, max_rss):
        self.sampler = sampler
        self.started = started
        self.max_seconds = max_seconds
        self.max_rss = max_rss

    @property
    def peak(self):
        peak = self.sampler.peak
        if perf_counter() - self.started > self.max_seconds or peak > self.max_rss:
            raise RuntimeError("Shared Concepts V2 resource budget reached; resume from checkpoint")
        return peak

    def check(self):
        return self.peak


def enrich_profiles(rows, documents, resolution):
    aliases = {p.entity_id: p.aliases for p in previous.profiles(documents, resolution)}
    return [row | dict(aliases=list(aliases[row["entity_id"]]),
                       confidence=max(m["source_entity"]["confidence"]
                                      for m in resolution["entities"][row["entity_id"]]))
            for row in rows]


def resolved_profiles(documents, checkpoint):
    docs = {key: row | dict(topic=row.get("topic", row["discipline"])) for key, row in documents.items()}
    resolution = previous.frozen.load_checkpoint(checkpoint, {"stage": "entity_resolution"}, lambda _: None)
    if resolution is None:
        raise ValueError("Invalid ER checkpoint")
    return enrich_profiles(previous.checked_entities(docs, resolution), docs, resolution)


def vector_recipe(model_files):
    return dict(
        model=MODEL, revision=REVISION, tokenizer_revision=REVISION, window=WINDOW,
        embedding=RECIPE, model_files_sha256=model_files, seed=contextual.SEED,
        code_sha256={name: sha(Path(__file__).with_name(name)) for name in
                     ("contextual_shared_concepts.py", "corpus_graph_inference.py")} |
                    {"semantic_concepts.py": sha(ROOT / "src/cgraph/semantic_concepts.py")},
        versions={pkg: version(pkg) for pkg in ("numpy", "torch", "tokenizers", "transformers")},
    )


def check_windows(profiles, windows, documents):
    geometry = {}
    for profile, row in zip(profiles, windows, strict=True):
        if row["entity_id"] != profile["entity_id"]:
            raise ValueError("Window rows are out of profile order")
        doc = documents[profile["document_id"]]
        if doc["document_id"] not in geometry:
            geometry[doc["document_id"]] = contextual.DocumentWindows(doc["text"])
        mentions = set()
        for m in profile["evidence"]:
            validate_span(doc["text"], m["mention_start"], m["mention_end"], m["mention_text"])
            if (m["document_id"], m["revision_id"], m["source_url"]) != (
                doc["document_id"], doc["revision_id"], doc["source_url"]
            ):
                raise ValueError("Source revision mismatch")
            mentions.add((m["source_entity_id"], tuple(m["source_mention_ids"]),
                          m["mention_start"], m["mention_end"], m["mention_text"]))
        if not row["contexts"] or len({c["text"] for c in row["contexts"]}) != len(row["contexts"]):
            raise ValueError("Expected distinct nonempty context windows")
        observed = set()
        for context in row["contexts"]:
            if context["input"] != profile["name"] + ". " + context["text"] or not context["provenance"]:
                raise ValueError("Context input or provenance mismatch")
            for ref in context["provenance"]:
                if any(ref[k] != doc[k] for k in ("document_id", "revision_id", "source_url", "text_sha256")):
                    raise ValueError("Window document identity mismatch")
                begin, end = ref["input_start"], ref["input_end"]
                start, stop = ref["mention_start"], ref["mention_end"]
                if not 0 <= ref["window_start"] <= begin <= start < stop <= end <= ref["window_end"] <= len(doc["text"]):
                    raise ValueError("Window does not contain the complete mention")
                if (ref["window_start"], ref["window_end"]) != geometry[doc["document_id"]].window(WINDOW, start, stop):
                    raise ValueError("Context window does not match the S1 geometry")
                validate_span(doc["text"], begin, end, context["text"])
                observed.add((ref["source_entity_id"], tuple(ref["source_mention_ids"]),
                              start, stop, ref["mention_text"]))
        if mentions != observed:
            raise ValueError("Source mentions and window evidence differ")


def frozen_corpus(directory=PREVIOUS_RUN, *, graph_inputs_only=False):
    """Проверяет цепочку доверия 3A.5.11 → 3A.5.12 → 3A.5.13."""
    manifest = read(FROZEN_MANIFEST)
    identity = read(directory / "identity.json")
    locked = manifest["identity"]
    expected_inputs = {p.replace("\\", "/"): h for p, h in locked["inputs_sha256"].items()}
    if (manifest["stage"] != "3A.5.13" or manifest["status"] != "complete"
            or manifest["fingerprint"] != signature(locked) or identity["stage"] != "3A.5.12"
            or identity["smoke"] or locked["smoke"] or locked["force_recompute"]
            or identity["windows"].get(WINDOW) != 1 or locked["windows"].get(WINDOW) != 1
            or tuple(locked["rules"]) != contextual.RULES
            or {p.replace("\\", "/"): h for p, h in identity["inputs_sha256"].items()} != expected_inputs
            or any(identity[k] != locked[k] for k in ("seed", "model_revision", "model_files_sha256",
                                                     "entity_ids_sha256", "pair_cosine"))
            or locked["model_revision"] != REVISION or locked["seed"] != contextual.SEED
            or locked["embedding"] != RECIPE
            or {k: v for k, v in identity["embedding"].items() if k not in {"threads", "workers"}} != RECIPE):
        raise ValueError("Incompatible Corpus100 embedding or input identity")
    used_inputs = expected_inputs
    if graph_inputs_only:
        # Исторические манифесты остаются неизменными, но для графа нужны только исходники и ER.
        used_paths = (
            previous.frozen.CORPUS, previous.frozen.FROZEN / "document_lock.json",
            previous.PRIOR / "manifest.json", previous.PRIOR / "checkpoints/entity_resolution/corpus.json",
        )
        used_inputs = {p.relative_to(ROOT).as_posix(): expected_inputs[p.relative_to(ROOT).as_posix()]
                       for p in used_paths}
    for path, value in used_inputs.items():
        if sha(ROOT / path) != value:
            raise ValueError(f"Frozen input checksum mismatch: {path}")
    recipe = vector_recipe(locked["model_files_sha256"])
    for name, checksum in recipe["code_sha256"].items():
        if locked["code_sha256"][name] != checksum or identity["code_sha256"][name] != checksum:
            raise ValueError(f"Frozen vector recipe changed: {name}")
    prior = signature(identity)
    files = {}
    for label, dtype in (("name", np.float32), (WINDOW, np.float64)):
        source = manifest["vector_and_score_artifacts"][label]
        filename = f"{label}_vectors.npy"
        if (source["source"] != previous.STAGE or source["fingerprint"] != prior
                or sha(directory / filename) != source["files"][filename]):
            raise ValueError(f"Frozen vector checksum mismatch: {label}")
        files[filename] = source["files"][filename]
        array = np.load(directory / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != (EXPECTED_EE["entities"], DIMENSION) or array.dtype != dtype:
            raise ValueError(f"Unexpected vector layout: {label}")
    windows_name = f"{WINDOW}_windows.json"
    extraction = read(directory / f"{WINDOW}_extract.json")
    checksum = extraction.pop("checksum")
    if (checksum != signature(extraction) or extraction["fingerprint"] != prior
            or sha(directory / windows_name) != extraction["files"][windows_name]):
        raise ValueError("Frozen window evidence checksum mismatch")
    files[windows_name] = extraction["files"][windows_name]
    docs = {d["document_id"]: d for d in previous.read_documents(previous.frozen.CORPUS)}
    resolution = previous.frozen_checkpoint(previous.PRIOR / "checkpoints/entity_resolution/corpus.json",
                                            read(previous.PRIOR / "manifest.json"), "entity_resolution")
    profiles = previous.checked_entities(docs, resolution)
    if (profiles != read(directory / "entities.json") or
            signature([p["entity_id"] for p in profiles]) != locked["entity_ids_sha256"]):
        raise ValueError("Frozen profile content or order mismatch")
    windows = read(directory / windows_name)
    check_windows(profiles, windows, docs)
    profiles = enrich_profiles(profiles, docs, resolution)
    names, contexts = (np.load(directory / f"{label}_vectors.npy", mmap_mode="r", allow_pickle=False)
                       for label in ("name", WINDOW))
    recipe_hash = digest(recipe)
    entities = [Entity.create(profile, row["contexts"], (n, c), recipe_hash)
                for profile, row, n, c in zip(profiles, windows, names, contexts, strict=True)]
    return entities, dict(recipe=recipe, corpus="Corpus100", input_hashes=used_inputs, vector_hashes=files,
                          prior_fingerprint=prior, embedding_source=previous.STAGE,
                          frozen_manifest_sha256=sha(FROZEN_MANIFEST), inference_seconds=0,
                          reused_entity_vectors=len(entities), inference_versions=identity["versions"])


def vectorize(profiles, documents, model, cache_dir, threads, budget):
    """Принимает профили действующего ER с исходными документами и offsets."""
    from tokenizers import Tokenizer

    locked = read(FROZEN_MANIFEST)["identity"]["model_files_sha256"]
    if any(sha(model / path) != value for path, value in locked.items()):
        raise ValueError("Local E5 model/tokenizer do not match the pinned revision")
    recipe = vector_recipe(locked)
    recipe_hash = digest(recipe)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    for profile in profiles:
        budget.check()
        doc = documents[profile["document_id"]]
        if not isinstance(doc["revision_id"], str) or not doc["revision_id"]:
            raise ValueError("Source document revision is required")
        if ("document_ids" in profile and set(profile["document_ids"]) != {doc["document_id"]}):
            raise ValueError("Multi-document ER profiles require an explicit policy")
        for m in profile["evidence"]:
            if (m["document_id"], m["revision_id"]) != (doc["document_id"], doc["revision_id"]):
                raise ValueError("Source mention revision mismatch")
            validate_span(doc["text"], m["mention_start"], m["mention_end"], m["mention_text"])
        if len(tokenizer.encode(contextual.PREFIX + profile["name"]).ids) > contextual.MAX_TOKENS:
            raise ValueError("Canonical name exceeds E5 token limit")
    geometry = {key: contextual.DocumentWindows(doc["text"]) for key, doc in documents.items()}
    windows, truncated = contextual.extract_windows(profiles, documents, geometry, tokenizer, WINDOW)
    check_windows(profiles, windows, documents)
    inputs = dict.fromkeys(chain((p["name"] for p in profiles),
                                (c["input"] for row in windows for c in row["contexts"])))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{recipe_hash}.sqlite3"
    # Одна модель и транзакционный кэш точных входов для обоих каналов.
    with FileLock(path.with_suffix(".lock"), timeout=0), closing(sqlite3.connect(path)) as db:
        cache = previous.EmbeddingCache(db, model, threads, budget, budget.started)
        scope = f"{WINDOW}:{digest(sorted(inputs))}"
        already_encoded, = db.execute("SELECT count(*) FROM vectors WHERE source_window = ?", (scope,)).fetchone()
        already_seconds, = db.execute(
            "SELECT coalesce(sum(seconds), 0) FROM batches WHERE source_window = ?", (scope,)
        ).fetchone()
        values, cumulative = cache.run(inputs, tokenizer, scope)
        encoded = cumulative["encoded"] - already_encoded
        usage = dict(unique_inputs=len(inputs), cache_hits=len(inputs) - encoded, encoded=encoded,
                     inference_seconds=cumulative["inference_seconds"] - already_seconds)
        if any(v.shape != (DIMENSION,) or not np.isclose(np.linalg.norm(v), 1., atol=2e-6, rtol=0)
               for v in values.values()):
            raise ValueError("Invalid individually normalized cache vector")
    entities = [Entity.create(p, row["contexts"],
                              (values[p["name"]], contextual.mean_embeddings(
                                  [values[c["input"]] for c in row["contexts"]])), recipe_hash)
                for p, row in zip(profiles, windows, strict=True)]
    budget.check()
    return entities, dict(recipe=recipe, truncated_windows=truncated, cache=usage)


def ee_edges(entities, incoming, check=lambda: None, block=128):
    """Точный EE для новых строк, с повторным использованием старой adjacency."""
    ordered = sorted(entities)
    if not incoming:
        return [], dict(entities=len(ordered), cross_document_pairs=0, raw_pairs=0, guarded_pairs=0,
                        entities_with_neighbors=0, documents_with_neighbors=0, guard_hits={},
                        name_pass=0, context_pass=0, documents_with_raw_neighbors=0,
                        entities_with_raw_neighbors=0, raw_document_pairs=0, guarded_document_pairs=0)
    profiles = [entities[key].profile for key in ordered]
    docs = np.asarray([row["document_id"] for row in profiles])
    vectors = unit([entities[key].vectors for key in ordered])
    indices = np.arange(len(ordered))
    is_new = np.asarray([key in incoming for key in ordered])
    rows = indices[is_new]
    counts = Counter(cross_document_pairs=0, raw_pairs=0, name_pass=0, context_pass=0)
    hits = Counter()
    raw_documents, raw_endpoints = set(), set()
    edges = []
    for start in tqdm(range(0, len(rows), block), desc="EE S1", unit="block"):
        check()
        i = rows[start:start + block]
        allowed = (docs[i, None] != docs) & ((i[:, None] < indices) | ~is_new)
        name = np.clip(vectors[i, 0] @ vectors[:, 0].T, -1, 1)
        context = np.clip(vectors[i, 1] @ vectors[:, 1].T, -1, 1)
        passed_name = allowed & (name >= EE[0])
        passed_context = allowed & (context >= EE[1])
        left, right = np.nonzero(passed_name & passed_context)
        source = i[left]
        raw_endpoints.update(source.tolist())
        raw_endpoints.update(right.tolist())
        raw_documents.update(tuple(sorted((a, b))) for a, b in zip(docs[source].tolist(), docs[right].tolist(), strict=True))
        guards = contextual.pair_guard_codes(profiles, source, right, np.ones(len(left), dtype=bool))
        counts.update(cross_document_pairs=int(allowed.sum()), raw_pairs=len(left),
                      name_pass=int(passed_name.sum()), context_pass=int(passed_context.sum()))
        for n, rule in enumerate(contextual.RULES):
            hits[rule] += int(np.count_nonzero(guards & (1 << n)))
        for l, r in zip(left[guards == 0], right[guards == 0], strict=True):
            a, b = sorted((ordered[i[l]], ordered[r]))
            edges.append((a, b, float(name[l, r]), float(context[l, r])))
    endpoints = {key for a, b, _, _ in edges for key in (a, b)}
    guarded_documents = {tuple(sorted((entities[a].document_id, entities[b].document_id))) for a, b, _, _ in edges}
    return sorted(edges), dict(counts, entities=len(ordered), guarded_pairs=len(edges),
                               entities_with_neighbors=len(endpoints),
                               documents_with_neighbors=len({entities[key].document_id for key in endpoints}),
                               documents_with_raw_neighbors=len({docs[i] for i in raw_endpoints}),
                               entities_with_raw_neighbors=len(raw_endpoints),
                               raw_document_pairs=len(raw_documents), guarded_document_pairs=len(guarded_documents),
                               guard_hits=dict(hits))


def summaries(catalog):
    concepts = list(catalog.concepts.values())
    counts = Counter(key for c in concepts for key in c.members)
    topics = {}
    for key in sorted(counts):
        entity = catalog.entities[key]
        group = topics.setdefault(entity.profile.get("topic", ""), dict(entities=0, documents=set(), memberships=0))
        group["entities"] += 1
        group["documents"].add(entity.document_id)
        group["memberships"] += counts[key]
    event_stats = {}
    for operation in ("initialize", "merge", "append"):
        events = [e for e in catalog.events if e["operation"] == operation]
        accepted = [e for e in events if e["reason"] == "PASS"]
        rounds = [len(e["rounds"]) for e in events if "rounds" in e]
        event_stats[operation] = dict(attempts=len(events), outcomes=dict(Counter(e["reason"] for e in events)),
                                      accepted_exclusions=sum(len(e.get("excluded", ())) for e in accepted),
                                      excluded_per_accept=dict(Counter(str(len(e.get("excluded", ()))) for e in accepted)),
                                      accepted_sizes=dict(Counter(str(len(e["members"])) for e in accepted)),
                                      ec_rounds=dict(min=min(rounds), max=max(rounds), mean=float(np.mean(rounds)),
                                                     median=float(np.median(rounds))) if rounds else {})
    doc_pairs = {(a, b) for c in concepts for i, a in enumerate(c.documents) for b in c.documents[i + 1:]}
    unseen = set(catalog.entities)
    components = Counter()
    while unseen:
        todo = [unseen.pop()]
        size = 0
        while todo:
            key = todo.pop()
            size += 1
            neighbors = catalog.adjacency.get(key, set()) & unseen
            unseen.difference_update(neighbors)
            todo.extend(neighbors)
        components[size] += 1
    return dict(concepts=len(concepts), memberships=sum(counts.values()), covered_entities=len(counts),
                covered_documents=len({d for c in concepts for d in c.documents}), document_pairs=len(doc_pairs),
                strict_inclusions=len(nesting_pairs(concepts)),
                concept_sizes=dict(Counter(str(len(c.members)) for c in concepts)),
                memberships_per_entity=dict(Counter(str(counts.get(key, 0)) for key in catalog.entities)),
                by_topic={key: value | dict(documents=len(value["documents"])) for key, value in topics.items()},
                cc=dict(catalog.cc_counts), operations=event_stats,
                ee_component_sizes={str(k): v for k, v in components.items()},
                exact_dedup=sum(e["operation"] == "exact_dedup" or e.get("exact_collision", False)
                                for e in catalog.events),
                neighborhoods=[e for e in catalog.events if e["operation"] == "neighborhoods"])


def source_identity():
    sources = (Path(__file__), ROOT / "src/cgraph/shared_concepts_v2.py",
               Path(contextual.__file__), Path(previous.__file__),
               Path(__file__).with_name("run_shared_concepts_v2.py"),
               Path(__file__).with_name("sweep_shared_concepts_v2.py"))
    return dict(python=sys.version.split()[0],
                dependencies={pkg: version(pkg) for pkg in ("numpy", "psutil", "tqdm", "threadpoolctl", "filelock")},
                code_sha256={path.name: sha(path) for path in sources})


def load_checked(path):
    data = read(path)
    checksum = data.pop("checksum")
    if checksum != digest(data):
        raise ValueError(f"Corrupt checkpoint: {path}")
    return data


def write_checked(path, data):
    payload = canonical(data).encode("utf-8")
    checksum = sha256(payload).hexdigest()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(memoryview(payload)[:-1])
        if data:
            stream.write(b",")
        stream.write(f'"checksum":"{checksum}"}}\n'.encode("ascii"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def historical_cases(catalog):
    """Сопоставляет 20 старых примеров с новыми проверками EE, EC и CC."""
    manifest = read(FROZEN_MANIFEST)
    name = "DIAGNOSTIC_CASES.csv"
    path = FROZEN_MANIFEST.with_name(name)
    if sha(path) != manifest["files_sha256"][name]:
        raise ValueError("Historical diagnostics checksum mismatch")
    cases = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["window"] != WINDOW or (float(row["threshold_name"]), float(row["threshold_context"])) != EE:
                continue
            a, b = (catalog.entities[row[f"{key}_entity_id"]] for key in ("a", "b"))
            score = aligned_cosine(a.vectors, b.vectors)
            guards = contextual.guard_reasons(a.profile, b.profile)
            passed = a.document_id != b.document_id and score[0] >= EE[0] and score[1] >= EE[1] and not guards
            if passed != (row["guarded_pass"] == "True") or passed != (b.entity_id in catalog.adjacency.get(a.entity_id, ())):
                raise ValueError(f"Historical EE case mismatch: {row['index']}")
            a_concepts = sorted(catalog.by_entity[a.entity_id])
            b_concepts = sorted(catalog.by_entity[b.entity_id])
            ec = []
            for entity, target in ((a, b_concepts), (b, a_concepts)):
                for key in target:
                    concept = catalog.concepts[key]
                    scores = aligned_cosine(entity.vectors, concept.sums)
                    ec.append(dict(entity=entity.entity_id, concept=key, already_member=entity.entity_id in concept.set_members,
                                   scores=scores.tolist(), thresholds_pass=bool(catalog.config.ec.passes(scores))))
            pairs = sorted({tuple(sorted((x, y))) for x in a_concepts for y in b_concepts if x != y})
            cc = []
            for x, y in pairs:
                first, second = catalog.concepts[x], catalog.concepts[y]
                scores = aligned_cosine(first.sums, second.sums)
                nested = first.set_members < second.set_members or second.set_members < first.set_members
                cc.append(dict(concepts=(x, y), scores=scores.tolist(), nested=nested,
                               thresholds_pass=bool(catalog.config.cc.passes(scores))))
            cases.append(dict(index=int(row["index"]), historical_candidate=row["old_candidate_id"],
                              historical_kind=row["old_kind"], entity=a.entity_id, reference=b.entity_id,
                              ee=dict(scores=score.tolist(), guard_reasons=guards, passed=bool(passed)),
                              ec=ec, cc=cc, shared_memberships=sorted(set(a_concepts) & set(b_concepts))))
    if sorted(row["index"] for row in cases) != list(range(1, 21)):
        raise ValueError("Expected all 20 historical diagnostics")
    return cases


def run_catalog(entities, config, namespace, output, *, metadata, base=None, regression=False,
                threads=4, max_seconds=1800, max_rss=2 * 1024**3, checkpoint_every=100, ee_cache=None):
    """Публикует целое файловое поколение, продолжает незавершённые расчёты по fingerprint."""
    if min(threads, max_seconds, max_rss, checkpoint_every) <= 0:
        raise ValueError("Resource limits must be positive")
    if metadata.get("corpus") == "Corpus100" and (base is not None or not namespace.startswith("research:corpus100:")):
        raise ValueError("Frozen Corpus100 is isolated from incremental catalogs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".lock", timeout=0), threadpool_limits(limits=threads), MemorySampler() as sampler:
        started = perf_counter()
        check = ResourceBudget(sampler, started, max_seconds, max_rss).check
        return _run(entities, config, namespace, output, metadata, base, regression, checkpoint_every,
                    check, sampler, started, threads, max_seconds, max_rss, ee_cache)


def _run(entities, config, namespace, output, metadata, base, regression, checkpoint_every,
         check, sampler, started, threads, max_seconds, max_rss, ee_cache):
    started_at = datetime.now(timezone.utc).isoformat()
    catalog = Catalog(entities, config, namespace, digest(metadata["recipe"]))
    source = source_identity()
    fingerprints = {key: e.fingerprint for key, e in sorted(catalog.entities.items())}
    documents = {}
    for entity in entities:
        for mention in entity.profile["evidence"]:
            revision = mention["revision_id"]
            if documents.setdefault(entity.document_id, revision) != revision:
                raise ValueError(f"Multiple source revisions for {entity.document_id}; rebuild required")
    identity = dict(config=config.to_dict(), config_hash=catalog.config_hash,
                    model=MODEL, window=WINDOW, ee=dict(name=EE[0], context=EE[1]),
                    ee_blas_threads=threads,
                    namespace=namespace, source=source, input=metadata, profiles=digest(fingerprints),
                    parent=[base["generation"], digest(base["state"])] if base is not None else None,
                    ee_regression=regression)
    if base is not None:
        before = base["identity"]
        same_research_graph = (
            namespace == RESEARCH_CATALOG_NAMESPACE
            and metadata.get("research_graph") == before["input"].get("research_graph") == RESEARCH_GRAPH_NAMESPACE
        )
        if (before["namespace"] != namespace or before["config"] != config.to_dict()
                or before["input"]["recipe"] != metadata["recipe"] or before["source"] != source
                or (before["input"].get("corpus") == "Corpus100" and not same_research_graph)):
            raise ValueError("Base catalog namespace, configuration, recipe or implementation mismatch")
        if any(fingerprints.get(key) != value for key, value in base["fingerprints"].items()):
            raise ValueError("Existing canonical profile changed or disappeared; rebuild required")
        if any(documents.get(key) != value for key, value in base["documents"].items()):
            raise ValueError("Existing document revision changed or disappeared; rebuild required")
    fingerprint = digest(identity)
    completed, checkpoint = output / "catalog.json", output / "checkpoint.json"
    if completed.exists():
        artifact = load_checked(completed)
        if artifact["fingerprint"] != fingerprint:
            raise ValueError("Output already belongs to a different generation")
        for filename, key in (("ee.json", "ee_artifact_sha256"), ("memberships.json", "memberships_sha256")):
            if sha(output / filename) != artifact[key]:
                raise ValueError(f"Invalid completed generation component: {filename}")
        return artifact
    edges_path = output / "ee.json"
    if edges_path.exists():
        ee = load_checked(edges_path)
        if ee["fingerprint"] != fingerprint:
            raise ValueError("EE checkpoint belongs to a different generation")
    else:
        incoming = catalog.entities.keys() - (base["fingerprints"].keys() if base is not None else set())
        ee_input = digest([fingerprints, sorted(incoming), source, EE, contextual.RULES, threads])
        if ee_cache is None:
            edges, metrics = ee_edges(catalog.entities, incoming, check)
            data = dict(edges=edges, metrics=metrics)
        else:
            ee_cache.mkdir(parents=True, exist_ok=True)
            cache_path = ee_cache / f"{ee_input}.json"
            with FileLock(cache_path.with_suffix(".lock"), timeout=0):
                if cache_path.exists():
                    data = load_checked(cache_path)
                    if data["ee_input"] != ee_input:
                        raise ValueError("Incompatible shared EE cache")
                else:
                    edges, metrics = ee_edges(catalog.entities, incoming, check)
                    data = dict(edges=edges, metrics=metrics, ee_input=ee_input)
                    write_checked(cache_path, data)
        ee = dict(fingerprint=fingerprint, **data)
        write_checked(edges_path, ee)
    if regression:
        actual = {key: ee["metrics"][key] for key in EXPECTED_EE}
        if actual != EXPECTED_EE:
            raise ValueError(f"Corpus100 EE regression failed: {actual}")
    if checkpoint.exists():
        saved = load_checked(checkpoint)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Catalog checkpoint belongs to a different generation")
        catalog.restore(saved["catalog"])
        catalog.event_seconds = dict(saved["event_seconds"])
        attempts = saved["attempts"]
        prior_seconds = saved["seconds"]
        prior_peak = saved["peak_rss_bytes"]
        started_at = saved["started_at"]
    else:
        attempts = prior_seconds = prior_peak = 0
        if base is not None:
            catalog.restore(base["state"])
            catalog.event_seconds = dict(base["event_seconds"])
        old_keys = {c.key for c in catalog.concepts.values()}
        pairs = [(a, b) for a, b, _, _ in ee["edges"]]
        catalog.add_edges(pairs)
        if base is not None:
            concept_ids = sorted(catalog.concepts)
            for key in tqdm(sorted(catalog.entities.keys() - base["fingerprints"].keys()), desc="Append", unit="entity"):
                check()
                for concept_id in concept_ids:
                    if concept_id in catalog.concepts:
                        catalog.append(concept_id, key)
        if base is None or pairs:
            catalog.initialize({key for pair in pairs for key in pair} if base is not None else catalog.entities,
                               partial(tqdm, desc="Initialize EC", unit="group"), check)
        if base is None or catalog.entities.keys() - base["fingerprints"].keys():
            catalog.event("before_cc", statistics=summaries(catalog))
        if base is None:
            catalog.prepare_cc(partial(tqdm, desc="CC comparisons", unit="concept"), check)
        else:
            changed = sorted(c.concept_id for c in catalog.concepts.values() if c.key not in old_keys)
            done = set(catalog.concepts) - set(changed)
            for key in tqdm(changed, desc="Changed CC comparisons", unit="concept"):
                check()
                catalog.enqueue(catalog.concepts[key], [catalog.concepts[other] for other in sorted(done)])
                done.add(key)
            if changed:
                catalog.retry_nesting()

    def save():
        write_checked(checkpoint, dict(fingerprint=fingerprint, catalog=catalog.snapshot(), attempts=attempts,
                                       event_seconds=catalog.event_seconds,
                                       started_at=started_at,
                                       seconds=prior_seconds + perf_counter() - started,
                                       peak_rss_bytes=max(prior_peak, sampler.peak)))

    save()
    with tqdm(initial=attempts, desc="CC merge attempts", unit="pair") as progress:
        while catalog.queue:
            check()
            if not catalog.merge_next():
                break
            attempts += 1
            progress.update()
            if attempts % checkpoint_every == 0:
                save()
    check()
    state = catalog.snapshot()
    memberships = []
    evidence = {key: catalog.entities[key].profile["evidence"] for key in catalog.by_entity}
    for c in state["concepts"]:
        for key, scores in zip(c["members"], c["scores"], strict=True):
            memberships.append(dict(relation="HAS_SHARED_CONCEPT", entity_id=key, concept_id=c["concept_id"],
                                    concept_version=c["version"], scores=dict(name=scores[0], context=scores[1]),
                                    profile_fingerprint=fingerprints[key], events=c["events"], evidence=evidence[key],
                                    factual=False, identity=False, fact_inheritance_allowed=False,
                                    generation=fingerprint, config_hash=catalog.config_hash))
    memberships_path = output / "memberships.json"
    write_checked(memberships_path, dict(generation=fingerprint, memberships=memberships))
    artifact = dict(fingerprint=fingerprint, identity=identity, fingerprints=fingerprints, state=state, documents=documents,
                    ee=ee["metrics"], ee_artifact_sha256=sha(edges_path), status="complete",
                    memberships_sha256=sha(memberships_path),
                    generation=fingerprint, statistics=summaries(catalog),
                    event_seconds=catalog.event_seconds,
                    historical_cases=historical_cases(catalog) if metadata.get("corpus") == "Corpus100" else [],
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=prior_seconds + perf_counter() - started, peak_rss_bytes=max(prior_peak, sampler.peak),
                    resources=dict(threads=threads, max_seconds=max_seconds, max_rss_bytes=max_rss))
    write_checked(completed, artifact)
    return artifact


def write_entity_bundle(path, entities):
    write_checked(path, dict(entities=[
        dict(profile=e.profile, windows=json.loads(e.windows_json), vectors=e.vectors.tolist(), recipe_hash=e.recipe_hash)
        for e in sorted(entities, key=lambda e: e.entity_id)
    ]))


def read_entity_bundle(path):
    return [Entity.create(**row) for row in load_checked(path)["entities"]]
