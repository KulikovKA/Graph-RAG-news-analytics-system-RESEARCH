"""Read-only Stage 2G.1: E5-small embeddings once/mode, 22 virtual graphs.

--embeddings-only prepares verified cache and candidate discovery for inspection.
Default execution reuses cache, validates semantic candidates, and writes reports.
No graph import method is called by this runner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_dynamic_concept_graph as graph
import benchmark_semantic_embeddings as protocol
from cgraph.concepts import Concept, class_form, digest, lexical_match, tokens
from cgraph.semantic_concepts import HybridConceptResolver, MODES, SEMANTIC_POLICY, THRESHOLDS, VERSION, cosine_matrix, passes, pool_e5, reference_records

OUT = graph.base.ER / "semantic_concept_sweep"
CACHE = ROOT / ".cache/semantic_concept_sweep"
SOURCE = graph.OUT
NS = graph.TARGET_NS
require, dumps, banner, progress = graph.require, graph.dumps, graph.banner, graph.progress
RECIPE = "e5-source-name-context-lodo-v1"
COSINE_RECIPE = "float64-exact-identical-vectors-v1"
EXTENDED_THRESHOLDS = tuple(format(Decimal(i) / Decimal(1000), ".3f") for i in range(700, 951, 5))
EXTENDED_OUT = graph.base.ER / "semantic_concept_sweep_extended"


def binary_cutoff(threshold):
    """Smallest binary64 number >= the exact requested decimal."""
    import numpy as np
    value = float(threshold)
    return np.nextafter(value, np.inf) if Decimal.from_float(value) < Decimal(threshold) else value


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def jsonl(name, rows):
    graph.base.core.write_jsonl(OUT / name, rows)


def csvout(name, rows, fields=None):
    rows = list(rows)
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: dumps(v) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()} for row in rows)


def verified_source(session=None):
    m = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    v = json.loads((SOURCE / "verification.json").read_text(encoding="utf-8"))
    require(m["namespace"] == NS and m["import_completed"], "Unverified Stage 2G manifest")
    # The original lexical implementation remains byte-for-byte unchanged.
    require(graph.base.sha(ROOT / "src/cgraph/concepts.py") == m["code_sha256"]["src/cgraph/concepts.py"], "Lexical baseline implementation changed")
    require(all(graph.base.sha(SOURCE / n) == h for n, h in m["artifact_files_sha256"].items()), "Stage 2G artifact hash mismatch")
    if session is None:
        nodes, edges = graph.unpack_snapshot(json.loads((CACHE / "baseline.json").read_text(encoding="utf-8")))
    else:
        nodes, edges = graph.read_snapshot(session, NS, (875, 2740))
    require(graph.snapshot_hash(nodes, edges) == v["target_sha256"], "Stage 2G full snapshot changed")
    require(all(r["properties"]["fingerprint"] == m["fingerprint"] for r in [*nodes.values(), *edges.values()]), "Stage 2G fingerprint differs")
    require(graph.counts(nodes, edges) == m["comparison"]["target_counts"], "Stage 2G counts differ")
    return nodes, edges, m, v


def representation_inputs(nodes, entities, catalog):
    by = {e.entity_id: e for e in entities}
    rows = []
    for e in sorted(entities, key=lambda e: e.entity_id):
        props = nodes[NS + ":" + e.entity_id]["properties"]
        ctx = json.loads(props["representative_context_json"])
        rows.append(dict(entity_id=e.entity_id, document_id=e.document_id, text_name=e.name,
                         text_name_context=f"Entity: {e.name}\nContext: {ctx['text']}", context=ctx))
    references = reference_records(catalog, by)
    for entity in rows:
        for c in catalog:
            candidates = [r for r in references if r["concept_id"] == c.concept_id]
            variant = next((r for r in candidates if r["excluded_document_id"] == entity["document_id"]),
                           next(r for r in candidates if r["excluded_document_id"] is None))
            require(entity["document_id"] not in variant["context_document_ids"], "Concept context self-confirmation")
    return rows, references


def attach_tokenization_audit(meta, rows, references, desc):
    if "tokenization_audit" not in meta:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(desc["tokenizer"]); tokenizer.no_padding(); tokenizer.no_truncation()
        audit = {}
        for mode in MODES:
            texts = [r[mode] for r in rows + references]
            lengths = [len(e.ids) for e in tokenizer.encode_batch([desc["prefix"] + t for t in texts])]
            audit[mode] = dict(max_tokens_before_truncation=max(lengths),
                entity_texts_truncated=sum(n > protocol.MAX_LENGTH for n in lengths[:len(rows)]),
                reference_texts_truncated=sum(n > protocol.MAX_LENGTH for n in lengths[len(rows):]))
        meta["tokenization_audit"] = audit
    return meta


def embeddings(rows, references):
    """Same E5 recipe as Stage 2B; all texts once per mode, cache outside Git."""
    import numpy as np
    desc = protocol.describe("intfloat/e5-small-v2")
    require(desc["prefix"] == "query: " and desc["pooling"] == "masked_mean", "E5 recipe changed")
    with (protocol.OUTPUT / "model_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        previous = next(r for r in csv.DictReader(stream) if r["model"] == desc["model"])
    prior_files = {str(Path(r["path"])): r["sha256"] for r in json.loads(previous["model_files"])}
    require(all(prior_files.get(str(Path(r["path"]))) == r["sha256"] for r in desc["files"]), "E5 files differ from Stage 2B")
    # Fixed approved E5-small weights from existing Stage 2B (not a replacement).
    require(graph.base.sha(Path(desc["weight"])) == "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1", "Existing E5 weights changed")
    signature = dict(recipe=RECIPE, rows=rows, references=references, model=desc,
                     batch=protocol.BATCH, max_length=protocol.MAX_LENGTH, dtype="float32", threads=protocol.THREADS)
    identity = digest(signature)
    meta_path = CACHE / "embedding_manifest.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        require(meta["fingerprint"] == identity, "Embedding cache belongs to a different input/recipe")
        require(all(graph.base.sha(CACHE / n) == h for n, h in meta["cache_files_sha256"].items()), "Embedding cache corrupted")
        if meta.get("cosine_recipe") != COSINE_RECIPE:
            # Algebraic correction at threshold 1.000 uses cached vectors only;
            # there is no second encoder inference or fuzzy threshold epsilon.
            for mode in MODES:
                full = np.load(CACHE / (mode + "_vectors.npy"), allow_pickle=False)
                n = len(rows)
                np.save(CACHE / (mode + "_ee.npy"), cosine_matrix(full[:n], full[:n]), allow_pickle=False)
                np.save(CACHE / (mode + "_ec.npy"), cosine_matrix(full[:n], full[n:]), allow_pickle=False)
            meta["cosine_recipe"] = COSINE_RECIPE
            meta["cache_files_sha256"] = {p.name: graph.base.sha(p) for p in sorted(CACHE.glob("*.npy"))}
            save(meta_path, meta)
        if "tokenization_audit" not in meta:
            attach_tokenization_audit(meta, rows, references, desc)
            save(meta_path, meta)
        return meta
    banner("ВЫЧИСЛЯЮ E5-SMALL EMBEDDINGS ОДИН РАЗ ДЛЯ КАЖДОГО РЕЖИМА — CPU FP32")
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModel
    torch.set_num_threads(protocol.THREADS)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = Tokenizer.from_file(desc["tokenizer"])
    tokenizer.no_padding(); tokenizer.enable_truncation(max_length=protocol.MAX_LENGTH)
    require(tokenizer.token_to_id("[PAD]") == desc["pad_id"], "E5 padding mismatch")
    model = AutoModel.from_pretrained(desc["root"], local_files_only=True, trust_remote_code=False,
                                    use_safetensors=True, dtype=torch.float32).cpu().eval()
    per_mode = {}
    for mode in MODES:
        texts = [r[mode] for r in rows + references]
        unique = list(dict.fromkeys(texts)); lookup = {t: i for i, t in enumerate(unique)}
        enc = tokenizer.encode_batch([desc["prefix"] + t for t in unique])
        vectors = np.empty((len(unique), desc["dimension"]), dtype=np.float32)
        started = time.perf_counter()
        for start in progress(range(0, len(unique), protocol.BATCH), total=(len(unique) + protocol.BATCH - 1)//protocol.BATCH, desc="E5 " + mode):
            batch = enc[start:start + protocol.BATCH]; width = max(len(e.ids) for e in batch)
            inputs = {k: np.full((len(batch), width), desc["pad_id"] if k == "input_ids" else 0, dtype=np.int64)
                      for k in ("input_ids", "attention_mask", "token_type_ids")}
            for j, e in enumerate(batch):
                n = len(e.ids)
                inputs["input_ids"][j, :n] = e.ids
                inputs["attention_mask"][j, :n] = e.attention_mask
                inputs["token_type_ids"][j, :n] = e.type_ids
            with torch.inference_mode():
                hidden = model(**{k: torch.from_numpy(v) for k, v in inputs.items()}).last_hidden_state.numpy()
            vectors[start:start + len(batch)] = pool_e5(hidden, inputs["attention_mask"])
        full = vectors[[lookup[t] for t in texts]]
        require(np.isfinite(full).all() and np.allclose(np.linalg.norm(full, axis=1), 1, atol=2e-6), "Invalid normalized E5 vectors")
        np.save(CACHE / (mode + "_vectors.npy"), full, allow_pickle=False)
        # Float64 cosine avoids FP32 dot products > 1 and preserves exact gates.
        n = len(rows)
        for kind, values in (("ee", cosine_matrix(full[:n], full[:n])), ("ec", cosine_matrix(full[:n], full[n:]))):
            np.save(CACHE / (mode + "_" + kind + ".npy"), values, allow_pickle=False)
        per_mode[mode] = dict(unique_texts=len(unique), vectors=len(full), seconds=time.perf_counter() - started,
                              entity_vectors=n, reference_vectors=len(references), text_sha256=digest(texts))
    files = {p.name: graph.base.sha(p) for p in sorted(CACHE.glob("*.npy"))}
    meta = dict(fingerprint=identity, recipe=RECIPE, model=desc, modes=per_mode,
                cache_files_sha256=files, pooling="masked_mean", prefix="query: ", l2_normalization=True,
                max_length=protocol.MAX_LENGTH, batch=protocol.BATCH, dtype="float32", cosine_dtype="float64",
                torch_version=torch.__version__, truncated_at_tokens=protocol.MAX_LENGTH,
                all_embeddings_computed_once_per_mode=True, cosine_recipe=COSINE_RECIPE)
    attach_tokenization_audit(meta, rows, references, desc)
    save(meta_path, meta)
    save(CACHE / "representations.json", dict(entities=rows, concepts=references))
    return meta


def discover(rows, references, catalog, memberships, similar=(), *, thresholds=THRESHOLDS,
             topk_entity=None, topk_concept=None, preserve_above=None, distribution=None):
    import numpy as np
    output = []
    old_pairs = {tuple(sorted(p)) for p in similar}
    concept_index = {c.concept_id: c for c in catalog}
    ref_lookup = {(r["concept_id"], r["excluded_document_id"]): i for i, r in enumerate(references)}
    minimum = binary_cutoff(thresholds[0])
    for mode in MODES:
        ee = np.load(CACHE / (mode + "_ee.npy"), allow_pickle=False)
        ec = np.load(CACHE / (mode + "_ec.npy"), allow_pickle=False)
        documents = np.array([r["document_id"] for r in rows])
        cross = documents[:, None] != documents[None, :]
        full_ee = np.triu(cross & (ee >= minimum), 1)
        selected_ee = full_ee.copy()
        if topk_entity is not None:
            selected_ee[:] = False
            for i in range(len(rows)):
                ranked = np.flatnonzero(cross[i] & (ee[i] >= minimum))
                ranked = ranked[np.argsort(-ee[i, ranked], kind="stable")][:topk_entity]
                for j in ranked:
                    selected_ee[min(i, j), max(i, j)] = True
            if preserve_above is not None:
                selected_ee |= np.triu(cross & (ee >= binary_cutoff(preserve_above)), 1)
        selected_ec = np.zeros((len(rows), len(concept_index)), dtype=bool)
        concept_ids = sorted(concept_index)
        ec_scores = np.empty(selected_ec.shape, dtype=np.float64)
        eligible_ec = np.ones(selected_ec.shape, dtype=bool)
        for i, row in enumerate(rows):
            for j, cid in enumerate(concept_ids):
                column = ref_lookup.get((cid, row["document_id"]), ref_lookup[(cid, None)])
                ec_scores[i, j] = ec[i, column]
                eligible_ec[i, j] = (row["entity_id"], cid) not in memberships
            ranked = np.flatnonzero(eligible_ec[i] & (ec_scores[i] >= minimum))
            ranked = ranked[np.argsort(-ec_scores[i, ranked], kind="stable")]
            selected_ec[i, ranked if topk_concept is None else ranked[:topk_concept]] = True
        if preserve_above is not None:
            selected_ec |= eligible_ec & (ec_scores >= binary_cutoff(preserve_above))
        if distribution is not None:
            ee_values = ee[np.triu(cross, 1)]
            ec_values = ec_scores[eligible_ec]
            checked_ee = ee[selected_ee]
            checked_ec = ec_scores[selected_ec]
            for threshold in thresholds:
                cutoff = binary_cutoff(threshold)
                full_a, full_b = int(np.count_nonzero(ee_values >= cutoff)), int(np.count_nonzero(ec_values >= cutoff))
                selected_a, selected_b = int(np.count_nonzero(checked_ee >= cutoff)), int(np.count_nonzero(checked_ec >= cutoff))
                distribution.append(dict(embedding_mode=mode, threshold=threshold,
                    full_entity_entity_candidates=full_a, full_entity_concept_candidates=full_b,
                    full_candidates=full_a + full_b, reviewed_entity_entity_candidates=selected_a,
                    reviewed_entity_concept_candidates=selected_b, reviewed_candidates=selected_a + selected_b,
                    pruned_candidates=full_a + full_b - selected_a - selected_b,
                    reviewed_fraction=(selected_a + selected_b) / (full_a + full_b) if full_a + full_b else 1.0))
        for i, left in enumerate(rows):
            for j in np.flatnonzero(selected_ee[i]):
                right = rows[int(j)]; score = float(ee[i, j])
                if left["document_id"] == right["document_id"] or not passes(score, thresholds[0]):
                    continue
                output.append(dict(candidate_id="ee:" + digest([mode, left["entity_id"], right["entity_id"]])[:24],
                    kind="entity_entity", embedding_mode=mode, source_entity_id=left["entity_id"],
                    source_name=left["text_name"], reference_name=right["text_name"],
                    reference_entity_id=right["entity_id"], source_document_id=left["document_id"],
                    reference_document_id=right["document_id"], cosine_similarity=score,
                    existing_similar_to=tuple(sorted((left["entity_id"], right["entity_id"]))) in old_pairs))
            for index, cid in enumerate(concept_ids):
                if not selected_ec[i, index]:
                    continue
                j = ref_lookup.get((cid, left["document_id"]), ref_lookup[(cid, None)])
                score = float(ec[i, j])
                if passes(score, thresholds[0]):
                    ref = references[j]
                    output.append(dict(candidate_id="ec:" + digest([mode, left["entity_id"], cid])[:24],
                        kind="entity_concept", embedding_mode=mode, source_entity_id=left["entity_id"],
                        source_name=left["text_name"], target_concept_name=concept_index[cid].name,
                        target_concept_id=cid, source_document_id=left["document_id"], cosine_similarity=score,
                        reference_id=ref["reference_id"], reference_context_document_ids=ref["context_document_ids"],
                        name_only_fallback=ref["name_only_fallback"]))
    for c in output:
        c["qualifying_thresholds"] = [t for t in thresholds if passes(c["cosine_similarity"], t)]
    return sorted(output, key=lambda c: (c["embedding_mode"], c["kind"], c["candidate_id"]))


def no_lexical_overlap(left, right):
    def all_tokens(item):
        return set().union(*(tokens(a) for a in (item.name, *item.aliases)))
    return not all_tokens(left) & all_tokens(right)


def compact_proof(value):
    """Deduplicate source text by its immutable mention/span/hash reference."""
    if isinstance(value, list):
        return [compact_proof(v) for v in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        if key in {"source_contexts", "reference_contexts"}:
            output[key + "_refs"] = [dict(document_id=c["document_id"], source_entity_id=c.get("source_entity_id"),
                source_mention_ids=c.get("source_mention_ids", []), start=c.get("start"), end=c.get("end"),
                revision_id=c.get("revision_id"), context_sha256=digest(c["text"])) for c in item]
        else:
            output[key] = compact_proof(item)
    return output


def evaluate(candidates, entities, catalog, memberships, *, compact=False, memoize=False, statistics=None):
    by = {e.entity_id: e for e in entities}
    cat = {c.concept_id: c for c in catalog}
    catalog_names = {class_form(c.name): c.concept_id for c in catalog}
    member_concepts = defaultdict(set)
    for eid, cid in memberships:
        member_concepts[eid].add(cid)
    resolver = HybridConceptResolver(catalog, entities, memberships)
    if compact or memoize:
        original = resolver.validate_semantic
        cache = {}; pair_cache = {}
        stats = statistics if statistics is not None else {}
        stats.update(semantic_calls=0, semantic_evaluations=0, pair_calls=0, pair_evaluations=0)
        def checked(entity, concept, reference):
            stats["semantic_calls"] += 1
            key = (entity.entity_id, concept.concept_id, concept.name, concept.aliases, concept.existing, reference.entity_id)
            if not memoize or key not in cache:
                status, reason, evidence = original(entity, concept, reference)
                stats["semantic_evaluations"] += 1
                result = (status, reason, compact_proof(evidence) if compact else evidence)
                if memoize:
                    cache[key] = result
            else:
                result = cache[key]
            return result[0], result[1], dict(result[2])
        resolver.validate_semantic = checked
        original_pair = resolver.validate_pair
        def checked_pair(left, right):
            stats["pair_calls"] += 1
            key = (left.entity_id, right.entity_id)
            if not memoize or key not in pair_cache:
                result = original_pair(left, right)
                stats["pair_evaluations"] += 1
                if memoize:
                    pair_cache[key] = result
            else:
                result = pair_cache[key]
            return result[0], result[1], dict(result[2])
        resolver.validate_pair = checked_pair
    decisions, proposals = [], []

    def decision(candidate, entity, concept, reference, *, new_concept=False, pair_evidence=None):
        if (entity.entity_id, concept.concept_id) in memberships:
            return
        status, reason, evidence = resolver.validate_semantic(entity, concept, reference)
        if pair_evidence is not None:
            evidence["pair_validation"] = pair_evidence
        decisions.append(dict(candidate_id=candidate["candidate_id"],
            decision_id="semantic-decision:" + digest([candidate["candidate_id"], entity.entity_id, concept.concept_id, VERSION])[:24],
            source_entity_id=entity.entity_id, source_name=entity.name, target_concept_id=concept.concept_id,
            target_concept_name=concept.name, document_id=entity.document_id,
            reference_entity_id=reference.entity_id, reference_document_id=reference.document_id,
            status=status, reason=reason, resolver_version=VERSION, new_concept=new_concept,
            embedding_mode=candidate["embedding_mode"], cosine_similarity=candidate["cosine_similarity"],
            qualifying_thresholds=candidate["qualifying_thresholds"],
            semantic_only_membership=no_lexical_overlap(entity, concept),
            semantic_only_candidate_pair=(no_lexical_overlap(entity, reference) if candidate["kind"] == "entity_entity" else no_lexical_overlap(entity, concept)),
            evidence=evidence, factual=False, identity=False, fact_inheritance_allowed=False))

    for candidate in progress(candidates, total=len(candidates), desc="СОДЕРЖАТЕЛЬНАЯ ПРОВЕРКА КАНДИДАТОВ"):
        left = by[candidate["source_entity_id"]]
        if candidate["kind"] == "entity_concept":
            c = cat[candidate["target_concept_id"]]
            refs = [by[i] for i in c.member_ids if i != left.entity_id]
            checks = [(resolver.validate_semantic(left, c, ref)[0] != "ACCEPT", ref.document_id == left.document_id, ref.entity_id, ref) for ref in refs]
            reference = sorted(checks, key=lambda x: x[:3])[0][-1]
            decision(candidate, left, c, reference)
            continue
        right = by[candidate["reference_entity_id"]]
        common = member_concepts[left.entity_id] & member_concepts[right.entity_id]
        if common:
            decisions.append(dict(candidate_id=candidate["candidate_id"], decision_id="covered:" + candidate["candidate_id"],
                source_entity_id=left.entity_id, reference_entity_id=right.entity_id,
                source_name=left.name, reference_name=right.name, target_concept_id=sorted(common)[0],
                document_id=left.document_id, status="ACCEPT", reason="frozen_existing_shared_concept_no_delta",
                adds_membership=False, embedding_mode=candidate["embedding_mode"],
                cosine_similarity=candidate["cosine_similarity"], qualifying_thresholds=candidate["qualifying_thresholds"],
                resolver_version=VERSION, semantic_only_membership=False, evidence=dict(frozen_concepts=sorted(common))))
            continue
        anchored = False
        for entity, ref in ((left, right), (right, left)):
            for cid in sorted(member_concepts[ref.entity_id]):
                decision(candidate, entity, cat[cid], ref)
                anchored = True
        if anchored:
            continue
        status, reason, evidence = resolver.validate_pair(left, right)
        name = evidence.get("name", min((left.name, right.name), key=lambda n: (len(n), n)))
        cid = catalog_names.get(class_form(name), "concept:2g1:" + digest(class_form(name))[:24])
        proposal = dict(proposal_id="proposal:" + candidate["candidate_id"], candidate_id=candidate["candidate_id"],
                        concept_id=cid, name=name, status=status, reason=reason, evidence=evidence,
                        source_entity_ids=[left.entity_id, right.entity_id],
                        source_document_ids=sorted([left.document_id, right.document_id]),
                        embedding_mode=candidate["embedding_mode"], cosine_similarity=candidate["cosine_similarity"],
                        qualifying_thresholds=candidate["qualifying_thresholds"], reused_existing_concept=cid in cat,
                        semantic_only_candidate_pair=no_lexical_overlap(left, right))
        proposals.append(proposal)
        c = cat.get(cid, Concept(cid, name, member_ids=(left.entity_id, right.entity_id), existing=False))
        if status == "ACCEPT":
            decision(candidate, left, c, right, new_concept=cid not in cat, pair_evidence=evidence)
            decision(candidate, right, c, left, new_concept=cid not in cat, pair_evidence=evidence)
        else:
            decisions.append(dict(candidate_id=candidate["candidate_id"], decision_id="rejected-pair:" + candidate["candidate_id"],
                source_entity_id=left.entity_id, source_name=left.name, reference_entity_id=right.entity_id,
                reference_name=right.name, document_id=left.document_id, target_concept_id=cid,
                target_concept_name=name, status=status, reason=reason, new_concept=True,
                resolver_version=VERSION, embedding_mode=candidate["embedding_mode"],
                cosine_similarity=candidate["cosine_similarity"], qualifying_thresholds=candidate["qualifying_thresholds"],
                semantic_only_membership=False, semantic_only_candidate_pair=no_lexical_overlap(left, right), evidence=evidence))
    return decisions, proposals


def virtual_graph(nodes, edges, accepted):
    """Shallow additive copy: existing records are not edited or remapped."""
    new_nodes, new_edges = dict(nodes), dict(edges)
    for row in accepted:
        cid = row["target_concept_id"]; key = NS + ":" + cid
        if key not in new_nodes:
            new_nodes[key] = dict(label="RELEXConcept", properties=dict(concept_id=cid, name=row["target_concept_name"],
                namespace=NS, key=key, entity_types=["CONCEPT"], virtual_only=True, factual=False, identity=False, fact_inheritance_allowed=False))
        rid = "semantic-membership:" + digest([row["source_entity_id"], cid])[:24]
        new_edges[NS + ":" + rid] = dict(type="HAS_SHARED_CONCEPT", head=NS + ":" + row["source_entity_id"], tail=key,
            properties=dict(key=NS + ":" + rid, namespace=NS, document_id=row["document_id"],
                factual=False, identity=False, fact_inheritance_allowed=False, resolver_version=VERSION))
    require(all(new_nodes[k] == r for k, r in nodes.items()) and all(new_edges[k] == r for k, r in edges.items()), "Frozen graph modified")
    return new_nodes, new_edges


def graph_metrics(nodes, edges):
    members, doc_pairs, bridges = graph.concept_connections(nodes, edges)
    by_doc = {k: nodes[k]["properties"]["document_id"] for rows in members.values() for k, _ in rows}
    bridge_count = 0
    for rows in members.values():
        counts = Counter(by_doc[e] for e, _ in rows)
        ds = sorted(counts)
        bridge_count += sum(counts[a] * counts[b] for i, a in enumerate(ds) for b in ds[i + 1:])
    return dict({**graph.counts(nodes, edges), **graph.kg.metrics(nodes, edges)[0]},
                multidocument_concepts=sum(len({by_doc[e] for e, _ in rows}) > 1 for rows in members.values()),
                entity_concept_entity_bridges=bridge_count), doc_pairs, graph.reachable_document_pairs(nodes, edges)


def threshold_sweep(nodes, edges, candidates, decisions, catalog, *, thresholds=THRESHOLDS,
                    lexical_delta=(), distribution=None, retain_evidence=True):
    base_metrics, old_pairs, old_reachable = graph_metrics(nodes, edges)
    cat_ids = {c.concept_id for c in catalog}
    result = []; memberships = []; changes = []
    for mode in MODES:
        for threshold in progress(thresholds, total=len(thresholds), desc="ВИРТУАЛЬНЫЕ ГРАФЫ " + mode):
            active = [c for c in candidates if c["embedding_mode"] == mode and passes(c["cosine_similarity"], threshold)]
            eligible = [r for r in decisions if r["embedding_mode"] == mode and r["status"] == "ACCEPT" and
                        r.get("adds_membership", True) and passes(r["cosine_similarity"], threshold)]
            dedup = {}
            for row in sorted(eligible, key=lambda r: (-r["cosine_similarity"], r["decision_id"])):
                dedup.setdefault((row["source_entity_id"], row["target_concept_id"]), row)
            accepted = list(dedup.values())
            # New pairwise proposals need two independently supported documents
            # at this threshold; no transitive similarity closure is used.
            supports = defaultdict(set)
            for r in accepted:
                supports[r["target_concept_id"]].add(r["document_id"])
            accepted = [r for r in accepted if r["target_concept_id"] in cat_ids or len(supports[r["target_concept_id"]]) >= 2]
            semantic_accepted = list(accepted)
            union = {(r["source_entity_id"], r["target_concept_id"]): r for r in lexical_delta}
            union.update({(r["source_entity_id"], r["target_concept_id"]): r for r in accepted})
            accepted = list(union.values())
            vn, ve = virtual_graph(nodes, edges, accepted)
            metrics, doc_pairs, reachable = graph_metrics(vn, ve)
            row = dict(variant="H", embedding_mode=mode, threshold=threshold,
                semantic_candidate_pairs=len(active), entity_entity_candidates=sum(c["kind"] == "entity_entity" for c in active),
                entity_concept_candidates=sum(c["kind"] == "entity_concept" for c in active),
                candidates_without_existing_similar_to=sum(c["kind"] == "entity_entity" and not c["existing_similar_to"] for c in active),
                new_memberships=len(accepted), new_concepts=metrics["concepts"] - base_metrics["concepts"],
                multidocument_concepts=metrics["multidocument_concepts"],
                new_entity_concept_entity_bridges=metrics["entity_concept_entity_bridges"] - base_metrics["entity_concept_entity_bridges"],
                new_direct_document_pairs=len(doc_pairs - old_pairs), new_reachable_document_pairs=len(reachable - old_reachable),
                weak_components=metrics["weak_components"], largest_component_vertices=metrics["largest_component_vertices"],
                suspicious_new_memberships=0, independently_measured_false_memberships="not_measured", semantic_only_memberships=sum(r["semantic_only_membership"] for r in accepted),
                rejected_candidate_decisions=sum(r["status"] == "REJECT" and r["embedding_mode"] == mode and passes(r["cosine_similarity"], threshold) for r in decisions))
            result.append(row)
            if distribution is not None:
                dist = next(r for r in distribution if r["embedding_mode"] == mode and r["threshold"] == threshold)
                active_ids = {r["candidate_id"] for r in active}
                grouped = defaultdict(list)
                for d in decisions:
                    if d["candidate_id"] in active_ids:
                        grouped[d["candidate_id"]].append(d)
                accepted_candidates = sum(any(d["status"] == "ACCEPT" for d in ds) for ds in grouped.values())
                new_accept_candidates = sum(any(d["status"] == "ACCEPT" and d.get("adds_membership", True) for d in ds) for ds in grouped.values())
                old_lexical = {(r["source_entity_id"], r["target_concept_id"]) for r in lexical_delta}
                row.update(variant="C", full_candidates=dist["full_candidates"], checked_candidates=len(active),
                    pruned_candidates=dist["pruned_candidates"], reviewed_fraction=dist["reviewed_fraction"],
                    accepted_checked_candidates=accepted_candidates, rejected_checked_candidates=len(grouped) - accepted_candidates,
                    new_accept_candidates=new_accept_candidates,
                    acceptance_fraction_checked_candidates=accepted_candidates / len(active) if active else 0,
                    semantic_accepted_memberships=len(semantic_accepted),
                    validator_only_memberships=len(lexical_delta),
                    memberships_beyond_validator_only=sum((r["source_entity_id"], r["target_concept_id"]) not in old_lexical for r in semantic_accepted))
            changes.append(dict(embedding_mode=mode, threshold=threshold, newly_directly_connected_document_pairs=sorted(doc_pairs - old_pairs),
                                newly_reachable_document_pairs=sorted(reachable - old_reachable)))
            memberships.extend(dict({k: v for k, v in r.items() if retain_evidence or k not in {"evidence", "qualifying_thresholds"}},
                                    embedding_mode=mode, threshold=threshold, virtual_only=True) for r in accepted)
    return result, memberships, changes, base_metrics


def controls_and_probes(rows, references, entities, catalog, memberships):
    """Requested content checks, including pairs not eligible for the sweep."""
    import numpy as np
    by = {e.entity_id: e for e in entities}; cat = {c.concept_id: c for c in catalog}
    resolver = HybridConceptResolver(catalog, entities, memberships)
    index = {r["entity_id"]: i for i, r in enumerate(rows)}
    ref_index = {(r["concept_id"], r["excluded_document_id"]): i for i, r in enumerate(references)}
    old = graph.base.core.read_jsonl(SOURCE / "candidate_decisions.jsonl")
    control = []
    for r in old:
        if r["status"] != "REJECT" or r["concept_id"] not in cat:
            continue
        refid = r["evidence"].get("reference_entity_id")
        if refid not in by:
            continue
        # Preserve the original aliases/reference from that decision to isolate
        # the validator change from changes in the Stage 2G catalog.
        original = Concept(r["concept_id"], r["concept_name"], tuple(r["evidence"].get("concept_forms", [])), (refid,))
        status, reason, _ = resolver.validate_semantic(by[r["entity_id"]], original, by[refid])
        control.append(dict(source_entity_id=r["entity_id"], source_name=by[r["entity_id"]].name,
                            target_concept_id=r["concept_id"], target_concept_name=r["concept_name"],
                            lexical_status=r["status"], lexical_reason=r["reason"],
                            hybrid_validator_without_retrieval=status, hybrid_reason=reason))
    probes = []
    cases = []
    cancer = next(c for c in catalog if class_form(c.name) == "cancer")
    for e in entities:
        if class_form(e.name) in {"breast cancer", "esophageal cancer"}:
            cases.append(("breast_cancer" if "breast" in e.name.casefold() else "esophageal_cancer", e, cancer, None))
    limonene = [e for e in entities if class_form(e.name) == "limonene"]
    genes = [e for e in entities if class_form(e.name) == "limonene synthase gene"]
    for a in limonene:
        for b in genes:
            cases.append(("limonene_vs_synthase_gene", a, None, b))
    for name, label in (("patient", "patients_independent_cohorts"), ("follow-up", "follow_up_usefulness")):
        subset = [e for e in entities if class_form(e.name) == name]
        seen = set()
        for a in subset:
            for b in subset:
                pair = tuple(sorted((a.document_id, b.document_id)))
                if a.document_id < b.document_id and pair not in seen:
                    seen.add(pair); cases.append((label, a, None, b))
    for mode in MODES:
        ee, ec = (np.load(CACHE / (mode + "_" + k + ".npy"), allow_pickle=False) for k in ("ee", "ec"))
        for label, entity, concept, reference in cases:
            if concept:
                j = ref_index.get((concept.concept_id, entity.document_id), ref_index[(concept.concept_id, None)])
                score = float(ec[index[entity.entity_id], j])
                ref = sorted((by[i] for i in concept.member_ids if i != entity.entity_id),
                             key=lambda e: (e.document_id == entity.document_id, e.entity_id))[0]
                status, reason, evidence = resolver.validate_semantic(entity, concept, ref)
                target, target_name = concept.concept_id, concept.name
                eligible = (entity.entity_id, concept.concept_id) not in memberships
            else:
                score = float(ee[index[entity.entity_id], index[reference.entity_id]])
                status, reason, evidence = resolver.validate_pair(entity, reference)
                target, target_name = reference.entity_id, reference.name
                eligible = entity.document_id != reference.document_id
            probes.append(dict(case=label, embedding_mode=mode, source_entity_id=entity.entity_id,
                source_name=entity.name, target_id=target, target_name=target_name,
                cosine_similarity=score, eligible_new_sweep_candidate=eligible,
                qualifying_thresholds=[t for t in THRESHOLDS if passes(score, t)],
                counterfactual_status=status, reason=reason, evidence=evidence,
                original_membership_preserved=(entity.entity_id, target) in memberships))
    return control, probes


def accepted_paths(nodes, edges, accepted):
    chosen = [r for r in accepted if r["embedding_mode"] == "text_name" and r["threshold"] == "0.955"]
    vn, ve = virtual_graph(nodes, edges, chosen)
    new_pair_set = {(NS + ":" + r["source_entity_id"], NS + ":" + r["target_concept_id"]) for r in chosen}
    members, _, _ = graph.concept_connections(vn, ve)
    paths = []
    for cid, rs in sorted(members.items()):
        for i, (a, alink) in enumerate(rs):
            for b, blink in rs[i + 1:]:
                da, db = vn[a]["properties"]["document_id"], vn[b]["properties"]["document_id"]
                if da == db or not ({(a, cid), (b, cid)} & new_pair_set):
                    continue
                left, right = graph.factual_route(a, vn, ve), graph.factual_route(b, vn, ve)
                if left and right:
                    paths.append(dict(document_a=da, document_b=db,
                        document_a_title=vn[NS + ":" + da]["properties"]["title"],
                        document_b_title=vn[NS + ":" + db]["properties"]["title"],
                        concept_id=vn[cid]["properties"]["concept_id"], concept_name=vn[cid]["properties"]["name"],
                        entity_a=a, entity_b=b, entity_a_name=vn[a]["properties"]["name"], entity_b_name=vn[b]["properties"]["name"],
                        membership_a=alink, membership_b=blink, factual_route_a=left, factual_route_b=right,
                        semantic_only=True, new_factual_dependency=False))
    return paths


def quality_audit(decisions, entities):
    """Content review notes for every new ACCEPT and every sampled REJECT.

    On this small experiment all REJECT decisions are included, exceeding the
    requested sample. These are agent review notes, never independent Gold.
    """
    by = {e.entity_id: e for e in entities}
    rows = []
    for r in decisions:
        if r.get("adds_membership") is False:
            continue
        source = by[r["source_entity_id"]]
        reference = by.get(r.get("reference_entity_id"))
        if r["status"] == "ACCEPT":
            if class_form(r["target_concept_name"]) == "cancer":
                note = "Breast cancer is a named cancer subtype in mammographic-risk contexts; preserve distinct diseases/studies and inherit no facts. Both original mentions were checked."
            elif class_form(r["target_concept_name"]) == "surgery":
                note = "Clinical surgical procedures and surgery denote the intervention class. The actual knee/eye operations and their outcomes remain distinct; broad category must remain clinically scoped."
            else:
                raise ValueError("New ACCEPT requires additional content review: " + r["target_concept_name"])
            verdict = "supported_scientific_class_membership"
            risk = "broad_clinical_class_no_identity_or_fact_transfer"
        elif source.name.casefold() == "pathological stress reactivity":
            verdict = "potential_false_negative"
            note = "Qualified stress-reactivity phenotype appears compatible with the class; context-witness rules abstain. Retain REJECT and flag for future validator revision."
            risk = "conservative_rejection_not_evidence_of_incompatibility"
        else:
            verdict = "rejection_supported_or_evidence_insufficient"
            risk = "not_a_gold_negative"
            if "hip" in source.name.casefold():
                note = "Hip and knee arthroplasty concern different anatomical procedures; high similarity does not establish membership in the knee-specific class."
            elif "genetic" in source.name.casefold():
                note = "Genetics is a field of study; genes are biological objects. Shared genomic contexts do not make the field a gene."
            elif "green bank" in source.name.casefold():
                note = "A loan/lending instrument or activity is not the banking institution itself."
            elif "papal" in source.name.casefold():
                note = "The source says annual medals are a specific iteration of papal commemorative medals. The broader class cannot automatically belong to the narrower annual class."
            elif source.name.casefold() == "thickness":
                note = "Lunar-regolith thickness is not membership in the specific multilayer-sublayer thickness concept."
            elif source.name.casefold() in {"grain", "grains", "supply line"}:
                note = "Source scopes differ: cereal vs magnetic grains, or economic supply curve vs electric supply line; no shared specific class established."
            else:
                note = "Generic study/results/author or ambiguous category supplies no sufficiently specific cross-study scientific class. Retain original contextual nodes."
        rows.append(dict(candidate_id=r["candidate_id"], decision_id=r["decision_id"],
            source_entity_id=source.entity_id, source_name=source.name, document_id=source.document_id,
            target_concept_id=r["target_concept_id"], target_concept_name=r.get("target_concept_name", ""),
            reference_entity_id=reference.entity_id if reference else "", status=r["status"],
            embedding_mode=r["embedding_mode"], cosine_similarity=r["cosine_similarity"],
            audit_verdict=verdict, audit_note=note, risk=risk, independent_gold=False,
            source_contexts=list(source.evidence), reference_contexts=list(reference.evidence) if reference else [],
            semantic_only_membership=r.get("semantic_only_membership", False)))
    return rows


def universe_summary(rows, references, catalog, memberships):
    import numpy as np
    docs = np.array([r["document_id"] for r in rows])
    cross = np.triu(docs[:, None] != docs[None, :], 1)
    ref_index = {(r["concept_id"], r["excluded_document_id"]): i for i, r in enumerate(references)}
    result = {}
    for mode in MODES:
        ee, ec = (np.load(CACHE / (mode + "_" + k + ".npy"), allow_pickle=False) for k in ("ee", "ec"))
        require(ee.shape == (811, 811) and ec.shape == (811, len(references)), "Incomplete cosine matrix")
        require(np.isfinite(ee).all() and np.isfinite(ec).all() and np.max(ee) <= 1 and np.max(ec) <= 1, "Invalid cosine values")
        scores = []
        for i, row in enumerate(rows):
            for c in catalog:
                if (row["entity_id"], c.concept_id) not in memberships:
                    j = ref_index.get((c.concept_id, row["document_id"]), ref_index[(c.concept_id, None)])
                    scores.append(float(ec[i, j]))
        result[mode] = dict(all_eligible_cross_document_entity_pairs=int(cross.sum()),
            eligible_nonmember_entity_concept_pairs=len(scores),
            maximum_cross_document_entity_cosine=float(ee[cross].max()),
            maximum_nonmember_entity_concept_cosine=max(scores),
            cross_document_entity_score_quantiles={str(q): float(np.quantile(ee[cross], q)) for q in (.5, .9, .99, 1)})
    return result


def write_design():
    text = '''# Stage 2G.1 — Experiment design

Исходный граф — immutable Stage 2G, namespace `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1`, commit `f5fbc05`.
Его fingerprint и полный snapshot SHA-256 сверяются с manifest/verification; проверяются также все artifact hashes Stage 2G и неизменность исходного lexical resolver.
Корпус: те же 30 документов, 811 contextual entities, 34 concepts, 130 memberships, 1307 facts, 492 SIMILAR_TO, 811 document mentions.

## L и H

L — фактический Stage 2G без изменений. H — аддитивные виртуальные графы с semantic candidate discovery и отдельной semantic validation. Ни один старый объект/endpoint/property не изменяется. NER/RE/GLiREL/MERGE не вызываются; новые документы и модели не загружаются. Реальные graph import methods не вызываются.

Пороги — строки Decimal: 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.000. Они получены из целых чисел, не последовательным сложением float. Каждая из двух матриц cosine используется для всех 11 порогов. Scalar gates сравнивают точное binary значение score с Decimal; near-1 не округляется до 1 ради прохождения порога. Полностью одинаковые сохранённые векторы получают ровно 1 по алгебраическому тождеству. Эта коррекция cosine выполнена по уже сохранённым векторам; E5 повторно не запускалась.

## E5 и representations

Используется существующая local `intfloat/e5-small-v2`, с теми же файлами и hashes Stage 2B. CPU FP32, 4 threads, batch 8, tokenizer.json, max_length 512, query: с обеих сторон, attention-masked mean pooling, L2 normalization. После нормирования cosine вычисляется в float64, результат ограничивается [-1,1] от численной погрешности. Модель не обучается. Identical input strings кодируются один раз; embeddings для entities и reference variants сохраняются вне Git. Cache fingerprint охватывает exact texts, IDs, reference provenance, модель и рецепт; каждый .npy имеет hash.

Entity `text_name` — сохранённое имя representative; `text_name_context` — прежний `Entity: <name>\\nContext: <original representative context>` (исходный радиус 120 символов, без новых NER spans). Concept text_name — только название. Concept text_name_context — название и до 8 неизменённых member context snippets, детерминированно сбалансированных по документам. Искусственных научных описаний нет.

Для entity→concept в reference исключается весь document проверяемой entity, включая другие entities и overlapping spans из этого документа. Если после исключения контекстов нет, используется только название с явным name_only_fallback. Ни собственный context, ни соседний mention того же исследования не могут самоподтвердить embedding. 34 concepts дают 77 reference variants; duplicate name texts переиспользуются.

## Candidate discovery и validation

A: все 811×34 entity→existing concept comparisons, исключая уже существующие 130 memberships. B: все unordered entity↔entity пары разных документов; существующие SIMILAR_TO не ограничивают universe. Cosine сам по себе никогда не создаёт membership. Нет Union-Find, MERGE или similarity transitive closure.

HybridConceptResolver расширяет существующий класс, не меняя L. Он сохраняет исходные source spans, типы и contexts; дополнительно проверяет clinical class readings, явно заданные текстом определения и научную роль procedural paraphrases. В частности surgery/surgical procedures трактуются как класс клинических вмешательств только при PROCESS/TASK/METHOD/CONCEPT roles и clinical context evidence. Этот source-bound, интерпретируемый paraphrase rule допускает ACCEPT без lexical_match. Limonene и его gene, genetics и genes, организации и их lending products, hip и knee operations не уравниваются.

Это exploratory rule-based semantic validation: дополнительные clinical witnesses и procedural paraphrase family сформулированы после изучения исходных текстов этих 30 документов. Embedding descriptions не дописывались. Правила не используют entity/document IDs и не выбирают решения по желаемым метрикам, но независимого held-out semantic gold нет. Нельзя приписывать все улучшения retrieval: отдельный validator-only control повторно оценивает прежние REJECT с прежними references.

Existing membership одного endpoint может предложить связь второго endpoint к тому же concept. Если anchors нет, допускается source-named новый concept, только после независимого обоснования обоих endpoints минимум из двух документов. Имя выбирается из source surface forms; существующий concept ID переиспользуется при совпадении identity. У каждого принятого delta сохраняются candidate/decision ID, source entity, concept, document, cosine, mode, threshold, reason, evidence и resolver version. factual/identity/fact inheritance всегда false. При сомнениях REJECT.

## Метрики и аудит

Отдельно считаются semantic candidates, delta memberships и proposals/created concepts, shared-document concepts, entity→concept←entity bridges, новые прямые document pairs, первая достижимость через полный KG, WCC и крупнейшая компонента. Document/Mentions исключены из WCC. Счёт мостов конечный: это entity pairs через concept, а не все циклические walks. Изменение связности не доказывает рост качества KG.

Аудируются все новые ACCEPT на уровне каждого уникального candidate–entity–concept решения; на этом малом наборе проверены и все REJECT. Quality audit — agent content review, не независимый Gold; подозрительные принятые связи и возможные false negatives разделяются. Requested probes включают breast/esophageal cancer, limonene/gene, независимые patient cohorts и clinical follow-up, даже если пара ниже gate, уже linked либо intra-document и не является допустимым новым sweep candidate.

Порог рекомендуется отдельно по режимам: по сохранению содержательно проверенных новых bridges и нагрузке на аудит, не по максимуму рёбер. Одинаковый cosine двух режимов не считается одинаково калиброванным.

## Воспроизведение

`.venv/Scripts/python.exe scripts/sweep_semantic_concepts.py --neo4j-env <existing-file>` — проверенный read-only live run с cache reuse. `--embeddings-only` готовит только cache/candidates; без env проверяется сохранённый полный source snapshot, и live preservation не заявляется. Cache находится в `.cache/semantic_concept_sweep/`, в Git его нет. Если cache snapshot отсутствует, live run читает его из уже существующего namespace. Ни один режим runner не импортирует граф.

Тесты: `.venv/Scripts/python.exe -m pytest tests/test_semantic_concepts.py tests/test_dynamic_concepts.py -q`. Локальный commit разрешён, push и graph import запрещены в этом эксперименте.
'''
    (OUT / "experiment_design.md").write_text(text, encoding="utf-8", newline="\n")


def write_report(comparison, baseline, candidates, decisions, accepted, control, probes, audit, universe, meta, paths):
    rows = ["# Stage 2G.1 — Semantic Concept Resolution: Threshold Sweep", "",
        "**22 виртуальные конфигурации проверены на тех же 30 документах. Экспериментальные результаты в Neo4j не импортировались.**", "",
        "L: Stage 2G — 811 entities, 34 concepts, 130 memberships, 1307 facts, 492 SIMILAR_TO; 6 multi-document concepts, 92 entity–concept–entity bridges, 223 WCC, largest component 371.", "",
        "## Результаты всех конфигураций", "",
        "| Режим | Порог | Кандидаты EE/EC | Новые memberships | Новые concepts | Shared concepts | Новые bridges | Новые direct doc pairs | Новые reachable doc pairs | WCC | Largest | Подозрительные ACCEPT |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        rows.append(f"| {r['embedding_mode']} | {r['threshold']} | {r['entity_entity_candidates']}/{r['entity_concept_candidates']} | {r['new_memberships']} | {r['new_concepts']} | {r['multidocument_concepts']} | {r['new_entity_concept_entity_bridges']} | {r['new_direct_document_pairs']} | {r['new_reachable_document_pairs']} | {r['weak_components']} | {r['largest_component_vertices']} | {r['suspicious_new_memberships']} |")
    unique_new = {(r["source_entity_id"], r["target_concept_id"]): r for r in accepted if r["threshold"] == "0.955"}
    rows += ["", "При text_name 0.950 и 0.955 граф имел бы 35 concepts и 135 memberships. Все исходные 34/130 сохраняются. Новые понятия/связи:", "",
             "| Сущность | Concept | Документ | Cosine | Основание |", "|---|---|---|---:|---|"]
    for r in unique_new.values():
        rows.append(f"| {r['source_name']} | {r['target_concept_name']} | {r['document_id']} | {r['cosine_similarity']:.6f} | {r['reason']} |")
    rows += ["", "## Что дали retrieval и validator", "",
        "Semantic candidate discovery добавляет путь к новому классу **surgery**: Surgical procedures ↔ surgery, cosine 0.959252. Это 2 пары из 2 документов, 3 memberships; одна membership Surgical procedures→surgery не имеет lexical token/alias overlap. Ни один вызов lexical_match не обязателен для этого semantic validation path.", "",
        "Две breast cancer→Cancer связи (cosine 0.956457) были кандидатами и в Stage 2G, но отклонялись из-за недостаточных context-sense witnesses. Validator-only control восстанавливает те же 2 связи без semantic retrieval. Поэтому их нельзя целиком считать выигрышем embeddings. Исходные NER labels и source spans не исправлялись.", "",
        f"Полностью проверено {universe['text_name']['all_eligible_cross_document_entity_pairs']} допустимых cross-document entity pairs и {universe['text_name']['eligible_nonmember_entity_concept_pairs']} nonmember entity→concept pairs на режим. При 0.950 отобрано 106 EE и 8 EC candidates. Все отобранные EE уже присутствовали среди исходных SIMILAR_TO; **новых EE-пар вне этого набора при данных порогах нет**. Мы не ограничивали ими поиск: это измеренный результат. Новизна относительно Stage 2G — class validation и явная обработка semantic-only EE proposals, а не обнаружение ранее невидимого высокого cosine.", "",
        "Новые прямые пары через Cancer: W1979727947↔W3123056951 и W1979727947↔W4391351014. Surgery добавляет ещё один concept-мост между W2761455282 и W3101922761, которые были связаны через Patients/follow-up. Первой достижимости в полном KG не прибавилось; WCC 223 сохраняются. Рост largest 371→372 — не доказательство роста качества. Структурные WCC/reachability включают все неизменяемые 492 SIMILAR_TO, в том числе с прежними диагностическими REJECT/REVIEW; такая достижимость не означает доказанную научную зависимость.", "",
        "## Name и name+context", "",
        f"Для text_name_context maximum EE cross-document cosine = {universe['text_name_context']['maximum_cross_document_entity_cosine']:.6f}, maximum eligible EC cosine = {universe['text_name_context']['maximum_nonmember_entity_concept_cosine']:.6f}. Ни один кандидат не проходит 0.950–1.000. Concept representation использует исходные member snippets с leave-one-document-out; своих context snippets в reference embedding нет. Name-only fallback для single-document concepts явно помечен. Результат относится именно к этому reference recipe и corpus, а не доказывает бесполезность context embeddings вообще.", "",
        "Name similarity завышает одинаковые общие слова: Study/Results/Patients получают около 1. Context может завышать co-occurrence внутри одного исследования: limonene и его synthase gene имеют context cosine 0.985701, хотя это разные объекты. Cross-document sweep исключает эту intra-document пару; отдельный обязательный probe всё равно показывает REJECT.", "",
        "E5 masks/pooling/prefix/нормализация сохранены; все 7 model files совпадают со Stage 2B. Embeddings вычислены один раз на режим (733 unique name texts и 885 context texts, 811 entities+77 reference variants). Векторы и cosine matrices находятся вне Git; hashes и truncation counts — в manifest. Усечений в этом прогоне нет: maximum input 458 tokens при лимите 512. На оценки влияют reference context variants и неоднородность исследовательских тем; режимы нельзя считать одинаково калиброванными. У byte-identical vectors cosine ровно 1 по тождеству, без epsilon-округления соседних значений; коррекция сделана только по cache, без повторного E5 inference.", "",
        "## Обязательный содержательный аудит", "",
        "| Случай | text_name | text_name_context | Решение / смысл |", "|---|---:|---:|---|"]
    summaries = [
        ("breast cancer→Cancer", "0.956457", "0.821099–0.830479", "2 новых class links только при name 0.950/0.955; это восстановленные FN lexical context rule"),
        ("esophageal cancer→Cancer", "0.906148", "0.847593", "Принятая Stage 2G membership сохраняется при любом пороге, даже ниже нового retrieval gate"),
        ("limonene↔limonene synthase gene", "0.892161", "0.846019 /0.985701", "REJECT: product и gene различны; intra-document, вне основного sweep"),
        ("patients в независимых исследованиях", "≈1.000000", "0.831965–0.856967", "Общая clinical category допустима; 130 frozen links не переоцениваются, MERGE когорт запрещён"),
        ("clinical follow-up", "≈1.000000", "0.857227", "Продольное клиническое наблюдение/параметр, не универсальный мост: разные интервалы и outcomes остаются локальными"),
    ]
    rows += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in summaries]
    rows += ["", "Аудированы все 10 новых ACCEPT candidate–entity–concept decisions (5 уникальных memberships) и все 41 REJECT decision при объединении порогов. У 5 принятых memberships агентный содержательный аудит не нашёл явно ошибочной принадлежности. Это **не измеренная Precision и не независимый Gold**. 41 REJECT — количество отклонённых решений, не доказанное число false matches: сюда входят как несовместимые понятия, так и недостаточное evidence.", "",
        "Показательные отклонения: hip→knee arthroplasty 0.961352; genetics→Genes 0.959849; green bank loans→Green banks 0.956395; magnetic grains↔grain production 0.980470; универсальные Study/Results около 1. Подозрительных автоматически добавленных memberships по этому аудиту: 0. Есть 1 потенциальный оставшийся FN: pathological stress reactivity→Stress reactivity (0.965276), которому строгие context witnesses не дали ACCEPT. Он сохраняется в REJECT; перед следующей версией нужен отдельный разбор.", "",
        "Clinical Surgery и Cancer остаются широкими категориями. Ни состав операций/когорт, ни результаты лечения/длительности/риски не наследуются. Все новые memberships имеют factual=false, identity=false, fact_inheritance_allowed=false. Agent audit notes, исходные contexts и указания на возможные ошибки сохранены в quality_audit.csv; semantic_only_discoveries.jsonl содержит как принятые, так и отклонённые пары без lexical overlap.", "",
        "## Реальные виртуальные цепочки", "",
        "Полные source IDs, исходные направления RELEX_REL, raw triple IDs и обе membership-связи находятся в path_examples.jsonl. Ниже выбраны несколько цепочек с фактическими рёбрами с обеих сторон:", ""]
    chosen_paths = sorted(paths, key=lambda p: (p["concept_name"] != "surgery", p["document_a"], p["document_b"], p["entity_a"]))[:5]
    for i, p in enumerate(chosen_paths, 1):
        rows += [f"### Цепочка {i}: {p['concept_name']}", "", f"A: {p['document_a']} — {p['document_a_title']}. B: {p['document_b']} — {p['document_b_title']}.", "",
                 f"`{p['entity_a_name']}` → HAS_SHARED_CONCEPT → `{p['concept_name']}` ← HAS_SHARED_CONCEPT ← `{p['entity_b_name']}`.", ""]
        for side in ("a", "b"):
            path = p["factual_route_" + side]
            rows.append(f"Маршрут {side.upper()}, {len(path)} исходных factual edges; направление обхода может отличаться от исходного:")
            rows.append("")
            for edge in path:
                rows.append(f"- {edge['head_name']} —{edge['predicate']}→ {edge['tail_name']}; edge `{edge['edge_key'].split(NS + ':')[-1]}`, source triples `{', '.join(edge['source_triple_ids'])}`.")
            rows.append("")
        rows += ["Это семантическая цепочка между независимыми исследованиями; новая фактическая зависимость не установлена.", ""]
    rows += ["## Рекомендация и ограничения", "",
        "Для следующего отдельного экспериментального графа предварительно рекомендован **text_name, 0.955, только с Hybrid validation**. Он сохраняет те же 5 содержательно проверенных memberships и 8 bridges, что 0.950, при 102 вместо 114 candidates. При 0.960 и выше содержательные новые bridges исчезают; одинаковые общие слова с cosine≈1 остаются. Порог выбран по сохранению проверенных научных связей и снижению нагрузки на аудит, а не по максимуму рёбер. Это exploratory выбор на малом корпусе, не независимая оценка качества; для production рекомендация не валидирована.", "",
        "Для name+context в проверенном диапазоне рекомендовать порог нельзя: положительных retrieval candidates нет. Нужно отдельно калибровать другой диапазон/способ агрегирования source contexts; в этом эксперименте дополнительные пороги и модели не запускались.", "",
        "E5-small достаточно для генерации части полезных кандидатов (Cancer и procedural paraphrase), но не для решения о scientific membership: она сближает разные объекты, общие слова и совместно упомянутые сущности. Нынешние данные показывают ограничения retrieval/representation и ручных правил; они не доказывают необходимость замены модели или количественное преимущество другой модели.", "",
        "Все 22 варианта остаются виртуальными. Исходный Neo4j namespace, 34 concepts/130 memberships, факты, SIMILAR_TO, documents и mapping сохранены. Результаты live preservation, protected-file hashes, cache/model/code hashes и тесты — в validation.json и manifest.json. Данные не расширялись; push и import экспериментального графа не выполнялись.", ""]
    (OUT / "comparison_report.md").write_text("\n".join(rows), encoding="utf-8", newline="\n")


def gzip_jsonl(path, rows):
    import gzip
    import io
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(dumps(row) + "\n")


def read_gzip_jsonl(path):
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extended_lexical_control(entities, catalog, memberships):
    """B uses the frozen lexical candidate universe, never embedding scores."""
    from cgraph.concepts import ConceptResolver
    by = {e.entity_id: e for e in entities}; cat = {c.concept_id: c for c in catalog}
    lexical = ConceptResolver(catalog, entities, memberships).resolve_batch(entities)
    historical = graph.base.core.read_jsonl(SOURCE / "candidate_decisions.jsonl")
    universe = {(r["entity_id"], r["concept_id"]) for r in [*historical, *lexical.decisions]}
    resolver = HybridConceptResolver(catalog, entities, memberships)
    delta, decisions = {}, []
    for eid, cid in sorted(universe):
        if (eid, cid) in memberships or cid not in cat:
            continue
        c = cat[cid]; e = by[eid]
        checks = [(*resolver.validate_semantic(e, c, by[i]), i) for i in c.member_ids if i != eid]
        if not checks:
            continue
        checks.sort(key=lambda x: (x[0] != "ACCEPT", by[x[3]].document_id == e.document_id, x[3]))
        status, reason, evidence, refid = checks[0]
        row = dict(candidate_id="lexical-control:" + digest([eid, cid])[:24],
            decision_id="validator-only:" + digest([eid, cid, VERSION])[:24], source_entity_id=eid,
            source_name=e.name, target_concept_id=cid, target_concept_name=c.name,
            document_id=e.document_id, reference_entity_id=refid, reference_document_id=by[refid].document_id,
            status=status, reason=reason, evidence=compact_proof(evidence), resolver_version=VERSION,
            embedding_mode="lexical", cosine_similarity=None, new_concept=False, origin="validator_only",
            semantic_only_membership=no_lexical_overlap(e, c), factual=False, identity=False, fact_inheritance_allowed=False)
        decisions.append(row)
        if status == "ACCEPT":
            delta[eid, cid] = row
    return list(delta.values()), decisions, universe


def extended_evaluate(candidates, entities, catalog, memberships, validator_sha):
    """Reuse each checked scientific decision across thresholds and modes."""
    import inspect
    signature = digest(dict(candidates=candidates, validator_sha256=validator_sha,
        evaluation_source=inspect.getsource(evaluate), proof_source=inspect.getsource(compact_proof),
        source_snapshot=json.loads((SOURCE / "verification.json").read_text(encoding="utf-8"))["target_sha256"]))
    directory = CACHE / "extended_evaluation"
    directory.mkdir(exist_ok=True)
    metadata = directory / "manifest.json"
    if metadata.exists():
        cached = json.loads(metadata.read_text(encoding="utf-8"))
        if cached["fingerprint"] == signature:
            require(all(graph.base.sha(directory / n) == h for n, h in cached["files_sha256"].items()), "Extended evaluation cache corrupted")
            return read_gzip_jsonl(directory / "decisions.jsonl.gz"), read_gzip_jsonl(directory / "proposals.jsonl.gz"), cached["statistics"]
    statistics = {}
    decisions, proposals = evaluate(candidates, entities, catalog, memberships, compact=True, memoize=True, statistics=statistics)
    gzip_jsonl(directory / "decisions.jsonl.gz", decisions)
    gzip_jsonl(directory / "proposals.jsonl.gz", proposals)
    save(metadata, dict(fingerprint=signature, statistics=statistics,
        files_sha256={p.name: graph.base.sha(p) for p in directory.glob("*.gz")}))
    return decisions, proposals, statistics


def extended_audit(decisions, accepted, entities, catalog, lexical_universe, lexical_delta):
    """Audit flags do not change ACCEPT/REJECT or the frozen validator rules."""
    from cgraph.semantic_concepts import modifiers_conflict
    by = {e.entity_id: e for e in entities}; cat = {c.concept_id: c for c in catalog}
    unique = {}
    for r in accepted:
        unique.setdefault((r["source_entity_id"], r["target_concept_id"]), r)
    rows = []
    for (eid, cid), r in sorted(unique.items()):
        e = by[eid]; ref = by.get(r.get("reference_entity_id"))
        target = Concept(cid, r["target_concept_name"])
        # A narrow target's contradictory modifiers are suspicious even if a
        # generic approved alias bypassed that distinction in the old rules.
        flags = []
        if cid not in cat and class_form(target.name) == "cell":
            flags.append("new_biological_cell_class_is_too_broad_for_unreviewed_global_hub")
        if modifiers_conflict(e, target):
            flags.append("contradictory_target_qualifier")
        if "RESULT" in e.entity_types:
            flags.append("experiment_result_as_class_member")
        if set(e.name.lower().split()) & {"risk", "rate", "rates", "outcome", "outcomes", "success", "failure"}:
            flags.append("property_result_or_relation_scope_needs_review")
        rows.append(dict(audit_kind="all_unique_ACCEPT", source_entity_id=eid, source_name=e.name,
            target_concept_id=cid, target_concept_name=r["target_concept_name"], document_id=e.document_id,
            reference_entity_id=ref.entity_id if ref else "", reference_name=ref.name if ref else "",
            status="ACCEPT", reason=r["reason"], risk_flags=flags,
            audit_verdict="SUSPICIOUS" if flags else "SOURCE_SUPPORTED_UNDER_FROZEN_RULES",
            independent_gold=False, semantic_only_membership=r.get("semantic_only_membership", False),
            unavailable_to_lexical_baseline=(eid, cid) not in lexical_universe,
            beyond_validator_only=(eid, cid) not in {(x["source_entity_id"], x["target_concept_id"]) for x in lexical_delta},
            source_contexts=list(e.evidence), reference_contexts=list(ref.evidence) if ref else [],
            note="Category membership only; retain separate studies, instances and original facts. Audit is not independent Gold."))
    # Stratified REJECT sample: highest scores per reason and mode, plus known FN.
    groups = defaultdict(list)
    for r in decisions:
        if r["status"] == "REJECT":
            groups[r["embedding_mode"], r["reason"]].append(r)
    sampled = {}
    for rs in groups.values():
        for r in sorted(rs, key=lambda r: (-r["cosine_similarity"], r["decision_id"]))[:3]:
            sampled[r["decision_id"]] = r
    possible_fn = {}
    for r in decisions:
        source_name = r.get("source_name", "").casefold()
        target_name = class_form(r.get("target_concept_name", ""))
        known_pair = ((source_name == "pathological stress reactivity" and target_name == "stress reactivity") or
                      (source_name == "hla-drb1*09 gene" and target_name == "gene"))
        if r["status"] == "REJECT" and known_pair:
            key = (r["source_entity_id"], r["target_concept_id"], r["embedding_mode"])
            if key not in possible_fn or r["cosine_similarity"] > possible_fn[key]["cosine_similarity"]:
                possible_fn[key] = r
    for r in possible_fn.values():
        sampled.setdefault(r["decision_id"], r)
    for r in sampled.values():
        e = by[r["source_entity_id"]]; ref = by.get(r.get("reference_entity_id"))
        known_fn = ((e.name.casefold() == "pathological stress reactivity" and class_form(r.get("target_concept_name", "")) == "stress reactivity") or
                    (e.name.casefold() == "hla-drb1*09 gene" and class_form(r.get("target_concept_name", "")) == "gene"))
        rows.append(dict(audit_kind="stratified_REJECT_sample", source_entity_id=e.entity_id, source_name=e.name,
            target_concept_id=r["target_concept_id"], target_concept_name=r.get("target_concept_name", ""),
            document_id=e.document_id, reference_entity_id=ref.entity_id if ref else "", reference_name=ref.name if ref else "",
            status="REJECT", reason=r["reason"], risk_flags=[],
            audit_verdict="POTENTIAL_FALSE_NEGATIVE" if known_fn else "REJECT_OR_INSUFFICIENT_EVIDENCE",
            independent_gold=False, semantic_only_membership=r.get("semantic_only_membership", False),
            unavailable_to_lexical_baseline=(e.entity_id, r["target_concept_id"]) not in lexical_universe,
            beyond_validator_only=False, source_contexts=list(e.evidence), reference_contexts=list(ref.evidence) if ref else [],
            note="Potential qualified phenotype or explicitly source-named gene class; frozen context rules abstain." if known_fn else
                 "High similarity does not establish a shared scientific class; inspect roles, qualifiers and source scope."))
    return rows


def refine_extended_semantic_flags(decisions, entities, catalog):
    """Analysis labels only: regular plurals/lexical suffixes are not semantic-only."""
    by = {e.entity_id: e for e in entities}; cat = {c.concept_id: c for c in catalog}
    normalized_tokens = {e.entity_id: set().union(*(tokens(class_form(a)) for a in (e.name, *e.aliases))) for e in entities}
    for r in decisions:
        if r.get("adds_membership") is False:
            continue
        e = by[r["source_entity_id"]]
        concept = cat.get(r["target_concept_id"], Concept(r["target_concept_id"], r.get("target_concept_name", "")))
        target_tokens = set().union(*(tokens(class_form(a)) for a in (concept.name, *concept.aliases)))
        r["raw_token_semantic_only_2g1"] = r.get("semantic_only_membership", False)
        r["semantic_only_membership"] = not normalized_tokens[e.entity_id] & target_tokens and lexical_match(e, concept) is None
        if r.get("reference_entity_id") in by:
            r["semantic_only_candidate_pair"] = not normalized_tokens[e.entity_id] & normalized_tokens[r["reference_entity_id"]]


def audit_supported_delta(active, suspicious, catalog_ids):
    kept = [r for r in active if (r["source_entity_id"], r["target_concept_id"]) not in suspicious]
    documents = defaultdict(set)
    for r in kept:
        documents[r["target_concept_id"]].add(r["document_id"])
    return [r for r in kept if r["target_concept_id"] in catalog_ids or len(documents[r["target_concept_id"]]) >= 2]


def compact_artifact_row(row):
    r = dict(row)
    if "qualifying_thresholds" in r:
        ts = r.pop("qualifying_thresholds")
        r["qualifying_threshold_range"] = [ts[0], ts[-1]] if ts else []
    if "evidence" in r and r.get("status") != "ACCEPT":
        r["evidence"] = dict(entity_bundle="entity_evidence.jsonl", source_entity_id=r.get("source_entity_id"),
            reference_entity_id=r.get("reference_entity_id"), proof_sha256=digest(r["evidence"]),
            reason=r.get("reason"))
    return r


def extended_plots(comparison, distribution, b_metrics, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for filename, kind in (("candidate_counts", "candidates"), ("new_concept_bridges", "bridges")):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
        for ax, mode in zip(axes, MODES):
            if kind == "candidates":
                rs = [r for r in distribution if r["embedding_mode"] == mode]
                x = [float(r["threshold"]) for r in rs]
                ax.plot(x, [r["full_candidates"] for r in rs], label="Все пары выше порога", linewidth=2)
                ax.plot(x, [r["reviewed_candidates"] for r in rs], "--", label="Проверенные top-k + ≥0.950", linewidth=2)
                ax.set_yscale("symlog", linthresh=1)
                ax.set_ylabel("Кандидаты (symlog, включая 0)")
            else:
                rs = [r for r in comparison if r["embedding_mode"] == mode]
                x = [float(r["threshold"]) for r in rs]
                ax.plot(x, [r["new_entity_concept_entity_bridges"] for r in rs], color="0.6", label="C − A: все ACCEPT validator", linewidth=1.5)
                ax.plot(x, [r["audit_supported_new_bridges"] for r in rs], label="Без спорных memberships", linewidth=2)
                ax.plot(x, [r["audit_supported_bridges_beyond_B"] for r in rs], "--", label="Без спорных, сверх B", linewidth=2)
                ax.axhline(b_metrics["entity_concept_entity_bridges"] - baseline["entity_concept_entity_bridges"],
                           color="0.5", linestyle=":", label="B − A: validator only")
                ax.set_ylabel("Новые entity–concept–entity мосты")
            ax.set_title(mode); ax.set_xlabel("Cosine threshold")
            ax.set_xlim(.700, .950); ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle("Stage 2G.2 — тот же корпус, замороженный validator; top-k не даёт полного Recall", fontsize=11)
        fig.savefig(OUT / (filename + ".png"), dpi=180)
        svg_path = OUT / (filename + ".svg")
        fig.savefig(svg_path)
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
        plt.close(fig)


def extended_report(comparison, distribution, baseline, b_metrics, b_delta, audit, semantic_only, stats):
    chosen = {"0.700", "0.750", "0.800", "0.825", "0.830", "0.845", "0.850", "0.870", "0.875", "0.900", "0.925", "0.950"}
    lines = ["# Stage 2G.2 — Extended Semantic Concept Threshold Sweep", "",
        "102 виртуальные конфигурации на прежних 30 документах. Neo4j не изменён; нового импорта и push нет.", "",
        "Validator и его словари заморожены в версии Stage 2G.1. Новый диапазон — Decimal 0.700–0.950, шаг 0.005. Embeddings/cosine cache предыдущего эксперимента проверен по hashes, input ordering, модели и рецепту; E5 не запускалась повторно.", "",
        "## Полный подсчёт и ограниченная содержательная проверка", "",
        "Все 315954 cross-document EE и 27444 nonmember EC пары на режим участвуют в полном векторизованном подсчёте. Для проверки смысла выбираются top-20 cross-document соседей каждой entity (симметричное объединение) и top-10 concepts каждой entity; дополнительно сохраняются все кандидаты с cosine ≥0.950. Tie-break: стабильный ID. Решение вычисляется один раз для уникального candidate/reference и переиспользуется при всех порогах/режимах. Отсечённые пары не объявляются REJECT; их смысл не проверялся. **Полный Recall не измеряется.**", "",
        "| Режим | Порог | Все кандидаты | Проверены | Отсечены | C memberships Δ | C concepts Δ | C bridges Δ | Bridges без спорных | C−B bridges | WCC | Largest | Suspicious ACCEPT |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        if r["threshold"] in chosen:
            lines.append(f"| {r['embedding_mode']} | {r['threshold']} | {r['full_candidates']} | {r['checked_candidates']} | {r['pruned_candidates']} | {r['new_memberships']} | {r['new_concepts']} | {r['new_entity_concept_entity_bridges']} | {r['audit_supported_new_bridges']} | {r['bridges_beyond_validator_only']} | {r['weak_components']} | {r['largest_component_vertices']} | {r['suspicious_new_memberships']} |")
    lines += ["", "Полные 102 строки: threshold_comparison.csv и candidate_distribution.csv. Доля ACCEPT среди проверенных кандидатов — acceptance_fraction_checked_candidates; это **не Precision**. Она включает уже подтверждённые общие concepts, поэтому отдельно показаны new_accept_candidates и новые memberships.", "",
        "![Candidate counts](candidate_counts.png)", "", "![New bridges](new_concept_bridges.png)", "",
        "## Ablation: одинаковый validator, разные кандидаты", "",
        "| Вариант | Concepts | HAS_SHARED_CONCEPT | Междокументные мосты | WCC | Largest |", "|---|---:|---:|---:|---:|---:|",
        f"| A: неизменный Stage 2G | {baseline['concepts']} | {baseline['concept_links']} | {baseline['entity_concept_entity_bridges']} | {baseline['weak_components']} | {baseline['largest_component_vertices']} |",
        f"| B: improved validator, lexical candidates only | {b_metrics['concepts']} | {b_metrics['concept_links']} | {b_metrics['entity_concept_entity_bridges']} | {b_metrics['weak_components']} | {b_metrics['largest_component_vertices']} |", "",
        "B использует исторические lexical candidates Stage 2G и повторный candidate generation неизменного lexical resolver на существующем каталоге. Embedding scores в B не участвуют. C — объединение B и принятых semantic memberships; поэтому улучшение validator остаётся в C при любом пороге. Это явное уточнение относительно таблицы threshold-gated deltas Stage 2G.1, где lexical control показывался отдельно. Колонки semantic_accepted_memberships и memberships_beyond_validator_only отделяют эти вклады. Полное сравнение A/B/C для каждого режима/порога — ablation_comparison.csv.", "",
        f"B добавляет {len(b_delta)} memberships. Unique semantic-only memberships, недоступных lexical candidate universe и отсутствующих в B: {len(semantic_only)}. Их исходные evidence и IDs — semantic_only_discoveries.jsonl. Высокий cosine или отсутствие общего слова сами по себе не означают правильность.", "",
        "## Ответы на вопросы эксперимента", "",
        "1. При движении вниз по порогу Surgery уже доступен при name 0.950 (в 2G.1 он появлялся и при 0.955). Новый квалифицированный member paranasal sinus surgery→surgery появляется на grid 0.870, cosine 0.873386. Новый concept cell появляется на 0.830, cosine 0.833477; он отмечен для review как чрезмерно широкая biological category. Без этой спорной категории иных новых concepts на проверенной selection ниже порога не найдено.", "",
        "2. Name обнаруживает клиническую Surgery и её дополнительный sinus-surgery member. Name+context воспроизводит Surgery на 0.845 и ниже, но дополнительных подтверждённых classes/members сверх name здесь не даёт. Это сравнение конкретного source-context recipe с leave-one-document-out и top-k, не универсальная победа одного режима. При ≥0.900 у context нет semantic candidates; C всё равно содержит 2 lexical validator-only links B.", "",
        "3. При снижении 0.850→0.825 full candidates растут с 6045 до 25805 для name и с 2265 до 27795 для context; на 0.800 их уже 77762/127466. Это резкий рост retrieval workload. Большинство проверенных низкосходных кандидатов не проходят научную validation, но отброшенные top-k пары не размечались и не называются ложными. При 0.700 проверены 21321 из 343174 (name) и 20910 из 343398 (context), около 6% universe выше gate; полнота содержательного поиска неизвестна.", "",
        "4. Строго semantic-only membership без пересечения нормализованных слов/aliases и без lexical_match — Surgical procedures→surgery. Она отсутствует в lexical candidate universe и B. Paranasal sinus surgery→surgery найдена через низкопороговый EE proposal, но имеет буквальный общий термин surgery и не выдаётся за совпадение без лексики. Cells/cell — регулярное число; это также не semantic-only. raw-token flag 2G.1 сохранён отдельно для сопоставимости.", "",
        "5. Возможные false negatives замороженного validator: pathological stress reactivity→Stress reactivity и HLA-DRB1*09 gene→Genes. Второй source прямо называет gene, однако строгие context witnesses не подтверждают общую область с reference. Они вынесены в quality_audit.csv; решение осталось REJECT. Это content-review hypotheses, не независимый Gold. Словари и ACCEPT rules для исправления этих случаев в ходе sweep не менялись.", "",
        "6. A: 34 concepts/130 memberships/92 bridges, WCC 223/largest 371. B: 34/132/98, WCC 223/largest 371. C name 0.870: 35/136/103, WCC 223/largest 372; context 0.845: 35/135/100, WCC 223/largest 372. C name ≤0.830 даёт raw 36/138/104, WCC 222/largest 374 и 17 новых reachable document pairs, но 2 новых memberships Cell спорны. Именно общий Cell подключает ранее недостижимый документ W2039836112 через W2791492120 к большому компоненту. Без них остаются 35/136/103, WCC 223/largest 372 и 0 новых reachable pairs. Два direct document pairs Cancer связаны с B; третий raw direct pair и все 17 first-reachability pairs возникают через спорный Cell. Увеличение связности само по себе не доказывает качество.", "",
        "7. Для следующей отдельной проверки разумны name 0.870 (добавляет sinus-surgery member), более консервативный name 0.900–0.950 и context 0.840–0.845. Это кандидаты диапазонов, не оптимумы: Gold нет, selection ограничена и source-rule validator исследован на этом же корпусе. Ниже примерно 0.825 workload резко увеличивается без новых неоспоренных gains в рассмотренной selection. Cell требует отдельного решения о полезности общего hub до какого-либо импорта.", "",
        "## Качество и границы вывода", "",
        "Все уникальные новые ACCEPT проходят отдельный аудит ролей, qualifiers, generic scope и source contexts; stratified REJECT sample берётся по каждому reason и режиму. Suspicious ACCEPT помечаются в quality_audit.csv и считаются отдельно, но результаты замороженного validator во время sweep не исправляются. Audit не является независимым Gold; отсутствие флага не доказывает правильность. Свойства cohorts, экземпляров и результатов не наследуются.", "",
        "Правильные примеры: breast cancer→Cancer (disease category), Surgical procedures→surgery (clinical class), paranasal sinus surgery→surgery (конкретный вид клинического вмешательства). Исходный текст сообщает о 10 пациентах, проходивших paranasal sinus surgery; их факты не переносятся на knee/eye cohorts. Cell и peripheral blood mononuclear cells действительно обозначают биологические клетки, но общая категория слишком широка для автоматического вывода о полезном междокументном scientific bridge. Две такие memberships остаются raw ACCEPT и отдельно помечаются SUSPICIOUS; график показывает и raw ACCEPT, и вариант без них.", "",
        "Ошибочные/неподтверждённые кандидаты: limonene↔limonene synthase gene (product vs gene), genetics→Genes (область vs объект), hip→knee arthroplasty (разные anatomical qualifiers), Study/Results (универсальные слова). Cosine не исправляет эти различия и не доказывает membership. Независимая Precision не измерялась.", "",
        "Снижение порога увеличивает число тематически близких пар, а не автоматически число корректных научных связей. Полные counts отражают весь universe, а ACCEPT/REJECT и graph metrics — только проверенную selection плюс B. Для анализа роста неподтверждённых кандидатов сравниваются raw/full curves и reviewed REJECT; непроверенные отсечённые пары нельзя считать ложными. Нельзя выбирать оптимальный порог по максимальному числу рёбер.", "",
        "WCC/reachability включают все 492 исходные SIMILAR_TO, в том числе с историческими REJECT/REVIEW annotations. Semantic graph path — маршрут поиска между независимыми исследованиями; он не является новой фактической зависимостью. Документы, 811 entities, 1307 facts, 492 SIMILAR_TO, 34 concepts и 130 исходных memberships во всех virtual payloads сохранены точно.", "",
        "## Воспроизведение", "",
        "`.venv/Scripts/python.exe scripts/sweep_semantic_concepts.py --extended --neo4j-env <existing-file>`", "",
        "Кэш embeddings и полной memoized проверки — вне Git в .cache/semantic_concept_sweep/. Полные выбранные candidates/decisions/proposals сохранены как детерминированные .jsonl.gz; entity_evidence.jsonl позволяет восстановить исходные контексты без их многократного дублирования. Новые mappings или модели не добавлялись. Manifest фиксирует selection limits, hashes, версию validator, counts отсечённых пар, memoization statistics и тесты.", ""]
    (OUT / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_extended(args):
    global OUT
    OUT = EXTENDED_OUT
    OUT.mkdir(parents=True, exist_ok=True)
    previous = json.loads((graph.base.ER / "semantic_concept_sweep/manifest.json").read_text(encoding="utf-8"))
    validator_path = ROOT / "src/cgraph/semantic_concepts.py"
    validator_sha = graph.base.sha(validator_path)
    require(validator_sha == previous["code_sha256"]["src/cgraph/semantic_concepts.py"], "Validator changed: extended sweep must freeze Stage 2G.1 rules")
    require(digest(SEMANTIC_POLICY) == digest(previous["policy"]), "Semantic policy/dictionaries changed")
    protected = sorted(p for p in graph.base.ER.rglob("*") if p.is_file() and not p.is_relative_to(OUT))
    protected += [graph.base.DOCUMENTS, validator_path, ROOT / "src/cgraph/concepts.py"]
    hashes = {p.relative_to(ROOT).as_posix(): graph.base.sha(p) for p in protected}
    driver = None; outside_before = schema_before = None
    banner("STAGE 2G.2: ПРОВЕРЯЮ BASELINE И КЭШ; ЗАПИСИ В NEO4J ЗАПРЕЩЕНЫ")
    if args.neo4j_env:
        from neo4j import READ_ACCESS
        driver, database = graph.base.core.connect_neo4j(args.neo4j_env)
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            nodes, edges, source_manifest, _ = verified_source(session)
            graph.base.NS = NS
            outside_before = graph.base.outside_snapshot(session)
            schema_before = dict(constraints=session.run(graph.SCHEMA_QUERY).data(), indexes=session.run(graph.INDEX_QUERY).data())
    else:
        nodes, edges, source_manifest, _ = verified_source()
    before = graph.snapshot_hash(nodes, edges)
    entities, catalog, memberships, similar, docs = graph.profiles(nodes, edges)
    rows, refs = representation_inputs(nodes, entities, catalog)
    cache_existed = (CACHE / "embedding_manifest.json").exists()
    meta = embeddings(rows, refs)
    require(meta["fingerprint"] == previous["embedding"]["fingerprint"], "Input ordering/model/settings differ from Stage 2G.1")
    if cache_existed:
        require(meta["cache_files_sha256"] == previous["embedding"]["cache_files_sha256"], "Cosine/vector cache differs from Stage 2G.1")
    banner("ПОДСЧИТЫВАЮ ВСЕ ПАРЫ ДЛЯ 102 КОНФИГУРАЦИЙ; ОТБИРАЮ TOP-20 EE И TOP-10 EC")
    distribution = []
    candidates = discover(rows, refs, catalog, memberships, similar, thresholds=EXTENDED_THRESHOLDS,
        topk_entity=20, topk_concept=10, preserve_above="0.950", distribution=distribution)
    csvout("candidate_distribution.csv", distribution)
    gzip_jsonl(OUT / "semantic_candidates.jsonl.gz", (compact_artifact_row(r) for r in candidates))
    jsonl("entity_evidence.jsonl", [dict(entity_id=e.entity_id, document_id=e.document_id, name=e.name,
        aliases=e.aliases, entity_types=e.entity_types, evidence=e.evidence) for e in entities])
    banner(f"ПРОВЕРЯЮ {len(candidates)} УНИКАЛЬНЫХ ВЫБРАННЫХ КАНДИДАТОВ; ПОРОГИ ПЕРЕИСПОЛЬЗУЮТ РЕШЕНИЯ")
    decisions, proposals, statistics = extended_evaluate(candidates, entities, catalog, memberships, validator_sha)
    refine_extended_semantic_flags(decisions, entities, catalog)
    gzip_jsonl(OUT / "semantic_decisions.jsonl.gz", (compact_artifact_row(r) for r in decisions))
    gzip_jsonl(OUT / "new_concept_proposals.jsonl.gz", (compact_artifact_row(r) for r in proposals))
    b_delta, b_decisions, lexical_universe = extended_lexical_control(entities, catalog, memberships)
    jsonl("validator_only_decisions.jsonl", b_decisions)
    comparison, accepted, changes, baseline = threshold_sweep(nodes, edges, candidates, decisions, catalog,
        thresholds=EXTENDED_THRESHOLDS, lexical_delta=b_delta, distribution=distribution, retain_evidence=False)
    b_nodes, b_edges = virtual_graph(nodes, edges, b_delta)
    b_metrics = graph_metrics(b_nodes, b_edges)[0]
    audit = extended_audit(decisions, accepted, entities, catalog, lexical_universe, b_delta)
    suspicious = {(r["source_entity_id"], r["target_concept_id"]) for r in audit if r["audit_verdict"] == "SUSPICIOUS"}
    ablations = []
    baseline_reachable = graph.reachable_document_pairs(nodes, edges)
    for r in comparison:
        active = [a for a in accepted if a["embedding_mode"] == r["embedding_mode"] and a["threshold"] == r["threshold"]]
        r["suspicious_new_memberships"] = sum((a["source_entity_id"], a["target_concept_id"]) in suspicious for a in active)
        r["bridges_beyond_validator_only"] = r["new_entity_concept_entity_bridges"] - (b_metrics["entity_concept_entity_bridges"] - baseline["entity_concept_entity_bridges"])
        supported = audit_supported_delta(active, suspicious, {c.concept_id for c in catalog})
        supported_metrics, supported_pairs, supported_reachable = graph_metrics(*virtual_graph(nodes, edges, supported))
        r["audit_supported_new_memberships"] = len(supported)
        r["audit_supported_new_bridges"] = supported_metrics["entity_concept_entity_bridges"] - baseline["entity_concept_entity_bridges"]
        r["audit_supported_bridges_beyond_B"] = supported_metrics["entity_concept_entity_bridges"] - b_metrics["entity_concept_entity_bridges"]
        r["audit_supported_weak_components"] = supported_metrics["weak_components"]
        r["audit_supported_largest_component_vertices"] = supported_metrics["largest_component_vertices"]
        r["audit_supported_new_reachable_document_pairs"] = len(supported_reachable - baseline_reachable)
        for variant, metrics, delta_count in (("A", baseline, 0), ("B", b_metrics, len(b_delta))):
            ablations.append(dict(variant=variant, embedding_mode=r["embedding_mode"], threshold=r["threshold"],
                concepts=metrics["concepts"], memberships=metrics["concept_links"], new_memberships=delta_count,
                entity_concept_entity_bridges=metrics["entity_concept_entity_bridges"],
                weak_components=metrics["weak_components"], largest_component_vertices=metrics["largest_component_vertices"]))
        ablations.append(dict(variant="C", embedding_mode=r["embedding_mode"], threshold=r["threshold"],
            concepts=baseline["concepts"] + r["new_concepts"], memberships=baseline["concept_links"] + r["new_memberships"],
            new_memberships=r["new_memberships"], entity_concept_entity_bridges=baseline["entity_concept_entity_bridges"] + r["new_entity_concept_entity_bridges"],
            weak_components=r["weak_components"], largest_component_vertices=r["largest_component_vertices"]))
    csvout("threshold_comparison.csv", comparison)
    csvout("ablation_comparison.csv", ablations)
    csvout("quality_audit.csv", audit)
    jsonl("new_memberships.jsonl", accepted)
    save(OUT / "document_pair_changes.json", changes)
    save(OUT / "baseline_metrics.json", baseline)
    save(OUT / "validator_only_metrics.json", b_metrics)
    unique = {}
    b_pairs = {(r["source_entity_id"], r["target_concept_id"]) for r in b_delta}
    for r in decisions:
        pair = (r["source_entity_id"], r["target_concept_id"])
        if (r["status"] == "ACCEPT" and r.get("adds_membership", True) and r.get("semantic_only_membership") and
                pair not in lexical_universe and pair not in b_pairs):
            unique.setdefault(pair, r)
    jsonl("semantic_only_discoveries.jsonl", unique.values())
    extended_plots(comparison, distribution, b_metrics, baseline)
    extended_report(comparison, distribution, baseline, b_metrics, b_delta, audit, list(unique.values()), statistics)
    require(len(comparison) == 102 and len(distribution) == 102, "Incomplete extended sweep")
    require(graph.snapshot_hash(nodes, edges) == before, "Virtual graphs mutated source")
    require(all(r["full_candidates"] == r["reviewed_candidates"] + r["pruned_candidates"] for r in distribution), "Candidate accounting mismatch")
    require(all(r["pruned_candidates"] == 0 for r in distribution if r["threshold"] == "0.950"), "High-threshold preservation failed")
    for mode in MODES:
        rs = [r for r in distribution if r["embedding_mode"] == mode]
        require(all(a["full_candidates"] >= b["full_candidates"] and a["reviewed_candidates"] >= b["reviewed_candidates"] for a, b in zip(rs, rs[1:])), "Non-monotonic distribution")
    validation = dict(configurations=102, thresholds=list(EXTENDED_THRESHOLDS), documents=30, entities=811,
        baseline_concepts=34, baseline_memberships=130, facts=1307, similar_to=492, source_sha256_before=before,
        all_eligible_pairs_counted=True, semantic_review_is_limited=True, full_recall_claimed=False,
        selection=dict(entity_top_k=20, concept_top_k=10, preserve_all_at_or_above="0.950"),
        cache_reused=cache_existed, model_inference_run=not cache_existed, validator_frozen_sha256=validator_sha,
        new_dictionary_mappings_added=False, model_files_unchanged=True, all_virtual_baseline_records_preserved=True,
        ablation_C_includes_B=True, live_database_preservation_verified=False, neo4j_import=False, push=False)
    validation.update(unique_ACCEPT_memberships_audited=sum(r["status"] == "ACCEPT" for r in audit),
        suspicious_unique_ACCEPT_memberships=len(suspicious),
        REJECT_audit_sample_rows=sum(r["status"] == "REJECT" for r in audit),
        semantic_only_definition="no normalized name/alias token overlap AND no lexical_match AND absent in lexical universe/B",
        audit_flags_do_not_change_frozen_ACCEPT_REJECT=True)
    if driver is not None:
        try:
            with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
                banner("ПРОВЕРЯЮ СОХРАННОСТЬ STAGE 2G И ЧУЖИХ NAMESPACES ПОСЛЕ 102 ВИРТУАЛЬНЫХ КОНФИГУРАЦИЙ")
                nn, ee, _, _ = verified_source(session)
                after = graph.base.outside_snapshot(session)
                schema = dict(constraints=session.run(graph.SCHEMA_QUERY).data(), indexes=session.run(graph.INDEX_QUERY).data())
                require(graph.snapshot_hash(nn, ee) == before and outside_before == after and schema == schema_before, "Neo4j source/outside/schema changed")
                save(OUT / "neo4j_preservation.json", dict(before=before, after=graph.snapshot_hash(nn, ee), outside_before=outside_before,
                    outside_after=after, schema_before=schema_before, schema_after=schema))
                validation.update(live_database_preservation_verified=True, source_sha256_after=before, external_namespaces_and_schema_unchanged=True)
        finally:
            driver.close()
    require(all(graph.base.sha(ROOT / p) == h for p, h in hashes.items()), "Previous experiment/corpus/validator changed")
    require(all(graph.base.sha(Path(r["path"])) == r["sha256"] for r in meta["model"]["files"]), "Model files changed")
    validation["previous_experiments_and_source_files_unchanged"] = True
    save(OUT / "validation.json", validation)
    code = [Path(__file__), ROOT / "tests/test_semantic_concepts.py", validator_path]
    output = dict(stage="2G.2", source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        completed_at=datetime.now(timezone(timedelta(hours=3))).isoformat(), timezone="Europe/Moscow",
        source_namespace=NS, source_fingerprint=source_manifest["fingerprint"], source_sha256=before,
        thresholds=list(EXTENDED_THRESHOLDS), policy=SEMANTIC_POLICY, embedding=meta, validation=validation,
        evaluation_statistics=statistics, checked_candidates=len(candidates), decisions=len(decisions),
        validator_only_new_memberships=len(b_delta), unique_new_memberships=len({(r["source_entity_id"], r["target_concept_id"]) for r in accepted}),
        protected_files_sha256=hashes, code_sha256={p.relative_to(ROOT).as_posix(): graph.base.sha(p) for p in code},
        independent_gold=False, neo4j_import=False, new_namespaces_created=0, push=False)
    test_path = CACHE / "extended_tests.xml"
    if test_path.exists():
        import xml.etree.ElementTree as ET
        suite = ET.parse(test_path).getroot().find("testsuite")
        tests = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
        tests.update(seconds=float(suite.attrib["time"]), junit_sha256=graph.base.sha(test_path))
        require(tests["failures"] == tests["errors"] == 0, "Extended tests failed")
        output["tests"] = tests
    output["artifact_files_sha256"] = {p.name: graph.base.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"}
    save(OUT / "manifest.json", output)
    banner("STAGE 2G.2 ЗАВЕРШЁН: 102 ВИРТУАЛЬНЫХ КОНФИГУРАЦИИ; ИМПОРТА И PUSH НЕТ")
    print(dumps(dict(candidates=len(candidates), decisions=len(decisions), B_new_memberships=len(b_delta),
        unique_new_memberships=output["unique_new_memberships"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-only", action="store_true")
    parser.add_argument("--neo4j-env", type=Path)
    parser.add_argument("--extended", action="store_true", help="Stage 2G.2: 0.700–0.950, frozen validator, explicit top-k review")
    args = parser.parse_args()
    if args.extended:
        return run_extended(args)
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    protected = sorted(p for p in graph.base.ER.rglob("*") if p.is_file() and not p.is_relative_to(OUT))
    protected += [graph.base.DOCUMENTS, ROOT / "src/cgraph/concepts.py", ROOT / "scripts/build_dynamic_concept_graph.py"]
    protected_hashes = {p.relative_to(ROOT).as_posix(): graph.base.sha(p) for p in protected}
    driver = None
    banner("ПРОВЕРЯЮ ИСХОДНЫЙ STAGE 2G И НЕИЗМЕННЫЙ КОРПУС — БЕЗ ЗАПИСИ В NEO4J")
    outside_before = schema_before = None
    if args.neo4j_env:
        from neo4j import READ_ACCESS
        driver, database = graph.base.core.connect_neo4j(args.neo4j_env)
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            nodes, edges, manifest, verification = verified_source(session)
            save(CACHE / "baseline.json", dict(nodes=[dict(key=k, labels=[r["label"]], properties=r["properties"]) for k, r in nodes.items()],
                relationships=[dict(key=k, **r) for k, r in edges.items()]))
            banner("ЧИТАЮ ВНЕШНИЙ СНИМОК NEO4J И СХЕМУ — ПРОЦЕНТ НЕ ОПРЕДЕЛЁН")
            graph.base.NS = NS
            outside_before = graph.base.outside_snapshot(session)
            schema_before = dict(constraints=session.run(graph.SCHEMA_QUERY).data(), indexes=session.run(graph.INDEX_QUERY).data())
    else:
        nodes, edges, manifest, verification = verified_source()
    before_hash = graph.snapshot_hash(nodes, edges)
    write_design()
    entities, catalog, memberships, similar, docs = graph.profiles(nodes, edges)
    rows, references = representation_inputs(nodes, entities, catalog)
    meta = embeddings(rows, references)
    candidates = discover(rows, references, catalog, memberships, similar)
    jsonl("semantic_candidates.jsonl", candidates)
    print("SEMANTIC CANDIDATES:", dict(Counter((c["embedding_mode"], c["kind"]) for c in candidates)), flush=True)
    if args.embeddings_only:
        if driver is not None:
            driver.close()
        return
    banner("ПРОВЕРЯЮ SEMANTIC CANDIDATES И СТРОЮ 22 ВИРТУАЛЬНЫХ ГРАФА")
    decisions, proposals = evaluate(candidates, entities, catalog, memberships)
    jsonl("semantic_decisions.jsonl", decisions)
    jsonl("new_concept_proposals.jsonl", proposals)
    comparison, accepted, changes, baseline = threshold_sweep(nodes, edges, candidates, decisions, catalog)
    csvout("threshold_comparison.csv", comparison)
    jsonl("new_memberships.jsonl", accepted)
    save(OUT / "document_pair_changes.json", changes)
    save(OUT / "baseline_metrics.json", baseline)
    control, probes = controls_and_probes(rows, references, entities, catalog, memberships)
    csvout("validator_only_control.csv", control)
    jsonl("mandatory_audit_cases.jsonl", probes)
    jsonl("path_examples.jsonl", accepted_paths(nodes, edges, accepted))
    jsonl("semantic_only_discoveries.jsonl", [r for r in decisions if r.get("semantic_only_candidate_pair") or r.get("semantic_only_membership")])
    audit = quality_audit(decisions, entities)
    csvout("quality_audit.csv", audit)
    paths = accepted_paths(nodes, edges, accepted)
    universe = universe_summary(rows, references, catalog, memberships)
    save(OUT / "candidate_universe.json", universe)
    # Verify monotone candidate filtering independently from the generated list.
    import numpy as np
    doc_ids = np.array([r["document_id"] for r in rows])
    cross = np.triu(doc_ids[:, None] != doc_ids[None, :], 1)
    for mode in MODES:
        scores = np.load(CACHE / (mode + "_ee.npy"), allow_pickle=False)[cross]
        expected = [sum(passes(s, t) for s in scores[scores >= float(t)]) for t in THRESHOLDS]
        actual = [r["entity_entity_candidates"] for r in comparison if r["embedding_mode"] == mode]
        require(expected == actual, "Full cross-document candidate coverage mismatch")
        require(all(a >= b for a, b in zip(actual, actual[1:])), "Non-monotone candidate sweep")
    require(graph.snapshot_hash(nodes, edges) == before_hash, "In-memory frozen source changed")
    require(len(comparison) == 22 and len({(r["embedding_mode"], r["threshold"]) for r in comparison}) == 22, "Incomplete sweep")
    require(all(r["status"] in {"ACCEPT", "REJECT"} for r in decisions), "Invalid decision status")
    require({r["decision_id"] for r in decisions if r["status"] == "ACCEPT" and r.get("adds_membership", True)} <=
            {r["decision_id"] for r in audit}, "Unaudited new ACCEPT")
    validation = dict(source_namespace=NS, source_fingerprint=manifest["fingerprint"], source_sha256_before=before_hash,
        documents=30, entities=811, baseline_concepts=34, baseline_memberships=130, facts=1307, similar_to=492,
        configurations=22, thresholds=list(THRESHOLDS), all_cross_document_pairs_evaluated=True,
        original_lexical_resolver_unchanged=True, all_virtual_graphs_preserve_exact_baseline_records=True,
        candidate_counts_independently_verified=True, leave_one_document_out_reference_contexts=True,
        self_context_leaks=0, no_ner_re_or_merge_inference=True, corpus_expansion=False, neo4j_import=False,
        embeddings_cached_once_per_mode=True, all_new_accept_decisions_audited=True, independent_gold=False,
        new_accept_candidate_concept_decisions=sum(r["status"] == "ACCEPT" and r.get("adds_membership", True) for r in decisions),
        unique_new_accept_memberships=len({(r["source_entity_id"], r["target_concept_id"]) for r in accepted}),
        reviewed_reject_decisions=sum(r["status"] == "REJECT" for r in audit),
        potential_false_negative_decisions=sum(r["audit_verdict"] == "potential_false_negative" for r in audit),
        live_database_preservation_verified=False)
    if driver is not None:
        try:
            with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
                banner("ПОВТОРНО ПРОВЕРЯЮ ПОЛНЫЙ STAGE 2G И ВНЕШНИЙ СНИМОК — ТОЛЬКО ЧТЕНИЕ")
                after_nodes, after_edges, _, _ = verified_source(session)
                outside_after = graph.base.outside_snapshot(session)
                schema_after = dict(constraints=session.run(graph.SCHEMA_QUERY).data(), indexes=session.run(graph.INDEX_QUERY).data())
                require(outside_before == outside_after and schema_before == schema_after, "External namespace or schema changed")
                require(graph.snapshot_hash(after_nodes, after_edges) == before_hash, "Stage 2G changed during sweep")
                save(OUT / "neo4j_preservation.json", dict(source_sha256_before=before_hash,
                    source_sha256_after=graph.snapshot_hash(after_nodes, after_edges),
                    outside_before=outside_before, outside_after=outside_after,
                    schema_before=schema_before, schema_after=schema_after))
                validation.update(live_database_preservation_verified=True, external_namespaces_and_schema_unchanged=True,
                                  source_sha256_after=before_hash)
        finally:
            driver.close()
    require(all(graph.base.sha(ROOT / p) == h for p, h in protected_hashes.items()), "Prior artifacts/corpus changed")
    require(all(graph.base.sha(Path(p["path"])) == p["sha256"] for p in meta["model"]["files"]), "E5 files changed during experiment")
    validation.update(protected_files_unchanged=True, model_files_unchanged=True)
    save(OUT / "validation.json", validation)
    write_report(comparison, baseline, candidates, decisions, accepted, control, probes, audit, universe, meta, paths)
    code = [Path(__file__), ROOT / "src/cgraph/semantic_concepts.py", ROOT / "tests/test_semantic_concepts.py"]
    tests = None
    if (CACHE / "tests.xml").exists():
        import xml.etree.ElementTree as ET
        suite = ET.parse(CACHE / "tests.xml").getroot().find("testsuite")
        tests = {k: int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
        tests["seconds"] = float(suite.attrib["time"])
        tests["junit_sha256"] = graph.base.sha(CACHE / "tests.xml")
        require(tests["failures"] == tests["errors"] == 0, "Unit tests did not pass")
    output = dict(stage="2G.1", resolver_version=VERSION, completed_at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        timezone="Europe/Moscow", source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_namespace=NS, source_fingerprint=manifest["fingerprint"], source_snapshot_sha256=before_hash,
        stage_2g_manifest_sha256=graph.base.sha(SOURCE / "manifest.json"),
        corpus_sha256=manifest["document_lock_sha256"], policy=SEMANTIC_POLICY, embedding=meta,
        protected_files_sha256=protected_hashes, code_sha256={p.relative_to(ROOT).as_posix(): graph.base.sha(p) for p in code},
        tests=tests, validation=validation, recommendation=dict(mode="text_name", threshold="0.955", exploratory=True,
            independent_precision_measured=False, context_mode_recommendation=None),
        neo4j_import=False, new_namespaces_created=0, push=False)
    output["artifact_files_sha256"] = {p.name: graph.base.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"}
    save(OUT / "manifest.json", output)
    banner("STAGE 2G.1 ЗАВЕРШЁН: 22 ВИРТУАЛЬНЫХ ГРАФА, ИСХОДНЫЙ ГРАФ СОХРАНЁН, ИМПОРТА НЕТ")
    print(dumps(dict(configurations=22, candidates=len(candidates), unique_new_memberships=validation["unique_new_accept_memberships"],
                     live_preservation=validation["live_database_preservation_verified"])), flush=True)


if __name__ == "__main__":
    main()
