"""Source-exact mention grouping and deterministic, stratified sampling."""
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from . import PROTOCOL_VERSION
from .storage import digest, file_hash, now, read_json, read_jsonl, write_json, write_jsonl


def source_paths(config, root):
    source = root / config["source_run"]
    return {"abstracts": source / "input_abstracts.jsonl",
            "mentions": source / "E3/mentions_035.jsonl",
            "ner_config": source / "E3/config.json",
            "ontology": root / config["ontology"]}


def load_candidates(config, root):
    paths = source_paths(config, root)
    ontology = read_json(paths["ontology"])
    labels = ontology["entity_types"]
    if len(labels) != 11 or config["detection_threshold"] != .35:
        raise ValueError("This protocol requires unchanged Ontology v2 (11 types) and detection 0.35")
    ner = read_json(paths["ner_config"])
    if (ner["configuration"] != "E3" or ner["normalization"] != "N3" or
            ner["checkpoint"]["repo_id"] != "fastino/gliner2.5-base-v1"):
        raise ValueError("Expected saved E3 GLiNER2.5 Base + N3")
    if (ner["ontology_sha256"] != file_hash(paths["ontology"]) or
            ner["input_sha256"] != file_hash(paths["abstracts"])):
        raise ValueError("Historical E3 source/ontology hash mismatch")
    documents = read_jsonl(paths["abstracts"])
    by_id = {d["document_id"]: d for d in documents}
    if len(by_id) != len(documents) or len(documents) != config["expected_documents"]:
        raise ValueError("Unexpected source document count/duplicate IDs")
    import hashlib
    for d in documents:
        if hashlib.sha256(d["text"].encode("utf-8")).hexdigest() != d["text_sha256"]:
            raise ValueError("Source text hash mismatch")
    groups = defaultdict(list)
    for m in tqdm(read_jsonl(paths["mentions"]), desc="Validate saved E3", unit="hint"):
        d = by_id[m["document_id"]]
        score = m["score"]
        if (type(score) not in (int, float) or not math.isfinite(score) or not .35 <= score <= 1 or
                m["ner_label_hint"] not in labels or m["configuration"] != "E3" or
                m["normalizer_version"] != "scientific-noun-head-nfc-v1" or
                m["revision_id"] != d["revision_id"] or m["source_reference"] != d["source_reference"]):
            raise ValueError("Invalid saved E3 hint/provenance")
        a, b = m["start"], m["end"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(d["text"]) or d["text"][a:b] != m["surface_text"]:
            raise ValueError("Offsets must match source Unicode codepoints exactly")
        groups[(m["document_id"], a, b)].append(m)
    candidates = []
    for (doc_id, start, end), hints in sorted(groups.items()):
        d = by_id[doc_id]
        if len({h["normalized_form"] for h in hints}) != 1:
            raise ValueError("Inconsistent saved N3 forms for one span")
        scores = {}
        for h in hints:
            label = h["ner_label_hint"]
            scores[label] = max(scores.get(label, 0), h["score"])
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        top, second = ranked[0], ranked[1] if len(ranked) > 1 else None
        candidates.append({
            "candidate_id": "c:" + digest([doc_id, d["revision_id"], d["text_sha256"], start, end])[:24],
            "document_id": doc_id, "start": start, "end": end,
            "surface_text": d["text"][start:end], "normalized_form": hints[0]["normalized_form"],
            "context": d["text"], "revision_id": d["revision_id"], "text_sha256": d["text_sha256"],
            "source_reference": d["source_reference"], "discipline": d["discipline"],
            "mention_ids": sorted({h["mention_id"] for h in hints}),
            "top1_type": top[0], "top1_score": top[1],
            "top2_type": second[0] if second else None, "top2_score": second[1] if second else None,
            "score_margin": top[1] - second[1] if second else None,
            "alternative_labels": [{"entity_type": label, "score": score} for label, score in ranked]})
    if len(candidates) != config["expected_candidates"]:
        raise ValueError(f"Expected {config['expected_candidates']} spans, got {len(candidates)}")
    return candidates, ontology, {key: file_hash(path) for key, path in paths.items()}


def stratified_sample(candidates, size=240, seed=4401):
    """Discipline × top1 type, minimum one per nonempty stratum, proportional remainder.

    Inside strata, cycle score bands × ambiguity before taking a second member.
    No outcome labels or previous assisted annotations are read.
    """
    groups = defaultdict(list)
    for c in sorted(candidates, key=lambda c: c["candidate_id"]):
        groups[(c["discipline"], c["top1_type"])].append(c)
    if not len(groups) <= size <= len(candidates):
        raise ValueError("Pilot size must cover all nonempty discipline/type strata")
    keys = sorted(groups)
    remaining = size - len(keys)
    capacity = sum(len(groups[k]) - 1 for k in keys)
    exact = {k: remaining * (len(groups[k]) - 1) / capacity if capacity else 0 for k in keys}
    quotas = {k: 1 + int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] % 1), k))[:size - sum(quotas.values())]:
        quotas[k] += 1
    rng = random.Random(seed)
    selected = []
    for k in keys:
        bins = defaultdict(list)
        for c in groups[k]:
            band = sum(c["top1_score"] >= t for t in (.5, .55, .6, .65))
            ambiguous = c["score_margin"] is not None and c["score_margin"] < .15
            bins[(band, ambiguous)].append(c)
        for pool in bins.values():
            rng.shuffle(pool)
        chosen = []
        while len(chosen) < quotas[k]:
            for b in sorted(bins):
                if bins[b] and len(chosen) < quotas[k]:
                    chosen.append(bins[b].pop())
        selected.extend(chosen)
    return sorted(selected, key=lambda c: (c["document_id"], c["start"], c["end"]))


def prepare(config, root, output, dataset):
    candidates, ontology, hashes = load_candidates(config, root)
    selected = stratified_sample(candidates, config["pilot_size"], config["pilot_seed"]) if dataset == "pilot" else candidates
    frozen = {"protocol_version": PROTOCOL_VERSION, "dataset": dataset,
              "source_hashes": hashes, "detection_threshold": .35,
              "candidate_ids": [c["candidate_id"] for c in selected],
              "candidate_digest": digest(selected), "ontology_digest": digest(ontology),
              "pilot_seed": config["pilot_seed"], "pilot_size": config["pilot_size"],
              "sampling": "discipline x top1; min-one proportional; score/ambiguity round-robin"}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old, existing, _ = load_frozen(output)
        if old["frozen"] != frozen or existing != selected:
            raise ValueError("Frozen dataset changed; choose a new --output directory")
        return old, selected
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_dataset.jsonl", selected)
    write_json(output / "ontology.json", ontology)
    manifest = {"experiment": config["experiment"], "created_at": now(), "frozen": frozen,
                "candidate_count": len(selected), "full_candidate_count": len(candidates),
                "strata": dict(sorted(Counter(c["discipline"] + " / " + c["top1_type"] for c in selected).items())),
                "runtime": None}
    write_json(manifest_path, manifest)
    write_jsonl(output / "failure_log.jsonl", [])
    return manifest, selected


def load_frozen(output):
    manifest = read_json(output / "manifest.json")
    candidates = read_jsonl(output / "candidate_dataset.jsonl")
    ontology = read_json(output / "ontology.json")
    frozen = manifest["frozen"]
    if (digest(candidates) != frozen["candidate_digest"] or digest(ontology) != frozen["ontology_digest"] or
            [c["candidate_id"] for c in candidates] != frozen["candidate_ids"] or
            len(set(frozen["candidate_ids"])) != len(candidates)):
        raise ValueError("Frozen manifest/dataset/ontology integrity mismatch")
    return manifest, candidates, ontology


def gliner_rows(candidates):
    keys = ("candidate_id", "document_id", "start", "end", "surface_text", "context",
            "top1_type", "top1_score", "top2_type", "top2_score", "score_margin", "alternative_labels")
    return [{**{k: c[k] for k in keys}, "entity_type": c["top1_type"], "typing_status": "TYPED"}
            for c in candidates]
