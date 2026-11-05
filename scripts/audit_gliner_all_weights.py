"""Complete, data-free parameter audit of the project's GLiNER2.5 checkpoint.

Every scalar is included in statistics; every 2D tensor gets its full spectrum.
No torch, model forward, training, optimizer, test corpus or labels are used.
"""
import csv
import hashlib
import json
import math
import os
import re
import struct
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/ner_v2/models/gliner25"
OUT = ROOT / "reports/gliner_weight_full_audit"
LIB = ROOT / ".venv-ner-v2/Lib/site-packages/gliner2"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def role(key):
    if key.startswith("encoder.embeddings."):
        return "token_embeddings_and_input_normalization", "active_shared_encoder"
    if key.startswith("encoder.encoder.layer."):
        part = "attention" if ".attention." in key else "feedforward"
        return "transformer_" + part, "active_shared_encoder"
    if key.startswith("encoder."):
        return "relative_position_embeddings_and_normalization", "active_shared_encoder"
    if key.startswith("boundary_head.boundary_encoder."):
        return "span_boundary_representation", "active_entity_extraction"
    if key.startswith("boundary_head.boundary_query_head."):
        return "type_conditioned_boundary_and_inside_scores", "active_entity_extraction"
    if key.startswith("boundary_head.shared_pool_builder."):
        return "candidate_pool_selection", "active_entity_extraction"
    if key.startswith("boundary_head.shared_pool_scorer."):
        return "joint_span_type_reranking", "active_entity_extraction"
    if key.startswith("boundary_head.boundary_proposer.") or key.startswith("boundary_head.pair_scorer."):
        return "legacy_per_query_proposals_and_scoring", "alternative_path_or_diagnostics_with_candidate_pool_shared"
    if key.startswith("boundary_head.candidate_encoder."):
        return "record_relation_candidate_features", "computed_but_not_in_plain_entity_pair_logits"
    if key.startswith("boundary_head.null_projection."):
        return "abstention_auxiliary_head", "auxiliary_output_decode_usage_must_be_traced"
    if key.startswith("boundary_head.count_head."):
        return "count_auxiliary_head", "auxiliary_output_decode_usage_must_be_traced"
    if key.startswith("classifier."):
        return "schema_text_classification", "separate_classification_task_not_entity_typing"
    if key.startswith("record_decoder."):
        return "structured_record_decoding", "not_plain_entity_extraction"
    if key.startswith("relation_scorer."):
        return "relation_extraction", "not_plain_entity_extraction"
    raise ValueError("Unclassified tensor " + key)


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(zip(("min", "median", "p95", "max"),
                    map(float, np.quantile(x, [0, .5, .95, 1]))))


def describe_spectrum(s):
    s = np.asarray(s, dtype=np.float64)
    energy = s * s
    total = float(energy.sum())
    positive = energy[energy > 0] / total if total else np.array([])
    threshold = max(len(s), 1) * np.finfo(np.float64).eps * (s[0] if len(s) else 0)
    result = {"singular_values": s.tolist(), "spectral_norm": float(s[0]),
              "smallest_singular_value": float(s[-1]),
              "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else None,
              "stable_rank": total / float(energy[0]) if total else 0.0,
              "entropy_effective_rank": float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0,
              "numerical_rank_float64_spectrum": int((s > threshold).sum()),
              "top1_energy_fraction": float(energy[0] / total) if total else 0.0,
              "top1_to_top2": float(s[0] / s[1]) if len(s) > 1 and s[1] else None,
              "top1_to_median": float(s[0] / np.median(s)) if np.median(s) else None}
    for fraction in (.9, .95, .99):
        result[f"rank_for_{int(fraction * 100)}pct_energy"] = int(np.searchsorted(np.cumsum(energy), fraction * total) + 1) if total else 0
    return result


class Checkpoint:
    def __init__(self, path):
        self.path = path
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            self.header = json.loads(stream.read(size))
        self.base = size + 8

    def array(self, key):
        spec = self.header[key]
        if spec["dtype"] != "F32":
            raise ValueError("Unexpected dtype: " + key)
        return np.memmap(self.path, mode="r", dtype="<f4", shape=tuple(spec["shape"]),
                         offset=self.base + spec["data_offsets"][0])

    def affine(self, stem):
        w = np.asarray(self.array(stem + ".weight"), dtype=np.float64)
        bias_key = stem + ".bias"
        b = np.asarray(self.array(bias_key), dtype=np.float64) if bias_key in self.header else np.zeros(w.shape[0])
        return np.column_stack((w, b))


def tensor_statistics(array):
    # Streaming float64 accumulation includes every weight, with bounded memory.
    flat = array.reshape(-1)
    count = flat.size
    finite = zero = 0
    total = squares = absolute = 0.0
    low, high = math.inf, -math.inf
    digest = hashlib.sha256()
    for start in range(0, count, 1024 * 1024):
        raw = np.asarray(flat[start:start + 1024 * 1024])
        digest.update(raw.tobytes())
        x = raw.astype(np.float64)
        finite += int(np.isfinite(x).sum())
        if not np.isfinite(x).all():
            raise ValueError("Non-finite checkpoint values")
        zero += int(np.count_nonzero(x == 0))
        total += float(x.sum())
        squares += float(x @ x)
        absolute += float(np.abs(x).sum())
        low = min(low, float(x.min()))
        high = max(high, float(x.max()))
    mean = total / count
    return {"numel": count, "finite_count": finite, "zero_count": zero,
            "mean": mean, "std": math.sqrt(max(0.0, squares / count - mean * mean)),
            "l1_norm": absolute, "frobenius_norm": math.sqrt(squares),
            "min": low, "max": high, "max_abs": max(abs(low), abs(high)),
            "sha256": digest.hexdigest()}


def matrix_spectrum(array):
    m, n = array.shape
    if max(m, n) <= 8192:
        w = np.asarray(array, dtype=np.float64)
        s = np.linalg.svd(w, compute_uv=False)
        rows = np.linalg.norm(w, axis=1)
        columns = np.linalg.norm(w, axis=0)
        method = "full_float64_svd"
    else:
        # Full spectrum of the 128011 x 768 embedding table, using every row.
        # Eigendecomposition of W^T W squares conditioning; tail accuracy is
        # therefore lower than direct SVD. No random or truncated SVD is used.
        if m < n:
            raise ValueError("Unsupported wide embedding")
        gram = np.zeros((n, n), dtype=np.float64)
        rows = np.empty(m, dtype=np.float64)
        for start in range(0, m, 4096):
            block = np.asarray(array[start:start + 4096], dtype=np.float64)
            gram += block.T @ block
            rows[start:start + len(block)] = np.linalg.norm(block, axis=1)
        eigenvalues = np.linalg.eigvalsh(gram)
        s = np.sqrt(np.maximum(eigenvalues, 0))[::-1]
        columns = np.sqrt(np.diag(gram))
        method = "full_float64_streamed_gram_eigendecomposition"
    return {**describe_spectrum(s), "method": method,
            "row_l2": distribution(rows), "column_l2": distribution(columns),
            "largest_norm_rows": [{"index": int(i), "norm": float(rows[i])}
                                  for i in np.argsort(rows)[-8:][::-1]],
            "largest_norm_columns": [{"index": int(i), "norm": float(columns[i])}
                                     for i in np.argsort(columns)[-8:][::-1]]}


def product_spectrum(left, right):
    """Nonzero spectrum of left.T @ right without the large output matrix."""
    _, rleft = np.linalg.qr(left.T, mode="reduced")
    _, rright = np.linalg.qr(right.T, mode="reduced")
    return np.linalg.svd(rleft @ rright.T, compute_uv=False)


def coupled_analysis(checkpoint):
    coupled = {}
    prefix = "boundary_head.boundary_query_head."
    for name, left, right in (
        ("start", "start_boundary_projection", "start_query_projection"),
        ("end", "end_boundary_projection", "end_query_projection"),
        ("inside", "inside_text_projection", "inside_query_projection"),
    ):
        a, b = checkpoint.affine(prefix + left), checkpoint.affine(prefix + right)
        s = product_spectrum(a, b) / math.sqrt(128)
        coupled["boundary_" + name] = {
            "left": prefix + left, "right": prefix + right,
            "meaning": "exact affine bilinear logit kernel on augmented inputs, before masks",
            "formula": "K = [W_left,b_left]^T [W_right,b_right] / sqrt(128)",
            "shape": [a.shape[1], b.shape[1]], "nonzero_spectrum": describe_spectrum(s)}
    stem = "boundary_head.shared_pool_builder."
    a, b = checkpoint.affine(stem + "start_projection"), checkpoint.affine(stem + "end_projection")
    coupled["candidate_pool_compatibility"] = {
        "meaning": "exact affine compatibility kernel before top-k and deduplication",
        "formula": "K = [W_start,b_start]^T [W_end,b_end] / sqrt(128)",
        "nonzero_spectrum": describe_spectrum(product_spectrum(a, b) / math.sqrt(128))}
    attention = []
    for layer in tqdm(range(12), desc="Coupled attention", unit="layer"):
        stem = f"encoder.encoder.layer.{layer}.attention."
        q = checkpoint.affine(stem + "self.query_proj")
        k = checkpoint.affine(stem + "self.key_proj")
        v = checkpoint.affine(stem + "self.value_proj")
        output = np.asarray(checkpoint.array(stem + "output.dense.weight"), dtype=np.float64)
        heads = []
        for head in range(12):
            sl = slice(head * 64, (head + 1) * 64)
            # DeBERTa content-content branch only; shared relative-key branches
            # must also be accounted for before editing Q or K.
            qk = product_spectrum(q[sl], k[sl]) / math.sqrt(64 * 3)
            ov = product_spectrum(output[:, sl].T, v[sl])
            heads.append({"head": head, "qk_content_nonzero_spectrum": describe_spectrum(qk),
                          "output_value_nonzero_spectrum": describe_spectrum(ov)})
        attention.append({"layer": layer, "heads": heads,
                          "caveat": "QK is content-content only; relative p2c/c2p, softmax and LayerNorm are not represented"})
    coupled["encoder_attention_by_head"] = attention
    return coupled


def main():
    started = time.perf_counter()
    # Modify only our own scheduling priority, never the running benchmark.
    try:
        import psutil
        if os.name == "nt":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (ImportError, OSError):
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(SOURCE / "model.safetensors")
    lock = json.loads((ROOT / "contracts/ner_v2/gliner25_research_lock.json").read_text(encoding="utf-8"))
    if source_hash != lock["files"]["model.safetensors"]["sha256"]:
        raise ValueError("Source weights differ from research lock")
    checkpoint = Checkpoint(SOURCE / "model.safetensors")
    rows = []
    spectra = {}
    keys = [key for key in checkpoint.header if key != "__metadata__"]
    for key in tqdm(keys, desc="All GLiNER tensors", unit="tensor", mininterval=1):
        spec = checkpoint.header[key]
        array = checkpoint.array(key)
        category, usage = role(key)
        row = {"name": key, "shape": spec["shape"], "dtype": spec["dtype"],
               "role": category, "usage": usage, **tensor_statistics(array)}
        if array.ndim == 2:
            spectra[key] = matrix_spectrum(array)
            for field in ("spectral_norm", "smallest_singular_value", "condition_number",
                          "stable_rank", "top1_energy_fraction", "top1_to_top2", "top1_to_median"):
                row[field] = spectra[key][field]
            relative_error = abs(np.linalg.norm(spectra[key]["singular_values"]) - row["frobenius_norm"]) / max(row["frobenius_norm"], 1e-300)
            row["spectrum_frobenius_relative_error"] = float(relative_error)
            if relative_error > 1e-9:
                raise ValueError("Spectrum energy mismatch: " + key)
        elif array.ndim != 1:
            raise ValueError("Unexpected tensor rank: " + key)
        rows.append(row)
        del array
        # A durable progress snapshot does not require rerunning inference.
        write(OUT / "progress.json", {"status": "ANALYZING", "completed_tensors": len(rows),
                                     "total_tensors": len(keys), "last_tensor": key})
    write(OUT / "all_tensor_statistics.json", rows)
    write(OUT / "all_matrix_spectra.json", spectra)
    coupled = coupled_analysis(checkpoint)
    write(OUT / "coupled_projections.json", coupled)
    groups = defaultdict(lambda: {"tensors": 0, "parameters": 0, "matrices": 0})
    for row in rows:
        g = groups[row["role"]]
        g["tensors"] += 1
        g["parameters"] += row["numel"]
        g["matrices"] += len(row["shape"]) == 2
    v1 = json.loads((ROOT / "models/gliner_weight_surgery_v1/weight_surgery_manifest.json").read_text(encoding="utf-8"))
    v1_checkpoint = Checkpoint(ROOT / "models/gliner_weight_surgery_v1/model.safetensors")
    changed = []
    for row in tqdm(rows, desc="Verify v1 tensor hashes", unit="tensor", mininterval=1):
        array = v1_checkpoint.array(row["name"])
        h = hashlib.sha256()
        flat = array.reshape(-1)
        for start in range(0, flat.size, 1024 * 1024):
            h.update(np.asarray(flat[start:start + 1024 * 1024]).tobytes())
        if h.hexdigest() != row["sha256"]:
            changed.append(row["name"])
        del array
    if changed != v1["changed_tensors"]:
        raise ValueError("V1 integrity mismatch")
    if sha256(SOURCE / "model.safetensors") != source_hash:
        raise ValueError("Source weights changed during read-only audit")
    config = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    ner_config = json.loads((ROOT / "reports/ner_v2/research_run2/E3/config.json").read_text(encoding="utf-8"))
    ontology = json.loads((ROOT / "contracts/ner_v2/ontology.v2.json").read_text(encoding="utf-8"))
    sources = [LIB / "models/boundary/model.py", LIB / "models/boundary/heads.py",
               LIB / "models/boundary/pool.py", LIB / "models/boundary/encoding.py",
               LIB / "inference/candidate_decoder.py", LIB / "inference/runtime.py",
               ROOT / "src/cgraph/entity_typing_v2/dataset.py",
               ROOT / "contracts/ner_v2/ontology.v2.json", SOURCE / "config.json"]
    manifest = {"status": "COMPLETE", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_weights_sha256": source_hash, "source_repo": lock["repo_id"],
                "revision": lock["revision"], "tensors": len(rows), "matrices": len(spectra),
                "vectors": sum(len(r["shape"]) == 1 for r in rows),
                "parameters": sum(r["numel"] for r in rows),
                "finite_parameters": sum(r["finite_count"] for r in rows),
                "groups": dict(groups), "source_unchanged": True,
                "full_coverage": "All scalar values and every 2D spectrum; no random sampling",
                "v1_changed_tensors_reverified": changed,
                "v1_surgery_is_task_conditioned": False,
                "task": {"labels": ner_config["labels"],
                         "ontology_definitions": {k: v["ner_description_en"] for k, v in ontology["entity_types"].items()},
                         "active_candidate_pool": config["boundary_head"]["candidate_pool"],
                         "max_input_tokens": ner_config["max_input_tokens"],
                         "entity_threshold": .35, "normalization": ner_config["normalization"]},
                "no_training": True, "no_gradients": True, "no_optimizers": True,
                "no_inference_or_tests": True, "no_corpus_or_reference_read": True,
                "new_weight_modifications": 0, "blas_threads": 1,
                "wall_seconds": time.perf_counter() - started,
                "sources_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "script_sha256": sha256(Path(__file__))}
    write(OUT / "manifest.json", manifest)
    with (OUT / "all_tensors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["name", "shape", "role", "usage", "numel", "finite_count", "zero_count", "mean", "std",
                  "frobenius_norm", "max_abs", "spectral_norm", "smallest_singular_value", "stable_rank",
                  "top1_to_top2", "top1_energy_fraction", "condition_number", "sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "shape": " x ".join(map(str, row["shape"]))})
    write(OUT / "progress.json", {"status": "COMPLETE", "completed_tensors": len(rows), "total_tensors": len(keys)})
    print(json.dumps({k: manifest[k] for k in ("status", "tensors", "matrices", "vectors", "parameters", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
