"""Joint, LayerNorm-aware span endpoint surgery after the complete weight audit.

Uses the original checkpoint, not v1. No model initialization or forward pass.
"""
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from audit_gliner_all_weights import (
    ROOT, SOURCE, OUT, Checkpoint, describe_spectrum, np, sha256, write,
)

DESTINATION = ROOT / "models/gliner_weight_surgery_v2"
REPORT = OUT / "surgery_v2"
STEM = "boundary_head.shared_pool_scorer."
TARGETS = [STEM + "start_projection.weight", STEM + "end_projection.weight"]
EPSILON = .001
TOP_SHRINK_LIMIT = .01


def tensor_digest(array):
    import hashlib
    flat = array.reshape(-1)
    h = hashlib.sha256()
    for start in range(0, flat.size, 1024 * 1024):
        h.update(np.asarray(flat[start:start + 1024 * 1024]).tobytes())
    return h.hexdigest()


def main():
    if DESTINATION.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite a v2 checkpoint or report")
    audit = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    interpretation = json.loads((OUT / "layernorm_and_task_analysis.json").read_text(encoding="utf-8"))
    if audit["status"] != "COMPLETE" or interpretation["status"] != "COMPLETE":
        raise ValueError("The full audit must complete before surgery")
    source = Checkpoint(SOURCE / "model.safetensors")
    if sha256(source.path) != audit["source_weights_sha256"]:
        raise ValueError("Original checkpoint identity mismatch")
    weights = [np.asarray(source.array(k), dtype=np.float64) for k in TARGETS]
    means = [w.mean(axis=0, keepdims=True) for w in weights]
    # Only the centered output directions survive the following LayerNorm.
    a = np.column_stack([w - mean for w, mean in zip(weights, means)])
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    if s[0] / s[1] <= 2:
        raise ValueError("The weight-only dominant-mode hypothesis is not supported by this checkpoint")
    budget = EPSILON * np.linalg.norm(a)
    tau = (1 - TOP_SHRINK_LIMIT) * s[0]
    if np.linalg.norm(np.maximum(s - tau, 0)) > budget:
        lower, upper = tau, s[0]
        for _ in range(80):
            middle = (lower + upper) / 2
            if np.linalg.norm(np.maximum(s - middle, 0)) > budget:
                lower = middle
            else:
                upper = middle
        tau = upper
    clipped = np.minimum(s, tau)
    update = (u * (clipped - s)) @ vt
    # Remove any float64 common-mode roundoff; H*update = update analytically.
    update -= update.mean(axis=0, keepdims=True)
    edited = [(w + update[:, i * 128:(i + 1) * 128]).astype("<f4")
              for i, w in enumerate(weights)]
    actual = np.column_stack([w.astype(np.float64) - w.astype(np.float64).mean(axis=0, keepdims=True)
                              for w in edited])
    actual_s = np.linalg.svd(actual, compute_uv=False)
    relative = float(np.linalg.norm(actual - a) / np.linalg.norm(a))
    if not 0 < relative <= EPSILON * 1.0001 or actual_s[0] >= s[0]:
        raise ValueError("Joint update violates the trust region or clipping direction")
    if not all(np.isfinite(w).all() for w in edited):
        raise ValueError("Nonfinite edited tensor")
    details = []
    for key, before, after in zip(TARGETS, weights, edited):
        delta = after.astype(np.float64) - before
        details.append({"name": key, "shape": list(before.shape), "dtype": "F32",
                        "before_spectrum": describe_spectrum(np.linalg.svd(before, compute_uv=False)),
                        "after_spectrum": describe_spectrum(np.linalg.svd(after.astype(np.float64), compute_uv=False)),
                        "delta_frobenius": float(np.linalg.norm(delta)),
                        "relative_delta_frobenius": float(np.linalg.norm(delta) / np.linalg.norm(before)),
                        "delta_spectral_norm": float(np.linalg.norm(delta, 2)),
                        "delta_max_abs": float(np.abs(delta).max()),
                        "output_mean_drift_max_abs": float(np.abs(delta.mean(axis=0)).max())})
    files = sorted(p for p in SOURCE.rglob("*") if p.is_file() and ".cache" not in p.parts)
    original_hashes = {p.relative_to(SOURCE).as_posix(): sha256(p) for p in files}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for path in files:
        dest = DESTINATION / path.relative_to(SOURCE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    new_path = DESTINATION / "model.safetensors"
    with new_path.open("r+b") as stream:
        for key, after in zip(TARGETS, edited):
            stream.seek(source.base + source.header[key]["data_offsets"][0])
            stream.write(after.tobytes(order="C"))
        stream.flush()
        os.fsync(stream.fileno())
    destination = Checkpoint(new_path)
    if destination.header != source.header:
        raise ValueError("Serialization header changed")
    tensor_audit = {}
    for key in source.header:
        if key == "__metadata__":
            continue
        before_hash = tensor_digest(source.array(key))
        after_hash = tensor_digest(destination.array(key))
        tensor_audit[key] = {"before_sha256": before_hash, "after_sha256": after_hash,
                             "changed": before_hash != after_hash}
    if [k for k, v in tensor_audit.items() if v["changed"]] != sorted(TARGETS):
        # Safetensors source keys are alphabetically ordered.
        raise ValueError("Unexpected set of changed tensors")
    from safetensors import safe_open
    with safe_open(str(new_path), framework="numpy") as reader:
        if set(reader.keys()) != set(tensor_audit):
            raise ValueError("Safetensors keys changed")
        for key, after in zip(TARGETS, edited):
            if not np.array_equal(reader.get_tensor(key), after):
                raise ValueError("Serialized tensor mismatch")
    source_after = {p.relative_to(SOURCE).as_posix(): sha256(p) for p in files}
    new_hashes = {name: sha256(DESTINATION / name) for name in original_hashes}
    if source_after != original_hashes:
        raise ValueError("Source checkpoint changed")
    if [k for k in original_hashes if new_hashes[k] != original_hashes[k]] != ["model.safetensors"]:
        raise ValueError("Non-weight files changed")
    REPORT.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "CHECKPOINT_SAVED_AND_SERIALIZATION_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": audit["source_repo"], "revision": audit["revision"],
        "source": str(SOURCE), "destination": str(DESTINATION),
        "parent": "original checkpoint, not v1; edits do not accumulate",
        "complete_audit_sha256": sha256(OUT / "manifest.json"),
        "layernorm_analysis_sha256": sha256(OUT / "layernorm_and_task_analysis.json"),
        "hypothesis": "A dominant joint centered endpoint direction may overemphasize shared span geometry relative to query-dependent discrimination; weak clipping is a data-free robustness experiment",
        "hypothesis_is_proven": False,
        "method": {
            "H": "I - ones(128,128)/128",
            "A": "[H W_start, H W_end]",
            "A_prime": "U diag(min(s_i,tau)) V^T",
            "W_prime": "W_i + corresponding block of (A_prime-A); preserve each uncentered output mean",
            "tau_rule": "smallest tau >= 0.99*s_1 with ||max(s-tau,0)||_2 <= 0.001*||A||_F",
            "tau": float(tau), "relative_joint_frobenius_budget": EPSILON,
            "maximum_top_relative_shrink": TOP_SHRINK_LIMIT,
            "joint_spectral_gap_before": float(s[0] / s[1]),
            "ideal_changed_singular_values": int(np.count_nonzero(s != clipped)),
            "minimality": "Frobenius-nearest matrix in the spectral-norm ball within the LayerNorm-centered endpoint map",
            "representation_bound": "||delta z_centered||_2 <= ||delta A||_2 * ||concat(boundary_start,boundary_end)||_2",
            "scope": "shared span representation before candidate_norm, affects all 11 dynamic entity type queries",
            "not_ontology_conditioned": True,
        },
        "joint_before": describe_spectrum(s), "joint_after": describe_spectrum(actual_s),
        "joint_delta_frobenius": float(np.linalg.norm(actual - a)),
        "joint_relative_delta_frobenius": relative,
        "joint_delta_spectral_norm": float(np.linalg.norm(actual - a, 2)),
        "changed_tensors": details, "unchanged_tensor_count": len(tensor_audit) - len(TARGETS),
        "source_files_unchanged": True, "source_files_sha256": original_hashes,
        "destination_files_sha256": new_hashes,
        "training": False, "gradients": False, "optimizers": False,
        "inference": False, "evaluation": "not run, per user instruction",
        "metrics_used_to_choose_or_tune": False,
        "verification": "per-tensor hashes, mathematical bounds and safetensors deserialization only",
        "quality_claim": "Unknown; may improve, preserve or worsen NER; no class-specific improvement established",
        "script_sha256": sha256(Path(__file__)),
    }
    write(DESTINATION / "weight_surgery_manifest.json", manifest)
    write(REPORT / "weight_surgery_manifest.json", manifest)
    write(REPORT / "tensor_integrity.json", tensor_audit)
    print(json.dumps({"status": manifest["status"], "destination": str(DESTINATION),
                      "changed_tensors": [{"name": d["name"], "relative_delta_percent": 100*d["relative_delta_frobenius"]} for d in details],
                      "joint_top_before": float(s[0]), "joint_top_after": float(actual_s[0]),
                      "joint_relative_delta_percent": 100*relative,
                      "unchanged_tensors": manifest["unchanged_tensor_count"],
                      "checkpoint_sha256": new_hashes["model.safetensors"]}, indent=2))


if __name__ == "__main__":
    main()
