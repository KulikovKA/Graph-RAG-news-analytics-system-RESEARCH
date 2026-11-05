"""LayerNorm-aware and path-aware interpretation of the complete weight audit."""
from audit_gliner_all_weights import (
    ROOT, SOURCE, OUT, LIB, Checkpoint, describe_spectrum, np, sha256, write,
)
import json


def centered_output(w):
    return w - w.mean(axis=0, keepdims=True)


def main():
    original = Checkpoint(SOURCE / "model.safetensors")
    stats = json.loads((OUT / "all_tensor_statistics.json").read_text(encoding="utf-8"))
    spectra = json.loads((OUT / "all_matrix_spectra.json").read_text(encoding="utf-8"))
    by_key = {r["name"]: r for r in stats}
    scorer = "boundary_head.shared_pool_scorer."
    direct = {}
    for name in ("start_projection", "end_projection", "length_projection",
                 "prior_projection", "content_projection"):
        key = scorer + name + ".weight"
        w = np.asarray(original.array(key), dtype=np.float64)
        u, s, _ = np.linalg.svd(w, full_matrices=False)
        hw = centered_output(w)
        direct[key] = {
            "raw": describe_spectrum(s),
            "centered": describe_spectrum(np.linalg.svd(hw, compute_uv=False)),
            "top_left_singular_vector_common_mode_energy": float(u[:, 0].sum() ** 2 / len(u)),
            "weight_energy_in_LayerNorm_invisible_direction": float(np.linalg.norm(w - hw) ** 2 / np.linalg.norm(w) ** 2),
        }
    # Centered concatenation measures a joint map from unit-norm concatenated
    # endpoint states. It is a structural bound, not an activation covariance.
    start = np.asarray(original.array(scorer + "start_projection.weight"), dtype=np.float64)
    end = np.asarray(original.array(scorer + "end_projection.weight"), dtype=np.float64)
    joined = np.column_stack((centered_output(start), centered_output(end)))
    joint = describe_spectrum(np.linalg.svd(joined, compute_uv=False))
    modified = Checkpoint(ROOT / "models/gliner_weight_surgery_v1/model.safetensors")
    new = np.asarray(modified.array(scorer + "start_projection.weight"), dtype=np.float64)
    delta = new - start
    v1 = {
        "raw_delta_frobenius": float(np.linalg.norm(delta)),
        "centered_delta_frobenius": float(np.linalg.norm(centered_output(delta))),
        "invisible_delta_energy_fraction": float(np.linalg.norm(delta - centered_output(delta)) ** 2 / np.linalg.norm(delta) ** 2),
        "centered_relative_delta_frobenius": float(np.linalg.norm(centered_output(delta)) / np.linalg.norm(centered_output(start))),
        "note": "Centering isolates the effect entering candidate LayerNorm, not the final output difference",
    }
    # Inspect every transformer output matrix immediately preceding LayerNorm.
    # Post-LN kills common output shifts even with the residual connection.
    post_ln = {}
    for layer in range(12):
        for stem in ("attention.output.dense", "output.dense"):
            key = f"encoder.encoder.layer.{layer}.{stem}.weight"
            w = np.asarray(original.array(key), dtype=np.float64)
            hw = centered_output(w)
            sc = np.linalg.svd(hw, compute_uv=False)
            post_ln[key] = {
                "raw_top1_to_top2": spectra[key]["top1_to_top2"],
                "raw_top1_energy_fraction": spectra[key]["top1_energy_fraction"],
                "centered": describe_spectrum(sc),
                "weight_energy_in_LayerNorm_invisible_direction": float(np.linalg.norm(w - hw) ** 2 / np.linalg.norm(w) ** 2),
            }
    # Candidate null-head bias is a meaningful detection control; count is not
    # used for span selection with adaptive_threshold=false in this checkpoint.
    null = original.affine("boundary_head.null_projection").ravel()
    result = {
        "status": "COMPLETE", "scorer_pre_LayerNorm_projections": direct,
        "joint_centered_endpoint_spectrum": joint,
        "v1_effect_before_LayerNorm": v1,
        "transformer_pre_LayerNorm_output_projections": post_ln,
        "null_head": {"weight_l2": float(np.linalg.norm(null[:-1])), "bias": float(null[-1]),
                      "activation": "abstain for an entire query when sigmoid(w^T h + b) > 0.5",
                      "bias_lowering_effect": "for identical query states, removes some abstentions; may add false positives",
                      "no_class_ranking_guarantee": True},
        "count_head": {"adaptive_threshold": False, "used_to_add_spans_in_this_configuration": False},
        "v1_assessment": "bounded spectral experiment; not ontology-conditioned adaptation",
        "semantic_identifiability": {
            "fixed_11_class_weight_rows": False,
            "type_representation": "contextual entity schema marker states in the shared encoder",
            "class_scores": "shared boundary + inside + span/query dot product + FiLM scores",
            "weight_spectra_identify_method_vs_process": False,
        },
        "intervention_rules": [
            "Never infer harmful noise from a large singular value alone",
            "Analyze H W with H=I-11^T/d for maps contributing directly before LayerNorm",
            "Analyze paired affine kernels for dot-product projections before modifying either factor",
            "A shared scoring change affects every schema query, not a named ontology class",
            "Leave encoder Q/K, embeddings, LayerNorm scales, biases and auxiliary/task heads intact unless there is a specific supported hypothesis",
        ],
        "new_checkpoint_created": False,
        "decision": "No evidence of a defective or semantically misdirected weight; no arbitrary second edit",
        "no_training_or_inference": True,
        "source_engine_sha256": sha256(LIB / "models/boundary/engine.py"),
        "script_sha256": sha256(ROOT / "scripts/analyze_gliner_surgery_targets.py"),
    }
    write(OUT / "layernorm_and_task_analysis.json", result)
    print(json.dumps({
        "v1_centered_delta": v1,
        "scorer_start": {k: v for k, v in direct[scorer + "start_projection.weight"].items() if k not in ("raw", "centered")},
        "joint_centered_endpoints": {k: v for k, v in joint.items() if k != "singular_values"},
        "ffn_layer3": {k: v for k, v in post_ln["encoder.encoder.layer.3.output.dense.weight"].items() if k != "centered"},
        "ffn_layer3_centered_topgap": post_ln["encoder.encoder.layer.3.output.dense.weight"]["centered"]["top1_to_top2"],
        "null_bias": float(null[-1]),
    }, indent=2))


if __name__ == "__main__":
    main()
