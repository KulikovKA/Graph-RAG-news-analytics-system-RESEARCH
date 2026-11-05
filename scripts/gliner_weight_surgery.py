"""Analytical, single-matrix GLiNER2.5 surgery. No model execution or training.

Run with .venv-ner-v2/Scripts/python.exe -X utf8 scripts/gliner_weight_surgery.py
The destination must not exist. NumPy only; the checkpoint is copied and one
F32 tensor payload is replaced, preserving all other bytes and the full header.
"""
import hashlib
import json
import os
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path

# Keep this small CPU operation from consuming the night experiment's workers.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/ner_v2/models/gliner25"
DESTINATION = ROOT / "models/gliner_weight_surgery_v1"
REPORTS = ROOT / "reports/gliner_weight_surgery_v1"
TARGET = "boundary_head.shared_pool_scorer.start_projection.weight"
EPSILON = 0.001
MAX_TOP_SHRINK = 0.01


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size)), size + 8


def matrix(path, metadata, base, key):
    item = metadata[key]
    if item["dtype"] != "F32" or len(item["shape"]) != 2:
        raise ValueError(f"Expected a float32 matrix: {key}")
    start, end = item["data_offsets"]
    with path.open("rb") as stream:
        stream.seek(base + start)
        return np.frombuffer(stream.read(end - start), dtype="<f4").reshape(item["shape"]).astype(np.float64)


def group(key):
    if ".attention." in key and key.startswith("encoder."):
        return "transformer_attention"
    if key.startswith("encoder.encoder.layer."):
        return "transformer_feed_forward_or_norm"
    if key.startswith("encoder."):
        return "transformer_embeddings_or_other"
    if key.startswith("boundary_head.boundary_encoder."):
        return "boundary_span_representation"
    if key.startswith("boundary_head.shared_pool_scorer."):
        return "shared_candidate_entity_scoring"
    if key.startswith("boundary_head."):
        return "boundary_proposal_pair_or_auxiliary_head"
    return "classification_record_or_relation_head"


def spectrum_summary(s):
    energy = s * s
    return {"singular_values": s.tolist(), "spectral_norm": float(s[0]),
            "frobenius_norm": float(np.linalg.norm(s)),
            "top1_to_top2": float(s[0] / s[1]) if len(s) > 1 and s[1] else None,
            "top1_to_median": float(s[0] / np.median(s)),
            "top1_energy_fraction": float(energy[0] / energy.sum()),
            "stable_rank": float(energy.sum() / energy[0]),
            "smallest_singular_value": float(s[-1])}


def tensor_hashes(path, metadata, base):
    result = {}
    with path.open("rb") as stream:
        for key, item in metadata.items():
            if key == "__metadata__":
                continue
            start, end = item["data_offsets"]
            stream.seek(base + start)
            remaining = end - start
            digest = hashlib.sha256()
            while remaining:
                block = stream.read(min(remaining, 4 * 1024 * 1024))
                if not block:
                    raise ValueError("Truncated checkpoint")
                digest.update(block)
                remaining -= len(block)
            result[key] = digest.hexdigest()
    return result


def main():
    lock = json.loads((ROOT / "contracts/ner_v2/gliner25_research_lock.json").read_text(encoding="utf-8"))
    source_files = sorted(p for p in SOURCE.rglob("*") if p.is_file() and ".cache" not in p.parts)
    before_files = {p.relative_to(SOURCE).as_posix(): sha256(p) for p in source_files}
    for name, info in lock["files"].items():
        if before_files.get(name) != info["sha256"]:
            raise ValueError(f"Original checkpoint differs from research lock: {name}")
    if DESTINATION.exists() or REPORTS.exists():
        raise FileExistsError("Refusing to overwrite a prior model/report; this v1 is immutable")

    weights = SOURCE / "model.safetensors"
    metadata, base = header(weights)
    inventory = [{"name": k, "shape": v["shape"], "dtype": v["dtype"], "group": group(k)}
                 for k, v in metadata.items() if k != "__metadata__"]
    # Weight-only exploration: all 2D head matrices plus representative final
    # transformer attention/FFN matrices. No texts, labels, metrics or predictions.
    analyzed = {}
    for row in inventory:
        key = row["name"]
        if len(row["shape"]) != 2:
            continue
        if (key.startswith("boundary_head.") or
                (key.startswith("encoder.encoder.layer.11.") and key.endswith(".weight"))):
            w = matrix(weights, metadata, base, key)
            analyzed[key] = spectrum_summary(np.linalg.svd(w, compute_uv=False))

    w = matrix(weights, metadata, base, TARGET)
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    budget = EPSILON * np.linalg.norm(s)
    tau = (1.0 - MAX_TOP_SHRINK) * s[0]
    # Smallest tau satisfying both a 1% maximum top-value reduction and the
    # Frobenius trust region. Bisection is a deterministic scalar constraint
    # calculation, not loss optimization or parameter fitting.
    if np.linalg.norm(np.maximum(s - tau, 0.0)) > budget:
        low, high = tau, s[0]
        for _ in range(80):
            middle = (low + high) / 2
            if np.linalg.norm(np.maximum(s - middle, 0.0)) > budget:
                low = middle
            else:
                high = middle
        tau = high
    s_after_ideal = np.minimum(s, tau)
    # Add only the changed singular components; avoid reconstructing unchanged
    # components, which would introduce unnecessary numerical roundoff.
    edited = (w + (u * (s_after_ideal - s)) @ vt).astype("<f4")
    delta = edited.astype(np.float64) - w
    actual_s = np.linalg.svd(edited.astype(np.float64), compute_uv=False)
    relative = float(np.linalg.norm(delta) / np.linalg.norm(w))
    if not np.isfinite(edited).all() or not 0 < relative <= EPSILON * 1.0001:
        raise ValueError("Surgery is nonfinite, empty or exceeds the trust region")
    if not actual_s[-1] > 0 or actual_s[0] > s[0]:
        raise ValueError("Invalid post-surgery spectrum")

    DESTINATION.mkdir(parents=True, exist_ok=False)
    for source in source_files:
        destination = DESTINATION / source.relative_to(SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    new_weights = DESTINATION / "model.safetensors"
    with new_weights.open("r+b") as stream:
        stream.seek(base + metadata[TARGET]["data_offsets"][0])
        stream.write(edited.tobytes(order="C"))
        stream.flush()
        os.fsync(stream.fileno())
    new_metadata, new_base = header(new_weights)
    if new_metadata != metadata or new_base != base:
        raise ValueError("Checkpoint header changed")
    original_tensor_hashes = tensor_hashes(weights, metadata, base)
    new_tensor_hashes = tensor_hashes(new_weights, metadata, base)
    changed = [k for k in original_tensor_hashes if original_tensor_hashes[k] != new_tensor_hashes[k]]
    if changed != [TARGET]:
        raise ValueError(f"Unexpected changed tensors: {changed}")
    after_original_files = {p.relative_to(SOURCE).as_posix(): sha256(p) for p in source_files}
    if before_files != after_original_files:
        raise ValueError("Original model changed during the operation")
    destination_files = {name: sha256(DESTINATION / name) for name in before_files}
    if [name for name in before_files if before_files[name] != destination_files[name]] != ["model.safetensors"]:
        raise ValueError("A non-weight file changed")
    # Validate serialization with the actual safetensors reader, with no torch
    # import, model initialization, forward pass or gradient computation.
    from safetensors import safe_open
    with safe_open(str(new_weights), framework="numpy") as reader:
        if set(reader.keys()) != set(original_tensor_hashes):
            raise ValueError("Safetensors key mismatch")
        if not np.array_equal(reader.get_tensor(TARGET), edited):
            raise ValueError("Serialized tensor mismatch")

    REPORTS.mkdir(parents=True, exist_ok=False)
    write_json(REPORTS / "tensor_inventory.json", inventory)
    write_json(REPORTS / "spectral_analysis.json", analyzed)
    audit = {k: {"before_sha256": original_tensor_hashes[k],
                 "after_sha256": new_tensor_hashes[k], "changed": k in changed}
             for k in original_tensor_hashes}
    write_json(REPORTS / "tensor_integrity.json", audit)
    manifest = {
        "status": "CHECKPOINT_SAVED_AND_SERIALIZATION_VERIFIED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": lock["repo_id"], "source_revision": lock["revision"],
        "source_path": str(SOURCE), "destination_path": str(DESTINATION),
        "architecture": "GLiNER2.5 boundary architecture version 1; DeBERTa-v3-base encoder",
        "source_files_sha256": before_files, "destination_files_sha256": destination_files,
        "source_files_unchanged": True, "tensor_count": len(inventory),
        "changed_tensors": changed, "unchanged_tensor_count": len(inventory) - 1,
        "transformation": {
            "method": "Frobenius-nearest projection onto a spectral-norm ball with a Frobenius trust region",
            "formula": "W=U diag(s) V^T; W'=U diag(min(s_i,tau)) V^T",
            "threshold_rule": "smallest tau >= 0.99*s_1 with ||max(s-tau,0)||_2 <= 0.001*||s||_2",
            "tau": float(tau), "max_top_relative_shrink": MAX_TOP_SHRINK,
            "relative_frobenius_budget": EPSILON,
            "selection": "One active shared-pool span-scoring start projection; stronger top spectral gap than the paired end projection",
            "caveat": "A spectral gap is not evidence of harmful noise; this is a bounded experimental perturbation, not a proven NER improvement",
        },
        "changed_tensor_details": {
            "name": TARGET, "shape": list(w.shape), "dtype": "F32",
            "before": spectrum_summary(s), "after": spectrum_summary(actual_s),
            "ideal_after_singular_values": s_after_ideal.tolist(),
            "ideal_changed_singular_values": int(np.count_nonzero(s != s_after_ideal)),
            "delta_frobenius": float(np.linalg.norm(delta)),
            "relative_delta_frobenius": relative,
            "delta_spectral_norm": float(np.linalg.norm(delta, ord=2)),
            "delta_max_abs": float(np.abs(delta).max()),
            "rank_before": int(np.linalg.matrix_rank(w)),
            "rank_after": int(np.linalg.matrix_rank(edited.astype(np.float64))),
        },
        "execution": {"numpy_version": np.__version__, "blas_threads": 1,
                      "training": False, "gradients": False, "optimizers": False,
                      "model_loads": 0, "forward_passes": 0, "test_data_read": False,
                      "evaluation": "Cancelled by user: weights only",
                      "verification": "Safetensors deserialization and per-tensor SHA256; no inference"},
        "script_sha256": sha256(Path(__file__)),
    }
    write_json(DESTINATION / "weight_surgery_manifest.json", manifest)
    write_json(REPORTS / "weight_surgery_manifest.json", manifest)
    (REPORTS / "README.md").write_text(report(manifest), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "model": str(DESTINATION),
                      "changed_tensors": changed, "relative_delta_frobenius": relative,
                      "top_singular_before": float(s[0]), "top_singular_after": float(actual_s[0]),
                      "checkpoint_sha256": destination_files["model.safetensors"],
                      "report": str(REPORTS / "README.md")}, ensure_ascii=False, indent=2))


def report(m):
    d = m["changed_tensor_details"]
    return f"""# GLiNER Weight Surgery v1 — без обучения

Сохранена отдельная модель `{DESTINATION.relative_to(ROOT).as_posix()}/`.
Источник: `{m['source_repo']}@{m['source_revision']}`.
Исходные файлы проверены по исследовательскому lock и остались неизменными.
По последнему указанию пользователя тестирование отменено: только изменение весов.

## Что изменено

Один тензор `{TARGET}`, размер 128×128, float32. Остальные {m['unchanged_tensor_count']}
тензора побитово совпадают с исходными. Конфигурация, токенизатор, архитектура,
смещения, LayerNorm, embedding, encoder и остальные головы сохранены.

Это GLiNER2.5 с boundary architecture, а не классический GLiNER SpanRepLayer.
Encoder — DeBERTa-v3-base: 12 слоёв, hidden=768, 12 attention heads,
Q/K/V и output projections 768×768, FFN 3072×768 и 768×3072.
BoundaryEncoder строит представления границ 128 измерений из левого/правого
контекста, двух attention blocks и residual SwiGLU. SharedPoolScorer объединяет
начало, конец, длину, prior и span content, применяет LayerNorm и сопоставляет
кандидаты запросам типов сущностей через dot product и FiLM.
Выбранная проекция используется в этом пути до candidate_norm.
Отдельный classifier и record/relation heads не изменены.

## Математика и ограничения

Для W=U diag(s) Vᵀ, s₁≥s₂≥…≥0, применяется
W′=U diag(min(sᵢ,τ)) Vᵀ. Порог τ — минимальный в [0.99s₁,s₁], удовлетворяющий
‖max(s−τ,0)‖₂≤0.001‖s‖₂. Его вычисление не использует тексты или метки.
Это ближайшая по Фробениусу матрица в шаре ‖X‖₂≤τ: сингулярные направления
сохранены, и подрезаются только значения выше τ. Ограничение доверительной
области гарантирует ‖W′−W‖F/‖W‖F≤0.001 с учётом проверенной погрешности F32.
Для линейного слоя ‖(W′−W)x‖₂≤‖W′−W‖₂‖x‖₂. Эта локальная оценка не гарантирует
малое изменение конечных предсказаний после LayerNorm, нелинейностей и порогов.
Bias оставлен прежним. Полный ранг сохранён.

Рассмотренные альтернативы: подавление верхних компонент уменьшает максимальное
линейное усиление; мягкая фильтрация малых значений может удалить редкие полезные
направления; глобальная перестройка encoder затрагивает все представления и
disentangled attention. Поэтому выбран только ограниченный спектральный cap
одной небольшой матрицы scoring head. Гиперпараметры 1% и 0.1% — заранее
заданный консервативный бюджет, а не оценка оптимальности или шумового порога.
Разрыв s₁/s₂≈1.989 у start projection (end≈1.962) мотивирует выбор для опыта,
но не доказывает наличие аномалии. Малые компоненты оставлены неизменными
аналитически; сохранение в float32 добавляет округление.

## Измеренные изменения

- ‖ΔW‖F = {d['delta_frobenius']:.12g}
- ‖ΔW‖F/‖W‖F = {d['relative_delta_frobenius']:.12g} ({100*d['relative_delta_frobenius']:.8f}%)
- ‖ΔW‖₂ = {d['delta_spectral_norm']:.12g}
- max|ΔWᵢⱼ| = {d['delta_max_abs']:.12g}
- s₁: {d['before']['spectral_norm']:.12g} → {d['after']['spectral_norm']:.12g}
- Порог τ = {m['transformation']['tau']:.12g}
- Аналитически изменённых сингулярных значений: {d['ideal_changed_singular_values']}
- Ранг: {d['rank_before']} → {d['rank_after']}

Полные спектры до/после находятся в `weight_surgery_manifest.json`;
`spectral_analysis.json` содержит исследованные матрицы head и последнего
transformer layer; `tensor_inventory.json` — все имена, размеры и группы;
`tensor_integrity.json` — SHA256 каждого тензора до/после.
Для создания checkpoint скопирован исходный safetensors и заменены только байты
одного float32 тензора; формат и metadata сохранены. Файл прочитан настоящим
safetensors reader. Модель в память не загружалась, forward не выполнялся.
Обучение, градиенты, optimizer, loss, LoRA/PEFT не использовались.
Численные операции выполнены NumPy, BLAS ограничен одним потоком.

## Результат исследования

Получен сохранённый checkpoint с контролируемым аналитическим изменением весов.
Качество NER не измерялось по последнему указанию пользователя. Precision,
Recall, F1, confusion matrix и изменение в процентных пунктах отсутствуют;
утверждать улучшение или ухудшение нельзя.

Обнаруженный ночной тест: PID 21304/17100, `scripts/run_entity_typing_v2.py night`,
конфигурация `config/entity_typing_v2.json`, результаты
`reports/entity_typing_v2/run2_two_servers/full/`. Исходные NER предсказания:
`reports/ner_v2/research_run2/E3/mentions_035.jsonl`; корпус:
`reports/ner_v2/research_run2/input_abstracts.jsonl`; онтология:
`contracts/ner_v2/ontology.v2.json`. Тест не изменялся и не останавливался.

## Локальные первичные источники

- `artifacts/ner_v2/models/gliner25/config.json` и research lock.
- `.venv-ner-v2/Lib/site-packages/gliner2/models/boundary/encoding.py`.
- `.venv-ner-v2/Lib/site-packages/gliner2/models/boundary/pool.py`, SharedPoolScorer.
- `.venv-ner-v2/Lib/site-packages/gliner2/models/boundary/model.py`.

Повторный запуск `scripts/gliner_weight_surgery.py` отклоняется, если папка v1
уже существует: checkpoint не перезаписывается.
"""


if __name__ == "__main__":
    main()
