"""Stage 2B, local-only 9 x 2 benchmark; no entity merges or graph operations.

Run: .venv/Scripts/python.exe scripts/benchmark_semantic_embeddings.py
One subprocess per model releases all model/runtime memory before the next model.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
ER = ROOT / "reports/entity_resolution/relex_base_t035"
INPUT = ER / "semantic_resolution/input/canonical_entities.jsonl"
OUTPUT = ER / "semantic_resolution/model_comparison"
MODEL_ROOT = (ROOT.parent / "embedding-models").resolve()
MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5",
          "intfloat/e5-small-v2", "intfloat/e5-base-v2", "intfloat/e5-large-v2",
          "thenlper/gte-small", "thenlper/gte-base", "thenlper/gte-large"]
MODES = ("text_name", "text_name_context")
THREADS, BATCH, MAX_LENGTH, TOP_K = 4, 8, 512, 5
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(THREADS)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_input():
    with INPUT.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(len(rows) == len({r["canonical_entity_id"] for r in rows}) == 1122, "Expected 1122 unique canonical IDs")
    require(len({r["document_id"] for r in rows}) == 30, "Expected 30 documents")
    require(all(isinstance(r[m], str) and r[m] for r in rows for m in MODES), "Missing input text")
    require(len({i for r in rows for i in r["source_entity_ids"]}) == 1211 and
            len({i for r in rows for i in r["source_mention_ids"]}) == 1327, "Incomplete original ID coverage")
    return rows


def describe(model):
    root = MODEL_ROOT / model.replace("/", "__")
    config = next((p for p in (root/"config.json", root/"onnx/config.json") if p.is_file()), None)
    tokenizer = next((p for p in (root/"tokenizer.json", root/"onnx/tokenizer.json") if p.is_file()), None)
    weight = root/"onnx/model.onnx"
    backend = "onnxruntime"
    if not weight.is_file():
        weight, backend = root/"model.safetensors", "pytorch"
    require(config is not None and tokenizer is not None and weight.is_file(), f"Incomplete local model: {model}")
    require(backend != "pytorch" or config.parent == root, f"Missing PyTorch root configuration: {model}")
    cfg = json.loads(config.read_text(encoding="utf-8"))
    require(cfg["model_type"] == "bert" and cfg["max_position_embeddings"] >= MAX_LENGTH, f"Unsupported model: {model}")
    pooling = "cls" if model.startswith("BAAI/") else "masked_mean"
    pool_config = root/"1_Pooling/config.json"
    if pool_config.is_file():
        pc = json.loads(pool_config.read_text(encoding="utf-8"))
        require(pc["pooling_mode_cls_token"] == (pooling == "cls") and
                pc["pooling_mode_mean_tokens"] == (pooling == "masked_mean"), f"Unexpected pooling: {model}")
    used = {config, tokenizer, weight}
    for p in (pool_config, config.parent/"tokenizer_config.json", config.parent/"special_tokens_map.json", root/"sentence_bert_config.json"):
        if p.is_file():
            used.add(p)
    return dict(model=model, root=str(root), config=str(config), tokenizer=str(tokenizer), weight=str(weight),
                backend=backend, dimension=cfg["hidden_size"], pooling=pooling,
                prefix="query: " if model.startswith("intfloat/") else "", pad_id=cfg.get("pad_token_id", 0),
                files=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(used)])


class Memory:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.peak_rss = self.peak_private = 0
        self.reset()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def tick(self):
        info = self.process.memory_info()
        with self.lock:
            self.peak_rss = max(self.peak_rss, info.rss)
            self.peak_private = max(self.peak_private, getattr(info, "private", 0))

    def sample(self):
        while not self.stop_event.wait(0.05):
            self.tick()

    def reset(self):
        with self.lock:
            self.peak_rss = self.peak_private = 0
        self.tick()

    def snapshot(self):
        self.tick()
        with self.lock:
            return {"peak_rss_mib": self.peak_rss/2**20, "peak_private_mib": self.peak_private/2**20}

    def cpu(self):
        x = self.process.cpu_times()
        return x.user+x.system


def emit(kind, **data):
    print(json.dumps(dict(kind=kind, **data), ensure_ascii=False), flush=True)


def worker(model, preflight=False):
    import numpy as np
    import tokenizers
    from tokenizers import Tokenizer
    desc = describe(model)
    memory = Memory()
    load_start = time.perf_counter()
    tokenizer = Tokenizer.from_file(desc["tokenizer"])
    tokenizer.no_padding()
    tokenizer.no_truncation()
    require(tokenizer.token_to_id("[PAD]") == desc["pad_id"], "Unexpected padding token")
    if desc["backend"] == "onnxruntime":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = THREADS, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(desc["weight"], sess_options=options, providers=["CPUExecutionProvider"])
        require(session.get_providers() == ["CPUExecutionProvider"], "Unexpected ONNX execution provider")
        input_names = [i.name for i in session.get_inputs()]
        require(set(input_names) <= {"input_ids", "attention_mask", "token_type_ids"}, "Unexpected ONNX inputs")
        output_name = next((o.name for o in session.get_outputs() if len(o.shape) == 3 and o.shape[-1] == desc["dimension"]), None)
        require(output_name is not None, "Missing token-level model output")
        def forward(batch):
            return session.run([output_name], {k: batch[k] for k in input_names})[0]
        runtime_version = ort.__version__
    else:
        import torch
        import transformers
        from transformers import AutoModel
        from transformers.utils import logging as transformer_logging
        transformer_logging.disable_progress_bar()
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        model_object = AutoModel.from_pretrained(desc["root"], local_files_only=True, trust_remote_code=False,
                                                 use_safetensors=True, dtype=torch.float32).cpu().eval()
        def forward(batch):
            with torch.inference_mode():
                return model_object(**{k: torch.from_numpy(v) for k, v in batch.items()}).last_hidden_state.numpy()
        runtime_version = f"torch={torch.__version__}; transformers={transformers.__version__}"
    load_seconds = time.perf_counter()-load_start
    load_memory = memory.snapshot()

    def pack(encodings):
        width = max(len(e.ids) for e in encodings)
        batch = {"input_ids": np.full((len(encodings), width), desc["pad_id"], dtype=np.int64),
                 "attention_mask": np.zeros((len(encodings), width), dtype=np.int64),
                 "token_type_ids": np.zeros((len(encodings), width), dtype=np.int64)}
        for j, e in enumerate(encodings):
            n = len(e.ids)
            batch["input_ids"][j, :n], batch["attention_mask"][j, :n], batch["token_type_ids"][j, :n] = e.ids, e.attention_mask, e.type_ids
        return batch

    def embed(batch):
        hidden = forward(batch)
        require(hidden.dtype == np.float32 and hidden.shape[-1] == desc["dimension"], "Non-FP32 or wrong dimension")
        if desc["pooling"] == "cls":
            pooled = hidden[:, 0, :].copy()
        else:
            mask = batch["attention_mask"].astype(np.float32)[..., None]
            pooled = (hidden*mask).sum(axis=1)/mask.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        require(np.isfinite(pooled).all() and (norms > 0).all(), "Invalid embeddings")
        return np.ascontiguousarray(pooled/norms, dtype=np.float32)

    warm_start = time.perf_counter()
    probe = pack(tokenizer.encode_batch([desc["prefix"]+"scientific entity resolution", desc["prefix"]+"a biomedical research term"]))
    require(embed(probe).shape == (2, desc["dimension"]), "Local forward failed")
    warm_seconds = time.perf_counter()-warm_start
    if preflight:
        emit("preflight", model=model, backend=desc["backend"], dimension=desc["dimension"], load_seconds=load_seconds, warmup_seconds=warm_seconds)
        return
    rows = read_input()
    ids = [r["canonical_entity_id"] for r in rows]
    id_rank = np.empty(len(ids), dtype=np.int64)
    id_rank[np.argsort(np.array(ids))] = np.arange(len(ids))
    docs = np.array([r["document_id"] for r in rows])
    same = docs[:, None] == docs[None, :]
    diagonal = np.eye(len(rows), dtype=bool)
    modes = []
    for mode in MODES:
        emit("progress", model=model, mode=mode, stage="inference")
        memory.reset()
        baseline_rss = memory.process.memory_info().rss/2**20
        cpu_start, start = memory.cpu(), time.perf_counter()
        texts = [desc["prefix"]+r[mode] for r in rows]
        token_start = time.perf_counter()
        encodings = tokenizer.encode_batch(texts)
        lengths = [len(e.ids) for e in encodings]
        truncated = sum(n > MAX_LENGTH for n in lengths)
        if truncated:
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            encodings = tokenizer.encode_batch(texts)
            tokenizer.no_truncation()
        token_seconds = time.perf_counter()-token_start
        vectors = np.empty((len(rows), desc["dimension"]), dtype=np.float32)
        progress_time = time.perf_counter()
        for left in range(0, len(rows), BATCH):
            vectors[left:left+BATCH] = embed(pack(encodings[left:left+BATCH]))
            if time.perf_counter()-progress_time >= 20:
                emit("progress", model=model, mode=mode, encoded=min(left+BATCH, len(rows)), total=len(rows))
                progress_time = time.perf_counter()
        inference_seconds = time.perf_counter()-start
        inference_cpu = memory.cpu()-cpu_start
        inference_memory = memory.snapshot()
        # Same first batch, same padding/shape: an exact local determinism probe.
        repeat_start = time.perf_counter()
        repeated = embed(pack(encodings[:BATCH]))
        require(np.array_equal(vectors[:BATCH], repeated), f"Non-reproducible first-batch embeddings: {model}/{mode}")
        probe_seconds = time.perf_counter()-repeat_start
        require(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6), "Embedding normalization failure")
        memory.reset()
        search_start = time.perf_counter()
        scores = vectors @ vectors.T
        require(np.isfinite(scores).all() and np.allclose(scores, scores.T, atol=1e-6), "Invalid cosine matrix")
        np.clip(scores, -1, 1, out=scores)  # Float32 roundoff only, no candidate threshold.
        pairs = []
        for scope, mask in (("within_document", same & ~diagonal), ("cross_document", ~same)):
            for i in range(len(rows)):
                eligible = np.flatnonzero(mask[i])
                ordered = eligible[np.lexsort((id_rank[eligible], -scores[i, eligible]))[:TOP_K]]
                pairs.extend([i, int(j), scope, rank, float(scores[i, j])] for rank, j in enumerate(ordered, 1))
        search_seconds = time.perf_counter()-search_start
        search_memory = memory.snapshot()
        repeat_search = []
        for scope, mask in (("within_document", same & ~diagonal), ("cross_document", ~same)):
            for i in range(len(rows)):
                eligible = np.flatnonzero(mask[i])
                ordered = sorted(eligible.tolist(), key=lambda j: (-float(scores[i, j]), ids[j]))[:TOP_K]
                repeat_search.extend([i, j, scope, rank, float(scores[i, j])] for rank, j in enumerate(ordered, 1))
        require(pairs == repeat_search and len(pairs) == 11220, "Top-k repeat/coverage failure")
        counts = Counter((i, scope) for i, _, scope, _, _ in pairs)
        require(len(counts) == 2244 and set(counts.values()) == {5} and all(i != j for i, j, _, _, _ in pairs), "Incomplete/invalid candidates")
        stats = dict(model=model, mode=mode, backend=desc["backend"], runtime_version=runtime_version,
                     device="CPU", dtype="float32", threads=THREADS, batch_size=BATCH, max_length=MAX_LENGTH,
                     pooling=desc["pooling"], prefix=desc["prefix"], dimension=desc["dimension"],
                     entities=len(rows), documents=30, candidate_rows=len(pairs),
                     load_seconds=load_seconds, warmup_seconds=warm_seconds, tokenization_seconds=token_seconds,
                     inference_seconds=inference_seconds, inference_cpu_seconds=inference_cpu, search_seconds=search_seconds,
                     determinism_probe_seconds=probe_seconds, first_batch_repeat_equal=True, search_repeat_equal=True,
                     load_peak_rss_mib=load_memory["peak_rss_mib"], baseline_rss_mib=baseline_rss,
                     inference_peak_rss_mib=inference_memory["peak_rss_mib"], inference_peak_private_mib=inference_memory["peak_private_mib"],
                     search_peak_rss_mib=search_memory["peak_rss_mib"], embedding_matrix_mib=vectors.nbytes/2**20,
                     maximum_tokens=max(lengths), mean_tokens=sum(lengths)/len(lengths), truncated_inputs=truncated,
                     embedding_sha256=hashlib.sha256(vectors.astype("<f4", copy=False).tobytes()).hexdigest(),
                     prepared_texts_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
                     input_sha256=sha(INPUT), model_files=desc["files"],
                     numpy_version=np.__version__, tokenizers_version=tokenizers.__version__)
        for scope, mask in (("within", np.triu(same, 1)), ("cross", np.triu(~same, 1))):
            values = scores[mask]
            stats[scope+"_all_pairs_count"] = len(values)
            stats[scope+"_all_pairs_mean"] = float(values.mean())
            stats[scope+"_all_pairs_std"] = float(values.std())
            for q, value in zip(("min", "p05", "p25", "median", "p75", "p95", "max"), np.quantile(values, [0, .05, .25, .5, .75, .95, 1])):
                stats[scope+"_all_pairs_"+q] = float(value)
            chosen = [p for p in pairs if p[2] == ("within_document" if scope == "within" else "cross_document")]
            stats[scope+"_candidate_mean"] = sum(p[4] for p in chosen)/len(chosen)
            stats[scope+"_top1_mean"] = sum(p[4] for p in chosen if p[3] == 1)/len(rows)
            stats[scope+"_candidate_type_conflict_count"] = sum(set(rows[i]["entity_types"]) != set(rows[j]["entity_types"]) for i, j, _, _, _ in chosen)
            stats[scope+"_unique_unordered_candidates"] = len({tuple(sorted((p[0], p[1]))) for p in chosen})
        modes.append(dict(stats=stats, pairs=pairs))
        del scores, vectors, encodings
        emit("progress", model=model, mode=mode, stage="complete", inference_seconds=inference_seconds, search_seconds=search_seconds)
    emit("result", model=model, configurations=modes)


def run_worker(model, preflight=False):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", model]
    if preflight:
        command.append("--preflight")
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    result = None
    try:
        for line in process.stdout:
            event = json.loads(line)
            if event["kind"] == "result":
                result = event["configurations"]
            else:
                print(json.dumps(event, ensure_ascii=False), flush=True)
                if event["kind"] == "preflight":
                    result = event
        require(process.wait() == 0 and result is not None, f"Local loading/inference failed: {model}; no incomplete comparison published")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    return result


def write_csv(path, records, fields=None):
    iterator = iter(records)
    first = next(iterator)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(first))
        writer.writeheader()
        for row in (first, *iterator) if isinstance(records, list) else _prepend(first, iterator):
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def _prepend(first, iterator):
    yield first
    yield from iterator


def main():
    import psutil
    start = time.perf_counter()
    rows = read_input()
    protected = {INPUT, ROOT/"reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl",
                 ROOT/"reports/re_v2/relex_comparison/threshold035/e3_base/relations.jsonl",
                 ROOT/"reports/ner_v2/research_run2/input_abstracts.jsonl"}
    for subdir in ("exact_matching_audit", "conservative_resolution", "relation_remapping", "graph_comparison", "semantic_resolution/input"):
        protected.update(p for p in (ER/subdir).iterdir() if p.is_file())
    protected_hashes = {p: sha(p) for p in sorted(protected)}
    descriptions = [describe(model) for model in MODELS]  # Check every model before any inference.
    print("All nine local file manifests validated; beginning sequential loading checks.", flush=True)
    preflight = [run_worker(model, True) for model in MODELS]
    print("All nine local loads and forward checks passed; beginning 18 configurations.", flush=True)
    configurations = []
    for model in MODELS:
        configurations.extend(run_worker(model))
    require(len(configurations) == 18 and len({(c["stats"]["model"], c["stats"]["mode"]) for c in configurations}) == 18, "Incomplete benchmark")
    neighbors = {}
    for c in configurations:
        sets = defaultdict(set)
        for i, j, scope, _, _ in c["pairs"]:
            sets[i, scope].add(j)
        neighbors[c["stats"]["model"], c["stats"]["mode"]] = sets
    for c in configurations:
        stats = c["stats"]
        own = neighbors[stats["model"], stats["mode"]]
        other_mode = neighbors[stats["model"], MODES[1] if stats["mode"] == MODES[0] else MODES[0]]
        for scope, short in (("within_document", "within"), ("cross_document", "cross")):
            stats[short+"_name_context_top5_overlap"] = sum(len(own[i, scope] & other_mode[i, scope])/TOP_K for i in range(len(rows)))/len(rows)
            agreement = []
            for other in MODELS:
                if other != stats["model"]:
                    compared = neighbors[other, stats["mode"]]
                    agreement.extend(len(own[i, scope] & compared[i, scope])/len(own[i, scope] | compared[i, scope]) for i in range(len(rows)))
            stats[short+"_other_models_mean_jaccard"] = sum(agreement)/len(agreement)
    require(all(sha(p) == digest for p, digest in protected_hashes.items()), "An input/stage 1A-2A artifact changed")
    for d in descriptions:
        require(all(sha(Path(f["path"])) == f["sha256"] for f in d["files"]), f"Model files changed: {d['model']}")
    def pair_rows():
        for c in configurations:
            stats = c["stats"]
            for i, j, scope, rank, score in c["pairs"]:
                a, b = rows[i], rows[j]
                yield dict(model=stats["model"], mode=stats["mode"], scope=scope, rank=rank, similarity=f"{score:.9f}",
                           query_canonical_id=a["canonical_entity_id"], candidate_canonical_id=b["canonical_entity_id"],
                           query_name=a["surface_text"], candidate_name=b["surface_text"], query_types=a["entity_types"], candidate_types=b["entity_types"],
                           query_document_id=a["document_id"], candidate_document_id=b["document_id"],
                           query_source_entity_ids=a["source_entity_ids"], candidate_source_entity_ids=b["source_entity_ids"],
                           query_source_mention_ids=a["source_mention_ids"], candidate_source_mention_ids=b["source_mention_ids"],
                           query_quality_flags=a["quality_flags"], candidate_quality_flags=b["quality_flags"],
                           type_set_conflict=set(a["entity_types"]) != set(b["entity_types"]), types_disjoint=not set(a["entity_types"]) & set(b["entity_types"]),
                           damaged_model_input="\ufffd" in a[stats["mode"]] or "\ufffd" in b[stats["mode"]],
                           same_name_cross_document=scope == "cross_document" and a["surface_text"].casefold() == b["surface_text"].casefold(),
                           source_input=INPUT.relative_to(ROOT).as_posix(), query_input_line=i+1, candidate_input_line=j+1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT/"candidate_pairs.csv", pair_rows())
    write_csv(OUTPUT/"model_comparison.csv", [c["stats"] for c in configurations])
    total_seconds = time.perf_counter()-start
    lines = ["# Этап 2B — benchmark embedding-моделей", "",
             f"Завершены все 18 конфигураций: 9 моделей × text_name/text_name_context, одни и те же 1122 сущности из 30 документов. Суммарное время, включая проверку загрузки, хеширование и запись CSV: {total_seconds:.2f} с. Entity Resolution, изменение графа и Relation Verification не выполнялись.", "",
             "## Протокол", "",
             f"CPU: {platform.processor()}, {psutil.cpu_count(logical=False)} физических / {psutil.cpu_count()} логических ядер; RAM {psutil.virtual_memory().total/2**30:.2f} GiB. ОС: {platform.platform()}. Для каждого процесса: {THREADS} вычислительных потока, batch={BATCH}, FP32, max_length={MAX_LENGTH}; GPU не использовался. Порядок моделей — BGE small/base/large, E5 small/base/large, GTE small/base/large; внутри модели сначала название, затем контекст. Нагрузка машины не изолирована.", "",
             "Все файлы проверены до инференса; все 9 локальных загрузок и коротких forward-pass прошли до основного сравнения. Модели запускались последовательно в отдельных процессах; завершение процесса освобождает веса/арены памяти. Загрузок из сети, новых зависимостей и дообучения нет. E5-small использует локальный PyTorch FP32 model.safetensors; остальные — локальные ONNX FP32 model.onnx. Это различие движков ограничивает прямое сравнение времени и RSS как свойств самих моделей.", "",
             "Рекомендации авторов: [BGE](https://huggingface.co/BAAI/bge-base-en-v1.5) — CLS pooling, без retrieval-инструкции для симметричного сравнения; [E5](https://huggingface.co/intfloat/e5-base-v2) — masked mean pooling и query: для обеих сторон; [GTE](https://huggingface.co/thenlper/gte-base) — masked mean pooling, без префикса. Используется tokenizer.json соответствующей модели, включая собственную нормализацию tokenizer; поля 2A не изменяются. Все векторы L2-нормированы. Рецепты BGE/E5/GTE применяются одинаково в двух режимах.", "",
             "Для каждого вектора вычисляется полный cosine со всеми 1122 векторами своей конфигурации. Сама сущность исключается; выбираются независимо top-5 внутри документа и top-5 между документами. Ровно 11220 направленных рекомендаций на конфигурацию, всего 201960 строк; взаимные рекомендации сохраняются отдельно. При равных float32 scores порядок определяется canonical ID. Нет порога similarity и фильтрации конфликтов/повреждений.", "",
             "inference_seconds включает токенизацию, padding, forward, pooling и нормализацию всех 1122 текстов; tokenization_seconds указан отдельно. Загрузка и warmup исключены из этого времени. search_seconds включает cosine-матрицу и ранжирование, без вычисления описательных квантилей и без повторной проверки. RSS/private — максимумы, измеренные каждые 50 мс для свежего дочернего процесса; RSS включает модель, runtime и арены, поэтому это не размер весов. load_seconds/RSS относятся к одной загрузке на модель и повторены в обеих её строках: складывать их дважды нельзя. Дисперсию времени по независимым повторным запускам не оценивали.", "",
             "## Скорость и память", "",
             "| Модель | Режим | Движок | d | Инференс, с | Поиск, с | Пик RSS инференса, MiB | Усечение |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for c in configurations:
        s = c["stats"]
        lines.append(f"| {s['model']} | {s['mode']} | {s['backend']} | {s['dimension']} | {s['inference_seconds']:.2f} | {s['search_seconds']:.3f} | {s['inference_peak_rss_mib']:.1f} | {s['truncated_inputs']} |")
    lines += ["", "## Распределение cosine similarity", "",
              "Ниже квантили всех неупорядоченных пар без самосравнений, отдельно внутри/между документами (не только отобранные top-5). Средние и остальные квантили, включая распределения кандидатов, записаны в model_comparison.csv. Шкалы разных моделей не калиброваны друг относительно друга.", "",
              "| Модель | Режим | Внутри: P05 / P50 / P95 | Между: P05 / P50 / P95 | Конфликты типов в top-5: внутри / между |", "|---|---|---|---|---:|"]
    for c in configurations:
        s = c["stats"]
        q = lambda prefix: " / ".join(f"{s[prefix+'_all_pairs_'+p]:.3f}" for p in ("p05", "median", "p95"))
        lines.append(f"| {s['model']} | {s['mode']} | {q('within')} | {q('cross')} | {s['within_candidate_type_conflict_count']} / {s['cross_candidate_type_conflict_count']} |")
    lines += ["", "## Изменение состава кандидатов", "",
              "Доля совпадения top-5 названия и контекста = |пересечение|/5, усреднённая по сущностям. Межмодельный Jaccard — |пересечение|/|объединение|, усреднённый по сущностям и восьми другим моделям в том же режиме; это согласие списков, не точность.", "",
              "| Модель | Совпадение name/context: внутри / между | Jaccard с другими моделями, контекст: внутри / между |", "|---|---:|---:|"]
    for model in MODELS:
        s = next(c["stats"] for c in configurations if c["stats"]["model"] == model and c["stats"]["mode"] == "text_name_context")
        lines.append(f"| {model} | {s['within_name_context_top5_overlap']:.3f} / {s['cross_name_context_top5_overlap']:.3f} | {s['within_other_models_mean_jaccard']:.3f} / {s['cross_other_models_mean_jaccard']:.3f} |")
    lines += ["", "## Примеры кандидатов и рисков", "",
              "Примеры ниже выбираются только из реально полученных top-5. Контекстные основания — иллюстрации для просмотра, не независимая gold-разметка и не разрешение на merge.", "",
              "| Категория | Модель / режим / область | Названия | cosine | Строки входа 2A | Основание/риск |", "|---|---|---|---:|---|---|"]
    examples = [({"mammographic density", "md"}, "Определение аббревиатуры в исходном документе; перспективный кандидат, требуется проверка типов и референта."),
                ({"penetrating keratoplasty", "pk"}, "В документе явно раскрыта PK; кандидат на проверку, не автоматическое тождество."),
                ({"breast cancer", "breast cancer risk"}, "Заболевание и риск заболевания тематически связаны, но это разные сущности."),
                ({"limonene", "limonene synthase gene"}, "Вещество и ген/фермент нельзя приравнивать по общей теме или включению названия.")]
    for names, note in examples:
        found = [(c, p) for c in configurations for p in c["pairs"] if {rows[p[0]]["surface_text"].casefold(), rows[p[1]]["surface_text"].casefold()} == names]
        if found:
            c, p = max(found, key=lambda x: (x[0]["stats"]["mode"] == "text_name_context", x[1][4]))
            a, b = rows[p[0]], rows[p[1]]
            lines.append(f"| Контекстный пример | {c['stats']['model']} / {c['stats']['mode']} / {p[2]} | {a['surface_text']} ↔ {b['surface_text']} | {p[4]:.6f} | {p[0]+1}, {p[1]+1} | {note} |")
    for label, predicate, note in [
        ("Одинаковое имя между документами", lambda a,b: a["surface_text"].casefold() == b["surface_text"].casefold() and a["document_id"] != b["document_id"], "Одинаковое имя, например patients, не доказывает одинаковую когорту/референт."),
        ("Повреждённый вход", lambda a,b: "damaged_name" in a["quality_flags"] or "damaged_name" in b["quality_flags"], "Повреждённые формулы сохраняются; близкие векторы могут отражать одинаковые дефекты текста.")]:
        matches = [(c,p) for c in configurations for p in c["pairs"] if predicate(rows[p[0]],rows[p[1]])]
        c,p = max(matches, key=lambda x:x[1][4])
        a,b = rows[p[0]],rows[p[1]]
        lines.append(f"| {label} | {c['stats']['model']} / {c['stats']['mode']} / {p[2]} | {a['surface_text']} ↔ {b['surface_text']} | {p[4]:.6f} | {p[0]+1}, {p[1]+1} | {note} |")
    lines += ["", "## Ограничения и сохранность", "",
              "Независимой проверенной разметки пар нет: Precision/Recall/F1, порог автоматического объединения и победитель по качеству не объявляются. Больший средний cosine и межмодельное согласие не доказывают правильность. Большие модели могут требовать больше ресурсов; измеренные скорость/RSS зависят также от backend и нагрузки CPU. Общий контекст способен поднимать тематически связанные, но разные термины. Контекст 2A — только окно представителя, а не все возможные значения термина.", "",
              "Все 15 повреждённых названий, 40 повреждённых контекстов, 108 множественных типов и 10 partial_token сохранены без фильтрации. candidate_pairs.csv содержит исходные entity/mention IDs обеих сторон, типы, документы, quality_flags, конфликт/непересечение типов и признак повреждения фактического текста режима. Контексты и полные provenance восстанавливаются по source_input + query_input_line/candidate_input_line и canonical ID; текст 2A не переписывается.", "",
              "Проверены все 18 конфигураций и одинаковое покрытие 1122 ID, отсутствие самопар, правильность области поиска и рангов 1–5, конечность и единичные нормы векторов, симметрия cosine-матрицы, детерминированные ties. В каждом режиме повторный forward первого полного batch дал идентичные векторы, а независимая повторная сортировка — те же top-5. Контрольные суммы векторов, фактических текстов с префиксом, файлов моделей и версий runtime записаны в model_comparison.csv. Времена и RSS при полном повторном запуске могут изменяться. Все хеши исходников и результатов 1A–2A проверены; веса моделей не изменены. Этап 2C и объединение сущностей не выполнялись.", "",
              "Запуск: `.venv/Scripts/python.exe scripts/benchmark_semantic_embeddings.py`. Только существующие numpy, tokenizers, onnxruntime, psutil и torch/transformers для E5-small; ничего не устанавливается. Векторы/матрицы существуют в памяти и отдельными файлами не сохраняются.", "", "## Идентичность моделей", "",
              "Полные пути, SHA-256 каждого использованного файла, размеры и версии runtime доступны в model_comparison.csv. Корень моделей: `../embedding-models`.", "",
              "| Модель | Файл весов | SHA-256 |", "|---|---|---|"]
    for d in descriptions:
        weight = next(f for f in d["files"] if f["path"] == d["weight"])
        lines.append(f"| {d['model']} | {Path(d['weight']).relative_to(MODEL_ROOT).as_posix()} | {weight['sha256']} |")
    lines += ["", "SHA-256 защищённых входов и результатов 1A–2A:", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}`: `{digest}`" for p,digest in protected_hashes.items()]
    (OUTPUT/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    require(all(sha(p) == digest for p,digest in protected_hashes.items()), "Protected file changed while writing results")
    print(json.dumps(dict(configurations=18, candidate_rows=sum(len(c["pairs"]) for c in configurations),
                          total_seconds=total_seconds, output=str(OUTPUT)), indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], "--preflight" in sys.argv)
    else:
        main()
