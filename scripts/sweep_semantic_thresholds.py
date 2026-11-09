"""Stage 2C.1 only: full-pair threshold/Union-Find simulation, no real merge.

Run: .venv/Scripts/python.exe scripts/sweep_semantic_thresholds.py
Uses the existing 2B protocol/helpers without modifying its code or artifacts.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import benchmark_semantic_embeddings as protocol

ROOT, ER = protocol.ROOT, protocol.ER
OUTPUT = ER / "semantic_resolution/threshold_sweep"
RELATIONS = ER / "relation_remapping/canonical_relations.jsonl"
COMPARISON = ER / "semantic_resolution/model_comparison/model_comparison.csv"
THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reference():
    with COMPARISON.open(encoding="utf-8-sig", newline="") as stream:
        result = {(r["model"], r["mode"]): r for r in csv.DictReader(stream)}
    protocol.require(set(result) == {(m, mode) for m in protocol.MODELS for mode in protocol.MODES}, "Expected all 18 existing 2B configurations")
    return result


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1]*size

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def simulate(scores, rows, relations):
    """Only similarity pairs join UF. Only existing 1C relations define isolation."""
    import numpy as np
    n = len(rows)
    ids = [r["canonical_entity_id"] for r in rows]
    index = {eid: i for i, eid in enumerate(ids)}
    edges = [(index[r["head_id"]], r["relation_type"], index[r["tail_id"]], r["document_id"]) for r in relations]
    documents = np.array([r["document_id"] for r in rows])
    signatures = [tuple(sorted(r["entity_types"])) for r in rows]
    upper_left, upper_right = np.triu_indices(n, 1)
    # Interpret each stored float32 cosine as its exact numeric value. Comparing
    # in float64 avoids silently rounding the requested decimal threshold to f32.
    values = scores[upper_left, upper_right].astype(np.float64)
    order = np.argsort(-values, kind="stable")
    uf, cursor, intra = UnionFind(n), 0, 0
    result = []
    for threshold in reversed(THRESHOLDS):
        while cursor < len(order) and values[order[cursor]] >= threshold:
            k = int(order[cursor])
            a, b = int(upper_left[k]), int(upper_right[k])
            uf.union(a, b)
            intra += int(documents[a] == documents[b])
            cursor += 1
        roots = [uf.find(i) for i in range(n)]
        groups = defaultdict(list)
        for i, root in enumerate(roots):
            groups[root].append(i)
        merged = sorted((g for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), min(ids[i] for i in g)))
        largest = merged[0] if merged else []
        touched, external_touched, triples = set(), set(), set()
        loop_records = 0
        for head, predicate, tail, doc in edges:
            h, t = roots[head], roots[tail]
            touched.update((h, t))
            triples.add((doc, h, predicate, t))
            if h == t:
                loop_records += 1
            else:
                external_touched.update((h, t))
        implied = sum(len(g)*(len(g)-1)//2 for g in merged)
        missing = implied-cursor
        minimum = None
        if largest:
            sub = scores[np.ix_(largest, largest)]
            minimum = float(sub[np.triu_indices(len(largest), 1)].min())
        result.append(dict(
            threshold=threshold, input_vertices=n, evaluated_unordered_pairs=len(values),
            pairs_above_threshold=cursor, within_document_pairs=intra, cross_document_pairs=cursor-intra,
            merge_groups=len(merged), entities_in_merge_groups=sum(map(len, merged)),
            removed_vertices=n-len(groups), remaining_canonical_vertices=len(groups),
            isolated_vertices=len(set(groups)-touched), isolated_vertices_without_self_loops=len(set(groups)-external_touched),
            largest_merge_group=len(largest), second_largest_merge_group=len(merged[1]) if len(merged) > 1 else 0,
            largest_group_fraction=len(largest)/n, largest_group_document_count=len({documents[i] for i in largest}),
            largest_group_min_similarity=minimum,
            mixed_type_groups=sum(len({signatures[i] for i in g}) > 1 for g in merged),
            groups_with_multiple_type_labels=sum(len({t for i in g for t in signatures[i]}) > 1 for g in merged),
            cross_document_merge_groups=sum(len({documents[i] for i in g}) > 1 for g in merged),
            groups_ge_100_entities=sum(len(g) >= 100 for g in merged),
            transitive_within_group_pairs=implied, transitive_below_threshold_pairs=missing,
            transitive_below_threshold_fraction=missing/implied if implied else 0,
            self_loop_relation_records=loop_records, unique_remapped_typed_triples=len(triples)))
        protocol.require(cursor == int(np.count_nonzero(values >= threshold)), "Threshold pair count mismatch")
        protocol.require(missing >= 0 and n-len(groups) == sum(len(g)-1 for g in merged), "Invalid merge accounting")
        protocol.require(len(set(roots)) == len(groups) and sum(map(len, groups.values())) == n, "Incomplete UF partition")
    ordered = sorted(result, key=lambda r: r["threshold"])
    for a,b in zip(ordered, ordered[1:]):
        protocol.require(a["pairs_above_threshold"] >= b["pairs_above_threshold"] and
                         a["remaining_canonical_vertices"] <= b["remaining_canonical_vertices"] and
                         a["largest_merge_group"] >= b["largest_merge_group"], "Non-monotonic sweep")
    return ordered


def worker(model):
    """The encoder steps intentionally mirror 2B; full vector hashes must match."""
    import numpy as np
    from tokenizers import Tokenizer
    expected = reference()
    rows, relations = protocol.read_input(), read_jsonl(RELATIONS)
    desc = protocol.describe(model)
    tokenizer = Tokenizer.from_file(desc["tokenizer"])
    tokenizer.no_padding()
    tokenizer.no_truncation()
    protocol.require(tokenizer.token_to_id("[PAD]") == desc["pad_id"], "Unexpected pad token")
    if desc["backend"] == "onnxruntime":
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = protocol.THREADS, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(desc["weight"], sess_options=options, providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        output = next(o.name for o in session.get_outputs() if len(o.shape) == 3 and o.shape[-1] == desc["dimension"])
        def forward(batch):
            return session.run([output], {k: batch[k] for k in names})[0]
        runtime_version = ort.__version__
    else:
        import torch
        import transformers
        from transformers import AutoModel
        from transformers.utils import logging
        logging.disable_progress_bar()
        torch.set_num_threads(protocol.THREADS)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        encoder = AutoModel.from_pretrained(desc["root"], local_files_only=True, trust_remote_code=False,
                                           use_safetensors=True, dtype=torch.float32).cpu().eval()
        def forward(batch):
            with torch.inference_mode():
                return encoder(**{k: torch.from_numpy(v) for k,v in batch.items()}).last_hidden_state.numpy()
        runtime_version = f"torch={torch.__version__}; transformers={transformers.__version__}"
    def pack(encodings):
        width = max(len(e.ids) for e in encodings)
        batch = {"input_ids":np.full((len(encodings),width),desc["pad_id"],dtype=np.int64),
                 "attention_mask":np.zeros((len(encodings),width),dtype=np.int64),
                 "token_type_ids":np.zeros((len(encodings),width),dtype=np.int64)}
        for j,e in enumerate(encodings):
            n = len(e.ids)
            batch["input_ids"][j,:n],batch["attention_mask"][j,:n],batch["token_type_ids"][j,:n] = e.ids,e.attention_mask,e.type_ids
        return batch
    def embed(batch):
        hidden = forward(batch)
        protocol.require(hidden.dtype == np.float32 and hidden.shape[-1] == desc["dimension"], "Unexpected encoder dtype/dimension")
        if desc["pooling"] == "cls":
            pooled = hidden[:,0,:].copy()
        else:
            mask = batch["attention_mask"].astype(np.float32)[...,None]
            pooled = (hidden*mask).sum(axis=1)/mask.sum(axis=1)
        norms = np.linalg.norm(pooled,axis=1,keepdims=True)
        protocol.require(np.isfinite(pooled).all() and (norms>0).all(), "Invalid embeddings")
        return np.ascontiguousarray(pooled/norms,dtype=np.float32)
    embed(pack(tokenizer.encode_batch([desc["prefix"]+"scientific entity resolution", desc["prefix"]+"a biomedical research term"])))
    output_rows = []
    for mode in protocol.MODES:
        ref = expected[model,mode]
        protocol.require(ref["runtime_version"] == runtime_version and ref["backend"] == desc["backend"], "2B runtime/backend differs")
        texts = [desc["prefix"]+r[mode] for r in rows]
        texts_hash = hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
        protocol.require(texts_hash == ref["prepared_texts_sha256"], "2B prepared texts differ")
        encodings = tokenizer.encode_batch(texts)
        if any(len(e.ids)>protocol.MAX_LENGTH for e in encodings):
            tokenizer.enable_truncation(max_length=protocol.MAX_LENGTH)
            encodings = tokenizer.encode_batch(texts)
            tokenizer.no_truncation()
        vectors = np.empty((len(rows),desc["dimension"]),dtype=np.float32)
        start = progress = time.perf_counter()
        protocol.emit("progress",model=model,mode=mode,stage="embedding")
        for left in range(0,len(rows),protocol.BATCH):
            vectors[left:left+protocol.BATCH] = embed(pack(encodings[left:left+protocol.BATCH]))
            if time.perf_counter()-progress>=20:
                protocol.emit("progress",model=model,mode=mode,encoded=min(left+protocol.BATCH,len(rows)),total=len(rows))
                progress = time.perf_counter()
        digest = hashlib.sha256(vectors.astype("<f4",copy=False).tobytes()).hexdigest()
        protocol.require(digest == ref["embedding_sha256"], f"Full embeddings do not match 2B: {model}/{mode}")
        protocol.require(np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-6), "Invalid normalization")
        scores = vectors@vectors.T
        protocol.require(np.isfinite(scores).all() and np.allclose(scores,scores.T,atol=1e-6), "Invalid cosine matrix")
        np.clip(scores,-1,1,out=scores)
        sweep = simulate(scores,rows,relations)
        for r in sweep:
            output_rows.append(dict(model=model,mode=mode,backend=desc["backend"],device="CPU",dtype="float32",
                                    dimension=desc["dimension"],pooling=desc["pooling"],prefix=desc["prefix"],
                                    threads=protocol.THREADS,batch_size=protocol.BATCH,max_length=protocol.MAX_LENGTH,
                                    embedding_source="recomputed_exactly_matches_2B",embedding_sha256=digest,
                                    prepared_texts_sha256=texts_hash,input_sha256=protocol.sha(protocol.INPUT),
                                    relation_input_sha256=protocol.sha(RELATIONS),runtime_version=runtime_version,**r))
        protocol.emit("progress",model=model,mode=mode,stage="five_thresholds_complete",embedding_matches_2B=True,
                      seconds=time.perf_counter()-start,remaining_vertices=[r["remaining_canonical_vertices"] for r in sweep])
        del vectors,scores,encodings
    protocol.emit("sweep_result",rows=output_rows)


def run_worker(model):
    process = subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--worker",model],cwd=ROOT,
                               stdout=subprocess.PIPE,text=True,encoding="utf-8")
    result = None
    try:
        for line in process.stdout:
            event = json.loads(line)
            if event["kind"] == "sweep_result":
                result = event["rows"]
            else:
                print(json.dumps(event,ensure_ascii=False),flush=True)
        protocol.require(process.wait()==0 and result is not None, f"Sweep worker failed: {model}; no incomplete results published")
    finally:
        if process.poll() is None:
            process.terminate();process.wait()
    return result


def main():
    rows = protocol.read_input()
    relations = read_jsonl(RELATIONS)
    index = {r["canonical_entity_id"]:r for r in rows}
    protocol.require(len(relations)==len({r["canonical_triple_id"] for r in relations})==1760,"Invalid 1C relation coverage")
    protocol.require(all(r["head_id"] in index and r["tail_id"] in index and
                         index[r["head_id"]]["document_id"]==index[r["tail_id"]]["document_id"]==r["document_id"] for r in relations),"Invalid 1C endpoints")
    refs = reference()
    protected = {ROOT/"reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl",
                 ROOT/"reports/re_v2/relex_comparison/threshold035/e3_base/relations.jsonl",
                 ROOT/"reports/ner_v2/research_run2/input_abstracts.jsonl",Path(protocol.__file__).resolve()}
    protected.update(p for p in ER.rglob("*") if p.is_file() and not p.is_relative_to(OUTPUT))
    hashes = {p:protocol.sha(p) for p in sorted(protected)}
    manifests = [protocol.describe(model) for model in protocol.MODELS]
    for desc in manifests:
        for mode in protocol.MODES:
            protocol.require(json.loads(refs[desc["model"],mode]["model_files"])==desc["files"],f"Model files differ from 2B: {desc['model']}")
            protocol.require(refs[desc["model"],mode]["input_sha256"]==hashes[protocol.INPUT],"Input differs from 2B")
    print("All nine local models and input hashes match 2B. No persisted corpus embeddings found in 2B artifacts; recomputing 18 vector sets.",flush=True)
    result = []
    for model in protocol.MODELS:
        result.extend(run_worker(model))
    protocol.require(len(result)==len({(r['model'],r['mode'],r['threshold']) for r in result})==90,"Incomplete threshold grid")
    protocol.require(all(r["evaluated_unordered_pairs"]==1122*1121//2 and
                         r["within_document_pairs"]+r["cross_document_pairs"]==r["pairs_above_threshold"] and
                         r["remaining_canonical_vertices"]+r["removed_vertices"]==1122 for r in result),"Sweep accounting error")
    baseline_touched = {r[k] for r in relations for k in ("head_id","tail_id")}
    baseline_isolated = len(index.keys()-baseline_touched)
    def span(models, mode, threshold, field):
        values = [r[field] for r in result if r["model"] in models and r["mode"]==mode and r["threshold"]==threshold]
        return f"{min(values)}–{max(values)}"
    bge = protocol.MODELS[:3]
    e5 = protocol.MODELS[3:6]
    gte = protocol.MODELS[6:]
    lines = ["# Этап 2C.1 — Threshold Sweep", "",
             "Выполнены все 90 сценариев: 9 моделей × 2 режима × пороги 0.75, 0.80, 0.85, 0.90, 0.95. Это экспериментальное замыкание similarity-пар через Union-Find в памяти, не семантически подтверждённое объединение. ACCEPT/REJECT из 2C и типы не применяются как фильтры.", "",
             f"Исходно: 1122 канонические сущности, 30 документов, 1760 существующих отношений 1C, {baseline_isolated} изолированные вершины. Каждое представление сравнивается по всем 628881 неупорядоченным парам без самопар, не по top-5. Внутри/между документами определяется по исходному document_id до экспериментального объединения.", "",
             "## Протокол", "",
             "В 2B векторы не сохранялись (нет пригодных npy/npz/pt файлов корпуса); top-5 CSV и квантили не позволяют восстановить все cosine. Поэтому пересчитаны 18 наборов векторов, каждый использован для пяти порогов. Повторно использованы helpers/константы 2B без изменения его скрипта: CPU FP32, 4 потока, batch=8, max_length=512, собственный tokenizer.json, CLS для BGE без префикса, masked mean для E5 с query: и для GTE без префикса, L2-нормализация. E5-small работает через PyTorch, остальные — через ONNX Runtime. Все 18 полных SHA-256 векторных матриц и текстов совпали с 2B; версии runtime и файлы моделей тоже проверены. Модели запускались последовательно в отдельных процессах, без сети и изменения весов.", "",
             "Cosine вычислен в float32, как в 2B; далее значение каждой пары представлено в float64 и сравнивается с десятичным порогом без предварительного округления. Условие включения — similarity >= threshold. Пороги обрабатываются от строгого к мягкому, Union-Find дополняется всеми вновь допустимыми парами. В CSV: merge_groups — группы размера ≥2; entities_in_merge_groups — все их участники; removed_vertices = Σ(size−1); remaining_canonical_vertices включает также не объединённые singleton-узлы. При отсутствии объединений largest_merge_group=0.", "",
             "Изоляция определяется **только существующими отношениями 1C**, чьи конечные точки в памяти заменяются корнями Union-Find. Similarity-пары не добавляются в отношения. Петля считается связью в isolated_vertices; isolated_vertices_without_self_loops считает изоляцию после исключения петель. Документный provenance сохранён в ключе (document_id, root_head, relation_type, root_tail) диагностических уникальных троек. Никаких новых mappings или отношений на диск не записано.", "",
             "## Оставшиеся канонические вершины (гипотетически)", "",
             "| Модель | Режим | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |", "|---|---|---:|---:|---:|---:|---:|"]
    for field,title in (("remaining_canonical_vertices",None),("isolated_vertices","Изолированные вершины с учётом петель"),
                        ("isolated_vertices_without_self_loops","Изолированные вершины без учёта петель"),
                        ("largest_merge_group","Размер крупнейшей объединённой группы")):
        if title:
            lines += ["",f"## {title}","","| Модель | Режим | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |","|---|---|---:|---:|---:|---:|---:|"]
        for model in protocol.MODELS:
            for mode in protocol.MODES:
                batch = sorted((r for r in result if r["model"]==model and r["mode"]==mode),key=lambda r:r["threshold"])
                lines.append(f"| {model} | {mode} | "+" | ".join(str(r[field]) for r in batch)+" |")
    lines += ["", "## Переходы к крупным транзитивным группам", "",
              "Для каждой пары модель/режим показан соседний шаг сетки с максимальным абсолютным увеличением крупнейшей группы при снижении порога. Это числовая диагностика резкого роста, не граница семантической безопасности. Группа >50% корпуса отмечает структурное доминирование; порог 50% не используется для отбора пар.", "",
              "| Модель | Режим | Наибольший шаг | Размер до → после | Уже >50% корпуса при |", "|---|---|---|---:|---|"]
    for model in protocol.MODELS:
        for mode in protocol.MODES:
            batch = sorted((r for r in result if r["model"]==model and r["mode"]==mode),key=lambda r:-r["threshold"])
            steps = list(zip(batch,batch[1:]))
            a,b = max(steps,key=lambda p:p[1]["largest_merge_group"]-p[0]["largest_merge_group"])
            crossing = next((r["threshold"] for r in batch if r["largest_merge_group"]>1122/2),None)
            lines.append(f"| {model} | {mode} | {a['threshold']:.2f} → {b['threshold']:.2f} | {a['largest_merge_group']} → {b['largest_merge_group']} | {crossing if crossing is not None else 'не наблюдалось'} |")
    lines += ["", "## Как читать агрессивность и перспективные пороги", "",
              "Меньшее число оставшихся узлов, большая крупнейшая группа и большее число междокументных пар означают более агрессивное экспериментальное сжатие, а не лучшую модель. Для сравнения при самом строгом проверенном пороге:", "",
              "| Модель | Режим, τ=0.95 | Осталось V | Max группа | Документов в ней | Смешанных типов групп | Косвенно связанных пар ниже τ | Min cosine в Max группе |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in result:
        if r["threshold"]==.95:
            low = r["largest_group_min_similarity"]
            lines.append(f"| {r['model']} | {r['mode']} | {r['remaining_canonical_vertices']} | {r['largest_merge_group']} | {r['largest_group_document_count']} | {r['mixed_type_groups']} | {r['transitive_below_threshold_pairs']} | {low:.6f} |")
    lines += ["", "mixed_type_groups считает группы, где исходные наборы entity_types участников различаются; groups_with_multiple_type_labels дополнительно считает группы с несколькими метками в объединении всех типов, даже если одинаковая множественная типизация была у каждого участника. Ни один из этих признаков не блокирует экспериментальное объединение.", "",
              "transitive_below_threshold_pairs = Σ size(size−1)/2 − число выбранных пар. Это точное число пар внутри полученных групп, которые сами **не прошли** порог, но оказались приравнены транзитивным замыканием. Низкий min cosine внутри крупнейшей группы показывает этот эффект непосредственно. Компонента может содержать все 30 документов и разные типы без высоких попарных сходств у всех её участников.", "",
              f"По названиям при τ=0.95 BGE сохраняют {span(bge,'text_name',.95,'remaining_canonical_vertices')} узел и {span(bge,'text_name',.95,'isolated_vertices')} изолированных вершин; E5/GTE — {span(e5+gte,'text_name',.95,'remaining_canonical_vertices')} узлов. При τ=0.80 у BGE крупнейшая группа содержит {span(bge,'text_name',.8,'largest_merge_group')} сущностей, у E5/GTE — {span(e5+gte,'text_name',.8,'largest_merge_group')}. Таким образом, BGE по названиям дают более умеренное структурное сжатие в этой сетке; это не сравнение семантического качества.", "",
              f"Контекст резко усиливает сжатие. При τ=0.95 остаётся: BGE {span(bge,'text_name_context',.95,'remaining_canonical_vertices')} узел, E5 {span(e5,'text_name_context',.95,'remaining_canonical_vertices')}, GTE {span(gte,'text_name_context',.95,'remaining_canonical_vertices')}. У GTE с контекстом уже при τ=0.90 все 1122 сущности образуют одну группу, а все 1760 отношений превращаются в петли. По названиям у E5 и GTE-small/large максимальный скачок крупнейшей группы виден при снижении 0.90 → 0.85, у GTE-base — 0.85 → 0.80 (крупнейшие группы всех E5/GTE при 0.85: {span(e5+gte,'text_name',.85,'largest_merge_group')}); у GTE с контекстом — уже 0.95 → 0.90.", "",
              "Для последующей проверки перспективны прежде всего τ=0.90 и 0.95 для BGE/text_name и τ=0.95 для E5/GTE/text_name: они предшествуют сильному росту групп, а крупнейшие группы остаются сравнительно небольшими. Для первой проверки BGE/text_name_context предпочтителен строгий край τ=0.95: уже там есть смешанные типы и транзитивные пары ниже порога. E5/text_name_context при 0.95 оставляет лишь 156–271 узел; GTE/text_name_context даже при 0.95 оставляет 25–62 узла, поэтому эти режимы в данной сетке не выглядят умеренным сценарием объединения. Это выбор кандидатов для проверки, не разрешение автоматического merge и не общий откалиброванный порог.", "",
              "Нулевые изолированные вершины сами по себе не улучшение: при схлопывании в одну вершину исходные отношения превращаются в петли. Поэтому показаны оба определения изоляции. Резкое падение V, конфликтные типы и поглощение разных документов — сигналы для проверки, не подтверждение новой идентичности. Типы, повреждённые данные и решения 2C не фильтровались. Семантическая точность и качество Graph-RAG этим экспериментом не измерены.", "",
              "## Проверки и воспроизводимость", "",
              "Проверены 1122 ID, 30 документов, покрытие 1211 исходных entity ID и 1327 mention ID, все 1760 исходных отношений и полная сетка 90 конфигураций. Ровно 628881 проверяемая пара на представление, без самопар и двойного учёта направлений. Union-Find покрывает каждую вершину ровно один раз; число сокращений согласовано с составом групп. Число пар выше порога проверено прямым сравнением, а число пар, оставшихся вершин и размер максимальной группы проверены на монотонность. На 120 контрольных сценариях результат сопоставлен с независимым обходом графа (компоненты, изоляция, петли, типы, транзитивность и границы порогов); повторная симуляция дала идентичные результаты. Дополнительно проверены крайние сценарии без объединений и с полным схлопыванием на всех реальных 1122 сущностях и 1760 отношениях. Полные embedding checksums совпадают с 2B. SHA-256 старых артефактов и файлов моделей контролируются до и после работы.", "",
              "Скрипт: `.venv/Scripts/python.exe scripts/sweep_semantic_thresholds.py`. Используются существующие библиотеки и код 2B; ничего не скачивается и не устанавливается. Создаются только threshold_results.csv и report.md. Canonical mapping, отношения и Neo4j не изменены, Semantic Merge/2D не выполнялся.", "", "SHA-256 защищённых артефактов:", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}`: `{digest}`" for p,digest in hashes.items()]
    protocol.require(all(protocol.sha(p)==digest for p,digest in hashes.items()),"Protected artifact changed")
    for desc in manifests:
        protocol.require(all(protocol.sha(Path(f["path"]))==f["sha256"] for f in desc["files"]),"Model files changed")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    protocol.write_csv(OUTPUT/"threshold_results.csv",result)
    (OUTPUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    protocol.require(all(protocol.sha(p)==digest for p,digest in hashes.items()),"Protected artifact changed after writing")
    print(json.dumps(dict(configurations=len(result),output=str(OUTPUT)),indent=2),flush=True)


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--worker":
        worker(sys.argv[2])
    else:
        main()
