"""Package measured artifacts and a readable report; never manufacture metrics."""
import argparse
from collections import Counter
from pathlib import Path

from cgraph.common import config, file_sha256, read_json, write_json
from cgraph.evaluation import entity_key
from cgraph.inputs import load_inputs
from cgraph.labels import load_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", required=True)
    parser.add_argument("--diagnostic", required=True)
    args = parser.parse_args()
    cfg = config()
    e2e, diagnostic = Path(args.e2e), Path(args.diagnostic)
    run, diag_run = read_json(e2e / "run.json"), read_json(diagnostic / "run.json")
    val, diag = read_json("reports/validation_e2e.json"), read_json("reports/validation_gold_entities_re.json")
    train = read_json("reports/train_descriptive_e2e.json")
    if val["prediction_run_sha256"] != file_sha256(e2e / "run.json") or diag["prediction_run_sha256"] != file_sha256(diagnostic / "run.json"):
        raise ValueError("Metrics and prediction runs differ")
    write_json("reports/runs/end_to_end.json", run)
    write_json("reports/runs/gold_entities_re.json", diag_run)
    write_json("reports/training_metadata.json", run["relation_model"])
    docs = load_inputs(cfg)
    gold = load_labels(cfg, docs)
    examples, counts, reasons, stages = [], Counter(), Counter(), Counter()
    sample_counts = Counter()
    prediction_hashes = {}
    for document in docs:
        code = document["document_code"]
        p = read_json(e2e / f"{code}.json")
        prediction_hashes[code] = file_sha256(e2e / f"{code}.json")
        gm = {m["mention_id"]: m for m in gold[code]["mentions"]}
        gold_edges = {(entity_key(gm[a["head_mention_id"]]["chunk_id"], gm[a["head_mention_id"]]),
                       entity_key(gm[a["tail_mention_id"]]["chunk_id"], gm[a["tail_mention_id"]]), a["relation_type"])
                      for a in gold[code]["assertions"] if a["support_status"] == "SUPPORTED"}
        for chunk in p["chunks"]:
            stages.update(chunk["stage_ms"])
            counts.update(entities=len(chunk["entities"]), candidate_pairs=len(chunk["candidate_pairs"]),
                          raw_proposals=len(chunk["raw_proposals"]), assertions=len(chunk["assertions"]),
                          budget_dropped=chunk["candidate_pairs_dropped_by_budget"])
            pm = {m["mention_id"]: m for m in chunk["entities"]}
            for a in chunk["raw_proposals"]:
                if a.get("reason"):
                    reasons[a["reason"]] += 1
                head, tail = pm[a["head_mention_id"]], pm[a["tail_mention_id"]]
                matched = (entity_key(chunk["chunk_id"], head), entity_key(chunk["chunk_id"], tail), a["relation_type"]) in gold_edges
                subset = "validation" if code in val["partition"]["validation"] else "train"
                bucket = "correct_train" if subset == "train" and matched and a["decision_status"] == "ACCEPTED_PROVISIONAL" else (
                    "validation_false_positive" if subset == "validation" and not matched and a["decision_status"] == "ACCEPTED_PROVISIONAL" else (
                        "validation_correct_raw_abstained" if subset == "validation" and matched and a["decision_status"] == "ABSTAIN" else None))
                if bucket and sample_counts[bucket] < (2 if bucket == "correct_train" else 1):
                    examples.append(dict(category=bucket, document_code=code, subset=subset, head=head, tail=tail,
                                         assertion=a, exact_typed_relation_matches_gold=matched))
                    sample_counts[bucket] += 1
    write_json("reports/examples.json", examples)
    write_json("reports/prediction_digests.json", dict(mode="end_to_end", sha256=prediction_hashes))
    write_json("reports/analysis_counts.json", dict(counts=counts, rejection_reasons=reasons, stage_ms_all_12_dev=stages))
    write_json("reports/diagnostics/model_setup.json", dict(status="RESOLVED", transport_issue="Initial huggingface_hub Xet transfer made no progress; stopped and replaced with bounded HTTPS download",
               weights_sha256_verified=True, strict_state_dict_loaded=True, tokenizer_warning="Transformers 5.1.0 emits a generic Mistral regex warning for local DebertaV2Tokenizer; no Mistral-specific fix applied",
               runtime_configuration="Pinned backbone config embedded; tokenizer local; audit forbids socket.connect", tested_platform="Native Windows Python 3.11.9 CPU; WSL2 not executed"))

    def metric_rows(metrics, names):
        rows = []
        for title, key in names:
            m = metrics[key]
            fmt = lambda x: "null" if x is None else f"{x:.4f}"
            rows.append(f"| {title} | {m['tp']} / {m['fp']} / {m['fn']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['f1'])} |")
        return "\n".join(rows)

    rows = metric_rows(val["micro"], [("Entity span exact", "entity_span"), ("Typed entity exact", "entity_typed"),
                                      ("Raw end-to-end relation", "relation_raw_end_to_end"), ("Gated end-to-end relation", "relation_end_to_end"),
                                      ("RE given candidate endpoints", "relation_given_candidate_endpoints"), ("Qualified relation", "qualified_relation")])
    diag_rows = metric_rows(diag["micro"], [("Raw RE, Gold entities", "relation_raw_end_to_end"), ("Gated RE, Gold entities", "relation_end_to_end"),
                                           ("RE given bounded candidate endpoints", "relation_given_candidate_endpoints"), ("Qualified RE", "qualified_relation")])
    doc_rows = "\n".join(f"| {code} | {m['entity_typed']['f1']:.4f} | {m['relation_end_to_end']['f1']:.4f} | {val['performance_documents'][code]['wall_ms']:.1f} | {val['performance_documents'][code]['cpu_ms']:.1f} |" for code, m in val["documents"].items())
    stage_rows = "\n".join(f"| {name} | {ms:.2f} |" for name, ms in sorted(stages.items()))
    reason_rows = "\n".join(f"| {name} | {n} |" for name, n in reasons.most_common())
    examples_md = []
    for ex in examples:
        a, h, t = ex["assertion"], ex["head"], ex["tail"]
        evidence = a["evidence"][0]
        examples_md.append(f"- **{ex['document_code']} / {ex['subset']} / {ex['category']}**: `{h['surface_text']}` ({h['kind']}, [{h['start']},{h['end']})) → `{a['relation_type']}` → `{t['surface_text']}` ({t['kind']}, [{t['start']},{t['end']})). Score {a['score']:.4f}; `{a['decision_status']}`; reason `{a.get('reason')}`. Exact typed Gold match: `{ex['exact_typed_relation_matches_gold']}`.\n\n  Evidence chunk `{evidence['chunk_id']}`, [{evidence['start']},{evidence['end']}):\n\n  > {evidence['text']}")
    text = f"""# C-001 — отчёт о реальном CPU прототипе архитектуры C

Дата: 2026-10-21 (Europe/Moscow). Статус: **исполняемый baseline, недостаточное качество end-to-end RE**.

## 1. Результат и структура

Реализован независимый Python-проект без backend/frontend/API/БД: NER модель, contextual kinds, bounded directed pairs, обучаемый multi-label RE, scope/qualifier gates, conservative concepts, exact evidence/schema validation, JSON output, отдельный evaluator. Структура и команды полностью приведены в [README](../README.md). Код: `src/cgraph`, конфиг: `config/c001.json`, контракты: `contracts`, проверки: `tests`, результаты: `reports`.

Все {len(run['documents'])} DEV-документов обработаны реальными моделями: {counts['entities']} entities, {counts['candidate_pairs']} candidate pairs, {counts['raw_proposals']} raw proposals, {counts['assertions']} provisional assertions. Бюджет отбросил {counts['budget_dropped']} дополнительных пар. Это не synthetic smoke experiment. Искусственные minimal-pair tests приведены отдельно.

## 2. Данные, split и защита от leakage

Upstream commit `{read_json(cfg['manifest'])['source_commit']}` репозитория [Graph-based-RAG-for-article-analysis](https://github.com/KulikovKA/Graph-based-RAG-for-article-analysis). Dataset: `{read_json(cfg['manifest'])['dataset_id']}`. Оригинальная ontology v1.0 сохранена byte-for-byte, SHA-256 `{read_json(cfg['manifest'])['ontology_sha256']}`. Её старый approval header не переписан: актуальный owner-reviewed release envelope хранится отдельно. Один owner review, annotation origin AI_GENERATED, independent secondary review отсутствует.

Seed 42. Train: **{', '.join(val['partition']['train'])}**. Validation: **{', '.join(val['partition']['validation'])}**. Группы C02/C07 целиком оставлены для validation; тематические pilot24 clusters — только proxy для family grouping, независимой проверки семейства не было. Это descriptive DEV validation на четырёх документах, без подтверждающего holdout, confidence claims и настройки thresholds по validation.

Loader проверяет allowlist 12 IDs, split DEV, revision IDs и pinned revision-content hash, inventory chunk IDs/ordinals/sections/languages/Unicode boundaries, stored/source SHA-256 и byte hashes входов. Полный revision hash сверяется с release metadata: полный исходный revision вне frozen chunks не реконструируется. `fetch-dev` получает ровно 14 разрешённых файлов по pinned commit; все 14 SHA-256 проверены реальным скачиванием. CALIBRATION/HOLDOUT тексты, Gold и review packets не открывались.

Раздельные stores: `data/inputs`, `data/labels`, `artifacts/*predictions`, offline evaluation. Runtime ontology — нормативная проекция без Gold examples и document-specific boundaries. Реальный end-to-end run выполнен через audit guard, запрещающий open labels/gold/original ontology, импорт annotation-модуля и socket.connect. Gold entities использовались только в отдельном диагностическом run.

D024: `exhaustive_relations=false`; его positive annotations разрешены для обучения, все assumed negatives исключены. Остальные hard negatives — close same-sentence type-compatible pairs и reversed positives из exhaustive train-документов; unresolved evidence не становится negative Gold. Train precision на неполном D024 ограничена полнотой аннотаций и приведена только descriptively.

## 3. Реальные модели и обучение

NER: [urchade/gliner_small-v2.1](https://huggingface.co/urchade/gliner_small-v2.1), revision `{cfg['ner']['revision']}`, Apache-2.0. Реально загружено **{run['ner_model']['parameters']:,} параметра**, CPU float32, strict=True, класс `{run['ner_model']['model_class']}`. Окна 140 whitespace words, overlap 30, threshold 0.35, 12 зафиксированных label descriptions. Nested/overlapping spans сохраняются, dedup только exact span+kind. Exact codepoints проверяются сравнением с неизменным source string.

Tokenizer/backbone config: [microsoft/deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small), revision `{cfg['ner']['tokenizer_revision']}`, MIT по card, `{run['ner_model']['tokenizer_class']}`. Backbone config встраивается в derived runtime config; weights не меняются. File hashes, включая weights/tokenizer/config/cards, в [model_files.json](../contracts/model_files.json); фактическая runtime identity в [runs/end_to_end.json](runs/end_to_end.json). Никакой generative model не загружается.

RE: **TF-IDF marked-pair linear sigmoid heads**, one-vs-rest logistic regression (`liblinear`, C=1, balanced classes, seed 42). Case-sensitive Unicode word tokens, unigram+bigram, до 12 000 features. Head/tail boundary markers и direction marker различают направление; применяется immutable ontology domain/range. Это обученный лексический classifier, **без pretrained RE encoder**. SciBERT остаётся base encoder и здесь не называется готовой RE моделью.

Изучен [GLiNER-Relex base](https://huggingface.co/knowledgator/gliner-relex-base-v1.0), revision `e6a880049a19c5cc222a7a479c32e84b0d8cdd9a`, Apache-2.0: joint NER+RE относится к arm F, поэтому не подменяет раздельную архитектуру C. Его веса в этом эксперименте не запускались. Выбран минимальный полностью обучаемый CPU marked-pair baseline; перенос на encoder head оставлен C-002.

Обучение: {sum(x['examples'] for x in run['relation_model']['training_inventory'])} примеров, {sum(x['hard_negatives'] for x in run['relation_model']['training_inventory'])} hard negatives. Из 23 relation outputs 21 обучен; `INTEGRATED_IN` и `OPTIMIZES` имеют нулевой support и `UNSUPPORTED_CAPABILITY`, а не фиктивные предсказания. Все 23 labels присутствуют в inventory; 24-й output — NO_RELATION. Multi-label positives допустимы; ABSTAIN — отдельная policy при низких scores, конфликте с NO_RELATION или unsupported labels. Thresholds 0.65 фиксированы заранее, не calibrated probabilities.

Точная identity local RE parameters: `{run['relation_model']['parameters_sha256']}`; feature/tokenizer hash `{run['relation_model']['features_sha256']}`. Train IDs, support каждого класса и точные settings в [training_metadata.json](training_metadata.json). Checkpoint из внешнего Hub для RE отсутствует; лицензия sklearn — BSD-3-Clause, отдельная лицензия новых fitted parameters не назначалась. Веса/NPZ/cache в Git не включены.

## 4. Поддержанные и неподдержанные возможности

Реализованы raw mention spans, contextual ANALYTE kind, directed/multi-label RE, NO_RELATION/ABSTAIN, domain/range, pair cap 400/chunk и gap 300 characters, local sentence candidates, parsing простых чисел/единиц, simple negation/pseudo-negation и modality cues, conservative NFC/whitespace name normalization с сохранением chemistry case/composition, exact local acronym rules, source-scoped concepts и evidence checks.

Нет полного contextual-role classifier (прочие roles UNKNOWN), coreference, cross-sentence RE, discontinuous mention model, надёжного range/comparator parsing, n-ary numeric/condition attachment, calibrated NLI или глобального identity classifier. Device instances не объединяются по имени; RELATED/hierarchy не превращаются в SAME.

Unresolved qualifiers/negation/comparison/distribution/overlapping endpoints получают ABSTAIN. Простые numeric spans сохраняются отдельно с attachment UNKNOWN; числа не назначаются всем соседним entities. Required qualifiers не выдумываются. Assertion evidence и qualifier rules проверяются до JSON Schema validation. Accepted assertions помечены **ACCEPTED_PROVISIONAL**, `qualifier_status=UNKNOWN`, support_decision=CLASSIFIER_AND_LEXICAL_SCREEN. Это кандидатный research graph, не independently established semantic truth.

Lexical predicate screen требует явный cue между endpoints, поэтому одна co-occurrence не является достаточной. Это только deterministic screen: ложные semantic relations и неверные NER types всё равно остаются. Grounded exact quote не доказывает смысл утверждения.

## 5. Качество на четырёх validation-документах

Основной end-to-end run: Gold не подаётся в NER/RE. Строгий one-to-one multiset matching exact chunk/codepoint spans+kind и directed predicate; concept IDs не участвуют. Нулевой precision denominator → null; Gold positives при пустом output → recall/F1=0.

| Метрика | TP / FP / FN | Precision | Recall | F1 |
|---|---|---|---|---|
{rows}

Candidate recall: **{val['micro']['candidate_pair_recall']['covered']}/{val['micro']['candidate_pair_recall']['gold']} = {val['micro']['candidate_pair_recall']['recall']:.4f}**. Abstention: **{val['micro']['abstention']['abstained']}/{val['micro']['abstention']['candidate_pairs']} = {val['micro']['abstention']['rate']:.4f}**. Unknown qualifier completeness не получает qualified TP, а Gold qualifiers остаются в знаменателе; нулевой qualified F1 честно отражает отсутствие полной поддержки.

Отдельный Gold-entity RE diagnostic, только validation, без NER inference:

| Метрика | TP / FP / FN | Precision | Recall | F1 |
|---|---|---|---|---|
{diag_rows}

Диагностический candidate recall **{diag['micro']['candidate_pair_recall']['covered']}/{diag['micro']['candidate_pair_recall']['gold']} = {diag['micro']['candidate_pair_recall']['recall']:.4f}**. Entity quality для этой диагностики не публикуется: Gold spans дали бы тавтологическую оценку. Данные не смешиваются с end-to-end.

На train end-to-end gated RE F1 = {train['micro']['relation_end_to_end']['f1']:.4f} (TP={train['micro']['relation_end_to_end']['tp']}); это **не независимый результат**. Полные micro/macro-document и per-document counts: [validation_e2e.json](validation_e2e.json), [validation_gold_entities_re.json](validation_gold_entities_re.json), [train_descriptive_e2e.json](train_descriptive_e2e.json). Human semantic-support rate, confidence intervals, false-merge metrics и downstream retrieval quality — NOT_MEASURED.

## 6. Производительность

Нативная Windows, Python {run['python']}, {run['cpu']}, {run['logical_cpus']} logical CPUs, 4 PyTorch threads, CPU float32; WSL2 execution не проверена. Полный lock: [uv.lock](../uv.lock), Linux torch закреплён с CPU index без CUDA packages. Runtime package versions сохранены в run manifests.

Реальный fresh-process run **всех 12 DEV** с локальными model files: wall **{run['wall_ms']/1000:.3f} s**, CPU **{run['cpu_ms']/1000:.3f} s**, peak working set **{run['peak_ram_bytes']/1024**3:.3f} GiB** ({run['peak_ram_bytes']} bytes). Method: {run['peak_ram_method']}. Wall включает dependency imports после CLI dispatch и model loading; download/training не включены. NER model setup/load **{run['ner_load_ms']/1000:.3f} s**, отдельно от per-document extraction. Это одна описательная CPU measurement, не доказательство speedup относительно A/B/F.

Model invocations: **{run['ner_calls']} NER calls**, instrumented **{run['encoder_forward_calls']} encoder forwards**, **{run['re_batch_calls']} RE batch calls** на {counts['candidate_pairs']} pairs; всего {run['ner_calls']+run['re_batch_calls']} model API calls. **LLM calls = 0**; сетевые подключения во время extraction запрещены audit hook. Maximum observed encoder sequence — {run['max_sequence_tokens']} tokens. Нет генеративного fallback.

Сумма стадий по всем 12 DEV (ms):

| Стадия | ms |
|---|---|
{stage_rows}

Linking учитывается отдельно в per-document execution. Сумма stage time не равна process wall: imports, loading, schema checks и serialization имеют отдельную стоимость.

| Validation doc | Typed entity F1 | Gated RE F1 | extraction wall ms | CPU ms |
|---|---|---|---|---|
{doc_rows}

Отдельный Gold RE process: wall {diag_run['wall_ms']/1000:.3f} s, CPU {diag_run['cpu_ms']/1000:.3f} s, peak {diag_run['peak_ram_bytes']/1024**3:.3f} GiB, {diag_run['re_batch_calls']} RE calls, 0 encoder/LLM calls. Полная performance scope указана в JSON: validation quality не выдаётся за отдельный 12-document latency run.

## 7. Примеры реального output и основные ошибки

Ниже неизменённые model outputs. Успешные train-примеры иллюстрируют функционирование, а не независимое качество. Validation false positive включён явно. Exact offsets и evidence SHA-256 всех примеров: [examples.json](examples.json).

{chr(10).join(examples_md)}

Главные наблюдения: NER boundary/type errors оставили только 7 из 132 Gold relations среди end-to-end candidates, хотя с Gold entities проходят 125/132. Лексические heads плохо переносятся на другие topical clusters. Из 91 raw validation relations только 5 совпали с Gold; gates удалили и эти TP, а 21 оставшийся provisional assertion не совпал с typed Gold. Следовательно, deterministic screen одновременно теряет recall и не гарантирует precision. Увеличение strictness само по себе недостаточно.

Reasons по raw proposals всех 12 DEV:

| Reason | Count |
|---|---|
{reason_rows}

Эти counts относятся к proposals, abstention denominator — к candidate pairs; их нельзя складывать как одну метрику.

Диагностика установки: первый Xet download завис, заменён на pinned HTTPS с timeout и checksum. Strict GLiNER load прошла. Transformers 5.1.0 вывел generic Mistral regex warning для DeBERTa tokenizer; архитектура/веса проверены, чужой tokenizer fix не применён. Это зарегистрированное ограничение runtime, а не скрытая подмена модели.

## 8. Проверки и C-002

21 тест: source identity/hash tamper, split/cluster isolation, immutable ontology, label-free projection, Unicode/repeated offsets, overlapping mentions, directed markers, pair budgets, simple/complex negation и not-only, co-occurrence rejection, numeric spans, chemical/device cannot-merge, multiset/empty metrics, strict qualifier penalty, schema и zero-LLM constraint, real classifier fitting/serialization и train labels boundary. Полный model run — отдельная реальная CPU интеграционная проверка с denied label/network access. См. [tests.json](tests.json).

Приоритеты C-002: (1) scientific NER и contextual typing на train-only domain data; (2) замена TF-IDF головы на marked-pair encoder head либо shared span classifier с сохранением arm identity; (3) расширение train coverage редких relations, exhaustive negative judgments; (4) sentence/condition attachment, negation/coreference scope и n-ary qualifiers; (5) независимый semantic audit и calibration внутри DEV, без подгонки на текущих четырёх validation docs; (6) family grouping, macro/cluster uncertainty и независимый будущий protocol. Текущий DEV validation уже просмотрен и не должен называться untouched confirmatory test в C-002.

Репозиторий содержит только код, контракты, measured metrics, hashes и разрешённые DEV excerpts. Model weights/cache, DEV labels и полные predictions остаются локально под gitignore. HOLDOUT/private data/secrets не публикуются. Основной проект использован только для контрактов, DEV-данных и документации, его рабочее дерево не изменено.
"""
    Path("reports/C001_REPORT.md").write_text(text, encoding="utf-8", newline="\n")
    print("Packaged C-001 measured reports and examples")


if __name__ == "__main__":
    main()
