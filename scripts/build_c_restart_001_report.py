"""Publish aggregates/provenance only. Full predictions and labels stay local."""
import argparse
from pathlib import Path
import shutil

from cgraph.common import file_sha256, read_json, write_json


def number(x):
    return "—" if x is None else f"{x:.4f}"


def quality_table(metrics):
    names = {"entity_span": "Entity span", "entity_typed": "Entity typed",
             "relation_raw_end_to_end": "Raw directed relation", "relation_end_to_end": "Final directed relation",
             "qualified_relation": "Qualified relation"}
    rows = ["| Метрика | TP / FP / FN | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for key, name in names.items():
        m = metrics[key]
        rows.append(f"| {name} | {m['tp']} / {m['fp']} / {m['fn']} | {number(m['precision'])} | {number(m['recall'])} | {number(m['f1'])} |")
    return "\n".join(rows)


def historical_all():
    source = [read_json(f"reports/{name}.json")["micro"] for name in ("validation_e2e", "train_descriptive_e2e")]
    output = {}
    for key in ("entity_span", "entity_typed", "relation_raw_end_to_end", "relation_end_to_end", "qualified_relation"):
        counts = {k: sum(x[key][k] for x in source) for k in ("tp", "fp", "fn", "gold", "predicted")}
        tp, ng, np = counts["tp"], counts["gold"], counts["predicted"]
        output[key] = dict(counts, precision=tp / np if np else None, recall=tp / ng if ng else None,
                           f1=2 * tp / (ng + np) if ng + np else None)
    covered, gold = (sum(x["candidate_pair_recall"][k] for x in source) for k in ("covered", "gold"))
    output["candidate_pair_recall"] = dict(covered=covered, gold=gold, recall=covered / gold)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/c_restart_001_nli_dev")
    args = parser.parse_args()
    directory = Path(args.predictions)
    out = Path("reports/c_restart_001")
    out.mkdir(parents=True, exist_ok=True)
    run = read_json(directory / "run.json")
    audit = read_json(directory / "access_audit.json")
    assert audit["run_sha256"] == file_sha256(directory / "run.json") and audit["completed"]
    for name, expected in run["source_code_files_sha256"].items():
        if file_sha256(name) != expected:
            raise ValueError(f"Inference code changed after run: {name}")
    predictions = [read_json(directory / f"{code}.json") for code in run["documents"]]
    verified_scores = 0
    for p in predictions:
        assert file_sha256(directory / f"{p['document_code']}.json") == run["prediction_files_sha256"][p["document_code"]]
        for chunk in p["chunks"]:
            entities = {m["mention_id"]: m for m in chunk["entities"]}
            def source_key(record):
                h, t = entities[record["head_mention_id"]], entities[record["tail_mention_id"]]
                return h["start"], h["end"], t["start"], t["end"], record["relation_type"]
            scored = {}
            for trace in chunk["re_inference_trace"]:
                if trace["passed_fixed_threshold"]:
                    scored[source_key(trace)] = max(scored.get(source_key(trace), 0), trace["scores"]["entailment"])
            for proposal in chunk["raw_proposals"]:
                if proposal["score"] != scored.get(source_key(proposal)):
                    raise ValueError("A raw proposal cannot be traced to an actual model score")
                verified_scores += 1
    dev = read_json(out / "dev.json")
    val = read_json(out / "legacy_validation.json")
    baseline = read_json("reports/validation_e2e.json")["micro"]
    all_baseline = historical_all()
    for name in ("run.json", "access_audit.json"):
        shutil.copyfile(directory / name, out / name)
    shutil.copyfile("artifacts/nli_checkpoint_probe.json", out / "nli_checkpoint_probe.json")
    if Path("artifacts/c_restart_001_dev/run.json").exists() and Path("artifacts/c_restart_001_probe_compatible.json").exists():
        initial = read_json("artifacts/c_restart_001_dev/run.json")
        failed = read_json("artifacts/c_restart_001_probe_compatible.json")
        diagnostic = dict(glirel_published_control_reproduced=False, cause="UNRESOLVED",
        expected_characters_scores_approximately=[0.99, 0.99], observed_characters_scores=failed["characters_scores"],
        checked_runtimes=["GLiREL 1.2.1 / Transformers 5.1.0", "GLiREL 1.2.1 / Transformers 4.51.3",
                          "GLiREL 1.0.1 / Flair 0.14.0 / Transformers 4.51.3",
                          "GLiREL 1.0.1 / Flair 0.14.0 / Transformers 4.44.2"],
        official_loader_same_result=True, fast_and_slow_tokenizers_same_result=True,
        initial_guarded_dev_run=dict(documents=initial["documents"], raw_predictions=0, final_predictions=0,
             wall_ms=initial["wall_ms"], cpu_ms=initial["cpu_ms"], peak_ram_bytes=initial["peak_ram_bytes"],
             ner_calls=initial["ner_calls"], re_calls=initial["re_batch_calls"], llm_calls=0,
             local_run_sha256=file_sha256("artifacts/c_restart_001_dev/run.json")),
            interpretation="Runtime diagnostic excluded from scientific-quality comparison; no threshold lowering or synthetic replacement")
    else:
        # A fresh clone can reproduce the primary NLI experiment without
        # repeating every historical runtime investigation.
        diagnostic = read_json(out / "glirel_diagnostic.json")
    write_json(out / "glirel_diagnostic.json", diagnostic)
    counts = {p["document_code"]: dict(entities=sum(len(c["entities"]) for c in p["chunks"]),
        candidate_pairs=sum(len(c["candidate_pairs"]) for c in p["chunks"]),
        raw_proposals=sum(len(c["raw_proposals"]) for c in p["chunks"]),
        final_assertions=sum(len(c["assertions"]) for c in p["chunks"]),
        nli_queries=sum(len(c["re_inference_trace"]) for c in p["chunks"]),
        skipped_context_queries=sum(len(c["skipped_re_contexts"]) for c in p["chunks"]),
        numeric_measurements=sum(len(c["measurements"]) for c in p["chunks"]),
        attached_qualifiers=sum(len(a["qualifiers"]) for c in p["chunks"] for a in c["assertions"])) for p in predictions}
    write_json(out / "counts.json", dict(documents=counts, totals={k: sum(row[k] for row in counts.values()) for k in next(iter(counts.values()))},
                                        raw_scores_traced_to_real_inference=verified_scores))
    write_json(out / "comparison.json", dict(legacy_validation=dict(c001=baseline, zero_shot=val["micro"]),
        all_dev=dict(c001=all_baseline, zero_shot=dev["micro"]),
        caveat="C-001 fitted on eight DEV docs; final NLI/NER runtime differs; DEV is exploratory; D024 incompletely annotated"))
    funnel = dev["loss_audit"]
    funnel_table = "\n".join(["| Стадия | Правильных edges осталось | Потеря на стадии |", "|---|---:|---:|"] +
        [f"| {name} | {funnel['correct_assertions_remaining'][name]} | {funnel['correct_assertions_lost'].get(name, 0)} |" for name in funnel["order"]])
    comparison_rows = []
    for name, m in (("Исторический C-001", baseline), ("Pretrained-only C", val["micro"])):
        comparison_rows.append(f"| {name} | {number(m['entity_typed']['f1'])} | {m['candidate_pair_recall']['covered']}/{m['candidate_pair_recall']['gold']} | {number(m['relation_raw_end_to_end']['f1'])} | {number(m['relation_end_to_end']['f1'])} | {number(m['qualified_relation']['f1'])} |")
    c = dev["micro"]["candidate_pair_recall"]
    abst = dev["micro"]["abstention"]
    total = {k: sum(row[k] for row in counts.values()) for k in next(iter(counts.values()))}
    report = f"""# C-RESTART-001 — pretrained-only архитектура C

Дата: 2026-10-29, Europe/Moscow. Реальный CPU inference завершён на всех 12 DEV.
Обучение, fine-tuning, fit, использование Gold в inference, генеративные LLM и
downstream reranker отсутствуют. **Исполнимость и exact evidence не означают
правильность графа**. Качество ниже пригодного для автоматической публикации фактов.

## Что реализовано

Независимые компоненты: scientific text → готовый GLiNER-small → 12 типов spans →
направленные candidate pairs → pretrained entailment как zero-shot RE → partial
scope/qualifiers → проверка evidence → conservative canonicalization → assertions.
Новый CLI `cgraph-zero` не содержит команды обучения. Исторические C-001 код,
конфиг и отчёты сохранены. Рабочее дерево C-002 осталось в исходном каталоге;
новая реализация находится в отдельном worktree/ветке.

Повторно использованы DEV allowlist/loader, frozen ontology v1.0, source hashes,
нормативная проекция без Gold examples, JSON Schema, exact multiset evaluator,
GLiNER adapter, evidence и conservative linking. Ontology не менялась.

## Модели и выбор RE

NER: `urchade/gliner_small-v2.1`, revision
`4e091416cf7c3481db542c2a3d26156916f3a47f`, Apache-2.0,
{run['ner_model']['parameters']:,} параметров. Сохранены 12 исходных label
descriptions, threshold 0.35, окна 140 words/overlap 30. Никаких Gold-based
boundary repairs или нового supervised entity typing.

RE: `cross-encoder/nli-deberta-v3-xsmall`, revision
`a150876415327c80daeff35ca6f68f5ed8cf5c24`, Apache-2.0,
{run['relation_model']['parameters']:,} параметров. Это готовая модель NLI с тремя
реальными классами contradiction/entailment/neutral, обученная её авторами на
SNLI/MultiNLI. На наших текстах и метках ничего не обучалось. Новые 23 heads
не создавались. Применение entailment к формулировкам отношений следует
[Sainz et al., EMNLP 2021](https://aclanthology.org/2021.emnlp-main.92/).
Результаты той статьи не являются оценкой нашей научной онтологии.

Выбор обусловлен отдельным RE-интерфейсом, доступным полным checkpoint,
CPU-стоимостью и лицензией. Это компонент RE, а не downstream reranker.
Данные модели: [model card](https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall).
Контроль из model card воспроизведён: entailment 0.9397 и contradiction 0.9996;
он исключён из DEV метрик и времени основного запуска.

Предварительно исследован GLiREL-large-v0 (466,577,920 параметров). Он не
воспроизвёл опубликованный пример: около 0.001–0.002 для `characters` вместо
около 0.99. SHA256 и strict load успешны; официальный loader даёт тот же результат.
Проверены старые/новые библиотеки, fast/slow tokenizer и обычная загрузка весов.
Причина **не установлена**. Первые 12-DEV predictions с нулём raw сохранены
локально; они исключены из валидированного сравнения качества, а не скрыты.
Диагностика: [glirel_diagnostic.json](c_restart_001/glirel_diagnostic.json).
GLiREL card/repository указывают CC BY-NC-SA 4.0 при Apache-2.0 в package metadata.
Joint GLiNER-Relex относится к F и не использовался.

Отдельно устранена несовместимость локальной DeBERTa tokenization в
Transformers 5.1.0: служебные CLS/SEP отсутствовали. Финальный стек закрепляет
Transformers 4.51.3/tokenizers 0.21.4 и проверяет их наличие. Это также меняет
NER runtime относительно C-001; сравнение не изолирует только замену RE.
Исходные pretrained веса сохранены без изменений.

## Зафиксированная политика и capabilities

До оценки записаны [аудит](C_RESTART_001_PROTOCOL.md) и
[окончательный протокол](C_RESTART_001_PROTOCOL_ADDENDUM.md).
Конфиг SHA256: `{run['config_sha256']}`.
Порог NLI entailment >0.5, batch 32, single-sentence context, без поиска thresholds
и verbalizations по Gold. Negative hypothesis дополнительно проверяется только
при linguistic negation в source. Полные три scores каждой проверки сохранены
локально в `re_inference_trace`. Ни один score не объявляется калиброванной
вероятностью правильности научного факта.

Все 23 relation capabilities явно перечислены в конфиге. 22 имеют статус
EXPERIMENTAL_ZERO_SHOT; это возможность выполнить проверку гипотезы, а не
доказательство поддержки научной семантики. `MEETS_REQUIREMENT` UNSUPPORTED:
совместный requirement bundle и collective scope требуют n-ary attachment.
`CHANGES_PARAMETER` требует явный direction cue; неизменность параметра и
сложные causal/explanatory claims поддерживаются неполно. Per-type фактические
TP/FP/FN находятся в `loss_audit.per_relation` отчётов JSON, включая нулевые TP.

Кандидаты перечисляются до domain/range pruning в обоих направлениях, без
прежнего cap 400/chunk и gap 300 characters. NLI оценивает только совместимые
с онтологией hypotheses. Cross-sentence/coreference в основном run не поддержаны.
Неоднозначные типы одного span сохраняются и могут увеличивать FP.
RE score относится к текстовым spans, поэтому raw proposals проецируются на
альтернативные NER kinds тех же spans до final domain/range gate; {funnel['rejected_raw_proposals'].get('ONTOLOGY_DOMAIN_RANGE', 0)} таких
proposals нарушили domain/range и были удалены. Число raw proposals отличается
от числа NLI queries: порог прошли {run['re_score_summary']['above_fixed_threshold']}
queries, затем альтернативные polarity/type записи объединяются/проецируются.
Все {verified_scores} raw scores проверены по реальным локальным NLI traces.
Это также ограничивает прямую сопоставимость raw output с C-001, который заранее
ограничивал classifier labels по domain/range; final matching policy едина.

Удалён обязательный узкий predicate cue между endpoints. Uppercase NO/NOx/NO₂
отделены от English negation; simple polarity/modality и scope проверяются
детерминированно. Неоднозначное отрицание/cross-clause scope дают ABSTAIN.
Числа/units сохраняются с exact evidence; произвольного присоединения к соседним
relations нет. Реализованы явные trailing temperature и physical direction
qualifiers, но completeness остаётся UNKNOWN: qualified TP за них не начисляется.
Опциональные unresolved qualifiers не удаляют все binary assertions.
На реальном DEV присоединённых qualifiers оказалось ноль; attachment подтверждён
только минимальными тестами, а не успешными qualified predictions этого run.

Каждый final assertion имеет exact source hash/quote/codepoint offsets и
ACCEPTED_PROVISIONAL. Evidence verification подтверждает происхождение текста,
**не entailment и не научную истинность**. Каноникализация сохраняет chemical
case/composition и не объединяет device instances по одному имени; глобальная
entity resolution не заявляется.

## Фактические метрики: все 12 DEV

{quality_table(dev['micro'])}

Candidate recall: **{c['covered']}/{c['gold']} = {number(c['recall'])}**.
Abstention по candidate pairs: **{abst['abstained']}/{abst['candidate_pairs']} = {number(abst['rate'])}**.
{total['entities']} entities, {total['candidate_pairs']} candidates,
{total['nli_queries']} реальных NLI queries, {total['raw_proposals']} raw proposals,
{total['final_assertions']} final assertions. Извлечено {total['numeric_measurements']}
числовых spans; в final assertions сохранено {total['attached_qualifiers']} partial qualifiers.
Контексты/queries сверх лимита: {total['skipped_context_queries']}.

Exact Unicode spans+kind, directed predicate, one-to-one multiset matching;
relaxed spans/coreference credit отсутствуют. UNKNOWN qualifier completeness
никогда не получает qualified TP. При пустом prediction denominator precision
обозначается null/«—», а не искусственным нулём/единицей.

Gold используется исключительно offline evaluator после сохранения predictions.
D024 имеет incomplete relation annotation: precision всех DEV ограничена её
полнотой. DEV уже исследовался в C-001 и не является untouched confirmatory set.
CALIBRATION/HOLDOUT не открывались. Один owner review, AI-origin annotations,
независимого второго annotator нет. Human semantic-support rate не измерен.

## Где теряются правильные утверждения

{funnel_table}

Это exact typed directed Gold edges, не автоматически подтверждённые научные
факты. Candidate recall ограничен уже наличием точных NER endpoints. Потери
raw→scope и причины отбрасывания отдельно записаны в `loss_audit`; там же
per-document funnel и per-relation counts. Не смешиваются pair abstention и
число rejected proposals. Canonicalization не меняет scoring keys, поэтому
может дать нулевую потерю на final даже без проверки глобальной identity.

## Сравнение с историческим C-001

Одни и те же четыре legacy validation DEV: D001, D003, D035, D092.

| Вариант | Typed entity F1 | Candidate coverage | Raw relation F1 | Final relation F1 | Qualified F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

Полная таблица нового варианта на этих четырёх документах:

{quality_table(val['micro'])}

C-001 обучался на восьми остальных DEV, новый C не обучался ни на одном.
Сравнение всех 12 для C-001 включает training documents и потому не является
независимым результатом: historical final F1 {number(all_baseline['relation_end_to_end']['f1'])},
новый final F1 {number(dev['micro']['relation_end_to_end']['f1'])}.
Все aggregate counts доступны в [comparison.json](c_restart_001/comparison.json).
Сравнение descriptive: маленький DEV, разные RE и runtime, без доверительных
интервалов, настройки на Gold и утверждения превосходства по общей архитектуре.
На legacy validation candidate recall остался 7/132: снятие pair caps не исправило
отсутствие точных NER endpoints. Новый scope сохраняет все 7 raw TP, но оставляет
596 FP. Raw F1 снизился с 0.0448 до 0.0141; рост final F1 с нуля до 0.0190
не означает пригодность графа. На всех DEV final precision составляет всего
1.07% (48 TP, 4455 FP). Неразборчивость NLI здесь столь же существенна, как NER.

## Измеренная стоимость

Windows CPU, Python {run['python']}, {run['logical_cpus']} logical CPUs,
{run['cpu_threads']} PyTorch threads, float32. Один fresh process; включает
imports, загрузку/hash verification моделей, inference и сериализацию документов;
исключает downloads, probes, невалидированные попытки и evaluator.

| Показатель | Измерено |
|---|---:|
| Wall time | {run['wall_ms']/1000:.3f} s |
| CPU process time | {run['cpu_ms']/1000:.3f} s |
| Peak working set | {run['peak_ram_bytes']/2**30:.3f} GiB ({run['peak_ram_bytes']} bytes) |
| NER API / encoder calls | {run['ner_calls']} / {run['encoder_forward_calls']} |
| RE batch API / encoder calls | {run['re_batch_calls']} / {run['re_encoder_forward_calls']} |
| Model API calls всего | {run['model_api_calls']} |
| LLM calls | 0 |
| Максимум NER / RE subtokens | {run['ner_max_sequence_tokens']} / {run['re_max_sequence_tokens']} |

Peak RAM: {run['peak_ram_method']}. RE scoring/queries и model API calls имеют
разные denominators: один batch обрабатывает до 32 premise/hypothesis pairs.
Одна measurement на общей машине; это не контролируемый speed benchmark.
Исторический C-001: 47.524 s wall, 90.266 s CPU, 1.624 GiB peak, 46 API calls
на 12 DEV; его обученный lexical RE существенно дешевле pretrained entailment.
Per-document/stage timing — в JSON. Промежуточные контрольные прогоны не
суммируются с основным временем и отдельно раскрыты в диагностике.

## Проверки, воспроизведение и вывод

`uv run --no-sync python -m pytest -q`: **37 passed, 1 deselected**.
Исключён только сохранённый C-001 supervised fitting test; он не запускался.
Проверены source/ontology hashes, DEV allowlist, Unicode/nested spans,
направленность/неограниченные пары, chemical vs linguistic negation, scope,
required direction, partial numeric evidence, raw/final domain filtering,
strict qualified penalty, loss funnel и отсутствие training imports/calls в
новых inference modules. Искусственные fixtures тестируют код и не входят в
экспериментальные метрики. Реальная интеграция — отдельные NLI control и 12 DEV.

Guard завершил реальный inference с запретом Gold/closed splits/network и
старых fitted RE. [access_audit.json](c_restart_001/access_audit.json) связан
hash с [run.json](c_restart_001/run.json). Source/config/lock/model/prediction
hashes сохранены; run выполнен до финального commit (`code_dirty=true`), точные
inference modules проверены по hashes. Полный lock закреплён. В Git публикуются
только код, contracts, aggregate metrics и provenance; полные predictions,
per-query traces, Gold labels и все веса остаются локально.

Воспроизведение — команды в [README](../README.md); для нового run нужен новый
output directory. Качество: [12 DEV](c_restart_001/dev.json),
[legacy validation](c_restart_001/legacy_validation.json),
[counts](c_restart_001/counts.json).

**Что работает:** реальные отдельные pretrained NER/RE, запрет обучения и
leakage, воспроизводимые predictions, exact provenance, детальная диагностика
потерь. **Что не доказано:** корректность scientific graph, полнота qualifiers,
каузальная/механистическая семантика, глобальная identity и downstream retrieval.
Низкий relation precision/recall нельзя скрыть structural PASS.

Продолжать C имеет смысл как ограниченное исследование bottlenecks, прежде всего
готового scientific NER/typing и семантики verbalizations, с заранее
зафиксированными Gold-free решениями. Текущий вариант не готов к автономному
построению надёжного научного графа. Ввиду стоимости NLI и потерь endpoints
не стоит масштабировать его без улучшения этих компонентов. Обучение на Gold,
reranker и joint F остаются вне текущего эксперимента.
"""
    Path("reports/C_RESTART_001_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    print({"report": "reports/C_RESTART_001_REPORT.md", "counts": total, "metrics": dev["micro"]})


if __name__ == "__main__":
    main()
