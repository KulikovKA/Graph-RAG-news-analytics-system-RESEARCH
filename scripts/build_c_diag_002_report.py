"""Render measured aggregates and selected real DEV examples into a reviewable report."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgraph.common import file_sha256, read_json


def n(value):
    return "—" if value is None else f"{value:.4f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    out = Path("reports/c_diag_002")
    names = dict(control="GLiNER control", chem="MatSciBERT CHEM", battery="Electrocatalyst NER*", union="GLiNER + оба NER", gold="GOLD entities (diagnostic)")
    results = {v: read_json(out / f"{v}.json") for v in names}
    cfg = read_json("config/c_diag_002.json")
    ontology = read_json(cfg["ontology"])
    control = results["control"]["micro"]
    union = results["union"]["micro"]
    gold = results["gold"]["micro"]
    gained = union["candidate_pair_recall"]["covered"] - control["candidate_pair_recall"]["covered"]
    conclusions = (
        "**Рекомендация: CLOSE C AND MOVE TO B.** Проверенный pretrained-only pipeline "
        "не обеспечивает точности, достаточной для автоматического построения научного графа. "
        "Это вывод о завершённом ограниченном исследовании, не доказательство невозможности всех "
        "pretrained NER/RE подходов. B здесь не измерялась и не объявляется доказанно лучшей.\n\n"
        "1. **Могут ли готовые scientific NER заметно улучшить candidate recall?** "
        f"В проверенном наборе получен только ограниченный прирост через union: самостоятельные CHEM и battery "
        f"имеют 0/390 typed relation candidates. Заранее заданный union добавил {gained} edges: "
        f"71/390 → {union['candidate_pair_recall']['covered']}/390 "
        f"(+{100*gained/390:.2f} процентного пункта), при падении typed entity F1 "
        f"{n(control['entity_typed']['f1'])} → {n(union['entity_typed']['f1'])}. "
        "Union нашёл ещё 20 exact MATERIAL и 9 OPERATING_CONDITION mentions, но не улучшил "
        "остальные kinds. CHEM — готовый scientific single-class NER; материалы/условия battery "
        "проверены как опубликованный domain checkpoint с неподтверждённой MatSciBERT lineage. "
        "Более широкие авторские MatBERT-NER archives остаются непроверенными: обобщать результат "
        "на весь scientific NER нельзя.\n\n"
        "2. **Способен ли готовый zero-shot RE извлекать наши 23 типа при правильных entities?** "
        f"Для проверенного NLI приемлемая точность не показана: Gold diagnostic final P/R/F1 = "
        f"{n(gold['relation_end_to_end']['precision'])}/{n(gold['relation_end_to_end']['recall'])}/"
        f"{n(gold['relation_end_to_end']['f1'])}, "
        f"{gold['relation_end_to_end']['tp']} TP и {gold['relation_end_to_end']['fp']} FP "
        "при правильных endpoint spans/kinds. Доступны только 22 experimental hypotheses, "
        "MEETS_REQUIREMENT unsupported; Qualified F1 = 0. GLiREL не воспроизвёл официальный "
        "control и остановлен, поэтому независимого рабочего RE comparator не получено. "
        "Нельзя переносить этот отрицательный результат на все существующие RE checkpoints.\n\n"
        "3. **Продолжать C или перейти к B?** Закрыть текущий pretrained-only C и перейти к "
        "отдельному протоколу B. Gold entities снимают NER bottleneck, однако 94.63% final "
        "Gold-mode predictions всё равно не совпадают с annotation. Union потребовал 53.3 минуты "
        "wall time на CPU против 6.0 минуты control и дал лишь 8 дополнительных TP при 27723 "
        "дополнительных FP; end-to-end F1 упал 0.0196 → 0.0034. Дальнейшее расширение "
        "candidate sets не решает семантическую точность RE или полноту qualifiers. "
        "C-001/C-RESTART-001 и все новые отрицательные результаты сохраняются как исследовательские baselines."
    )
    parts = ["# C-DIAG-002 — финальная диагностика pretrained-only архитектуры C",
             "Дата завершения: 2026-10-30, Europe/Moscow; primary inference 29–30 октября. "
             "Все пять вариантов реально выполнены на тех же 12 DEV. "
             "Результаты exploratory: DEV уже исследован; это не независимый финальный тест.",
             conclusions,
             "## Протокол, данные и доступ",
             "Base `b368a63`; отдельная ветка `c-diag-002-pretrained-only`. "
             "[Протокол](C_DIAG_002_PROTOCOL.md) зафиксирован коммитом `36f2b97` до новых predictions; "
             "[config](../config/c_diag_002.json), модельные hashes и evaluator закреплены до scoring. "
             "Gold F1 не использован для выбора mapping, thresholds, verbalizations или объединения. "
             "Ontology v1.0 (12 kinds, 23 relations), C-001 и C-RESTART-001 сохранены. "
             "CALIBRATION/HOLDOUT не открывались. Обучения, fit, новых heads и генеративных LLM нет.",
             "Gold применяется только offline evaluator и в отдельно разрешённом режиме `gold_entities_re_diagnostic`. "
             "В этом режиме RE получает проекцию Gold spans/kinds/IDs с UNKNOWN roles; "
             "Gold relations, polarity и qualifiers не передаются. Source-only extraction защищён process-wide "
             "audit hook от Gold, network, закрытых split и исторического fitted RE. "
             "Все пять predictions сохранены до общего scoring; hashes проверены. "
             "Audits подтверждают режим доступа и завершение процесса, не научную истинность assertions.",
             f"Protocol SHA256: `{file_sha256('reports/C_DIAG_002_PROTOCOL.md')}`. "
             f"Config file SHA256: `{file_sha256('config/c_diag_002.json')}`.",
             "## Scientific NER: доступность, head и mapping",
             "Сводка рассмотренных checkpoints: [model_survey.json](c_diag_002/model_survey.json). "
             "Проверены доступные families MatSciBERT/MatBERT/SciBERT и model cards/configs. "
             "Это ограниченный обзор: невыбранные модели не объявляются неработающими или недоступными."]
    rows = []
    for alias, spec in cfg["scientific_ner"].items():
        model = read_json(out / f"{alias}_run.json")["ner_model"]
        rows.append([names[alias], f"`{spec['checkpoint']}@{spec['revision']}`", str(model["classifier_shape"]),
                     "; ".join(f"{label} → {kind or 'unsupported'}" for label, kind in spec["mapping"].items())])
    parts += [table(["Модель", "Checkpoint / revision", "Saved classifier.weight", "Mapping"], rows),
              "Первичные источники: [CHEM model card](https://huggingface.co/teman67/matscibert-chem-ner/blob/e8bd203aec2ec59c8705f199c108b976d170d953/README.md), "
              "[electrocatalyst model card](https://huggingface.co/Dmjdxb/batterybert-electrocatalyst-ner-v2/blob/13e458cef185382b7a54ad2352f51946aa18b0f3/README.md). "
              "Точные config URLs всех проверенных моделей сохранены в survey; заявленные авторами external F1 "
              "не переносятся на нашу ontology и DEV. CHEM использует MatSciBERT backbone, но NER "
              "fine-tuning его автором выполнен на CHEMDNER (биомедицинские химические названия), "
              "а не на нашей materials ontology. Electrocatalyst card заявляет fine-tuned MatSciBERT, "
              "но подробно не документирует обучающий корпус; история обучения независимо не аудировалась. "
              "Strict loading доказывает наличие сохранённой head, а не достоверность external benchmark.",
              "**\* Electrocatalyst provenance caveat:** после freeze, до Gold scoring обнаружено расхождение "
              "с card: сохранённый model/tokenizer vocabulary 28996, cased; официальный MatSciBERT "
              "vocabulary 31090, а CHEM checkpoint — 31090, uncased. Пакет battery внутренне согласован "
              "(model vocab = tokenizer vocab), поэтому использован как опубликован, без замены tokenizer. "
              "Его происхождение от MatSciBERT не подтверждено. Это проверка domain-labelled NER checkpoint, "
              "а не независимо подтверждённого materials-pretrained backbone. Исходный protocol сохраняет "
              "первоначальную предпосылку выбора; [metadata audit](c_diag_002/metadata_audit.json) фиксирует "
              "обнаруженное ограничение. Названия выбранных вариантов и mapping не изменены по Gold.",
              "Оба checkpoint загружены как `BertForTokenClassification` с нулём missing/unexpected/mismatched keys; "
              "num_labels и id2label не переопределялись. Сохранённая NER-head действительно присутствует. "
              "Авторы описывают их как fine-tuned NER checkpoints; в данном исследовании параметры не обучались. "
              "CHEM не определяет target role, поэтому не превращается в ANALYTE. DURATION/CURRENT/FAILURE_MODE "
              "не имеют однозначного kind и не заменяются OTHER. Scientific NER не поддерживают остальные kinds. "
              "Mapping CONDITION → OPERATING_CONDITION является заранее принятой интерпретацией source schema, "
              "а не Gold-based contextual typing.",
              "Контроль: `urchade/gliner_small-v2.1@4e091416cf7c3481db542c2a3d26156916f3a47f`, "
              "threshold .35, прежние 12 prompts, окна 140/30 words. Scientific decoding — pretrained BIO argmax + "
              "Transformers simple aggregation, без дополнительного threshold. Chem: окна 45/10, limit 128 subtokens; "
              "battery: 100/20, limit 512. Boundary repairs отсутствуют. Union задан заранее: exact span/kind union "
              "трёх моделей, max score для дубликата, conflicting kinds/nested spans сохраняются.",
              "Base encoders `m3rg-iitd/matscibert` и `allenai/scibert_scivocab_uncased` не приняты за готовый NER. "
              "MaterioMiner checkpoint имеет LABEL_n без опубликованного decoding в model card. "
              "Другие saved heads с неполными cards или иной schema перечислены как untested, без выдуманного mapping. "
              "Наличие готовых scientific NER подтверждено; готовая полноценная поддержка всех 12 kinds не найдена "
              "в рамках выбранных двух моделей. Дополнительно подтверждена публикация "
              "[авторского архива MatBERT-NER](https://figshare.com/articles/software/MatBERT-NER_models/15087276), "
              "4.65 GB zip, version 1 ([metadata](https://api.figshare.com/v2/articles/15087276)); "
              "его отдельный [runtime](https://github.com/CederGroupHub/MatBERT_NER) не интегрировался "
              "в рамках бюджета двух выбранных checkpoints. Он остаётся UNTESTED, а не UNAVAILABLE. "
              "Поэтому результат нельзя трактовать как доказательство отсутствия материаловедческого NER "
              "или невозможности улучшить NER другими готовыми моделями.",
              "## Качество NER и candidate recall"]
    rows = []
    for v in ("control", "chem", "battery", "union"):
        m = results[v]["micro"]
        span, typed, cand = m["entity_span"], m["entity_typed"], m["candidate_pair_recall"]
        rows.append([names[v], results[v]["counts"]["entities"],
                     "/".join(n(span[k]) for k in ("precision", "recall", "f1")),
                     "/".join(n(typed[k]) for k in ("precision", "recall", "f1")),
                     f"{cand['covered']}/{cand['gold']} = {n(cand['recall'])}"])
    parts += [table(["Вариант", "Entities", "Exact span P/R/F1", "Typed P/R/F1", "Candidate recall"], rows),
              "Одинаковый denominator всех 12 DEV; exact Unicode offsets, one-to-one multiset matching. "
              "Две kind-гипотезы одного span остаются отдельными predictions и могут увеличить FP даже "
              "в untyped metric; relaxed/coreference matching отсутствует. Контроль воспроизвёл исходные "
              "C-RESTART-001 counts и метрики точно, включая 71/390 candidates и 4503 final assertions.",
              "Покрытие Gold по kind: **typed TP / Gold (exact boundary TP без требования kind)**. "
              "Ноль у неподдержанного kind — реальное ограничение данного mapping; denominator не сокращён."]
    rows = []
    for kind in ontology["entities"]:
        row = [kind]
        for v in ("control", "chem", "battery", "union"):
            k = results[v]["entity_per_kind"][kind]
            row.append(f"{k['typed']['tp']}/{k['gold']} ({k['exact_boundary_covered']})")
        rows.append(row)
    parts += [table(["Kind", "Control", "CHEM", "Battery", "Union"], rows),
              "Положительный локальный NER signal: CHEM нашёл exact границы 57/77 Gold ANALYTE "
              "против 41/77 у control, но назначает им MATERIAL, поскольку contextual target role "
              "не входит в его source schema. Поэтому это не 57 typed ANALYTE TP и не готовое решение "
              "для ontology. В данном протоколе не добавлялись Gold-based role repairs или новые "
              "contextual typing rules после evaluation.",
              "## RE с предсказанными entities и Gold-диагностика",
              "Один и тот же `cross-encoder/nli-deberta-v3-xsmall@a150876415327c80daeff35ca6f68f5ed8cf5c24`: "
              "готовая трёхклассовая NLI-head, entailment >.5, batch 32, single sentence, 512 subtokens. "
              "22 frozen hypotheses, `MEETS_REQUIREMENT` unsupported. Это entailment-based zero-shot RE, "
              "а не специализированный scientific relation classifier. 22 статуса EXPERIMENTAL_ZERO_SHOT "
              "не означают доказанной поддержки scientific semantics. Scores не являются "
              "калиброванными вероятностями истинности научного факта."]
    rows = []
    for v in names:
        m = results[v]["micro"]
        raw, final, qualified = m["relation_raw_end_to_end"], m["relation_end_to_end"], m["qualified_relation"]
        rows.append([names[v], "/".join(n(raw[k]) for k in ("precision", "recall", "f1")),
                     "/".join(n(final[k]) for k in ("precision", "recall", "f1")),
                     f"{final['tp']}/{final['fp']}/{final['fn']}", final["predicted"], n(qualified["f1"])])
    parts += [table(["Вариант", "Raw P/R/F1", "Final P/R/F1", "Final TP/FP/FN", "Final count", "Qualified F1"], rows),
              "Строка GOLD — только RE diagnostic, **не end-to-end качество архитектуры**. "
              "Основной relation denominator сохраняет все 390 supported Gold assertions, включая "
              "недоступные contexts/capabilities. Qualified metric дополнительно требует polarity, modality "
              "и полный qualifier set; UNKNOWN completeness никогда не получает TP. Exact evidence "
              "проверяет происхождение цитаты, не entailment или scientific truth."]
    g = results["gold"]["micro"]
    parts += [f"При Gold entities candidate recall: **{g['candidate_pair_recall']['covered']}/390**. "
              f"Фактически запрошены правильные typed endpoints + Gold label для "
              f"{results['gold']['actual_gold_label_query_coverage']['covered']}/390 edges; "
              "эта проверка по model traces отделяет serving pair от serving конкретного label. "
              "Условная final RE метрика на доступных candidate endpoints: " +
              "/".join(n(g["relation_given_candidate_endpoints"][k]) for k in ("precision", "recall", "f1")) +
              "; это дополнительный diagnostic view с меньшим denominator, не замена основной метрике.",
              "## Где теряются Gold relations"]
    stages = results["control"]["loss_audit"]["order"]
    parts += [table(["Осталось exact Gold edges"] + list(names.values()),
                    [[s] + [results[v]["loss_audit"]["correct_assertions_remaining"][s] for v in names] for s in stages]),
              "Разложение final FN без двойного счёта: **control** — 304 boundary losses + 10 kind losses "
              "+ 5 candidate exclusions + 21 RE losses + 2 scope losses = 342 FN. **Union** — "
              "278 + 22 + 5 + 27 + 2 = 334 FN. **Gold diagnostic** — 0 NER losses, 16 candidate "
              "exclusions, 1 дополнительный unsupported capability, 70 RE losses, 20 scope losses = 107 FN. "
              "Всего Gold содержит два MEETS_REQUIREMENT: один уже исключён candidate stage, поэтому "
              "последовательный capability gate теряет только один дополнительный edge.",
              "Funnel — последовательный учёт доступности exact typed directed Gold edges. "
              "Это не отдельный causal ablation каждого gate. NLI запрашивает только domain/range-compatible "
              "hypotheses, а raw score проецируется на альтернативные kind-гипотезы тех же spans; "
              "final domain/range gate удаляет несовместимые проекции. Полные stage losses, reasons и "
              "raw/final per-relation counts опубликованы для каждого варианта в JSON. "
              "Candidate stage ограничивает endpoints одной sentence и исключает пары одинаковых spans "
              "даже с разными kinds; отдельного distance/budget cap нет.",
              "Gold diagnostic: total / TP / FP после каждого gate:",
              table(["Stage", "Total", "TP", "FP"],
                    [[stage, row["total"], row["tp"], row["fp"]] for stage, row in results["gold"]["gate_totals"].items()]),
              "Evidence checks при невалидных offsets/hash прерывают run; в завершённых runs "
              "отдельного удаления assertions за invalid evidence не произошло. Нулевые потери на evidence "
              "не означают подтверждения смысла relation. В Gold mode domain/range удалил 44 FP и 0 TP; "
              "scope удалил 754 FP и 20 TP. Из этих 20 TP: 15 потеряны на NEGATION_ATTACHMENT, "
              "4 на REQUIRED_DIRECTION, 1 на CROSS_CLAUSE_SCOPE. Финальная precision выросла лишь "
              "с 4.97% до 5.37%; фильтры не устранили основную массу RE ошибок.",
              "Final FP разложены по first applicable category; правильные endpoints не гарантируют правильного relation. "
              "Категория ‘relation’ включает несовпадение с разметкой и потенциально неаннотированные связи."]
    cats = ["at_least_one_non_gold_boundary", "gold_boundaries_wrong_entity_kind", "exact_gold_endpoints_wrong_or_unannotated_relation"]
    parts += [table(["Вариант", "Boundary FP", "Kind FP", "Relation/unannotated FP"],
                    [[names[v]] + [results[v]["final_fp_categories"].get(k, 0) for k in cats] for v in names]),
              "## Все 23 типа отношений",
              "Final TP/FP/FN; значения 0/0/0 означают отсутствие и Gold support, и predictions. "
              "Zero-support types нельзя считать проверенной научной capability."]
    rows = []
    for label in ontology["relations"]:
        row = [label]
        for v in ("control", "union", "gold"):
            m = results[v]["loss_audit"]["per_relation"][label]["final"]
            row.append("/".join(str(m[k]) for k in ("tp", "fp", "fn")))
        rows.append(row)
    parts += [table(["Relation", "Control", "Union", "Gold diagnostic"], rows),
              "## Abstentions и unsupported"]
    rows = []
    for v in names:
        m = results[v]["micro"]
        abst = m["abstention"]
        rows.append([names[v], f"{abst['abstained']}/{abst['candidate_pairs']} = {n(abst['rate'])}",
                     str(results[v]["unsupported_gold_relations"]), results[v]["counts"]["skipped_re_contexts"]])
    parts += [table(["Вариант", "Pair abstentions", "Gold unsupported labels", "RE context skips"], rows),
              "MEETS_REQUIREMENT не оценивается моделью и не переименовывается в другое отношение. "
              "Отсутствие unsupported proposals само по себе не означает поддержку всех 23 labels. "
              "Pair abstentions и число rejected raw proposals имеют разные denominators; один pair может "
              "иметь несколько raw labels/reasons. Предсказанные неподдержанные source-NER классы battery "
              "сосчитаны в `scientific_output_counts` manifests, без переноса в ontology.",
              "## Стоимость реального CPU inference",
              "Windows, Python 3.11.9, 8 physical / 16 logical CPU, host RAM около 31.3 GiB, "
              "четыре torch threads, float32, deterministic algorithms. Варианты запускались последовательно "
              "в fresh processes. Wall/CPU timer начинается до ML imports и включает load/hash checks, "
              "inference и serialization predictions. Final provenance hashing/manifest serialization, "
              "interpreter startup, downloads, tests, probes и evaluator исключены. "
              "Один shared-host measurement без доверительных интервалов и контроля file cache."]
    rows = []
    for v in names:
        p, s = results[v]["performance"], results[v]["stage_ms"]
        run = read_json(out / f"{v}_run.json")
        rows.append([names[v], f"{p['wall_ms']/1000:.3f}", f"{p['cpu_ms']/1000:.3f}",
                     f"{p['peak_ram_bytes']/1024**3:.3f}", f"{s['ner_ms']/1000:.3f}",
                     f"{s['pretrained_re_ms']/1000:.3f}", run["re_score_summary"]["pair_label_scores"],
                     f"{p['ner_encoder_calls']}/{p['re_encoder_calls']}"])
    parts += [table(["Вариант", "Wall s", "CPU s", "Peak GiB", "NER inference wall s", "RE wall s", "NLI queries", "NER/RE forwards"], rows),
              "Peak — Windows PeakWorkingSetSize за весь процесс с обоими компонентами, не отдельная "
              "NER peak memory. NER column — сумма wall timers predict(), без загрузки; в primary run "
              "NER CPU отдельно не инструментировалось. Gold column NER time — чтение fixed entity inventory, "
              "без NER encoder. Все peaks ниже 32 GiB. Полные load times, API calls, token limits и file "
              "hashes — в *_run.json. Время неудачных запусков не суммируется с основными вариантами. "
              "Дополнительный source-only NER cost probe повторил battery NER на всех 12 DEV: 790 entities, "
              "16712 направленных pairs и 28060 type-compatible hypotheses до проверки linguistic negation. "
              "Он объясняет рост RE workload и не использует Gold. Этот короткий probe выполнялся во время "
              "battery RE; его нагрузка не вычтена из измеренного battery wall/CPU, что дополнительно "
              "ограничивает точность сравнения скорости. [Профиль](c_diag_002/battery_ner_cost_probe.json).",
              "Отдельные NER-only fresh-process measurements по заранее записанному "
              "[cost supplement](C_DIAG_002_COST_PROTOCOL.md), после primary runs и до Gold scoring. "
              "RE и Gold не загружаются; все predicted inventories точно совпали с primary. "
              "Peak включает только NER, source loading и небольшой inventory check. Total times включают "
              "ML imports/load, predict times — только NER.predict()."]
    costs = read_json(out / "ner_costs.json")
    parts += [table(["NER", "Total wall s", "Total CPU s", "Predict wall s", "Predict CPU s", "Peak GiB"],
                    [[names[v], f"{p['wall_s']:.3f}", f"{p['cpu_s']:.3f}", f"{p['predict_wall_s']:.3f}",
                      f"{p['predict_cpu_s']:.3f}", f"{p['peak_ram_bytes']/1024**3:.3f}"] for v, p in costs.items()]),
              "## GLiREL: ограниченная проверка воспроизводимости"]
    probe = read_json(out / "glirel_probe.json")
    parts += [f"Повторён [официальный Doctor Who control](https://github.com/jackboyla/GLiREL#usage) "
              "через `GLiREL.from_pretrained(..., strict=True)` "
              f"для `jackboyla/glirel-large-v0@{probe['revision']}`, pinned runtime "
              "GLiREL 1.0.1 / Flair 0.14.0 / Transformers 4.51.3. "
              "Сохранены исходные tokens/NER/labels, threshold=0, top_k=1. "
              "Ожидались два characters с score около .99. Фактически получены " +
              ", ".join(f"{p['label']}={p['score']:.6f}" for p in probe["output"]) +
              "; characters отсутствует. Strict loading завершился, совпал SHA256 полного checkpoint. "
              "Локальная runtime config меняет лишь путь backbone; официальный loader получает "
              "канонический encoder export из того же checkpoint.",
              f"Wall {probe['wall_s']:.3f} s, CPU {probe['cpu_s']:.3f} s, peak "
              f"{probe['peak_ram_bytes']/1024**3:.3f} GiB, budget 600 s. **FAIL; GLiREL остановлен.** "
              "Blocker: опубликованное поведение не воспроизведено, root cause не установлен. "
              "Это согласуется с предыдущей диагностикой нескольких runtimes; нельзя объявлять причиной "
              "конкретную библиотеку без доказательств. Нулевые DEV predictions из невалидированного runtime "
              "не включены в таблицу качества. Рабочего независимого standalone zero-shot RE comparator "
              "в этом ограниченном исследовании не получено; межмодельного сравнения RE на одинаковых "
              "entities нет. NLI на разных entities отвечает на вопрос о bottleneck, но не заменяет такой comparator.",
              "REBEL исключён как генеративный, GLiNER-Relex как joint arm F; fixed-label "
              "TACRED/SciERC heads не предоставляют наши 23 класса. Это не доказательство отсутствия "
              "других пригодных checkpoints. Полный output/hashes: [glirel_probe.json](c_diag_002/glirel_probe.json).",
              "## Реальные DEV примеры"]
    for v in ("chem", "battery"):
        for key in ("entity_TP", "entity_FP_KIND", "entity_FP_BOUNDARY"):
            e = results[v]["selected_examples"].get(key)
            if e:
                parts += [f"**{names[v]} / {e['document']} / {key}**: `{e['surface']}` "
                          f"[{e['start']},{e['end']}) → {e['predicted_kind']}; "
                          f"Gold kinds на точной границе: {e['gold_kinds_at_exact_boundary'] or 'нет'}. "
                          "BOUNDARY означает отсутствие exact Gold span, без самостоятельной semantic adjudication.",
                          "> " + e["evidence"].replace("\n", " ").rstrip()]
    for v, key in [("control", "relation_TP"), ("control", "relation_FP"), ("control", "relation_FN_NER"),
                   ("union", "relation_FN_NER"), ("gold", "relation_FP"), ("gold", "relation_FN_RE_OR_CAPABILITY"), ("gold", "relation_FN_GATES")]:
        e = results[v]["selected_examples"].get(key)
        if e:
            parts += [f"**{names[v]} / {e['document']} / {key}**: `{e['head']}` ({e['head_kind']}) "
                      f"→ {e['relation']} → `{e['tail']}` ({e['tail_kind']}). " +
                      (f"Score {e['score']:.4f}." if "score" in e else f"Loss stage: {e['stage']}."),
                      "> " + e["evidence"].replace("\n", " ").rstrip()]
    parts += ["Примеры выбраны детерминированно после predictions для иллюстрации; они не использованы "
              "для исправления prompts/thresholds. Более подробные examples и chunk IDs — в variant JSON.",
              "## Проверки",
              "`python -m pytest -q`: **47 passed, 1 deselected**. Исключён только сохранённый "
              "historical_training test с C-001 fitting; он не выполнялся. Проверены frozen ontology, "
              "DEV allowlist/hashes, Unicode/nested spans, directed pairs, scope, strict qualifiers, "
              "multiset matching, exact union, Gold entity-only projection, запрет закрытых путей и "
              "различие pair serving от фактической оценки relation label. "
              "[tests.json](c_diag_002/tests.json). Реальные model runs и NER-only benchmarks — "
              "отдельные integration measurements, не synthetic fixtures. Evaluator проверил "
              "prediction/config/source hashes и audits всех пяти runs, полное совпадение control "
              "с C-RESTART-001 и разбиение всех final FP по причинам. Все четыре NER-only inventories "
              "точно совпали с основными predictions. Исторические C-001/C-RESTART-001 отчёты, "
              "код zero-shot pipeline и ontology byte-for-byte сохранены; main остался на 06f2bac, "
              "c-restart-001-zero-shot — на b368a63.",
              "## Ограничения и воспроизведение",
              "12 повторно использованных DEV, один owner review и AI-origin annotations; независимого "
              "второго annotator и human semantic-support audit нет. D024 имеет incomplete relations, "
              "поэтому часть unmatched predictions может отражать неполноту annotation. Exact matching "
              "чувствителен к границам и contextual kinds. Mapping/schema mismatch и узкое покрытие двух "
              "scientific моделей не позволяют обобщить отрицательный результат на все scientific NER. "
              "Single-sentence contexts не проверяют coreference/cross-sentence RE. Не все 23 типа могут "
              "иметь Gold support на этих DEV. Qualifier attachment остаётся неполным. GLiREL runtime "
              "не валидирован, поэтому прямой scientific RE comparator отсутствует. B здесь не запускалась; "
              "переход к B не означает, что её превосходство уже доказано. Пересечение авторских "
              "pretraining/fine-tuning corpora с исходной литературой DEV независимо не проверялось.",
              "[Команды и сведения о неудачной instrumentation-попытке](C_DIAG_002_REPRODUCE.md). "
              "Полные predictions/query traces/Gold/weights остаются локально под gitignore; "
              "публикуются aggregate metrics, immutable hashes, access audits и короткие DEV excerpts. "
              "[comparison.json](c_diag_002/comparison.json), [control](c_diag_002/control.json), "
              "[chem](c_diag_002/chem.json), [battery](c_diag_002/battery.json), "
              "[union](c_diag_002/union.json), [Gold diagnostic](c_diag_002/gold.json)."]
    Path("reports/C_DIAG_002_REPORT.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
