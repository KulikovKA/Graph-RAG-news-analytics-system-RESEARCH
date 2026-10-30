# C-DIAG-002 — финальная диагностика pretrained-only архитектуры C

Дата завершения: 2026-10-30, Europe/Moscow; primary inference 29–30 октября. Все пять вариантов реально выполнены на тех же 12 DEV. Результаты exploratory: DEV уже исследован; это не независимый финальный тест.

**Рекомендация: CLOSE C AND MOVE TO B.** Проверенный pretrained-only pipeline не обеспечивает точности, достаточной для автоматического построения научного графа. Это вывод о завершённом ограниченном исследовании, не доказательство невозможности всех pretrained NER/RE подходов. B здесь не измерялась и не объявляется доказанно лучшей.

1. **Могут ли готовые scientific NER заметно улучшить candidate recall?** В проверенном наборе получен только ограниченный прирост через union: самостоятельные CHEM и battery имеют 0/390 typed relation candidates. Заранее заданный union добавил 14 edges: 71/390 → 85/390 (+3.59 процентного пункта), при падении typed entity F1 0.4592 → 0.2802. Union нашёл ещё 20 exact MATERIAL и 9 OPERATING_CONDITION mentions, но не улучшил остальные kinds. CHEM — готовый scientific single-class NER; материалы/условия battery проверены как опубликованный domain checkpoint с неподтверждённой MatSciBERT lineage. Более широкие авторские MatBERT-NER archives остаются непроверенными: обобщать результат на весь scientific NER нельзя.

2. **Способен ли готовый zero-shot RE извлекать наши 23 типа при правильных entities?** Для проверенного NLI приемлемая точность не показана: Gold diagnostic final P/R/F1 = 0.0537/0.7256/0.0999, 283 TP и 4990 FP при правильных endpoint spans/kinds. Доступны только 22 experimental hypotheses, MEETS_REQUIREMENT unsupported; Qualified F1 = 0. GLiREL не воспроизвёл официальный control и остановлен, поэтому независимого рабочего RE comparator не получено. Нельзя переносить этот отрицательный результат на все существующие RE checkpoints.

3. **Продолжать C или перейти к B?** Закрыть текущий pretrained-only C и перейти к отдельному протоколу B. Gold entities снимают NER bottleneck, однако 94.63% final Gold-mode predictions всё равно не совпадают с annotation. Union потребовал 53.3 минуты wall time на CPU против 6.0 минуты control и дал лишь 8 дополнительных TP при 27723 дополнительных FP; end-to-end F1 упал 0.0196 → 0.0034. Дальнейшее расширение candidate sets не решает семантическую точность RE или полноту qualifiers. C-001/C-RESTART-001 и все новые отрицательные результаты сохраняются как исследовательские baselines.

## Протокол, данные и доступ

Base `b368a63`; отдельная ветка `c-diag-002-pretrained-only`. [Протокол](C_DIAG_002_PROTOCOL.md) зафиксирован коммитом `36f2b97` до новых predictions; [config](../config/c_diag_002.json), модельные hashes и evaluator закреплены до scoring. Gold F1 не использован для выбора mapping, thresholds, verbalizations или объединения. Ontology v1.0 (12 kinds, 23 relations), C-001 и C-RESTART-001 сохранены. CALIBRATION/HOLDOUT не открывались. Обучения, fit, новых heads и генеративных LLM нет.

Gold применяется только offline evaluator и в отдельно разрешённом режиме `gold_entities_re_diagnostic`. В этом режиме RE получает проекцию Gold spans/kinds/IDs с UNKNOWN roles; Gold relations, polarity и qualifiers не передаются. Source-only extraction защищён process-wide audit hook от Gold, network, закрытых split и исторического fitted RE. Все пять predictions сохранены до общего scoring; hashes проверены. Audits подтверждают режим доступа и завершение процесса, не научную истинность assertions.

Protocol SHA256: `589c6b8e9bfb02292604b2852978c8cb87575fd01c93e9ea69217d6493ccf5cd`. Config file SHA256: `4daef3ec271ae15a23efd8edb5576faccf27a06f451dd07d20762d72741933ae`.

## Scientific NER: доступность, head и mapping

Сводка рассмотренных checkpoints: [model_survey.json](c_diag_002/model_survey.json). Проверены доступные families MatSciBERT/MatBERT/SciBERT и model cards/configs. Это ограниченный обзор: невыбранные модели не объявляются неработающими или недоступными.

| Модель | Checkpoint / revision | Saved classifier.weight | Mapping |
|---|---|---|---|
| MatSciBERT CHEM | `teman67/matscibert-chem-ner@e8bd203aec2ec59c8705f199c108b976d170d953` | [3, 768] | CHEM → MATERIAL |
| Electrocatalyst NER* | `Dmjdxb/batterybert-electrocatalyst-ner-v2@13e458cef185382b7a54ad2352f51946aa18b0f3` | [11, 768] | MATERIAL → MATERIAL; CONDITION → OPERATING_CONDITION; DURATION → unsupported; FAILURE_MODE → unsupported; CURRENT → unsupported |

Первичные источники: [CHEM model card](https://huggingface.co/teman67/matscibert-chem-ner/blob/e8bd203aec2ec59c8705f199c108b976d170d953/README.md), [electrocatalyst model card](https://huggingface.co/Dmjdxb/batterybert-electrocatalyst-ner-v2/blob/13e458cef185382b7a54ad2352f51946aa18b0f3/README.md). Точные config URLs всех проверенных моделей сохранены в survey; заявленные авторами external F1 не переносятся на нашу ontology и DEV. CHEM использует MatSciBERT backbone, но NER fine-tuning его автором выполнен на CHEMDNER (биомедицинские химические названия), а не на нашей materials ontology. Electrocatalyst card заявляет fine-tuned MatSciBERT, но подробно не документирует обучающий корпус; история обучения независимо не аудировалась. Strict loading доказывает наличие сохранённой head, а не достоверность external benchmark.

**\* Electrocatalyst provenance caveat:** после freeze, до Gold scoring обнаружено расхождение с card: сохранённый model/tokenizer vocabulary 28996, cased; официальный MatSciBERT vocabulary 31090, а CHEM checkpoint — 31090, uncased. Пакет battery внутренне согласован (model vocab = tokenizer vocab), поэтому использован как опубликован, без замены tokenizer. Его происхождение от MatSciBERT не подтверждено. Это проверка domain-labelled NER checkpoint, а не независимо подтверждённого materials-pretrained backbone. Исходный protocol сохраняет первоначальную предпосылку выбора; [metadata audit](c_diag_002/metadata_audit.json) фиксирует обнаруженное ограничение. Названия выбранных вариантов и mapping не изменены по Gold.

Оба checkpoint загружены как `BertForTokenClassification` с нулём missing/unexpected/mismatched keys; num_labels и id2label не переопределялись. Сохранённая NER-head действительно присутствует. Авторы описывают их как fine-tuned NER checkpoints; в данном исследовании параметры не обучались. CHEM не определяет target role, поэтому не превращается в ANALYTE. DURATION/CURRENT/FAILURE_MODE не имеют однозначного kind и не заменяются OTHER. Scientific NER не поддерживают остальные kinds. Mapping CONDITION → OPERATING_CONDITION является заранее принятой интерпретацией source schema, а не Gold-based contextual typing.

Контроль: `urchade/gliner_small-v2.1@4e091416cf7c3481db542c2a3d26156916f3a47f`, threshold .35, прежние 12 prompts, окна 140/30 words. Scientific decoding — pretrained BIO argmax + Transformers simple aggregation, без дополнительного threshold. Chem: окна 45/10, limit 128 subtokens; battery: 100/20, limit 512. Boundary repairs отсутствуют. Union задан заранее: exact span/kind union трёх моделей, max score для дубликата, conflicting kinds/nested spans сохраняются.

Base encoders `m3rg-iitd/matscibert` и `allenai/scibert_scivocab_uncased` не приняты за готовый NER. MaterioMiner checkpoint имеет LABEL_n без опубликованного decoding в model card. Другие saved heads с неполными cards или иной schema перечислены как untested, без выдуманного mapping. Наличие готовых scientific NER подтверждено; готовая полноценная поддержка всех 12 kinds не найдена в рамках выбранных двух моделей. Дополнительно подтверждена публикация [авторского архива MatBERT-NER](https://figshare.com/articles/software/MatBERT-NER_models/15087276), 4.65 GB zip, version 1 ([metadata](https://api.figshare.com/v2/articles/15087276)); его отдельный [runtime](https://github.com/CederGroupHub/MatBERT_NER) не интегрировался в рамках бюджета двух выбранных checkpoints. Он остаётся UNTESTED, а не UNAVAILABLE. Поэтому результат нельзя трактовать как доказательство отсутствия материаловедческого NER или невозможности улучшить NER другими готовыми моделями.

## Качество NER и candidate recall

| Вариант | Entities | Exact span P/R/F1 | Typed P/R/F1 | Candidate recall |
|---|---|---|---|---|
| GLiNER control | 478 | 0.5795/0.4703/0.5192 | 0.5126/0.4160/0.4592 | 71/390 = 0.1821 |
| MatSciBERT CHEM | 231 | 0.4719/0.1851/0.2659 | 0.2251/0.0883/0.1268 | 0/390 = 0.0000 |
| Electrocatalyst NER* | 790 | 0.0544/0.0730/0.0624 | 0.0228/0.0306/0.0261 | 0/390 = 0.0000 |
| GLiNER + оба NER | 1367 | 0.2414/0.5603/0.3374 | 0.2004/0.4652/0.2802 | 85/390 = 0.2179 |

Одинаковый denominator всех 12 DEV; exact Unicode offsets, one-to-one multiset matching. Две kind-гипотезы одного span остаются отдельными predictions и могут увеличить FP даже в untyped metric; relaxed/coreference matching отсутствует. Контроль воспроизвёл исходные C-RESTART-001 counts и метрики точно, включая 71/390 candidates и 4503 final assertions.

Покрытие Gold по kind: **typed TP / Gold (exact boundary TP без требования kind)**. Ноль у неподдержанного kind — реальное ограничение данного mapping; denominator не сокращён.

| Kind | Control | CHEM | Battery | Union |
|---|---|---|---|---|
| MATERIAL | 104/161 (108) | 52/161 (52) | 9/161 (9) | 124/161 (126) |
| DEVICE | 23/57 (28) | 0/57 (0) | 0/57 (0) | 23/57 (28) |
| PROPERTY | 34/75 (34) | 0/75 (0) | 0/75 (0) | 34/75 (34) |
| PERFORMANCE | 21/63 (31) | 0/63 (0) | 0/63 (0) | 21/63 (31) |
| ANALYTE | 41/77 (41) | 0/77 (57) | 0/77 (23) | 41/77 (65) |
| PROCESS | 13/51 (15) | 0/51 (0) | 0/51 (0) | 13/51 (15) |
| MECHANISM | 1/13 (2) | 0/13 (0) | 0/13 (0) | 1/13 (2) |
| OPERATING_CONDITION | 0/14 (0) | 0/14 (0) | 9/14 (9) | 9/14 (9) |
| MORPHOLOGY | 1/38 (6) | 0/38 (0) | 0/38 (2) | 1/38 (8) |
| TECHNOLOGY | 4/5 (4) | 0/5 (0) | 0/5 (0) | 4/5 (4) |
| APPLICATION | 3/28 (7) | 0/28 (0) | 0/28 (0) | 3/28 (7) |
| OTHER | 0/7 (1) | 0/7 (0) | 0/7 (0) | 0/7 (1) |

Положительный локальный NER signal: CHEM нашёл exact границы 57/77 Gold ANALYTE против 41/77 у control, но назначает им MATERIAL, поскольку contextual target role не входит в его source schema. Поэтому это не 57 typed ANALYTE TP и не готовое решение для ontology. В данном протоколе не добавлялись Gold-based role repairs или новые contextual typing rules после evaluation.

## RE с предсказанными entities и Gold-диагностика

Один и тот же `cross-encoder/nli-deberta-v3-xsmall@a150876415327c80daeff35ca6f68f5ed8cf5c24`: готовая трёхклассовая NLI-head, entailment >.5, batch 32, single sentence, 512 subtokens. 22 frozen hypotheses, `MEETS_REQUIREMENT` unsupported. Это entailment-based zero-shot RE, а не специализированный scientific relation classifier. 22 статуса EXPERIMENTAL_ZERO_SHOT не означают доказанной поддержки scientific semantics. Scores не являются калиброванными вероятностями истинности научного факта.

| Вариант | Raw P/R/F1 | Final P/R/F1 | Final TP/FP/FN | Final count | Qualified F1 |
|---|---|---|---|---|---|
| GLiNER control | 0.0090/0.1282/0.0168 | 0.0107/0.1231/0.0196 | 48/4455/342 | 4503 | 0.0000 |
| MatSciBERT CHEM | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0/1306/390 | 1306 | 0.0000 |
| Electrocatalyst NER* | 0.0000/0.0000/0.0000 | 0.0000/0.0000/0.0000 | 0/9002/390 | 9002 | 0.0000 |
| GLiNER + оба NER | 0.0016/0.1487/0.0031 | 0.0017/0.1436/0.0034 | 56/32178/334 | 32234 | 0.0000 |
| GOLD entities (diagnostic) | 0.0497/0.7769/0.0935 | 0.0537/0.7256/0.0999 | 283/4990/107 | 5273 | 0.0000 |

Строка GOLD — только RE diagnostic, **не end-to-end качество архитектуры**. Основной relation denominator сохраняет все 390 supported Gold assertions, включая недоступные contexts/capabilities. Qualified metric дополнительно требует polarity, modality и полный qualifier set; UNKNOWN completeness никогда не получает TP. Exact evidence проверяет происхождение цитаты, не entailment или scientific truth.

При Gold entities candidate recall: **374/390**. Фактически запрошены правильные typed endpoints + Gold label для 373/390 edges; эта проверка по model traces отделяет serving pair от serving конкретного label. Условная final RE метрика на доступных candidate endpoints: 0.0537/0.7567/0.1002; это дополнительный diagnostic view с меньшим denominator, не замена основной метрике.

## Где теряются Gold relations

| Осталось exact Gold edges | GLiNER control | MatSciBERT CHEM | Electrocatalyst NER* | GLiNER + оба NER | GOLD entities (diagnostic) |
|---|---|---|---|---|---|
| gold | 390 | 390 | 390 | 390 | 390 |
| untyped_endpoints | 86 | 3 | 0 | 112 | 390 |
| typed_endpoints | 76 | 0 | 0 | 90 | 390 |
| candidate_pairs | 71 | 0 | 0 | 85 | 374 |
| model_served | 71 | 0 | 0 | 85 | 374 |
| re_capability | 71 | 0 | 0 | 85 | 373 |
| raw_re | 50 | 0 | 0 | 58 | 303 |
| domain_range | 50 | 0 | 0 | 58 | 303 |
| scope | 48 | 0 | 0 | 56 | 283 |
| evidence | 48 | 0 | 0 | 56 | 283 |
| final | 48 | 0 | 0 | 56 | 283 |

Разложение final FN без двойного счёта: **control** — 304 boundary losses + 10 kind losses + 5 candidate exclusions + 21 RE losses + 2 scope losses = 342 FN. **Union** — 278 + 22 + 5 + 27 + 2 = 334 FN. **Gold diagnostic** — 0 NER losses, 16 candidate exclusions, 1 дополнительный unsupported capability, 70 RE losses, 20 scope losses = 107 FN. Всего Gold содержит два MEETS_REQUIREMENT: один уже исключён candidate stage, поэтому последовательный capability gate теряет только один дополнительный edge.

Funnel — последовательный учёт доступности exact typed directed Gold edges. Это не отдельный causal ablation каждого gate. NLI запрашивает только domain/range-compatible hypotheses, а raw score проецируется на альтернативные kind-гипотезы тех же spans; final domain/range gate удаляет несовместимые проекции. Полные stage losses, reasons и raw/final per-relation counts опубликованы для каждого варианта в JSON. Candidate stage ограничивает endpoints одной sentence и исключает пары одинаковых spans даже с разными kinds; отдельного distance/budget cap нет.

Gold diagnostic: total / TP / FP после каждого gate:

| Stage | Total | TP | FP |
|---|---|---|---|
| raw_re | 6091 | 303 | 5788 |
| domain_range | 6047 | 303 | 5744 |
| scope | 5273 | 283 | 4990 |
| evidence | 5273 | 283 | 4990 |
| final | 5273 | 283 | 4990 |

Evidence checks при невалидных offsets/hash прерывают run; в завершённых runs отдельного удаления assertions за invalid evidence не произошло. Нулевые потери на evidence не означают подтверждения смысла relation. В Gold mode domain/range удалил 44 FP и 0 TP; scope удалил 754 FP и 20 TP. Из этих 20 TP: 15 потеряны на NEGATION_ATTACHMENT, 4 на REQUIRED_DIRECTION, 1 на CROSS_CLAUSE_SCOPE. Финальная precision выросла лишь с 4.97% до 5.37%; фильтры не устранили основную массу RE ошибок.

Final FP разложены по first applicable category; правильные endpoints не гарантируют правильного relation. Категория ‘relation’ включает несовпадение с разметкой и потенциально неаннотированные связи.

| Вариант | Boundary FP | Kind FP | Relation/unannotated FP |
|---|---|---|---|
| GLiNER control | 2771 | 543 | 1141 |
| MatSciBERT CHEM | 873 | 180 | 253 |
| Electrocatalyst NER* | 8959 | 18 | 25 |
| GLiNER + оба NER | 29254 | 1271 | 1653 |
| GOLD entities (diagnostic) | 0 | 0 | 4990 |

## Все 23 типа отношений

Final TP/FP/FN; значения 0/0/0 означают отсутствие и Gold support, и predictions. Zero-support types нельзя считать проверенной научной capability.

| Relation | Control | Union | Gold diagnostic |
|---|---|---|---|
| HAS_PROPERTY | 7/170/45 | 7/718/45 | 49/208/3 |
| HAS_PERFORMANCE | 4/64/48 | 4/229/48 | 44/86/8 |
| HAS_MORPHOLOGY | 1/3/16 | 1/12/16 | 14/68/3 |
| HAS_COMPONENT | 4/1325/53 | 10/12987/47 | 19/997/38 |
| HAS_MEMBER | 0/2069/44 | 0/15716/44 | 20/2197/24 |
| USES_MATERIAL | 5/24/18 | 7/121/16 | 17/61/6 |
| FUNCTIONS_AS | 0/36/3 | 0/203/3 | 3/87/0 |
| DETECTS | 3/12/10 | 3/12/10 | 9/18/4 |
| USED_FOR | 10/191/37 | 10/890/37 | 47/239/0 |
| FABRICATED_BY | 7/75/7 | 7/283/7 | 14/89/0 |
| IMPROVES | 1/5/6 | 1/5/6 | 7/36/0 |
| ACTS_ON | 2/86/11 | 2/205/11 | 10/105/3 |
| INTEGRATED_IN | 0/11/1 | 0/11/1 | 1/34/0 |
| HAS_MECHANISM | 0/9/6 | 0/57/6 | 3/26/3 |
| MECHANISM_INVOLVES | 0/1/9 | 0/1/9 | 6/10/3 |
| LOCATED_IN | 0/2/2 | 0/6/2 | 2/65/0 |
| OPTIMIZES | 0/14/1 | 0/34/1 | 1/37/0 |
| FACILITATES | 0/97/3 | 0/342/3 | 3/117/0 |
| CAUSES | 0/48/3 | 0/133/3 | 2/71/1 |
| EXPLAINS | 4/103/7 | 4/103/7 | 9/282/2 |
| CHANGES_PARAMETER | 0/0/8 | 0/0/8 | 1/12/7 |
| DEPENDS_ON | 0/110/2 | 0/110/2 | 2/145/0 |
| MEETS_REQUIREMENT | 0/0/2 | 0/0/2 | 0/0/2 |

## Abstentions и unsupported

| Вариант | Pair abstentions | Gold unsupported labels | RE context skips |
|---|---|---|---|
| GLiNER control | 1425/3942 = 0.3615 | {'MEETS_REQUIREMENT': 2} | 0 |
| MatSciBERT CHEM | 359/1110 = 0.3234 | {'MEETS_REQUIREMENT': 2} | 0 |
| Electrocatalyst NER* | 10751/16712 = 0.6433 | {'MEETS_REQUIREMENT': 2} | 0 |
| GLiNER + оба NER | 22578/41590 = 0.5429 | {'MEETS_REQUIREMENT': 2} | 0 |
| GOLD entities (diagnostic) | 2460/5532 = 0.4447 | {'MEETS_REQUIREMENT': 2} | 0 |

MEETS_REQUIREMENT не оценивается моделью и не переименовывается в другое отношение. Отсутствие unsupported proposals само по себе не означает поддержку всех 23 labels. Pair abstentions и число rejected raw proposals имеют разные denominators; один pair может иметь несколько raw labels/reasons. Предсказанные неподдержанные source-NER классы battery сосчитаны в `scientific_output_counts` manifests, без переноса в ontology.

## Стоимость реального CPU inference

Windows, Python 3.11.9, 8 physical / 16 logical CPU, host RAM около 31.3 GiB, четыре torch threads, float32, deterministic algorithms. Варианты запускались последовательно в fresh processes. Wall/CPU timer начинается до ML imports и включает load/hash checks, inference и serialization predictions. Final provenance hashing/manifest serialization, interpreter startup, downloads, tests, probes и evaluator исключены. Один shared-host measurement без доверительных интервалов и контроля file cache.

| Вариант | Wall s | CPU s | Peak GiB | NER inference wall s | RE wall s | NLI queries | NER/RE forwards |
|---|---|---|---|---|---|---|---|
| GLiNER control | 358.585 | 1322.156 | 1.684 | 9.603 | 321.116 | 6841 | 27/223 |
| MatSciBERT CHEM | 124.178 | 436.125 | 1.193 | 8.805 | 99.910 | 2248 | 66/76 |
| Electrocatalyst NER* | 1296.734 | 4952.438 | 1.382 | 8.252 | 1265.769 | 28212 | 32/890 |
| GLiNER + оба NER | 3198.087 | 12265.000 | 2.749 | 24.127 | 3128.936 | 70933 | 125/2227 |
| GOLD entities (diagnostic) | 417.645 | 1563.469 | 1.024 | 0.001 | 399.542 | 9226 | 0/298 |

Peak — Windows PeakWorkingSetSize за весь процесс с обоими компонентами, не отдельная NER peak memory. NER column — сумма wall timers predict(), без загрузки; в primary run NER CPU отдельно не инструментировалось. Gold column NER time — чтение fixed entity inventory, без NER encoder. Все peaks ниже 32 GiB. Полные load times, API calls, token limits и file hashes — в *_run.json. Время неудачных запусков не суммируется с основными вариантами. Дополнительный source-only NER cost probe повторил battery NER на всех 12 DEV: 790 entities, 16712 направленных pairs и 28060 type-compatible hypotheses до проверки linguistic negation. Он объясняет рост RE workload и не использует Gold. Этот короткий probe выполнялся во время battery RE; его нагрузка не вычтена из измеренного battery wall/CPU, что дополнительно ограничивает точность сравнения скорости. [Профиль](c_diag_002/battery_ner_cost_probe.json).

Отдельные NER-only fresh-process measurements по заранее записанному [cost supplement](C_DIAG_002_COST_PROTOCOL.md), после primary runs и до Gold scoring. RE и Gold не загружаются; все predicted inventories точно совпали с primary. Peak включает только NER, source loading и небольшой inventory check. Total times включают ML imports/load, predict times — только NER.predict().

| NER | Total wall s | Total CPU s | Predict wall s | Predict CPU s | Peak GiB |
|---|---|---|---|---|---|
| GLiNER control | 32.134 | 57.406 | 8.672 | 33.797 | 1.684 |
| MatSciBERT CHEM | 20.149 | 39.016 | 7.389 | 28.609 | 0.749 |
| Electrocatalyst NER* | 19.771 | 36.438 | 6.623 | 25.531 | 0.750 |
| GLiNER + оба NER | 46.709 | 116.141 | 23.121 | 89.438 | 2.001 |

## GLiREL: ограниченная проверка воспроизводимости

Повторён [официальный Doctor Who control](https://github.com/jackboyla/GLiREL#usage) через `GLiREL.from_pretrained(..., strict=True)` для `jackboyla/glirel-large-v0@40a523e12a8432d6da364cf2a195a28755ff04d3`, pinned runtime GLiREL 1.0.1 / Flair 0.14.0 / Transformers 4.51.3. Сохранены исходные tokens/NER/labels, threshold=0, top_k=1. Ожидались два characters с score около .99. Фактически получены followed by=0.002801, followed by=0.002741; characters отсутствует. Strict loading завершился, совпал SHA256 полного checkpoint. Локальная runtime config меняет лишь путь backbone; официальный loader получает канонический encoder export из того же checkpoint.

Wall 29.989 s, CPU 31.328 s, peak 2.404 GiB, budget 600 s. **FAIL; GLiREL остановлен.** Blocker: опубликованное поведение не воспроизведено, root cause не установлен. Это согласуется с предыдущей диагностикой нескольких runtimes; нельзя объявлять причиной конкретную библиотеку без доказательств. Нулевые DEV predictions из невалидированного runtime не включены в таблицу качества. Рабочего независимого standalone zero-shot RE comparator в этом ограниченном исследовании не получено; межмодельного сравнения RE на одинаковых entities нет. NLI на разных entities отвечает на вопрос о bottleneck, но не заменяет такой comparator.

REBEL исключён как генеративный, GLiNER-Relex как joint arm F; fixed-label TACRED/SciERC heads не предоставляют наши 23 класса. Это не доказательство отсутствия других пригодных checkpoints. Полный output/hashes: [glirel_probe.json](c_diag_002/glirel_probe.json).

## Реальные DEV примеры

**MatSciBERT CHEM / D004 / entity_TP**: `graphene` [243,251) → MATERIAL; Gold kinds на точной границе: ['MATERIAL']. BOUNDARY означает отсутствие exact Gold span, без самостоятельной semantic adjudication.

> hat the products corresponded to agglomerated structures of graphene and SnO 2 particles, with small secondary SnO x ( x ≤ 2) na

**MatSciBERT CHEM / D001 / entity_FP_KIND**: `NO 2` [755,759) → MATERIAL; Gold kinds на точной границе: ['ANALYTE']. BOUNDARY означает отсутствие exact Gold span, без самостоятельной semantic adjudication.

> bility (84 days), low theoretical detection limit (43.5 ppb NO 2 ), great mechanical deformation tolerance (3000 bending cyc

**MatSciBERT CHEM / D001 / entity_FP_BOUNDARY**: `graphene oxide` [516,530) → MATERIAL; Gold kinds на точной границе: нет. BOUNDARY означает отсутствие exact Gold span, без самостоятельной semantic adjudication.

> astic threads as flexible/stretchable templates and reduced graphene oxide/mesoporous zinc oxide nanosheets as sensing layers to form 

**Electrocatalyst NER* / D001 / entity_TP**: `room-temperature` [897,913) → OPERATING_CONDITION; Gold kinds на точной границе: ['OPERATING_CONDITION']. BOUNDARY означает отсутствие exact Gold span, без самостоятельной semantic adjudication.

>  cycles and 65% strain strength), and washing durability in room-temperature gas detection.

**Electrocatalyst NER* / D003 / entity_FP_KIND**: `NO` [711,713) → MATERIAL; Gold kinds на точной границе: ['ANALYTE']. BOUNDARY означает отсутствие exact Gold span, без самостоятельной semantic adjudication.

> ng parameters, exhibits a large response of 4.18‰ ppm −1 to NO and 6.66‰ ppm −1 to NO 2 , an ultralow detection limit of 8

**Electrocatalyst NER* / D001 / entity_FP_BOUNDARY**: `E` [0,1) → MATERIAL; Gold kinds на точной границе: нет. BOUNDARY означает отсутствие exact Gold span, без самостоятельной semantic adjudication.

> E-textiles are gaining growing popularity recently due to low

**GLiNER control / D003 / relation_TP**: `LIG electrode` (DEVICE) → HAS_MORPHOLOGY → `stretchable serpentine structure` (MORPHOLOGY). Score 0.9923.

> The design of a stretchable serpentine structure in the LIG electrode and strain isolation from the stiff island allows the gas sensor to be stretched by 30%.

**GLiNER control / D001 / relation_FP**: `E-textiles` (MATERIAL) → HAS_PROPERTY → `light weight` (PROPERTY). Score 0.9877.

> E-textiles are gaining growing popularity recently due to low cost, light weight, and conformable compatibility with clothes in wearable and portable smart electronics.

**GLiNER control / D001 / relation_FN_NER**: `E-textiles` (DEVICE) → HAS_PROPERTY → `low cost` (PROPERTY). Loss stage: NER.

> E-textiles are gaining growing popularity recently due to low cost, light weight, and conformable compatibility with clothes in wearable

**GLiNER + оба NER / D001 / relation_FN_NER**: `E-textiles` (DEVICE) → HAS_PROPERTY → `low cost` (PROPERTY). Loss stage: NER.

> E-textiles are gaining growing popularity recently due to low cost, light weight, and conformable compatibility with clothes in wearable

**GOLD entities (diagnostic) / D001 / relation_FP**: `E-textiles` (DEVICE) → INTEGRATED_IN → `clothes` (OTHER). Score 0.6084.

> E-textiles are gaining growing popularity recently due to low cost, light weight, and conformable compatibility with clothes in wearable and portable smart electronics.

**GOLD entities (diagnostic) / D001 / relation_FN_RE_OR_CAPABILITY**: `reduced graphene oxide/mesoporous zinc oxide nanosheets` (MATERIAL) → HAS_COMPONENT → `reduced graphene oxide` (MATERIAL). Loss stage: RE_OR_CAPABILITY.

> vailable cotton/elastic threads as flexible/stretchable templates and reduced graphene oxide/mesoporous zinc oxide nanosheets as sensing layers to form conducting fibers. The as-prepared fiber de

**GOLD entities (diagnostic) / D024 / relation_FN_GATES**: `create pores in graphene` (PROCESS) → CHANGES_PARAMETER → `selectivity` (PERFORMANCE). Loss stage: GATES.

> investigations, using either electron beams or bottom-up synthesis to create pores in graphene, suggest that it may be possible to employ such techniques to engineer variable-sized, graphene nanopores to tune selectivity and molecular diffusivity. Hence, we propose using porous graphene sh

Примеры выбраны детерминированно после predictions для иллюстрации; они не использованы для исправления prompts/thresholds. Более подробные examples и chunk IDs — в variant JSON.

## Проверки

`python -m pytest -q`: **47 passed, 1 deselected**. Исключён только сохранённый historical_training test с C-001 fitting; он не выполнялся. Проверены frozen ontology, DEV allowlist/hashes, Unicode/nested spans, directed pairs, scope, strict qualifiers, multiset matching, exact union, Gold entity-only projection, запрет закрытых путей и различие pair serving от фактической оценки relation label. [tests.json](c_diag_002/tests.json). Реальные model runs и NER-only benchmarks — отдельные integration measurements, не synthetic fixtures. Evaluator проверил prediction/config/source hashes и audits всех пяти runs, полное совпадение control с C-RESTART-001 и разбиение всех final FP по причинам. Все четыре NER-only inventories точно совпали с основными predictions. Исторические C-001/C-RESTART-001 отчёты, код zero-shot pipeline и ontology byte-for-byte сохранены; main остался на 06f2bac, c-restart-001-zero-shot — на b368a63.

## Ограничения и воспроизведение

12 повторно использованных DEV, один owner review и AI-origin annotations; независимого второго annotator и human semantic-support audit нет. D024 имеет incomplete relations, поэтому часть unmatched predictions может отражать неполноту annotation. Exact matching чувствителен к границам и contextual kinds. Mapping/schema mismatch и узкое покрытие двух scientific моделей не позволяют обобщить отрицательный результат на все scientific NER. Single-sentence contexts не проверяют coreference/cross-sentence RE. Не все 23 типа могут иметь Gold support на этих DEV. Qualifier attachment остаётся неполным. GLiREL runtime не валидирован, поэтому прямой scientific RE comparator отсутствует. B здесь не запускалась; переход к B не означает, что её превосходство уже доказано. Пересечение авторских pretraining/fine-tuning corpora с исходной литературой DEV независимо не проверялось.

[Команды и сведения о неудачной instrumentation-попытке](C_DIAG_002_REPRODUCE.md). Полные predictions/query traces/Gold/weights остаются локально под gitignore; публикуются aggregate metrics, immutable hashes, access audits и короткие DEV excerpts. [comparison.json](c_diag_002/comparison.json), [control](c_diag_002/control.json), [chem](c_diag_002/chem.json), [battery](c_diag_002/battery.json), [union](c_diag_002/union.json), [Gold diagnostic](c_diag_002/gold.json).
