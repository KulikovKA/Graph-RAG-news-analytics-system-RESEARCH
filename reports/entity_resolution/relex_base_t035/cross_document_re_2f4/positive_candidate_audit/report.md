# Stage 2F.4 — Positive Candidate Discovery & Scientific Audit

Дата: 2026-11-12. Ветка: `design-v2-sequential-kg`. Исходный commit: `3c19306` (`research: compare A B C cross-document ReLEx prompts on bridge entities`). Рабочее дерево перед исследованием было чистым.

## Результат и его научный смысл

Найдены **6 содержательно мотивированных междокументных пар: 4 STRONG_CANDIDATE и 2 PLAUSIBLE_CANDIDATE**, охватывающие пять пар документов и девять документов. Ориентир 10–20 не достигнут: слабые случаи намеренно не включены. Дополнительно проверены десять новых альтернатив (9 WEAK, 1 REJECTED) и сохранены 12 контролей из прежних 68 bridge-пар.

Четыре STRONG — кандидаты на **типизированные связи между научными понятиями**. Они не являются четырьмя новыми эмпирическими открытиями. Два PLAUSIBLE — **условные гипотезы применимости**, не утверждения о реально проведённых вмешательствах или вычислениях. Новых доказанных эмпирических cross-document отношений найдено **0**. Положительный candidate set нельзя считать gold benchmark фактических отношений.

Четыре пары соединяли бы разные baseline WCC. У всех шести отсутствуют прямой SIMILAR_TO, общий концепт и запись как пары в прежнем candidate inventory из 682 пар и bridge pool из 128 пар. Поэтому прежний результат 0 plausible predictions не исчерпывает научные пересечения корпуса. Одновременно состав из 30 разнородных абстрактов существенно ограничивает ожидаемый выход новых конкретных фактов.

## Состав артефактов и определения

| Файл | Содержимое |
|---|---|
| `corpus_overlap_report.md` | Все 30 документов: дисциплина по содержанию, объекты, процессы, методы, измерения, результаты и карта всех 435 пар документов |
| `positive_candidates.csv` | Только P01–P06: 4 STRONG и 2 PLAUSIBLE; направление, уровни связи, IDs, контексты, evidence refs, ограничения и baseline-покрытие |
| `negative_controls.csv` | Только N01–N12 из прежнего эксперимента; сохранены исходные bridgepair IDs и конкретные ошибочные claims |
| `candidate_evidence.jsonl` | 28 записей: 6 positive_candidate, 10 reviewed_nonpositive W01–W10, 12 negative_control; полные native/canonical identifiers, source mentions, offsets, revisions, абстракты и factual-окрестности |
| `report.md` | Выводы, содержательный аудит, причины пропусков, ограничения и предложение следующего опыта |

CSV — UTF-8, comma delimiter, одна строка на пару, JSON-массивы внутри quoted cells. JSONL связывается с CSV по `audit_id`. E2/E3 — endpoints этой конкретной пары, без требования существующего E1–E2–E3–E4 path и без переориентации по прежним модельным scores. IDs не изменялись. `E2_canonical_entity_id`/`E3_canonical_entity_id` — представители Stage 2F.3; `*_source_canonical_ids` — все исходные canonical IDs до этого объединения; native `e:` и `m:` лежат в JSONL. `source_canonical_ids` не смешиваются с native IDs.

`auditpass:<document_id>:<text_sha256>:<start>:<end>` — локальный locator точного исходного passage, введённый только для аудита. Это **не** новое model-extracted evidence, triple или KG-ребро. Native factual refs сохраняют исходные `ct:`, `t:` и `w:` IDs. Источник каждого полного абстракта содержит source URL, DOI, revision и source_reference. В JSONL `local_factual_neighborhood` — все существующие incident RELEX_REL обоих endpoints с provenance. Они сохраняются как model-extracted claims, а не становятся научно подтверждёнными через включение в аудит.

Значение STRONG означает убедительную совместимость терминов и гипотезы о конкретном отношении в указанном **scope**. Ни STRONG, ни PLAUSIBLE не означает established empirical fact. `direct_cross_document_empirical_support=False` у всех записей задан намеренно. Общая категория, сходство, identity, типизированный concept relation, multi-hop chain и research hypothesis различаются через `relation_category`, формулировку predicate и ограничения; общего RELATED_TO вместо них нет.

## Содержательный аудит шести кандидатов

### P01 — ESRD → CKD: отношение стадии

E2: `end-stage renal disease`, W2761455282 (D09). E3: `chronic kidney disease`, W3123056951 (D16). Вердикт: **STRONG_CANDIDATE**, `IS_ADVANCED_STAGE_OF`, уровень понятий, разные стадии не тождественны.

D09 описывает ESRD у пациентов на диализе, которым выполнена TKA. D16 начинает с применения rHuEPO при low hemoglobin, в том числе related to CKD. Клиническая классификация связывает эти понятия: CKD может прогрессировать до kidney failure/ESKD. Этот таксономический шаг проверен по [CDC: Chronic Kidney Disease Basics](https://www.cdc.gov/kidney-disease/about/index.html). Он явно внешнее знание: два абстракта сами не определяют ESRD через CKD.

Локальные `ct:d3e83c5f4cd3a059007ff6a6cb2eea95263e4ccf23f0250e9d6e5f3c72885b88` и `ct:ee68917c5caeecd12bca05d7341e59e989078346a4da00fd6470e2e34fb53d19` дают контексты ESRD/complication rate и rHuEPO/CKD соответственно, но не доказывают междокументное отношение. Нельзя переносить частоту осложнений TKA, PRCA или HLA-статус между когортами. Полезность для Graph-RAG — поиск по конкретной стадии почечной болезни с сохранением ограничений результатов.

### P02 — rHuEPO → подгруппа пациентов на диализе

E2: `Recombinant human erythropoietin`, W3123056951 (D16). E3: `dialysis patients`, W2761455282 (D09). Вердикт: **PLAUSIBLE_CANDIDATE**, `POTENTIALLY_APPLICABLE_TO_SUBGROUP`, исследовательская гипотеза.

Через P01 появляется осмысленный вопрос о релевантности rHuEPO диализным пациентам при соответствующем низком Hb/анемии. D09 сообщает только ESRD, диализ, TKA и исходы. В нём нет Hb, анемии, экспозиции rHuEPO, дозы, HLA или PRCA. Поэтому утверждение «эти 21 пациент получали rHuEPO» не поддержано. Ещё менее поддержаны лечение этих пациентов новыми мутантами или объяснение их осложнений иммуногенностью препарата.

Контекст D16 непосредственно описывает drug/CKD, D09 — TKA/dialysis population. Существующий `ct:31942044390e6008f2dc7077d01c3c8ef177a3eb36350824d07ab918eef3eb12` (`primary TKA APPLIED_TO dialysis patients`) подтверждает только локальную тему. Для дальнейшего вывода нужны клинические сведения о подгруппе. Это полезная условная поисковая цепочка, не рекомендация пациенту и не факт о лечении когорты.

### P03 — T cell activity → T-cells

E2: `T cell activity`, W2791492120 (D08). E3: `T-cells`, W3123056951 (D16). Вердикт: **STRONG_CANDIDATE**, `ACTIVITY_OF_CELL_CLASS`, процесс/клеточный класс.

D08 прямо говорит о подавлении T-cell activity регуляторными B-клетками. D16 описывает распознавание EPO epitopes T-клетками в контексте иммуногенности. Активность и её клеточный участник согласованы по смыслу; это конкретнее общей рубрики immunology. При этом клетки разных исследований не являются одной популяцией, а разные антигенные контексты не тождественны.

У обоих endpoints нет incident RELEX_REL в baseline. Основание — точные source passages, а не выдуманный factual flank. Запрещённый переход: Breg suppression в D08 -> снижение anti-rHuEPO response в D16. Для него данных нет. В Graph-RAG допустим переход к двум контекстам клеточной активности с явным предупреждением о недоказанном переносе механизма.

### P04 — overexpression → gene expression

E2: `overexpression`, W3138328568 (D19). E3: `gene expression`, W2076293586 (D18). Вердикт: **STRONG_CANDIDATE**, `IS_SPECIALIZED_EXPRESSION_PROCESS`, процесс и специальный режим.

D19 недвусмысленно описывает coexpression/overexpression генов дрожжей и изменения limonene tolerance/production. D18 анализирует expression при mouse CNV. Связь специального режима экспрессии с общим биологическим процессом содержательна даже при разных видах и генах. Это не идентичность событий экспрессии.

Критичное ограничение D18: простая корреляция gene dosage и gene expression найдена не во всех случаях. Существующий `ct:f6e85f2e378703a524c0a02e738150952a94dfb23aa3de192fd217c015bc2c0e` (`gene expression ASSOCIATED_WITH gene dosage`) теряет это ограничение, если читать только label. Источник имеет приоритет. D19 не сообщает, что overexpression достигнута через увеличение copy number. Именно поэтому отдельный W01 dosage → overexpression остался WEAK. Полезность P04 — сравнение наблюдательного и инженерного изучения expression, без универсального причинного вывода.

### P05 — coherent fiber optic bundle → optical fibers

E2: `coherent fiber optic bundle`, W2084709926 (D11). E3: `optical fibers`, W3081936674 (D15). Вердикт: **STRONG_CANDIDATE**, `HAS_COMPONENT_TYPE`, прибор/тип компонента.

D11 передаёт изображение divertor через волоконный пучок, D15 моделирует optical pulses in optical fibers. Компонентная связь к типу optical fiber осмысленна. Для проверки значения bundle использован пример конструкции из individual fibers в первичной работе [Parker et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12549214/). Эта внешняя статья не добавлена к корпусу и не идентифицирует конфигурацию NSTX.

Нельзя трактовать E3 как те же физические волокна, которые стоят в NSTX, или утверждать soliton regime в пучке. Для переноса уравнения D15 нужны материал, геометрия, wavelength, dispersion и nonlinear operating conditions. У bundle нет incident RELEX_REL; у E3 есть `ct:38efbf9b92359aebd875e5098c029aeb268280e4a320260ae0d667e89703a636` (`optical pulses LOCATED_IN optical fibers`). Он не подтверждает applicability к NSTX. Полезность — переход между приборной категорией и теорией распространения, а не готовая физическая модель установки.

### P06 — OB-NLM → mammographic images

E2: `blockwise non-local means`, W4230116451 (D02). E3: `mammographic images`, W1979727947 (D07). Вердикт: **PLAUSIBLE_CANDIDATE**, `CANDIDATE_PREPROCESSING_METHOD_FOR`, методический перенос.

D02 реализует GPU OB-NLM для MRI denoising и оценивает время. D07 извлекает mammographic image features для MD и риска. Возможный конкретный эксперимент — проверить NLM preprocessing перед оценкой MD. Применимость семейства NLM к mammogram denoising встречается в первичном исследовании [Rajaguru and Chakravarthy, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7294012/). Оно использует другой вариант фильтра и не доказывает пригодность конкретной MRI/GPU реализации или пользу для MD.

Локальные `ct:2ec3cc4dbe1b26100264c6949be25bb07a16a5e6dfb2cc65cc392ca74c989e07` (`GPU parallel algorithm DERIVED_FROM blockwise non-local means`) и `ct:684dae1a456ee636f8aa0d2fff520d239b771a0f5f721b8d8281eb9a6651de41` (`automated measure DERIVED_FROM mammographic images`) согласуются с тематикой источников. Вместе они не означают, что D07 использовал этот алгоритм. Нужны проверки noise model, preservation of density/texture, Cumulus correlation, risk calibration и execution time. Дальнейшая задача проверяема; ожидаемый выигрыш остаётся неизвестным.

## Почему остальные новые предложения не стали положительными

| ID | Пара | Вердикт и конкретная причина |
|---|---|---|
| W01 | mouse gene dosage / yeast overexpression | WEAK: dosage-expression не универсальна в D18, механизм copy-number engineering не задан в D19 |
| W02 | supply-and-demand theorem / fares | WEAK: есть экономическая идея price-demand, но нет одинаковой формальной модели или конкретного переносимого результата |
| W03 | embedded Pt nanoparticles / Fe-Co-Zr grains | WEAK: morphology-property аналогия не задаёт общий материал и механизм |
| W04 | lightning protection / CIED malfunctions | REJECTED: transient over-voltage/SPD и ionizing radiation являются разными воздействиями; эффективность защиты не переносится |
| W05 | ARPN / mammographic images | WEAK: tiny-person detection и density/risk estimation — разные задачи; shared image analysis недостаточно |
| W06 | inbred mouse models / Cx43 knockout mice | WEAK: общий организм, разные генотипы, вмешательства и фенотипы |
| W07 | psychiatric diseases / relational wounds | WEAK: отсутствуют общий диагноз, эксперимент и причинный механизм между мышиной генетикой и концептуальным IFS |
| W08 | CIO organizing drives / unions | WEAK: labor-history category без общего именованного события, архива или интервью; D28 усечён |
| W09 | synod on synodality / papal politics | WEAK: общая церковная среда без связи между современным текстом и материалом 1605–1700 |
| W10 | nanostructure fabrication / protein computational design | WEAK: engineering/design обозначает разные операции, методы и объекты |

Эти случаи не отклонялись только из-за отсутствия общего предложения или cross-citation. Проверялись значения терминов, методы, объекты, направления и недостающие условия. P02/P06 показывают, что обоснованная гипотеза может сохраниться без прямого общего evidence passage.

## Отрицательные контроли

N01 supply line/electric supply line и N02 grains/grain проверяют омонимию. N03 resolution/resolution, N04 thicknesses/thickness и N12 energy consumption/energy consumption — различение общей характеристики и конкретного измерения/влияния. N05/N06 results — независимые локальные результаты. N07 study — разные методы protein design и yeast engineering. N08 cooperation — отсутствие причинной связи банков и перевозчиков. N09 patients — независимые когорты. N10 reliability — историческая достоверность и техническая надёжность. N11 mouse gene dosage/HLA-DRB1*09 — неподтверждённое генетическое сопоставление.

Из 12 контролей семь REJECTED и пять WEAK. Набор включает прежние ложноположительные случаи и случаи, по которым модель воздержалась, с сохранением исходных pair IDs. Контроли выбраны по значениям и исходным текстам; прежние prediction IDs не используются как доказательство ошибки. Они проверяют конкретные `invalid_claims`, а не универсальное отсутствие любых отношений. Например, patients могут принадлежать общей человеческой категории, две energy consumption — общей физической величине, а две cooperation — общему виду деятельности. Разумная общая абстракция не должна ошибочно считаться ложным фактическим предсказанием, как и не должна засчитываться новым фактом.

## Ответы на восемь вопросов

1. **Сколько содержательных пар?** Шесть candidate-пар, пять документных пар, девять документов. Из десяти дополнительно рассмотренных предложений девять WEAK, одно REJECTED. Новых доказанных эмпирических отношений — ноль.
2. **STRONG/PLAUSIBLE?** 4/2. STRONG имеют scope понятия/типа; PLAUSIBLE требуют новых наблюдений или проверки переносимости.
3. **Какие отношения?** Disease-stage, activity-to-cell-class, expression-process specialization, bundle-to-component-type; условная drug relevance к подгруппе и перенос denoising метода. Identity и причинный перенос результатов не установлены.
4. **Межкомпонентные пары?** Четыре: P03, P04, P05, P06. Среди STRONG — три, среди PLAUSIBLE — одна. Это потенциальные endpoint bridges; новый граф не создавался и никакие рёбра не добавлялись.
5. **Уже связанные через SIMILAR_TO?** Прямых SIMILAR_TO нет у всех шести; общих concepts также нет. P01/P02 при этом уже находятся внутри крупнейшей компоненты. Отсутствие прямого ребра не означает отсутствия любого более длинного пути.
6. **Почему прежний поиск пропустил пары?** Все шесть отсутствуют даже в 682 candidate-парах и 128 bridge-парах, то есть причина не только позднее ранжирование. A/B источников прежнего discovery не было без прямого similarity/concept; C требовал существующий semantic/concept bridge и factual flank с обеих сторон. Система D искала только разные WCC, одинаковую broad discipline, общие name/context tokens, compatible extracted types и factual neighbors. P01/P02 находятся в одной WCC и в разных source disciplines Medicine/Biology; P03 также междисциплинарен и имеет два endpoints без factual neighbors; у P05 bundle без factual neighbors; P04 не имеет общего точного name token (`overexpression` против `gene expression`); P06 междисциплинарен и не имеет общего name token. В A/B/C comparison использовались intermediate endpoints существующих paths; новые смысловые соответствия там не искались. Для P03/P05 отсутствие factual flank является прямой структурной причиной пропуска, а не низким score ReLEx.
7. **Повторять A/B/C?** Полный повтор как оценку качества factual extraction сейчас не обоснован. Небольшой диагностический опыт на новом наборе может быть полезен при явном разделении concept relations, hypotheticals и invalid claims. Конкретный протокол ниже. В этом аудите он не запускался.
8. **Достаточен ли корпус?** Для демонстрации потерь candidate discovery и проверки нескольких concept-level связей — да. Для устойчивого сравнения factual cross-document extraction, recall или выбора рабочего prompt/threshold — нет. Нужен более плотный тематический корпус с перекрывающимися объектами/методами, совместимыми условиями и независимо проверенными отношениями. Здесь преобладают несопоставимые узкие задачи и ограниченные абстракты.

## Предложение следующего опыта, без запуска

После решения пользователя возможен небольшой фиксированный диагностический набор **6 новых пар + 12 контролей**, без искусственного дополнения до 60–100. Текущие IDs, revisions и тексты уже сохранены для подготовки будущих contexts. Это ещё не готовые tokenizer-checked model inputs.

Перед inference следует утвердить target scope и допустимые mappings к существующим A/B/C labels. P01/P04 допускают class-level IS_A в A/C при сохранении stage/specialization смысла. P05 допускает HAS_PART только как отношение bundle к **типу** компонента, не к конкретным волокнам D15. P03 ACTIVITY_OF не имеет точного label в текущей онтологии: это отдельный тест ontology gap, а не гарантированный gold positive для HAS_ATTRIBUTE. Labels B не выражают многие из этих направленных отношений точно; совпадение общего CONCEPTUALLY_RELATED_TO с сильной типизированной связью нужно учитывать отдельно. Настройки и prompts в текущем аудите не менялись.

P02/P06 оценивать как гипотезы: не требовать безусловного APPLIED_TO как gold fact. Для них отдельно фиксировать корректные условия, отсутствие заявления о фактическом использовании и разумное воздержание. Контроли оценивать по содержанию и `invalid_claims`; valid common-category связь не является causal/factual success. Все пары и оценки зафиксировать до forward, использовать один и тот же context для A/B/C, затем вручную проверить выходы по первоисточникам. Исследовать влияние локальных summaries можно отдельным последующим опытом, чтобы не смешивать два изменения сразу.

Отчёт такого небольшого опыта должен показывать все пары и ошибки по категориям. Он не обоснует статистическое превосходство модели, полный recall или рабочий threshold. Если цель следующего этапа — именно **новые эмпирические факты**, сначала целесообразно собрать более плотный тематический набор и gold-разметку, а не запускать очередное широкое A/B/C на этих 30 абстрактах.

## Проверки и границы выполненной работы

Из локальных артефактов заново восстановлены 811 contextual entities + 32 concepts = 843 KG vertices, 1307 factual edges, 492 semantic links и 118 concept links. Связность посчитана как weak connectivity по этим типам рёбер, включая изолированные vertices и self-loops. Документы и MENTIONS исключены. Получено 223 WCC, крупнейшая 369 vertices / 17 documents, в точности как Stage 2F.3.

Проверены: SHA-256 текстов всех 30 документов и revision IDs; соответствие mention/context slices всех 811 profiles; mapping всех 905 source canonical memberships; offsets 993 native entities и 1101 mentions, включая их revisions; все 1307 factual IDs, source triple IDs и remapped endpoints против lossless preview. Проверены отсутствие дубликатов и внутридокументных пар в 28 записях, ссылки CSV ↔ JSONL и наличие всех controls в прежней 68-парной выборке. Расчёт компонент не использовал связи Document–MENTIONS, которые искусственно объединили бы все сущности документа.

Чтение source_factual_evidence не равнозначно научному подтверждению каждого model-extracted label. Примеры ограничений: dosage-expression требует сохранения отрицательной оговорки D18; `airlines PERFORMS fares` не является осмысленным доказательством W02; `TKA DEPENDS_ON ESRD` не используется как необходимая причинная связь. Исходные сомнительные отношения сохранены.

Работа выполнена offline: существующий Neo4j namespace не открывался, новый live snapshot базы не снимался. Baseline — импортированный Stage 2F.3, восстанавливаемый из versioned artifacts, а не независимая проверка текущего состояния сервера. NER, GLiNER ReLEx и embedding inference не запускались; model/prompt/ER/source documents не менялись; SIMILAR_TO сохранены. Stage 2F.5 не начинался. Добавлены только пять файлов этого исследовательского аудита. Локальные одноразовые скрипты сериализации и проверок не являются частью pipeline и не входят в коммит.

Оценки выполнены одним рецензентом по абстрактам. Независимого expert adjudication и исчерпывающего gold по всем entity-парам нет. Внешние источники использованы только там, где это явно указано для P01/P05/P06; они не пополняют 30-документный корпус и не превращают гипотезу в факт. Отсутствие доказанных новых фактов при таком корпусе не диагностируется как программная ошибка.

## Машинно проверяемая сводка

Следующие таблицы фиксируют baseline-memberships, размеры покрытия и контрольные хеши использованных входов. Нумерация `wcc:000` и далее: компоненты отсортированы по убыванию размера, при равенстве — по минимальному vertex ID. Исходные source artifacts и все их IDs остаются авторитетными; audit IDs локальны этому каталогу.

<!-- GENERATED AUDIT SUMMARY -->

| Кандидат | Оценка | E2 WCC / размер | E3 WCC / размер | Разные WCC | SIMILAR_TO / общий concept | Старый inventory / bridge pool |
|---|---|---|---|---|---|---|
| P01 | STRONG_CANDIDATE | wcc:000 / 369 | wcc:000 / 369 | False | 0 / 0 | отсутствует / отсутствует |
| P02 | PLAUSIBLE_CANDIDATE | wcc:000 / 369 | wcc:000 / 369 | False | 0 / 0 | отсутствует / отсутствует |
| P03 | STRONG_CANDIDATE | wcc:119 / 1 | wcc:159 / 1 | True | 0 / 0 | отсутствует / отсутствует |
| P04 | STRONG_CANDIDATE | wcc:000 / 369 | wcc:007 / 17 | True | 0 / 0 | отсутствует / отсутствует |
| P05 | STRONG_CANDIDATE | wcc:047 / 1 | wcc:034 / 2 | True | 0 / 0 | отсутствует / отсутствует |
| P06 | PLAUSIBLE_CANDIDATE | wcc:032 / 2 | wcc:000 / 369 | True | 0 / 0 | отсутствует / отсутствует |

Размеры относятся к vertices, не документам. P03 связывал бы два изолированных endpoints; P05 имеет изолированный bundle. Наличие этих сущностей проверено по native offsets, а отсутствие factual edges не использовалось как основание научного отклонения.

| Проверка | Результат |
|---|---:|
| Исходные абстракты / корректные SHA-256 и revisions | 30 / 30 |
| Профили / корректные target и context offsets | 811 / 811 |
| Исходные canonical memberships | 905 |
| Native entity IDs / source mention IDs | 993 / 1101 |
| Фактические IDs / совпадение mapped endpoints, типов и source triples | 1307 / 1307 |
| SIMILAR_TO / HAS_SHARED_CONCEPT / concepts | 492 / 118 / 32 |
| KG vertices / WCC / largest / largest documents | 843 / 223 / 369 / 17 |
| Исходные evidence paths, учтённые при проверке прежнего покрытия | 3236 |
| Сохранённые старые contexts: target spans и source fragments проверены offline | 68 / 68 |
| Новые positive / nonpositive / старые controls | 6 / 10 / 12 |
| Positive STRONG / PLAUSIBLE | 4 / 2 |
| Controls REJECTED / WEAK | 7 / 5 |
| Positive cross-WCC / direct SIMILAR_TO / shared concept | 4 / 0 / 0 |
| Positive в старом inventory 682 / pool 128 / sample 68 | 0 / 0 / 0 |
| Новые модельные forwards / embedding calls / Neo4j calls | 0 / 0 / 0 |

### Хеши исходных артефактов

Пути ниже заданы относительно корня репозитория. Это версии, на которых основаны identifiers, контексты, компоненты и выводы. Полная история последних этапов проверена через `git log`: `5143288` (2F.2 proposal), `e9f1994` (2F.3 lossless import), `2e7c740` (2F.4 evidence-grounded audit), `3c19306` (A/B/C).

| Вход | SHA-256 |
|---|---|
| `reports/ner_v2/research_run2/input_abstracts.jsonl` | `684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/entity_profiles.jsonl` | `9516bc6306b7c3b0aa8cd2d2b80ee4f388a011591df7b8657c328467d2e65158` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/source_factual_evidence.jsonl` | `325238221430cdbfda2aeedb5166b23fb23e71cf00bef1b5790e0e0b10cecd47` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/candidate_pairs.jsonl` | `d4ae6ecd8cde258b2934288839459e2d01af910c7651a6e02c943ee5d4e160c4` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/all_evidence_paths.jsonl` | `ff1cdd6748fb38855ee41b8d6ec23ec3561643331021214e594e90ffc1f5fc28` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/candidate_report.md` | `a6a37bdf837af652d2918f3f7796b1d1f336467eefe8e635a9d8ccd2b06e9175` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/prompt_comparison/comparison_report.md` | `a21ac294d9c95b7411d66bbb73701bdddf8b1d926b337df2ea8bef12cfc3a3aa` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/prompt_comparison/findings.md` | `eb2d1e2f49003a0307f59d2e4ef0c97a730177068048a9e7667e2fbc274c9735` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/prompt_comparison/bridge_pool.jsonl` | `3ebb8b3b061f0ce936e9619db182ff169fafcf3f2d7204eba6f9aeda586dd7c3` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/prompt_comparison/selected_bridge_pairs.jsonl` | `e95181327e958e822ad2f140cf4714249cee1f682782624d69fb8922668072cc` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/prompt_comparison/quality_review.csv` | `f9169cb1c7e17bbeeec65da2e388594fd3a5b352bd1ecfc0985e7c809aa80fe6` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/prompt_comparison/prepared_contexts.jsonl` | `4a449c1700e126ae398e7625ff6e353a8ce3706bcd6e8020861ae6f3cec5778b` |
| `reports/entity_resolution/relex_base_t035/cross_document_re_2f4/prompt_comparison/configurations.json` | `8f41ebfda73540af3bef24c9176af5269e2d5f58e2a895aa9028cfefd49b5c1e` |
| `reports/entity_resolution/relex_base_t035/global_kg_2f3/manifest.json` | `b0aaf7c7dcfbe199f5034fc3ee4abefc849ba6d3efc14afa130ac09ea41c8b4b` |
| `reports/entity_resolution/relex_base_t035/global_kg_2f3/canonical_mapping.jsonl` | `1c3f825e522279ceee8b054d3cfa734a4072872c7dddf4cf8f01eedbee68a714` |
| `reports/entity_resolution/relex_base_t035/global_entity_resolution/proposed_mapping.jsonl` | `50cf8728892d65f95ad23ee8be824702570e8314701a966a9d786ec817d1dd48` |
| `reports/entity_resolution/relex_base_t035/global_entity_resolution/preview_factual_relations.jsonl` | `032c86d6db994cf779c029de6cc0f560266133178ae50c2cfa44fdb2aaaed2bd` |
| `reports/entity_resolution/relex_base_t035/global_entity_resolution/preview_semantic_links.jsonl` | `f1363c52cc37a59d8ef4adbf4ea86dbf3d3487012f3290ec1f7ba4842ad6597c` |
| `reports/entity_resolution/relex_base_t035/global_entity_resolution/preview_concept_links.jsonl` | `7f0e9ccf9b69869ced203f2ebce2b9b0b85d0312ca8b2b22561a67d6ead0b930` |
| `scripts/analyze_cross_document_re_2f4.py` | `ed1da840786f0191a04648eaecc82bd79f49ca1895a449374eebe409e5fc47f5` |
| `scripts/compare_cross_document_prompts.py` | `e5d3d31b1f7d32dc8caed8324b60d21125ea476a06c54759e1d840426ae70c2a` |
