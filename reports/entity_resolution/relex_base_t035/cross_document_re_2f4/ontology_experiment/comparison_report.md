# Stage 2F.4 — Cross-Document Ontology Experiment A/B/C/D

Дата: 2026-11-12. Source HEAD `97b8196`, ветка `design-v2-sequential-kg`. В этом опыте разработана одна новая онтология D, выполнен весь pilot/main/sanity protocol. E не создавалась.

## Результат

**Содержательной междокументной связи выше 0.35 не распознано ни одной конфигурацией.** D содержит 9 типизированных направленных отношений и повышает проектное точное ontology coverage шести P-смыслов с A=2, B=0, C=2 до D=6. Однако на P01–P06 у всех A/B/C/D нет ни одного выхода >0.35. Это development-set diagnostic, не независимый gold и не полный recall корпуса.

| Конфигурация | Labels | Предсказания >0.35 | Пары с выходом / 28 | Supported / plausible / uncertain / rejected | Exact P hits / covered | Ошибочные N-пары / 12 |
|---|---:|---:|---:|---|---|---:|
| A | 20 | 2 | 1 | 0 / 0 / 0 / 2 | 0 / 2 | 1 |
| B | 9 | 2 | 1 | 0 / 0 / 0 / 2 | 0 / 0 | 1 |
| C | 20 | 7 | 3 | 0 / 0 / 0 / 7 | 0 / 2 | 3 |
| D | 9 | 0 | 0 | 0 / 0 / 0 / 0 | 0 / 6 | 0 |

D не дала ложноположительных выходов на N/W, но полностью воздержалась и на P. Это уменьшение ложных срабатываний при нулевом candidate yield, а не доказанная более высокая precision: для пустого выхода precision не определена. Новых эмпирических фактов и новых потенциальных межкомпонентных мостов на заданных порогах нет. Никаких рёбер в Neo4j не добавлялось.

## Что именно разработано

В `cross_document_ontology.json` и `ontology_design.md` зафиксированы definition, head/tail roles, direction, positive/boundary examples и evidence requirements для девяти codes. В модель передавались только короткие prompts:

| Code | Точный английский prompt |
|---|---|
| SUBTYPE_OF | is a type of |
| STAGE_OF | is a stage of |
| PART_OF | is part of |
| HAS_PARTICIPANT | has participant |
| HAS_PROPERTY | has property |
| USED_FOR | is used for |
| APPLIED_TO | is applied to |
| POTENTIALLY_APPLICABLE_TO | may be applied to |
| DERIVED_FROM | is derived from |

Различаются subtype и stage, component и participant, property и outcome, function и actual application, actual и potential applicability, derivation и causal influence. Все D-relations направленные. PART_OF является inverse старого HAS_PART. Симметричные untyped association/similarity, causal и автоматические dependency links из D исключены. Это проектное ограничение смысла, не автоматический postfilter выходов модели. Исходная intra-document ontology не менялась.

Дизайн опирается на разграничение hierarchy/association в [SKOS Reference](https://www.w3.org/TR/skos-reference/#semantic-relations), на process/participant/part distinctions в [OBO RO](https://oborel.github.io/obo-relations/process-relations/) и на научную типизацию relations/coreference в [SciERC/SciIE](https://aclanthology.org/D18-1360/). Наши class-level shortcuts не объявляются эквивалентными формальным RO properties. [Model card ReLEx](https://huggingface.co/knowledgator/gliner-relex-base-v1.0) описывает текстовые relation labels; определения в отчёте не добавлялись в model prompt.

D не состоит из шести отдельных answer labels: property, function, provenance, participation, structure, subtype и stage применимы к научным областям всего корпуса. Potential applicability охватывает методы, инструменты и вмешательства с явно указанными условиями. E без самостоятельного научного вопроса повторила бы broad semantic B и потому не запускалась.

## Постановка и контроль эксперимента

Основной набор: ровно 28 уникальных unordered pairs — P01–P06, N01–N12, W01–W10. Дубликатов нет, исходная идентичность записей и native/canonical provenance сохранены. Sanity set из 6 intra-document pairs отделён в `sanity_pairs.jsonl` и `sanity_predictions.jsonl`; он не входит в основную пороговую таблицу или виртуальную междокументную связность.

Модель одна: локальный `knowledgator/gliner-relex-base-v1.0`, CPU, 2 threads, seed=0, deterministic algorithms, eval/inference mode, offline checkpoint, token limit 512. A/B/C exact labels/order взяты из прежнего configurations.json без изменений, D frozen до первого forward. Контексты и fixed spans общие для всех четырёх конфигураций, выбран общий radius по самому длинному prompt. Реальные максимумы tokens: {"A": 427, "B": 409, "C": 439, "D": 411}. Проверялись и tokenizer, и actual collator; relation head получил ожидаемые embeddings и ordered label mapping.

Во входе только точные source fragments и нейтральные `[DOCUMENT ...]` / `[END DOCUMENT ...]` boundaries. Исходные source contexts каждого endpoint и полные IDs лежат в metadata. Локальные 1307 model-extracted RELEX_REL доступны по refs, но summaries не вставлены, чтобы неверный локальный label не подсказывал cross-document ответ. Expected labels, статусы STRONG/PLAUSIBLE/REJECTED, внешние определения и оценки аудитора в model text не вставлялись. Новые исходные документы не создавались.

Для каждой directed pair сохранён полный вектор sigmoid scores по всем labels, включая значения <0.35 (`raw_scores`), и отдельный positive inventory >0.35 (`relations`). Это scores, не logits и не калиброванные вероятности научной корректности. Ровно 0.35 среди raw scores не встретилось; поэтому ручной inventory >0.35 полностью покрывает threshold >=0.35.

Пилот: P01/P03/P06/N01/N07/W04 во всех конфигурациях, затем sanity set. Все 30 положительных pilot/sanity выходов проверены до main; labels/contexts не менялись. Технических ошибок не было, нулевой выход D не использовался как stop gate. Main дополнил checkpoint только оставшимися 22 парами на конфигурацию. Итого 139 forwards: 112 main (включая 24 pilot), 24 sanity, 3 compatibility. Threshold forwards=0. JSONL fsync после каждой пары, negative rows сохранены, signature-checked resume и восстановление оборванного tail реализованы.

Совместимость: исторический adapter из `2e7c740`, текущий default и explicit A дали полностью идентичные raw scores на всех 40 directed/label scores P01, default audit также идентичен. Адаптер не изменялся. Выполнены 39 существующих тестов адаптера/экспериментов и 3 новых теста: неполный checkpoint tail, rejection чужой signature/duplicate, одинаковые поверхности mentions из разных документов. Все 42 прошли.

## Ontology coverage отдельно от результатов модели

Покрытие зафиксировано до inference в `ontology_coverage.csv`. EXACT — сохранён конкретный смысл и направление; APPROXIMATE — существенный аспект потерян. Здесь NONE не понадобился: у B всегда есть broad association, но она не является точным выражением stage/participant/conditional application. APPROXIMATE не засчитывается exact hit.

| P-пара | A | B | C | D |
|---|---|---|---|---|
| P01 | APPROXIMATE / IS_A / E2_TO_E3 | APPROXIMATE / CONCEPTUALLY_RELATED_TO / EITHER | APPROXIMATE / IS_A / E2_TO_E3 | EXACT / STAGE_OF / E2_TO_E3 |
| P02 | APPROXIMATE / APPLIED_TO / E2_TO_E3 | APPROXIMATE / RELATED_TO / EITHER | APPROXIMATE / APPLIED_TO / E2_TO_E3 | EXACT / POTENTIALLY_APPLICABLE_TO / E2_TO_E3 |
| P03 | APPROXIMATE / PERFORMS / E3_TO_E2 | APPROXIMATE / CONCEPTUALLY_RELATED_TO / EITHER | APPROXIMATE / CONCEPTUALLY_RELATED_TO / EITHER | EXACT / HAS_PARTICIPANT / E2_TO_E3 |
| P04 | EXACT / IS_A / E2_TO_E3 | APPROXIMATE / CONCEPTUALLY_RELATED_TO / EITHER | EXACT / IS_A / E2_TO_E3 | EXACT / SUBTYPE_OF / E2_TO_E3 |
| P05 | EXACT / HAS_PART / E2_TO_E3 | APPROXIMATE / CONCEPTUALLY_RELATED_TO / EITHER | EXACT / HAS_PART / E2_TO_E3 | EXACT / PART_OF / E3_TO_E2 |
| P06 | APPROXIMATE / APPLIED_TO / E2_TO_E3 | APPROXIMATE / ADDRESSES_RELATED_PROBLEM / EITHER | APPROXIMATE / APPLIED_TO / E2_TO_E3 | EXACT / POTENTIALLY_APPLICABLE_TO / E2_TO_E3 |

P01 STAGE_OF сохраняет стадию; IS_A допускает более грубую классификацию. P03 HAS_PARTICIPANT не предполагает специальной agency. P04 SUBTYPE_OF/IS_A описывают specialization процесса, не одинаковые гены или события. P05 PART_OF требует **E3→E2** и component-type scope; волокна из разных работ не становятся тем же оборудованием. P02/P06 требуют modality и дополнительных клинических/изобразительных условий, не actual application. Все эти смыслы использовались в дизайне и потому не являются holdout gold.

Exact candidate coverage-conditioned hits: A 0/2, B denominator=0 (не 0% recall), C 0/2, D 0/6. Это счётчик заданных candidate meanings, не полный corpus recall. Все шесть P-пар дополнительно имеют zero any-label output во всех конфигурациях. Ошибочный approximate label не может заменить exact hit, а отсутствие exact label не штрафуется как model miss.

| Кандидат | Frozen D target / direction | Raw score D для ожидаемого смысла | Максимальный D score на паре |
|---|---|---:|---|
| P01 | STAGE_OF / E2_TO_E3 | 0.00041253 | STAGE_OF E3→E2 0.00234839 |
| P03 | HAS_PARTICIPANT / E2_TO_E3 | 0.00006884 | PART_OF E3→E2 0.00926681 |
| P06 | POTENTIALLY_APPLICABLE_TO / E2_TO_E3 | 0.00000403 | SUBTYPE_OF E3→E2 0.00008235 |
| P02 | POTENTIALLY_APPLICABLE_TO / E2_TO_E3 | 0.01095484 | APPLIED_TO E2→E3 0.07829803 |
| P04 | SUBTYPE_OF / E2_TO_E3 | 0.00148867 | PART_OF E3→E2 0.00563346 |
| P05 | PART_OF / E3_TO_E2 | 0.00063710 | APPLIED_TO E2→E3 0.00101319 |

Показываются raw scores для диагностики, а не основание понизить порог постфактум. Например, более высокий score APPLIED_TO на P02 не является доказательством actual drug exposure. Малые scores concept/stage relations могут отражать отсутствие нужного утверждения в двух локальных текстах и/или трудность новых формулировок; причинно отделить эти факторы этот опыт не позволяет.

## Все положительные междокументные предсказания

| Config / pair | Head → code → tail | Score | Вердикт |
|---|---|---:|---|
| A / N05 | E3 → HAS_RESULT → E2 | 0.76167351 | REJECTED |
| A / N05 | E2 → HAS_RESULT → E3 | 0.65046901 | REJECTED |
| B / N07 | E2 → USES_SAME_METHOD_AS → E3 | 0.65934896 | REJECTED |
| B / N07 | E3 → USES_SAME_METHOD_AS → E2 | 0.49165270 | REJECTED |
| C / N07 | E2 → USES_SAME_METHOD_AS → E3 | 0.41241628 | REJECTED |
| C / N07 | E2 → USES → E3 | 0.44402343 | REJECTED |
| C / N05 | E2 → HAS_RESULT → E3 | 0.41097948 | REJECTED |
| C / N05 | E3 → PRODUCES → E2 | 0.37926105 | REJECTED |
| C / N05 | E3 → HAS_RESULT → E2 | 0.53520751 | REJECTED |
| C / N06 | E2 → HAS_RESULT → E3 | 0.48149049 | REJECTED |
| C / N06 | E3 → HAS_RESULT → E2 | 0.42203167 | REJECTED |

Все 11 main outputs отклонены по источникам, а не из-за принадлежности к N. N05 связывает тарифные results и lunar/regolith results как HAS_RESULT/PRODUCES. N06 смешивает mammographic-density outcomes и glaucoma/graft outcomes. N07 USES_SAME_METHOD_AS связывает protein computational design и yeast transcriptomics/evolutionary engineering; USES не имеет evidence использования одной работы другой. Независимые локальные results/study нельзя превращать в происхождение, причинность или method identity. Максимальный rejected score A достигает примерно 0.762.

На W01–W10 у всех конфигураций нулевой выход >0.35. Это сохранённое наблюдение, не доказательство невозможности любого концептуального отношения. N01 supply-line homonym и N02 grains/grain в данном контексте не дали ошибок, но их отсутствие не подтверждает освоение контекстной семантики вообще. Все N оцениваются по конкретному type/direction/scope; common-category или корректная modal hypothesis не отклонялись бы автоматически.

## Sanity set: модель реагирует на часть новых labels

| Config | Предсказания | Supported / plausible / uncertain / rejected | Source-checked ожидаемые смыслы, распознанные при >0.35 |
|---|---:|---|
| A | 9 | 6 / 0 / 1 / 2 | S01, S02, S03, S04, S05, S06 |
| B | 3 | 1 / 0 / 1 / 1 | S04 |
| C | 8 | 5 / 0 / 1 / 2 | S01, S04, S05, S06 |
| D | 6 | 6 / 0 / 0 / 0 | S01, S02, S03, S04, S05, S06 |

D выдаёт ровно по одному ожидаемому отношению на всех S01–S06: subtype, property (два примера), derivation, actual application, intended use. Следовательно, динамические prompts работают, и полный нулевой cross-document выход D не является отсутствием любого отклика на D ontology. Это не проверка всех девяти labels: STAGE_OF, PART_OF, HAS_PARTICIPANT и POTENTIALLY_APPLICABLE_TO не имеют positive sanity в этом наборе.

Sanity также выявляет ошибки и неоднозначности: A даёт MEASURES между composite и strength-to-weight ratio, хотя материал является носителем свойства, а не прибором; C даёт IS_A на той же паре. Для GPU algorithm/OB-NLM DERIVED_FROM/BUILDS_ON согласуются с based-on construction; USES_SAME_METHOD_AS смешивает реализацию с методом и отклонён. DEPENDS_ON и COMPLEMENTS оставлены UNCERTAIN: текст подтверждает derivation, но не формализует отдельную parameter dependency или complementary function. Роль GPU в intended diagnosis workflow не превращает его в самостоятельного агента PERFORMS diagnosis.

## Все 12 порогов и виртуальная связность

Фильтр >=t применяется к сохранённым float scores; модель не запускается заново. Распределение **всех** labels, в том числе нулевых, и все счётчики P/N/W находятся в `threshold_comparison.csv`. Sanity исключён из таблицы ниже.

| Config | t | Predictions / pairs | Informative / rejected | Exact P hits | Raw WCC / largest / docs | Accepted cross-WCC pairs |
|---|---:|---|---|---:|---|---:|
| A | 0.35 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| A | 0.40 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| A | 0.45 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| A | 0.50 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| A | 0.55 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| A | 0.60 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| A | 0.65 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| A | 0.70 | 1 / 1 | 0 / 1 | 0 | 223 / 369 / 17 | 0 |
| A | 0.75 | 1 / 1 | 0 / 1 | 0 | 223 / 369 / 17 | 0 |
| A | 0.80 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| A | 0.85 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| A | 0.90 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| B | 0.35 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| B | 0.40 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| B | 0.45 | 2 / 1 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| B | 0.50 | 1 / 1 | 0 / 1 | 0 | 223 / 369 / 17 | 0 |
| B | 0.55 | 1 / 1 | 0 / 1 | 0 | 223 / 369 / 17 | 0 |
| B | 0.60 | 1 / 1 | 0 / 1 | 0 | 223 / 369 / 17 | 0 |
| B | 0.65 | 1 / 1 | 0 / 1 | 0 | 223 / 369 / 17 | 0 |
| B | 0.70 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| B | 0.75 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| B | 0.80 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| B | 0.85 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| B | 0.90 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.35 | 7 / 3 | 0 / 7 | 0 | 223 / 369 / 17 | 0 |
| C | 0.40 | 6 / 3 | 0 / 6 | 0 | 223 / 369 / 17 | 0 |
| C | 0.45 | 2 / 2 | 0 / 2 | 0 | 223 / 369 / 17 | 0 |
| C | 0.50 | 1 / 1 | 0 / 1 | 0 | 223 / 369 / 17 | 0 |
| C | 0.55 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.60 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.65 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.70 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.75 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.80 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.85 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| C | 0.90 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.35 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.40 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.45 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.50 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.55 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.60 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.65 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.70 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.75 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.80 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.85 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |
| D | 0.90 | 0 / 0 | 0 / 0 | 0 | 223 / 369 / 17 | 0 |

Baseline восстановлен offline из 811 contextual entities, 32 concepts, 1307 RELEX_REL, 492 SIMILAR_TO и 118 HAS_SHARED_CONCEPT. Document/MENTIONS исключены. Сохраняются 223 WCC, largest 369 vertices / 17 documents во всех 48 config/threshold combinations. Нет даже raw cross-WCC outputs. Отдельные virtual_raw, virtual_supported и virtual_accepted_including_hypotheses поля совпали с baseline. В общем случае raw structural merges не являлись бы научным улучшением; hypotheses и concept relations не становятся empirical KG facts.

A становится пустым при 0.80, B при 0.70, C при 0.55; D пуст на всех заданных порогах. Повышение порога здесь удаляет проверенные ошибки, но не восстанавливает P-покрытие и не выбирает рабочего победителя. Ошибка A остаётся даже при 0.75.

## Ответы на исследовательские вопросы

1. **Новая онтология:** D из девяти коротких labels, отделяющая taxonomy/stage, structure/participation, property/function и actual/potential applicability. Определения, роли, examples и основания frozen до inference.

2. **Отличие от A/B/C:** отсутствуют untyped relatedness и необоснованные causal shortcuts, добавлены точные stage/participant/modal смыслы. Меняется и состав, и формулировка labels; это не изолированная prompt-wording ablation.

3. **Более точная представимость:** стадия заболевания P01, клеточный участник P03 и modality P02/P06. P04/P05 A/C уже могли выражать точно при class/component-type scope.

4. **Распознана ли содержательная междокументная связь?** Нет при всех заданных порогах. Sanity-success относится к внутридокументным примерам и не подменяет main result.

5. **Улучшение D?** Улучшено проектное coverage и исчезли наблюдаемые false positives на этом наборе. Прирост meaningful extraction/coverage-conditioned candidate hits не показан; нулевой выход не доказывает высокую precision или superiority.

6. **Ошибки слабых пар:** main false positives остались на N05/N06/N07 у A/B/C; D не дала их. W cases все negative-output, но универсальный gold NO_RELATION из этого не следует.

7. **Сохраняющиеся классы ошибок:** lexical results/study shortcuts, неверные bearer/measurement roles и implementation/method conflation. У D главная наблюдаемая проблема main — abstention/missing candidate relations; его sanity показывает реакцию лишь на часть labels.

8. **Пригодность D для второго прохода:** как исследовательская схема аннотирования и разделения meaning/modality — пригодна; как подтверждённая рабочая ontology для извлечения этим checkpoint — пока не обоснована. Предсказания импортировать нельзя.

9. **Главное ограничение:** старый candidate discovery действительно пропускал P-смыслы, а A/B/C имели ontology gaps. В текущем диагностическом наборе человечески мотивированные кандидаты и D coverage уже есть, но выходов нет: непосредственно ограничивает связка модель + два локальных контекста + задача неявного междокументного вывода. Нельзя приписать всё только весам модели: отсутствует context/label-wording ablation, часть P требует внешнего знания, а четыре D labels не имеют отдельного positive sanity.

10. **Следующий шаг:** прекратить подбор labels на этих же шести парах как поиск рабочего решения. Предпочтительно разделить concept linking и evidence-grounded relation extraction: типизированные conceptual links проверять по определениям/таксономии, empirical relations требовать из источников с совместимыми условиями. До смены модели можно спроектировать один независимый context/scope diagnostic (например, те же значения в явном локальном утверждении против двух раздельных источников), с новым holdout и без post-hoc tuning. Для оценки научной генерализации нужен тематически более плотный независимо размеченный корпус. Эти варианты только предложены; новые LLM, модели или Stage 2F.5 не запускались.

## Воспроизводимость и ограничения

`manifest.json` содержит source commit, все 1183 protected tracked-file hashes, checkpoint/config/tokenizer hashes, точные ordered labels, runner/adapter hashes, frozen input hashes, signatures, runtime и accounting forwards. `validation.json` фиксирует technical preparation, compatibility, pilot gate, завершённость и итоговые проверки. Каждый prediction row имеет signature, input SHA, raw scores и head audit. `quality_review.csv` покрывает все 11 main и 26 sanity positive predictions; `pilot_review.csv` неизменён после gate.

Воспроизвести score analysis из сохранённых результатов:

```powershell
.\.venv\Scripts\python.exe scripts/report_cross_document_ontology.py --stage analyze
```

Inference runner имеет этапы `prepare`, `pilot`, `infer`. Prepare отказывается перезаписывать уже frozen inputs. На завершённом наборе `infer` проверяет hashes/signatures и использует checkpoints; повторять модель для thresholds не требуется. Для независимого повторения можно скопировать frozen ontology/design/coverage/contexts/manifest/validation/pilot review в отдельный каталог, использовать тот же runner с OUT, направленным на копию, и отдельные пустые output checkpoints. Изменение runner source или labels требует новой experimental signature, а не смешения в текущий набор.

Все оценки сделаны одним рецензентом по оригинальным абстрактам. Санити и P-смыслы не независимы от известного корпуса/проектирования. Human adjudication использует контексты и явные дополнительные знания из прошлого аудита; эти знания не добавлены модели. Label scores между A/B/C/D не считаются одинаково калиброванными. Нельзя считать 0/6 полным recall или разность false-positive counts статистическим превосходством. Пустые и непустые результаты сохранены без подбора labels после pilot.

В сравнении со старым опытом 68 пар изменились и выборка, и source-only context recipe. Поэтому меньшие A/B/C counts здесь нельзя приписать одному улучшению discovery или отсутствию summaries. Внутри нового опыта контексты и вычислительные настройки одинаковы для всех конфигураций.

Модельные веса, NER 0.46, ER, все 1307 factual relations, 492 semantic links, 118 concept links, исходные документы и старые отчёты сохранены; byte hashes проверены до и после inference/analysis. База Neo4j не открывалась: отсутствие записи обеспечивается отсутствием graph calls, а актуальное состояние внешнего сервера отдельно не проверялось. Никаких новых empirical edges, inference сверх описанного, push или перехода к Stage 2F.5 в этом задании нет.
