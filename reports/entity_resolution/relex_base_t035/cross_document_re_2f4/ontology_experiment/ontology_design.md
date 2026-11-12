# Stage 2F.4 — проект Cross-Document Ontology D

Дата: 2026-11-12. Исходный HEAD `97b8196`, рабочее дерево перед изменениями чистое. Это отдельный эксперимент второго прохода. Первый проход, его 20 predicates, веса и старые отчёты не меняются.

## Исследовательская гипотеза

Обнаружение связи между независимыми научными контекстами требует различать значение терминов, уровень абстракции и силу утверждения. Низкое качество прежнего A/B/C нельзя объяснить только labels: 45/68 прежних примеров были patient/generic controls, все шесть P-пар открыты позже. Новый опыт одновременно проверяет A/B/C на лучшем candidate set и добавляет D. Он не является чистой ablation одного label и не сравним по числу рёбер со старым набором из 68 пар.

A выражает actual-use, membership, structure, properties и causal/dependency statements. Однако у него нет точного stage relation, общего process-participant relation и модальной применимости. IS_A не уточняет стадию; PERFORMS имеет более узкий agency-смысл, чем participation; APPLIED_TO не говорит «возможно при условиях». B смешивает свободную ассоциацию, сходство значения, общность задач, методов и влияние. RELATED_TO/CONCEPTUALLY_RELATED_TO недостаточно специфичны для научного вывода. USES_SAME_METHOD_AS требует совпадения методов, а не общего слова study; BUILDS_ON требует фактической опоры на другую работу. C наследует эти трудности и удаляет девять original codes: эффект C нельзя приписывать только добавлению semantic labels.

CAUSES/INFLUENCES/DEPENDS_ON между разными опытами требуют причинной/зависимостной модели, совместимых референтов и условий. Идентичность, co-reference, similar naming и class hierarchy не дают этого автоматически. Relation confidence не является доказательством и не калиброван между наборами.

## Основания выбора семейства отношений

[SKOS Reference, §8](https://www.w3.org/TR/skos-reference/#semantic-relations) разделяет hierarchical и associative связи понятий и задаёт симметрию related. Для этого опыта важен сам принцип различения: SKOS related не заменяет типизированную связь и не означает причинность. D не является реализацией SKOS и не наследует автоматически его аксиомы.

[OBO Relation Ontology: Process relations](https://oborel.github.io/obo-relations/process-relations/) различает process parts, participants и роли input/output/agent. Отсюда мотивировано выделение HAS_PARTICIPANT вместо смешения участия, части и действия субъекта. Это вдохновение семантической моделью, не импорт RO или доказательство эквивалентности наших class-level shortcuts формальным RO properties.

[Luan et al., SciERC/SciIE, EMNLP 2018](https://aclanthology.org/D18-1360/) рассматривают извлечение научных entities, relations и coreference совместно. В схеме исследования используются, в частности, различения hyponymy, part-whole и usage. Это основание сохранять конкретные виды связей, а не считать независимые одноимённые mentions одним научным объектом. Из статьи не следует, что ReLEx должен выводить гипотезы между независимыми документами.

[Model card GLiNER ReLEx base v1.0](https://huggingface.co/knowledgator/gliner-relex-base-v1.0) показывает передачу текстовых relation labels. Рабочий локальный адаптер кодирует prompt embeddings и оценивает направленные пары span representations. Определения в JSON предназначены рецензенту: в модель передаются только короткие английские labels, не эти инструкции. Поддержка произвольного текста label не гарантирует распознавание незнакомой формулировки. Проверка новых labels проводится экспериментально.

## Окончательная онтология

Единственный авторитетный набор — `cross_document_ontology.json`, version `cross-document-D-v1`. Там для каждой из девяти меток сохранены definition, head/tail roles, направление, положительный и отрицательный примеры вне целевых P-пар, отличие от старых отношений и необходимые основания.

| Code | Точный prompt | Значение и направление |
|---|---|---|
| SUBTYPE_OF | is a type of | Частный научный класс → более общий класс |
| STAGE_OF | is a stage of | Стадия/фаза → заболевание, развитие или жизненный цикл |
| PART_OF | is part of | Компонент → составной объект/процесс, в том числе явно оговорённый component-type scope |
| HAS_PARTICIPANT | has participant | Процесс/активность → участник |
| HAS_PROPERTY | has property | Носитель → свойство/параметр, не конкретный результат |
| USED_FOR | is used for | Метод/инструмент/вмешательство → задача/назначение |
| APPLIED_TO | is applied to | Метод/вмешательство → объект/данные/популяция при подтверждённом фактическом применении |
| POTENTIALLY_APPLICABLE_TO | may be applied to | Метод/вмешательство → совместимая цель при явно неустановленных дополнительных условиях |
| DERIVED_FROM | is derived from | Результат/модель/объект → источник/исходный материал при установленном происхождении |

Все девять направленные. Обратное направление не засчитывается автоматически. Не добавляются парные inverse labels: это увеличило бы пересечение prompt semantics без новой научной категории. Типы head/tail используются при смысловом аудите, не как автоматический filter по шумным NER labels. Предсказания несовместимых ролей сохраняются и проверяются.

Симметричные equality/similarity/common-method отношения сознательно не включены. Они могут быть научно полезны, но в текущем опыте вопрос о достаточности конкретной типизации и modality, а не о расширении associative graph. Поэтому отдельная E не создаётся: широкая E без новой проверяемой постановки повторила бы B.

STAGE_OF применим к заболеваниям, развитию организмов, физическим переходам, промышленным и программным жизненным циклам. HAS_PARTICIPANT применим к биологии, реакциям, социальным и вычислительным процессам. PART_OF — к материалам, приборам и процессам. HAS_PROPERTY, USED_FOR, APPLIED_TO, DERIVED_FROM взяты из общих научных потребностей всего корпуса, в том числе WSN, оптических/магнитных измерений, green banking, исторических источников и алгоритмов. Таким образом, D не представляет собой шесть уникальных меток под P01–P06.

## Риски интерпретации labels

SUBTYPE_OF и STAGE_OF могут пересекаться, если поздняя стадия также классифицируется как подтип. Они разделяются по точности сохранения заданного смысла. PART_OF относится к структуре, HAS_PARTICIPANT — к роли в процессе. HAS_PROPERTY не означает принадлежность одного измерения другому. USED_FOR выражает назначение, APPLIED_TO — применение к объекту. POTENTIALLY_APPLICABLE_TO — единственная модальная связь: её score не является вероятностью применимости. DERIVED_FROM не доказывает dependency или citation relation между исследованиями.

Короткое may модель может игнорировать, смешать potential с actual use или воспринимать потенциальность как разрешение соединять любые методы и данные. Это центральный проверяемый риск, а не обещание улучшения. Новые labels могут оказаться хуже освоены, чем стандартные is a/has part. Число и состав меток меняются относительно A/C, поэтому сравнение не выделяет отдельно длину prompt, формулировку и удаление неподходящих классов. Sigmoid scores не следует интерпретировать как softmax-конкуренцию.

## Заморозка выборки, coverage и оценивания до inference

Основной набор — все P01–P06, N01–N12 и W01–W10, всего 28 уникальных unordered endpoint pairs. Исходные canonical/native IDs, document IDs, revision IDs и mentions сохраняются. W и N не являются универсальными gold NO_RELATION. Модельный текст содержит только исходные fragments и нейтральные document boundaries. Expected labels, статусы аудита, внешние определения и аргументация P01/P05/P06 в текст не вставляются. Существующие локальные RELEX_REL сохраняются как refs, но summaries не вставляются: их ошибки не должны подсказывать целевой label. Это один общий context policy для A/B/C/D, без ablation context recipes.

Одинаковое окно выбирается по самому длинному prompt из A/B/C/D, с проверкой tokenizer и actual collator. Все 2 directed fixed-span pairs сохраняются без NER selection. Raw sigmoid scores хранятся для каждого label в обоих направлениях, включая ниже 0.35; пороги 0.35–0.90 рассчитываются без дополнительных forwards. Positive inventory для ручного review — scores строго >0.35; сравнение порогов uses >=t. Возможный score ровно 0.35 отдельно проверяется, чтобы не потерять его в threshold analysis.

`ontology_coverage.csv` фиксируется до первого forward. EXACT означает представимость конкретного смысла, APPROXIMATE — потерю существенного аспекта, NONE — отсутствие подходящего типа. Только exact, scientifically accepted, correctly directed predictions считаются exact candidate hit. Наличие label не равно model recall; отсутствие exact label исключает пару из соответствующего coverage-conditioned denominator. Отдельно сохраняются допустимые приблизительные выводы: они не засчитываются exact hit.

Оценки каждого выхода: SUPPORTED_CONCEPT_RELATION, PLAUSIBLE_HYPOTHESIS, UNCERTAIN, REJECTED. Проверяются type, direction, referents, abstraction scope, conditions и direct textual support. У P05 конкретные волокна разных работ не тождественны. У P02 и P06 actual application не утверждалась. Даже у правильного class-level вывода shared passage может отсутствовать. Causal/empirical novel facts оцениваются отдельно и не возникают автоматически из concept relations.

Sanity set — шесть исходных внутридокументных пар, выбранных и прочитанных до inference: rHuEPO/drug, composite/strength-to-weight ratio, WSN/energy consumption, GPU algorithm/OB-NLM, XMCD/multilayers, GPU/real-time diagnosis. Ожидаемые D-смыслы: subtype, два property, derivation, actual application, function. Это convenience set на известных источниках, не независимый benchmark; stage, participant и modality не получают локального positive sanity здесь, и нулевой выход отдельной метки не докажет её техническую неисправность.

Pilot IDs: P01, P03, P06, N01, N07, W04 для всех четырёх конфигураций; затем шесть sanity pairs. Проверка pilot включает все положительные выходы. Низкий выход и содержательные ошибки не являются stop gate; остановка только при технической проблеме, offsets/mapping ошибке или изменении protected files. После pilot labels и contexts не изменяются. Основной запуск дополняет checkpoint без повторного inference уже сохранённых пар.

P01–P06 использованы при проектировании. Результат на них — development-set diagnostic с одним рецензентом и внешними знаниями, а не независимый тест обобщения, полный recall или основание статистически объявлять победителя. Послеэкспериментальные идеи сохраняются только как будущие гипотезы.
