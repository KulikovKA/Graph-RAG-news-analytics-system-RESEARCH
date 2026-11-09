# Этап 2C — предварительная проверка кандидатов

Объединены 201960 направленных рекомендаций 18 конфигураций в **49336 уникальных неупорядоченных пар**. Самопар нет; направления исходных рекомендаций, все модели, режимы, ранги и cosine сохранены в observations. Это анализ пар, не объединение сущностей.

| Область | Всего пар | ACCEPT | REVIEW | REJECT | Индивидуально разобрано |
|---|---:|---:|---:|---:|---:|
| within_document | 10632 | 45 | 10547 | 40 | 126 |
| cross_document | 38704 | 1 | 38681 | 22 | 26 |
| Всего | 49336 | 46 | 49228 | 62 | 152 |

Индивидуальная выборка: 152 пар из фактически найденных кандидатов; 44 из них остались REVIEW после разбора. Остальные 49184 пар **не проверены индивидуально**, имеют individually_reviewed=False и REVIEW, а не фиктивное заключение об идентичности. Каталог включает явные аббревиатуры/синонимы, узкие/широкие понятия, классы/экземпляры, когорты/измерения, типовые конфликты и повреждения; выборка целевая, не случайная и не репрезентативная. Несопоставленные с кандидатами случаи каталога в статистику решений не входят.

Состав выборки: 122 найденных случаев из базового каталога 127 контекстных примеров и 30 дополнительных пар, охвачены 27 документов. Дополнительный отбор воспроизводим: по 6 внекаталожных пар с наибольшим числом моделей/конфигураций в каждой области и по одной паре, найденной только одной моделью, на модель и область; сначала лучший rank, затем canonical ID. Счётчики режимов и направлений не умножают число моделей. Отбор выполнен до присвоения решений этим дополнительным парам; он не даёт оснований обобщать частоты ACCEPT/REJECT на весь корпус.

Использованы все контексты исходных span участников из members 2A и полные исходные абстракты, если короткого окна недостаточно. В verified_pairs.csv сохранены left/right_contexts для всех участников, исходные entity/mention IDs, document ID, типы/флаги и точные цитаты source_evidence с offsets и revision_id для индивидуальных решений. left/right упорядочены по canonical ID для единственности пары; reason_entity_order явно задаёт порядок участников в объяснении («первый/второй»). В таблице примеров имена показаны в порядке объяснения. Решения не принимаются по score: это детерминированное применение явно записанного контекстного разбора к фиксированному снимку, без дополнительной LLM-разметки.

## Согласие моделей

Число моделей считается без повторов по режимам и направлениям. Модель могла найти пару в одном или двух представлениях; все оценки сохранены отдельно. Согласие не является доказательством корректности.

| Число моделей | Внутридокументных пар | Междокументных пар |
|---|---:|---:|
| 1 | 1989 | 21010 |
| 2 | 1213 | 7774 |
| 3 | 829 | 3756 |
| 4 | 711 | 2152 |
| 5 | 601 | 1269 |
| 6 | 531 | 838 |
| 7 | 530 | 553 |
| 8 | 661 | 445 |
| 9 | 3567 | 907 |

## Диагностика моделей на целевой выборке

Числа ACCEPT/REJECT ниже — сколько разобранных пар было найдено моделью в каждом режиме; это не Precision, Recall, F1 или оценка на независимой разметке. Отбор из объединённых top-5 смещён в пользу уже найденных случаев. Время и RAM взяты из 2B; PyTorch E5-small нельзя напрямую ранжировать по скорости с ONNX как чистое сравнение моделей.

| Модель | ACCEPT: name / context | REJECT: name / context | Инференс двух режимов, с | Максимальный RSS, MiB |
|---|---:|---:|---:|---:|
| BAAI/bge-base-en-v1.5 | 40 / 33 | 43 / 25 | 85.63 | 599.0 |
| BAAI/bge-large-en-v1.5 | 41 / 37 | 44 / 30 | 386.43 | 1473.0 |
| BAAI/bge-small-en-v1.5 | 38 / 33 | 40 / 28 | 23.62 | 280.8 |
| intfloat/e5-base-v2 | 40 / 34 | 43 / 27 | 89.78 | 601.6 |
| intfloat/e5-large-v2 | 39 / 32 | 42 / 26 | 374.06 | 1512.9 |
| intfloat/e5-small-v2 | 39 / 32 | 43 / 25 | 41.23 | 569.2 |
| thenlper/gte-base | 42 / 34 | 42 / 27 | 82.36 | 599.3 |
| thenlper/gte-large | 41 / 33 | 42 / 27 | 320.63 | 1473.2 |
| thenlper/gte-small | 38 / 36 | 39 / 25 | 23.94 | 285.2 |

Практически перспективны BGE-small и GTE-small как экономичные CPU-кандидаты: два режима заняли 23.62 и 23.94 с при пиках RSS 280.8 и 285.2 MiB. GTE-base — вариант для сопоставления с ними: 82.36 с, name/context находят 42/34 предварительно принятых случаев. Это shortlist по стоимости и целевым диагностическим примерам, **не доказанный рейтинг качества**. Large-модели дороже, независимого подтверждения выигрыша ER здесь нет.

Контекст сильно меняет списки: пересечение name/context top-5 составляет 0.228–0.261 внутри документа и 0.041–0.099 между документами. На этой выборке у всех 9 моделей name находит больше ACCEPT-пар, чем context, хотя context добавляет отдельные полезные случаи. Поэтому контекстный режим нельзя объявить безусловно лучшим или использовать как единственную замену именам; разные окна могут усиливать тематическую близость и оттеснять корректные синонимы.

## Контекстные решения и высокие ложные сходства

Ниже приведены иллюстративные случаи; полный разбор выборки сохранён в verified_pairs.csv. Диапазон cosine объединяет разные шкалы только для просмотра примера, не служит общей метрикой качества и не используется в решении.

| Область | Названия | Решение | Моделей | Min / max cosine | Уровень и основание |
|---|---|---|---:|---|---|
| within_document | preoperative IOP ↔ IOP | ACCEPT | 9 | 0.809 / 0.997 | same_preoperative_measurement: Оба span обозначают одно preoperative mean IOP 34.0 в том же выражении; не объединяется с послеоперационным IOP. |
| within_document | chronic rhinosinusitis ↔ chronic rhinosinusitis with nasal polyposis | REVIEW | 9 | 0.859 / 0.996 | broader_disease_vs_specified_subtype: Короткий span не сохраняет with nasal polyposis; без решения о границах нельзя приравнять общее заболевание уточнённому подтипу. |
| within_document | dataset ↔ training set | REJECT | 9 | 0.691 / 0.978 | dataset_vs_partition: Набор делят на обучающую и тестовую части; весь dataset не равен training set. |
| cross_document | cancer ↔ cancer | ACCEPT | 9 | 0.719 / 1.000 | general_disease_concept: Оба употребления обозначают общий медицинский концепт cancer без уточнения органа; речь о болезни как категории, не о тождестве пациентов, опухолей или клинических случаев. |
| cross_document | medical complications ↔ complications | REVIEW | 9 | 0.745 / 0.967 | complication_class_vs_study_events: Общий медицинский класс осложнений может пересекаться, но фактические эпизоды в TKA и GDD/PK-когортах различны; неясен concept/event уровень CONDITION. |
| cross_document | Corneal outcome ↔ outcome | REJECT | 9 | 0.732 / 0.946 | different_clinical_outcomes: Corneal outcome измеряют rejection/failure/visual acuity; outcome в TKA оценивают Knee Society/Arthritis Index. Это разные клинические исходы и носители. |
| within_document | genes ↔ genes | ACCEPT | 9 | 0.914 / 1.000 | same_named_gene_pair: The genes ilvB and ilvH затем повторены как these genes в том же предложении; оба span относятся к этой паре, а не к следующему acpP1. Это уточнение контекста, ранее оставленного на проверке. |
| within_document | preoperative IOP ↔ mean IOP | REJECT | 9 | 0.759 / 0.982 | different_timepoint_measurements: Первое явно preoperative, второе после postoperative follow-up; значения 34.0 и 12.7 не одно измерение. |
| within_document | Cyclotella sp. WC03_2 ↔ Cyclotella sp. L04_2 | REJECT | 9 | 0.850 / 0.975 | named_strain: В перечне даны два разных strain-идентификатора WC03_2 и L04_2; общий род не означает идентичность. |
| within_document | rHuEPO ↔ anti-rHuEPO neutralizing antibodies | REJECT | 9 | 0.741 / 0.894 | drug_vs_antibodies: Антитела направлены против rHuEPO; связывание/нейтрализация не тождество лекарства и антител. |
| within_document | Co0� 45Fe0� 45Zr0� 1 ↔ Co0� 45Fe0� 45Zr0� 1 | REVIEW | 9 | 0.932 / 1.000 | damaged_formula: U+FFFD повреждает состав формулы, а контексты обсуждают sublayers и grains; одинаковая повреждённая строка не доказывает тождество. |
| within_document | Limonene ↔ limonene | REVIEW | 9 | 1.000 / 1.000 | substance_vs_extraction_fragment: Второй limonene входит в limonene synthase gene; границы упоминания не позволяют автоматически признать тот же химический референт. |
| within_document | HLA-DRB1*09 ↔ HLA-DRB1*09 | REVIEW | 9 | 0.979 / 1.000 | gene_vs_allele_or_inexact_source_naming: 611 сопровождается gene, позднее то же имя названо allele; требуется отличить уровни gene/allele и возможную неточность самого источника. Совпадение идентификатора и один OBJECT не снимают эту неоднозначность. |
| within_document | novel adaptive region proposal network ↔ ARPN | REVIEW | 9 | 0.954 / 0.980 | named_model_with_type_conflict: ARPN явно раскрывается, но METHOD против ACTOR; требуется согласовать уровень/исходную типизацию. |
| within_document | breast cancer ↔ breast cancer risk | REJECT | 9 | 0.814 / 0.970 | disease_vs_risk: Диагноз заболевания и характеристика его риска — разные концепты; общий текст и совпадение CONDITION не делают их эквивалентными. |
| within_document | patients ↔ patients | REJECT | 9 | 1.000 / 1.000 | cohort_vs_subset: Исходная когорта содержит 21 пациента, позднее Six patients имеют осложнения; вся когорта и подмножество не равны. |

Проверка транзитивности принятых пар: противоречивых ACCEPT–ACCEPT–REJECT цепочек 0; незамкнутых ACCEPT–ACCEPT цепочек без принятой третьей пары 0. Они не замыкаются автоматически и не создают merge-групп.

## Выводы и ограничения

Явные раскрытия MD/PK/OB-NLM и названия именованных объектов дают контекстные основания сильнее cosine. В то же время высокая близость возникает у разных видов/штаммов, до-/послеоперационного IOP, болезни и её риска, препарата и антител, группы и подмножества. Повреждённые формулы и извлечённые фрагменты способны давать почти одинаковые векторы. Неизвестный автор другой статьи не считается заведомо другим человеком; при нехватке подтверждений сохранён REVIEW. При переходе между статьями различается общий научный концепт и конкретная когорта/объект.

Результат — собственная предварительная контекстная классификация, **не независимая человеческая gold-разметка**. Объективные Precision/Recall/F1, calibrated identity probabilities и числовые пороги не заявляются. Выводы о качестве Graph-RAG не делаются. Правила предложены в proposed_merge_rules.md; их применение и Semantic Merge (2D) не выполнялись.

## Проверки и воспроизводимость

Проверены 18 конфигураций и по 5 кандидатов каждой области для всех 1122 ID, все 201960 наблюдений, единственность неупорядоченных пар, source entity/mention IDs, названия, типы, области и флаги. Контекстные цитаты сверены с исходным текстом и revision_id, решения привязаны к SHA-256 входа 2A. Повторный запуск трёх артефактов 2C проверен побайтово. Хеши источников и результатов 1A–2B неизменны. Скрипт использует только стандартную библиотеку, модели не запускает.

Запуск: `python scripts/verify_semantic_candidates.py`. Создаются только verified_pairs.csv, proposed_merge_rules.md и report.md. Neo4j, canonical mapping и отношения не изменяются.

SHA-256 защищённых источников и результатов:

- `reports/entity_resolution/relex_base_t035/conservative_resolution/canonical_mapping.jsonl`: `f97e8f920cf7cced3483caca39bdf87079f2b8a9c7d5240b812c886cc766c8b3`
- `reports/entity_resolution/relex_base_t035/conservative_resolution/report.md`: `d9ae051c9bb57019f215aa5db9ab74150c144358850324337460db4b3930f9cc`
- `reports/entity_resolution/relex_base_t035/conservative_resolution/review_groups.csv`: `6ea86012816fc3d04603fc61a778c6577351611fba91984611f2beacb060551b`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/ambiguous_groups.csv`: `fdf596b77ed5e0fd063b52a64ee814da14675085d1520d4eb70e1606ce3e0795`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/exact_groups.csv`: `fda84b2faa3fddf13103f711ab1cf6bd04b86b802691b6d9d5c005b1583e0972`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/exact_merge_preview.jsonl`: `d68c01c9d72622a29deb3378079d6887641baef4f590a949bf2db9918bacd855`
- `reports/entity_resolution/relex_base_t035/exact_matching_audit/report.md`: `92158a6653b224524cbc231f04470465db3542771a6521187bda3863e84b595b`
- `reports/entity_resolution/relex_base_t035/graph_comparison/report.md`: `cff7dc986261ff26ed376ae309443291b1381fda3b349b53b6a6f4261815f14c`
- `reports/entity_resolution/relex_base_t035/relation_remapping/canonical_relations.jsonl`: `febcd7b9e717575bed79e3e1483402fdd81aae54b59d72b81d6b91582fccda6a`
- `reports/entity_resolution/relex_base_t035/relation_remapping/remapped_relations.jsonl`: `4b0ea8f647aed02a4b9a42a274d1e87611a2158fffe10cf3362f1d83884025c6`
- `reports/entity_resolution/relex_base_t035/relation_remapping/report.md`: `b760c94584a5f7dbf27bf410e88f678b1fba711292206f622c52ce1b69e546a4`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/input/canonical_entities.jsonl`: `34fe3fd42e110b5b7a8b054360eee2a6182c67782e7afa54d2298e0491bf9151`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/input/report.md`: `bd99b9d37b9461fcf932ff37428d1cbefb0d7fabb2b40977ae73430a212714dc`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/candidate_pairs.csv`: `5b5101945ce513243d3a28969b77425c4e1549a96bb31bb616546727d815ae2f`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/model_comparison.csv`: `e89fa1b0d133f33c0710b6bf595e1bbddc7fcc6c21d77bc2c9c222bf454aac6c`
- `reports/entity_resolution/relex_base_t035/semantic_resolution/model_comparison/report.md`: `40f5f0d7d5b0100a02c83b774430ec6538b99e3fde668c030f75c43c81a881bc`
- `reports/ner_v2/research_run2/input_abstracts.jsonl`: `684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1`
