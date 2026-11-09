# Этап 2B — benchmark embedding-моделей

Завершены все 18 конфигураций: 9 моделей × text_name/text_name_context, одни и те же 1122 сущности из 30 документов. Суммарное время, включая проверку загрузки, хеширование и запись CSV: 1587.02 с. Entity Resolution, изменение графа и Relation Verification не выполнялись.

## Протокол

CPU: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD, 8 физических / 16 логических ядер; RAM 31.31 GiB. ОС: Windows-10-10.0.26200-SP0. Для каждого процесса: 4 вычислительных потока, batch=8, FP32, max_length=512; GPU не использовался. Порядок моделей — BGE small/base/large, E5 small/base/large, GTE small/base/large; внутри модели сначала название, затем контекст. Нагрузка машины не изолирована.

Все файлы проверены до инференса; все 9 локальных загрузок и коротких forward-pass прошли до основного сравнения. Модели запускались последовательно в отдельных процессах; завершение процесса освобождает веса/арены памяти. Загрузок из сети, новых зависимостей и дообучения нет. E5-small использует локальный PyTorch FP32 model.safetensors; остальные — локальные ONNX FP32 model.onnx. Это различие движков ограничивает прямое сравнение времени и RSS как свойств самих моделей.

Рекомендации авторов: [BGE](https://huggingface.co/BAAI/bge-base-en-v1.5) — CLS pooling, без retrieval-инструкции для симметричного сравнения; [E5](https://huggingface.co/intfloat/e5-base-v2) — masked mean pooling и query: для обеих сторон; [GTE](https://huggingface.co/thenlper/gte-base) — masked mean pooling, без префикса. Используется tokenizer.json соответствующей модели, включая собственную нормализацию tokenizer; поля 2A не изменяются. Все векторы L2-нормированы. Рецепты BGE/E5/GTE применяются одинаково в двух режимах.

Для каждого вектора вычисляется полный cosine со всеми 1122 векторами своей конфигурации. Сама сущность исключается; выбираются независимо top-5 внутри документа и top-5 между документами. Ровно 11220 направленных рекомендаций на конфигурацию, всего 201960 строк; взаимные рекомендации сохраняются отдельно. При равных float32 scores порядок определяется canonical ID. Нет порога similarity и фильтрации конфликтов/повреждений.

inference_seconds включает токенизацию, padding, forward, pooling и нормализацию всех 1122 текстов; tokenization_seconds указан отдельно. Загрузка и warmup исключены из этого времени. search_seconds включает cosine-матрицу и ранжирование, без вычисления описательных квантилей и без повторной проверки. RSS/private — максимумы, измеренные каждые 50 мс для свежего дочернего процесса; RSS включает модель, runtime и арены, поэтому это не размер весов. load_seconds/RSS относятся к одной загрузке на модель и повторены в обеих её строках: складывать их дважды нельзя. Дисперсию времени по независимым повторным запускам не оценивали.

## Скорость и память

| Модель | Режим | Движок | d | Инференс, с | Поиск, с | Пик RSS инференса, MiB | Усечение |
|---|---|---|---:|---:|---:|---:|---:|
| BAAI/bge-small-en-v1.5 | text_name | onnxruntime | 384 | 3.17 | 0.216 | 235.8 | 0 |
| BAAI/bge-small-en-v1.5 | text_name_context | onnxruntime | 384 | 20.45 | 0.218 | 280.8 | 0 |
| BAAI/bge-base-en-v1.5 | text_name | onnxruntime | 768 | 10.23 | 0.210 | 532.2 | 0 |
| BAAI/bge-base-en-v1.5 | text_name_context | onnxruntime | 768 | 75.39 | 0.248 | 599.0 | 0 |
| BAAI/bge-large-en-v1.5 | text_name | onnxruntime | 1024 | 48.23 | 0.256 | 1399.3 | 0 |
| BAAI/bge-large-en-v1.5 | text_name_context | onnxruntime | 1024 | 338.20 | 0.240 | 1473.0 | 0 |
| intfloat/e5-small-v2 | text_name | pytorch | 384 | 7.35 | 0.488 | 511.7 | 0 |
| intfloat/e5-small-v2 | text_name_context | pytorch | 384 | 33.89 | 0.273 | 569.2 | 0 |
| intfloat/e5-base-v2 | text_name | onnxruntime | 768 | 13.01 | 0.260 | 533.5 | 0 |
| intfloat/e5-base-v2 | text_name_context | onnxruntime | 768 | 76.77 | 0.253 | 601.6 | 0 |
| intfloat/e5-large-v2 | text_name | onnxruntime | 1024 | 48.04 | 0.212 | 1403.4 | 0 |
| intfloat/e5-large-v2 | text_name_context | onnxruntime | 1024 | 326.03 | 0.257 | 1512.9 | 0 |
| thenlper/gte-small | text_name | onnxruntime | 384 | 3.31 | 0.231 | 236.5 | 0 |
| thenlper/gte-small | text_name_context | onnxruntime | 384 | 20.64 | 0.202 | 285.2 | 0 |
| thenlper/gte-base | text_name | onnxruntime | 768 | 10.05 | 0.210 | 532.8 | 0 |
| thenlper/gte-base | text_name_context | onnxruntime | 768 | 72.31 | 0.205 | 599.3 | 0 |
| thenlper/gte-large | text_name | onnxruntime | 1024 | 37.94 | 0.209 | 1399.8 | 0 |
| thenlper/gte-large | text_name_context | onnxruntime | 1024 | 282.68 | 0.225 | 1473.2 | 0 |

## Распределение cosine similarity

Ниже квантили всех неупорядоченных пар без самосравнений, отдельно внутри/между документами (не только отобранные top-5). Средние и остальные квантили, включая распределения кандидатов, записаны в model_comparison.csv. Шкалы разных моделей не калиброваны друг относительно друга.

| Модель | Режим | Внутри: P05 / P50 / P95 | Между: P05 / P50 / P95 | Конфликты типов в top-5: внутри / между |
|---|---|---|---|---:|
| BAAI/bge-small-en-v1.5 | text_name | 0.479 / 0.573 / 0.811 | 0.441 / 0.527 / 0.623 | 3478 / 3989 |
| BAAI/bge-small-en-v1.5 | text_name_context | 0.635 / 0.768 / 0.931 | 0.466 / 0.549 / 0.643 | 4271 / 4784 |
| BAAI/bge-base-en-v1.5 | text_name | 0.429 / 0.534 / 0.780 | 0.384 / 0.483 / 0.579 | 3460 / 3959 |
| BAAI/bge-base-en-v1.5 | text_name_context | 0.674 / 0.815 / 0.942 | 0.510 / 0.618 / 0.738 | 4269 / 4842 |
| BAAI/bge-large-en-v1.5 | text_name | 0.452 / 0.558 / 0.777 | 0.410 / 0.510 / 0.613 | 3534 / 3784 |
| BAAI/bge-large-en-v1.5 | text_name_context | 0.629 / 0.762 / 0.921 | 0.479 / 0.559 / 0.645 | 4208 / 4661 |
| intfloat/e5-small-v2 | text_name | 0.749 / 0.793 / 0.905 | 0.738 / 0.777 / 0.831 | 3541 / 3813 |
| intfloat/e5-small-v2 | text_name_context | 0.825 / 0.878 / 0.968 | 0.756 / 0.793 / 0.830 | 4268 / 4882 |
| intfloat/e5-base-v2 | text_name | 0.723 / 0.773 / 0.894 | 0.706 / 0.749 / 0.818 | 3525 / 3793 |
| intfloat/e5-base-v2 | text_name_context | 0.802 / 0.862 / 0.961 | 0.717 / 0.755 / 0.800 | 4246 / 4597 |
| intfloat/e5-large-v2 | text_name | 0.724 / 0.774 / 0.890 | 0.708 / 0.753 / 0.825 | 3586 / 3917 |
| intfloat/e5-large-v2 | text_name_context | 0.796 / 0.853 / 0.964 | 0.720 / 0.756 / 0.796 | 4272 / 4919 |
| thenlper/gte-small | text_name | 0.743 / 0.783 / 0.904 | 0.722 / 0.763 / 0.805 | 3528 / 4059 |
| thenlper/gte-small | text_name_context | 0.870 / 0.930 / 0.980 | 0.779 / 0.842 / 0.891 | 4257 / 4847 |
| thenlper/gte-base | text_name | 0.724 / 0.765 / 0.892 | 0.702 / 0.742 / 0.784 | 3530 / 4027 |
| thenlper/gte-base | text_name_context | 0.893 / 0.952 / 0.987 | 0.810 / 0.868 / 0.905 | 4270 / 4821 |
| thenlper/gte-large | text_name | 0.733 / 0.773 / 0.891 | 0.710 / 0.751 / 0.792 | 3529 / 3911 |
| thenlper/gte-large | text_name_context | 0.913 / 0.953 / 0.984 | 0.845 / 0.878 / 0.911 | 4185 / 4802 |

## Изменение состава кандидатов

Доля совпадения top-5 названия и контекста = |пересечение|/5, усреднённая по сущностям. Межмодельный Jaccard — |пересечение|/|объединение|, усреднённый по сущностям и восьми другим моделям в том же режиме; это согласие списков, не точность.

| Модель | Совпадение name/context: внутри / между | Jaccard с другими моделями, контекст: внутри / между |
|---|---:|---:|
| BAAI/bge-small-en-v1.5 | 0.242 / 0.071 | 0.732 / 0.115 |
| BAAI/bge-base-en-v1.5 | 0.246 / 0.042 | 0.711 / 0.114 |
| BAAI/bge-large-en-v1.5 | 0.254 / 0.099 | 0.725 / 0.107 |
| intfloat/e5-small-v2 | 0.229 / 0.045 | 0.738 / 0.090 |
| intfloat/e5-base-v2 | 0.232 / 0.072 | 0.748 / 0.117 |
| intfloat/e5-large-v2 | 0.228 / 0.041 | 0.739 / 0.093 |
| thenlper/gte-small | 0.243 / 0.049 | 0.701 / 0.108 |
| thenlper/gte-base | 0.239 / 0.044 | 0.704 / 0.127 |
| thenlper/gte-large | 0.261 / 0.071 | 0.683 / 0.112 |

## Примеры кандидатов и рисков

Примеры ниже выбираются только из реально полученных top-5. Контекстные основания — иллюстрации для просмотра, не независимая gold-разметка и не разрешение на merge.

| Категория | Модель / режим / область | Названия | cosine | Строки входа 2A | Основание/риск |
|---|---|---|---:|---|---|
| Контекстный пример | thenlper/gte-large / text_name_context / within_document | Mammographic density ↔ MD | 0.977201 | 1, 10 | Определение аббревиатуры в исходном документе; перспективный кандидат, требуется проверка типов и референта. |
| Контекстный пример | thenlper/gte-small / text_name_context / within_document | breast cancer ↔ breast cancer risk | 0.998090 | 11, 12 | Заболевание и риск заболевания тематически связаны, но это разные сущности. |
| Контекстный пример | thenlper/gte-large / text_name_context / within_document | limonene ↔ limonene synthase gene | 0.998032 | 811, 812 | Вещество и ген/фермент нельзя приравнивать по общей теме или включению названия. |
| Одинаковое имя между документами | BAAI/bge-small-en-v1.5 / text_name / cross_document | RESULTS ↔ results | 1.000000 | 27, 310 | Одинаковое имя, например patients, не доказывает одинаковую когорту/референт. |
| Повреждённый вход | BAAI/bge-small-en-v1.5 / text_name / within_document | Co0� 45Fe0� 45Zr0� 1 ↔ Co0� 45Fe0� 45Zr0� 1 | 1.000000 | 354, 360 | Повреждённые формулы сохраняются; близкие векторы могут отражать одинаковые дефекты текста. |

## Ограничения и сохранность

Независимой проверенной разметки пар нет: Precision/Recall/F1, порог автоматического объединения и победитель по качеству не объявляются. Больший средний cosine и межмодельное согласие не доказывают правильность. Большие модели могут требовать больше ресурсов; измеренные скорость/RSS зависят также от backend и нагрузки CPU. Общий контекст способен поднимать тематически связанные, но разные термины. Контекст 2A — только окно представителя, а не все возможные значения термина.

Все 15 повреждённых названий, 40 повреждённых контекстов, 108 множественных типов и 10 partial_token сохранены без фильтрации. candidate_pairs.csv содержит исходные entity/mention IDs обеих сторон, типы, документы, quality_flags, конфликт/непересечение типов и признак повреждения фактического текста режима. Контексты и полные provenance восстанавливаются по source_input + query_input_line/candidate_input_line и canonical ID; текст 2A не переписывается.

Проверены все 18 конфигураций и одинаковое покрытие 1122 ID, отсутствие самопар, правильность области поиска и рангов 1–5, конечность и единичные нормы векторов, симметрия cosine-матрицы, детерминированные ties. В каждом режиме повторный forward первого полного batch дал идентичные векторы, а независимая повторная сортировка — те же top-5. Контрольные суммы векторов, фактических текстов с префиксом, файлов моделей и версий runtime записаны в model_comparison.csv. Времена и RSS при полном повторном запуске могут изменяться. Все хеши исходников и результатов 1A–2A проверены; веса моделей не изменены. Этап 2C и объединение сущностей не выполнялись.

Запуск: `.venv/Scripts/python.exe scripts/benchmark_semantic_embeddings.py`. Только существующие numpy, tokenizers, onnxruntime, psutil и torch/transformers для E5-small; ничего не устанавливается. Векторы/матрицы существуют в памяти и отдельными файлами не сохраняются.

## Идентичность моделей

Полные пути, SHA-256 каждого использованного файла, размеры и версии runtime доступны в model_comparison.csv. Корень моделей: `../embedding-models`.

| Модель | Файл весов | SHA-256 |
|---|---|---|
| BAAI/bge-small-en-v1.5 | BAAI__bge-small-en-v1.5/onnx/model.onnx | 828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35 |
| BAAI/bge-base-en-v1.5 | BAAI__bge-base-en-v1.5/onnx/model.onnx | 9bc579acdba21c253c62a9bf866891355a63ffa3442b52c8a37d75b2ccb91848 |
| BAAI/bge-large-en-v1.5 | BAAI__bge-large-en-v1.5/onnx/model.onnx | 69ed3f810d3b6d13f70dff9ca89966f39c0a0e877fb88211be7bcc070df2a2ce |
| intfloat/e5-small-v2 | intfloat__e5-small-v2/model.safetensors | 45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1 |
| intfloat/e5-base-v2 | intfloat__e5-base-v2/onnx/model.onnx | 157f97ef1957d34f52efa26f8031371bf9043acc45460cec7ebe94631ac0e96b |
| intfloat/e5-large-v2 | intfloat__e5-large-v2/onnx/model.onnx | f560bbc5da586764fc6ccf54990bac71f94e5a872ed4e1889039391c849cbcf8 |
| thenlper/gte-small | thenlper__gte-small/onnx/model.onnx | 0b01312b59bec0a2558a626f2937be4cbe4bb16d1511560153f598cec488f1f8 |
| thenlper/gte-base | thenlper__gte-base/onnx/model.onnx | dbfc7a6898c7c95fc53e52aaaf8302b5b2f5e8ec90d0eafa8e3d4acd26abef39 |
| thenlper/gte-large | thenlper__gte-large/onnx/model.onnx | 8b74fc4b120c681d10f89231f66cadf8ac53cc30eba939c76254dc651bfbd12c |

SHA-256 защищённых входов и результатов 1A–2A:

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
- `reports/ner_v2/research_run2/input_abstracts.jsonl`: `684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1`
- `reports/re_v2/relex_comparison/threshold035/e3_base/entities.jsonl`: `19039d87119130dcc37d2c472a7f2242c6a3ad5059e46a1ff431e6fa37e3b6ba`
- `reports/re_v2/relex_comparison/threshold035/e3_base/relations.jsonl`: `bbb4e602abccedcc776b59cfd526f8ffe7ffb46bdea7f7dc3c38df33174dd099`
