# Концепты из первых 10 000 строк запроса

Срез содержит 19 узлов и 40 memberships. Для каждой строки отдельно указано полное состояние соответствующего класса в Corpus100. E5 cosine рассчитан по всем парам названий участников. Качественная оценка предварительная. Она основана на видимых названиях и типах участников.

| Концепт и ID | Участники / документы в срезе | Участники / документы в полном графе | E5 mean / min в срезе | E5 mean / min в полном графе | Оценка однородности |
|---|---:|---:|---:|---:|---|
| breast cancer · `concept:2g:fc3ff0578073ca80dc3e2d70` | 1 / 1 | 4 / 2 | — | 1.000 / 1.000 | Участники имеют одинаковое название и относятся к одному классу заболевания. |
| classification · `concept:2g:980757100f7ec986a67ce8b0` | 2 / 2 | 5 / 2 | 0.866 / 0.866 | 0.918 / 0.866 | Общий класс включает общий термин и специализированные задачи классификации заболеваний. |
| Clinical note · `concept:2g:1c1098b968dee15acecbda8e` | 3 / 2 | 9 / 4 | 1.000 / 1.000 | 0.990 / 0.955 | Текстовые записи одного класса, включая аннотированные clinical notes. |
| clinician · `concept:2g:354759ae4e278f53d0a57237` | 2 / 1 | 3 / 2 | 0.978 / 0.978 | 0.985 / 0.978 | Одна профессиональная роль в единственном и множественном числе. |
| demographic · `concept:2g:2bd5657c9a33fdf63edba0e9` | 2 / 2 | 3 / 3 | 0.919 / 0.919 | 0.937 / 0.919 | Лексически близкий класс. Прилагательное, признаки и совокупность демографических сведений требуют различения. |
| Hospital · `concept:2g:811df7b2ef435e127cb8eacd` | 2 / 2 | 12 / 9 | 0.854 / 0.854 | 0.881 / 0.811 | Общий институциональный класс. Включает конкретные учреждения и общее название. |
| Hospital readmission · `concept:2g:28d46d40d1aa400b6cbc214a` | 3 / 3 | 11 / 7 | 0.997 / 0.995 | 0.963 / 0.910 | Один класс события с уточнениями срока, причины и плановости. |
| hospital stay · `concept:2g:ab70ed8a62c6220afddc2a7c` | 1 / 1 | 2 / 2 | — | 1.000 / 1.000 | Лексически однородный класс. Длительности отдельных госпитализаций сохраняют свой контекст. |
| Named Entity Recognition · `concept:2g:52b187382441cf8af13e2d43` | 1 / 1 | 4 / 3 | — | 0.822 / 0.737 | Класс задачи NER с предметными уточнениями. Пересекается с отдельным узлом NER. |
| NER · `concept:2g:84f1590cc2adafedd13f7c33` | 1 / 1 | 4 / 3 | — | 0.913 / 0.864 | Аббревиатура задачи и конкретные варианты. Кандидат на проверку полноформатного алиаса Named Entity Recognition. |
| patient data · `concept:2g1:a6f4cf7dca97e3aaf05f55e9` | 1 / 1 | 3 / 2 | — | 0.943 / 0.914 | Отдельные виды данных о пациентах. Родовой класс без переноса происхождения и свойств данных. |
| patient population · `concept:2g:ab037c09d7df9a3c7a164a9a` | 10 / 7 | 35 / 20 | 0.931 / 0.876 | 0.961 / 0.845 | Родовой клинический класс с разнородными когортами. Группы пациентов остаются самостоятельными. |
| prediction · `concept:2g:fa24800ef4cb5706552ad634` | 1 / 1 | 4 / 2 | — | 0.879 / 0.851 | Широкий класс прогнозирования. Целевые переменные и способы прогноза различаются. |
| readmission · `concept:2g:4ebea4554f71ad46927fa753` | 3 / 3 | 17 / 8 | 0.997 / 0.995 | 0.951 / 0.880 | Родовой класс события. Пересекается с Hospital readmission. |
| receiver operating characteristic curve · `concept:2g:9559c892c1ea51a31bf259bd` | 1 / 1 | 3 / 2 | — | 0.840 / 0.781 | Смешение названия ROC-кривой и AUC как измеряемой площади. Требуется проверка и потенциальное разведение классов. |
| recognition · `concept:2g:e58e8dc1c7a323a91f2e8444` | 3 / 3 | 7 / 4 | 0.802 / 0.753 | 0.839 / 0.737 | Широкий класс распознавания. Наличие NER и специализированных вариантов требует проверки границ. |
| research · `concept:2g1:5f8368d643022c015855422d` | 1 / 1 | 5 / 3 | — | 0.927 / 0.875 | Широкий класс исследовательской деятельности с несколькими специализациями. |
| time · `concept:2g:c894b67229a62ab939dfc675` | 1 / 1 | 2 / 2 | — | 1.000 / 1.000 | Общий термин. Одинаковая поверхность не разрешает смысл периода, момента или длительности. |
| transformer · `concept:2g:75a877be52ede94e187f567d` | 1 / 1 | 2 / 2 | — | 1.000 / 1.000 | Одинаковая форма Transformers. Контекстную интерпретацию архитектуры нужно подтвердить по каждому упоминанию. |

## Участники и документы

Столбец «В срезе» отмечает membership, возвращённый среди первых 10 000 строк. Document ID позволяет восстановить источник. Разные Document ID одной группы подтверждают междокументность.

### breast cancer

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4403989939` | breast cancer | CONDITION |
| да | `W4416217660` | breast cancer | CONDITION |
| — | `W4416217660` | breast cancer | CONDITION |
| — | `W4416217660` | breast cancer | CONDITION |

### classification

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4411629692` | disease classification | TASK |
| да | `W4411629692` | thorax disease classification | TASK |
| — | `W4411629692` | thorax disease classification | TASK |
| да | `W4412636006` | classification | PROCESS |
| — | `W4412636006` | classification | PROCESS |

### Clinical note

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| да | `W4403989939` | clinical notes | DATA_RESOURCE, OBJECT |
| да | `W4403989939` | clinical notes | DATA_RESOURCE, OBJECT |
| — | `W4403989939` | clinical notes | DATA_RESOURCE, OBJECT |
| — | `W4412617629` | annotated clinical notes | DATA_RESOURCE |
| да | `W4412617629` | clinical notes | OBJECT |
| — | `W4412617629` | clinical notes | OBJECT |
| — | `W4413112971` | Clinical notes | OBJECT |
| — | `W4413112971` | clinical notes | OBJECT |
| — | `W4413924632` | clinical notes | DATA_RESOURCE |

### clinician

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| да | `W4414892439` | clinician | ACTOR |
| да | `W4414892439` | clinicians | ACTOR |
| — | `W4415291590` | clinicians | ACTOR |

### demographic

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| да | `W4411531688` | patient demographics | ATTRIBUTE |
| да | `W4413924632` | demographic | ATTRIBUTE |
| — | `W4414167173` | demographics | ATTRIBUTE |

### Hospital

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| да | `W4377292807` | hospital | ATTRIBUTE, DATA_RESOURCE |
| — | `W4377292807` | hospital | DATA_RESOURCE |
| — | `W4377292807` | hospital | ATTRIBUTE, DATA_RESOURCE |
| — | `W4386832365` | hospitals | ACTOR |
| — | `W4402926611` | Hospital | ACTOR |
| да | `W4412424119` | Tongji Hospital | ACTOR |
| — | `W4412617629` | hospital | ACTOR |
| — | `W4412743408` | Beijing Children’s Hospital | ACTOR |
| — | `W4412743408` | Chongqing Children’s Hospital | ACTOR |
| — | `W4414892439` | major hospital | ACTOR |
| — | `W4415291590` | Shanghai Chest Hospital | ACTOR |
| — | `W4416807702` | Second Xiangya Hospital | DATA_RESOURCE |

### Hospital readmission

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4290467241` | hospital readmissions | CONDITION, PROCESS |
| да | `W4312090231` | hospital readmission | PROCESS |
| да | `W4377292807` | hospital readmission | CONDITION |
| — | `W4386832365` | HF hospital readmission | CONDITION |
| — | `W4386832365` | hospital readmission | CONDITION |
| — | `W4386832365` | hospital readmission | CONDITION |
| — | `W4402900266` | Early hospital readmission | CONDITION |
| — | `W4402900266` | unplanned hospital readmissions | CONDITION |
| — | `W4408725062` | hospital readmission | CONDITION |
| да | `W4413112971` | hospital readmissions | CONDITION |
| — | `W4413112971` | unplanned hospital readmissions | CONDITION |

### hospital stay

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| да | `W4404832627` | hospital stay | CONDITION |
| — | `W4412743408` | hospital stay | CONDITION |

### Named Entity Recognition

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4401237319` | Biomedical named entity recognition | CONCEPT |
| — | `W4412617629` | medical named entity recognition | TASK |
| да | `W4416217660` | NER | CONCEPT, MECHANISM, METHOD |
| — | `W4416217660` | NER | CONCEPT, METHOD |

### NER

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4401237319` | multi-class NER | CONCEPT, TASK |
| — | `W4412617629` | medical named entity recognition (NER) | TASK |
| да | `W4416217660` | NER | CONCEPT, MECHANISM, METHOD |
| — | `W4416217660` | NER | CONCEPT, METHOD |

### patient data

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| да | `W4377292807` | patient data | DATA_RESOURCE |
| — | `W4377292807` | remotely-monitored patient data | DATA_RESOURCE |
| — | `W4414007093` | patient data | DATA_RESOURCE |

### patient population

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4290467241` | patients | ACTOR |
| — | `W4290467241` | patients | ACTOR |
| — | `W4290467241` | patients | ACTOR |
| — | `W4312090231` | patients | OBJECT |
| — | `W4312090231` | patients | ACTOR, OBJECT |
| — | `W4377292807` | patients | ACTOR |
| — | `W4386832365` | patients | ACTOR |
| да | `W4402900266` | Intensive Care Unit patients | ACTOR |
| да | `W4402900266` | patients | ACTOR |
| — | `W4402900266` | patients | ACTOR |
| — | `W4403989939` | patients | ACTOR |
| да | `W4404832627` | patients | ACTOR |
| да | `W4405257212` | patients | ACTOR |
| — | `W4407519249` | patients | ACTOR |
| да | `W4408363098` | patients | ACTOR |
| да | `W4408725062` | HNSCC patients | ACTOR |
| — | `W4408725062` | patients | ACTOR |
| да | `W4412017072` | cancer patients | ACTOR, OBJECT |
| да | `W4412017072` | cancer patients | ACTOR, OBJECT |
| — | `W4412017072` | cancer patients | ACTOR, OBJECT |
| — | `W4412424119` | patients | ACTOR |
| — | `W4412424119` | patients | ACTOR |
| — | `W4412743408` | hospitalized pediatric patients | ACTOR |
| да | `W4412743408` | pediatric patients | ACTOR |
| да | `W4412743408` | pediatric patients | ACTOR |
| — | `W4413112971` | patient | ACTOR, OBJECT |
| — | `W4413112971` | patients | ACTOR |
| — | `W4413257864` | patients | ACTOR |
| — | `W4414007093` | patients | ACTOR, OBJECT |
| — | `W4414007093` | patients | ACTOR, OBJECT |
| — | `W4414167173` | UK surgical patients | ACTOR |
| — | `W4414167173` | patients | ACTOR, OBJECT |
| — | `W4414167173` | patients | OBJECT |
| — | `W4415291590` | patients | ACTOR |
| — | `W7115171724` | patients | ACTOR |

### prediction

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4413924632` | Hospital readmission prediction | TASK |
| — | `W4413924632` | hospital readmission prediction | PROCESS |
| — | `W4413924632` | predictions | ATTRIBUTE |
| да | `W4416807702` | early risk prediction | TASK |

### readmission

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4290467241` | hospital readmissions | CONDITION, PROCESS |
| — | `W4290467241` | readmissions | PROCESS |
| да | `W4312090231` | hospital readmission | PROCESS |
| — | `W4312090231` | readmission | PROCESS |
| да | `W4377292807` | hospital readmission | CONDITION |
| — | `W4386832365` | HF hospital readmission | CONDITION |
| — | `W4386832365` | hospital readmission | CONDITION |
| — | `W4386832365` | hospital readmission | CONDITION |
| — | `W4402900266` | Early hospital readmission | CONDITION |
| — | `W4402900266` | early readmission | CONDITION |
| — | `W4402900266` | unplanned hospital readmissions | CONDITION |
| — | `W4405257212` | readmissions | PROCESS |
| — | `W4408725062` | hospital readmission | CONDITION |
| — | `W4408725062` | readmission | CONDITION, PROCESS |
| — | `W4408725062` | readmission | CONDITION |
| да | `W4413112971` | hospital readmissions | CONDITION |
| — | `W4413112971` | unplanned hospital readmissions | CONDITION |

### receiver operating characteristic curve

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4403759266` | AUC-ROC | ATTRIBUTE |
| — | `W4408155557` | area under the receiver operating characteristic curve | ATTRIBUTE |
| да | `W4408155557` | receiver operating characteristic curve | ATTRIBUTE |

### recognition

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4401237319` | Biomedical named entity recognition | CONCEPT |
| да | `W4403989939` | entity recognition | CONCEPT, PROCESS |
| — | `W4412617629` | medical named entity recognition | TASK |
| да | `W4412617629` | recognition | TASK |
| да | `W4416217660` | NER | CONCEPT, MECHANISM, METHOD |
| — | `W4416217660` | NER | CONCEPT, METHOD |
| — | `W4416217660` | medical entity recognition | CONCEPT |

### research

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4401237319` | drug development research | PROCESS |
| да | `W4401237319` | medical research | PROCESS |
| — | `W4403989939` | cancer research | TASK |
| — | `W4403989939` | cancer research | TASK |
| — | `W4416217660` | research | PROCESS |

### time

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| да | `W4414892439` | time | ATTRIBUTE |
| — | `W4416217660` | time | ATTRIBUTE |

### transformer

| В срезе | Документ | Сущность | Типы |
|---|---|---|---|
| — | `W4412617629` | Transformers | ACTOR |
| да | `W4416217660` | Transformers | ACTOR |
