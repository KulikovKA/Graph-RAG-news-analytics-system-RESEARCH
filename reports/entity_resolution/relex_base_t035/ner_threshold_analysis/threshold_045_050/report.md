# Уточнение NER thresholds: 0.45–0.50, шаг 0.01

Продолжение этапа 2F.0 по запросу пользователя. Исходный coarse sweep и его отчёт сохранены без изменений. Используются те же 30 документов, 1122 canonical entities из 1211 source entities, 1760 RELEX_REL и 685 готовых SIMILAR_TO (194 междокументных). Исходный NER cutoff — 0.35.

## Вывод

**В заданном диапазоне предпочтителен осторожный кандидат 0.45 (max member): 925 вершин, удалено 197 (17.56%), сохраняются 1352 RELEX_REL, 510 SIMILAR_TO и 126 междокументных связей.** По сравнению с уже рассчитанным 0.40 это ещё 87 удалённых вершин и потеря 45 междокументных связей; поэтому даже 0.45 не является безусловно безопасным улучшением.

**0.46 — промежуточный кандидат; 0.47 — более агрессивный.** При 0.46 остаётся 905 вершин, сохраняется полный span `Pearson product-moment correlation coefficient` (0.464365). При 0.47 остаётся 886 вершин и те же 121 междокументная связь, но этот span, `endophenotypes` и `average recall (AR)` уже удалены. При переходе 0.46 → 0.47 удаляются 19 вершин и 62 фактических отношения: отсутствие потерь междокументных рёбер не означает отсутствие содержательных потерь.

**У 0.48 наблюдается структурный перелом**, но его нельзя использовать как самостоятельное доказательство качества порога: один из удалённых междокументных мостов связывает общие `paper`. Пороги 0.48–0.50 дают дальнейшее сокращение ценой потерь научных названий; преимущества по качеству извлечений не доказаны.

## Методика

Рассчитаны ровно шесть порогов 0.45, 0.46, 0.47, 0.48, 0.49, 0.50 для двух прежних стратегий. Representative — confidence source entity с canonical ID; max member — максимум source entity confidence в существующей группе. `V={v:score(v)>=t}`; исходное ребро сохраняется тогда и только тогда, когда присутствуют оба конца. Модели, Entity Merge, пересчёт cosine и независимый от вершин отбор рёбер не выполнялись; Neo4j не открывался.

Все сущности, исходные confidence, типы, provenance и контексты прочитаны из прежних файлов. Для каждого source entity confidence остаётся максимумом его исходных typed mention scores. В слабых компонентах игнорируются направления, учитываются изолированные вершины; документы и MENTIONS не входят. Петля даёт ненулевую степень. Среднее рассчитано на все 30 документов. Многотипная вершина учитывается в каждом её исходном типе.

## Полное сравнение

### max_member

| t | Вершин | На док. | Удалено | RELEX_REL | SIMILAR_TO | Междок. | Изоляты (%) | WCC | Max WCC | Док. max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 925 | 30.83 | 197 | 1352 | 510 | 126 | 183 (19.78%) | 238 | 382 | 17 |
| 0.46 | 905 | 30.17 | 217 | 1307 | 492 | 121 | 175 (19.34%) | 230 | 378 | 17 |
| 0.47 | 886 | 29.53 | 236 | 1245 | 488 | 121 | 172 (19.41%) | 227 | 366 | 17 |
| 0.48 | 863 | 28.77 | 259 | 1223 | 474 | 119 | 167 (19.35%) | 221 | 301 | 13 |
| 0.49 | 849 | 28.30 | 273 | 1187 | 460 | 114 | 168 (19.79%) | 222 | 294 | 13 |
| 0.50 | 830 | 27.67 | 292 | 1134 | 452 | 112 | 171 (20.60%) | 223 | 288 | 13 |

| Тип | 0.45 | 0.46 | 0.47 | 0.48 | 0.49 | 0.50 |
|---|---|---|---|---|---|---|
| ACTOR | 109 | 108 | 107 | 105 | 105 | 105 |
| ATTRIBUTE | 155 | 153 | 145 | 144 | 142 | 139 |
| CONCEPT | 92 | 87 | 86 | 77 | 73 | 71 |
| CONDITION | 78 | 78 | 76 | 74 | 74 | 74 |
| DATA_RESOURCE | 54 | 54 | 53 | 51 | 50 | 48 |
| MECHANISM | 61 | 58 | 57 | 56 | 56 | 54 |
| METHOD | 57 | 55 | 54 | 53 | 53 | 51 |
| OBJECT | 253 | 249 | 247 | 242 | 241 | 238 |
| PROCESS | 95 | 92 | 90 | 87 | 85 | 84 |
| RESULT | 43 | 42 | 41 | 41 | 40 | 38 |
| TASK | 38 | 37 | 36 | 35 | 32 | 29 |

### representative

| t | Вершин | На док. | Удалено | RELEX_REL | SIMILAR_TO | Междок. | Изоляты (%) | WCC | Max WCC | Док. max |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 919 | 30.63 | 203 | 1338 | 507 | 126 | 182 (19.80%) | 238 | 381 | 17 |
| 0.46 | 899 | 29.97 | 223 | 1294 | 489 | 121 | 174 (19.35%) | 230 | 377 | 17 |
| 0.47 | 880 | 29.33 | 242 | 1233 | 486 | 121 | 170 (19.32%) | 226 | 366 | 17 |
| 0.48 | 858 | 28.60 | 264 | 1212 | 472 | 119 | 165 (19.23%) | 220 | 301 | 13 |
| 0.49 | 843 | 28.10 | 279 | 1169 | 457 | 113 | 167 (19.81%) | 222 | 294 | 13 |
| 0.50 | 825 | 27.50 | 297 | 1123 | 450 | 112 | 169 (20.48%) | 222 | 288 | 13 |

| Тип | 0.45 | 0.46 | 0.47 | 0.48 | 0.49 | 0.50 |
|---|---|---|---|---|---|---|
| ACTOR | 109 | 108 | 107 | 105 | 105 | 105 |
| ATTRIBUTE | 155 | 153 | 145 | 144 | 142 | 139 |
| CONCEPT | 92 | 87 | 86 | 77 | 73 | 71 |
| CONDITION | 77 | 77 | 75 | 74 | 74 | 74 |
| DATA_RESOURCE | 52 | 52 | 51 | 49 | 48 | 46 |
| MECHANISM | 61 | 58 | 57 | 56 | 56 | 54 |
| METHOD | 57 | 55 | 54 | 53 | 53 | 51 |
| OBJECT | 250 | 246 | 244 | 239 | 237 | 235 |
| PROCESS | 95 | 92 | 90 | 87 | 85 | 84 |
| RESULT | 43 | 42 | 41 | 41 | 40 | 38 |
| TASK | 38 | 37 | 36 | 35 | 32 | 29 |

Междокументных RELEX_REL нет во всех вариантах: междокументный столбец содержит только SIMILAR_TO. Полные проценты сохранения и все дополнительные счётчики — [comparison.csv](comparison.csv). Все 30 документов сохраняют хотя бы одну вершину при каждом из шести порогов.

## Что меняется на каждом шаге

| Переход | Удалено вершин | Потеря RELEX_REL | Потеря SIMILAR_TO | Потеря междок. | Убыль max WCC | Убыль док. max |
|---|---|---|---|---|---|---|
| 0.45 → 0.46 | 20 | 45 | 18 | 5 | 4 | 0 |
| 0.46 → 0.47 | 19 | 62 | 4 | 0 | 12 | 0 |
| 0.47 → 0.48 | 23 | 22 | 14 | 2 | 65 | 4 |
| 0.48 → 0.49 | 14 | 36 | 14 | 5 | 7 | 0 |
| 0.49 → 0.50 | 19 | 53 | 8 | 2 | 6 | 0 |

Разности для обеих стратегий — [step_deltas.csv](step_deltas.csv). Все конкретные удалённые вершины каждого шага с confidence и контекстами — [marginal_removed_entities.csv](marginal_removed_entities.csv).

### Разрыв при 0.47 → 0.48

Из 366 вершин прежней крупнейшей компоненты удаляются 6; оставшиеся разделяются на компоненты 301 вершина / 13 документов и 59 вершин / 4 документа. Ещё 17 из 23 удалённых на этом шаге вершин находились в других компонентах. Четыре документа переходят во вторую компоненту: W2317432785 (Physics), W2364973104 (Physics), W2906067930 (Economics), W4381572413 (History). Исчезновение документов из крупнейшей компоненты не означает удаления самих документов или всех их сущностей.

На этом шаге теряются две междокументные пары:

| Левая вершина | Документ | NER confidence | Правая вершина | Документ | NER confidence |
|---|---|---|---|---|---|
| paper | W2991563224 | 0.475494 | paper | W2906067930 | 0.692119 |
| genetics | W1995028339 | 0.551332 | genes | W3138328568 | 0.474246 |

Полные ID, исходные similarity и контексты сохранены в `cross_document_losses_047_048.csv`; состав отделившихся частей — в `largest_component_split_047_048.csv` и `components.csv`. Связь `paper ↔ paper` соединяет общее самоупоминание публикаций разных дисциплин. Сохранение такого пути повышает формальную связность, но его содержательная полезность не подтверждена. Здесь не выполнялись дополнительные симуляции удаления отдельных вершин или фильтрация качества рёбер.

## patients и results

Весь диапазон не различает эти группы: при каждом пороге и обеих стратегиях остаются 14 из 16 canonical `patients` и 6 из 8 casefold `results`. Два заголовка `RESULTS` с 0.446378 и 0.449832 уже удалены при 0.45; третий (0.564522) остаётся даже при 0.50. Пять общих ссылок `results` остаются, включая 0.522291–0.892064. Наличие реальных участников клинической когорты у `patients` не делает их бесполезными; общий surface name и ошибка извлечения не тождественны.

Повышение 0.45 → 0.50 не даёт дополнительных удалений точных названий `patients` и `results`, но удаляет 95 других canonical entities. Это не означает, что все 95 полезны или все бесполезны; ниже приведены контекстные контрпримеры.

## Содержательные и малоинформативные потери

Разбор иллюстративный, по локальным контекстам, без внешней модели и без оценки precision/recall. Удаление конкретного названия не означает полного исчезновения концепта: сокращения и другие spans не объединялись.

- **connexin**, `W2039836112`, Biology, **0.450992**. Конкретный белок в описании gap junctions; потеря при 0.46.
  Контекст: «Spinal astrocytes are coupled by connexin (Cx) gap junctions and express pannexin 1 (Px1) and purinergic receptors. Fibroblast growth factor 1 (FGF-1), which is »

- **periodic wave solutions**, `W3081936674`, Physics, **0.458547**. Перечисленная форма математических решений; потеря при 0.46.
  Контекст: «tained solutions are in the form of dark soliton, combined soliton, symmetrical Lucas sine, Lucas cosine functions, and periodic wave solutions. We present also some graphical simulations of the solutions corresponding to values of parameters which leads to a bet»

- **time varying parameter formulation**, `W2078771716`, Economics, **0.454852**. Конкретная формулировка модели в исследовании конвергенции; потеря при 0.46.
  Контекст: «se two considerations lead us to a preferred definition of convergence and a proposed framework for testing, based on a time varying parameter formulation and estimation by Kalman filter techniques. We consider the relationship between our proposed framework and other metho»

- **Pearson product-moment correlation coefficient**, `W1979727947`, Medicine, **0.464365**. Конкретный коэффициент для сравнения измерений; потеря при 0.47.
  Контекст: «ing set for model building (733 cases, 748 controls) and a test set for model assessment (765 cases, 747 controls). The Pearson product-moment correlation coefficient (r) was used to compare the MD measurements by Cumulus and our automated measure, which mimics Cumulus. The likelihood »

- **endophenotypes**, `W2076293586`, Biology, **0.469926**. Предмет исследования copy number variants; потеря при 0.47.
  Контекст: «biased approach we are currently focusing on the discovery of copy number variants (CNVs) and their implications in the endophenotypes of these animals. Two array approaches were used to reveal regions of CNVs that differ between the extremes of the resp»

- **average recall (AR)**, `W4389076493`, Computer Science / Machine Learning, **0.463260**. Названная метрика оценки detection; потеря при 0.47.
  Контекст: «nd show obvious improvement over other competitors in terms of two performance indicators of average precision (AP) and average recall (AR). The code will be available at https://github.com/kbzhang0505/ARPN.»

- **nonlinear optics**, `W3081936674`, Physics, **0.478372**. Научная область, из которой возникает исследуемая система; потеря при 0.48.
  Контекст: «Abstract In this work, we consider a (2+1) dimensional nonlinear Schrödinger system which appears in the theory of nonlinear optics and describe transmission of the optical pulses in optical fibers. We attain certain special type traveling wave soluti»

- **co-investment**, `W4285525942`, Economics, **0.478441**. Названный финансовый инструмент; потеря при 0.48.
  Контекст: «The article analyzes the main financial instruments of green banking: green bank lending, green mortgages, green bonds, co-investment of green banks, securitization of green bank assets. The analysis of characteristic features and peculiarities of appli»

- **synthetic biology**, `W3138328568`, Biology, **0.479533**. Область, обеспечивающая engineering microbial cell factories; потеря при 0.48.
  Контекст: «esis limit the sustainability of industrial limonene production. Recently, the advancement of metabolic engineering and synthetic biology has facilitated the engineering of microbes into microbial cell factories for producing limonene. However, the deleteri»

- **lunar regolith's thickness**, `W2096307598`, Physics, **0.485732**. Измеряемая характеристика лунного реголита; потеря при 0.49.
  Контекст: « to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lunar surface.»

- **fare optimization**, `W4309030044`, Economics, **0.482791**. Непосредственная тема транспортной модели; потеря при 0.49.
  Контекст: «In the business of intermodal passenger transport, fare optimization of intermodal products has significant effects on corporate revenue and passenger travel convenience. This study takes »

- **diatom phylogeny**, `W1995028339`, Biology, **0.497369**. Предмет обсуждения эволюции пластид; потеря при 0.50.
  Контекст: «the heterokont lineage. Variable plastid genome sizes and extensive genome rearrangements have been observed across the diatom phylogeny, but little is known about plastid genome evolution within order- or family-level clades. The Thalassiosirales is one o»

- **magnetic resonance imaging**, `W4230116451`, Computer Science / Machine Learning, **0.493728**. Конкретная технология исследованного case study; потеря при 0.50.
  Контекст: «g steps of a distributed cloud e-health system by the use of graphics processing units (GPUs). In the case study of the magnetic resonance imaging (MRI), for improving the quality of denoising and helping the real-time diagnosis, we have implemented a GPU parallel a»

- **cardiac implantable electronic device (CIED)**, `W4391351014`, Medicine, **0.497576**. Полное название устройства исследования; потеря при 0.50, другие CIED-spans могут остаться.
  Контекст: «(1) Background: The prevalence of cancer patients relying on cardiac implantable electronic device (CIED) is steadily rising. The aim of this study was to evaluate RT-related malfunctions of CIEDs. (2) Methods: We retrospecti»

- **retrospectively**, `W4391351014`, Medicine, **0.452601**. Наречие из описания метода; слабая самостоятельная вершина, удаляется при 0.46.
  Контекст: «vice (CIED) is steadily rising. The aim of this study was to evaluate RT-related malfunctions of CIEDs. (2) Methods: We retrospectively analyze sixteen patients with esophageal cancer who were treated with radiotherapy between 2012 and 2022 at the Univers»

- **green**, `W4285525942`, Economics, **0.459313**. Одиночный модификатор, слабое самостоятельное название; удаляется при 0.46.
  Контекст: «ors and developers with green banks, standardizes the conditions of such cooperation, and increases the transparency of green banking institutions. The article analyzes the main financial instruments of green banking: green bank lending, green m»

- **paper**, `W2991563224`, Computer Science / Machine Learning, **0.475494**. Самоупоминание статьи; удаляется при 0.48 и одновременно теряет формальный междокументный мост.
  Контекст: «In this paper, we propose a new approach to improve the protocol LEACH performance. It has been accomplished by increasing the energy»

- **clear**, `W3101922761`, Medicine, **0.490462**. В контексте прозрачность роговичного трансплантата: слабое самостоятельное имя, но клинический смысл есть; потеря при 0.50.
  Контекст: « 45%, 37%, 32%, and 26% of eyes at 1, 2, 3, 4, 5, 7, and 10 years, respectively. Only 7 corneal grafts (17.5%) remained clear at last follow-up. CONCLUSIONS: A GDD can successfully control intractable glaucoma even after a very long period of ti»

## По дисциплинам и документам

| Дисциплина | Baseline | 0.45 | 0.46 | 0.47 | 0.48 | 0.49 | 0.50 |
|---|---|---|---|---|---|---|---|
| Biology | 259 | 216 | 209 | 205 | 196 | 190 | 186 |
| Computer Science / Machine Learning | 154 | 117 | 116 | 113 | 110 | 108 | 106 |
| Economics | 171 | 141 | 135 | 134 | 131 | 128 | 125 |
| History | 116 | 89 | 89 | 89 | 86 | 84 | 84 |
| Medicine | 238 | 204 | 203 | 194 | 192 | 192 | 186 |
| Physics | 184 | 158 | 153 | 151 | 148 | 147 | 143 |

| Документ | Дисциплина | Baseline | 0.45 | 0.46 | 0.47 | 0.48 | 0.49 | 0.50 |
|---|---|---|---|---|---|---|---|---|
| W1979727947 | Medicine | 37 | 32 | 32 | 29 | 29 | 29 | 28 |
| W1995028339 | Biology | 60 | 49 | 48 | 47 | 44 | 43 | 42 |
| W2039836112 | Biology | 59 | 50 | 46 | 44 | 43 | 43 | 42 |
| W2076293586 | Biology | 47 | 41 | 39 | 38 | 37 | 36 | 34 |
| W2078771716 | Economics | 37 | 35 | 34 | 33 | 33 | 33 | 33 |
| W2084709926 | Physics | 36 | 30 | 30 | 30 | 30 | 30 | 30 |
| W2096307598 | Physics | 37 | 32 | 32 | 32 | 32 | 31 | 29 |
| W2130236240 | Economics | 13 | 9 | 9 | 9 | 8 | 7 | 7 |
| W2317432785 | Physics | 45 | 41 | 40 | 40 | 39 | 39 | 38 |
| W2326480042 | History | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| W2364973104 | Physics | 40 | 35 | 33 | 31 | 31 | 31 | 30 |
| W2394633461 | History | 37 | 31 | 31 | 31 | 29 | 28 | 28 |
| W2761455282 | Medicine | 70 | 60 | 60 | 57 | 57 | 57 | 56 |
| W2791492120 | Medicine | 40 | 37 | 37 | 36 | 35 | 35 | 34 |
| W2906067930 | Economics | 40 | 33 | 32 | 32 | 32 | 32 | 30 |
| W2991563224 | Computer Science / Machine Learning | 18 | 13 | 13 | 13 | 12 | 12 | 12 |
| W3027457624 | Computer Science / Machine Learning | 29 | 26 | 26 | 26 | 26 | 26 | 26 |
| W3081936674 | Physics | 26 | 20 | 18 | 18 | 16 | 16 | 16 |
| W3101922761 | Medicine | 55 | 49 | 49 | 49 | 49 | 49 | 47 |
| W3123056951 | Biology | 38 | 32 | 32 | 32 | 31 | 30 | 30 |
| W3138328568 | Biology | 55 | 44 | 44 | 44 | 41 | 38 | 38 |
| W4230116451 | Computer Science / Machine Learning | 24 | 18 | 18 | 18 | 18 | 18 | 16 |
| W4234628002 | Computer Science / Machine Learning | 32 | 26 | 25 | 25 | 25 | 25 | 25 |
| W4285525942 | Economics | 54 | 44 | 41 | 41 | 40 | 39 | 38 |
| W4309030044 | Economics | 27 | 20 | 19 | 19 | 18 | 17 | 17 |
| W4312422299 | History | 23 | 13 | 13 | 13 | 13 | 12 | 12 |
| W4381572413 | History | 21 | 17 | 17 | 17 | 16 | 16 | 16 |
| W4389076493 | Computer Science / Machine Learning | 51 | 34 | 34 | 31 | 29 | 27 | 27 |
| W4391351014 | Medicine | 36 | 26 | 25 | 23 | 22 | 22 | 21 |
| W7215137876 | History | 25 | 18 | 18 | 18 | 18 | 18 | 18 |

Таблицы выше используют max member; полные разрезы для обеих стратегий — `retention_by_discipline.csv` и `retention_by_document.csv`. Распределения исходных confidence не меняются; для удобства сохранены повторно тем же кодом в `distribution_summary.csv` и `confidence_histogram.csv`.

## Примеры для каждого варианта

По два ближайших к границе сохранённых и удалённых примера. Отбор детерминированный, не репрезентативная выборка качества. Confidence и контекст относятся к текущей стратегии. Полные ID и дополнительный крайний пример на группу — в `examples.csv`.

### max_member, t=0.45

- Удалена: **RESULTS**, **0.449832**, `W1979727947`, `e:d4a0d3d5c1ae…`.
  Контекст: «sion models for breast cancer risk, which included our measure capturing additional information in mammographic images. RESULTS: We observed a high correlation between the Cumulus measure and our measure mimicking Cumulus (r = 0.884; 95% CI, 0.872»

- Удалена: **accessibility**, **0.449302**, `W4381572413`, `e:b234a364591f…`.
  Контекст: «al history, because it became then clear that “domestic” conservation could not give any guarantees of preservation and accessibility to these unique sources. The bibliographic selection offered by this essay well illuminates how this approach can be em»

- Сохранена: **action**, **0.450367**, `W2039836112`, `e:c56dd127b2cb…`.
  Контекст: « given 1 h after FGF-1. We infer that ATP is initially released from vesicles and then it mediates continued release by action on P2X(7) receptors and opening of HCs. These changes in HCs and gap junction channels may promote inflammation and dep»

- Сохранена: **volume**, **0.450751**, `W4285525942`, `e:0ed666546827…`.
  Контекст: « green mortgages as one of the priority instruments of green banking, and emphasizes the objective need to increase the volume of green bonds. The joint investment of green banks and its essence has been described in the article. The author belie»

### max_member, t=0.46

- Удалена: **HSR**, **0.459981**, `W4309030044`, `e:95b9531443da…`.
  Контекст: «between aviation and HSR, and a reasonable pricing strategy is more likely to motivate cooperation between aviation and HSR.»

- Удалена: **lightening protection device**, **0.459879**, `W2364973104`, `e:da1b59403aaf…`.
  Контекст: «s lighting protection module device is of high reliability and safety,and can overcome the barriers of the conventional lightening protection device,and helps to improve the indirect lighting protection capability and anti-weather fighting ability of the military powe»

- Сохранена: **complete clinical and radiographic evaluations**, **0.461519**, `W2761455282`, `e:23377d0ce48e…`.
  Контекст: «d 2015. A preoperative medical evaluation was performed in all patients to assess the risk of surgery. All patients had complete clinical and radiographic evaluations, and the outcome was scored using American Knee Society Scores and the Western Ontario and McMaster Universities Arthri»

- Сохранена: **quality of proposal boxes**, **0.461849**, `W4389076493`, `e:147675c1168c…`.
  Контекст: «ntional RPNs exhibit two key weaknesses in the majority of existing two-stage object detection approaches. Firstly, the quality of proposal boxes generated by the RPN is heavily reliant on rich feature representations extracted from the FPN backbone. Secondly, the »

### max_member, t=0.47

- Удалена: **systemic evaluation**, **0.469956**, `W4391351014`, `e:130f28264bad…`.
  Контекст: « who were treated with radiotherapy between 2012 and 2022 at the University Hospital Heidelberg. All patients underwent systemic evaluation including pre-therapeutic cardiological examinations of the CIED functionality and after every single irradiation. (3) »

- Удалена: **endophenotypes**, **0.469926**, `W2076293586`, `e:a6a68d647f4c…`.
  Контекст: «biased approach we are currently focusing on the discovery of copy number variants (CNVs) and their implications in the endophenotypes of these animals. Two array approaches were used to reveal regions of CNVs that differ between the extremes of the resp»

- Сохранена: **crowd counting**, **0.470468**, `W4389076493`, `e:d91175337c8b…`.
  Контекст: «s of interest in a coarse-to-fine manner. On the other hand, a mini-density map, which is inspired by the philosophy of crowd counting, is elaborately designed to adaptively determine the number of region proposal boxes. This approach significantly reduc»

- Сохранена: **papal commemorative**, **0.471277**, `W2394633461`, `e:9c7ef8208104…`.
  Контекст: «r annual medals (sede vacante produced generic annual medals in 1667 and 1691). Annual medals are a unique iteration of papal commemorative medals and they celebrate an important papal achievement from the preceding year. The production of annual medals was a»

### max_member, t=0.48

- Удалена: **synthetic biology**, **0.479533**, `W3138328568`, `e:413c8d1a16fa…`.
  Контекст: «esis limit the sustainability of industrial limonene production. Recently, the advancement of metabolic engineering and synthetic biology has facilitated the engineering of microbes into microbial cell factories for producing limonene. However, the deleteri»

- Удалена: **Fe**, **0.478921**, `W2317432785`, `e:0f6ad991c6b0…`.
  Контекст: « absorption spectroscopy (XANES) and X-ray magnetic circular dichroism (XMCD) at the Zr L2� 3 and K absorption edges of Fe and Co. The local probes of electronic and magnetic properties are fully consistent with macroscopic data in assumption»

- Сохранена: **feature representations**, **0.481755**, `W4389076493`, `e:4069687796e7…`.
  Контекст: «age object detection approaches. Firstly, the quality of proposal boxes generated by the RPN is heavily reliant on rich feature representations extracted from the FPN backbone. Secondly, the fixed number of generated proposal boxes limits adaptability to the dist»

- Сохранена: **characteristic features**, **0.482742**, `W4285525942`, `e:369869d2873f…`.
  Контекст: «nding, green mortgages, green bonds, co-investment of green banks, securitization of green bank assets. The analysis of characteristic features and peculiarities of application of these financial instruments has been carried out. Different estimates of the total »

### max_member, t=0.49

- Удалена: **change**, **0.489547**, `W2394633461`, `e:aa1caa9077f2…`.
  Контекст: «edal, and ending in 1700 at the conclusion of the papacy of Innocent XII (r. 1691–1700), a reign that marked a distinct change in papal politics in advance of the eighteenth century. The article mines a wealth of numismatics images and places it »

- Удалена: **genes**, **0.488093**, `W3138328568`, `e:199ca809b8b6…`.
  Контекст: «lly coexpressed the tolerance-enhancing genes with a limonene synthase gene. Indeed, expression of 5 of the 8 candidate genes enhanced limonene production in Y. lipolytica . Particularly, overexpressing YALI0F19492p led to an 8-fold improvement »

- Сохранена: **SNP-based invariant genomic probes**, **0.490154**, `W2076293586`, `e:9233c1047d44…`.
  Контекст: «es were used to reveal regions of CNVs that differ between the extremes of the respective mouse lines (aCGH, NimbleGen; SNP-based invariant genomic probes, Jackson Laboratories). In total, we identified approximately 100 CNV regions ranging from around 10 kb to 7 Mb. To tes»

- Сохранена: **clear**, **0.490462**, `W3101922761`, `e:508a88858db4…`.
  Контекст: « 45%, 37%, 32%, and 26% of eyes at 1, 2, 3, 4, 5, 7, and 10 years, respectively. Only 7 corneal grafts (17.5%) remained clear at last follow-up. CONCLUSIONS: A GDD can successfully control intractable glaucoma even after a very long period of ti»

### max_member, t=0.50

- Удалена: **food problems**, **0.499482**, `W2906067930`, `e:75e2175ca10c…`.
  Контекст: «g of the condition and mechanism for the Harvest Paradox, but also maybe offers some little reference values in solving food problems of China. Following the methods of Samuelson and Nordhaus, this paper utilizes the supply and demand theorem to study t»

- Удалена: **tendency to increase**, **0.497763**, `W4285525942`, `e:4d561107e82f…`.
  Контекст: « these financial instruments has been carried out. Different estimates of the total value of green bank loans and their tendency to increase and against the background of still low share in total lending have been compared. The article evaluates the structure »

- Сохранена: **composite part**, **0.500281**, `W3027457624`, `e:ac5045ca7350…`.
  Контекст: «unts of counterfeit composite parts showing up in the complex aerospace supply chain, it is absolutely vital to track a composite part throughout its lifecycle-from production to usage and to disposal. Existing barcoding methods are invasive, affect the »

- Сохранена: **ethidium**, **0.500734**, `W2039836112`, `e:fbfa33eb42b0…`.
  Контекст: «spinal astrocytes in culture, induced secretion of ATP, and permeabilized them to relatively large fluorescent tracers [ethidium (Etd) and lucifer yellow (LY)] through "hemichannels" (HCs). HCs can be formed by connexins or pannexins; they can open»

### representative, t=0.45

- Удалена: **RESULTS**, **0.449832**, `W1979727947`, `e:d4a0d3d5c1ae…`.
  Контекст: «sion models for breast cancer risk, which included our measure capturing additional information in mammographic images. RESULTS: We observed a high correlation between the Cumulus measure and our measure mimicking Cumulus (r = 0.884; 95% CI, 0.872»

- Удалена: **accessibility**, **0.449302**, `W4381572413`, `e:b234a364591f…`.
  Контекст: «al history, because it became then clear that “domestic” conservation could not give any guarantees of preservation and accessibility to these unique sources. The bibliographic selection offered by this essay well illuminates how this approach can be em»

- Сохранена: **action**, **0.450367**, `W2039836112`, `e:c56dd127b2cb…`.
  Контекст: « given 1 h after FGF-1. We infer that ATP is initially released from vesicles and then it mediates continued release by action on P2X(7) receptors and opening of HCs. These changes in HCs and gap junction channels may promote inflammation and dep»

- Сохранена: **volume**, **0.450751**, `W4285525942`, `e:0ed666546827…`.
  Контекст: « green mortgages as one of the priority instruments of green banking, and emphasizes the objective need to increase the volume of green bonds. The joint investment of green banks and its essence has been described in the article. The author belie»

### representative, t=0.46

- Удалена: **HSR**, **0.459981**, `W4309030044`, `e:95b9531443da…`.
  Контекст: «between aviation and HSR, and a reasonable pricing strategy is more likely to motivate cooperation between aviation and HSR.»

- Удалена: **lightening protection device**, **0.459879**, `W2364973104`, `e:da1b59403aaf…`.
  Контекст: «s lighting protection module device is of high reliability and safety,and can overcome the barriers of the conventional lightening protection device,and helps to improve the indirect lighting protection capability and anti-weather fighting ability of the military powe»

- Сохранена: **complete clinical and radiographic evaluations**, **0.461519**, `W2761455282`, `e:23377d0ce48e…`.
  Контекст: «d 2015. A preoperative medical evaluation was performed in all patients to assess the risk of surgery. All patients had complete clinical and radiographic evaluations, and the outcome was scored using American Knee Society Scores and the Western Ontario and McMaster Universities Arthri»

- Сохранена: **quality of proposal boxes**, **0.461849**, `W4389076493`, `e:147675c1168c…`.
  Контекст: «ntional RPNs exhibit two key weaknesses in the majority of existing two-stage object detection approaches. Firstly, the quality of proposal boxes generated by the RPN is heavily reliant on rich feature representations extracted from the FPN backbone. Secondly, the »

### representative, t=0.47

- Удалена: **systemic evaluation**, **0.469956**, `W4391351014`, `e:130f28264bad…`.
  Контекст: « who were treated with radiotherapy between 2012 and 2022 at the University Hospital Heidelberg. All patients underwent systemic evaluation including pre-therapeutic cardiological examinations of the CIED functionality and after every single irradiation. (3) »

- Удалена: **endophenotypes**, **0.469926**, `W2076293586`, `e:a6a68d647f4c…`.
  Контекст: «biased approach we are currently focusing on the discovery of copy number variants (CNVs) and their implications in the endophenotypes of these animals. Two array approaches were used to reveal regions of CNVs that differ between the extremes of the resp»

- Сохранена: **crowd counting**, **0.470468**, `W4389076493`, `e:d91175337c8b…`.
  Контекст: «s of interest in a coarse-to-fine manner. On the other hand, a mini-density map, which is inspired by the philosophy of crowd counting, is elaborately designed to adaptively determine the number of region proposal boxes. This approach significantly reduc»

- Сохранена: **papal commemorative**, **0.471277**, `W2394633461`, `e:9c7ef8208104…`.
  Контекст: «r annual medals (sede vacante produced generic annual medals in 1667 and 1691). Annual medals are a unique iteration of papal commemorative medals and they celebrate an important papal achievement from the preceding year. The production of annual medals was a»

### representative, t=0.48

- Удалена: **synthetic biology**, **0.479533**, `W3138328568`, `e:413c8d1a16fa…`.
  Контекст: «esis limit the sustainability of industrial limonene production. Recently, the advancement of metabolic engineering and synthetic biology has facilitated the engineering of microbes into microbial cell factories for producing limonene. However, the deleteri»

- Удалена: **Fe**, **0.478921**, `W2317432785`, `e:0f6ad991c6b0…`.
  Контекст: « absorption spectroscopy (XANES) and X-ray magnetic circular dichroism (XMCD) at the Zr L2� 3 and K absorption edges of Fe and Co. The local probes of electronic and magnetic properties are fully consistent with macroscopic data in assumption»

- Сохранена: **feature representations**, **0.481755**, `W4389076493`, `e:4069687796e7…`.
  Контекст: «age object detection approaches. Firstly, the quality of proposal boxes generated by the RPN is heavily reliant on rich feature representations extracted from the FPN backbone. Secondly, the fixed number of generated proposal boxes limits adaptability to the dist»

- Сохранена: **characteristic features**, **0.482742**, `W4285525942`, `e:369869d2873f…`.
  Контекст: «nding, green mortgages, green bonds, co-investment of green banks, securitization of green bank assets. The analysis of characteristic features and peculiarities of application of these financial instruments has been carried out. Different estimates of the total »

### representative, t=0.49

- Удалена: **change**, **0.489547**, `W2394633461`, `e:aa1caa9077f2…`.
  Контекст: «edal, and ending in 1700 at the conclusion of the papacy of Innocent XII (r. 1691–1700), a reign that marked a distinct change in papal politics in advance of the eighteenth century. The article mines a wealth of numismatics images and places it »

- Удалена: **genes**, **0.488093**, `W3138328568`, `e:199ca809b8b6…`.
  Контекст: «lly coexpressed the tolerance-enhancing genes with a limonene synthase gene. Indeed, expression of 5 of the 8 candidate genes enhanced limonene production in Y. lipolytica . Particularly, overexpressing YALI0F19492p led to an 8-fold improvement »

- Сохранена: **SNP-based invariant genomic probes**, **0.490154**, `W2076293586`, `e:9233c1047d44…`.
  Контекст: «es were used to reveal regions of CNVs that differ between the extremes of the respective mouse lines (aCGH, NimbleGen; SNP-based invariant genomic probes, Jackson Laboratories). In total, we identified approximately 100 CNV regions ranging from around 10 kb to 7 Mb. To tes»

- Сохранена: **clear**, **0.490462**, `W3101922761`, `e:508a88858db4…`.
  Контекст: « 45%, 37%, 32%, and 26% of eyes at 1, 2, 3, 4, 5, 7, and 10 years, respectively. Only 7 corneal grafts (17.5%) remained clear at last follow-up. CONCLUSIONS: A GDD can successfully control intractable glaucoma even after a very long period of ti»

### representative, t=0.50

- Удалена: **food problems**, **0.499482**, `W2906067930`, `e:75e2175ca10c…`.
  Контекст: «g of the condition and mechanism for the Harvest Paradox, but also maybe offers some little reference values in solving food problems of China. Following the methods of Samuelson and Nordhaus, this paper utilizes the supply and demand theorem to study t»

- Удалена: **tendency to increase**, **0.497763**, `W4285525942`, `e:4d561107e82f…`.
  Контекст: « these financial instruments has been carried out. Different estimates of the total value of green bank loans and their tendency to increase and against the background of still low share in total lending have been compared. The article evaluates the structure »

- Сохранена: **composite part**, **0.500281**, `W3027457624`, `e:ac5045ca7350…`.
  Контекст: «unts of counterfeit composite parts showing up in the complex aerospace supply chain, it is absolutely vital to track a composite part throughout its lifecycle-from production to usage and to disposal. Existing barcoding methods are invasive, affect the »

- Сохранена: **ethidium**, **0.500734**, `W2039836112`, `e:fbfa33eb42b0…`.
  Контекст: «spinal astrocytes in culture, induced secretion of ATP, and permeabilized them to relatively large fluorescent tracers [ethidium (Etd) and lucifer yellow (LY)] through "hemichannels" (HCs). HCs can be formed by connexins or pannexins; they can open»

## Проверки и воспроизведение

Независимый union-find подтвердил точные наборы вершин, рёбер и метрики компонент всех 12 вариантов. Обе строки 0.50 полностью совпали с предыдущим анализом по всем столбцам. SHA-256 всех защищённых входов и предыдущих результатов сверены; прежний coarse sweep не изменён. Подробности — `validation.json` и `independent_validation.json`.

```powershell
.venv/Scripts/python.exe reports/entity_resolution/relex_base_t035/ner_threshold_analysis/threshold_045_050/run.py
.venv/Scripts/python.exe reports/entity_resolution/relex_base_t035/ner_threshold_analysis/threshold_045_050/write_report.py
```

`run.py` переиспользует прежний стандартно-библиотечный `../analyze.py`, меняя только выходной каталог и список порогов. Он не вызывает прежний генератор отчёта, не меняет предыдущий вывод. `virtual_subgraphs.jsonl` содержит vertex IDs и исходные edge IDs каждого варианта; свойства остаются в неизменённых источниках. Все компоненты и их состав сохранены в `components.csv`.

**Выполнен только запрошенный анализ NER 0.45–0.50 с шагом 0.01. Фильтр к KG не применён.**
