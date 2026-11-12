# Stage 2F.4 — пересечения корпуса

Дата аудита: 2026-11-12. Исходный commit: `3c19306`, ветка `design-v2-sequential-kg`.

## Область и порядок исследования

Прочитаны все 30 исходных абстрактов из `reports/ner_v2/research_run2/input_abstracts.jsonl`, затем все 811 профилей сущностей, сгруппированные по документам. Сначала сопоставлялись научные задачи документов, затем выбирались конкретные endpoints. Существующие 1307 RELEX_REL, 492 SIMILAR_TO, 118 HAS_SHARED_CONCEPT и 3236 evidence paths использовались для проверки окрестностей, provenance и прежнего покрытия. Предсказания A/B/C не использовались для открытия или обоснования положительных кандидатов.

Это аудит доступных абстрактов, а не полных статей. Скрининг охватывает 435 неупорядоченных пар документов. Междокументных пар контекстных сущностей возможно 315954; их исчерпывающая ручная разметка не проводилась. Воспроизведена структура всего baseline KG; научная проверка локальных model-extracted отношений выполнена для обсуждаемых выводов, а не для всех 1307 рёбер.

Обозначения D01–D30 ниже — только индекс строк исходного файла. Они не заменяют исходные document IDs. Метаданные содержат шесть дисциплинарных групп по пять документов, но научное содержание неоднородно даже внутри групп. Например, D04 помечен Computer Science / Machine Learning, хотя посвящён материаловедению, D26 помечен History, хотя посвящён практическому богословию и психотерапевтической концепции. Эти метки нельзя использовать как достаточное условие совместимости или запрет междисциплинарного поиска.

## Все документы: объекты, методы, процессы и результаты

В последнем столбце названы наиболее содержательные направления сопоставления. Отсутствие кандидата означает недостаток конкретной поддержки в абстрактах, а не доказанную независимость всех знаний в полных статьях.

| Код / document ID и название | Научная область по содержанию | Объекты и процессы | Методы | Характеристики и результаты в источнике | Пересечения и пределы |
|---|---|---|---|---|---|
| D01 / W2991563224 — Energy efficiency of organized cluster election method in wireless sensor networks | Сети беспроводных датчиков | WSN, cluster-head election, передача пакетов | Математический анализ, simulation, сравнение с LEACH | Источник сообщает снижение energy consumption около 3.5%, увеличение lifecycle 44% и packets 185% относительно LEACH | D19 имеет energy consumption другого процесса; это контроль N12, а не перенос эффективности |
| D02 / W4230116451 — A GPU parallel optimised blockwise NLM algorithm in a distributed computing system | Вычислительная обработка медицинских изображений | MRI denoising в distributed e-health system | GPU-параллелизация optimized blockwise NLM | Улучшение execution time; числа в абстракте нет | D07: конкретный возможный preprocessing-этап, P06; D05: общий image processing без общего метода |
| D03 / W4234628002 — Sistem Informasi Terintegrasi Evaluasi Kegiatan Mengajar Dosen Sebagai Implementasi Sistem Penjaminan Mutu Internal | Информационные системы образования | SPMI, оценивание преподавания студентами, управление результатами | Laravel/PHP, RAD, component-based construction | Две интегрированные страницы для questionnaire и management; экспериментальных метрик нет | Других исследований этой системы или образовательного качества нет; совпадение results не содержательно |
| D04 / W3027457624 — “Shape-Coding”: Morphology-Based Information System for Polymers and Composites | Материаловедение и идентификация изделий | Fiber-reinforced composites, embedded Pt nanostructures, morphology barcodes | Solid-state in situ reduction, кодирование по морфологии | Код 1 × 1 cm представляет от 3.4 billion до 95 trillion parts в зависимости от resolution | D12: слабая аналогия microstructure W03; D14 resolution — контроль N03; optical fibers D15 не являются подтверждёнными reinforcement fibers |
| D05 / W4389076493 — An Adaptive Region Proposal Network With Progressive Attention Propagation for Tiny Person Detection From UAV Images | Computer vision | Tiny-person detection, proposal boxes, density-adaptive proposal count | ARPN, FAM/BAM attention, mini-density map, FPN | Улучшение AP/AR без чисел в абстракте | D07: иная imaging-задача, W05; ни общего detector, ни mammography training данных нет |
| D06 / W3101922761 — Ten-Year Outcome of Glaucoma Drainage Device Surgery After Penetrating Keratoplasty | Офтальмология | Glaucoma, GDD, corneal graft survival после PK | Retrospective records, 40 eyes/40 patients, длительное наблюдение | IOP 34.0 -> 12.7 mm Hg; 7 grafts clear at last follow-up; разные показатели glaucoma success и graft survival | D09/D10 также retrospective surgery/device cohorts, но другие органы, вмешательства и исходы; общие patients/follow-up не связывают результаты |
| D07 / W1979727947 — High-throughput mammographic-density measurement: a tool for risk prediction of breast cancer | Маммография и статистическое моделирование риска | MD, mammographic image features, breast-cancer risk | Automated image processing, penalized/logistic regression, train/test split, Pearson r, likelihood-ratio test | r=0.884 с Cumulus; дополнительный image feature улучшает fit, P=0.0002 | D02: P06, возможный denoising; D05 W05; D22 — только общая статистическая постановка |
| D08 / W2791492120 — Impact and Modulations of Peripheral and Edaphic B Cell Subpopulations in Chronic Rhinosinusitis With Nasal Polyposis | Иммунология CRSwNP | Breg, mature/memory B cells, T-cell activity в blood/tissue | 10 пациентов, polyp/blood samples, multicolor flow cytometry | Разные частоты B-cell subpopulations в nasal polyps и PBMC | D16: P03 activity-to-T-cell-class; эффект Breg на rHuEPO-иммуногенность не измерялся |
| D09 / W2761455282 — Perioperative Complications of Total Knee Arthroplasty in Dialysis Patients | Ортопедия и нефрология | TKA у ESRD/dialysis patients | Retrospective series: 21 пациент/26 knees; clinical/radiographic evaluation, AKS и WOMAC | Один death; шесть пациентов с medical complications; 19 улучшившихся outcomes; наблюдение остальных в среднем 5.8 years | D16: P01 ESRD/CKD и P02 условная релевантность rHuEPO; анемия и экспозиция препарату не сообщены |
| D10 / W4391351014 — Effects of Ionizing Radiation on Cardiac Implantable Electronic Devices (CIEDs) in Patients with Esophageal Cancer Undergoing Radiotherapy: A Pilot Study | Онкорадиология и кардиальные устройства | CIED function при radiotherapy esophageal cancer | 16 пациентов, обследования до лечения и после каждой irradiation | 28 fractions, total 58.8 Gy, mean max CIED dose 1.8 Gy; severe malfunctions не зарегистрированы в этой серии | D13 W04: overvoltage/SPD и ionizing radiation — разные механизмы; D16 cancer слишком широкая категория |
| D11 / W2084709926 — NSTX tangential divertor camera | Диагностика плазмы и приборная оптика | Turbulence/impurity emission у divertor x-point, imaging optics | Sapphire window, mirror, coherent fiber optic bundle, fast camera; UEDGE/BOUT для интерпретации | Описана конфигурация наблюдения, до 40500 frames/s в примере камеры; результат по турбулентности не установлен | D15 P05: тип optical fiber как элемент bundle; soliton regime не подтверждён |
| D12 / W2317432785 — Structural and Magnetic Studies of [Co0.45Fe0.45Zr0.1/a-Si]N Multilayers | Физика магнитных многослойных материалов | Sublayers, grains, silicides, interface microstructure | TKE, low-angle X-ray reflectivity, XANES, XMCD | Thickness-dependent grain/silicide formation согласуется с enhancement/reduction TKE | D04 W03 microstructure; D14 thickness — N04; D21 grain — N02. Формулы в исходном абстракте частично повреждены |
| D13 / W2364973104 — Research on the Influence Factors and Limiting Methods for Lighting Over-Voltage in Military Power Supply | Электротехника и защита от молнии | Generator/control/cables, transient overvoltage, SPD | Electromagnetic transient simulation PSCAD/EMTDC, shielding/filtering proposal | Предложена схема защиты; количественной экспериментальной оценки нет | D10 W04 проверен и отклонён; D21 supply line — N01; D29 reliability — N10 |
| D14 / W2096307598 — New radio observations of the Moon atLband | Радиоастрономия и планетология | Lunar brightness temperature, polarization, regolith properties | L-band synthesis telescope observations | Resolution 159'' × 87'', sensitivity 1.7 K, disk brightness около 233 K; пространственная неоднородность polarization/thickness | D12/D04 совпадения thickness/resolution не означают механизм; другой труд о Луне в корпусе отсутствует |
| D15 / W3081936674 — Optical soliton solutions to a (2+1) dimensional Schrödinger equation using a couple of integration architectures | Нелинейная оптика и математическая физика | Optical pulses in fibers, traveling-wave solutions | Finite series, auxiliary ODE, exp(-phi), modified Kudryashov, tanh ansatz, graphical simulations | Получены типы soliton и periodic solutions; совместимость с каким-либо прибором не проверена | D11 P05 компонентная категория; перенос soliton solutions на NSTX не обоснован |
| D16 / W3123056951 — Modified recombinant human erythropoietin with potentially reduced immunogenicity | Protein engineering и иммунология | rHuEPO, HLA-DRB1*09, T-cells, EPO epitopes, low hemoglobin/CKD | Computational hotspot/mutant design, in vitro immunogenicity/bioactivity | 17 mutants предсказаны; пять менее immunogenic с сохранённой или слегка сниженной bioactivity | D09 P01/P02, D08 P03; D18 HLA/gene dosage N11; D19 общее engineering не общий метод |
| D17 / W1995028339 — Conserved Gene Order and Expanded Inverted Repeats Characterize Plastid Genomes of Thalassiosirales | Сравнительная геномика диатомей | Plastid genomes, gene order, repeats, gene losses | Сравнение семи полных genomes и филогенетическая интерпретация | 116459–129498 bp; conserved order с исключениями; repeat expansion и lineage-specific gene loss | D18/D19 общее слово genes/genetics не задаёт общий ген, организм, expression assay или механизм |
| D18 / W2076293586 — Copy number variants in two mouse models of affective disorders | Поведенческая генетика | Inbred mouse lines, CNV, gene dosage/expression, affective endophenotypes | aCGH и SNP-based genomic probes, сопоставление expression и CNV | Около 100 CNV regions, 10 kb–7 Mb; простой dosage-expression correlation найден не во всех случаях | D19 P04 процесс expression; W01 причинный перенос не поддержан; D20 W06 общий model organism |
| D19 / W3138328568 — Simultaneous Improvement of Limonene Production and Tolerance in Yarrowia lipolytica through Tolerance Engineering and Evolutionary Engineering | Микробная биотехнология | Yeast, limonene cytotoxicity, tolerance, product titer | Transcriptomics, targeted overexpression/coexpression, adaptive evolution, morphology/membrane analysis | 82 upregulated genes; 8 улучшают tolerance, 5 production; YALI0F19492p даёт 8-fold titer | D18 P04 expression specialization; CNV-механизм не показан; D01 energy consumption N12 |
| D20 / W2039836112 — FGF-1 induces ATP release from spinal astrocytes in culture and opens pannexin and connexin hemichannels | Нейробиология и клеточная сигнализация | Astrocytes, FGF-1, ATP, Px1/Cx43 hemichannels, permeability | Pharmacology, siRNA Px1 knockdown, Cx43 knockout, fluorescent tracers | Различается участие Px1/Cx43 при 2h/7h; предлагается vesicular ATP/autocrine-paracrine механизм | D18 W06 мыши используются для другой цели; D08 inflammation/immune biology не устанавливает общий экспериментальный механизм |
| D21 / W2906067930 — Research on the Phenomenon That the Increase of Grain Production Did Not Cause the Harvest Paradox in China from 2003 to 2014 | Аграрная экономика | Grain production, demand/supply shifts, grain price и farmers income | Supply-and-demand theorem в интерпретации Samuelson/Nordhaus | Интерпретируется отсутствие Harvest Paradox в Китае 2003–2014 | D23 W02 demand/pricing overlap недостаточен для конкретного переноса модели; grain/supply line дают N02/N01 |
| D22 / W2078771716 — Measuring Economic Convergence | Макроэконометрика | ERM/EMU, exchange/interest/inflation series | Time-varying parameters, Kalman filtering, convergence tests | В источнике разные выводы для nominal FX, interest и inflation convergence | D07: общая статистическая проверка, но другие модели и переменные; D24 monetary policy не общий эмпирический объект |
| D23 / W4309030044 — Market competition oriented air-rail ticket fare optimization | Экономика транспорта | HSR/airline competition, connectivity, fares | Elastic-demand stochastic user equilibrium, bi-level programming, particle swarm и successive averages | Модельные выводы о pricing и cooperation; чисел нет | D21 W02 price-demand framework; D24 cooperation N08 относится к другим институтам |
| D24 / W4285525942 — FEATURES OF FINANCIAL INSTRUMENTS OF GREEN BANKING IN THE GLOBAL MARKET OF BANKING SERVICES | Финансы и green banking | Loans, mortgages, bonds, co-investment, securitization, risks | Описательный сравнительный анализ инструментов и тенденций | Выводы о развитии форм финансирования и регулировании, без чисел в абстракте | D22/D23 имеют лишь общие economic institutions/policy категории; причинная связь cooperation не задана |
| D25 / W2130236240 — The Impact of Advance Notice: A Comment on a Study by Nord and Ting | Экономика труда | Advance layoff notice, joblessness, earnings losses | Комментарий к study и Displaced Worker Surveys 1984/1986; различение formal/informal notice | Излагаются утверждения Nord/Ting и трудность интерпретации notice; собственный эффект не оценён в абстракте | D29 unions — иной исторический материал; отдельного исследования тех же plant closings/workers нет |
| D26 / W7215137876 — Emotional Maturity in the Light of Synodality: Contributions from Internal Family System Therapy | Практическое богословие и терапевтическая концепция | Synodal mindset, attachment/relational wounds, emotional skills | Концептуальное сопоставление IFS и synodal practices | Предлагается возможная польза inner work; клинического испытания и измеренного эффекта нет | D18 W07, D30 W09: тематические связи; не молекулярное объяснение и не историческая причинность |
| D27 / W4312422299 — Land as Regenerative Space in The Book of Mormon | Религиоведение и литературный анализ | Nephites, Lehi, Jerusalem, movement/escape/reproduction | Интерпретация повествования Book of Mormon | Сформулирован вопрос воспроизводства народа Бога; количественного исследования нет | D26/D30 религиозная тематика, но другой корпус, эпоха и задача; общих именованных объектов нет |
| D28 / W2326480042 — Between the 30s and the 60s | Интеллектуальная и политическая история | US radicals, New Left, counterculture, CIO | Авторское сравнение поколений/десятилетий | Утверждается необходимость сравнения 1930s/1960s; абстракт обрывается на San Francis247 | D29 W08 labor-history category; нет подтверждённого общего архива или интервью |
| D29 / W4381572413 — “First of all, let’s go learn something” | История и методология oral history | Oral interviews, shared authorship, archives, Italian historiography | Историографический/библиографический обзор | Обсуждены preservation/accessibility и применение к разным темам | D28 W08 наиболее близок по labor-history теме; oral interviews не идентифицированы как источник D28; reliability N10 |
| D30 / W2394633461 — Pressing Metal, Pressing Politics: Papal Annual Medals, 1605–1700 | История и нумизматика | Papal annual medals, identity creation, politics | Анализ изображений медалей в историческом нарративе | 10 popes, 94 medals за 96 лет; роли builder/warrior/impresario | D26 W09 общая церковная среда без доказанного влияния; D27 другая религиозная традиция/источники |

## Какие документные пересечения выдержали проверку

| Пара документов | Положительные кандидаты | Что действительно связывает тексты | Что не следует из пересечения |
|---|---|---|---|
| D09–D16 | P01 STRONG, P02 PLAUSIBLE | ESRD/CKD как отношение стадии; возможная релевантность rHuEPO при определённых состояниях диализных пациентов | Идентичность когорт, назначение препарата в D09, влияние rHuEPO/PRCA на результаты TKA |
| D08–D16 | P03 STRONG | Активность T-клеточного класса и T-cells как участник | Breg-эффект на anti-rHuEPO response, общие клетки или единый эксперимент |
| D18–D19 | P04 STRONG | Expression/overexpression как процесс и его специальный режим | Дозовый CNV-механизм overexpression в дрожжах или одинаковые гены/фенотипы |
| D11–D15 | P05 STRONG | Компонентная связь к типу optical fiber | Те же физические волокна, soliton propagation в bundle NSTX или применимость уравнения D15 |
| D02–D07 | P06 PLAUSIBLE | Проверяемый перенос NLM-family preprocessing в mammography/MD | Использование того же алгоритма в D07, доказанное улучшение MD/risk model |

Всего шесть пар сущностей в пяти парах документов, девять разных документов. Четыре STRONG относятся к типизированным связям понятий. Два PLAUSIBLE — исследовательские гипотезы с недостающими условиями. Ни одна категория не означает новое установленное эмпирическое отношение между двумя исследованиями.

Десять новых проверенных, но не принятых пар W01–W10 сохранены в `candidate_evidence.jsonl` с полными endpoints и аргументацией. Девять WEAK и одна REJECTED не включены в `positive_candidates.csv`. Это позволяет проверить, что положительный набор не пополнялся за счёт общих категорий. Контроли N01–N12 из прежнего эксперимента вынесены в отдельный CSV.

## Ограничения состава корпуса

Научно близкие узкие тематические серии почти отсутствуют: нет второго исследования того же диатомового порядка, yeast strain, HLA-mutant design, NSTX turbulence, Moon regolith, ARPN detector или cohort surgery. Пять сохранённых документных пересечений являются выбранными случаями, а не оценкой истинной доли всех научных связей. Большинство остальных абстрактов не дают конкретной совместимой цели, переменной, метода или именованного объекта для нового междокументного утверждения.

D28 усечён, D12 содержит повреждённые символы формул, D25 и D27 дают преимущественно постановку/комментарий, D26 не описывает клиническое испытание, а D11 в том числе описывает планируемую интерпретацию. Эти ограничения нельзя компенсировать уверенной типизацией LLM/ReLEx. Исходные документы и NER/ER в аудите не исправлялись.

Идентичность объекта, сходство объектов, общая категория, типизированная связь понятий, многошаговый путь и новая гипотеза различаются. Одинаковые genes, patients, energy consumption, cooperation или reliability дают полезные рубрики поиска, но не разрешают перенос фактов. Единого RELATED_TO вместо этих различий не вводилось.

## Полная карта скрининга 435 пар документов

Матрица ниже сохраняет решение скрининга для каждой неупорядоченной пары. P — хотя бы один STRONG/PLAUSIBLE кандидат; W — новая рассмотренная пара W01–W10 без положительного решения; C — дополнительная близкая тема или прежний контроль, не принятые как положительный мост; . — в доступных абстрактах не найдено достаточно конкретного пересечения для выдвижения пары. Пробелом обозначена нижняя половина и диагональ. P имеет приоритет над W и C. Точка не является gold NO_RELATION для полных текстов.

Для C, не представленных контролями: D02–D05 объединяет обработка изображений, D07–D22 — статистическая проверка разных моделей, D08–D20 — иммунные/воспалительные контексты разных клеток, D16–D19 — разные подходы биоинженерии, D17–D18 — разные геномные изменения в разных организмах, D22–D24 — общий макроэкономический/регуляторный контекст, D25–D29 — трудовая тематика без общих источников, D26–D27 и D27–D30 — широкая религиозная тематика. Они не добавляют отдельного положительного endpoint relation. D04–D15 не отождествляет reinforcement fibers с optical fibers.

<!-- GENERATED SCREEN MATRIX -->

```text
    01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
01     .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  C  .  .  .  .  .  .  .  .  .  .  .
02        .  .  C  .  P  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
03           .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
04              .  .  .  .  .  .  .  W  .  C  C  W  .  .  .  .  .  .  .  .  .  .  .  .  .  .
05                 .  W  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
06                    C  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
07                       .  .  .  .  .  .  .  .  .  .  .  .  .  .  C  .  .  .  .  .  .  .  .
08                          C  .  .  .  .  .  .  P  .  .  .  C  .  .  .  .  .  .  .  .  .  .
09                             .  .  .  .  .  .  P  .  .  .  .  .  .  .  .  .  .  .  .  .  .
10                                .  .  W  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
11                                   .  .  .  P  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
12                                      .  C  .  .  .  .  .  .  C  .  .  .  .  .  .  .  .  .
13                                         .  .  .  .  .  .  .  C  .  .  .  .  .  .  .  C  .
14                                            .  .  .  .  .  .  .  .  C  .  .  .  .  .  .  .
15                                               .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
16                                                  .  C  C  .  .  .  .  .  .  .  .  .  .  .
17                                                     C  .  .  .  .  .  .  .  .  .  .  .  .
18                                                        P  W  .  .  .  .  .  W  .  .  .  .
19                                                           .  .  .  .  .  .  .  .  .  .  .
20                                                              .  .  .  .  .  .  .  .  .  .
21                                                                 .  W  .  .  .  .  .  .  .
22                                                                    .  C  .  .  .  .  .  .
23                                                                       C  .  .  .  .  .  .
24                                                                          .  .  .  .  .  .
25                                                                             .  .  .  C  .
26                                                                                C  .  .  W
27                                                                                   .  .  C
28                                                                                      W  .
29                                                                                         .
30
```

Итоги скрининга: P=5, W=9, C=21, .=400; всего 435. Число W меньше числа W01–W10, поскольку W01 проверяет альтернативный механизм внутри уже положительной документной пары D18–D19.

## Покрытие сущностей и source revisions

Все перечисленные counts относятся к сохранённым сущностям Stage 2F.3, а не к заново выполненному NER. Полный текст каждой строки доступен по document ID в исходном JSONL.

| Документ | Сущностей | Source discipline | Revision ID |
|---|---:|---|---|
| D01 / W2991563224 | 12 | Computer Science / Machine Learning | `sha256:f5626ce38fed5e64846696dc2fe4c1da711561d42bfc116ad1cfe850e3f618fd` |
| D02 / W4230116451 | 16 | Computer Science / Machine Learning | `sha256:be8d449de0b368ff95d367594803fe257989c989633702f3bee4ceab8be839ed` |
| D03 / W4234628002 | 23 | Computer Science / Machine Learning | `sha256:884a49c84f4b71814e4b064ce4b04e94e91e16ec2abd50deb975942e8458be47` |
| D04 / W3027457624 | 25 | Computer Science / Machine Learning | `sha256:e2c9df190288a816f9dd29f08558c97eb10702fd4efc6a6113cbb9934674c583` |
| D05 / W4389076493 | 28 | Computer Science / Machine Learning | `sha256:f4cf0293b69944ea4057fbb9b4935928b2ddc909e1b7e72e7b9ea3e9414ed1fe` |
| D06 / W3101922761 | 45 | Medicine | `sha256:4ad8779832806262a62448df71e477ddf9e742488e35b3b32cb1551579bfb44b` |
| D07 / W1979727947 | 30 | Medicine | `sha256:e884e0e10d07e2f4d99fb3a768c3d6513389a47d10dfea36b2b61938ddb00505` |
| D08 / W2791492120 | 26 | Medicine | `sha256:4508e79f0f2d74bcf709e0c404536b736c62dd03e01ec5aced75810fb984d94b` |
| D09 / W2761455282 | 53 | Medicine | `sha256:f9d06039150fe307bf30002731fdd81b94f576d1a641bab8cb4e7b06e3ed0584` |
| D10 / W4391351014 | 22 | Medicine | `sha256:6685fb1fcac700267e0e5d147571e87c246ae22ffbc81c8811aa051eb0b16a12` |
| D11 / W2084709926 | 19 | Physics | `sha256:89ad5e956f2937ab847f226fda41b82bf0389aa31229ae7c59c95420741fbf9a` |
| D12 / W2317432785 | 37 | Physics | `sha256:c742c4247bf619005be9c9bb04322d1179e96ef169edd80c38c01254d3cff1e6` |
| D13 / W2364973104 | 32 | Physics | `sha256:c5e143f3d69dc7ca232226044216bb6b65e4a571a448cf4ee9eba8c746ea440c` |
| D14 / W2096307598 | 27 | Physics | `sha256:10fc278cebea8bfe80c4c66ab01d21e88833a68cbd4c3913b86a0525d1d856bd` |
| D15 / W3081936674 | 18 | Physics | `sha256:9a0f1e3257cc64d931aa684e38adc3063a1502b60f2d1c43e046ba0c77429695` |
| D16 / W3123056951 | 28 | Biology | `sha256:aa04a14796bd6c13a2dbcfd79d237898ec2abe12c095cd1dfb23a9c36bce0fa5` |
| D17 / W1995028339 | 44 | Biology | `sha256:0d85abeec77604f49039eb07af4a689ef78fe525a61c8f57142214508e4110ac` |
| D18 / W2076293586 | 37 | Biology | `sha256:cfea73ac727094f3729f0a5102f832e50fad3813546b3c3fe2b4957e6e90463b` |
| D19 / W3138328568 | 40 | Biology | `sha256:4e04e88c833477762332b30f68ea9f9cc8b444e20aaa56f6252b0011937b3b25` |
| D20 / W2039836112 | 40 | Biology | `sha256:9b07bbf92eed6b7b5c36f4357836c6736cb8e3a85b0828367913e4f1f32319f1` |
| D21 / W2906067930 | 29 | Economics | `sha256:7cad1d4676ba282b72e20667d454da06526d16c0aa9914a0d749c4e82862d1fd` |
| D22 / W2078771716 | 31 | Economics | `sha256:172a5233fdffe70f136c8575078f653a75467e8f035028e6aac68f3184cd7f3c` |
| D23 / W4309030044 | 16 | Economics | `sha256:148e40e3759e1995c899fa0f1be30517e1afc6f52a71c05c1f74bf3d79b05217` |
| D24 / W4285525942 | 41 | Economics | `sha256:1f59574097e2558bffa2181b837aeef9646420f22490c1079799aaf3060192a9` |
| D25 / W2130236240 | 8 | Economics | `sha256:9e231f1ce7251317d477b15c090001eee5efab8336e78362621e30329140e900` |
| D26 / W7215137876 | 18 | History | `sha256:7348a443574c79025e8d91562a546eaddbb34a1731bcfe890d09037c05581a14` |
| D27 / W4312422299 | 11 | History | `sha256:fe5c937534750dc1a8835106222989c570bd7f2f820233f97e69c429a61c4c52` |
| D28 / W2326480042 | 9 | History | `sha256:22b10a66ff08a431db39dce70a17d17784fd9e59e9e10a8ff978d1cb827ceff2` |
| D29 / W4381572413 | 16 | History | `sha256:8c231ed4d765453627674ff77b2912ad48e256fb7a99ee645374b28f220dcb9a` |
| D30 / W2394633461 | 30 | History | `sha256:d5203d9ab3268c0e6f2ba115965411aaed58b46a836a80da8a75c460e5941ef9` |

Итого: 811 сущностей, 30 документов.
