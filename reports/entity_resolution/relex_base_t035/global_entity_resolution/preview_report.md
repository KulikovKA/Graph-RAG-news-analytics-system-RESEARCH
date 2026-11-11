# Stage 2F.2 — Global Entity Resolution: proposed mapping и preview

Дата: 11 ноября 2026, Europe/Moscow. Ветка `design-v2-sequential-kg`. Статус: **предложение для проверки, не импорт и не новая версия KG**.

Актуальный namespace: `relex_sem_e5_small_name_t095_ner_t046_max`. NER ≥0.46, confidence=max_member; E5-small / text_name / similarity≥0.95. Все счётчики ниже относятся к **905**, а не историческим 1122 вершинам.

## Основной результат

Контекстно обоснованы **69 proposed MERGE-групп** из 163 текущих canonical IDs. Это позволяет устранить 94 повторов: **905 → 811 contextual entities**. Предлагаются **32 отдельных научных концепта**, итоговая схема — **843 вершин**, без документов. Все подтверждённые identity-группы находятся внутри одного документа; **междокументного тождества конкретных референтов не доказано**. Глобальный слой связывает контексты общими концептами.

Все **1307 исходных RELEX_REL** сохранены в восстанавливаемом виде. После переназначения это 1076 уникальных структурных троек: 164 групп параллельных отношений, 231 дополнительных записей с теми же концами/типом. Петли: 12 исходных → 39 записей, из них 27 новые. Это диагностические последствия ER, а не новые достоверные факты.

Из 349 одноимённых SIMILAR_TO **254 можно считать структурно избыточными в предложенной двухслойной схеме**: 49 схлопываются по доказанной идентичности, 205 заменимы путём contextual entity → concept ← contextual entity. Ещё 27 отклонены по содержанию, 42 требуют REVIEW, 26 остаются RELATED. Отклонение связи и избыточность — разные причины.

Из 121 существующего междокументного перехода **90 имеют положительное контекстное основание на заявленном семантическом уровне**: 77 SHARED_CONCEPT и 13 RELATED. Это **не 90 подтверждённых фактических отношений**. 30 отклонены, 1 требует REVIEW. Все исходные 492 semantic records сохранены для аудита.

Числа MERGE — результат текущей аналитической проверки контекстов агентом, не независимая человеческая gold-разметка и не разрешение применить mapping. 18 групп имеют несовпадающие исходные наборы типов; основания исключений явно записаны. Перед применением потребуются проверка предлагаемой политики типизации и проверка отношений, которые становятся петлями. Этап 2F.3 не начат.

## Baseline и независимый аудит повторов

| Показатель | Baseline |
|---|---|
| Документы | 30 |
| Canonical entities | 905 |
| Source entities / typed mentions в их provenance | 993 / 1101 |
| RELEX_REL / SIMILAR_TO | 1307 / 492 |
| Междокументные SIMILAR_TO / RELEX_REL | 121 / 0 |
| Изоляты / слабые компоненты | 175 / 230 |
| Крупнейшая компонента: entities / documents | 378 / 17 |
| Группы нормализованных имён / вершины в них | 96 / 259 |
| Одноимённые SIMILAR_TO / междокументные одноимённые | 349 / 91 |

Нормализация переиспользует 1A: NFC → casefold → NFC → сжатие Unicode-пробелов; пунктуация, числа, дефисы, квалификаторы и повреждения не стираются. **87 групп** содержат повторы внутри документа, **11** — между документами; 2 группы (`patients`, `energy consumption`) входят в обе категории. Все 349 неупорядоченных exact-name пар текущего снимка уже имеют SIMILAR_TO. Сам факт совпадения вектора имени не используется как доказательство идентичности.

Baseline проверен полным read-only чтением Neo4j: все 935 физических узлов (30+905), 2704 отношения (905 MENTIONS +1307 facts +492 semantic), source_json, max-member confidence, endpoints и fingerprint сопоставлены с локальными источниками. Снимок перед заключительной симуляцией и после неё сверяется SHA-256; подробности — `live_baseline_validation.json`.

## Что учтено из предыдущих этапов

- [1A](../exact_matching_audit/report.md): совпадение имён — только генератор кандидатов; различать геномный/таксономический order, разные множества genes/eyes/patients и измерения в разные моменты.
- [1B](../conservative_resolution/report.md): существующие source-member группы не разбираются и не переписываются. В mapping каждая текущая canonical запись сохранена целиком, включая прежние решения и все исходные member IDs. Контекстная идентичность и типизация исследуются отдельно.
- [2C](../semantic_resolution/candidate_verification/proposed_merge_rules.md): проверять уровень референта и внешние модификаторы, не голосовать cosine. 113 индивидуально разобранных прежних пар, чьи концы сохранились при NER .46, включены как evidence; старые ACCEPT не импортированы автоматически как MERGE. 2C REJECT внутри proposed MERGE-клик отсутствуют. REJECT на тождество из 2C отдельно переосмыслен: например, разные strain IDs, XANES/XMCD, hip/knee arthroplasty и AP/AR могут получить RELATED, не MERGE; самостоятельная полезная близость и identity различаются.
- [2E](../semantic_resolution/global_kg_import/report.md) и [2F.1](../ner_threshold_analysis/global_kg_t046_max/report.md): используют lossless provenance, один SIMILAR_TO на неупорядоченную пару и отдельные document-mentions. Актуальный набор берётся из membership/manifest 2F.1; старый граф не подменяет baseline.

## Правила Global Entity Resolution

1. **MERGE** требует одного референта на явно выбранном уровне: именованный вид/метод/молекулярный вид, документно определённый объект/набор или одна явно определённая когорта/оценка. Один объектный класс не равен физическим экземплярам; область применения процедуры не равна операциям разных пациентов. Для каждой группы заданы полный состав и положительное основание; каждая пара внутри группы имеет собственную строку MERGE. Клики не пересекаются, транзитивного замыкания похожести нет.
2. **SHARED_CONCEPT** сохраняет contextual IDs и добавляет отдельный `concept:2f2:*`. Связь `HAS_SHARED_CONCEPT` — контекстная принадлежность научной теме/классу, **не identity, не доказанная IS_A-инстанциация, не факт пациента и не наследование фактов**. Фактические рёбра никогда не перенаправляются на concept_id.
3. **RELATED** означает обоснованную близость при разных уровнях, носителях или условиях: болезнь/риск, класс/подтип, параметр/значение, действие/объект. Идентификаторы не объединяются; новых типизированных фактических отношений не создаётся.
4. **REJECT** отклоняет предлагаемый прямой кандидат как ошибочную/малоинформативную связь: полисемия, generic self-reference разных публикаций, heading, modifier fragment или подтверждённое различие референтов без полезного прямого основания. Это не удаление исходного узла или его фактов.
5. **REVIEW** оставляет ID раздельно при неизвестном авторстве/составе/времени, повреждённом имени, неоднозначной границе или неразрешённом уровне gene/allele, instrument/scores, class/instance. REVIEW-вершины сохраняются; их старые semantic links остаются явно pending.
6. Для аббревиатур использовать раскрытие в **том же документе**, проверить все member contexts и модификаторы. Сокращение может задавать один named kind, но IOP до/после операции не становится одним измерением. PK/RT объединяются на уровне procedure kind, а не individual operations/fractions.
7. Между документами одинаковое имя/высокий cosine не доказывают идентичность конкретного объекта. Требуются общий устойчивый идентификатор и совпадение уровня/спецификации либо прямое документальное подтверждение. В этом корпусе таких cross-document MERGE-групп не предложено.
8. Тип — диагностический сигнал. Равенство типов не доказывает identity; несовместимость типов может быть NER-ошибкой. В отличие от строгого gate 1B/2C, 2F.2 допускает **явно обоснованное исключение** для определения имени/аббревиатуры или точно указанной когорты. Source types сохраняются; proposed_primary_type — только предложение для проверки, не применённая коррекция.
9. Представитель — существующий ID: минимальный document_id/start, затем наиболее полное имя при равном start, затем canonical ID. Это не выбор по NER confidence. Все source scores/offsets/provenance сохраняются. У концептов нет выдуманной calibrated confidence.
10. Existing RELEX_REL учитываются как диагностические последствия решения, **не независимое доказательство identity**. Автоматически извлечённые HAS_PART/COMPARED_WITH между двумя именами одного объекта могут стать петлями; они сохраняются с пометкой relation_review_required.

## Покрытие кандидатов и решения

В итоговой таблице **593 пар** и **463 различных contextual IDs**. Исходный pool: 492 SIMILAR_TO, включая все 349 exact-name пар; 113 прежних индивидуальных 2C-проверок и 30 пар из сохранённых abbreviation candidates (множества пересекаются). Получено 579 уникальных пар. Ещё 14 пар добавлены только для полного явного попарного рассмотрения предложенных клик. Это не новый semantic search и не полный перебор 905×904/2.

| Решение | Пар |
|---|---|
| MERGE | 126 |
| SHARED_CONCEPT | 240 |
| RELATED | 97 |
| REJECT | 51 |
| REVIEW | 79 |

MERGE-пар больше, чем MERGE-групп: например, clique из четырёх участников имеет шесть пар. Все 96 exact-name групп имеют конкретные group-level основания и полный состав в `duplicate_groups.csv`; вся существующая semantic-сеть имеет решение в `resolution_decisions.csv`. Узлы вне выбранного candidate pool сохранены отдельными и **не объявлены глобально уникальными**. Необоснованных fallback-решений без отдельного контекстного основания в итоговой таблице нет.

## patients: когорты, подгруппы и общий Patients

В baseline 14 одноимённых вершин из пяти документов. Пять документов **не означают пять одинаково определённых клинических когорт**:

- W2761455282: TKA-серия **21 ESRD-пациент / 26 knees, 2007–2015**; только её явно анафорические упоминания входят в одну MERGE-группу N398/N399/N400. Background ESRD-популяция в тексте про hip arthroplasty и these patients в постановке вопроса не автоматически равны этой enrolled серии.
- В том же документе: **other 20** после смерти одного, **6** с осложнениями и **19** с improved outcomes — самостоятельные подгруппы. Пересечения групп событий не выводятся. **Most patients** в заключении может означать те же 19, но это REVIEW, а не доказанное тождество.
- W2791492120: **10** CRSwNP-пациентов, полипоидная ткань/кровь. N455/N463/N474 — одна явная серия; CONDITION у qualified patient phrase ошибочен для людей, но не делает их диагнозом CRSwNP.
- W3101922761: **40 пациентов / 40 глаз** с GDD after PK. Пациенты и глаза не объединяются, даже когда их количество совпадает.
- W4391351014: **16** пациентов с esophageal cancer, radiotherapy 2012–2022, Heidelberg — отдельная серия; background cancer patients relying on CIED не приравниваются enrolled группе.
- W3123056951: **общая популяция rHuEPO recipients** и **some patients с anti-rHuEPO neutralizing antibodies/PRCA**. Численность собственной клинической когорты не указана; работа описывает дизайн и in-vitro проверку белков. Нельзя выдумывать enrolled cohort.

| Alias / ID prefix | Документ | Offset(s) | Типы | Max confidence | Референт / решение |
|---|---|---|---|---|---|
| N386 / e:4d249c81a4ed… | W2761455282 | [38] | ACTOR | 0.754086 | Общая ESRD-популяция в background о total hip arthroplasty, не enrolled TKA cohort. Сохранить отдельно; Patients context link. |
| N391 / e:93516aed6123… | W2761455282 | [201] | ACTOR | 0.764892 | These patients с ESRD в постановке пробела по TKA; population-level, не явный состав 21. Сохранить отдельно; Patients context link. |
| N399 / e:4c01c07b43a2… | W2761455282 | [427] | ACTOR | 0.570194 | Пациенты этой больницы за past decade, непосредственно уточнённые как 21 TKA patients. MERGE M029 |
| N400 / e:c7f6b860fc0a… | W2761455282 | [516, 650, 694] | ACTOR | 0.898713 | Когорта 21 пациент / 26 knees, primary TKA 2007–2015; includes all-patients follow-up members 650 и 694. MERGE M029 |
| N417 / e:847fa491143b… | W2761455282 | [1054] | ACTOR/OBJECT | 0.826873 | Other 20 patients после смерти одного пациента; отдельная surviving/followed subgroup, ACTOR/OBJECT. Сохранить отдельно; Patients context link. |
| N418 / e:9fe2f2c54b56… | W2761455282 | [1114] | ACTOR | 0.825249 | Six patients (28.5%) с medical complications; подмножество, не исходная когорта 21. Сохранить отдельно; Patients context link. |
| N426 / e:e13e94a1cd71… | W2761455282 | [1297] | ACTOR | 0.779344 | 19 patients (90.5%) с improved clinical outcomes at latest follow-up; не равны 20 survivors или 6 complications. Сохранить отдельно; Patients context link. |
| N436 / e:ad37de4afac5… | W2761455282 | [1626] | ACTOR | 0.784162 | Most patients с improved outcome в conclusion; точный состав/число не повторены. Сохранить отдельно; Patients context link. |
| N455 / e:f90203005b5e… | W2791492120 | [615, 853, 1171, 1287] | ACTOR | 0.756118 | 10 пациентов sinus surgery в CRSwNP-study; четыре анафорических source members этой же серии. MERGE M033 |
| N582 / e:a156f1ad92a3… | W3101922761 | [361, 466] | ACTOR | 0.874928 | 40 пациентов GDD after PK; один глаз на пациента не означает равенство сущностей Patients и Eyes. Сохранить отдельно; Patients context link. |
| N622 / e:f6e4ef94482c… | W3123056951 | [79] | ACTOR | 0.880514 | Общая популяция получателей rHuEPO с low hemoglobin при CKD/cancer/anemia. Сохранить отдельно; Patients context link. |
| N626 / e:0603ca063d4d… | W3123056951 | [181] | ACTOR | 0.940641 | Some patients с neutralizing antibodies и PRCA после повторного rHuEPO; отдельная подгруппа общей популяции N622. Сохранить отдельно; Patients context link. |
| N864 / e:22d3cd3da78f… | W4391351014 | [41] | ACTOR | 0.680332 | Общая растущая популяция cancer patients relying on CIED в background, не 16 включённых пациентов. Сохранить отдельно; Patients context link. |
| N869 / e:a5809b87f459… | W4391351014 | [246, 553, 649] | ACTOR | 0.931172 | 16 пациентов с esophageal cancer, RT 2012–2022, Heidelberg; includes enrolled/all-patients members. Сохранить отдельно; Patients context link. |

Предлагаемый `concept:2f2:patients` охватывает 20 исходных contextual IDs, включая qualified/dialysis/cancer-patient forms; после identity MERGE остаются **16** context→concept связей в пяти документах. Никакие HAS_RESULT/CAUSES/USES, описывающие отдельную когорту, не переносятся на Patients. При обходе такой путь означает общую категорию, а не «другая статья подтверждает тот же результат у тех же пациентов».

Три показательных контекста:

- N400, 21 patients: «cal outcome of TKA in dialysis patients at our hospital during the past decade. METHODS: We retrospectively assessed 21 patients (26 knees) with ESRD who underwent primary TKA between 2007 and 2015. A preoperative medical evaluation was performed i»
- N418, 6 patients: «ery (4.8% mortality). The mean follow-up period of other 20 patients was 5.8 years (69.7 months, 24 to 119 months). Six patients (28.5%) had medical complications, including pneumonia (n = 3), stroke (n = 2), and cardiovascular events (n = 3), afte»
- N869, 16 patients: «The aim of this study was to evaluate RT-related malfunctions of CIEDs. (2) Methods: We retrospectively analyze sixteen patients with esophageal cancer who were treated with radiotherapy between 2012 and 2022 at the University Hospital Heidelberg. »

## results: общий концепт не предлагается

6 вершин / 6 документов: пять анафорических указателей и один заголовок. Все 15 exact-name попарных связей отклонены как полезная глобальная идентификация; существующие factual edges исходных узлов не удаляются. N725 имеет локальную информационную роль в PNM, однако это не основание для общего научного Results.

| Alias | Документ | Offset | Max confidence | Контекстная оценка |
|---|---|---|---|---|
| N030 | W1979727947 | [1504] | 0.784150 | Анафорическое Our results из заключения mammographic-density статьи, не конкретно названный результат. |
| N195 | W2078771716 | [1322] | 0.707956 | Taken together these results в статье об economic convergence; самостоятельного содержательного имени нет. |
| N258 | W2096307598 | [1158] | 0.892064 | The results show — общий указатель на lunar regolith measurements, не отдельная единая scientific entity. |
| N600 | W3101922761 | [799] | 0.564522 | RESULTS: — заголовок раздела перед mean follow-up, а не результат исследования. |
| N725 | W4234628002 | [688] | 0.522291 | Results as decision support materials в системе PNM; есть локальная информационная роль, но не общий научный Results. |
| N793 | W4309030044 | [716] | 0.765157 | The results suggest — анафорическая ссылка на результаты HSR/airline модели. |

- Пример heading N600: «ure, and visual acuity. Surgical procedures before and during the study period, and their complications were evaluated. RESULTS: The mean follow-up was 125.0±52.3 (median, 116.5) months. Twenty of 40 eyes had a follow-up of at least 10 years. The »
- Пример generic reference N793: «per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS»

## Другие обязательные кейсы

### convergence

Вводятся разные definitions/hypothesis, затем отдельно выводы об exchange, interest и inflation rates. Один общий Convergence связывает контексты без MERGE этих результатов. Факт положительной сходимости nominal exchange rates нельзя переносить на inflation rates, где convergence can be rejected.

- N170, `e:f58284aa87e7…`, `W2078771716`, offsets [144], types CONCEPT, max=0.827681. Head convergence внутри economic convergence: исходная цель межстранового сближения.
  Контекст: «e ERM and the feasibility of EMU is dependent upon the economies of the EC achieving the appropriate degree of economic convergence. One of the difficulties in determining this is the absence of a broadly acceptable measure of economic convergence. In»

- N172, `e:f4af9e39c557…`, `W2078771716`, offsets [323], types CONCEPT, max=0.823090. Возможные определения convergence for time series, без выбранного единственного определения.
  Контекст: « the absence of a broadly acceptable measure of economic convergence. In this paper we consider possible definitions of convergence for time series. To be useful the definition must allow for the stochastic nature of the data under consideration, and »

- N174, `e:2a066f992df0…`, `W2078771716`, offsets [495], types CONCEPT, max=0.616806. Convergence hypothesis, которую должны проверять формальные тесты.
  Контекст: «tion must allow for the stochastic nature of the data under consideration, and for the development of formal tests of a convergence hypothesis. These two considerations lead us to a preferred definition of convergence and a proposed framework for test»

- N176, `e:e2d3f6141f6f…`, `W2078771716`, offsets [581], types CONCEPT, max=0.988801. Preferred definition, предлагаемое авторами определение.
  Контекст: « development of formal tests of a convergence hypothesis. These two considerations lead us to a preferred definition of convergence and a proposed framework for testing, based on a time varying parameter formulation and estimation by Kalman filter tec»

- N181, `e:9bbb18aac533…`, `W2078771716`, offsets [811], types CONCEPT, max=0.935616. Объект proposed/other methods for testing convergence.
  Контекст: « by Kalman filter techniques. We consider the relationship between our proposed framework and other methods for testing convergence. It is argued that our method encompasses most of the tests that exist in the literature. Tests for convergence are the»

- N183, `e:4779fbb7299d…`, `W2078771716`, offsets [923], types CONCEPT, max=0.971831. Convergence в tests for exchange, interest и inflation rates.
  Контекст: «testing convergence. It is argued that our method encompasses most of the tests that exist in the literature. Tests for convergence are then applied to exchange rates, interest rates and inflation rates for a sample of countries including members of t»

- N189, `e:2e7d225bd5d9…`, `W2078771716`, offsets [1114], types CONCEPT/PROCESS, max=0.978531. Положительный/осторожный вывод о nominal exchange rates.
  Контекст: «ation rates for a sample of countries including members of the ERM. We conclude that the ERM may have helped to promote convergence in the nominal exchange rates. Although interest rate differentials have declined, convergence does not seem to be occu»

- N193, `e:296d120f732c…`, `W2078771716`, offsets [1209], types CONCEPT/PROCESS, max=0.984864. Для interest rate differentials convergence does not seem to be occurring.
  Контекст: «y have helped to promote convergence in the nominal exchange rates. Although interest rate differentials have declined, convergence does not seem to be occurring. For inflation rates convergence can be rejected. Taken together these results suggest re»

- N194, `e:e76ad5e3b054…`, `W2078771716`, offsets [1272], types CONCEPT/PROCESS, max=0.788884. Для inflation rates convergence can be rejected — отрицательный результат.
  Контекст: «tes. Although interest rate differentials have declined, convergence does not seem to be occurring. For inflation rates convergence can be rejected. Taken together these results suggest reasons for the recent disruption to the ERM and bring into quest»

### glaucoma

Различать диагноз и слово внутри compound names. Только N573/N609 предлагаются как одна disease-kind entity этого контекста. Device/outcome/surgery/medications/success не исправляются и не сливаются с диагнозом по одному head glaucoma. Full intractable glaucoma остаётся RELATED к head-span, квалификатор не стирается.

- N573, `e:4a59e7a64dfd…`, `W3101922761`, offsets [33, 1617], types CONDITION, max=0.978620. Сам диагноз intractable glaucoma в PRECIS/conclusion; original canonical уже содержит два member.
  Контекст: «PRECIS: In eyes with intractable glaucoma, drainage devices provide long-term control of intraocular pressure also after penetrating keratoplasty (PK). There is »

- N580, `e:0bfaf9171506…`, `W3101922761`, offsets [260], types CONDITION, max=0.866517. Glaucoma как модификатор внутри glaucoma drainage device; не полный device name.
  Контекст: «lasty (PK). There is a high incidence of corneal graft failure. PURPOSE: To compare very long-term results of eyes with glaucoma drainage device (GDD) after PK. METHODS: We retrospectively reviewed medical records of all patients who underwent GDD »

- N587, `e:a9e88bc46a60…`, `W3101922761`, offsets [489], types CONDITION, max=0.846223. Glaucoma как модификатор внутри Glaucoma outcome.
  Контекст: « who underwent GDD placement after PK at our institution between 2001 and 2017. Forty eyes of 40 patients were studied. Glaucoma outcome was assessed by postoperative intraocular pressure (IOP), number of medications, and need for further glaucoma »

- N592, `e:39ce5f7a0099…`, `W3101922761`, offsets [608], types CONDITION, max=0.922511. Glaucoma как модификатор внутри further glaucoma surgery.
  Контекст: « Glaucoma outcome was assessed by postoperative intraocular pressure (IOP), number of medications, and need for further glaucoma surgery. Corneal outcome was assessed by graft rejection, failure, and visual acuity. Surgical procedures before and du»

- N604, `e:7bd6610cebd0…`, `W3101922761`, offsets [1009], types CONDITION, max=0.901132. Glaucoma как модификатор medications при preoperative IOP.
  Контекст: « follow-up of at least 10 years. The mean preoperative IOP was 34.0±8.3 (median, 32.0) mm Hg with 3.2±1.3 (median, 3.5) glaucoma medications. At last postoperative follow-up, the mean IOP decreased to 12.7±4.9 (median, 14.0) mm Hg with 1.0±1.2 (med»

- N606, `e:3f64a1fd7aa3…`, `W3101922761`, offsets [1147], types CONDITION, max=0.908122. Glaucoma как модификатор medications при postoperative IOP.
  Контекст: «ns. At last postoperative follow-up, the mean IOP decreased to 12.7±4.9 (median, 14.0) mm Hg with 1.0±1.2 (median, 0.0) glaucoma medications. GDD implantation successfully controlled glaucoma in 88%, 88%, 85%, 80%, 78%, 75%, and 70% of eyes, at 1, »

- N609, `e:fec086ad8409…`, `W3101922761`, offsets [1210], types CONDITION, max=0.986562. Само заболевание как объект controlled glaucoma в выводе по GDD implantation.
  Контекст: «12.7±4.9 (median, 14.0) mm Hg with 1.0±1.2 (median, 0.0) glaucoma medications. GDD implantation successfully controlled glaucoma in 88%, 88%, 85%, 80%, 78%, 75%, and 70% of eyes, at 1, 2, 3, 4, 5, 7, and 10 years, respectively. At last follow-up 68»

- N612, `e:3e1f23cd6b3d…`, `W3101922761`, offsets [1347], types CONDITION, max=0.912023. Glaucoma как модификатор success — outcome label, не тождество исходу и болезни.
  Контекст: «88%, 85%, 80%, 78%, 75%, and 70% of eyes, at 1, 2, 3, 4, 5, 7, and 10 years, respectively. At last follow-up 68% showed glaucoma success. The corneal grafts remained clear in 74%, 63%, 45%, 45%, 37%, 32%, and 26% of eyes at 1, 2, 3, 4, 5, 7, and 10»

### eyes

N579/N586 подтверждают исходную 40-eye серию. 20 long-followed eyes и доли глаз с разными outcomes не та же группа. Eyes — отдельный концепт, а не Patients даже при 40/40.

- N571, `e:e231f8c5a379…`, `W3101922761`, offsets [11], types OBJECT, max=0.959397. Обобщение Eyes with intractable glaucoma из PRECIS; конкретная мощность множества не задана.
  Контекст: «PRECIS: In eyes with intractable glaucoma, drainage devices provide long-term control of intraocular pressure also after penetrating ke»

- N579, `e:4985db187b92…`, `W3101922761`, offsets [250], types OBJECT, max=0.934042. Eyes with GDD after PK, объявленная цель исследования; далее явно серия 40.
  Контекст: «ng keratoplasty (PK). There is a high incidence of corneal graft failure. PURPOSE: To compare very long-term results of eyes with glaucoma drainage device (GDD) after PK. METHODS: We retrospectively reviewed medical records of all patients who »

- N586, `e:3a2668eee06d…`, `W3101922761`, offsets [455], types OBJECT, max=0.928966. Forty eyes of 40 patients — исходная изученная серия.
  Контекст: «ed medical records of all patients who underwent GDD placement after PK at our institution between 2001 and 2017. Forty eyes of 40 patients were studied. Glaucoma outcome was assessed by postoperative intraocular pressure (IOP), number of medic»

- N602, `e:ccd00537fdc0…`, `W3101922761`, offsets [879], types OBJECT, max=0.856073. Twenty of 40 eyes с follow-up at least 10 years.
  Контекст: «and their complications were evaluated. RESULTS: The mean follow-up was 125.0±52.3 (median, 116.5) months. Twenty of 40 eyes had a follow-up of at least 10 years. The mean preoperative IOP was 34.0±8.3 (median, 32.0) mm Hg with 3.2±1.3 (median,»

- N610, `e:c75dc53152e8…`, `W3101922761`, offsets [1263], types OBJECT, max=0.984772. Доли глаз с успешно контролируемой glaucoma при разных сроках; состава единого набора нет.
  Контекст: «.0) glaucoma medications. GDD implantation successfully controlled glaucoma in 88%, 88%, 85%, 80%, 78%, 75%, and 70% of eyes, at 1, 2, 3, 4, 5, 7, and 10 years, respectively. At last follow-up 68% showed glaucoma success. The corneal grafts rem»

- N614, `e:a72ad3305f79…`, `W3101922761`, offsets [1443], types OBJECT, max=0.978011. Доли глаз с clear grafts при разных сроках, не то же что glaucoma-controlled eyes.
  Контекст: «st follow-up 68% showed glaucoma success. The corneal grafts remained clear in 74%, 63%, 45%, 45%, 37%, 32%, and 26% of eyes at 1, 2, 3, 4, 5, 7, and 10 years, respectively. Only 7 corneal grafts (17.5%) remained clear at last follow-up. CONCLU»

### ATP

MERGE на уровне именованного молекулярного вида. OBJECT/ATTRIBUTE — несовместимая исходная типизация; предлагаем OBJECT отдельно. Endogenous/reagent pools, exposure times и процессы secretion не отождествляются.

- N089, `e:7b9c8188b695…`, `W2039836112`, offsets [254, 1362], types OBJECT, max=0.899140. 
  Контекст: «factor 1 (FGF-1), which is released in spinal cord injury, activated spinal astrocytes in culture, induced secretion of ATP, and permeabilized them to relatively large fluorescent tracers [ethidium (Etd) and lucifer yellow (LY)] through "hemic»

- N106, `e:1339994efd3f…`, `W2039836112`, offsets [877], types ATTRIBUTE, max=0.596040. 
  Контекст: « Permeabilization was reduced by apyrase (APY), an ATPase, and by P2X(7) receptor antagonists, implicating secretion of ATP and autocrine and/or paracrine action. Increased permeability of cells exposed to FGF-1 or ATP for 2 h was mediated lar»

- N109, `e:c741550b3929…`, `W2039836112`, offsets [972], types ATTRIBUTE, max=0.694920. 
  Контекст: «implicating secretion of ATP and autocrine and/or paracrine action. Increased permeability of cells exposed to FGF-1 or ATP for 2 h was mediated largely by Px1 HCs activated by P2X(7) receptors. After a 7-h treatment, the permeability was medi»

### FGF-1

Явное Fibroblast growth factor 1 (FGF-1) и устойчивое имя во всех contexts. MERGE по именованному фактору с сохранением всех ACTOR/ATTRIBUTE/CONCEPT/OBJECT source labels. Условия before/after и 2h/7h не становятся одним событием.

- N085, `e:305a23924d5a…`, `W2039836112`, offsets [116], types CONCEPT/OBJECT, max=0.471553. 
  Контекст: «Spinal astrocytes are coupled by connexin (Cx) gap junctions and express pannexin 1 (Px1) and purinergic receptors. Fibroblast growth factor 1 (FGF-1), which is released in spinal cord injury, activated spinal astrocytes in culture, induced secretion of ATP, and»

- N108, `e:99a9fa5da229…`, `W2039836112`, offsets [963], types ATTRIBUTE, max=0.621148. 
  Контекст: «gonists, implicating secretion of ATP and autocrine and/or paracrine action. Increased permeability of cells exposed to FGF-1 or ATP for 2 h was mediated largely by Px1 HCs activated by P2X(7) receptors. After a 7-h treatment, the permeability w»

- N113, `e:e13a311a5f8c…`, `W2039836112`, offsets [1126], types ACTOR/ATTRIBUTE, max=0.693435. 
  Контекст: «y Px1 HCs activated by P2X(7) receptors. After a 7-h treatment, the permeability was mediated by both Cx43 and Px1 HCs. FGF-1 also caused reduction in gap junctional communication. Botulinum neurotoxin A, a blocker of vesicular release, reduced »

- N117, `e:1d185c7ef971…`, `W2039836112`, offsets [1293, 1341], types ACTOR, max=0.655584. 
  Контекст: «ommunication. Botulinum neurotoxin A, a blocker of vesicular release, reduced permeabilization when given 30 min before FGF-1 application, but not when given 1 h after FGF-1. We infer that ATP is initially released from vesicles and then it medi»

### TKA

Полное имя и class-level TKA объединяются на уровне типа процедуры. Primary TKA в конкретной серии N402 сохраняется отдельно и связан с концептом Total knee arthroplasty. Total hip arthroplasty — другой вид операции и не объединяется.

- N392, `e:a74d56d82562…`, `W2761455282`, offsets [216], types MECHANISM, max=0.870785. 
  Контекст: «ated with a high complication rate. However, the perioperative complication and mortality rates in these patients after total knee arthroplasty (TKA) have rarely been investigated. The purposes of this study were to measure the mortality and complication rates an»

- N397, `e:34f65f3c5336…`, `W2761455282`, offsets [411], types MECHANISM, max=0.469073. 
  Контекст: «f this study were to measure the mortality and complication rates and to evaluate the 2- to 10-year clinical outcome of TKA in dialysis patients at our hospital during the past decade. METHODS: We retrospectively assessed 21 patients (26 knees»

- N402, `e:3b60d7d60948…`, `W2761455282`, offsets [568], types ACTOR, max=0.555037. TKA в primary TKA 2007–2015 у конкретных пациентов; outside modifier primary значим.
  Контекст: «tal during the past decade. METHODS: We retrospectively assessed 21 patients (26 knees) with ESRD who underwent primary TKA between 2007 and 2015. A preoperative medical evaluation was performed in all patients to assess the risk of surgery. A»

- N430, `e:0c7e451818bd…`, `W2761455282`, offsets [1363, 1537, 1787], types METHOD, max=0.892079. 
  Контекст: « discharge from hospital. At the latest follow-up, 19 patients (90.5%) had improved clinical outcomes of the knee after TKA according to the American Knee Society Scores (P < .01) and Western Ontario and McMaster Universities Arthritis Index s»

### genes

82 up-regulated, восемь tolerance-enhancing и пять successful candidates — разные наборы. Только these genes / tolerance-enhancing genes в одном предложении обозначают восемь; proposed MERGE N678/N680. Genetics как дисциплина — RELATED, не один набор genes.

- N671, `e:22632f9f229f…`, `W3138328568`, offsets [708], types CONCEPT, max=0.485305. 82 up-regulated genes при limonene exposure.
  Контекст: «ing high-titer production of limonene in engineered microbes. In this study, by using transcriptomics, we identified 82 genes from the nonconventional yeast Yarrowia lipolytica that were up-regulated when exposed to limonene. When overexpressed,»

- N678, `e:32697b3b8272…`, `W3138328568`, offsets [968], types CONCEPT, max=0.505474. These genes отсылает к 8 tolerance-enhancing candidates.
  Контекст: «candidates improved tolerance of this yeast to exogenously added limonene. To determine whether overexpression of these genes could also improve limonene production, we individually coexpressed the tolerance-enhancing genes with a limonene synth»

- N680, `e:a1cfe6b00589…`, `W3138328568`, offsets [1066], types CONCEPT, max=0.474246. Tolerance-enhancing genes — те же 8, которые coexpressed with synthase gene.
  Контекст: «erexpression of these genes could also improve limonene production, we individually coexpressed the tolerance-enhancing genes with a limonene synthase gene. Indeed, expression of 5 of the 8 candidate genes enhanced limonene production in Y. lipo»

- N684, `e:199ca809b8b6…`, `W3138328568`, offsets [1146], types CONCEPT, max=0.488093. 5 of the 8 candidate genes с улучшенным limonene production; отдельный поднабор.
  Контекст: «lly coexpressed the tolerance-enhancing genes with a limonene synthase gene. Indeed, expression of 5 of the 8 candidate genes enhanced limonene production in Y. lipolytica . Particularly, overexpressing YALI0F19492p led to an 8-fold improvement »

### breast cancer

N002 — болезнь; N007 вложен в breast cancer risk, граница требует REVIEW. Breast cancer vs breast cancer risk — RELATED, а general cancer vs esophageal cancer не равны. Одинаковый CONDITION не разрешает identity.

- N002, `e:3672e490ca33…`, `W1979727947`, offsets [81, 253], types CONDITION, max=0.997830. Заболевание breast cancer; два source members с risk factor/biomarker contexts.
  Контекст: «INTRODUCTION: Mammographic density (MD) is a strong, independent risk factor for breast cancer, but measuring MD is time consuming and reader dependent. Objective MD measurement in a high-throughput fashion would e»

- N007, `e:078a0d999a4a…`, `W1979727947`, offsets [600], types CONDITION, max=0.968120. Вложенный аргумент breast cancer внутри breast cancer risk; boundary level требует REVIEW.
  Контекст: «ribe measures that incorporate additional features of mammographic images for improving the risk associations of MD and breast cancer risk. METHODS: We randomly partitioned our dataset into a training set for model building (733 cases, 748 controls) and»

## Синонимы, аббревиатуры и корректные/ошибочные объединения

| Первый референт | Второй | Решение | Основание |
|---|---|---|---|
| N001 Mammographic density | N006 MD | MERGE | Mammographic density (MD) определено в начале того же документа; последующие MD обозначают эту характеристику, не отдельные значения измерений. |
| N083 pannexin 1 | N111 Px1 | MERGE | pannexin 1 (Px1) задано явно; Px1 в Px1 HCs обозначает тот же именованный компонент, не самостоятельного ACTOR. |
| N845 adaptive region proposal network | N846 ARPN | MERGE | Предложен один novel adaptive region proposal network (ARPN); все последующие ARPN ссылаются на него, без версий или альтернативных моделей с этим именем. |
| N833 feature pyramid network | N840 FPN | MERGE | Явное feature pyramid network (FPN) и FPN backbone того же текста; конфликт MECHANISM/ACTOR относится к типизации архитектуры. |
| N387 end-stage renal disease | N444 ESRD | MERGE | Явное end-stage renal disease (ESRD) в background и ESRD в выводе задают один вид заболевания; конкретные ESRD patients не объединяются с диагнозом. |
| N446 chronic rhinosinusitis with nasal polyposis | N464 CRSwNP | MERGE | Явное chronic rhinosinusitis with nasal polyposis (CRSwNP) и повторные disease-labels; ACTOR/OBJECT не создают отдельные болезни или людей. Узлы patients with CRSwNP сюда не входят. |
| N716 Politeknik Negeri Madiun | N735 PNM | MERGE | PNM в том же описании SPMI и доменах student.pnm.ac.id / imonev.pnm.ac.id отсылает к названному Politeknik Negeri Madiun; конфликт DATA_RESOURCE/ACTOR не обозначает другой вуз. |
| N620 Recombinant human erythropoietin | N648 rHuEPO | MERGE | Явное Recombinant human erythropoietin (rHuEPO); rHuEPO как comparator — тот же исходный препарат, не мутанты или anti-rHuEPO antibodies. |
| N641 HLA-DRB1*09 | N651 HLA-DRB1*09 allele | MERGE | HLA-DRB1*09 как распознаваемый allele, the allele и HLA-DRB1*09 allele прямо связываются текстом; span, явно названный gene, исключён. |
| N636 HLA-DRB1*09 | N651 HLA-DRB1*09 allele | REVIEW | Текст колеблется между HLA-DRB1*09 gene и allele. Внешняя биология не используется для молчаливого устранения этой неточности источника. |
| N392 total knee arthroplasty | N402 TKA | RELATED | Primary TKA в series 2007–2015 — конкретно специфицированные операции; общий вид TKA не отождествляется с ними, хотя оба контекста связаны концептом процедуры. |
| N575 intraocular pressure | N603 IOP | SHARED_CONCEPT | IOP раскрыто как intraocular pressure, но class/endpoint и pre/post численные измерения различаются. Аббревиатура не стирает timepoint. |
| N603 IOP | N605 IOP | SHARED_CONCEPT | 34.0±8.3 до операции и 12.7±4.9 после операции — разные точки времени; одинаковая аббревиатура не повод MERGE. |
| N044 Cyclotella sp. WC03_2 | N045 Cyclotella sp. L04_2 | RELATED | Два явно разных strain IDs Cyclotella WC03_2 и L04_2 имеют полезную близость как исследованные штаммы, но не тождество. Прежний 2C REJECT относился к identity, не запрещает RELATED. |
| N010 training set | N009 dataset | RELATED | Training set — часть исходного dataset (733 cases, 748 controls); dataset также имеет test partition, поэтому целое и часть не равны. |
| N782 high-speed rail (HSR) | N783 high-speed rail | MERGE | Два span в high-speed rail (HSR) обозначают один транспортный mode/sector. Конкретные поезда или перевозчики не идентифицируются. |
| N299 grains | N485 grain | REJECT | Grains Co-Fe-Zr в материале и grain как сельскохозяйственная продукция — разные значения слова. |

Раскрытие аббревиатуры само по себе задаёт лексическое значение, но не identity любых численных результатов или наборов. Поэтому class-level MD/FGF-1/ARPN и именованное учреждение PNM могут иметь положительное основание, а pre-/postoperative IOP, instrument vs scores и HLA gene vs allele — разные результаты. Полные evidence, source mention IDs и точные source fragments для каждой пары — в `resolution_decisions.csv`.

## Влияние типизации и старых factual edges

Из 96 exact-name групп **28** имеют различающиеся наборы исходных типов, **29** — хотя бы одно многотипное извлечение. В предложенных 69 MERGE-группах **18** имеют type-set conflict, **15** содержат multi-type members (счётчики пересекаются). Strict same-single-type gate 1B заблокировал бы такие случаи; считать их разными объектами только по label тоже неверно.

Явные примеры: FGF-1 помечен ACTOR/ATTRIBUTE, ARPN — ACTOR/CONCEPT, PNM — DATA_RESOURCE, PBMC — DATA_RESOURCE/OBJECT, CRSwNP — ACTOR/CONDITION/OBJECT, qualified patients with CRSwNP — CONDITION. Текст определяет соответственно фактор, модель, вуз, клетки, заболевание и людей. Original types не исправлены в provenance; proposed_primary_type указан отдельно в MERGE-группе. Неизвестный тип ERM оставлен без обязательной предлагаемой primary label.

**14 MERGE-групп создают 27 новых factual-loop records.** Проверены исходные neighbourhoods и все новые loop cases; `new_loop_diagnostics.csv` содержит каждую запись. Среди примеров — PNM HAS_MEMBER Politeknik Negeri Madiun, HSR COMPARED_WITH high-speed rail и ARPN COMPARED_WITH ARPN. Контекстное тождество имени не доказывает корректность этих машинных отношений. Их нельзя молча удалить, объявить новыми фактами или использовать как доказательство плохого/хорошего ER.

Пример genes: source HAS_PART между tolerance-enhancing genes и limonene synthase gene подозрителен, поскольку текст говорит coexpressed with; ER сохраняет его как исходное предсказание, а не превращает в факт общего концепта Genes. Исправление ReLEx в этот этап не входит.

## Предлагаемые общие концепты

Это локальные concept_id проекта без внешнего ontology ID. Все новые связи имеют отдельный тип HAS_SHARED_CONCEPT, factual=false, identity=false, fact_inheritance_allowed=false. У концепта нет собственного document_id; число документов в компонентах считает только реальные source documents.

| Concept | Contextual targets после MERGE | Документов | Назначение |
|---|---|---|---|
| Patients | 16 | 5 | Класс пациентов. Независимые исследования, общие популяции и подгруппы сохраняют собственные ID, состав, времена и факты. |
| Convergence | 11 | 1 | Общая научная тема economic/time-series convergence. Определения, гипотезы, разные ряды и положительные/отрицательные результаты не тождественны. |
| Genes | 3 | 1 | Общий биологический класс genes; контекстные наборы 82, 8 и 5 генов остаются разными. |
| Eyes | 5 | 1 | Глаз как анатомический объект/класс; исходная серия, долгосрочная подгруппа и различные outcome groups сохраняются. |
| Annual papal medals | 4 | 1 | Класс annual papal medals; 94 медали и generic sede-vacante examples не превращаются в одну коллекцию. |
| B cells | 3 | 1 | Общий B-cell класс; tissue compartments, модификаторы и клеточные подгруппы сохраняются. |
| Bioactivity | 2 | 1 | Биоактивность как оцениваемое свойство, отдельно от результатов разных mutant sets. |
| Brightness temperature | 3 | 1 | Физический параметр; mean disk, quadrant vector и maximum остаются отдельными измерениями с разными значениями. |
| Copy number variation | 5 | 1 | Класс CNV; observed variant sets, regions и data sets различаются; контекстная связь не объявляет data set отдельным вариантом. |
| Corneal grafts | 2 | 1 | Класс corneal grafts; общий набор трансплантатов и семь surviving grafts не отождествляются. |
| Degree of polarization | 3 | 1 | Измеряемый физический параметр, отдельно от spatial aggregation и profile observations. |
| Diatom plastid genomes | 5 | 1 | Класс пластидных геномов диатомей; набор семи, subset Thalassiosirales и общие выводы не имеют одинакового состава. |
| Diatoms | 2 | 1 | Класс диатомей; other diatoms в конкретном сравнении сохраняет отличие от общего класса. |
| Energy consumption | 3 | 2 | Один физический параметр энергопотребления; WSN и chemical synthesis имеют разные носители, единицы нормирования и экспериментальные утверждения. |
| Exchange rates | 3 | 1 | Общий класс экономических рядов exchange rates; nominal specificity и результаты тестирования не стираются. |
| Farmers | 2 | 1 | Класс носителей экономических характеристик; текст не устанавливает один и тот же индивидуальный состав популяции во всех временных утверждениях. |
| Gap junctions | 3 | 1 | Класс структуры gap junction; connexin/pannexin и различие mammalian/invertebrate contexts сохраняются. |
| Glaucoma drainage devices | 2 | 1 | Класс GDD. Не отождествляет физические имплантаты, хирургические вмешательства и общий вывод об эффективности. |
| Gene expression | 3 | 1 | Общий научный параметр/процесс, отдельно от измеренных associations, отрицательного результата и future work. |
| Green banks | 3 | 1 | Класс финансовых институтов; конкретные наборы инвестирующих банков и их активы не становятся одной организацией. |
| Hemichannels | 3 | 1 | Класс HCs; конкретные molecular families и временные channel pools не отождествляются. |
| Income | 3 | 1 | Общий экономический показатель; носитель residents/farmers, период и модальность каждого contextual entity сохраняются. |
| Intraocular pressure | 4 | 1 | Общий параметр IOP; preoperative 34.0 и postoperative 12.7 остаются разными измерениями, не фактами о concept node. |
| Medical complications | 3 | 2 | Общий класс клинических осложнений; реальные события TKA и другой GDD/PK-серии и общий риск разделены. |
| Mortality | 4 | 1 | Общий показатель mortality; perioperative risk/endpoint и one-year observed death не смешиваются. |
| Region proposal boxes | 3 | 1 | Класс region proposals; boxes разных моделей и разные наборы/условия генерации не имеют общей идентичности. |
| Pearson correlation coefficient | 2 | 1 | Статистика Pearson r как общий концепт; определение и конкретная оценка r=0.884 остаются разными уровнями. |
| Stress reactivity | 2 | 1 | Общий исследуемый trait; clinical/pathological scope и mouse-line selection conditions сохраняются. |
| Stroke | 2 | 1 | Общий диагноз/тип события; fatal stroke одного пациента и два strokes в осложнениях не объявляются одним событием. |
| Sublayer thickness | 5 | 1 | Толщина слоёв конкретного multilayer-исследования как параметр; значения larger/smaller и material phases сохраняются. Не связана автоматически с regolith thickness. |
| Cardiac implantable electronic devices | 2 | 1 | Общий класс CIED; определение device kind и устройства серии 16 пациентов сохраняют разные contextual IDs. |
| Total knee arthroplasty | 2 | 1 | Общий вид процедуры TKA; class-level labels и primary operations конкретной серии разделены. Не включает total hip arthroplasty. |

Всего **118** context→concept связей; только **3** концепта охватывают несколько документов: Patients (5), Energy consumption (2) и Medical complications (2). Остальные сохраняют различия контекстов внутри одной работы и могут быть проверены как будущие глобальные классы. Общих Results, Paper, Study, Author, Reliability или Resolution не создаётся.

## Симуляция структуры

Все сценарии — только in-memory bookkeeping; ни один граф не импортирован. Направления игнорируются для слабых компонент; петли сохраняются и дают ненулевую степень; документы/Mentions не входят. Для ties используется минимальный ID, а число документов вычисляется из provenance contextual members, без фиктивного документа концепта.

| Сценарий | Вершин | Изоляты | WCC | Крупнейшая | Документов в ней |
|---|---|---|---|---|---|
| Текущий baseline | 905 | 175 | 230 | 378 | 17 |
| Только MERGE, все старые SIM | 811 | 172 | 225 | 346 | 17 |
| MERGE + concepts, все старые SIM | 843 | 172 | 223 | 369 | 17 |
| Proposed display + concepts | 843 | 181 | 239 | 132 | 5 |

Последний display-view сохраняет все 1307 factual records (1076 структурных buckets), добавляет 118 conceptual edges и оставляет **150** SIMILAR_TO (80 RELATED +70 pending REVIEW). Остальные direct semantic edges представлены audit-статусами: 72 identity-redundant, 233 replaceable-by-concept-path, 37 rejected. Все 492 оригинальных semantic records остаются восстановимы; это предложение отображения, не фактическое удаление из Neo4j.

Верхние два proposed views намеренно разделены: один показывает чистый эффект adding concepts при всех старых semantic edges, другой — отдельный эффект предложенной содержательной классификации direct links. Уменьшение числа WCC при добавлении hub-концепта — топологический эффект абстракции; оно не создаёт новых подтверждений фактов. Рост изоляции в proposed display показывает, что часть прежней связанности давали отклонённые generic связи. Ни 905→811, ни рост/падение largest component не являются оценкой улучшения качества.

### Сохранность фактических отношений

| Метрика | Значение |
|---|---|
| Сохранённых canonical RELEX_REL records | 1307 |
| Вложенных raw source triple IDs | 1490 |
| Уникальных (document,head,type,tail) после mapping | 1076 |
| Групп параллельных отношений | 164 |
| Дополнительных записей в этих группах | 231 |
| Исходных / итоговых loop records | 12 / 39 |
| Новых loop records / уникальных loop structures | 27 / 26 |
| Фактических отношений на concept nodes | 0 |
| Междокументных factual edges после mapping | 0 |

`preview_factual_relations.jsonl` хранит full original_relations для каждого структурного bucket: исходные canonical IDs, source triples, confidence, evidence и provenance. Распаковка строго восстанавливает все 1307 исходных объектов и 1490 raw source relations. Структурная дедупликация не удаляет записи, не суммирует confidence и не порождает новые scientific facts.

### Смысл междокументных переходов

| Решение по существующим cross-document SIMILAR_TO | Число |
|---|---|
| MERGE | 0 |
| SHARED_CONCEPT | 77 |
| RELATED | 13 |
| REJECT | 30 |
| REVIEW | 1 |

90 положительно обоснованных переходов — это 77 классово-концептуальных и 13 тематически RELATED кандидатов; 66 из них одноимённые. RELATED, например genetics↔genes или breast cancer↔cancer, сохраняет различие референтов/специфичности, а не поддерживает MERGE. Один pending cross-document case — неизвестные author из разных публикаций: без ORCID/имён нельзя доказать ни тождество, ни различие людей.

В proposed display 77 прямых SHARED_CONCEPT links заменимы путями через соответствующие абстракции; они не считаются 77 фактическими междокументными отношениями. 13 RELATED links остаются прямыми, REVIEW author остаётся отдельно pending. Число всех возможных пар через концептуальный hub не выдаётся за число независимых доказательств. Старые CROSS_DOC_RE_CANDIDATE из отдельного эксперимента не входят в baseline и не добавлялись.

## Полный каталог 96 групп

M/S/R/X/? в колонке решений означают MERGE / SHARED_CONCEPT / RELATED / REJECT / REVIEW и показывают число пар, а не узлов. Полные ID, все member contexts, source offsets/types/confidence/provenance и incident factual IDs сохранены в duplicate_groups.csv.

| Группа | Имя | Вершин / док. | Решения пар | Конкретное основание |
|---|---|---|---|---|
| G001 | airlines | 2 / 1 | M:1 | В этой статье оба употребления обозначают один обсуждаемый класс авиаперевозчиков в модели HSR/aviation, не отдельные компании. |
| G002 | american knee society scores | 2 / 1 | M:1 | Одна именованная шкала оценки коленного сустава; P<.01 относится к оценённым исходам, а не определяет вторую шкалу. |
| G003 | annual medals | 4 / 1 | S:6 | 94 ежегодные медали, две generic annual medals и общее определение класса имеют разный объём; не одна коллекция. |
| G004 | arpn | 3 / 1 | M:3 | Введён ровно один proposed ARPN; METHOD/ACTOR/CONCEPT расходятся из-за типизации, а не из-за разных моделей. |
| G005 | atp | 3 / 1 | M:3 | Один именованный молекулярный вид ATP, обсуждаемый как выделяемый/применяемый агент; не тождество молекул или отдельных пулов ATP. |
| G006 | author | 2 / 2 | ?:1 | Самоупоминания авторов двух разных публикаций о green banking и synodality; имён/ORCID нет. Ни тождество, ни различие людей не доказаны. |
| G007 | b cells | 3 / 1 | S:3 | Частота B-клеток в nasal polyps, общий состав и edaphic B cells не задают одну клеточную популяцию; общий класс B cells допустим. |
| G008 | bioactivity | 2 / 1 | S:1 | Предсказываемая сохранность bioactivity у 17 мутантов и наблюдение пяти мутантов относительно rHuEPO относятся к разным наборам/оценкам. |
| G009 | breast cancer | 2 / 1 | ?:1 | Первый узел обозначает заболевание; второй span входит в breast cancer risk. Требуется решение о самостоятельности вложенного аргумента, а не слияние риска и болезни. |
| G010 | brightness temperature | 3 / 1 | S:3 | Среднее по диску 233 K, четыре квадрантных средних и максимум 257 K — разные измерительные результаты. |
| G011 | camera | 2 / 1 | ?:1 | fast camera — роль в диагностической схеме; камера с указанными 40 500 frames/s дана как example. Нельзя автоматически приравнять роль конкретному примеру. |
| G012 | cancer | 2 / 2 | R:1 | В rHuEPO перечислена общая категория cancer, в RT-когорте esophageal cancer; внешнее уточнение органа нельзя отбрасывать. |
| G013 | cnv | 2 / 1 | S:1 | CNV входит в 100 CNV regions и CNV data sets; это не два явно одинаковых варианта или один и тот же набор данных. |
| G014 | cnvs | 3 / 1 | S:3 | Определение copy number variants, регионы различий mouse lines и найденные CNVs имеют общий класс, но разные контекстные объёмы. |
| G015 | co0� 45fe0� 45zr0� 1 | 3 / 1 | ?:3 | U+FFFD повреждает состав; контексты различают исходные sublayers, образованные grains и весь sublayer. Повреждённое совпадение не идентификатор материала. |
| G016 | convergence | 9 / 1 | S:36 | Общий концепт convergence объединяет тему, но preferred definition, hypothesis и результаты для exchange/interest/inflation rates не тождественны. |
| G017 | cooperation | 2 / 2 | R:1 | Частные инвесторы/developers/green banks и aviation/HSR — разные взаимодействия; сходство общего глагольного класса не даёт идентичности. |
| G018 | corneal grafts | 3 / 1 | M:1, S:2 | Общая серия трансплантатов и 7 оставшихся прозрачными grafts различаются по составу; заключение о низкой survival относится ко всей серии. |
| G019 | crswnp | 3 / 1 | M:3 | CRSwNP явно раскрыто как chronic rhinosinusitis with nasal polyposis; это одна категория болезни, не сами пациенты. |
| G020 | degree of polarization | 3 / 1 | S:3 | Общие 6%, радиальный профиль и неоднородность вдоль limb — разные характеристики одного физического параметра. |
| G021 | demand line | 2 / 1 | R:1 | Наблюдавшийся сдвиг demand line и рекомендация should move farther вправо-вверх имеют разные модальности/времена. |
| G022 | destruction | 2 / 1 | R:1 | Угроза семье Lehi и избегаемое уничтожение Nephite race различаются носителями и повествовательным масштабом. |
| G023 | dialysis patients | 3 / 1 | S:3 | Наблюдавшиеся пациенты больницы, общая применимость TKA и будущие кандидаты на TKA — разные уровни популяции. |
| G024 | diatom plastid genomes | 2 / 1 | S:1 | Шесть геномов при обсуждении ilvB/ilvH и общий класс в выводе о потере photosynthetic genes не одна именованная выборка. |
| G025 | diatoms | 2 / 1 | S:1 | Начальный общий класс Diatoms и other diatoms в сравнении с Thalassiosirales не одно множество организмов. |
| G026 | economic convergence | 2 / 1 | S:1 | Условие осуществимости EMU и проблема отсутствия общепринятого измерителя относятся к общей теме, но не одному результату измерения. |
| G027 | endophenotypes | 2 / 1 | ?:1 | В работе две линии с разными anxiety/stress extremes; these endophenotypes не уточняет равенство наборов. Дополнительно расходится многотипизация. |
| G028 | energy consumption | 3 / 2 | S:3 | Энергопотребление WSN, его снижение на 3.5% и высокое энергопотребление химического синтеза имеют общий параметр, но разных носителей/условия. |
| G029 | erm | 4 / 1 | M:6 | Одно ERM обсуждается через sustainability, members, влияние и disruption. Внешняя расшифровка не добавляется; конфликт типов не доказывает разные ERM. |
| G030 | exchange rates | 2 / 1 | S:1 | Перечень тестируемых рядов и вывод о nominal exchange rates не один результат или гарантированно одна спецификация ряда. |
| G031 | extremes | 3 / 1 | R:3 | Anxiety-related behavior и stress reactivity — разные оси отбора; extremes of respective mouse lines не один экстремум. |
| G032 | eyes | 6 / 1 | M:1, S:14 | Общий клинический класс, исходные 40 глаз, 20 с ≥10 лет follow-up и группы с контролем glaucoma/clear grafts не тождественны. |
| G033 | failure | 2 / 1 | ?:1 | Corneal graft failure в background и failure в составной оценке Corneal outcome могут относиться к одному типу исхода, но не доказанно к одному событию/измерению. |
| G034 | farmers | 2 / 1 | S:1 | Farmers как носители total income в 2003–2014 и целевая популяция рекомендации могут пересекаться; индивидуальный состав не задан. |
| G035 | fgf-1 | 3 / 1 | M:3 | Fibroblast growth factor 1 (FGF-1) задан явно. Разные временные экспозиции не превращаются в один эксперимент; объединяется именованный фактор. |
| G036 | follow-up | 2 / 2 | R:1 | Latest follow-up TKA и mean follow-up GDD/PK — независимые исследования, сроки и носители. |
| G037 | gap junctions | 2 / 1 | S:1 | Connexin junctions spinal astrocytes и замечание о pannexins in invertebrates имеют общий класс структуры, не одни физические каналы. |
| G038 | gdd | 2 / 1 | S:1 | Имплантированные GDD в серии и обобщение A GDD can control glaucoma относятся к классу устройств; конкретные имплантаты не идентифицированы. |
| G039 | gene expression | 3 / 1 | S:3 | Изменения expression, отсутствие корреляции и необходимость будущих исследований — разные утверждения о параметре, а не один результат. |
| G040 | gene order | 2 / 1 | R:1 | Сохранность gene order в таксоне и конкретный общий ancestral order трёх видов не один доказанно одинаковый порядок всех геномов. |
| G041 | genes | 4 / 1 | M:1, S:5 | 82 up-regulated genes, 8 tolerance-enhancing candidates и 5 удачных кандидатов различаются составом. Только these genes и tolerance-enhancing genes подтверждают один набор восьми. |
| G042 | genomes | 2 / 1 | S:1 | Семь перечисленных геномов и подмножество Thalassiosirales различаются; anaphoric their genomes уже сохранено внутри первого canonical node. |
| G043 | glaucoma | 8 / 1 | M:1, ?:27 | Большинство spans входят в glaucoma drainage device/outcome/surgery/medications/success; это не восемь подтверждённых тождественных клинических сущностей. |
| G044 | green banks | 3 / 1 | S:3 | Инвесторы взаимодействуют с green banks, перечислены их инструменты и joint investment; конкретный одинаковый набор банков не задан. |
| G045 | harvest paradox | 3 / 1 | M:3 | Один именованный Harvest Paradox обсуждается как явление, условие/механизм и последствие политик. CONCEPT/MECHANISM — различие типизации. |
| G046 | hcs | 2 / 1 | S:1 | На 7 h действуют и Cx43, и Px1 HCs; opening/changes in HCs не одна явно та же подгруппа каналов. |
| G047 | hla-drb1*09 | 2 / 1 | ?:1 | Первое упоминание прямо названо gene; остальные allele. Нельзя исправлять gene→allele по внешнему знанию и автоматически объединять. |
| G048 | income | 3 / 1 | S:3 | Domestic residents income, farmers total income и рекомендуемый рост farmers income имеют разных носителей/модальность. |
| G049 | indirect lightning protection | 2 / 1 | ?:1 | Название функции/подхода в proposal и shielding/filtering конкретного module не подтверждают одну операцию защиты; исходные типы различаются. |
| G050 | intraocular pressure | 2 / 1 | S:1 | Общий параметр и postoperative outcome variable не одно численное измерение; preoperative/postoperative значения сохраняются отдельно. |
| G051 | iop | 2 / 1 | S:1 | 34.0±8.3 до операции и 12.7±4.9 после операции — разные точки времени; одинаковая аббревиатура не повод MERGE. |
| G052 | knee | 2 / 1 | X:1 | Анатомическое knee в clinical outcomes и Knee внутри American Knee Society Scores не имеют основания для прямой эквивалентности. |
| G053 | lightning protection | 3 / 1 | R:3 | Защита military power station в целом и indirect lightning protection не совпадают по специфичности. |
| G054 | limonene | 2 / 1 | ?:1 | Второй span limonene находится внутри limonene synthase gene; нельзя подменять ген веществом или молча менять границы. |
| G055 | limonene production | 2 / 1 | R:1 | Общий вопрос улучшения production и улучшение у 5/8 генов описывают разные экспериментальные объёмы. |
| G056 | mammographic images | 2 / 1 | ?:1 | Упомянуты additional features и информация в images, но training/test partitions не дают основания считать эти два множества изображений идентичными. |
| G057 | market | 2 / 1 | R:1 | Market of grain purchase and sales и power of market — рынок товаров и рыночный механизм; уровни различаются. |
| G058 | medals | 2 / 1 | R:1 | 94 annual medals — часть papal commemorative medals; общий head medals не стирает родовидовое различие. |
| G059 | medical complications | 2 / 1 | S:1 | Наблюдавшиеся осложнения у шести человек и возможный риск у будущих пациентов — события и общий медицинский класс. |
| G060 | memory b cells | 2 / 1 | M:1 | Оба упоминания обозначают memory B-cell subpopulation в nasal polyps той же CRSwNP-серии, противопоставленной PBMC. |
| G061 | microbes | 2 / 1 | R:1 | Возможность engineering microbes в cell factories и engineered microbes текущей работы имеют разный уровень спецификации. |
| G062 | monetary | 2 / 1 | X:1 | Фраза monetary, monetary and prudential policies содержит повтор модификатора; самостоятельного научного референта у двух isolated adjectives не установлено. |
| G063 | mortality | 4 / 1 | S:6 | Perioperative mortality rates, цель измерить mortality, recorded outcome и 4.8% death at one year не одно измерение/событие. |
| G064 | nasal polyps | 3 / 1 | M:3 | Один polypoid-tissue compartment в сравнении с blood PBMC у десяти CRSwNP-пациентов; не утверждается тождество отдельных полипов. |
| G065 | object detection | 2 / 1 | R:1 | Один span вложен в tiny object detection, другой в two-stage object detection approaches; условия задачи различаются. |
| G066 | oral history | 2 / 1 | M:1 | Одна дисциплина Oral history в Италии: историческое развитие и отношение архивов; речь не о двух конкретных интервью. |
| G067 | order | 2 / 1 | X:1 | Ancestral gene order — порядок генов; order при synapomorphy — таксономический ранг Thalassiosirales. Это разные значения слова. |
| G068 | papal politics | 2 / 1 | M:1 | Оба употребления явно относятся к papal politics XVII века в рамках одного исследуемого периода. |
| G069 | paper | 2 / 2 | X:1 | Self-reference в статье о grain economics и в статье о LEACH относится к разным document_id/DOI; общего научного концепта Paper не предлагается. |
| G070 | patients | 14 / 5 | M:1, S:89, ?:1 | Общий Patients допустим как класс; когорты разных исследований, 21/20/6/19 пациентов и общие популяции не одна сущность. |
| G071 | patients with crswnp | 2 / 1 | M:1 | Обе формулировки анафорически обозначают тех же десятерых участников CRSwNP-исследования. CONDITION — ошибочная типизация пациентов, не болезни. |
| G072 | pbmc | 3 / 1 | M:3 | Явное раскрытие peripheral blood mononuclear cells (PBMC) и последующие сравнения относятся к одному blood-cell compartment CRSwNP-серии. |
| G073 | popes | 2 / 1 | ?:1 | Ten popes исследуемой серии и popes through XVII century могут иметь разный объём. Состав второго множества не перечислен. |
| G074 | proposal boxes | 3 / 1 | S:3 | Общий элемент two-stage detectors, boxes RPN и fixed-number outputs не одни физические/цифровые boxes нового ARPN. |
| G075 | r | 2 / 1 | S:1 | Обозначение Pearson coefficient в методах и r=0.884 во внешнем test set — определение статистики и конкретная оценка. |
| G076 | real-time diagnosis | 2 / 1 | M:1 | Один заявленный класс задач real-time diagnosis в e-health; последующий MRI case study адресует именно этот challenge, без отдельных клинических эпизодов. |
| G077 | reliability | 2 / 2 | X:1 | Надёжность lightning-protection device и достоверность oral history — разные смыслы свойства и носители; полезного прямого моста текст не подтверждает. |
| G078 | resolution | 2 / 2 | X:1 | Угловое разрешение радионаблюдений 159×87 arcsec и resolution morphology-based barcodes — разные величины и носители. |
| G079 | results | 6 / 6 | X:15 | Пять анафорических results разных работ и заголовок RESULTS не обозначают один научный результат; общего концепта Results не создаётся. |
| G080 | rhizosolenia imbricata | 2 / 1 | M:1 | В перечне исследованных видов и в сообщении о потере генов явно один названный вид; CONDITION/RESULT — ошибки типизации второго упоминания. |
| G081 | secretion | 2 / 1 | R:1 | Индукция ATP secretion FGF-1 и вывод по pharmacological antagonists — связанные процессы/доказательства, но тождество экспериментальных событий не доказано. |
| G082 | spinal astrocytes | 2 / 1 | R:1 | Общий класс spinal astrocytes и activated spinal astrocytes in culture различаются состоянием/экспериментальной популяцией. |
| G083 | stress reactivity | 2 / 1 | S:1 | Клинический trait патологической reactivity и критерий отбора extremes связаны одним научным параметром, не одним состоянием животного. |
| G084 | stroke | 2 / 1 | S:1 | Один fatal stroke через год и два strokes в перечне complications не гарантированно одна группа событий; общий диагноз Stroke сохраняется отдельно. |
| G085 | students | 2 / 1 | M:1 | Один класс пользователей PNM: возможность оценить преподавание и student.pnm.ac.id как questionnaire page; не разные перечисленные учебные когорты. |
| G086 | study | 4 / 4 | X:6 | TKA, CRSwNP, rHuEPO и limonene studies имеют разные DOI/документы, темы и участников; generic Study не даёт полезного концептуального моста. |
| G087 | sublayer thicknesses | 2 / 1 | R:1 | Параметр, определяющий material phases, и larger/smaller thicknesses в объяснении TKE — разные условия, не одна численная толщина. |
| G088 | sublayers | 2 / 1 | ?:1 | В multilayer присутствуют Co-Fe-Zr и a-Si; второй span явно ferromagnetic Co-Fe-Zr. Повреждённая формула и разные материалы не допускают exact MERGE. |
| G089 | surgery | 2 / 1 | R:1 | Риск TKA в preoperative evaluation и операция, после которой конкретный пациент умер через год, имеют разные уровни референта. |
| G090 | synthesis telescope | 2 / 1 | M:1 | Два span внутри одной фразы однозначно называют telescope Dominion Radio Astrophysical Observatory; DATA_RESOURCE/METHOD не отражает природу прибора. |
| G091 | temperature | 2 / 1 | R:1 | Temperature внутри instrument sensitivity 1.7 K и внутри lunar brightness temperature 233 K — разные параметры/результаты. |
| G092 | testing | 2 / 1 | R:1 | Proposed framework for testing и other methods for testing convergence не обозначают один тест или алгоритм. |
| G093 | thalassiosirales | 3 / 1 | M:3 | Один таксономический order в перечне геномов и выводах об ancestral order/acpP1; ACTOR здесь не указывает другого действующего субъекта. |
| G094 | thicknesses | 2 / 1 | R:1 | Общая вариация layer thicknesses и конкретные условия создания grains не одна численная величина; не объединяется с толщиной regolith. |
| G095 | tka | 3 / 1 | M:1, R:2 | TKA как тип процедуры и primary TKA у 21 участника различаются уровнем/спецификацией; ACTOR/MECHANISM/METHOD не решают вопрос идентичности. |
| G096 | turbulence | 3 / 1 | M:3 | Один обсуждаемый phenomenon turbulence около нижней x-point NSTX: prediction, visualization и localization; не заявляется один временной turbulence realization. |

## Риски, границы и проверки

- Это контекстная аналитическая классификация, не независимая экспертная gold-разметка. Не оценивались precision/recall, retrieval quality, answer faithfulness или истинность самих научных публикаций.
- Не все возможные синонимы среди 905 вершин перебраны. Candidate pool ограничен существующими связями, exact names, сохранёнными явными раскрытиями и прежними индивидуальными 2C evidence. Источники с редкими формулировками могут остаться вне pool; KEEP_SEPARATE не означает доказанную уникальность.
- Один canonical node 1B уже может содержать несколько source mentions, в том числе low-confidence members. Они не расщепляются; max_member policy исходного графа сохранена. При NER .46 присутствуют 993 source entities /1101 mentions, остальные старые пороговые предсказания не возвращались.
- Повреждённые Co–Fe–Zr формулы, uncertainty gene/allele, границы instrument/scores, whole/subset и unspecified authors не разрешаются similarity. Низкий confidence не отменяет научное содержание, высокий не доказывает identity.
- Применение 18 type-conflicted MERGE-групп потребовало бы отдельного принятия описанной политики semantic identity vs source label. Proposed_primary_type — справочное предложение, raw types не исправлялись. 14 групп, создающих новые loops, требуют проверки отношений до практической перестройки.
- Обобщённые concept nodes не получают factual edges или агрегированную достоверность исследований. Маршрут через Patients означает общую категорию и должен сохранять document/cohort context; результаты independent cohorts не наследуются.
- Модели NER/ReLEx/embeddings/LLM, embeddings/cosine и новый поиск similarity не запускались. Не создавались новые factual relations. Neo4j только читался; MERGE, смена схемы/типов и создание новой версии Global KG не выполнялись.

Автоматические проверки: все 905 ID ровно один раз; target idempotence; disjoint explicit cliques; решение MERGE по каждой паре клики; отсутствие negative/REVIEW pair, попавшей в одно identity; покрытие всех 96 групп и всех 492 semantic records; точное восстановление всех factual/source objects; отсутствие factual ends на concept nodes; source offsets/revisions и неизменность SHA-256 предыдущих артефактов. Независимый DSU повторяет все четыре component views из экспортированных файлов.

Команда воспроизведения (только локальные новые артефакты):

```powershell
.venv/Scripts/python.exe scripts/preview_global_entity_resolution.py
```

Для отдельной read-only проверки фактического baseline: тот же скрипт с `--verify-live` и существующими NEO4J_* либо `--neo4j-env <existing-file>`. Учётные данные не записываются. Подробности проверок: `validation.json`, `independent_validation.json`, `live_baseline_validation.json`.

Артефакты: обязательные `duplicate_groups.csv`, `resolution_decisions.csv`, `concept_candidates.csv`, `proposed_mapping.jsonl`, `preview_report.md`; дополнительные `review_plan.py` (явный source-bound review), `entity_inventory.csv`, `candidate_inventory.csv`, `merge_groups.json`, `typing_audit.csv`, `new_loop_diagnostics.csv`, `preview_factual_relations.jsonl`, `preview_semantic_links.jsonl`, `preview_concept_links.jsonl`, `simulation_summary.json` и файлы валидации.

**Этап 2F.2 завершён только proposed mapping и симуляцией. Изменений Neo4j и запуска этапа 2F.3 нет.**
