# Stage 2G.1 — Semantic Concept Resolution: Threshold Sweep

**22 виртуальные конфигурации проверены на тех же 30 документах. Экспериментальные результаты в Neo4j не импортировались.**

L: Stage 2G — 811 entities, 34 concepts, 130 memberships, 1307 facts, 492 SIMILAR_TO; 6 multi-document concepts, 92 entity–concept–entity bridges, 223 WCC, largest component 371.

## Результаты всех конфигураций

| Режим | Порог | Кандидаты EE/EC | Новые memberships | Новые concepts | Shared concepts | Новые bridges | Новые direct doc pairs | Новые reachable doc pairs | WCC | Largest | Подозрительные ACCEPT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text_name | 0.950 | 106/8 | 5 | 1 | 7 | 8 | 2 | 0 | 223 | 372 | 0 |
| text_name | 0.955 | 95/7 | 5 | 1 | 7 | 8 | 2 | 0 | 223 | 372 | 0 |
| text_name | 0.960 | 86/2 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 0.965 | 84/1 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 0.970 | 82/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 0.975 | 82/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 0.980 | 80/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 0.985 | 79/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 0.990 | 79/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 0.995 | 79/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name | 1.000 | 74/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.950 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.955 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.960 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.965 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.970 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.975 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.980 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.985 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.990 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 0.995 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |
| text_name_context | 1.000 | 0/0 | 0 | 0 | 6 | 0 | 0 | 0 | 223 | 371 | 0 |

При text_name 0.950 и 0.955 граф имел бы 35 concepts и 135 memberships. Все исходные 34/130 сохраняются. Новые понятия/связи:

| Сущность | Concept | Документ | Cosine | Основание |
|---|---|---|---:|---|
| Surgical procedures | surgery | W3101922761 | 0.959252 | source_grounded_procedure_paraphrase_and_scientific_role |
| surgery | surgery | W2761455282 | 0.959252 | hybrid_supported_lexical_class |
| surgery | surgery | W2761455282 | 0.959252 | hybrid_supported_lexical_class |
| breast cancer | cancer | W1979727947 | 0.956457 | source_grounded_clinical_class_with_expanded_context_witnesses |
| breast cancer | cancer | W1979727947 | 0.956457 | source_grounded_clinical_class_with_expanded_context_witnesses |

## Что дали retrieval и validator

Semantic candidate discovery добавляет путь к новому классу **surgery**: Surgical procedures ↔ surgery, cosine 0.959252. Это 2 пары из 2 документов, 3 memberships; одна membership Surgical procedures→surgery не имеет lexical token/alias overlap. Ни один вызов lexical_match не обязателен для этого semantic validation path.

Две breast cancer→Cancer связи (cosine 0.956457) были кандидатами и в Stage 2G, но отклонялись из-за недостаточных context-sense witnesses. Validator-only control восстанавливает те же 2 связи без semantic retrieval. Поэтому их нельзя целиком считать выигрышем embeddings. Исходные NER labels и source spans не исправлялись.

Полностью проверено 315954 допустимых cross-document entity pairs и 27444 nonmember entity→concept pairs на режим. При 0.950 отобрано 106 EE и 8 EC candidates. Все отобранные EE уже присутствовали среди исходных SIMILAR_TO; **новых EE-пар вне этого набора при данных порогах нет**. Мы не ограничивали ими поиск: это измеренный результат. Новизна относительно Stage 2G — class validation и явная обработка semantic-only EE proposals, а не обнаружение ранее невидимого высокого cosine.

Новые прямые пары через Cancer: W1979727947↔W3123056951 и W1979727947↔W4391351014. Surgery добавляет ещё один concept-мост между W2761455282 и W3101922761, которые были связаны через Patients/follow-up. Первой достижимости в полном KG не прибавилось; WCC 223 сохраняются. Рост largest 371→372 — не доказательство роста качества. Структурные WCC/reachability включают все неизменяемые 492 SIMILAR_TO, в том числе с прежними диагностическими REJECT/REVIEW; такая достижимость не означает доказанную научную зависимость.

## Name и name+context

Для text_name_context maximum EE cross-document cosine = 0.887371, maximum eligible EC cosine = 0.898728. Ни один кандидат не проходит 0.950–1.000. Concept representation использует исходные member snippets с leave-one-document-out; своих context snippets в reference embedding нет. Name-only fallback для single-document concepts явно помечен. Результат относится именно к этому reference recipe и corpus, а не доказывает бесполезность context embeddings вообще.

Name similarity завышает одинаковые общие слова: Study/Results/Patients получают около 1. Context может завышать co-occurrence внутри одного исследования: limonene и его synthase gene имеют context cosine 0.985701, хотя это разные объекты. Cross-document sweep исключает эту intra-document пару; отдельный обязательный probe всё равно показывает REJECT.

E5 masks/pooling/prefix/нормализация сохранены; все 7 model files совпадают со Stage 2B. Embeddings вычислены один раз на режим (733 unique name texts и 885 context texts, 811 entities+77 reference variants). Векторы и cosine matrices находятся вне Git; hashes и truncation counts — в manifest. Усечений в этом прогоне нет: maximum input 458 tokens при лимите 512. На оценки влияют reference context variants и неоднородность исследовательских тем; режимы нельзя считать одинаково калиброванными. У byte-identical vectors cosine ровно 1 по тождеству, без epsilon-округления соседних значений; коррекция сделана только по cache, без повторного E5 inference.

## Обязательный содержательный аудит

| Случай | text_name | text_name_context | Решение / смысл |
|---|---:|---:|---|
| breast cancer→Cancer | 0.956457 | 0.821099–0.830479 | 2 новых class links только при name 0.950/0.955; это восстановленные FN lexical context rule |
| esophageal cancer→Cancer | 0.906148 | 0.847593 | Принятая Stage 2G membership сохраняется при любом пороге, даже ниже нового retrieval gate |
| limonene↔limonene synthase gene | 0.892161 | 0.846019 /0.985701 | REJECT: product и gene различны; intra-document, вне основного sweep |
| patients в независимых исследованиях | ≈1.000000 | 0.831965–0.856967 | Общая clinical category допустима; 130 frozen links не переоцениваются, MERGE когорт запрещён |
| clinical follow-up | ≈1.000000 | 0.857227 | Продольное клиническое наблюдение/параметр, не универсальный мост: разные интервалы и outcomes остаются локальными |

Аудированы все 10 новых ACCEPT candidate–entity–concept decisions (5 уникальных memberships) и все 41 REJECT decision при объединении порогов. У 5 принятых memberships агентный содержательный аудит не нашёл явно ошибочной принадлежности. Это **не измеренная Precision и не независимый Gold**. 41 REJECT — количество отклонённых решений, не доказанное число false matches: сюда входят как несовместимые понятия, так и недостаточное evidence.

Показательные отклонения: hip→knee arthroplasty 0.961352; genetics→Genes 0.959849; green bank loans→Green banks 0.956395; magnetic grains↔grain production 0.980470; универсальные Study/Results около 1. Подозрительных автоматически добавленных memberships по этому аудиту: 0. Есть 1 потенциальный оставшийся FN: pathological stress reactivity→Stress reactivity (0.965276), которому строгие context witnesses не дали ACCEPT. Он сохраняется в REJECT; перед следующей версией нужен отдельный разбор.

Clinical Surgery и Cancer остаются широкими категориями. Ни состав операций/когорт, ни результаты лечения/длительности/риски не наследуются. Все новые memberships имеют factual=false, identity=false, fact_inheritance_allowed=false. Agent audit notes, исходные contexts и указания на возможные ошибки сохранены в quality_audit.csv; semantic_only_discoveries.jsonl содержит как принятые, так и отклонённые пары без lexical overlap.

## Реальные виртуальные цепочки

Полные source IDs, исходные направления RELEX_REL, raw triple IDs и обе membership-связи находятся в path_examples.jsonl. Ниже выбраны несколько цепочек с фактическими рёбрами с обеих сторон:

### Цепочка 1: surgery

A: W3101922761 — Ten-Year Outcome of Glaucoma Drainage Device Surgery After Penetrating Keratoplasty. B: W2761455282 — Perioperative Complications of Total Knee Arthroplasty in Dialysis Patients.

`Surgical procedures` → HAS_SHARED_CONCEPT → `surgery` ← HAS_SHARED_CONCEPT ← `surgery`.

Маршрут A, 4 исходных factual edges; направление обхода может отличаться от исходного:

- Surgical procedures —EVALUATED_ON→ eyes; edge `ct:20261a6638e48f7b9fc02bfd56ea6c886d802f8b41d64c405adbbe7c04cd0827`, source triples `t:83ca00b743deb4bed2d97ecc77f8d9145a03a7f09ab7fbf585afed230a68f96c`.
- GDD placement —APPLIED_TO→ eyes; edge `ct:02f974e3795ce44d36027c04cbcb2e745797806a378b6b056df796e0cbd247cf`, source triples `t:556d9adef818fd5ddde75b03dada962305567782d6d9ef93d8c342663ea15855`.
- GDD placement —DERIVED_FROM→ medical records; edge `ct:968f5c7799a6f1a11e453354fe4a2f121ed6b446c5477e4be61e5e19d941781b`, source triples `t:f719ccec5a8ad5af270d6999734219e76f6cd79f3aa7ceb715fa1f54ca80cd74`.
- Glaucoma outcome —DERIVED_FROM→ medical records; edge `ct:5d5b396b4c437d387f4cd810c04fb82cf2b8dcb61645b6561bedba2b82f667e2`, source triples `t:b62d7fc00a9197438c2703c0342b5915a0c05abf3ea1e609c5ad2e40b3a99c6b`.

Маршрут B, 4 исходных factual edges; направление обхода может отличаться от исходного:

- surgery —HAS_RESULT→ improved clinical outcomes; edge `ct:1ca18373c7a123ffce972255f33c02add68edb2fc63b79a8c62d803272c2191c`, source triples `t:8ce2f37d241ea001e28545cdd2928a21a7300f8a942e80487317cab98c2bf129`.
- improved clinical outcomes —APPLIED_TO→ dialysis patients; edge `ct:4b2cc6b7aec1c964e26cbbf8ee6755336cc74fd4a921f60c67b1a47322ae6420`, source triples `t:3deb0c96278b7902e9f3280ec814a14ffabd09209c74b88537f7289c9db82594`.
- preoperative medical evaluation —APPLIED_TO→ dialysis patients; edge `ct:bb5295ea18bc4e5aa0a0b343a1fbcc0c225aa6f42201207297b7a8007ec8d5ad`, source triples `t:a8f99c3b6fa4998da4ae05a2339d42b6eb8f46984825973ebd9056c48740917b`.
- preoperative medical evaluation —APPLIED_TO→ patients; edge `ct:18a16bb420d200fcd5fdbb631d2693c54d25395ecbc8494c18413b8f718559c7`, source triples `t:3f272fde5d3297cbd9edbacf3d4fd914d72f6be988c8359739601c17ffbbd274`.

Это семантическая цепочка между независимыми исследованиями; новая фактическая зависимость не установлена.

### Цепочка 2: surgery

A: W3101922761 — Ten-Year Outcome of Glaucoma Drainage Device Surgery After Penetrating Keratoplasty. B: W2761455282 — Perioperative Complications of Total Knee Arthroplasty in Dialysis Patients.

`Surgical procedures` → HAS_SHARED_CONCEPT → `surgery` ← HAS_SHARED_CONCEPT ← `surgery`.

Маршрут A, 4 исходных factual edges; направление обхода может отличаться от исходного:

- Surgical procedures —EVALUATED_ON→ eyes; edge `ct:20261a6638e48f7b9fc02bfd56ea6c886d802f8b41d64c405adbbe7c04cd0827`, source triples `t:83ca00b743deb4bed2d97ecc77f8d9145a03a7f09ab7fbf585afed230a68f96c`.
- GDD placement —APPLIED_TO→ eyes; edge `ct:02f974e3795ce44d36027c04cbcb2e745797806a378b6b056df796e0cbd247cf`, source triples `t:556d9adef818fd5ddde75b03dada962305567782d6d9ef93d8c342663ea15855`.
- GDD placement —DERIVED_FROM→ medical records; edge `ct:968f5c7799a6f1a11e453354fe4a2f121ed6b446c5477e4be61e5e19d941781b`, source triples `t:f719ccec5a8ad5af270d6999734219e76f6cd79f3aa7ceb715fa1f54ca80cd74`.
- Glaucoma outcome —DERIVED_FROM→ medical records; edge `ct:5d5b396b4c437d387f4cd810c04fb82cf2b8dcb61645b6561bedba2b82f667e2`, source triples `t:b62d7fc00a9197438c2703c0342b5915a0c05abf3ea1e609c5ad2e40b3a99c6b`.

Маршрут B, 4 исходных factual edges; направление обхода может отличаться от исходного:

- surgery —APPLIED_TO→ dialysis patients; edge `ct:105e74d296cfbd4cba22ed6b945e941c6507ecaaab25907ebd46862a4d80e6c8`, source triples `t:23109626a8fb06e59642ee7be345e135ab193775e8b506cd61051e82cc336764, t:5e80b5eb8c053195170a1b79b240f53908c6f23c232d5a54733e410e1eaf374e, t:c7d9dd9149d58cac96e3172eb25bea900da4a287291f5c017ae711dfda675351`.
- follow-up —APPLIED_TO→ dialysis patients; edge `ct:9d7fe3c9e13b384073b5cccfa4deaf3ac872bae4361ffdbc378b298e9b46e933`, source triples `t:5728dbef016ee4a8d6c210db55826361d69e69aa647781a1b2fb4a62febec529`.
- follow-up —APPLIED_TO→ dialysis patients; edge `ct:6d0d60d1a462b067104d054ffe998e16a65309b51df1592d1f8bffef651dd86b`, source triples `t:b0f95005706656211b4188480a10d04512f93972b279f50a1d6ebaffbf9c49c7`.
- discharge —APPLIED_TO→ dialysis patients; edge `ct:67fa118c0d86ab279c1e987ab095fb79c3773fc4ef5c494ba4d02d8718cb7ace`, source triples `t:082f2288f00af523096eb89e82abd20014b298a7dd106550759e6481c931b57a`.

Это семантическая цепочка между независимыми исследованиями; новая фактическая зависимость не установлена.

### Цепочка 3: cancer

A: W3123056951 — Modified recombinant human erythropoietin with potentially reduced immunogenicity. B: W1979727947 — High-throughput mammographic-density measurement: a tool for risk prediction of breast cancer.

`cancer` → HAS_SHARED_CONCEPT → `cancer` ← HAS_SHARED_CONCEPT ← `breast cancer`.

Маршрут A, 5 исходных factual edges; направление обхода может отличаться от исходного:

- Recombinant human erythropoietin —ASSOCIATED_WITH→ cancer; edge `ct:805cc674fcfc5df192e78da2744b64c74e588d135ca9aa9588968015cb202d4e`, source triples `t:f3007d7385cc5e7a5058fabe9459852128918300f4fb41718e57d93cce72f572`.
- engineered proteins —COMPARED_WITH→ Recombinant human erythropoietin; edge `ct:7434d0a397664db96b8fb79e5aa012e598c5c02440935d83522fcf2fac4a4878`, source triples `t:f6f6957f61998146e52a33e8a00b6476afd27b547765e7fe5f980d0ad113f495`.
- engineered proteins —COMPARED_WITH→ HLA-DRB1*09; edge `ct:35d9891dbc5c3b110ac243245824196ce0606cce90a3331f5b63b4f492db6a9f`, source triples `t:933e4b6b7f969226a13d8cbba273196c46e07901c1283958b4547a226361f929`.
- HLA-DRB1*09 —LOCATED_IN→ Thai population; edge `ct:d3293d03f00dfe687bc2f4740e25d64c2260f2beca3c05f7c22e8b288c3f3b88`, source triples `t:57e17eb3925ac5d6963f16b8364fe014508e6cc6162843e84793dffa610b0d44`.
- HLA-DRB1*09 gene —LOCATED_IN→ Thai population; edge `ct:056e9a2a4ee011159236e0e4438ddd31b05b385a23beb2a54d4bc50190cf003e`, source triples `t:23e57348c141c1f72e8072e556f879c991c9eb2139c63eef84ffce9c14217c70`.

Маршрут B, 5 исходных factual edges; направление обхода может отличаться от исходного:

- Mammographic density —ASSOCIATED_WITH→ breast cancer; edge `ct:13e8331fb20b1ed3ce991b6be9e4cde0f20d92c2e00ce868a4455a510d7e492d`, source triples `t:70b88d3bd9ba4d3b4755a2fedebdeb1f85f232a62d43817ae488c68677f8153d, t:d54f962f788601d7c897a65b750baff2bafa0954547c5fa6c444fff564290411`.
- Cumulus —MEASURES→ Mammographic density; edge `ct:2c247923d0feb464d7f39ab622a117ca8574d42d256c31a233ca91cc556fb044`, source triples `t:3f5a75b9d395531a69e6981e345cf0bbd22246e460532bbe38bee2fbaca7d0d6, t:a34852c5a194370b2726c9522b32ef8a69f0842de60759945f61f6eaea291d19`.
- Cumulus —HAS_RESULT→ high correlation; edge `ct:e41608080f2d465f2e3e78ab04ad8a2bb96d73b47bb92c1866d3638cffb8e967`, source triples `t:3cf986ff12dd1212c2c22391ddda5bdc3f79a8da655b3789137af9d3ce85396c, t:445dc3e3485e43740d2fd8039ec4f767f8f1e3d5f8d98f180d81f1d6ad7c71ae`.
- logistic regression model —HAS_RESULT→ high correlation; edge `ct:2f62a2b1dfe8a98f71d6c0aed9641020978460a0457008db2c4b632e2e693a2e`, source triples `t:004ea0f221c8db75e243d7e32a476174ca8953b4c120d9fddabd34f7721ff134`.
- logistic regression model —APPLIED_TO→ clinical practice; edge `ct:59a907252e06aa5f44e0c4b70b3671e43e2e2ec3cc1946112d4186b7599ed46e`, source triples `t:1a7ba07dffd33cc937332b77587461d05e40d932a2be0031d342d0944e9b1002`.

Это семантическая цепочка между независимыми исследованиями; новая фактическая зависимость не установлена.

### Цепочка 4: cancer

A: W4391351014 — Effects of Ionizing Radiation on Cardiac Implantable Electronic Devices (CIEDs) in Patients with Esophageal Cancer Undergoing Radiotherapy: A Pilot Study. B: W1979727947 — High-throughput mammographic-density measurement: a tool for risk prediction of breast cancer.

`esophageal cancer` → HAS_SHARED_CONCEPT → `cancer` ← HAS_SHARED_CONCEPT ← `breast cancer`.

Маршрут A, 4 исходных factual edges; направление обхода может отличаться от исходного:

- radiotherapy —APPLIED_TO→ esophageal cancer; edge `ct:b678c19bcb8c61c37482a96e9d3b8b5beb29c19914b67fb2f1112d6b7f0154f3`, source triples `t:62999a31d96142b6faafc757d9ed6fcb1ccd95897eb22fab019e95b8590457ce`.
- radiotherapy —HAS_RESULT→ registered events; edge `ct:04a2adc76a7399a6be669f164ae48117965459e09425c9887345a96541af11fa`, source triples `t:09fbe9a72f39e3425847f26a7e45ec075d84a5bd558262e0e64daf34161247d0, t:3746641acc52bb1fdfcb1f17dcb091f40d158fd5039d81d2bbdebbb0733e40ca`.
- patients —HAS_RESULT→ registered events; edge `ct:ec609e3d4bf4d4c83744bc640ea97ba1bd58d1bd685adcce1b3771baa08013fd`, source triples `t:e40dd7bdefe2e86472c82a4253cafbef93d07a841132bee278d7120ae552c862`.
- patients —HAS_ATTRIBUTE→ mean age; edge `ct:72bf78cdbf033eac174efbb1a05bfcdc5b58ea60a521c8669d4f0809056f644d`, source triples `t:518c02c8ea5a43f25f30bf84eb99b27ffe995e15bbdf015a6d71d1be6fcfa117`.

Маршрут B, 5 исходных factual edges; направление обхода может отличаться от исходного:

- Mammographic density —ASSOCIATED_WITH→ breast cancer; edge `ct:13e8331fb20b1ed3ce991b6be9e4cde0f20d92c2e00ce868a4455a510d7e492d`, source triples `t:70b88d3bd9ba4d3b4755a2fedebdeb1f85f232a62d43817ae488c68677f8153d, t:d54f962f788601d7c897a65b750baff2bafa0954547c5fa6c444fff564290411`.
- Cumulus —MEASURES→ Mammographic density; edge `ct:2c247923d0feb464d7f39ab622a117ca8574d42d256c31a233ca91cc556fb044`, source triples `t:3f5a75b9d395531a69e6981e345cf0bbd22246e460532bbe38bee2fbaca7d0d6, t:a34852c5a194370b2726c9522b32ef8a69f0842de60759945f61f6eaea291d19`.
- Cumulus —HAS_RESULT→ high correlation; edge `ct:e41608080f2d465f2e3e78ab04ad8a2bb96d73b47bb92c1866d3638cffb8e967`, source triples `t:3cf986ff12dd1212c2c22391ddda5bdc3f79a8da655b3789137af9d3ce85396c, t:445dc3e3485e43740d2fd8039ec4f767f8f1e3d5f8d98f180d81f1d6ad7c71ae`.
- logistic regression model —HAS_RESULT→ high correlation; edge `ct:2f62a2b1dfe8a98f71d6c0aed9641020978460a0457008db2c4b632e2e693a2e`, source triples `t:004ea0f221c8db75e243d7e32a476174ca8953b4c120d9fddabd34f7721ff134`.
- logistic regression model —APPLIED_TO→ clinical practice; edge `ct:59a907252e06aa5f44e0c4b70b3671e43e2e2ec3cc1946112d4186b7599ed46e`, source triples `t:1a7ba07dffd33cc937332b77587461d05e40d932a2be0031d342d0944e9b1002`.

Это семантическая цепочка между независимыми исследованиями; новая фактическая зависимость не установлена.

## Рекомендация и ограничения

Для следующего отдельного экспериментального графа предварительно рекомендован **text_name, 0.955, только с Hybrid validation**. Он сохраняет те же 5 содержательно проверенных memberships и 8 bridges, что 0.950, при 102 вместо 114 candidates. При 0.960 и выше содержательные новые bridges исчезают; одинаковые общие слова с cosine≈1 остаются. Порог выбран по сохранению проверенных научных связей и снижению нагрузки на аудит, а не по максимуму рёбер. Это exploratory выбор на малом корпусе, не независимая оценка качества; для production рекомендация не валидирована.

Для name+context в проверенном диапазоне рекомендовать порог нельзя: положительных retrieval candidates нет. Нужно отдельно калибровать другой диапазон/способ агрегирования source contexts; в этом эксперименте дополнительные пороги и модели не запускались.

E5-small достаточно для генерации части полезных кандидатов (Cancer и procedural paraphrase), но не для решения о scientific membership: она сближает разные объекты, общие слова и совместно упомянутые сущности. Нынешние данные показывают ограничения retrieval/representation и ручных правил; они не доказывают необходимость замены модели или количественное преимущество другой модели.

Все 22 варианта остаются виртуальными. Исходный Neo4j namespace, 34 concepts/130 memberships, факты, SIMILAR_TO, documents и mapping сохранены. Результаты live preservation, protected-file hashes, cache/model/code hashes и тесты — в validation.json и manifest.json. Данные не расширялись; push и import экспериментального графа не выполнялись.
