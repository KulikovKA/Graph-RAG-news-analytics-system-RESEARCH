# Stage 2G — Dynamic Global Concept Linking

Новый namespace импортирован и проверен двумя проходами.

Baseline: `relex_sem_e5_small_name_t095_ner_t046_er2f3`. Target: `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1`.

Корпус строго тот же: 30 документов Stage 2F.3. NER/RE, обучение и расширение корпуса не запускались.

| Метрика | Stage 2F.3 | Новый граф |
|---|---:|---:|
| Документы | 30 | 30 |
| Контекстные сущности | 811 | 811 |
| Концепты | 32 | 34 |
| RELEX_REL | 1307 | 1307 |
| SIMILAR_TO | 492 | 492 |
| HAS_SHARED_CONCEPT | 118 | 130 |
| RELEX_MENTIONS | 811 | 811 |
| Концепты в нескольких документах | 3 | 6 |
| Weakly connected components | 223 | 223 |
| Изолированные KG-вершины | 172 | 172 |
| Крупнейшая компонента | 369 | 371 |

Добавлено концептов: **2**, memberships: **12**.
Решения: `{"ACCEPT": 12, "REJECT": 117}`. По уточнению пользователя только ACCEPT/REJECT; недостаточные доказательства дают REJECT без ребра.

## Правила и границы доказательства

Кандидаты: исходные имена, нормализация NFC, ограниченная регулярная морфология, явно определённые в исходном тексте аббревиатуры, каталог 32 концептов, пересечения токенов имён и все 492 SIMILAR_TO. Новые модели и embeddings не вычисляются. Совпадение темы, имени, типа или cosine само по себе недостаточно.

ACCEPT требует явно названного класса/алиаса, проверенных source spans, контекстных признаков общего смысла и совместимой абстракции. Словари смысловых признаков — явные эвристики, а не независимая научная верификация. Новые концепты требуют поддержанных memberships минимум в двух документах. RESULT, общие Study/Results и неоднозначные категории отклоняются. Рёбра не переносят факты: factual=false, identity=false, fact_inheritance_allowed=false.

## Междокументные пути

Пар entity→concept←entity было 84, стало 92; добавлено 8. Это конечный подсчёт простых мостов, не всех возможных маршрутов/циклических обходов. Новых комбинаций (concept, document A, document B): 3.

Пары, впервые получившие прямой мост через общий концепт (могли быть связаны косвенно и раньше):

- `W1995028339` ↔ `W3138328568`

Полные документы, source URLs, revision IDs, offsets и текстовые evidence: `document_lock.json`, `candidate_decisions.jsonl`, `memberships.jsonl`. Все новые пути с выбранными фактическими маршрутами: `path_examples.jsonl`.

Новых пар документов, ранее вообще недостижимых через KG, — 0. Прямой новый concept-мост и первая достижимость в полном KG — разные метрики.

### Пример 1: W3101922761 ↔ W2761455282 через follow-up

A: Ten-Year Outcome of Glaucoma Drainage Device Surgery After Penetrating Keratoplasty. B: Perioperative Complications of Total Knee Arthroplasty in Dialysis Patients.

`relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:20cd845c73bfd1ef630457afd8ceda0158a4cebb0f6a404d55f970fa776f8830` → HAS_SHARED_CONCEPT → `concept:2g:71ee9b93bf4c09c81568f921` ← HAS_SHARED_CONCEPT ← `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:38a747cce1d828da4d749c98621569dde1e8372c72058e56d5d3dddc7a83d93e`.

Фактический маршрут в `W3101922761` (длина 5; обход может идти против исходного направления):

- GDD implantation —HAS_RESULT→ follow-up; IDs `e:9667ee951cad35e23036f23ba41ab3831b5bc29f92f6dba8bba0abb88e945c4c` → `e:20cd845c73bfd1ef630457afd8ceda0158a4cebb0f6a404d55f970fa776f8830`; edge `ct:573657fb02b5acfd0782413225a7f98cbd4efac694038f624b347cdfddc926aa`; raw triples `t:df6cdd746259d0f0bffc1fe03fb106e0c946e629e7523448a1d485ed896040fe`.
- GDD implantation —APPLIED_TO→ eyes; IDs `e:9667ee951cad35e23036f23ba41ab3831b5bc29f92f6dba8bba0abb88e945c4c` → `e:4985db187b92b0df25db8f7d0106140e8cbff83ccdab81ead031e0c4be5062bf`; edge `ct:47919e2cff6bce713b8e01184e2a8edbc2e66463364c02316f97352f24f535cd`; raw triples `t:097f3aa619419a98c1531fb03c9d61dca7bfaefc3423bbe1d123702f7d709457`.
- GDD placement —APPLIED_TO→ eyes; IDs `e:b137bcdd3a1135ba05552d7c0ea6dcf03eef62b32a3e731c000089884243a97c` → `e:4985db187b92b0df25db8f7d0106140e8cbff83ccdab81ead031e0c4be5062bf`; edge `ct:02f974e3795ce44d36027c04cbcb2e745797806a378b6b056df796e0cbd247cf`; raw triples `t:556d9adef818fd5ddde75b03dada962305567782d6d9ef93d8c342663ea15855`.
- GDD placement —DERIVED_FROM→ medical records; IDs `e:b137bcdd3a1135ba05552d7c0ea6dcf03eef62b32a3e731c000089884243a97c` → `e:2a32f865f8e2bf3b1b8a97400c60a51452e3243e05f6e6d3dd3984a147ca7fed`; edge `ct:968f5c7799a6f1a11e453354fe4a2f121ed6b446c5477e4be61e5e19d941781b`; raw triples `t:f719ccec5a8ad5af270d6999734219e76f6cd79f3aa7ceb715fa1f54ca80cd74`.
- Glaucoma outcome —DERIVED_FROM→ medical records; IDs `e:d978c8d1433b9fac1ab7c2a156229a2a2c4cf67a1d0437e3ab17c66668060f51` → `e:2a32f865f8e2bf3b1b8a97400c60a51452e3243e05f6e6d3dd3984a147ca7fed`; edge `ct:5d5b396b4c437d387f4cd810c04fb82cf2b8dcb61645b6561bedba2b82f667e2`; raw triples `t:b62d7fc00a9197438c2703c0342b5915a0c05abf3ea1e609c5ad2e40b3a99c6b`.

Фактический маршрут в `W2761455282` (длина 3; обход может идти против исходного направления):

- follow-up —APPLIED_TO→ dialysis patients; IDs `e:38a747cce1d828da4d749c98621569dde1e8372c72058e56d5d3dddc7a83d93e` → `e:0e64fa773f1a3f618b003c3507c25facd2e7cdec291d5e2673030a2c855ab48b`; edge `ct:6d0d60d1a462b067104d054ffe998e16a65309b51df1592d1f8bffef651dd86b`; raw triples `t:b0f95005706656211b4188480a10d04512f93972b279f50a1d6ebaffbf9c49c7`.
- preoperative medical evaluation —APPLIED_TO→ dialysis patients; IDs `e:d3ec4fb8d4e251c1697123928393930f39c54fc3f57b65f61262e34fc257f50d` → `e:0e64fa773f1a3f618b003c3507c25facd2e7cdec291d5e2673030a2c855ab48b`; edge `ct:bb5295ea18bc4e5aa0a0b343a1fbcc0c225aa6f42201207297b7a8007ec8d5ad`; raw triples `t:a8f99c3b6fa4998da4ae05a2339d42b6eb8f46984825973ebd9056c48740917b`.
- preoperative medical evaluation —APPLIED_TO→ patients; IDs `e:d3ec4fb8d4e251c1697123928393930f39c54fc3f57b65f61262e34fc257f50d` → `e:4d249c81a4ed564e403795b242430ec3af604ba07960831bc4da6100d4ae503a`; edge `ct:18a16bb420d200fcd5fdbb631d2693c54d25395ecbc8494c18413b8f718559c7`; raw triples `t:3f272fde5d3297cbd9edbacf3d4fd914d72f6be988c8359739601c17ffbbd274`.

Это семантический маршрут между исследованиями; новая фактическая зависимость не установлена.

### Пример 2: W3138328568 ↔ W1995028339 через Genes

A: Simultaneous Improvement of Limonene Production and Tolerance in Yarrowia lipolytica through Tolerance Engineering and Evolutionary Engineering. B: Conserved Gene Order and Expanded Inverted Repeats Characterize Plastid Genomes of Thalassiosirales.

`relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:11aa16588639658ad8769035b49b247a2e3aabd81dff0f78fdbf4452ff450fd2` → HAS_SHARED_CONCEPT → `concept:2f2:genes` ← HAS_SHARED_CONCEPT ← `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`.

Фактический маршрут в `W3138328568` (длина 4; обход может идти против исходного направления):

- limonene synthase gene —DERIVED_FROM→ nonconventional yeast Yarrowia lipolytica; IDs `e:11aa16588639658ad8769035b49b247a2e3aabd81dff0f78fdbf4452ff450fd2` → `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081`; edge `ct:268622ec1ed3a42a1ad10c2f16197b4aef3a002b9f12ea60fba384d99dd95558`; raw triples `t:204dd8b7595358db25dc6064e8828d37cd756a62b5af3a1d696345e926c45af3`.
- evolutionary engineering —APPLIED_TO→ nonconventional yeast Yarrowia lipolytica; IDs `e:40fa23d4e5b9b5d6d9574f2522c1f8834b38063a88831bb903989c03eafa02dc` → `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081`; edge `ct:c808cfd9fb836d0b35dc327e63b35da72ab11ccf2d33c5f6b4eb76e204ed88fa`; raw triples `t:3e2f16b813938bd1412e238e29f2540c54f0d7856baa37a9427a9eae284aea20`.
- study —USES→ evolutionary engineering; IDs `e:c438e0cdc9b76e3afe2b49791bf395c433fc6b8f3c8d61659718ad59acb28b32` → `e:40fa23d4e5b9b5d6d9574f2522c1f8834b38063a88831bb903989c03eafa02dc`; edge `ct:f6fe1868634d776391a7b48861ca4bbfcd8e78416f0034b4aff4c8ed3287a314`; raw triples `t:44c67499bbf0a0259f9da10ff1f577c1113db568609749c88810d5cf1de6f6ed`.
- study —USES→ transcriptomics; IDs `e:c438e0cdc9b76e3afe2b49791bf395c433fc6b8f3c8d61659718ad59acb28b32` → `e:223a8312186b5266525ea22513b37127c803e800615906b75dff9a18094c5966`; edge `ct:8f0dcbcbd3a86c419cc9382adb0ca22781f0d0ecadb4b0331a8ee51c157eba66`; raw triples `t:e3d9a54ffa890468f8fcdc4564d8c7d88abf3c96887b4fe1b2a144025c26defc`.

Фактический маршрут в `W1995028339` (длина 3; обход может идти против исходного направления):

- Thalassiosirales —HAS_ATTRIBUTE→ acpP1 gene; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`; edge `ct:9a6d07cd9fc91d4c9a9e67f49b80b1813d6e381a9a232498ecf2135c10afe122`; raw triples `t:0baf2583a37927a09801b68e1357c7702dd77ec222f8b6856ad3227ceba58461`.
- Thalassiosirales —HAS_ATTRIBUTE→ Gene content; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa`; edge `ct:0e5a8c160ac5ac320c70b30186fb47b4c9d0b310585571cece2924ef9a7106b2`; raw triples `t:42db5c08d5b65ef1ce4923f92efea2060fd177ba55000758168757aa177f1f1a`.
- Gene content —CAUSES→ expansion; IDs `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa` → `e:0a5ea201bafa16958ea8569eefc37ee43784eb8da5f6a3e846b7faaed2fa7097`; edge `ct:f9bf9b1c526d3fa4ac081ff0f3555b16dd1bc7a0e8fbf0daa84790d1ac5f3844`; raw triples `t:b8091516e68876f2f3af232f510cb582212a5f652b4b9ab28a383a474df0512c`.

Это семантический маршрут между исследованиями; новая фактическая зависимость не установлена.

### Пример 3: W3138328568 ↔ W1995028339 через Genes

A: Simultaneous Improvement of Limonene Production and Tolerance in Yarrowia lipolytica through Tolerance Engineering and Evolutionary Engineering. B: Conserved Gene Order and Expanded Inverted Repeats Characterize Plastid Genomes of Thalassiosirales.

`relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:199ca809b8b67f945ebc0db994aff7a8695b5f062357410381d3c3af21f42177` → HAS_SHARED_CONCEPT → `concept:2f2:genes` ← HAS_SHARED_CONCEPT ← `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`.

Фактический маршрут в `W3138328568` (длина 4; обход может идти против исходного направления):

- genes —DERIVED_FROM→ nonconventional yeast Yarrowia lipolytica; IDs `e:199ca809b8b67f945ebc0db994aff7a8695b5f062357410381d3c3af21f42177` → `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081`; edge `ct:300fb4245636e3f9baca820e7480eb964295d0c78de24d75e05ca22dec0cbb1e`; raw triples `t:e57a930861591675080348832cc7044901da2dc0d71cfc76ecf7b47ee8e2ccf0`.
- nonconventional yeast Yarrowia lipolytica —PRODUCES→ Limonene; IDs `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081` → `e:1d05cadd8a7c4335cbe9a8c2eead3a4d40855753b3d88a436000d1bba2adabd5`; edge `ct:0fe0c4a9a7e07fddb9c3a6b70a3490f8665c0084f4f273b7a0d8993bc4d73244`; raw triples `t:1ef48fc5496f94bfc72cb9d296231a714342f5798e103c4a0eb370f0de0cdfb8, t:b87bc6b46e84faf518a2efdddef6f306fb3e341a70166b969e02b45bd7caf089`.
- microbes —PRODUCES→ Limonene; IDs `e:1d21a3fb6aad416d430f9af7d559580ba74b9dd2b5f9aaf7ed26a6205e3fdaae` → `e:1d05cadd8a7c4335cbe9a8c2eead3a4d40855753b3d88a436000d1bba2adabd5`; edge `ct:f951a403469ab6d8d0041e3bfbfe1e6c58b6434de0c6ff7f2cf89650f13ac6a4`; raw triples `t:4034b7b222876e3d47e8d3de3307dc3ab8b1e261cdbf8c387b20c61867051504, t:890f053e60b1d1692ca910626ad411a0dadfa87ea98e4410f56874ddaac83323`.
- microbes —LOCATED_IN→ engineered microbes; IDs `e:1d21a3fb6aad416d430f9af7d559580ba74b9dd2b5f9aaf7ed26a6205e3fdaae` → `e:fa09a9cb3c7defa3671ba4e2393ef139e9bae4b230b640568f12b246ef116844`; edge `ct:8a4c9b78bddc5e9ea98bec776866c6a041fce8996ec783e1a0deb9902e525739`; raw triples `t:bf37368165823b1cf4e313ef8d69f27f9e1372d6d2ac102e8ae029d7559b5422`.

Фактический маршрут в `W1995028339` (длина 3; обход может идти против исходного направления):

- Thalassiosirales —HAS_ATTRIBUTE→ acpP1 gene; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`; edge `ct:9a6d07cd9fc91d4c9a9e67f49b80b1813d6e381a9a232498ecf2135c10afe122`; raw triples `t:0baf2583a37927a09801b68e1357c7702dd77ec222f8b6856ad3227ceba58461`.
- Thalassiosirales —HAS_ATTRIBUTE→ Gene content; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa`; edge `ct:0e5a8c160ac5ac320c70b30186fb47b4c9d0b310585571cece2924ef9a7106b2`; raw triples `t:42db5c08d5b65ef1ce4923f92efea2060fd177ba55000758168757aa177f1f1a`.
- Gene content —CAUSES→ expansion; IDs `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa` → `e:0a5ea201bafa16958ea8569eefc37ee43784eb8da5f6a3e846b7faaed2fa7097`; edge `ct:f9bf9b1c526d3fa4ac081ff0f3555b16dd1bc7a0e8fbf0daa84790d1ac5f3844`; raw triples `t:b8091516e68876f2f3af232f510cb582212a5f652b4b9ab28a383a474df0512c`.

Это семантический маршрут между исследованиями; новая фактическая зависимость не установлена.

### Пример 4: W3138328568 ↔ W1995028339 через Genes

A: Simultaneous Improvement of Limonene Production and Tolerance in Yarrowia lipolytica through Tolerance Engineering and Evolutionary Engineering. B: Conserved Gene Order and Expanded Inverted Repeats Characterize Plastid Genomes of Thalassiosirales.

`relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:22632f9f229f74e1d82cf83da079c10c9cd2d963645822e29b43764d7aab69af` → HAS_SHARED_CONCEPT → `concept:2f2:genes` ← HAS_SHARED_CONCEPT ← `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`.

Фактический маршрут в `W3138328568` (длина 4; обход может идти против исходного направления):

- genes —DERIVED_FROM→ nonconventional yeast Yarrowia lipolytica; IDs `e:22632f9f229f74e1d82cf83da079c10c9cd2d963645822e29b43764d7aab69af` → `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081`; edge `ct:0fd78aba696ec48817dbb176d11be6357c13847d0941c9a87c5a0e5ccf0fe5b1`; raw triples `t:c59c0454fad2edba93625075039fa460a47949836b516d38f3f33c406debaa93`.
- nonconventional yeast Yarrowia lipolytica —PRODUCES→ Limonene; IDs `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081` → `e:1d05cadd8a7c4335cbe9a8c2eead3a4d40855753b3d88a436000d1bba2adabd5`; edge `ct:0fe0c4a9a7e07fddb9c3a6b70a3490f8665c0084f4f273b7a0d8993bc4d73244`; raw triples `t:1ef48fc5496f94bfc72cb9d296231a714342f5798e103c4a0eb370f0de0cdfb8, t:b87bc6b46e84faf518a2efdddef6f306fb3e341a70166b969e02b45bd7caf089`.
- microbes —PRODUCES→ Limonene; IDs `e:1d21a3fb6aad416d430f9af7d559580ba74b9dd2b5f9aaf7ed26a6205e3fdaae` → `e:1d05cadd8a7c4335cbe9a8c2eead3a4d40855753b3d88a436000d1bba2adabd5`; edge `ct:f951a403469ab6d8d0041e3bfbfe1e6c58b6434de0c6ff7f2cf89650f13ac6a4`; raw triples `t:4034b7b222876e3d47e8d3de3307dc3ab8b1e261cdbf8c387b20c61867051504, t:890f053e60b1d1692ca910626ad411a0dadfa87ea98e4410f56874ddaac83323`.
- microbes —LOCATED_IN→ engineered microbes; IDs `e:1d21a3fb6aad416d430f9af7d559580ba74b9dd2b5f9aaf7ed26a6205e3fdaae` → `e:fa09a9cb3c7defa3671ba4e2393ef139e9bae4b230b640568f12b246ef116844`; edge `ct:8a4c9b78bddc5e9ea98bec776866c6a041fce8996ec783e1a0deb9902e525739`; raw triples `t:bf37368165823b1cf4e313ef8d69f27f9e1372d6d2ac102e8ae029d7559b5422`.

Фактический маршрут в `W1995028339` (длина 3; обход может идти против исходного направления):

- Thalassiosirales —HAS_ATTRIBUTE→ acpP1 gene; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`; edge `ct:9a6d07cd9fc91d4c9a9e67f49b80b1813d6e381a9a232498ecf2135c10afe122`; raw triples `t:0baf2583a37927a09801b68e1357c7702dd77ec222f8b6856ad3227ceba58461`.
- Thalassiosirales —HAS_ATTRIBUTE→ Gene content; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa`; edge `ct:0e5a8c160ac5ac320c70b30186fb47b4c9d0b310585571cece2924ef9a7106b2`; raw triples `t:42db5c08d5b65ef1ce4923f92efea2060fd177ba55000758168757aa177f1f1a`.
- Gene content —CAUSES→ expansion; IDs `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa` → `e:0a5ea201bafa16958ea8569eefc37ee43784eb8da5f6a3e846b7faaed2fa7097`; edge `ct:f9bf9b1c526d3fa4ac081ff0f3555b16dd1bc7a0e8fbf0daa84790d1ac5f3844`; raw triples `t:b8091516e68876f2f3af232f510cb582212a5f652b4b9ab28a383a474df0512c`.

Это семантический маршрут между исследованиями; новая фактическая зависимость не установлена.

### Пример 5: W3138328568 ↔ W1995028339 через Genes

A: Simultaneous Improvement of Limonene Production and Tolerance in Yarrowia lipolytica through Tolerance Engineering and Evolutionary Engineering. B: Conserved Gene Order and Expanded Inverted Repeats Characterize Plastid Genomes of Thalassiosirales.

`relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:32697b3b8272542d2b8784c8ed71c0a41ede3e56f129d173ef3df0ce0b6c3c0d` → HAS_SHARED_CONCEPT → `concept:2f2:genes` ← HAS_SHARED_CONCEPT ← `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1:e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`.

Фактический маршрут в `W3138328568` (длина 4; обход может идти против исходного направления):

- genes —DERIVED_FROM→ nonconventional yeast Yarrowia lipolytica; IDs `e:32697b3b8272542d2b8784c8ed71c0a41ede3e56f129d173ef3df0ce0b6c3c0d` → `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081`; edge `ct:0ed7a1022ca2315e7132a0948e699535062a509d5a0097817fdee858529c650e`; raw triples `t:19b1003624a0f9293032f2c629c3a3d3996189e997db43a36a16ea2af91acf37`.
- evolutionary engineering —APPLIED_TO→ nonconventional yeast Yarrowia lipolytica; IDs `e:40fa23d4e5b9b5d6d9574f2522c1f8834b38063a88831bb903989c03eafa02dc` → `e:a819a412a4761dd0023cc77f7c4b71c9c375840116cfd76d9e344a736ba9f081`; edge `ct:c808cfd9fb836d0b35dc327e63b35da72ab11ccf2d33c5f6b4eb76e204ed88fa`; raw triples `t:3e2f16b813938bd1412e238e29f2540c54f0d7856baa37a9427a9eae284aea20`.
- study —USES→ evolutionary engineering; IDs `e:c438e0cdc9b76e3afe2b49791bf395c433fc6b8f3c8d61659718ad59acb28b32` → `e:40fa23d4e5b9b5d6d9574f2522c1f8834b38063a88831bb903989c03eafa02dc`; edge `ct:f6fe1868634d776391a7b48861ca4bbfcd8e78416f0034b4aff4c8ed3287a314`; raw triples `t:44c67499bbf0a0259f9da10ff1f577c1113db568609749c88810d5cf1de6f6ed`.
- study —USES→ transcriptomics; IDs `e:c438e0cdc9b76e3afe2b49791bf395c433fc6b8f3c8d61659718ad59acb28b32` → `e:223a8312186b5266525ea22513b37127c803e800615906b75dff9a18094c5966`; edge `ct:8f0dcbcbd3a86c419cc9382adb0ca22781f0d0ecadb4b0331a8ee51c157eba66`; raw triples `t:e3d9a54ffa890468f8fcdc4564d8c7d88abf3c96887b4fe1b2a144025c26defc`.

Фактический маршрут в `W1995028339` (длина 3; обход может идти против исходного направления):

- Thalassiosirales —HAS_ATTRIBUTE→ acpP1 gene; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:b9a8c3c64d4d443e08339142d1811a97a6cb51dc4dd7ae6a894f248c3e3720c2`; edge `ct:9a6d07cd9fc91d4c9a9e67f49b80b1813d6e381a9a232498ecf2135c10afe122`; raw triples `t:0baf2583a37927a09801b68e1357c7702dd77ec222f8b6856ad3227ceba58461`.
- Thalassiosirales —HAS_ATTRIBUTE→ Gene content; IDs `e:31bd290a905e51076aeccdad12b4da52794de1c0cfb6ddbe2d1d8b9aafc49bcf` → `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa`; edge `ct:0e5a8c160ac5ac320c70b30186fb47b4c9d0b310585571cece2924ef9a7106b2`; raw triples `t:42db5c08d5b65ef1ce4923f92efea2060fd177ba55000758168757aa177f1f1a`.
- Gene content —CAUSES→ expansion; IDs `e:2b97e70e8e77c802502d5e7474908108d8f23f31d6633878c9cf45a0cd82defa` → `e:0a5ea201bafa16958ea8569eefc37ee43784eb8da5f6a3e846b7faaed2fa7097`; edge `ct:f9bf9b1c526d3fa4ac081ff0f3555b16dd1bc7a0e8fbf0daa84790d1ac5f3844`; raw triples `t:b8091516e68876f2f3af232f510cb582212a5f652b4b9ab28a383a474df0512c`.

Это семантический маршрут между исследованиями; новая фактическая зависимость не установлена.

## Ошибки и hubs

Потенциальные ошибки: регулярное приведение числа, многозначность имён, грубые словари смыслов, шум исходного NER/RE. ACCEPT — воспроизводимое решение алгоритма, не gold precision. Непроверенные варианты остаются в REJECT с evidence; исходные 118 memberships сохраняются независимо от новой политики.

Крупнейшие концепты (полный аудит: `concept_audit.json`):

| Концепт | Сущности | Документы | Новые memberships |
|---|---:|---:|---:|
| Patients | 16 | 5 | 0 |
| Genes | 5 | 2 | 2 |
| Medical complications | 4 | 2 | 1 |
| Energy consumption | 3 | 2 | 0 |
| cancer | 3 | 2 | 3 |
| follow-up | 2 | 2 | 2 |
| Convergence | 11 | 1 | 0 |
| Annual papal medals | 5 | 1 | 1 |
| B cells | 5 | 1 | 2 |
| Copy number variation | 5 | 1 | 0 |

Роста связности недостаточно для вывода об улучшении качества KG. Широкие Patients/Genes сохраняют только категориальную общность; по ним нельзя наследовать состав когорт, измерения или результаты. Новый общий Study/Results не создаётся.

## Сохранность и повторный запуск

Каждый исходный ID, label/type, endpoint и property проверен точно; только namespace/key/fingerprint заменены для отдельной копии. Исходный fingerprint/key сохранены дополнительно. Полный SHA-256 baseline сверяется с verification Stage 2F.3. Manifest фиксирует исходные файлы, корпус, код, параметры и hashes. Снимки инвентаря до/после включают per-record hashes. Importer, atomic transaction, fingerprint guard, exact read-back и no-op повторного запуска взяты из Stage 2F.3. Глобальные constraints не создаются и не меняются.

Воспроизведение: `.venv/Scripts/python.exe scripts/build_dynamic_concept_graph.py --neo4j-env <existing-file>`; подготовка без записи: добавить `--prepare-only`. Offline: `--prepare-only --baseline-cache <verified-snapshot.json>`.

## Фактически выполненные проверки

- Первый импорт: 875 узлов /2740 отношений.
- Повторный импорт: 0 узлов /0 отношений, полное совпадение target hashes.
- Baseline до/после: `5daa32d6b8e389592db3e3d0a5326f04cf06706b198448950ee7d9250ca431cf`.
- Target после двух проходов: `7cc8e0eb06fcae421df0d3935faecd453454c31e9071f4bf9989b658aa34a64d`.
- Внешние namespace, constraints/indexes и все 196 исходных файлов неизменны.
- Duplicate concepts/memberships и cross-namespace relationships: 0.
- Независимый Cypher: 223 WCC, 845 KG-вершин, крупнейшая компонента 371/17 документов; 8 новых concept-мостов.
- Все 10 graph_stats запросов выполнены; full-view возвращает все 875 узлов и 2740 отношений.
- Тесты: 57 passed, 0 failures/errors/skips. Семь существующих предупреждений torch.jit относятся к синтетическим тестам span layer; pretrained extractor не запускается.

Детали: `verification.json`, `independent_validation.json`, `test_results.json`.
Решения раздельно: [ACCEPT](accept.csv), [REJECT](reject.csv). Полное evidence каждого решения: `candidate_decisions.jsonl`.
