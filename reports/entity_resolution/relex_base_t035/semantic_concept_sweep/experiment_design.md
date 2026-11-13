# Stage 2G.1 — Experiment design

Исходный граф — immutable Stage 2G, namespace `relex_sem_e5_small_name_t095_ner_t046_er2f3_concept_v1`, commit `f5fbc05`.
Его fingerprint и полный snapshot SHA-256 сверяются с manifest/verification; проверяются также все artifact hashes Stage 2G и неизменность исходного lexical resolver.
Корпус: те же 30 документов, 811 contextual entities, 34 concepts, 130 memberships, 1307 facts, 492 SIMILAR_TO, 811 document mentions.

## L и H

L — фактический Stage 2G без изменений. H — аддитивные виртуальные графы с semantic candidate discovery и отдельной semantic validation. Ни один старый объект/endpoint/property не изменяется. NER/RE/GLiREL/MERGE не вызываются; новые документы и модели не загружаются. Реальные graph import methods не вызываются.

Пороги — строки Decimal: 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.000. Они получены из целых чисел, не последовательным сложением float. Каждая из двух матриц cosine используется для всех 11 порогов. Scalar gates сравнивают точное binary значение score с Decimal; near-1 не округляется до 1 ради прохождения порога. Полностью одинаковые сохранённые векторы получают ровно 1 по алгебраическому тождеству. Эта коррекция cosine выполнена по уже сохранённым векторам; E5 повторно не запускалась.

## E5 и representations

Используется существующая local `intfloat/e5-small-v2`, с теми же файлами и hashes Stage 2B. CPU FP32, 4 threads, batch 8, tokenizer.json, max_length 512, query: с обеих сторон, attention-masked mean pooling, L2 normalization. После нормирования cosine вычисляется в float64, результат ограничивается [-1,1] от численной погрешности. Модель не обучается. Identical input strings кодируются один раз; embeddings для entities и reference variants сохраняются вне Git. Cache fingerprint охватывает exact texts, IDs, reference provenance, модель и рецепт; каждый .npy имеет hash.

Entity `text_name` — сохранённое имя representative; `text_name_context` — прежний `Entity: <name>\nContext: <original representative context>` (исходный радиус 120 символов, без новых NER spans). Concept text_name — только название. Concept text_name_context — название и до 8 неизменённых member context snippets, детерминированно сбалансированных по документам. Искусственных научных описаний нет.

Для entity→concept в reference исключается весь document проверяемой entity, включая другие entities и overlapping spans из этого документа. Если после исключения контекстов нет, используется только название с явным name_only_fallback. Ни собственный context, ни соседний mention того же исследования не могут самоподтвердить embedding. 34 concepts дают 77 reference variants; duplicate name texts переиспользуются.

## Candidate discovery и validation

A: все 811×34 entity→existing concept comparisons, исключая уже существующие 130 memberships. B: все unordered entity↔entity пары разных документов; существующие SIMILAR_TO не ограничивают universe. Cosine сам по себе никогда не создаёт membership. Нет Union-Find, MERGE или similarity transitive closure.

HybridConceptResolver расширяет существующий класс, не меняя L. Он сохраняет исходные source spans, типы и contexts; дополнительно проверяет clinical class readings, явно заданные текстом определения и научную роль procedural paraphrases. В частности surgery/surgical procedures трактуются как класс клинических вмешательств только при PROCESS/TASK/METHOD/CONCEPT roles и clinical context evidence. Этот source-bound, интерпретируемый paraphrase rule допускает ACCEPT без lexical_match. Limonene и его gene, genetics и genes, организации и их lending products, hip и knee operations не уравниваются.

Это exploratory rule-based semantic validation: дополнительные clinical witnesses и procedural paraphrase family сформулированы после изучения исходных текстов этих 30 документов. Embedding descriptions не дописывались. Правила не используют entity/document IDs и не выбирают решения по желаемым метрикам, но независимого held-out semantic gold нет. Нельзя приписывать все улучшения retrieval: отдельный validator-only control повторно оценивает прежние REJECT с прежними references.

Existing membership одного endpoint может предложить связь второго endpoint к тому же concept. Если anchors нет, допускается source-named новый concept, только после независимого обоснования обоих endpoints минимум из двух документов. Имя выбирается из source surface forms; существующий concept ID переиспользуется при совпадении identity. У каждого принятого delta сохраняются candidate/decision ID, source entity, concept, document, cosine, mode, threshold, reason, evidence и resolver version. factual/identity/fact inheritance всегда false. При сомнениях REJECT.

## Метрики и аудит

Отдельно считаются semantic candidates, delta memberships и proposals/created concepts, shared-document concepts, entity→concept←entity bridges, новые прямые document pairs, первая достижимость через полный KG, WCC и крупнейшая компонента. Document/Mentions исключены из WCC. Счёт мостов конечный: это entity pairs через concept, а не все циклические walks. Изменение связности не доказывает рост качества KG.

Аудируются все новые ACCEPT на уровне каждого уникального candidate–entity–concept решения; на этом малом наборе проверены и все REJECT. Quality audit — agent content review, не независимый Gold; подозрительные принятые связи и возможные false negatives разделяются. Requested probes включают breast/esophageal cancer, limonene/gene, независимые patient cohorts и clinical follow-up, даже если пара ниже gate, уже linked либо intra-document и не является допустимым новым sweep candidate.

Порог рекомендуется отдельно по режимам: по сохранению содержательно проверенных новых bridges и нагрузке на аудит, не по максимуму рёбер. Одинаковый cosine двух режимов не считается одинаково калиброванным.

## Воспроизведение

`.venv/Scripts/python.exe scripts/sweep_semantic_concepts.py --neo4j-env <existing-file>` — проверенный read-only live run с cache reuse. `--embeddings-only` готовит только cache/candidates; без env проверяется сохранённый полный source snapshot, и live preservation не заявляется. Cache находится в `.cache/semantic_concept_sweep/`, в Git его нет. Если cache snapshot отсутствует, live run читает его из уже существующего namespace. Ни один режим runner не импортирует граф.

Тесты: `.venv/Scripts/python.exe -m pytest tests/test_semantic_concepts.py tests/test_dynamic_concepts.py -q`. Локальный commit разрешён, push и graph import запрещены в этом эксперименте.
