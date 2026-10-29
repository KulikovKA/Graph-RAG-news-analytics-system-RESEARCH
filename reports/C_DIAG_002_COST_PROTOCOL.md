# C-DIAG-002: measurement supplement before evaluation

Зафиксировано до первого Gold scoring. Модели, revisions, mapping, thresholds,
правила, primary predictions и frozen protocol не меняются.

После пяти основных runs выполнить четыре отдельных fresh-process NER-only
measurement: control, chem, battery, union; те же 12 source-only DEV, четыре CPU
threads, seed 42, float32. Без RE и Gold. Измерить total wall/CPU и Windows process
peak RAM с imports/load, а также NER predict wall/CPU отдельно. Это уточняет
запрошенное сравнение памяти NER: peak основного pipeline включает и RE.

Predicted (chunk,start,end,kind,surface) inventories обязаны точно совпасть с уже
сохранённым соответствующим primary run. Иначе остановка и диагностика, без
исправления spans по Gold. Inference audit guard остаётся source-only. Результаты
записать в `reports/c_diag_002/ner_costs.json`; запускать последовательно, без
параллельного primary inference. Эти измерения не добавляют новых model variants.

Дополнительный ранний source-only battery NER workload probe, выполненный во время
battery RE, описан отдельно в reproduction; он не является NER speed benchmark.
Обнаруженный metadata lineage mismatch battery записан до evaluation в
`metadata_audit.json`; пакет и mapping используются без изменений.
