# 2F.4A — Candidate discovery

```json
{
  "discovered_pairs": 682,
  "selected_for_inference": 36,
  "source_original_cross_similar_to": 121,
  "cross_similar_unique_pairs": 113,
  "source_counts": {
    "A": 113,
    "B": 84,
    "C": 554,
    "D": 1
  },
  "selected_source_counts": {
    "A": 18,
    "B": 6,
    "C": 17,
    "D": 1
  },
  "discovered_documents": 21,
  "selected_documents": 13,
  "generic_controls": 6,
  "generic_path_controls": 6,
  "all_enumerated_paths": 3236,
  "saved_representative_paths": 1129,
  "baseline": {
    "weak_components": 223,
    "largest_component_vertices": 369,
    "largest_component_documents": 17
  },
  "policy": {
    "version": "2f4-v1",
    "inference_C_limit": 160,
    "inference_D_limit": 40,
    "generic_control_limit": 6,
    "paths_per_pair_in_model": 1,
    "generic_path_control_limit": 6,
    "max_saved_paths_per_pair": 3,
    "D_max_per_document_pair": 2,
    "D_min_shared_context_terms": 3,
    "D_min_shared_name_terms": 1,
    "D_min_context_jaccard": 0.045,
    "min_threshold": 0.35,
    "threshold_comparison": "confidence >= threshold; stored adapter scores strictly > 0.35",
    "candidate_pairs_unordered": true,
    "inference_both_directions": true,
    "path_search": "one cross-document SIMILAR_TO or two HAS_SHARED_CONCEPT edges; one factual edge on each side",
    "screening": "frozen lexical/contextual ranking, no prediction score in selection"
  }
}
```

Методика зафиксирована до inference. Все 121 SIMILAR_TO учтены, после ER это 113 уникальных пар. Список содержит весь обнаруженный пул; selected_for_inference задаёт ограниченную исследовательскую выборку. Источники A–D перекрываются. Граф, включая общие слова и все старые рёбра, не меняется.

C: только одна междокументная семантическая перемычка либо один общий концепт и по одному factual ребру с каждой стороны. Более длинные цепочки не исследуются. Полный счёт путей сохранён, тексты/provenance вынесены в source_factual_evidence.jsonl и entity_profiles.jsonl; до трёх реальных путей на пару сохранены для проверки. Все bridge IDs сохранены в sources.C. Сам путь не доказывает транзитивности и не разрешает перенос фактов через концепт.

D: узкий поиск по пересечению слов названий и контекстов, дисциплине, типам и наличию factual окрестностей. Это ограничивает полноту: парафразы без общих слов не обнаруживаются; broad discipline не гарантирует научной совместимости.

В модель подаются два исходных фрагмента с документными границами, точные spans и один сохранённый путь с двумя source excerpts. Путь явно помечен как non-entailing; synthetic input не source evidence. Два направления оцениваются одним вызовом адаптера.

## Реальные примеры до запуска модели

Сильных в смысле доказанного научного вывода до аудита нет. В первой группе — лишь относительно перспективные кандидаты для проверки. Первичный вариант отбора выявил доминирование patients как промежуточного узла: основной C-отбор исправлен до полного inference; пути только через общие промежуточные термины ограничены шестью диагностическими парами.

| Категория | Сущности / документы | Причина и ограничение |
|---|---|---|
| Перспективный для проверки, не доказанный | study period (W3101922761) ↔ mortality (W2761455282) | Factual flank / semantic-or-concept bridge / factual flank; non-transitive path; слова: complications, follow-up, mean, months, period; это приоритет проверки, не поддержанное отношение |
| Перспективный для проверки, не доказанный | mortality (W2761455282) ↔ eyes (W3101922761) | Factual flank / semantic-or-concept bridge / factual flank; non-transitive path; слова: follow-up, years; это приоритет проверки, не поддержанное отношение |
| Перспективный для проверки, не доказанный | eyes (W3101922761) ↔ mortality (W2761455282) | Factual flank / semantic-or-concept bridge / factual flank; non-transitive path; слова: after; это приоритет проверки, не поддержанное отношение |
| Слабый / общий термин | follow-up (W3101922761) ↔ follow-up (W2761455282) | Existing cross-document SIMILAR_TO; no identity assumed; слова: follow-up, had; это приоритет проверки, не поддержанное отношение |
| Слабый / общий термин | cooperation (W4285525942) ↔ cooperation (W4309030044) | Existing cross-document SIMILAR_TO; no identity assumed; слова: cooperation; это приоритет проверки, не поддержанное отношение |
| Слабый / общий термин | study period (W3101922761) ↔ dialysis patients (W2761455282) | Factual flank / semantic-or-concept bridge / factual flank; non-transitive path; слова: assessed, during, outcome; это приоритет проверки, не поддержанное отношение |
| Сомнительный cross-component | HLA-DRB1*09 gene (W3123056951) ↔ gene dosage (W2076293586) | Different baseline WCC, shared name/context terms, compatible type, same discipline, factual neighbors; слова: association, gene, which; это приоритет проверки, не поддержанное отношение |

Полный архив всех 3236 обнаруженных путей: all_evidence_paths.jsonl. Файл evidence_paths.jsonl содержит представительные пути, выбранные до inference; его байты после запуска модели не менялись.
