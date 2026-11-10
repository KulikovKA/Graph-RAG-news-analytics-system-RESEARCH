# Cross-document ReLEx experiment

```json
{
  "namespace": "relex_crossdoc_e5_small_t095",
  "processed_pairs": 194,
  "positive_pairs": 24,
  "unique_cross_doc_candidates": 40,
  "linked_documents": 12,
  "full_kg_component_bridges": 0,
  "original_factual_component_bridges": 40,
  "suspicious_type_hints": 0,
  "without_direct_evidence": 40,
  "relation_types": {
    "HAS_RESULT": 32,
    "DERIVED_FROM": 2,
    "PRODUCES": 1,
    "USED_FOR": 1,
    "CAUSES": 2,
    "USES": 2
  },
  "connectivity": {
    "entities": 1122,
    "isolated": 200,
    "weak_components": 258,
    "largest_component_vertices": 580,
    "largest_component_documents": 23
  },
  "import_created": {
    "nodes": 1152,
    "relationships": 3607
  },
  "repeated_import_created": {
    "nodes": 0,
    "relationships": 0
  },
  "confidence": {
    "min": 0.363546222448349,
    "median": 0.47023679316043854,
    "mean": 0.5259911052882671,
    "max": 0.8410337567329407
  },
  "confidence_bins": {
    "0.35..0.5": 22,
    "0.5..0.7": 13,
    "0.7..0.9": 5,
    "0.9..1": 0
  }
}
```

Две параллельные копии исходного ReLEx-base, неизменённый fixed-span адаптер и 20 labels. Сохранены все предсказания строго >0.35 в обоих направлениях. Smoke 10 (включая patients) повторён с идентичным результатом. JSONL: одна metadata-строка и 194 pair_result со списками relations; пустой список — полноценный отрицательный результат.

Это экспериментальные кандидаты CROSS_DOC_RE_CANDIDATE, не подтверждённые факты. У всех отсутствует общий исходный passage: два source fragments сохранены отдельно, синтетический вход НЕ evidence. Типовые подсказки — диагностика, не фильтр.

Новые рёбра не соединяют разные компоненты полного Global KG: их концы уже связаны SIMILAR_TO. В исходном фактическом подграфе без SIMILAR_TO концы находятся в разных документных компонентах. Сохранены 30 документов, 1122 сущности, 1760 RELEX_REL, 685 SIMILAR_TO и 1122 служебных RELEX_MENTIONS.

## 20 примеров

| Head / document | Relation / confidence | Tail / document | Head source fragment | Tail source fragment |
|---|---|---|---|---|
| results / W4309030044 | HAS_RESULT / 0.841034 | results / W2078771716 | per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS | lined, convergence does not seem to be occurring. For inflation rates convergence can be rejected. Taken together these results suggest reasons for the recent disruption to the ERM and bring into question its sustainability. © 1997 John Wiley & So |
| results / W4309030044 | HAS_RESULT / 0.829875 | results / W2096307598 | per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun |
| results / W1979727947 | HAS_RESULT / 0.798265 | results / W2096307598 | y, significantly improved the fit of the logistic regression model of breast cancer risk (P = 0.0002). CONCLUSIONS: Our results demonstrate the potential to facilitate the integration of mammographic density measurements into large-scale research  |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun |
| results / W4309030044 | HAS_RESULT / 0.726049 | results / W1979727947 | per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS | y, significantly improved the fit of the logistic regression model of breast cancer risk (P = 0.0002). CONCLUSIONS: Our results demonstrate the potential to facilitate the integration of mammographic density measurements into large-scale research  |
| results / W2096307598 | HAS_RESULT / 0.723370 | results / W1979727947 |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun | y, significantly improved the fit of the logistic regression model of breast cancer risk (P = 0.0002). CONCLUSIONS: Our results demonstrate the potential to facilitate the integration of mammographic density measurements into large-scale research  |
| RESULTS / W2761455282 | HAS_RESULT / 0.696235 | results / W1979727947 | the Western Ontario and McMaster Universities Arthritis Index. Perioperative complications and mortality were recorded. RESULTS: One patient died of stroke 1 year after surgery (4.8% mortality). The mean follow-up period of other 20 patients was 5 | y, significantly improved the fit of the logistic regression model of breast cancer risk (P = 0.0002). CONCLUSIONS: Our results demonstrate the potential to facilitate the integration of mammographic density measurements into large-scale research  |
| results / W2078771716 | HAS_RESULT / 0.696174 | results / W4309030044 | lined, convergence does not seem to be occurring. For inflation rates convergence can be rejected. Taken together these results suggest reasons for the recent disruption to the ERM and bring into question its sustainability. © 1997 John Wiley & So | per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS |
| results / W1979727947 | HAS_RESULT / 0.684255 | results / W4234628002 | y, significantly improved the fit of the logistic regression model of breast cancer risk (P = 0.0002). CONCLUSIONS: Our results demonstrate the potential to facilitate the integration of mammographic density measurements into large-scale research  | rers. A special access for PNM's Center for Education Quality Assurance and Development (P4MP) to manage and define the results as decision support materials, as part of the SPMI activity cycle at the end of each semester. The information system i |
| RESULTS / W3101922761 | HAS_RESULT / 0.647350 | results / W1979727947 | ure, and visual acuity. Surgical procedures before and during the study period, and their complications were evaluated. RESULTS: The mean follow-up was 125.0±52.3 (median, 116.5) months. Twenty of 40 eyes had a follow-up of at least 10 years. The  | y, significantly improved the fit of the logistic regression model of breast cancer risk (P = 0.0002). CONCLUSIONS: Our results demonstrate the potential to facilitate the integration of mammographic density measurements into large-scale research  |
| results / W2078771716 | HAS_RESULT / 0.627261 | results / W2096307598 | lined, convergence does not seem to be occurring. For inflation rates convergence can be rejected. Taken together these results suggest reasons for the recent disruption to the ERM and bring into question its sustainability. © 1997 John Wiley & So |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun |
| RESULTS / W1979727947 | HAS_RESULT / 0.626210 | results / W2096307598 | sion models for breast cancer risk, which included our measure capturing additional information in mammographic images. RESULTS: We observed a high correlation between the Cumulus measure and our measure mimicking Cumulus (r = 0.884; 95% CI, 0.872 |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun |
| results / W2096307598 | HAS_RESULT / 0.588727 | results / W4309030044 |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun | per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS |
| results / W2078771716 | DERIVED_FROM / 0.588405 | results / W4309030044 | lined, convergence does not seem to be occurring. For inflation rates convergence can be rejected. Taken together these results suggest reasons for the recent disruption to the ERM and bring into question its sustainability. © 1997 John Wiley & So | per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS |
| results / W2096307598 | HAS_RESULT / 0.564764 | RESULTS / W1979727947 |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun | sion models for breast cancer risk, which included our measure capturing additional information in mammographic images. RESULTS: We observed a high correlation between the Cumulus measure and our measure mimicking Cumulus (r = 0.884; 95% CI, 0.872 |
| results / W1979727947 | HAS_RESULT / 0.555620 | RESULTS / W3101922761 | y, significantly improved the fit of the logistic regression model of breast cancer risk (P = 0.0002). CONCLUSIONS: Our results demonstrate the potential to facilitate the integration of mammographic density measurements into large-scale research  | ure, and visual acuity. Surgical procedures before and during the study period, and their complications were evaluated. RESULTS: The mean follow-up was 125.0±52.3 (median, 116.5) months. Twenty of 40 eyes had a follow-up of at least 10 years. The  |
| RESULTS / W2761455282 | HAS_RESULT / 0.553261 | results / W2096307598 | the Western Ontario and McMaster Universities Arthritis Index. Perioperative complications and mortality were recorded. RESULTS: One patient died of stroke 1 year after surgery (4.8% mortality). The mean follow-up period of other 20 patients was 5 |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun |
| cooperation / W4285525942 | CAUSES / 0.527328 | cooperation / W4309030044 | celerates the accumulation of global capital in green infrastructure, but also significantly facilitates the process of cooperation between private investors and developers with green banks, standardizes the conditions of such cooperation, and increas | tor and improve the connectivity between aviation and HSR, and a reasonable pricing strategy is more likely to motivate cooperation between aviation and HSR. |
| results / W2096307598 | HAS_RESULT / 0.506626 | RESULTS / W2761455282 |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun | the Western Ontario and McMaster Universities Arthritis Index. Perioperative complications and mortality were recorded. RESULTS: One patient died of stroke 1 year after surgery (4.8% mortality). The mean follow-up period of other 20 patients was 5 |
| RESULTS / W3101922761 | HAS_RESULT / 0.479979 | results / W2096307598 | ure, and visual acuity. Surgical procedures before and during the study period, and their complications were evaluated. RESULTS: The mean follow-up was 125.0±52.3 (median, 116.5) months. Twenty of 40 eyes had a follow-up of at least 10 years. The  |  the new data are used to study the properties of regolith, such as dielectric constant and thickness distribution. The results show that the lunar regolith's thickness increases from the NW (mare area) to the SE (highland area) regions on the lun |
| results / W4309030044 | HAS_RESULT / 0.471190 | RESULTS / W1979727947 | per and lower models are solved using the particle swarm algorithm and method of successive averages, respectively. The results suggest that airlines are willing to cooperate with the HSR sector and improve the connectivity between aviation and HS | sion models for breast cancer risk, which included our measure capturing additional information in mammographic images. RESULTS: We observed a high correlation between the Cumulus measure and our measure mimicking Cumulus (r = 0.884; 95% CI, 0.872 |
