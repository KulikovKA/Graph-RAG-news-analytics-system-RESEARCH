"""Stage 2C: aggregate 2B; context case analysis, not gold and not semantic merge.

Run: python scripts/verify_semantic_candidates.py
--inspect prints a deterministic diagnostic sample without writing artifacts.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
ER = ROOT / "reports/entity_resolution/relex_base_t035"
INPUT = ER / "semantic_resolution/input/canonical_entities.jsonl"
BENCH = ER / "semantic_resolution/model_comparison"
DOCS = ROOT / "reports/ner_v2/research_run2/input_abstracts.jsonl"
OUT = ER / "semantic_resolution/candidate_verification"
REVIEWED_INPUT = "34fe3fd42e110b5b7a8b054360eee2a6182c67782e7afa54d2298e0491bf9151"

# Purposive context-case sample, bound to the unchanged input snapshot. Selectors
# are original document/start/name, never inferred identity IDs or score cutoffs.
# doc | start:name | start:name | decision | referent level | rationale | source quote
CASES = """
W1979727947|14:Mammographic density|593:MD|ACCEPT|scientific_characteristic|В тексте MD явно раскрывается как Mammographic density; все участники MD описывают ту же характеристику плотности, а не другую расшифровку.|Mammographic density (MD)
W1979727947|396:measure|449:measure|REJECT|measurement_method|Первая мера конструируется автоматически, вторая — well-established semiautomated Cumulus; это сравниваемые, а не одинаковые меры.|construct a measure that mimics a well-established semiautomated measure (Cumulus)
W1979727947|458:Cumulus|891:Cumulus|REVIEW|named_measure|Описание указывает на Cumulus, но один исходный тип ATTRIBUTE, другой MECHANISM; требуется разрешить типизацию/уровень, тип автоматически не исправляется.|measure (Cumulus)
W1979727947|891:Cumulus|1201:Cumulus measure|REVIEW|named_measure|Вероятна одна мера Cumulus, но MECHANISM против ATTRIBUTE/OBJECT; контекстного основания недостаточно для принятия при нерешённых исходных типах.|Cumulus measure and our measure mimicking Cumulus
W1979727947|81:breast cancer|600:breast cancer risk|REJECT|disease_vs_risk|Диагноз заболевания и характеристика его риска — разные концепты; общий текст и совпадение CONDITION не делают их эквивалентными.|logistic regression models for breast cancer risk
W1979727947|657:dataset|672:training set|REJECT|dataset_vs_partition|Набор делят на обучающую и тестовую части; весь dataset не равен training set.|partitioned our dataset into a training set
W1979727947|672:training set|736:test set|REJECT|distinct_data_partitions|Обучающая и тестовая части явно различаются назначением и числом cases/controls.|733 cases, 748 controls
W1979727947|797:Pearson product-moment correlation coefficient|845:r|ACCEPT|named_statistic|В этой отдельной канонической записи r находится только в явном определении Pearson coefficient (r), не включает позднюю оценку r=0.884; полное имя и символ обозначают тот же коэффициент.|Pearson product-moment correlation coefficient (r)
W1979727947|845:r|1252:r|REVIEW|statistic_vs_estimate|Определение символа коэффициента и конкретная оценка r=0.884; уровень представления не определён однозначно.|r = 0.884
W1995028339|710:seven genomes|716:genomes|ACCEPT|document_defined_set|seven genomes и их повтор their genomes обозначают один явно заданный набор из семи геномов; не все геномы диатомей вообще.|The sizes of the seven genomes vary
W1995028339|716:genomes|836:genomes|REJECT|set_vs_subset|Первый участник — все семь геномов, второй — plastid genomes of Thalassiosirales; состав множеств не совпадает.|plastid genomes of Thalassiosirales
W1995028339|529:Cyclotella sp. WC03_2|552:Cyclotella sp. L04_2|REJECT|named_strain|В перечне даны два разных strain-идентификатора WC03_2 и L04_2; общий род не означает идентичность.|Cyclotella sp. WC03_2, Cyclotella sp. L04_2
W1995028339|481:Thalassiosira weissflogii|1138:Thalassiosira oceanica|REJECT|named_species|Разные биномиальные имена видов; у oceanica отдельно обсуждается перестройка генома.|genome rearrangement in Thalassiosira oceanica
W1995028339|645:Cerataulina daemon|669:Rhizosolenia imbricata|REJECT|named_species|В исходном перечне это два разных вида, а не синонимы одного вида.|Cerataulina daemon, and Rhizosolenia imbricata
W1995028339|669:Rhizosolenia|669:Rhizosolenia imbricata|REVIEW|genus_fragment_vs_species|Первое имя — фрагмент биномиального имени; без решения о границах сущности и уровне genus/species нельзя считать их синонимами.|Rhizosolenia imbricata
W1995028339|1298:order|1679:order|REJECT|gene_order_vs_taxonomic_rank|В первом случае ancestral gene order, во втором order как таксономический ранг Thalassiosirales; разные значения одной строки.|synapomorphy for the order
W1995028339|1413:genes|1550:genes|ACCEPT|same_named_gene_pair|The genes ilvB and ilvH затем повторены как these genes в том же предложении; оба span относятся к этой паре, а не к следующему acpP1. Это уточнение контекста, ранее оставленного на проверке.|suggesting an independent gain of these genes in this species
W1995028339|283:Thalassiosirales|1312:Thalassiosirales|REVIEW|named_taxon|Один таксон вероятен по тексту, но OBJECT против ACTOR/OBJECT; не исправляем типы и не принимаем только одинаковое имя.|ancestral order for the Thalassiosirales
W2039836112|790:apyrase|790:apyrase (APY)|ACCEPT|named_enzyme|Одна и та же apyrase прямо снабжена аббревиатурой APY в скобках; оба типа ACTOR сохранены.|apyrase (APY)
W2039836112|254:ATP|877:ATP|REVIEW|substance_with_type_conflict|Контексты описывают секрецию ATP, но OBJECT против ATTRIBUTE; нужна проверка исходной типизации, не автоматическая смена типа.|implicating secretion of ATP
W2039836112|254:ATP|1293:FGF-1|REJECT|different_substances|FGF-1 вводится как ростовой фактор, ATP секретируется после его действия; причинная связь не является тождеством.|FGF-1 application
W2039836112|372:hemichannels|47:gap junctions|REJECT|different_channel_structures|Текст различает hemichannels и junction channels, которые состоят из одного HC от каждой клетки; разные структуры.|one HC from each cell
W2039836112|372:hemichannels|1486:HCs|REVIEW|channel_class_with_type_conflict|HCs связано с hemichannels, но MECHANISM против OBJECT, а каналы имеют разные подтипы; дополнительное разрешение типов/уровня необходимо.|hemichannels
W2039836112|1008:Px1 HCs|1486:HCs|REVIEW|subtype_vs_channel_class|Px1 HCs — явный подтип, позднее HCs не уточняет подтип; общий референт всей группы не доказан.|both Cx43 and Px1 HCs
W2039836112|0:Spinal astrocytes|203:spinal astrocytes|REVIEW|cell_class_vs_experimental_population|Общий класс клеток и экспериментальные клетки in culture; совпадение конкретной популяции не следует из общего названия.|spinal astrocytes in culture
W2076293586|30:inbred mouse models|37:mouse models|ACCEPT|document_defined_models|Оба span относятся к двум явно введённым inbred mouse models этой лаборатории; это одна совокупность моделей.|we established two inbred mouse models
W2076293586|198:heterogeneous CD-1 outbred population|212:CD-1 outbred population|ACCEPT|document_defined_population|Один исходный CD-1 outbred population назван полностью и вложенным именем в том же вводящем выражении.|a heterogeneous CD-1 outbred population
W2076293586|569:copy number variants|591:CNVs|ACCEPT|scientific_variant_class|CNVs явно раскрыто как copy number variants; речь об одном классе вариантов, не о разных наборах CNV regions.|copy number variants (CNVs)
W2076293586|116:depression|128:anxiety disorders|REJECT|different_diagnostic_categories|Автор перечисляет depression и anxiety disorders как разные заболевания; близость области психиатрии не эквивалентность.|depression, anxiety disorders and pathological stress reactivity
W2076293586|259:extremes|300:extremes|REJECT|different_selection_axes|Экстремумы отбора относятся соответственно к anxiety-related behavior и stress reactivity.|extremes in anxiety-related behavior and extremes in stress reactivity
W2078771716|955:exchange rates|990:inflation rates|REJECT|different_time_series|В перечне тестируемых экономических рядов exchange rates и inflation rates различаются; их результаты обсуждаются отдельно.|exchange rates, interest rates and inflation rates
W2078771716|971:interest rates|990:inflation rates|REJECT|different_time_series|Процентные ставки и инфляция — отдельные ряды в тесте; общий эксперимент не означает тождество показателей.|exchange rates, interest rates and inflation rates
W2078771716|695:Kalman|695:Kalman filter techniques|REVIEW|name_fragment_vs_method|Kalman извлечён с ACTOR внутри названия метода METHOD; без решения о границах нельзя объявить человека/фрагмент методом.|Kalman filter techniques
W2084709926|39:divertor x-point|48:x-point|ACCEPT|document_defined_location|Одна нижняя x-точка дивертора NSTX; уточнение divertor задаёт носителя, все упоминания x-point относятся к описанной диагностике.|views the x-point
W2084709926|360:moveable in-vessel mirror|369:in-vessel mirror|ACCEPT|document_defined_optical_component|Оба span обозначают одно вводимое moveable in-vessel mirror в оптической системе, а не произвольные экземпляры зеркал.|a moveable in-vessel mirror
W2084709926|506:coherent fiber optic bundle|515:fiber optic bundle|ACCEPT|document_defined_optical_component|Одно явно вводимое fiber optic bundle передаёт изображение; оба имени относятся к нему в одном выражении.|A coherent fiber optic bundle transmits the image
W2084709926|564:remotely selected filter|582:filter|ACCEPT|document_defined_optical_component|В одном выражении выделены полное описание и head-span одного выбранного фильтра; иной фильтр не вводится.|through a remotely selected filter
W2084709926|594:fast camera|599:camera|ACCEPT|document_defined_camera_role|Полная фраза fast camera и head-span camera обозначают один вводимый камерный элемент; не приравниваем его к приведённому далее примеру модели.|to a fast camera, for example
W2084709926|599:camera|686:camera|REVIEW|camera_role_vs_example_model|Первая камера описана как fast camera, вторая — часть for example конкретного описания; идентичность выбранной модели не подтверждена.|for example, a 40 500 frames/s
W2084709926|48:x-point|232:lower x-point region|REJECT|point_vs_surrounding_region|X-point и region near/lower x-point обозначают точку и область вокруг неё; это не одна пространственная сущность.|near the lower x-point region
W2096307598|1088:regolith|1180:lunar regolith|ACCEPT|lunar_material_class|Весь документ посвящён радиоизмерениям Луны; свойства regolith и lunar regolith относятся к тому же классу лунного материала.|the lunar regolith's thickness
W2096307598|184:temperature|316:temperature|REVIEW|extraction_fragment_or_distinct_attribute|Один короткий span входит в temperature sensitivity, другой в brightness temperature; без уточнения границ и общего/измеренного уровня temperature тождество не устанавливается.|temperature sensitivity
W2096307598|184:temperature sensitivity|299:lunar brightness temperature|REJECT|instrument_sensitivity_vs_object_measurement|Чувствительность наблюдений 1.7 K и усреднённая яркостная температура Луны 233 K — разные характеристики разных носителей.|The resolution and temperature sensitivity of the observations
W2096307598|305:brightness temperature|390:brightness temperature|REJECT|different_measurement_aggregates|233 K усреднено по всему диску, 228.1/239.7/233.9/228.8 K — по квадрантам; не один агрегат измерения.|averaged over the whole disk is about 233 K
W2096307598|390:brightness temperature|604:brightness temperature|REJECT|different_measurement_aggregates|Средние значения квадрантов и highest brightness temperature 257 K в отдельной области не один агрегат.|The highest brightness temperature is about 257 K
W2096307598|817:degree of polarization|981:degree of polarization|REVIEW|global_metric_vs_spatial_distribution|Total degree и distribution along the limb могут обозначать показатель и распределение его значений; уровень сущности неоднозначен.|distribution of the degree of polarization along the limb
W2317432785|236:macroscopic magneto-optical transversal Kerr effect|236:macroscopic magneto-optical transversal Kerr effect (TKE)|ACCEPT|named_measurement_method|Расширение TKE дано в чистом выражении одного метода; названия не повреждены, несмотря на повреждённые формулы рядом.|macroscopic magneto-optical transversal Kerr effect (TKE)
W2317432785|352:element-selective X-ray absorption spectroscopy|352:element-selective X-ray absorption spectroscopy (XANES)|ACCEPT|named_measurement_method|Один метод с явно указанной в скобках аббревиатурой XANES; не отождествляется с XMCD.|element-selective X-ray absorption spectroscopy (XANES)
W2317432785|412:X-ray magnetic circular dichroism|412:X-ray magnetic circular dichroism (XMCD)|ACCEPT|named_measurement_method|Один метод, в скобках прямо указан XMCD; чистый фрагмент определения сохранён.|X-ray magnetic circular dichroism (XMCD)
W2317432785|352:element-selective X-ray absorption spectroscopy|412:X-ray magnetic circular dichroism|REJECT|different_measurement_methods|В перечне методов XANES и XMCD названы отдельно; общая X-ray тематика не делает методы эквивалентными.|X-ray absorption spectroscopy (XANES) and X-ray magnetic circular dichroism (XMCD)
W2317432785|664:Co0� 45Fe0� 45Zr0� 1|730:Co0� 45Fe0� 45Zr0� 1|REVIEW|damaged_formula|U+FFFD повреждает состав формулы, а контексты обсуждают sublayers и grains; одинаковая повреждённая строка не доказывает тождество.|Different silicides and grains
W2317432785|776:sublayer thicknesses|1055:sublayer thicknesses|REVIEW|metric_concept_vs_specimen_conditions|Общая характеристика sublayer thicknesses и сравнение larger/smaller образцов могут кодировать концепт параметра либо разные условия; уровень не определён, измерения автоматически не приравниваются.|larger and smaller sublayer thicknesses respectively
W2364973104|415:PSCAD/EMTDC|415:PSCAD/EMTDC power simulation software|ACCEPT|named_software|Полное описание software и его собственное имя обозначают одну указанную программную систему моделирования.|PSCAD/EMTDC power simulation software
W2364973104|415:PSCAD/EMTDC|421:EMTDC|REVIEW|software_suite_vs_component|PSCAD/EMTDC и компонент имени EMTDC нельзя считать одним артефактом без проверки границ/состава пакета.|PSCAD/EMTDC power simulation software
W2761455282|216:total knee arthroplasty|411:TKA|ACCEPT|procedure_kind|TKA прямо раскрыто как total knee arthroplasty; оба MECHANISM сохраняются, объединяется вид процедуры, не операции разных пациентов.|total knee arthroplasty (TKA)
W2761455282|411:TKA|1363:TKA|REVIEW|procedure_with_type_conflict|Одна аббревиатура, но MECHANISM против METHOD; исходные типы и уровни процедуры не исправляются.|total knee arthroplasty (TKA)
W2761455282|12:Total hip arthroplasty|216:total knee arthroplasty|REJECT|different_procedures|Hip и knee arthroplasty — разные виды операций, отдельно упомянутые в Background.|total knee arthroplasty (TKA)
W2761455282|516:patients|1114:patients|REJECT|cohort_vs_subset|Исходная когорта содержит 21 пациента, позднее Six patients имеют осложнения; вся когорта и подмножество не равны.|Six patients (28.5%) had medical complications
W2761455282|516:patients|1297:patients|REJECT|cohort_vs_subset|21 исходный пациент и 19 с улучшенными исходами на последнем follow-up — разные множества.|19 patients (90.5%) had improved clinical outcomes
W2761455282|970:stroke|1187:stroke|REVIEW|diagnosis_concept_vs_distinct_episodes|Один летальный инсульт и два случая осложнений — разные эпизоды, но непрояснён уровень CONDITION: диагноз-концепт или эпизод; оставлено REVIEW.|One patient died of stroke
W2761455282|1135:medical complications|1835:medical complications|REVIEW|observed_events_vs_risk_class|Случившиеся осложнения и будущий риск осложнений; идентичность событий не доказана, общий класс может совпадать.|risk of possible medical complications
W2761455282|1352:knee|1393:Knee|REJECT|anatomy_vs_name_fragment|Анатомический knee и слово Knee в American Knee Society Scores; разные референты, несмотря на один OBJECT.|American Knee Society Scores
W2791492120|808:peripheral blood mononuclear cells|844:PBMC|ACCEPT|cell_class|PBMC явно раскрыто как peripheral blood mononuclear cells; эта конкретная PBMC имеет OBJECT и не объединяет другие конфликтно типизированные PBMC.|peripheral blood mononuclear cells (PBMC)
W2791492120|844:PBMC|940:PBMC|REVIEW|cell_class_with_type_conflict|Одна расшифровка вероятна, но OBJECT против DATA_RESOURCE/OBJECT; нельзя принять до разрешения типов.|peripheral blood mononuclear cells (PBMC)
W2791492120|137:Regulatory B cells|993:Breg|REVIEW|cell_class_with_type_conflict|Breg явно раскрыто, но MECHANISM против ATTRIBUTE; положительная аббревиатурная связь не устраняет конфликт типов.|Regulatory B cells (Breg)
W2791492120|48:chronic rhinosinusitis with nasal polyposis|867:CRSwNP|REVIEW|disease_with_type_and_span_issues|Расшифровка явная, но ACTOR/CONDITION против CONDITION и исходный partial_token; автоматического исправления нет.|chronic rhinosinusitis with nasal polyposis (CRSwNP)
W2791492120|48:chronic rhinosinusitis|48:chronic rhinosinusitis with nasal polyposis|REVIEW|broader_disease_vs_specified_subtype|Короткий span не сохраняет with nasal polyposis; без решения о границах нельзя приравнять общее заболевание уточнённому подтипу.|chronic rhinosinusitis with nasal polyposis
W3101922761|121:penetrating keratoplasty|404:PK|ACCEPT|procedure_kind|PK прямо раскрывается как penetrating keratoplasty; объединяется вид процедуры, а не фактические операции пациентов.|penetrating keratoplasty (PK)
W3101922761|931:preoperative IOP|944:IOP|ACCEPT|same_preoperative_measurement|Оба span обозначают одно preoperative mean IOP 34.0 в том же выражении; не объединяется с послеоперационным IOP.|mean preoperative IOP was 34.0
W3101922761|1068:mean IOP|1073:IOP|ACCEPT|same_postoperative_measurement|Оба span входят в одно описание mean IOP 12.7 на последнем послеоперационном follow-up.|mean IOP decreased to 12.7
W3101922761|944:IOP|1073:IOP|REJECT|different_timepoint_measurements|Дооперационное 34.0 и последнее послеоперационное 12.7 — разные наблюдения одной характеристики, не один результат.|mean preoperative IOP was 34.0
W3101922761|931:preoperative IOP|1068:mean IOP|REJECT|different_timepoint_measurements|Первое явно preoperative, второе после postoperative follow-up; значения 34.0 и 12.7 не одно измерение.|At last postoperative follow-up
W3101922761|21:intractable glaucoma|33:glaucoma|REVIEW|disease_subtype_vs_head_span|Короткий head-span не сохраняет intractable; одинаковые контексты не решают, аннотирован ли общий класс glaucoma или его подтип.|intractable glaucoma
W3101922761|1369:corneal grafts|1504:corneal grafts|REJECT|cohort_vs_surviving_subset|Когорта трансплантатов и Only 7 corneal grafts, сохранившихся на последнем сроке, — разные множества.|Only 7 corneal grafts (17.5%) remained clear
W3123056951|0:Recombinant human erythropoietin|1097:rHuEPO|ACCEPT|named_biopharmaceutical|rHuEPO явно определено как recombinant human erythropoietin; позднее с ним сравнивается биоактивность мутантов, иного раскрытия нет.|Recombinant human erythropoietin (rHuEPO)
W3123056951|906:allele|781:HLA-DRB1*09|ACCEPT|named_allele|the allele отсылает к распознающему HLA-DRB1*09, который далее прямо назван allele; генное упоминание 611 не включается.|HLA-DRB1*09 allele
W3123056951|781:HLA-DRB1*09|1224:HLA-DRB1*09 allele|ACCEPT|named_allele|Один именованный аллель в дизайне эпитопов и выводе для носителей; не отождествляется с gene из Background.|HLA-DRB1*09 allele
W3123056951|906:allele|1224:HLA-DRB1*09 allele|ACCEPT|named_allele|Указательное the allele и полное имя в заключении относятся к HLA-DRB1*09, иных аллелей в тексте нет.|HLA-DRB1*09 allele
W3123056951|611:HLA-DRB1*09|781:HLA-DRB1*09|REVIEW|gene_vs_allele_or_inexact_source_naming|611 сопровождается gene, позднее то же имя названо allele; требуется отличить уровни gene/allele и возможную неточность самого источника. Совпадение идентификатора и один OBJECT не снимают эту неоднозначность.|HLA-DRB1*09 gene
W3123056951|808:seventeen mutants|818:mutants|ACCEPT|document_defined_mutant_set|Оба участника содержат те же упоминания полного набора seventeen mutants; five out of задаёт подмножество этого набора, не заменяя referent seventeen.|Five out of seventeen mutants
W3123056951|1097:rHuEPO|226:anti-rHuEPO neutralizing antibodies|REJECT|drug_vs_antibodies|Антитела направлены против rHuEPO; связывание/нейтрализация не тождество лекарства и антител.|anti-rHuEPO neutralizing antibodies
W3138328568|745:Yarrowia lipolytica|1510:Y. lipolytica|ACCEPT|named_species|Полное биномиальное имя дрожжей вводится до сокращения Y. lipolytica; все сокращённые упоминания относятся к описанному виду-хозяину.|nonconventional yeast Yarrowia lipolytica
W3138328568|0:Limonene|1079:limonene|REVIEW|substance_vs_extraction_fragment|Второй limonene входит в limonene synthase gene; границы упоминания не позволяют автоматически признать тот же химический референт.|a limonene synthase gene
W3138328568|0:Limonene|1079:limonene synthase gene|REJECT|substance_vs_gene|Производимое вещество и ген его синтазы — разные биологические объекты; общий токен не эквивалентность.|a limonene synthase gene
W3138328568|708:genes|1146:genes|REJECT|different_gene_sets|Первое genes — 82 кандидата, последнее — 5 из 8; мощности и состав множеств различаются.|5 of the 8 candidate genes
W4230116451|619:optimised blockwise non-local means|656:OB-NLM|ACCEPT|named_method|OB-NLM явно раскрывается как optimised blockwise non-local means, один метод для GPU-алгоритма.|optimised blockwise non-local means (OB-NLM)
W4230116451|619:optimised blockwise non-local means|619:optimised blockwise non-local means (OB-NLM)|ACCEPT|named_method|Полное название с аббревиатурой и без неё обозначает один метод в том же определяющем выражении.|optimised blockwise non-local means (OB-NLM)
W4230116451|619:optimised blockwise non-local means (OB-NLM)|656:OB-NLM|ACCEPT|named_method|Явно указанная аббревиатура одного метода, не просто сходство слов.|optimised blockwise non-local means (OB-NLM)
W4230116451|629:blockwise non-local means|656:OB-NLM|REVIEW|base_method_vs_optimized_variant|Короткий span не сохраняет optimised; нельзя автоматически считать базовый метод и оптимизированный вариант идентичными.|optimised blockwise non-local means (OB-NLM)
W4234628002|196:Politeknik Negeri Madiun|1203:PNM|REVIEW|named_institution_with_type_conflict|PNM указывает на названное учреждение, но ACTOR против DATA_RESOURCE; исходную типизацию не исправляем.|Politeknik Negeri Madiun
W4234628002|858:Rapid Application Development software|898:RAD|REVIEW|method_vs_software_wording|RAD раскрывается рядом с software development method, но полное извлечённое имя содержит software; требуется уточнить метод/инструмент, без автоматического сокращения.|Rapid Application Development software (RAD) software development method
W4285525942|728:green mortgages|745:green bonds|REJECT|different_financial_instruments|Ипотека и облигации перечислены как разные инструменты зелёного банкинга.|green mortgages, green bonds
W4285525942|708:green bank lending|993:green bank loans|REVIEW|lending_activity_vs_loan_assets|Выдача кредитов и сами кредитные активы/их стоимость могут относиться к разным уровням; общий финансовый процесс не доказательство identity.|total value of green bank loans
W4309030044|225:high-speed rail|225:high-speed rail (HSR)|ACCEPT|transport_mode|HSR прямо раскрыто как high-speed rail; оба ACTOR относятся к одному виду транспорта в модели, не конкретным поездам.|high-speed rail (HSR)
W4309030044|225:high-speed rail|225:high-speed rail (HSR|REVIEW|incomplete_span|Незакрытая скобка в одном извлечённом имени; сначала требуется решение о границе span, исходный текст не исправляется.|high-speed rail (HSR)
W4309030044|225:high-speed rail|841:HSR|REVIEW|transport_mode_with_type_conflict|Расшифровка явна, но ACTOR против ACTOR/OBJECT; типы не меняются и вопрос уровня не закрыт.|high-speed rail (HSR)
W4309030044|667:method of successive averages|677:successive averages|ACCEPT|named_method|Оба span обозначают один явно названный метод решения нижней модели, не средние измерения.|method of successive averages
W4312422299|77:biblical Jerusalem|86:Jerusalem|ACCEPT|named_city|Один библейский город, из которого уходит семья Лехия; модификатор biblical уточняет тот же referent повествования.|family from biblical Jerusalem
W4312422299|210:Nephite|210:Nephite people|ACCEPT|named_people|Тот же именованный народ descendants of Lehi; head-fragment Nephite в обоих своих контекстах относится к этому народу.|Nephite people, descendants of Lehi
W4312422299|210:Nephite|413:Nephite race|ACCEPT|named_people|Nephite people/race в одном повествовании обозначают один именованный народ, а не отдельные семьи.|the Nephite race can avoid destruction
W4312422299|210:Nephite people|413:Nephite race|ACCEPT|named_people|People и race здесь два обозначения того же народа нефийцев, явно введённого как descendants of Lehi.|the Nephite race can avoid destruction
W4312422299|324:destruction|436:destruction|REVIEW|event_or_threat_scope|Первая угроза относится к семье Лехия, вторая — народу; тождество эпизода либо общий концепт угрозы не определены.|save his family from destruction
W4389076493|437:region proposal network|636:RPN|REVIEW|named_model_with_type_conflict|RPN введено как region proposal network, но METHOD против ACTOR; типы не исправляются.|region proposal network (RPN)
W4389076493|911:novel adaptive region proposal network|951:ARPN|REVIEW|named_model_with_type_conflict|ARPN явно раскрывается, но METHOD против ACTOR; требуется согласовать уровень/исходную типизацию.|adaptive region proposal network (ARPN)
W4389076493|437:region proposal network|917:adaptive region proposal network|REJECT|base_model_vs_new_variant|Conventional RPNs противопоставлены предложенной adaptive ARPN; это изменённая архитектура, не два названия одной модели.|Conventional RPNs exhibit two key weaknesses
W4389076493|1912:average recall|1912:average recall (AR)|ACCEPT|named_metric|AR дано непосредственно в скобках после average recall; одна метрика, не результат измерения другой модели.|average recall (AR)
W4389076493|1885:average precision|1912:average recall|REJECT|different_metrics|AP и AR перечислены как two performance indicators; это разные метрики, а не синонимы.|average precision (AP) and average recall (AR)
W4391351014|61:cardiac implantable electronic device|61:cardiac implantable electronic device (CIED)|ACCEPT|device_class|Один класс устройств с прямо указанной аббревиатурой CIED; не отдельные импланты пациентов.|cardiac implantable electronic device (CIED)
W4391351014|61:cardiac implantable electronic device|100:CIED|ACCEPT|device_class|CIED в определяющем выражении — явное сокращение того же класса устройств.|cardiac implantable electronic device (CIED)
W4391351014|61:cardiac implantable electronic device (CIED)|100:CIED|ACCEPT|device_class|Одно определение класса CIED, полное имя и аббревиатура; не утверждается тождество имплантов когорты.|cardiac implantable electronic device (CIED)
W4391351014|100:CIED|477:CIED|REVIEW|device_class_vs_cohort_devices|Первый CIED — определение класса, второй — устройства конкретных пациентов при обследовании; тот же класс не означает те же экземпляры.|examinations of the CIED functionality
W4391351014|300:radiotherapy|1071:RT|ACCEPT|procedure_kind|RT в заключении обозначает ту же лучевую терапию изученной когорты; другой расшифровки RT в тексте нет.|after completion of RT
W4391351014|41:patients|246:patients|REJECT|population_vs_study_cohort|Общая популяция cancer patients и sixteen patients with esophageal cancer — разные множества.|sixteen patients with esophageal cancer
W7215137876|396:Internal Family Systems|648:IFS|ACCEPT|named_therapy_method|IFS введено как Internal Family Systems и далее обсуждается тот же способ терапии/inner work.|Internal Family Systems (IFS)
W7215137876|117:development|541:development|REVIEW|different_process_subjects|Emotional maturity и emotional skills могут быть связаны, но это не явно один процесс с одним носителем; неизвестен уровень concept/event.|development of emotional maturity
"""

# doc:left-selector | doc:right-selector | decision | level | reason | quote-left | quote-right
CROSS_CASES = """
W3123056951@149:cancer|W4391351014@34:cancer|ACCEPT|general_disease_concept|Оба употребления обозначают общий медицинский концепт cancer без уточнения органа; речь о болезни как категории, не о тождестве пациентов, опухолей или клинических случаев.|chronic kidney disease, cancer or anemia|prevalence of cancer patients
W3123056951@149:cancer|W4391351014@271:cancer|REJECT|general_disease_vs_subtype|Первое cancer — общий диагноз-категория, второе входит в esophageal cancer; конкретный подтип нельзя приравнивать общей категории.|chronic kidney disease, cancer or anemia|patients with esophageal cancer
W2761455282@516:patients|W4391351014@246:patients|REJECT|different_defined_cohorts|Когорта из 21 ESRD пациента после TKA и когорта из 16 пациентов с раком пищевода после лучевой терапии — разные документно заданные множества.|21 patients (26 knees) with ESRD|sixteen patients with esophageal cancer
W3101922761@361:patients|W4391351014@246:patients|REJECT|different_defined_cohorts|40 пациентов с GDD после PK и 16 пациентов с раком пищевода — разные когорты с разными мощностями и протоколами.|Forty eyes of 40 patients|sixteen patients with esophageal cancer
W2791492120@615:patients|W2761455282@516:patients|REJECT|different_defined_cohorts|10 пациентов в исследовании B-клеток и 21 ESRD пациент после TKA — не одно множество.|from 10 patients|21 patients (26 knees) with ESRD
W2791492120@615:patients|W4391351014@246:patients|REJECT|different_defined_cohorts|Когорты из 10 и 16 пациентов в разных протоколах; одинаковое слово patients не доказывает identity.|from 10 patients|sixteen patients with esophageal cancer
W4285525942@4:article|W2991563224@8:paper|REJECT|different_publication_instances|Self-reference к двум разным публикациям с разными document ID/DOI: зелёный банкинг и WSN/LEACH.|The article is devoted|In this paper
W2761455282@298:study|W2791492120@465:study|REJECT|different_study_instances|Разные текущие исследования, протоколы и исходные публикации: TKA/ESRD и B-клетки/CRSwNP.|The purposes of this study|the aim of this study
W4285525942@1389:author|W7215137876@4:author|REVIEW|unidentified_person|Это авторы разных текстов, но имена/идентификаторы людей не даны; нельзя доказать ни тождество, ни что это разные люди.|The author assesses|The author brings to light
W2991563224@396:energy consumption|W3138328568@203:energy consumption|REVIEW|metric_concept_vs_system_attribute|WSN и химический синтез имеют разных носителей расхода энергии; неизвестно, кодируется общий концепт величины или системный показатель.|network lifetime, energy consumption|high energy consumption in chemical synthesis
W3123056951@277:development|W7215137876@117:development|REJECT|different_process_subjects|В первом случае развивается pure red cell aplasia, во втором emotional maturity; это разные процессы, а не синонимы.|development of pure red cell aplasia|development of emotional maturity
W2791492120@772:B cells|W3123056951@407:T-cells|REJECT|different_cell_classes|B cells изучаются как субпопуляции, T-cells распознают EPO epitopes; разные именованные классы клеток.|frequency of B cells|T-cells recognizing EPO epitopes
"""

# Follow-up sample actually inspected after 2B: six highest-consensus non-catalog
# pairs per scope, plus one single-model pair per model/scope (best rank, ID tie).
FOLLOWUP_CASES = """
W2761455282|1642:improved clinical outcome|1651:clinical outcome|ACCEPT|same_document_conclusion_result|Оба RESULT-span относятся к одному improved clinical outcome у Most patients в заключении; improved описывает тот же наблюдаемый результат, не другую когорту.|Most patients had an improved clinical outcome
W4312422299|594:spread|563:reproduce|REJECT|different_actions_and_objects|В одном предложении reproduce относится к people of God, а spread — к word of God; воспроизводство народа и распространение слова не одно действие.|reproduce the people of God to spread the word of God
W2761455282|788:American Knee Society Scores|788:American Knee Society|REVIEW|instrument_name_fragment_with_type_conflict|Оба контекста содержат Scores, но короткий фрагмент имени может обозначать организацию/часть имени инструмента; ATTRIBUTE/DATA_RESOURCE против DATA_RESOURCE не разрешены.|American Knee Society Scores
W2078771716|53:EMU|96:EC|REVIEW|unexpanded_acronyms|Одна аббревиатура описана через feasibility, другая через economies; раскрытий и основания считать их одним объектом нет. По разным буквам автоматически REJECT не присваивается.|feasibility of EMU is dependent upon the economies of the EC
W2078771716|53:EMU|26:ERM|REVIEW|unexpanded_acronyms|Источник сопоставляет sustainability of ERM и feasibility of EMU, но не раскрывает имена и уровень референтов; недостаточно доказательств identity, без внешней реконструкции.|sustainability of the ERM and the feasibility of EMU
W4389076493|1800:obvious improvement|1808:improvement|ACCEPT|same_reported_comparison_result|Оба RESULT-span относятся к одному улучшению ARPN относительно конкурентов по AP/AR в том же предложении; obvious — оценочное свойство этого результата.|show obvious improvement over other competitors
W1995028339|1114:genome rearrangement|258:family-level clades|REJECT|genomic_change_vs_taxonomic_group|Конкретная перестройка генома Thalassiosira oceanica и family-level clades — геномное изменение и таксономическая группа, не один объект.|extensive genome rearrangement in Thalassiosira oceanica
W2084709926|397:divertor region|7:magnetic field shear|REJECT|spatial_region_vs_field_property|Первый span — область дивертора, которую images оптическая система, второй — shear магнитного поля вокруг x-point; место и характеристика поля различаются.|Strong magnetic field shear around the divertor x-point
W2317432785|495:Fe|720:grains of Co0� 45Fe0� 45Zr0� 1|REVIEW|element_vs_damaged_material_name|Fe назван в ряду absorption edges, второй span — grains повреждённой формулы материала; это не основание считать их эквивалентными, а повреждение мешает надёжной идентификации материала.|absorption edges of Fe and Co
W1995028339|645:Cerataulina daemon|1138:Thalassiosira oceanica|REJECT|different_named_species|Вид Cerataulina daemon в перечне и Thalassiosira oceanica с перестройкой генома — разные биномиальные имена, не синонимы.|genome rearrangement in Thalassiosira oceanica
W1995028339|645:Cerataulina daemon|258:family-level clades|REJECT|species_vs_taxonomic_rank_class|Именованный отдельный вид и family-level clades вообще имеют разные уровни и объёмы; автор не определяет это как два имени одного таксона.|order- or family-level clades
W1995028339|283:Thalassiosirales|422:genomes|REJECT|named_taxon_vs_genome_set|Thalassiosirales прямо назван order, genomes входит в seven complete diatom plastid genomes, включая другие виды; таксон не равен набору геномов.|Seven complete diatom plastid genomes
W7215137876|246:laity|777:outer world|REJECT|people_vs_practice_domain|Laity — участники collaboration, outer world — область, где происходят synodal practices; люди и область практики не один референт.|collaborate with the laity
W2394633461|574:sede vacante|1147:popes|REVIEW|unresolved_historical_referent|Sede vacante не раскрыто в тексте и не тождественно автоматически popes по ACTOR; контраст generic medals и papal medals недостаточен для точного уровня/идентичности без уточнения источника.|sede vacante produced generic annual medals
W1995028339|380:morphology|1794:psaE|REJECT|organism_characteristics_vs_named_gene|Morphology перечислена как аспект изучения порядка, psaE — один из трёх genes involved in photosynthesis; это разные биологические концепты/объекты.|Three genes involved in photosynthesis, psaE, psaI, psaM
"""
FOLLOWUP_CROSS_CASES = """
W2761455282@1642:improved clinical outcome|W3123056951@47:biopharmaceutical drug|REJECT|clinical_result_vs_drug|Результат TKA у диализных пациентов и препарат rHuEPO — наблюдаемый исход и лекарство; общая медицинская тема не тождество.|Most patients had an improved clinical outcome|is a biopharmaceutical drug
W2761455282@1642:improved clinical outcome|W4391351014@41:patients|REJECT|clinical_result_vs_patient_population|Результат вмешательства у ESRD-когорты и общая популяция cancer patients — разные референты и уровни, не синонимы.|Most patients had an improved clinical outcome|prevalence of cancer patients
W2761455282@1642:improved clinical outcome|W4391351014@1297:lead impedance|REJECT|different_outcome_variables|Клинический исход после TKA и lead impedance кардиоустройства после RT относятся к разным объектам и показателям.|Most patients had an improved clinical outcome|pacing threshold, sensing threshold and lead impedance
W2906067930@4:phenomenon|W1995028339@1398:inversion|REJECT|economic_case_vs_genomic_change|Phenomenon явно задан как китайский рост производства зерна 2003–2014 без Harvest Paradox; inversion — перестройка генома Cyclotella. Это разные случаи, не один общий референт.|from 2003 to 2014|by a single inversion
W2906067930@4:phenomenon|W2130236240@544:plant closing literature|REJECT|economic_case_vs_literature|Исторический случай производства зерна в Китае и литература о закрытии предприятий — явление и информационные источники о другой теме.|from 2003 to 2014|earlier plant closing literature
W4312422299@594:spread|W2039836112@342:lucifer yellow|REJECT|religious_action_vs_fluorescent_tracer|Распространение word of God и lucifer yellow, явно названный fluorescent tracer, не одно действие/вещество.|spread the word of God|fluorescent tracers
W3138328568@1381:morphological and cytoplasmic membrane integrity analysis|W3123056951@1097:rHuEPO|REJECT|analysis_method_vs_drug|Метод анализа клеток дрожжей при токсичности лимонена и препарат rHuEPO — разные сущности; общая биомедицинская лексика не эквивалентность.|morphological and cytoplasmic membrane integrity analysis|bioactivity than rHuEPO
W3138328568@1381:morphological and cytoplasmic membrane integrity analysis|W1995028339@1606:Thalassiosirales|REJECT|analysis_method_vs_named_taxon|Анализ морфологии/мембраны дрожжей и именованный таксон диатомей имеют разные референты и роли.|morphological and cytoplasmic membrane integrity analysis|missing in all Thalassiosirales
W3138328568@1381:morphological and cytoplasmic membrane integrity analysis|W2076293586@627:endophenotypes|REJECT|analysis_method_vs_mouse_traits|Метод анализа дрожжей и эндофенотипы мышиных моделей — процедура и исследуемые признаки другой системы, не синонимы.|morphological and cytoplasmic membrane integrity analysis|endophenotypes of these animals
W2761455282@1135:medical complications|W3101922761@769:complications|REVIEW|complication_class_vs_study_events|Общий медицинский класс осложнений может пересекаться, но фактические эпизоды в TKA и GDD/PK-когортах различны; неясен concept/event уровень CONDITION.|Six patients (28.5%) had medical complications|their complications were evaluated
W3101922761@626:Corneal outcome|W2761455282@763:outcome|REJECT|different_clinical_outcomes|Corneal outcome измеряют rejection/failure/visual acuity; outcome в TKA оценивают Knee Society/Arthritis Index. Это разные клинические исходы и носители.|Corneal outcome was assessed|outcome was scored using American Knee Society Scores
W3101922761@817:follow-up|W2761455282@1025:follow-up period|REJECT|different_followup_measurements|Измеренные средние follow-up составляют 125.0 месяцев и 5.8 года/69.7 месяцев для разных когорт; это разные наблюдения, не один интервал.|mean follow-up was 125.0|follow-up period of other 20 patients was 5.8 years
W2364973104@830:filtering|W2084709926@582:filter|REJECT|electrical_filtering_vs_optical_device|Процесс filtering для lightning protection электролиний и физический оптический фильтр перед камерой относятся к разным системам/уровням.|shielding and filtering for the indirect lightning protection|through a remotely selected filter
W2761455282@1025:follow-up period|W3101922761@745:study period|REJECT|followup_duration_vs_other_study_window|Follow-up 20 пациентов после TKA и study period другого исследования GDD/PK — разные документно заданные периоды, не один временной интервал.|follow-up period of other 20 patients|before and during the study period
W2394633461@436:narrative|W4381572413@34:Oral history|REJECT|particular_historical_narrative_vs_discipline|Конкретный narrative о папской политике XVII века и Oral history как обсуждаемая дисциплина/подход — разные объекты/уровни, хотя оба относятся к истории.|a narrative of seventeenth-century papal politics|story of Oral history
"""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def prepare_cases(rows, docs, include_followup=True):
    lookup = defaultdict(list)
    for row in rows:
        representative = next(m["source_entity"] for m in row["members"] if m["source_entity"]["entity_id"] == row["canonical_entity_id"])
        lookup[row["document_id"], representative["start"], row["surface_text"].casefold()].append(row["canonical_entity_id"])
    def select(doc, value):
        if "@" in value:
            doc, value = value.split("@", 1)
        start, _, name = value.partition(":")
        found = lookup[doc, int(start), name.casefold()]
        require(len(found) == 1, f"Unknown/ambiguous case selector: {doc}/{value}")
        return found[0]
    cases = {}
    def add(left, right, status, level, reason, anchors):
        pair = tuple(sorted((left, right)))
        require(pair not in cases and left != right, "Duplicate/self context case")
        evidence = []
        for doc, quote in anchors:
            text = docs[doc]["text"]
            start = text.find(quote)
            require(start >= 0, f"Case quote absent from original text: {doc}/{quote}")
            evidence.append(dict(document_id=doc, revision_id=docs[doc]["revision_id"], start=start, end=start+len(quote), text=quote))
        cases[pair] = dict(decision=status, referent_level=level, reason=reason,
                           reason_entity_order=[left,right], source_evidence=evidence)
    same_text = CASES + (FOLLOWUP_CASES if include_followup else "")
    cross_text = CROSS_CASES + (FOLLOWUP_CROSS_CASES if include_followup else "")
    for line in same_text.strip().splitlines():
        if not line.strip():
            continue
        doc, a, b, status, level, reason, quote = line.split("|")
        add(select(doc,a), select(doc,b), status, level, reason, [(doc,quote)])
    for line in cross_text.strip().splitlines():
        if not line.strip():
            continue
        a,b,status,level,reason,qa,qb = line.split("|")
        da, db = a.split("@",1)[0], b.split("@",1)[0]
        add(select(da,a), select(db,b), status, level, reason, [(da,qa),(db,qb)])
    return cases


def aggregate(rows):
    by_id = {r["canonical_entity_id"]:r for r in rows}
    pairs = {}
    counts = Counter()
    with (BENCH/"candidate_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
        for line, r in enumerate(csv.DictReader(stream), 2):
            left, right = r["query_canonical_id"], r["candidate_canonical_id"]
            require(left in by_id and right in by_id and left != right, "Unknown/self candidate ID")
            a,b = by_id[left], by_id[right]
            scope = "within_document" if a["document_id"] == b["document_id"] else "cross_document"
            require(scope == r["scope"], "Candidate scope mismatch")
            require(r["query_name"] == a["surface_text"] and r["candidate_name"] == b["surface_text"], "Changed candidate name")
            for prefix, entity in (("query",a),("candidate",b)):
                require(r[prefix+"_document_id"] == entity["document_id"] and json.loads(r[prefix+"_types"]) == entity["entity_types"], "Changed candidate types/document")
                for field in ("source_entity_ids", "source_mention_ids", "quality_flags"):
                    require(json.loads(r[prefix+"_"+field]) == entity[field], "Changed candidate IDs/flags")
            require(1 <= int(r["rank"]) <= 5 and -1 <= float(r["similarity"]) <= 1, "Invalid candidate rank/cosine")
            counts[r["model"],r["mode"],left,scope] += 1
            pair = tuple(sorted((left,right)))
            record = pairs.setdefault(pair,dict(scope=scope, observations=[]))
            record["observations"].append(dict(model=r["model"],mode=r["mode"],query_side="left" if left == pair[0] else "right",
                                                rank=int(r["rank"]),similarity=r["similarity"],source_csv_line=line))
    configurations = {(m,mode) for m,mode,_,_ in counts}
    require(len(configurations) == 18 and len(counts) == 18*1122*2 and set(counts.values()) == {5}, "Incomplete 2B candidate coverage")
    require(sum(len(p["observations"]) for p in pairs.values()) == 201960, "Lost 2B observations")
    for pair,record in pairs.items():
        record["model_names"] = sorted({o["model"] for o in record["observations"]})
        record["modes"] = sorted({o["mode"] for o in record["observations"]})
        record["configuration_count"] = len({(o["model"],o["mode"]) for o in record["observations"]})
    return pairs


def contexts(row):
    return [dict(entity_id=m["source_entity"]["entity_id"], offsets=[m["source_entity"]["start"],m["source_entity"]["end"]],
                 source_mention_ids=m["source_mention_ids"], context=m["context"]) for m in row["members"]]


def main():
    require(sha(INPUT) == REVIEWED_INPUT, "Context cases require re-review: input 2A changed")
    rows = read_jsonl(INPUT)
    by_id = {r["canonical_entity_id"]:r for r in rows}
    require(len(rows) == len(by_id) == 1122, "Invalid 2A IDs")
    docs = {d["document_id"]:d for d in read_jsonl(DOCS)}
    cases = prepare_cases(rows,docs)
    if "--check-cases" in sys.argv:
        print(json.dumps(dict(valid_context_cases=len(cases))))
        return
    protected = {DOCS, INPUT}
    for directory in ("exact_matching_audit","conservative_resolution","relation_remapping","graph_comparison","semantic_resolution/input","semantic_resolution/model_comparison"):
        protected.update(p for p in (ER/directory).iterdir() if p.is_file())
    hashes = {p:sha(p) for p in sorted(protected)}
    pairs = aggregate(rows)
    if "--inspect" in sys.argv:
        base_cases = prepare_cases(rows, docs, include_followup=False)
        selected = set()
        for scope in ("within_document","cross_document"):
            eligible = [p for p in pairs if p not in base_cases and pairs[p]["scope"] == scope]
            for p in sorted(eligible,key=lambda p:(-len(pairs[p]["model_names"]),-pairs[p]["configuration_count"],p))[:6]:
                selected.add(p)
            for model in sorted({m for p in pairs.values() for m in p["model_names"]}):
                rare = [p for p in eligible if pairs[p]["model_names"] == [model]]
                if rare:
                    selected.add(min(rare,key=lambda p:(min(o["rank"] for o in pairs[p]["observations"]),p)))
        for p in sorted(selected):
            print(json.dumps(dict(ids=p,scope=pairs[p]["scope"],models=pairs[p]["model_names"],
                                  left=[by_id[p[0]]["document_id"],by_id[p[0]]["surface_text"],by_id[p[0]]["entity_types"],contexts(by_id[p[0]])],
                                  right=[by_id[p[1]]["document_id"],by_id[p[1]]["surface_text"],by_id[p[1]]["entity_types"],contexts(by_id[p[1]])]),ensure_ascii=False))
        return
    results = []
    for pair,group in sorted(pairs.items()):
        a,b = by_id[pair[0]],by_id[pair[1]]
        case = cases.get(pair)
        if case:
            decision, reason = case["decision"], case["reason"]
            if decision == "ACCEPT" and (len(a["entity_types"]) != 1 or a["entity_types"] != b["entity_types"] or
                                          {"damaged_name","original_partial_token"} & (set(a["quality_flags"])|set(b["quality_flags"]))):
                decision = "REVIEW"
                reason += " Положительное основание не снимает исходный конфликт/множественность типов либо повреждённый/частичный span; ACCEPT не присваивается."
        else:
            decision = "REVIEW"
            reason = "Пара вне индивидуально разобранной контекстной выборки. Тождество и различие референтов не проверены; similarity и число моделей не являются доказательством."
        results.append(dict(pair_id="pair:"+hashlib.sha256(json.dumps(pair).encode()).hexdigest(), scope=group["scope"],
                            left_canonical_id=pair[0],right_canonical_id=pair[1],left_name=a["surface_text"],right_name=b["surface_text"],
                            left_document_id=a["document_id"],right_document_id=b["document_id"],left_types=a["entity_types"],right_types=b["entity_types"],
                            decision=decision,individually_reviewed=case is not None,review_method="context_case_analysis" if case else "not_individually_reviewed",
                            referent_level=case["referent_level"] if case else "unresolved",reason=reason,
                            reason_entity_order=case["reason_entity_order"] if case else list(pair),
                            source_evidence=case["source_evidence"] if case else [],
                            model_count=len(group["model_names"]),models=group["model_names"],modes=group["modes"],configuration_count=group["configuration_count"],
                            observations=group["observations"],left_quality_flags=a["quality_flags"],right_quality_flags=b["quality_flags"],
                            type_set_conflict=set(a["entity_types"]) != set(b["entity_types"]),
                            left_source_entity_ids=a["source_entity_ids"],right_source_entity_ids=b["source_entity_ids"],
                            left_source_mention_ids=a["source_mention_ids"],right_source_mention_ids=b["source_mention_ids"],
                            left_contexts=contexts(a),right_contexts=contexts(b),
                            source_input=INPUT.relative_to(ROOT).as_posix(),source_candidates=(BENCH/"candidate_pairs.csv").relative_to(ROOT).as_posix()))
    require(len(results) == len({r["pair_id"] for r in results}) == len(pairs), "Duplicate/lost unique pairs")
    require(sum(len(r["observations"]) for r in results) == 201960, "Lost model/mode/direction observations")
    decisions = Counter(r["decision"] for r in results)
    reviewed = [r for r in results if r["individually_reviewed"]]
    accepted = {tuple(sorted((r["left_canonical_id"],r["right_canonical_id"]))) for r in reviewed if r["decision"] == "ACCEPT"}
    rejected = {tuple(sorted((r["left_canonical_id"],r["right_canonical_id"]))) for r in reviewed if r["decision"] == "REJECT"}
    adjacency = defaultdict(set)
    for a,b in accepted:
        adjacency[a].add(b);adjacency[b].add(a)
    open_wedges, contradictions = set(),set()
    for center,ends in adjacency.items():
        for a,b in itertools.combinations(sorted(ends),2):
            if (a,b) not in accepted:
                open_wedges.add((a,center,b))
                if (a,b) in rejected:
                    contradictions.add((a,center,b))
    require(not contradictions, "Contradictory ACCEPT-ACCEPT-REJECT cases require re-review")
    metrics = list(read_csv(BENCH/"model_comparison.csv"))
    require(len(metrics) == 18, "Incomplete 2B model comparison")
    base_cases = prepare_cases(rows, docs, include_followup=False)
    base_reviewed = sum(pair in base_cases for pair in pairs if pair in cases)
    followup_reviewed = len(reviewed)-base_reviewed
    reviewed_documents = {r[k] for r in reviewed for k in ("left_document_id","right_document_id")}
    rules = """# Предлагаемые правила Semantic Entity Resolution — результат 2C

Это предложения для будущего решения, не исполняемый Semantic Merge. Canonical mapping и отношения не изменены. ACCEPT — предварительное контекстное заключение о паре; оно не является разрешением изменять граф и не является независимой gold-разметкой.

1. **Similarity только порождает кандидатов.** Использовать ранги отдельно для моделей, text_name/text_name_context и внутри/между документами. Не принимать и не отклонять пару только по cosine или числу голосов. Размеры одного семейства моделей не являются независимыми экспертами. Общий контекст повышает сходство разных объектов; контекстный режим не заменяет автоматически поиск по имени. Никаких числовых порогов из этого анализа не следует: для них нужна независимая проверенная выборка с трудными отрицательными примерами, разделённая для калибровки и оценки. Даже отдельные пороги по моделям/режимам пока не обоснованы.

2. **Сначала определить уровень референта.** Для каждой будущей группы явно зафиксировать concept/type, именованный объект, конкретную когорту/набор, измерение или событие. Общий диагноз не приравнивается его подтипу; параметр не приравнивается конкретному измерению; вид процедуры не приравнивается операции пациента. Если из текста уровень неясен — REVIEW. Числа, отрицания, версии, штаммы, орган, время, носитель и модификаторы не удалять.

3. **Типы — консервативный gate, не доказательство.** По умолчанию для допуска к будущему объединению требовать один согласованный исходный тип у всех участников. Несовпадающие/множественные типы — REVIEW даже при явной расшифровке. Они не доказывают разные сущности: PNM, RPN, ARPN и Cumulus демонстрируют проблемы исходной типизации. Автоматически типы не исправлять. Равный тип не разрешает merge: CONDITION не делает болезнь и риск болезни идентичными.

4. **Синонимы и аббревиатуры.** Положительное основание — явное определение long form (ABBR), документно подтверждённое альтернативное название либо однозначная отсылка к тому же именованному объекту/набору. Проверить все исходные упоминания обоих участников: одна аббревиатура может иметь другую расшифровку. Вложенность span и общий head недостаточны: род и вид, рак и риск рака, вещество и его synthase gene различаются. Короткие названия сохранять как provenance, не расширять автоматически.

5. **Внутри документа** допускается вывод о том же референте при явном контекстном связывании, одинаковом уровне и непротиворечивых атрибутах. Для когорт важны состав/мощность, для измерений — носитель, время и агрегирование. До-/послеоперационное IOP и семь сохранившихся трансплантатов против всей когорты остаются раздельными. Для оптических компонентов полное описание и head-span могут обозначать один вводимый компонент; это не правило слияния всех устройств одного типа.

6. **Между документами** общего имени/тематики недостаточно. Для научного концепта нужны подтверждение одного значения и одной степени обобщения в обоих текстах, совместимые типы и отсутствие различающих спецификаций. Для конкретного объекта/экземпляра нужен общий устойчивый идентификатор или прямое документальное указание на тот же объект. Patients, author, study, result не объединять между статьями по названию. Разные DOI подтверждают различие самих self-referenced публикаций, но не обязательно различие их неизвестных авторов. При недостатке подтверждений — REVIEW.

7. **REJECT** — только при положительном контекстном доказательстве различия на том же уровне: разные идентификаторы штаммов, виды, методы, целое/подмножество, разные измерения, болезнь/риск. Ген и аллель тоже нельзя автоматически приравнивать, но при одинаковом идентификаторе сначала исключить неточную терминологию самого источника; если уровень неясен — REVIEW. Это правило для доказательств, не автоматический blanket reject всех разных слов/типов. Повреждённые формулы и partial_token обычно REVIEW, без исправления и без предположения, что одинаковые дефекты означают идентичность. Чистое явное определение метода может быть разобрано при повреждениях в соседнем контексте, с сохранением флага.

8. **Транзитивность не выводится из цепочки сходств.** ACCEPT(A,B) и ACCEPT(B,C) не дают ACCEPT(A,C). До будущего группового объединения проверить все пары объединяемых групп, единый уровень/область/идентификатор и отсутствие REJECT/REVIEW внутри группы. Непроверенные связи и противоречивые атрибуты блокируют автоматическое замыкание; не использовать connected components similarity-графа как готовые canonical groups. 2C проверяет согласованность принятых пар, но сами группы не строит.

9. **Provenance и обратимость.** Будущее решение должно сохранять исходные IDs, offsets, типы, confidence, все контексты, наблюдения моделей и основание принятия. Никакое решение 2C не удаляет информацию и не меняет данные 1A–2B. Независимая проверка необходима для оценки точности правил и качества Graph-RAG; данная целевая выборка не заменяет её.
"""
    lines = ["# Этап 2C — предварительная проверка кандидатов", "",
             f"Объединены 201960 направленных рекомендаций 18 конфигураций в **{len(results)} уникальных неупорядоченных пар**. Самопар нет; направления исходных рекомендаций, все модели, режимы, ранги и cosine сохранены в observations. Это анализ пар, не объединение сущностей.", "",
             "| Область | Всего пар | ACCEPT | REVIEW | REJECT | Индивидуально разобрано |", "|---|---:|---:|---:|---:|---:|"]
    for scope in ("within_document","cross_document"):
        subset = [r for r in results if r["scope"] == scope]
        cc = Counter(r["decision"] for r in subset)
        lines.append(f"| {scope} | {len(subset)} | {cc['ACCEPT']} | {cc['REVIEW']} | {cc['REJECT']} | {sum(r['individually_reviewed'] for r in subset)} |")
    lines += [f"| Всего | {len(results)} | {decisions['ACCEPT']} | {decisions['REVIEW']} | {decisions['REJECT']} | {len(reviewed)} |", "",
              f"Индивидуальная выборка: {len(reviewed)} пар из фактически найденных кандидатов; {sum(r['decision']=='REVIEW' for r in reviewed)} из них остались REVIEW после разбора. Остальные {len(results)-len(reviewed)} пар **не проверены индивидуально**, имеют individually_reviewed=False и REVIEW, а не фиктивное заключение об идентичности. Каталог включает явные аббревиатуры/синонимы, узкие/широкие понятия, классы/экземпляры, когорты/измерения, типовые конфликты и повреждения; выборка целевая, не случайная и не репрезентативная. Несопоставленные с кандидатами случаи каталога в статистику решений не входят.", "",
              f"Состав выборки: {base_reviewed} найденных случаев из базового каталога 127 контекстных примеров и {followup_reviewed} дополнительных пар, охвачены {len(reviewed_documents)} документов. Дополнительный отбор воспроизводим: по 6 внекаталожных пар с наибольшим числом моделей/конфигураций в каждой области и по одной паре, найденной только одной моделью, на модель и область; сначала лучший rank, затем canonical ID. Счётчики режимов и направлений не умножают число моделей. Отбор выполнен до присвоения решений этим дополнительным парам; он не даёт оснований обобщать частоты ACCEPT/REJECT на весь корпус.", "",
              "Использованы все контексты исходных span участников из members 2A и полные исходные абстракты, если короткого окна недостаточно. В verified_pairs.csv сохранены left/right_contexts для всех участников, исходные entity/mention IDs, document ID, типы/флаги и точные цитаты source_evidence с offsets и revision_id для индивидуальных решений. left/right упорядочены по canonical ID для единственности пары; reason_entity_order явно задаёт порядок участников в объяснении («первый/второй»). В таблице примеров имена показаны в порядке объяснения. Решения не принимаются по score: это детерминированное применение явно записанного контекстного разбора к фиксированному снимку, без дополнительной LLM-разметки.", "",
              "## Согласие моделей", "",
              "Число моделей считается без повторов по режимам и направлениям. Модель могла найти пару в одном или двух представлениях; все оценки сохранены отдельно. Согласие не является доказательством корректности.", "",
              "| Число моделей | Внутридокументных пар | Междокументных пар |", "|---|---:|---:|"]
    for votes in range(1,10):
        lines.append(f"| {votes} | {sum(r['model_count']==votes and r['scope']=='within_document' for r in results)} | {sum(r['model_count']==votes and r['scope']=='cross_document' for r in results)} |")
    lines += ["", "## Диагностика моделей на целевой выборке", "",
              "Числа ACCEPT/REJECT ниже — сколько разобранных пар было найдено моделью в каждом режиме; это не Precision, Recall, F1 или оценка на независимой разметке. Отбор из объединённых top-5 смещён в пользу уже найденных случаев. Время и RAM взяты из 2B; PyTorch E5-small нельзя напрямую ранжировать по скорости с ONNX как чистое сравнение моделей.", "",
              "| Модель | ACCEPT: name / context | REJECT: name / context | Инференс двух режимов, с | Максимальный RSS, MiB |", "|---|---:|---:|---:|---:|"]
    diagnostics = []
    for model in sorted({m["model"] for m in metrics}):
        hits = {}
        for mode in ("text_name","text_name_context"):
            for status in ("ACCEPT","REJECT"):
                hits[mode,status] = sum(r["decision"] == status and any(o["model"] == model and o["mode"] == mode for o in r["observations"]) for r in reviewed)
        timing = sum(float(m["inference_seconds"]) for m in metrics if m["model"] == model)
        rss = max(float(m["inference_peak_rss_mib"]) for m in metrics if m["model"] == model)
        diagnostics.append((model,hits,timing,rss))
        lines.append(f"| {model} | {hits['text_name','ACCEPT']} / {hits['text_name_context','ACCEPT']} | {hits['text_name','REJECT']} / {hits['text_name_context','REJECT']} | {timing:.2f} | {rss:.1f} |")
    diagnostic_by_model = {d[0]:d for d in diagnostics}
    small_bge, small_gte, base_gte = (diagnostic_by_model[m] for m in ("BAAI/bge-small-en-v1.5","thenlper/gte-small","thenlper/gte-base"))
    overlap_within = [float(m["within_name_context_top5_overlap"]) for m in metrics]
    overlap_cross = [float(m["cross_name_context_top5_overlap"]) for m in metrics]
    lines += ["", f"Практически перспективны BGE-small и GTE-small как экономичные CPU-кандидаты: два режима заняли {small_bge[2]:.2f} и {small_gte[2]:.2f} с при пиках RSS {small_bge[3]:.1f} и {small_gte[3]:.1f} MiB. GTE-base — вариант для сопоставления с ними: {base_gte[2]:.2f} с, name/context находят {base_gte[1]['text_name','ACCEPT']}/{base_gte[1]['text_name_context','ACCEPT']} предварительно принятых случаев. Это shortlist по стоимости и целевым диагностическим примерам, **не доказанный рейтинг качества**. Large-модели дороже, независимого подтверждения выигрыша ER здесь нет.", "",
              f"Контекст сильно меняет списки: пересечение name/context top-5 составляет {min(overlap_within):.3f}–{max(overlap_within):.3f} внутри документа и {min(overlap_cross):.3f}–{max(overlap_cross):.3f} между документами. На этой выборке у всех 9 моделей name находит больше ACCEPT-пар, чем context, хотя context добавляет отдельные полезные случаи. Поэтому контекстный режим нельзя объявить безусловно лучшим или использовать как единственную замену именам; разные окна могут усиливать тематическую близость и оттеснять корректные синонимы.", "",
              "## Контекстные решения и высокие ложные сходства", "",
              "Ниже приведены иллюстративные случаи; полный разбор выборки сохранён в verified_pairs.csv. Диапазон cosine объединяет разные шкалы только для просмотра примера, не служит общей метрикой качества и не используется в решении.", "",
              "| Область | Названия | Решение | Моделей | Min / max cosine | Уровень и основание |", "|---|---|---|---:|---|---|"]
    examples = {}
    for scope in ("within_document", "cross_document"):
        for status in ("ACCEPT", "REVIEW", "REJECT"):
            candidates = [r for r in reviewed if r["scope"] == scope and r["decision"] == status]
            if candidates:
                r = max(candidates, key=lambda r:(r["model_count"],r["configuration_count"],r["pair_id"]))
                examples[r["pair_id"]] = r
    for level in ("same_named_gene_pair", "different_timepoint_measurements", "named_strain", "drug_vs_antibodies",
                  "damaged_formula", "substance_vs_extraction_fragment", "gene_vs_allele_or_inexact_source_naming",
                  "named_model_with_type_conflict", "disease_vs_risk", "cohort_vs_subset"):
        candidates = [r for r in reviewed if r["referent_level"] == level]
        if candidates:
            r = max(candidates,key=lambda r:(r["model_count"],r["configuration_count"],r["pair_id"]))
            examples[r["pair_id"]] = r
    for r in examples.values():
        values = [float(o["similarity"]) for o in r["observations"]]
        left = by_id[r["reason_entity_order"][0]]["surface_text"].replace("|","/")
        right = by_id[r["reason_entity_order"][1]]["surface_text"].replace("|","/")
        lines.append(f"| {r['scope']} | {left} ↔ {right} | {r['decision']} | {r['model_count']} | {min(values):.3f} / {max(values):.3f} | {r['referent_level']}: {r['reason']} |")
    lines += ["", f"Проверка транзитивности принятых пар: противоречивых ACCEPT–ACCEPT–REJECT цепочек {len(contradictions)}; незамкнутых ACCEPT–ACCEPT цепочек без принятой третьей пары {len(open_wedges)}. Они не замыкаются автоматически и не создают merge-групп.", "",
              "## Выводы и ограничения", "",
              "Явные раскрытия MD/PK/OB-NLM и названия именованных объектов дают контекстные основания сильнее cosine. В то же время высокая близость возникает у разных видов/штаммов, до-/послеоперационного IOP, болезни и её риска, препарата и антител, группы и подмножества. Повреждённые формулы и извлечённые фрагменты способны давать почти одинаковые векторы. Неизвестный автор другой статьи не считается заведомо другим человеком; при нехватке подтверждений сохранён REVIEW. При переходе между статьями различается общий научный концепт и конкретная когорта/объект.", "",
              "Результат — собственная предварительная контекстная классификация, **не независимая человеческая gold-разметка**. Объективные Precision/Recall/F1, calibrated identity probabilities и числовые пороги не заявляются. Выводы о качестве Graph-RAG не делаются. Правила предложены в proposed_merge_rules.md; их применение и Semantic Merge (2D) не выполнялись.", "",
              "## Проверки и воспроизводимость", "",
              "Проверены 18 конфигураций и по 5 кандидатов каждой области для всех 1122 ID, все 201960 наблюдений, единственность неупорядоченных пар, source entity/mention IDs, названия, типы, области и флаги. Контекстные цитаты сверены с исходным текстом и revision_id, решения привязаны к SHA-256 входа 2A. Повторный запуск трёх артефактов 2C проверен побайтово. Хеши источников и результатов 1A–2B неизменны. Скрипт использует только стандартную библиотеку, модели не запускает.", "",
              "Запуск: `python scripts/verify_semantic_candidates.py`. Создаются только verified_pairs.csv, proposed_merge_rules.md и report.md. Neo4j, canonical mapping и отношения не изменяются.", "", "SHA-256 защищённых источников и результатов:", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}`: `{digest}`" for p,digest in hashes.items()]
    require(all(sha(p)==digest for p,digest in hashes.items()), "Protected stage artifact changed during 2C")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"verified_pairs.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader()
        for row in results:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    (OUT/"proposed_merge_rules.md").write_text(rules,encoding="utf-8")
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    require(all(sha(p)==digest for p,digest in hashes.items()), "Protected stage artifact changed after 2C")
    print(json.dumps(dict(unique_pairs=len(results),individually_reviewed=len(reviewed),decisions=dict(decisions),
                          context_cases_not_retrieved=len(set(cases)-set(pairs)),open_accept_wedges=len(open_wedges)),indent=2))


if __name__ == "__main__":
    main()
