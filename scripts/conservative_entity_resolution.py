"""Stage 1B only: snapshot-bound context rules, no inference or graph operations.

Run from the repository root: python scripts/conservative_entity_resolution.py
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from exact_matching_audit import ROOT, SOURCE, DOCUMENTS, REVIEWED_HASHES, normalize, read_jsonl, sha256

AUDIT = ROOT / "reports/entity_resolution/relex_base_t035/exact_matching_audit"
OUT = AUDIT.parent / "conservative_resolution"

# Explicit evidence rules for additional groups in the reviewed source snapshot.
# Columns: document | exact name | allowed start offsets | positive evidence |
# reason the remaining members stay separate. No rule generalizes to other data.
ADDITIONAL = """
W1979727947|breast cancer|81,253|Один диагноз: MD назван фактором риска для breast cancer, затем его биомаркером; оба упоминания относятся к заболеванию.|600 входит в breast cancer risk; фрагмент названия показателя риска оставлен отдельно.
W1979727947|cumulus|891,939,1201,1243|Одна явно названная полуаутоматическая мера Cumulus: измерение, его имитация и корреляция между ними; не сливаются Cumulus и автоматическая мера.|458 имеет ATTRIBUTE вместо MECHANISM; конфликтующие типы не объединяются.
W1995028339|genomes|716,767|The sizes of the seven genomes и their genomes в одном предложении явно обозначают один набор из семи геномов.|422 имеет ATTRIBUTE; 836 относится к подмножеству геномов Thalassiosirales.
W1995028339|rhizosolenia|669,1827|В обоих случаях тот же фрагмент полного имени Rhizosolenia imbricata: перечень видов и повторное описание потери генов; другого носителя имени нет.|-
W1995028339|thalassiosirales|283,847,964,1052|Один именованный таксон: вводится как order, затем обсуждаются размеры геномов, состав и порядок генов внутри этого же таксона.|1312 и 1606 имеют ACTOR/OBJECT, в отличие от OBJECT у выбранных упоминаний.
W2039836112|atp|254,1362|Одно вещество в описанной цепи: induced secretion of ATP и ATP is initially released from vesicles; сохраняется уровень вещества, не отдельных молекул.|877 и 972 имеют ATTRIBUTE вместо OBJECT.
W2039836112|fgf-1|1293,1341|В одном предложении сопоставлены before FGF-1 application и after FGF-1: одна явно указанная аппликация FGF-1.|963 и 1126 имеют другие наборы типов; исходный ACTOR у выбранной пары сохранён без исправления.
W2039836112|hcs|1486,1508|opening of HCs немедленно уточняется как These changes in HCs: прямая анафора к тем же описанным каналам.|1121 вложено в Px1 HCs, тогда как последующие HCs не уточняют подтип.
W2039836112|px1 hcs|1008,1117|В обоих случаях явно назван один молекулярный тип каналов Px1 HCs; меняется вклад этого типа через 2 и 7 часов, не его название/тип.|-
W2076293586|animals|237,651|animals were bred вводит животных двух моделей, затем these animals явно отсылает к этой совокупности.|-
W2078771716|inflation rates|990,1256|Одна переменная тестирования в описанной выборке стран: сначала перечислены inflation rates, затем дан вывод For inflation rates convergence can be rejected.|-
W2096307598|thickness|1130,1197|Одна характеристика одного носителя: свойства regolith, such as ... thickness distribution, затем lunar regolith's thickness; локальные числовые измерения не отождествляются.|-
W2364973104|military power station|122,247|Один тип целевой военной электростанции: проблема её lightning protection и модель на основе low-noise motor lightning in military power station; иная станция в тексте не вводится.|-
W2761455282|american knee society|788,1384|Оба вхождения входят в одинаковое полное название American Knee Society Scores — шкалы оценки в методах и результатах; самостоятельная организация здесь не вводится.|-
W2761455282|patients|516,650,694|21 patients вводит исследуемую когорту, последующие all patients и All patients относятся к той же когорте до разбиения по исходам.|Другие вхождения обозначают общую популяцию, умершего/20 оставшихся, шесть или 19 пациентов, либо общий вывод; 1054 также имеет конфликт типов.
W2761455282|tka|1363,1537,1787|Одна названная хирургическая методика Total Knee Arthroplasty (TKA) в результатах и выводах, а не тождество операций разных пациентов.|411 и 568 имеют MECHANISM и ACTOR вместо METHOD.
W2761455282|western ontario|825,1427|Оба вхождения являются тем же фрагментом полного названия Western Ontario and McMaster Universities Arthritis Index — одной шкалы оценки.|-
W2791492120|patients|615,853,1171,1287|В методах введена одна когорта из 10 пациентов; результаты и заключение повторяют пациентов этой когорты с CRSwNP, без введения другой когорты.|-
W2791492120|patients with crswnp|1171,1287|В результатах и заключении явно повторяется исследуемая когорта CRSwNP; подгруппа или другая выборка не вводятся.|853 содержит partial_token из исходной разметки; границы не исправляются.
W2906067930|grain|40,687|В обоих вхождениях контекст grain production in China указывает на один класс продукции в том же историческом случае 2003–2014; отдельные партии зерна не отождествляются.|-
W2906067930|grain production|40,687|Первое вхождение задаёт China's increase ... from 2003 to 2014, второе повторяет увеличение in China for 11 years from 2003 to 2014: тот же исторический процесс.|-
W2906067930|harvest paradox|74,791|Один именованный экономический принцип: отсутствие Harvest Paradox в 2003–2014 и повтор inconsistent with the Harvest Paradox in economics.|376 и 1053 имеют другие наборы типов.
W2906067930|phenomenon|4,290,591,607|В первом предложении phenomenon определён как рост производства зерна без Harvest Paradox в Китае в 2003–2014; все три the phenomenon отсылают к этому определённому случаю.|-
W3101922761|glaucoma|33,1617|Оба вхождения являются частью одного точного диагноза intractable glaucoma во вводном PRECIS и заключении.|Прочие вхождения входят в drainage device, outcome, surgery, medications, success либо обозначают контроль болезни без того же уточнения; автоматически не приравниваются.
W3101922761|patients|361,466|all patients в критерии ретроспективного отбора и 40 patients were studied описывают одну выборку GDD после PK за 2001–2017; другая выборка не вводится.|-
W3101922761|pk|404,1675|Одна названная процедура penetrating keratoplasty (PK) в описании отбора и в заключении after PK; объединяется вид процедуры, не отдельные операции.|-
W3123056951|allele|906,1236|the allele отсылает к ранее названному HLA-DRB1*09; в конце тот же аллель назван явно HLA-DRB1*09 allele.|-
W3123056951|hla-drb1*09|781,1224|При распознавании эпитопов 781 далее уточняется как the allele; 1224 прямо назван HLA-DRB1*09 allele: один именованный аллель.|611 явно сопровождается gene; ген и аллель не сливаются.
W3123056951|mutants|818,997|Оба вхождения входят в seventeen mutants: один предсказанный набор из 17, второе упоминает five out of этого набора; не весь набор приравнивается к пяти.|-
W3138328568|limonene|0,283,465,548,617,804,993,1161,1485,1586,1626|Первое предложение определяет химическое вещество Limonene; выбранные контексты описывают его производство, добавление, токсичность, толерантность и титр; другой химический референт не вводится.|1079 входит в limonene synthase gene и исключено как потенциальный фрагмент имени фермента/гена.
W4234628002|pnm|1203,1390|Одна названная Politeknik Negeri Madiun: деятельность SPMI in PNM и доступ её подразделения PNM P4MP к тому же порталу.|Оба исходных DATA_RESOURCE сохранены; организация не перетипизирована.
W4234628002|students|467,501|В одном предложении доступ by students немедленно продолжается allows the students: прямая анафора к тем же пользователям.|1303 задаёт целевую роль на странице questionnaire page for students; тождество конкретной совокупности пользователей не утверждается.
W4285525942|cooperation|456,563|cooperation between private investors and developers with green banks затем явно повторяется как such cooperation в том же предложении.|-
W4309030044|fares|440,583|Одна целевая переменная описанного исследования: researches the optimization of fares, затем модель for the optimization of fares; конкретные цены разных перевозчиков не приравниваются.|-
W4312422299|nephite|210,413|Один именованный народ в том же повествовании: Nephite people, descendants of Lehi, затем Nephite race; не разные семьи/поколения как отдельные лица.|-
W4389076493|arpn|951,1135|Одна модель, явно введённая как a novel adaptive region proposal network (ARPN), затем the ARPN; не отождествляется с Conventional RPNs.|1379 и 1786 имеют конфликтующие наборы типов; исходный ACTOR выбранной пары сохранён.
W4391351014|patients|246,553,649|В методах sixteen patients with esophageal cancer, в результатах Sixteen patients were enrolled, затем All patients received 28 fractions: та же когорта.|41 обозначает общую популяцию cancer patients в Background, а не исследованные 16 человек.
"""

# Contextual non-merge reasons for groups without a sufficient positive rule.
# Groups blocked by unresolved types/partial spans/damaged text also retain those
# concrete flags and their complete original contexts in review_groups.csv.
NOTES = """
W1979727947|mammographic images|Упоминаются дополнительные признаки и информация в изображениях, но не задан одинаковый набор снимков; есть training/test разбиение.
W1979727947|measure|Конструируемая автоматическая мера и well-established semiautomated measure (Cumulus) явно различаются.
W1979727947|r|Символ в определении Pearson coefficient и наблюдаемое значение r=0.884; показатель и конкретная оценка не приравниваются.
W1979727947|results|Заголовок RESULTS и self-reference Our results; заголовок раздела не считается отдельной сущностью того же набора результатов.
W1995028339|diatoms|Класс Diatoms и other diatoms вне Thalassiosirales имеют разный объём.
W1995028339|gene order|Общий порядок генов Thalassiosirales и общая конфигурация трёх названных видов; разные носители.
W1995028339|genes|ilvB/ilvH и acpP1 — разные гены.
W2039836112|secretion|Индукция секреции ATP и вывод из эксперимента с ингибиторами; тождество конкретного события не задано.
W2039836112|spinal astrocytes|Общий класс клеток и экспериментальные клетки in culture; нельзя отождествить популяции.
W2076293586|cnv|CNV regions и CNV data sets: фрагменты названий областей генома и данных о них.
W2076293586|extremes|Экстремумы по тревожности и стрессовой реактивности, далее экстремумы разных линий мышей.
W2076293586|stress reactivity|Патологическое состояние в общем введении и ось селекции мышей; совпадение носителя/состояния не установлено.
W2078771716|exchange rates|Exchange rates в наборе тестируемых переменных и более узкие nominal exchange rates; одинаковая область показателей не доказана.
W2084709926|camera|Fast camera и for example конкретная модель; пример не подтверждает выбор этой модели.
W2084709926|turbulence|Общая турбулентность, визуализация и предполагаемая локализация газовым импульсом; один эпизод/режим не задан.
W2096307598|degree of polarization|Общая величина и пространственное распределение по диску/лимбу; разные агрегаты.
W2096307598|temperature|Temperature sensitivity прибора и brightness temperature Луны — разные характеристики.
W2317432785|sublayer thicknesses|Разные толщины и образцы; в выводах larger и smaller, их измерения не приравниваются.
W2317432785|sublayers|Слои многослойной структуры в целом и конкретные Co-Fe-Zr слои; структура содержит также a-Si.
W2317432785|thicknesses|Толщины разных слоёв/образцов; отсутствует тождество набора измерений.
W2394633461|annual medals|94 медали, generic medals 1667/1691 и класс annual medals в целом; состав множеств различается.
W2394633461|medals|94 annual medals, более широкий класс commemorative medals и обсуждаемые источники; одинаковый набор не подтверждён.
W2394633461|papal|Прилагательное в papal commemorative medals и papal achievement; не имя одного ACTOR.
W2394633461|papal politics|Изменение на границе XVIII века и политика XVII века в целом; временные области не совпадают явно.
W2394633461|popes|10 изученных пап и попы XVII века в общем заключении; одинаковый состав явно не задан.
W2761455282|clinical outcome|Цель исследования всей когорты и улучшенный исход у большинства; результат и цель/агрегат различаются.
W2761455282|dialysis patients|Госпитальная когорта и общая популяция кандидатов на TKA; нельзя приравнять выборку к классу.
W2761455282|knee|Анатомический knee и фрагмент American Knee Society: разные значения.
W2761455282|medical complications|Случившиеся осложнения у шести пациентов и риск возможных будущих осложнений.
W2761455282|stroke|Один летальный инсульт и два случая осложнений; один диагноз не означает одно событие.
W2761455282|surgery|Предоперационный риск у когорты и операция умершего пациента; объём событий различается.
W2906067930|farmers|Историческая совокупность фермеров и адресаты будущей политики; одинаковая совокупность не определена.
W2906067930|income|Доходы domestic residents и farmers, исторические и будущие показатели; разные носители/периоды.
W2991563224|energy consumption|Общий показатель протокола и наблюдаемое снижение WSN; не установлено тождество измерений/режимов.
W2991563224|number of packets|Показатель пакетов, переданных base station, и сравнительный результат sent; область подсчёта/режим явно не отождествлены.
W3101922761|clear|Состояние трансплантатов в разные сроки и у семи сохранившихся; разные носители/время.
W3101922761|corneal grafts|Исходная когорта, подмножество из семи и общий вывод о выживаемости; состав меняется.
W3101922761|eyes|Общий класс, 40 глаз, подмножество из 20 и разные сроки наблюдения; состав и время различаются.
W3101922761|gdd|Имплантации устройств у пациентов и A GDD в общем выводе; экземпляр и класс не приравниваются, исходный ACTOR не исправляется.
W3101922761|intraocular pressure|Общий контролируемый показатель и postoperative показатель; тождество наблюдений не задано.
W3101922761|iop|Preoperative IOP 34.0 и postoperative IOP 12.7 — разные измерения.
W3123056951|bioactivity|Целевая биоактивность проектируемого набора и наблюдаемая активность пяти мутантов; разные носители/оценки.
W3123056951|development|Развитие PRCA в общей популяции и anti-rHuEpo-associated PRCA; одно событие не задано.
W3123056951|patients|Общая популяция с низким гемоглобином, подмножество с антителами и будущие кандидаты с HLA-DRB1*09; разные группы.
W3138328568|genes|Наборы из 82, 8, толерантность-повышающих и 5 генов; разный состав.
W3138328568|limonene production|Цель коэкспрессии и результат у пяти конструкций; тождество конкретных процессов/конструкций не задано.
W3138328568|microbes|Общий класс возможных микробных фабрик и engineered microbes; один набор организмов не определён.
W4230116451|real-time diagnosis|Общая задача e-health и её частный MRI case study; общий класс не приравнивается частному случаю.
W4285525942|financial instruments|Green financial instruments глобального рынка и main instruments of green banking; границы классов/наборов не установлены.
W4285525942|monetary|Дублированное прилагательное monetary, monetary and prudential policies; возможный дефект текста/выделения.
W4309030044|airlines|Конкуренты в разных рынках и желающие сотрудничать перевозчики; один набор компаний не задан.
W4312422299|destruction|Угроза семье Лехия и угроза народу нефийцев; разные носители/временные масштабы.
W4389076493|feature representations|Общие multi-scale representations и признаки FPN backbone; один набор тензоров/представлений не задан.
W4389076493|object detection|Tiny object detection на UAV и two-stage object detection вообще; разные объёмы задачи.
W4391351014|cancer|Cancer вообще и esophageal cancer — разные объёмы диагноза.
W4391351014|cied|Общий класс устройств в определении и устройства обследуемой когорты; экземпляры не отождествляются с классом.
W7215137876|development|Развитие emotional maturity и emotional skills; разные предметы процесса.
"""


def table(text, width):
    rows = [line.split("|") for line in text.strip().splitlines()]
    assert all(len(r) == width for r in rows)
    result = {(r[0], r[1]): r[2:] for r in rows}
    assert len(result) == len(rows)
    return result


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    inputs = [AUDIT / name for name in ("report.md", "exact_groups.csv", "ambiguous_groups.csv", "exact_merge_preview.jsonl")]
    inputs += [SOURCE / "entities.jsonl", SOURCE / "relations.jsonl", DOCUMENTS,
               Path(__file__).with_name("exact_matching_audit.py")]
    hashes = {p: sha256(p) for p in inputs}
    for p in inputs:
        if p.name in REVIEWED_HASHES:
            require(hashes[p] == REVIEWED_HASHES[p.name], f"Source snapshot changed: {p}")
    metadata, *preview = read_jsonl(AUDIT / "exact_merge_preview.jsonl")
    for item in metadata["source_manifest"]:
        require(sha256(ROOT / item["path"]) == item["sha256"], "1A source manifest mismatch")
    original = read_jsonl(SOURCE / "entities.jsonl")
    docs = {d["document_id"]: d for d in read_jsonl(DOCUMENTS)}
    entities = {e["entity_id"]: e for e in original}
    source_mentions = {m["mention_id"]: (e["entity_id"], m) for e in original for m in e["source_mentions"]}
    require(len(entities) == len(original) == 1211, "Expected 1211 unique entity IDs")
    require(len(source_mentions) == sum(len(e["source_mentions"]) for e in original) == 1327, "Invalid source mention IDs")
    groups = [g for g in preview if g["record_type"] == "exact_group"]
    initial = [g for g in groups if g["decision"] == "conservative_candidate"]
    csv_groups, csv_review = read_csv(AUDIT / "exact_groups.csv"), read_csv(AUDIT / "ambiguous_groups.csv")
    require((len(groups), len(initial), len(csv_groups), len(csv_review)) == (152, 25, 152, 127), "Unexpected 1A counts")
    require({g["group_id"] for g in csv_review} == {g["group_id"] for g in groups if g["decision"] == "review"}, "1A review CSV mismatch")
    preview_members = [m for g in preview for m in g["members"]]
    require(Counter(m["entity_id"] for m in preview_members) == Counter(entities.keys()), "1A coverage mismatch")
    for member in preview_members:
        require(all(member[k] == v for k, v in entities[member["entity_id"]].items()), "1A original properties mismatch")
    by_group = {g["group_id"]: g for g in groups}
    for row in csv_groups:
        g = by_group[row["group_id"]]
        require(json.loads(row["members"]) == g["members"] and row["decision"] == g["decision"], "1A exact CSV mismatch")
    additional, notes = table(ADDITIONAL, 5), table(NOTES, 3)
    group_keys = {(g["document_id"], g["normalized_name"]) for g in groups}
    require((set(additional) | set(notes)) <= group_keys and not set(additional) & set(notes), "Invalid context rule keys")
    mapping = {eid: eid for eid in entities}
    membership, reasons, group_for_entity = {}, {}, {}
    accepted, review = [], []
    for g in groups:
        key = (g["document_id"], g["normalized_name"])
        members = g["members"]
        for e in members:
            require(e["entity_id"] not in group_for_entity, "Overlapping exact groups")
            group_for_entity[e["entity_id"]] = g
        chosen, reason, remainder = [], "", ""
        if g["decision"] == "conservative_candidate":
            chosen, reason = members, g["decision_evidence"]
        elif key in additional:
            starts, reason, remainder = additional[key]
            offsets = {int(n) for n in starts.split(",")}
            chosen = [e for e in members if e["start"] in offsets]
            require(len(chosen) == len(offsets), f"Missing rule offsets: {key}")
        else:
            if key in notes:
                remainder = notes[key][0]
            else:
                require(g["type_conflict"] or g["multiple_types"] or
                        {"partial_token_span", "damaged_surface_text"} & set(g["risk_flags"]), f"Unassessed group: {key}")
                blockers = []
                if g["type_conflict"] or g["multiple_types"]:
                    blockers.append("Неразрешённые исходные типы по start: " + "; ".join(
                        f"{e['start']}={','.join(e['entity_types'])}" for e in members) + ".")
                if "partial_token_span" in g["risk_flags"]:
                    blockers.append("Есть исходный partial_token; границы span не исправляются.")
                if "damaged_surface_text" in g["risk_flags"]:
                    blockers.append("В исходном имени есть повреждение кодировки U+FFFD.")
                remainder = " ".join(blockers) + " Положительного правила тождества для совместимого поднабора нет. " + g["decision_evidence"]
        if chosen:
            require(len(chosen) >= 2 and reason, "Unsupported merge")
            require(len({tuple(e["entity_types"]) for e in chosen}) == 1 and len(chosen[0]["entity_types"]) == 1, f"Conflicting types: {key}")
            for e in chosen:
                require(e["document_id"] == g["document_id"] and normalize(e["surface_text"]) == key[1], "Cross-document/non-exact merge")
                require("\ufffd" not in e["surface_text"] and not any(t.get("partial_token", False) for m in e["source_mentions"] for t in m.get("tokens", [])), "Damaged/partial mention in merge")
            chosen.sort(key=lambda e: (e["start"], e["end"], e["entity_id"]))
            representative = chosen[0]["entity_id"]
            ids = [e["entity_id"] for e in chosen]
            for eid in ids:
                require(mapping[eid] == eid and eid not in membership, "Overlapping merge components")
                mapping[eid], membership[eid], reasons[eid] = representative, ids, reason
            accepted.append((g, ids, reason))
        if len(chosen) != len(members):
            require(remainder and remainder != "-", f"Missing remainder explanation: {key}")
            components = defaultdict(list)
            for e in members:
                components[mapping[e["entity_id"]]].append(e["entity_id"])
            review.append(dict(group_id=g["group_id"], document_id=key[0], normalized_name=key[1],
                               status="partial_merge" if chosen else "keep_separate",
                               reason=remainder, original_risk_flags=g["risk_flags"],
                               type_conflict=g["type_conflict"], multiple_types=g["multiple_types"],
                               canonical_components=dict(components),
                               members=[{**entities[e["entity_id"]], "context_start": e["context_start"],
                                         "context_end": e["context_end"], "context": e["context"],
                                         "canonical_entity_id": mapping[e["entity_id"]]} for e in members]))
    output = []
    for e in sorted(original, key=lambda e: (e["document_id"], e["start"], e["end"], e["entity_id"])):
        eid, doc = e["entity_id"], docs[e["document_id"]]
        require(doc["text"][e["start"]:e["end"]] == e["surface_text"], "Entity offset mismatch")
        for m in e["source_mentions"]:
            require((m["document_id"], m["start"], m["end"], m["surface_text"], m["revision_id"]) ==
                    (e["document_id"], e["start"], e["end"], e["surface_text"], doc["revision_id"]), "Source mention/revision mismatch")
        ids = membership.get(eid, [eid])
        merge_mentions = sorted(m["mention_id"] for i in ids for m in entities[i]["source_mentions"])
        g = group_for_entity.get(eid)
        reason = reasons.get(eid, next((r["reason"] for r in review if g and r["group_id"] == g["group_id"]), "Единственное упоминание exact-ключа в документе; оставлено отдельно."))
        contexts = []
        for i in ids:
            item = entities[i]
            a, b = max(0, item["start"] - 120), min(len(doc["text"]), item["end"] + 120)
            contexts.append(dict(entity_id=i, start=a, end=b, text=doc["text"][a:b]))
        decision = dict(stage="1B", method="snapshot_bound_context_rule", action="merge" if len(ids) > 1 else "keep",
                        audit_group_id=g["group_id"] if g else None,
                        origin="revalidated_1A" if g and g["decision"] == "conservative_candidate" else "1B",
                        reason=reason, context_evidence=contexts, source_revision_id=doc["revision_id"])
        for m in sorted(e["source_mentions"], key=lambda m: m["mention_id"]):
            output.append({**e, "mention_id": m["mention_id"], "source_mention": m,
                           "canonical_entity_id": mapping[eid], "canonical_member_entity_ids": ids,
                           "canonical_member_mention_ids": merge_mentions, "resolution": decision})
    require(len(output) == len({r["mention_id"] for r in output}) == 1327, "Incomplete/duplicate mention mapping")
    require({r["entity_id"] for r in output} == set(entities), "Incomplete entity mapping")
    for row in output:
        require(source_mentions[row["mention_id"]] == (row["entity_id"], row["source_mention"]), "Mention provenance mismatch")
        require(all(row[k] == v for k, v in entities[row["entity_id"]].items()), "Original properties changed")
        require(entities[row["canonical_entity_id"]]["document_id"] == row["document_id"], "Cross-document mapping")
        require(mapping[row["canonical_entity_id"]] == row["canonical_entity_id"], "Non-idempotent mapping")
    stats = dict(source_entities=len(entities), source_mentions=len(source_mentions),
                 merge_components=len(accepted), merged_entity_mentions=sum(len(ids) for _, ids, _ in accepted),
                 removed_repeats=sum(len(ids)-1 for _, ids, _ in accepted), canonical_entities=len(set(mapping.values())),
                 fully_resolved_groups=len(groups)-len(review), partial_groups=sum(r["status"] == "partial_merge" for r in review),
                 review_groups=len(review), revalidated_1A_groups=len(initial), additional_groups=len(accepted)-len(initial))
    require(stats["source_entities"] - stats["removed_repeats"] == stats["canonical_entities"], "Canonical count mismatch")
    require(all(sha256(p) == digest for p, digest in hashes.items()), "Input changed during resolution")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "canonical_mapping.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (OUT / "review_groups.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review[0]))
        writer.writeheader()
        for row in review:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    lines = ["# Этап 1B — Conservative Entity Resolution", "",
             "Источник: неизменённые результаты 1A и исходные JSONL namespace relex_base_t035. Только отображение ID внутри документа; операции с графом и отношениями не выполнялись.", "",
             "| Показатель | Количество |", "|---|---:|"]
    labels = [("Исходных сущностей (уникальных span/entity ID)", "source_entities"), ("Исходных NER mention ID", "source_mentions"),
              ("Объединений (компонент с ≥2 entity ID)", "merge_components"), ("Упоминаний-сущностей в объединениях", "merged_entity_mentions"),
              ("Устранённых повторов (сумма n−1)", "removed_repeats"), ("Канонических сущностей", "canonical_entities"),
              ("Полностью разрешённых exact-групп", "fully_resolved_groups"), ("Частично объединённых групп", "partial_groups"),
              ("Групп в очереди проверки, включая частичные", "review_groups")]
    lines += [f"| {label} | {stats[key]} |" for label, key in labels]
    lines += ["", f"Все 25 исходных кандидатов 1A повторно проверены и сохранены (32 сокращения). Все 127 оставшихся групп получили решение: {stats['additional_groups']} дополнительных групп объединены полностью или частично. У остальных сохранены отдельные ID и конкретные причины в review_groups.csv.", "",
              "Правило: точное название после нормализации 1A, один документ, положительное основание из исходного контекста, одинаковый единственный исходный тип у выбранных участников, отсутствие повреждённых/частичных span. Одинаковая строка, тип или confidence отдельно не доказывают тождество. Если основания относятся лишь к части группы, остальные ID остаются отдельными. Канонический ID — существующий entity ID первого участника по (start, end, ID); это обозначение в отображении, новый узел не создаётся.", "",
              "Разрешение относится к именованному объекту/классу, явно обсуждаемому в документе. Не утверждается тождество отдельных молекул, особей, приборов одного типа, лечебных событий или разных измерений. Исходные типы не исправлены даже там, где они выглядят ошибочными (например, FGF-1/ARPN: ACTOR, PNM: DATA_RESOURCE). Такие исходные ошибки ограничивают интерпретацию результата.", "",
              f"В 152 исходных группах: {sum(g['type_conflict'] for g in groups)} конфликтуют наборами типов, {sum(g['multiple_types'] for g in groups)} содержат несколько типов у одного span; признаки пересекаются. Разные значения order, разные множества genes/patients, IOP до/после операции, разные толщины и damaged Co-Fe-Zr оставлены раздельными. Конфликтующие типы никогда не входят в одно объединение.", "",
              "Примеры частичных решений: Cumulus — только четыре MECHANISM, без ATTRIBUTE; HLA-DRB1*09 — аллель без упоминания gene; limonene — вещество без фрагмента limonene synthase gene; patients — только явно повторённая когорта, без общей популяции или другого подмножества.", "",
              "Формат: canonical_mapping.jsonl содержит 1327 строк с настоящим исходным m:mention_id → canonical_entity_id и entity_id; покрыты все 1211 e:entity_id. Несколько NER mention ID одного span получают одинаковый canonical ID. Исходные поля entity, source_mentions и конкретный source_mention сохранены. canonical_member_entity_ids / canonical_member_mention_ids задают полный состав каждого объединения; resolution содержит основание и исходные контексты с offsets. Это позволяет восстановить также отображение всех entity_id → canonical_entity_id без неоднозначности.", "",
              "review_groups.csv: одна строка на не полностью разрешённую исходную группу, включая частичные решения; canonical_components показывает уже принятые подгруппы, members сохраняет offsets, типы, confidence, provenance и контекст. Отсутствие объединения означает недостаточность доказательств либо явное различие референтов.", "",
              "Проверены уникальность 1327 mention ID и 1211 entity ID источника, полнота, точность offsets/ревизий, неизменность свойств/provenance, однозначность и идемпотентность отображения, отсутствие междокументных объединений и конфликтов типов внутри принятых компонент. Детерминированные правила привязаны к SHA-256 снимка; модели, LLM, embeddings, cosine similarity и семантический поиск в скрипте не используются. Результат не является независимой человеческой gold-разметкой.", "",
              "Запуск: `python scripts/conservative_entity_resolution.py`. Создаются только три артефакта 1B. Импорта в Neo4j, переназначения отношений и этапов 1C/1D нет.", "", "SHA-256 входов (контроль до и после записи):", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}`: `{digest}`" for p, digest in hashes.items()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(all(sha256(p) == digest for p, digest in hashes.items()), "Input changed after writing")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
