"""Offline, standard-library-only exact matching audit. Never connects to Neo4j.

Run: python scripts/exact_matching_audit.py
Check edge cases: python scripts/exact_matching_audit.py --self-test
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/re_v2/relex_comparison/threshold035/e3_base"
DOCUMENTS = ROOT / "reports/ner_v2/research_run2/input_abstracts.jsonl"
OUTPUT = ROOT / "reports/entity_resolution/relex_base_t035/exact_matching_audit"
NAMESPACE = "relex_base_t035"
# Context decisions below apply exclusively to this reviewed input snapshot.
REVIEWED_HASHES = {
    "entities.jsonl": "19039d87119130dcc37d2c472a7f2242c6a3ad5059e46a1ff431e6fa37e3b6ba",
    "relations.jsonl": "bbb4e602abccedcc776b59cfd526f8ffe7ffb46bdea7f7dc3c38df33174dd099",
    "input_abstracts.jsonl": "684ca3a758a128c6382ab924f5c131f0e27faa9f7625b431ae0d6db975e543b1",
}
# Affirmative context evidence, not a name/type whitelist for other documents.
# Resolution concerns the named referent/class explicitly discussed, not individual
# members of a species, specimens, measurements or separate treatment events.
SUPPORTED = {
    ("W1979727947", "breast cancer risk"): "Одна целевая характеристика риска: построение, проверка и улучшение логистической модели этого риска.",
    ("W1979727947", "md"): "MD явно определено как mammographic density; оба упоминания относятся к измерению той же характеристики.",
    ("W1995028339", "cerataulina daemon"): "Один названный вид в перечне семи видов и затем как исключение по наличию ilvB/ilvH.",
    ("W1995028339", "roundia cardiophora"): "Один названный вид в перечне и в сравнении порядка генов; других одноимённых референтов нет.",
    ("W1995028339", "thalassiosira weissflogii"): "Один названный вид в перечне и в сравнении порядка генов.",
    ("W2084709926", "divertor"): "Оба упоминания относятся к дивертору установки NSTX, где размещён описываемый диагностический порт.",
    ("W2084709926", "x-point"): "Одна нижняя x-точка дивертора NSTX: геометрия, наблюдение и локализация турбулентности.",
    ("W2096307598", "moon"): "Оба упоминания обозначают Луну как единственный объект представленных радиоизмерений.",
    ("W2096307598", "regolith"): "Один класс материала — лунный реголит; сначала перечислены свойства, затем описано распределение толщины.",
    ("W2364973104", "military power supply"): "Обсуждение перенапряжения и заключение о защите относятся к описанной системе военного электропитания.",
    ("W2761455282", "study"): "Оба self-reference указывают на текущее ретроспективное исследование 21 пациента, не на прошлые работы.",
    ("W2791492120", "patients with crswnp"): "В результатах и заключении повторно обозначена исследуемая когорта CRSwNP; нет смешения с контрольной группой.",
    ("W2791492120", "study"): "Цель и заключение явно ссылаются на текущее исследование B-клеток.",
    ("W3027457624", "fiber-reinforced composites"): "Один класс материала: вводное описание и предложенный способ записи информации в нём.",
    ("W3101922761", "intractable glaucoma"): "Одна клиническая индикация для GDD после PK во вводном PRECIS и заключении.",
    ("W3123056951", "seventeen mutants"): "Один явно заданный набор из 17 предсказанных мутантов; второе упоминание задаёт подмножество five out of этого набора.",
    ("W3138328568", "y. lipolytica"): "Один вид дрожжей-хозяина; обсуждаются инженерия толерантности и выход продукта в этом виде.",
    ("W4285525942", "article"): "Все четыре self-reference относятся к текущей статье о зелёных банковских инструментах.",
    ("W4285525942", "author"): "Все три self-reference относятся к автору текущей статьи, другие авторы не введены.",
    ("W4285525942", "green bonds"): "Один класс зелёных облигаций: элемент списка инструментов и рекомендация увеличить его объём.",
    ("W4285525942", "green mortgages"): "Один класс зелёной ипотеки: элемент списка инструментов и оценка спроса на него.",
    ("W4309030044", "aviation"): "Один авиационный сектор в двух последовательных утверждениях о сотрудничестве с HSR.",
    ("W4312422299", "god"): "Один референт в одном повествовании: повеление Лехию, народ Бога и слово Бога.",
    ("W4312422299", "jerusalem"): "Один библейский город: место отправления семьи и место, которое велено покинуть.",
    ("W4391351014", "radiotherapy"): "Один вид лечения в том же протоколе исследуемой когорты; методы, результаты и наблюдение после лечения.",
    ("W4391351014", "rt"): "Оба упоминания RT в заключении обозначают тот же протокол лучевой терапии, без другой расшифровки.",
}
CONTEXT_RISKS = {
    ("W1995028339", "order"): "Полисемия: таксономический ранг order и ancestral gene order.",
    ("W1995028339", "genes"): "Разные явно названные гены: ilvB/ilvH и acpP1; одинаковый общий термин не задаёт один объект.",
    ("W1995028339", "gene order"): "Порядок генов разных таксонов: общее утверждение о Thalassiosirales и общая конфигурация трёх видов.",
    ("W2076293586", "extremes"): "Разные оси отбора: тревожность и стрессовая реактивность, затем экстремумы разных линий.",
    ("W2078771716", "convergence"): "Экономическая конвергенция в целом и разные выводы для валютных курсов, процентных ставок и инфляции.",
    ("W2084709926", "camera"): "Описание быстрой камеры и приведённый пример модели: пример не доказывает, что выбрана именно эта модель.",
    ("W2096307598", "brightness temperature"): "Разные агрегаты/области: весь диск, четыре квадранта и локальный максимум; нельзя сливать измерения.",
    ("W2096307598", "degree of polarization"): "Общая величина и пространственное распределение; различать характеристику и локальные измерения.",
    ("W2761455282", "knee"): "Анатомическое knee и фрагмент собственного названия American Knee Society: разные значения.",
    ("W2761455282", "stroke"): "Один летальный случай инсульта и два случая осложнений; один диагноз не доказывает одно событие.",
    ("W3101922761", "iop"): "Дооперационное и послеоперационное IOP: один показатель, разные измерения и значения.",
    ("W3101922761", "corneal grafts"): "Исходная группа трансплантатов и подмножество из семи сохранившихся; состав/время различаются.",
    ("W3123056951", "hla-drb1*09"): "Упоминания гена и аллеля; требуются явное решение о гранулярности и проверка типов референта.",
    ("W3138328568", "genes"): "Множества из 82, 8 и 5 генов и дополнительные гены синтазы; состав различается.",
    ("W3138328568", "limonene"): "Большинство упоминаний обозначает вещество, но span 1079:1087 входит в limonene synthase gene. Для полного объединения группы нужно проверить границу между веществом и фрагментом названия фермента/гена; безопасная подгруппа отдельно не моделируется.",
    ("W2906067930", "income"): "Доходы жителей в целом и доходы фермеров; различаются носители характеристики.",
    ("W3081936674", "soliton"): "Разные решения: dark soliton и combined soliton; модификаторы вне выделенного span существенны.",
    ("W4285525942", "monetary"): "Повтор при перечислении monetary, monetary and prudential policies; возможный дефект текста/извлечения.",
    ("W4391351014", "cancer"): "Общая популяция cancer patients и более узкая esophageal cancer; объём понятия различается.",
}


def normalize(text):
    """NFC preserves compatibility distinctions; no lemma/stopword/modifier edits."""
    return " ".join(unicodedata.normalize("NFC", unicodedata.normalize("NFC", text).casefold()).split())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate(entities, relations, mapping):
    """Keep every original relation record; also report optional dedup/no-loop views."""
    targets = {mapping[e["entity_id"]] for e in entities}
    triples, touched, externally_touched = set(), set(), set()
    self_loops = 0
    for r in relations:
        head, tail = mapping[r["head_id"]], mapping[r["tail_id"]]
        triples.add((r["document_id"], head, r["relation_type"], tail))
        touched.update((head, tail))
        if head == tail:
            self_loops += 1
        else:
            externally_touched.update((head, tail))
    return {
        "canonical_entities": len(targets), "removed_entity_nodes": len(entities) - len(targets),
        "isolated_nodes": len(targets - touched),
        "isolated_nodes_if_self_loops_removed": len(targets - externally_touched),
        "retained_relation_records": len(relations), "self_loop_records": self_loops,
        "unique_directed_typed_triples": len(triples),
        "duplicate_triple_records": len(relations) - len(triples),
        "unique_triples_without_self_loops": sum(h != t for _, h, _, t in triples),
    }


def audit(entities, relations, documents):
    by_id = {e["entity_id"]: e for e in entities}
    if len(by_id) != len(entities):
        raise ValueError("Duplicate entity IDs")
    incident, incoming, outgoing = defaultdict(list), Counter(), Counter()
    for r in relations:
        h, t = r["head_id"], r["tail_id"]
        if by_id[h]["document_id"] != r["document_id"] or by_id[t]["document_id"] != r["document_id"]:
            raise ValueError("Cross-document or missing relation endpoint")
        outgoing[h] += 1
        incoming[t] += 1
        for eid in {h, t}:
            incident[eid].append({k: r[k] for k in ("triple_id", "head_id", "tail_id", "relation_type", "confidence")})
    grouped = defaultdict(list)
    for e in entities:
        doc = documents[e["document_id"]]
        start, end = e["start"], e["end"]
        if not 0 <= start < end <= len(doc["text"]) or doc["text"][start:end] != e["surface_text"]:
            raise ValueError(f"Offset mismatch: {e['entity_id']}")
        for m in e["source_mentions"]:
            if (m["start"], m["end"], m["document_id"], m["surface_text"], m["revision_id"]) != (
                    start, end, e["document_id"], e["surface_text"], doc["revision_id"]):
                raise ValueError("Source mention or revision mismatch")
        grouped[e["document_id"], normalize(e["surface_text"])].append(e)
    full = {eid: eid for eid in by_id}
    conservative = full.copy()
    records = []
    by_doc = defaultdict(list)
    for e in entities:
        by_doc[e["document_id"]].append(e)
    for (doc_id, name), members in sorted(grouped.items()):
        members.sort(key=lambda e: (e["start"], e["end"], e["entity_id"]))
        doc = documents[doc_id]
        # An existing ID is only a deterministic representative in this preview.
        representative = members[0]["entity_id"]
        enriched = []
        for e in members:
            a, b = e["start"], e["end"]
            left, right = max(0, a - 160), min(len(doc["text"]), b + 160)
            containers = sorted({normalize(o["surface_text"]) for o in by_doc[doc_id]
                                 if o["start"] <= a and b <= o["end"] and o["entity_id"] != e["entity_id"]})
            enriched.append({**e, "neo4j_key": f"{NAMESPACE}:{e['entity_id']}",
                             "incident_relation_count": len(incident[e["entity_id"]]),
                             "in_relation_count": incoming[e["entity_id"]],
                             "out_relation_count": outgoing[e["entity_id"]],
                             "isolated": not incident[e["entity_id"]],
                             "context_start": left, "context_end": right,
                             "context": doc["text"][left:right], "containing_entity_names": containers,
                             "incident_relations": incident[e["entity_id"]]})
        types = {tuple(e["entity_types"]) for e in members}
        type_conflict = len(types) > 1
        multi_type = any(len(t) != 1 for t in types)
        contexts_differ = len({normalize(e["context"]) for e in enriched}) > 1
        nested_differ = len({tuple(e["containing_entity_names"]) for e in enriched}) > 1
        partial = any(t.get("partial_token", False) for e in members for m in e["source_mentions"] for t in m.get("tokens", []))
        invalid = "\ufffd" in name or any(unicodedata.category(c) == "Cc" and not c.isspace() for c in name)
        reasons = []
        if type_conflict:
            reasons.append("type_set_conflict")
        if multi_type:
            reasons.append("multiple_or_missing_types")
        if nested_differ:
            reasons.append("different_containing_entity_names")
        if partial:
            reasons.append("partial_token_span")
        if invalid:
            reasons.append("damaged_surface_text")
        key = (doc_id, name)
        supported = len(members) > 1 and key in SUPPORTED and not (type_conflict or multi_type or partial or invalid)
        if key in CONTEXT_RISKS:
            reasons.append("referent_or_scope_difference")
            supported = False
        if len(members) > 1 and not supported:
            reasons.append("identity_requires_review")
        explanation = SUPPORTED[key] if supported else CONTEXT_RISKS.get(key, "")
        if not explanation and len(members) > 1:
            explanation = "Нет достаточного положительного подтверждения общего референта; совпадение названия/типа и наличие связей сами по себе не доказывают идентичность."
        group_id = "exact:" + hashlib.sha256(json.dumps([doc_id, name], ensure_ascii=False).encode()).hexdigest()[:20]
        isolated_count = sum(e["isolated"] for e in enriched)
        mixed = 0 < isolated_count < len(members)
        for e in members:
            full[e["entity_id"]] = representative
            if supported:
                conservative[e["entity_id"]] = representative
        records.append({
            "record_type": "exact_group" if len(members) > 1 else "singleton",
            "group_id": group_id, "document_id": doc_id, "normalized_name": name,
            "mention_count": len(members), "potential_reductions": len(members) - 1,
            "full_exact_representative_id": representative,
            "conservative_representative_ids": {e["entity_id"]: conservative[e["entity_id"]] for e in members},
            "decision": "conservative_candidate" if supported else ("review" if len(members) > 1 else "keep"),
            "type_conflict": type_conflict, "multiple_types": multi_type,
            "contexts_differ": contexts_differ, "potential_ambiguity": len(members) > 1 and not supported,
            "risk_flags": reasons, "decision_evidence": explanation,
            "isolated_mentions": isolated_count, "linked_mentions": len(members) - isolated_count,
            "mixed_linked_isolated": mixed, "mixed_same_object_supported": mixed and supported,
            "incident_relation_count": len({r["triple_id"] for e in enriched for r in e["incident_relations"]}),
            "document_title": doc["title"], "document_revision_id": doc["revision_id"],
            "document_text": doc["text"], "members": enriched,
        })
    return records, {"baseline": simulate(entities, relations, {eid: eid for eid in by_id}),
                     "full_exact": simulate(entities, relations, full),
                     "conservative": simulate(entities, relations, conservative)}


def write_csv(path, groups):
    fields = ["group_id", "document_id", "normalized_name", "mention_count", "potential_reductions",
              "decision", "type_conflict", "multiple_types", "contexts_differ", "potential_ambiguity",
              "risk_flags", "decision_evidence", "isolated_mentions", "linked_mentions", "mixed_linked_isolated",
              "mixed_same_object_supported", "incident_relation_count", "entity_ids", "offsets", "types",
              "confidence", "surface_texts", "relation_counts", "contexts", "members"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for g in groups:
            members = g["members"]
            row = {k: g[k] for k in fields if k in g}
            row.update(entity_ids=[e["entity_id"] for e in members],
                       offsets=[[e["start"], e["end"]] for e in members],
                       types=[e["entity_types"] for e in members],
                       confidence=[e["confidence"] for e in members],
                       surface_texts=[e["surface_text"] for e in members],
                       relation_counts=[e["incident_relation_count"] for e in members],
                       contexts=[e["context"] for e in members])
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def report(groups, scenarios, manifest):
    safe = [g for g in groups if g["decision"] == "conservative_candidate"]
    review = [g for g in groups if g["decision"] == "review"]
    mixed = [g for g in groups if g["mixed_linked_isolated"]]
    baseline = scenarios["baseline"]["isolated_nodes"]
    lines = ["# Exact Matching Audit — relex_base_t035", "",
             "Dry-run по исходным JSONL, без подключения к Neo4j. Сохранённые summary.json и отчёт импорта подтверждают namespace: 30 документов, 1211 сущностей, 1971 отношение. Текущее состояние живой БД не проверялось.", "",
             "## Результат", "",
             f"Найдено **{len(groups)} групп**, {sum(g['mention_count'] for g in groups)} узлов-упоминаний, **{sum(g['potential_reductions'] for g in groups)} потенциально устраняемых повторов**. Между документами объединение не рассматривается.", "",
             f"Консервативно допустимы **{len(safe)} групп / {sum(g['potential_reductions'] for g in safe)} повторов**: проверен контекст и записано положительное основание. **{len(review)} групп / {sum(g['potential_reductions'] for g in review)} повторов требуют проверки**. Это оценка по тексту, не gold-разметка и не гарантия идентичности.", "",
             "| Сценарий | Канонических сущностей | Сокращение узлов | Изолированных | Снижение изолированных |",
             "|---|---:|---:|---:|---:|"]
    for name, label in [("baseline", "Исходный"), ("full_exact", "Полный exact merge (верхняя граница)"), ("conservative", "Консервативный")]:
        s = scenarios[name]
        lines.append(f"| {label} | {s['canonical_entities']} | {s['removed_entity_nodes']} | {s['isolated_nodes']} | {baseline-s['isolated_nodes']} |")
    lines += ["", "Изоляция означает отсутствие входящих и исходящих RELEX_REL. Служебные RELEX_MENTIONS от документа не учитываются. Петля считается связью; отдельный расчёт без петель приведён ниже.", "",
              "## Изолированные упоминания в связанных группах", "",
              f"Всего **{len(mixed)} смешанных групп**, в них **{sum(g['isolated_mentions'] for g in mixed)} изолированных упоминаний**. Консервативно подтверждён общий предполагаемый референт в **{sum(g['mixed_same_object_supported'] for g in mixed)} группах**, где **{sum(g['isolated_mentions'] for g in mixed if g['mixed_same_object_supported'])} изолированных упоминаний** получают связи канонического представителя.", ""]
    for label, selected in [("Полный exact", groups), ("Консервативный", safe)]:
        absorption = sum(g["isolated_mentions"] for g in selected if g["mixed_linked_isolated"])
        collapse = sum(g["mention_count"] - 1 for g in selected if g["isolated_mentions"] == g["mention_count"])
        lines.append(f"- {label}: снижение изолированных = {absorption} за счёт присоединения к связанным + {collapse} за счёт схлопывания полностью изолированных групп. Новые фактические связи не извлекаются.")
    lines += ["", "Смешанные группы с подтверждением контекста:", "",
              "| Документ | Название | Связанных / изолированных | Основание |", "|---|---|---:|---|"]
    for g in mixed:
        if g["mixed_same_object_supported"]:
            lines.append(f"| {g['document_id']} | {g['normalized_name']} | {g['linked_mentions']} / {g['isolated_mentions']} | {g['decision_evidence']} |")
    lines += ["", "Все смешанные группы доступны через mixed_linked_isolated; mixed_same_object_supported выделяет консервативно подтверждённые. Остальные смешанные группы остаются на проверке.", "",
              "## Ограничения и признаки риска", "",
              f"Конфликт наборов типов: **{sum(g['type_conflict'] for g in groups)} групп**; несколько типов хотя бы у одного узла: **{sum(g['multiple_types'] for g in groups)}**; отличающиеся локальные контексты: **{sum(g['contexts_differ'] for g in groups)}**. Эти признаки могут пересекаться. Различие контекстов само по себе не доказывает различие объектов.", "",
              "Примеры, оставленные на проверке: order — таксономический ранг против порядка генов; convergence — разные экономические показатели; genes — разные гены/множества; knee — анатомия против названия шкалы; soliton — dark против combined; IOP — измерения до и после операции. Проблемы кодировки (U+FFFD), частичные токены, разные внешние модификаторы/вложенные сущности также отмечены.", "",
              "## Метод", "",
              "Ключ: (document_id, NFC → casefold → NFC → сжатие Unicode-пробелов). Используется исходный surface_text, не normalized_forms предыдущего нормализатора. Пунктуация, дефисы, диакритика, числа, смысловые слова и модификаторы сохраняются; NFC не сводит совместимые символы как NFKC. Offsets — исходные Python Unicode-индексы [start,end), проверены по тексту и source_mentions. Уверенность NER и все исходные типы/оценки сохранены; confidence не является вероятностью идентичности.", "",
              "Консервативное решение требует явного объяснения общего референта в конкретном документе и одного согласованного типа без повреждённого текста/частичных токенов. Проверены контексты исходных документов; основания в SUPPORTED скрипта привязаны к SHA-256 снимка. Нет embeddings, семантического поиска, новых NER/RE или объединения разных строк. Разрешение проводится на уровне названного объекта/класса; не утверждается тождество отдельных особей, образцов или лечебных событий. Не разрешённые уверенно группы целиком оставлены на проверке; частичное слияние подгрупп не моделируется.", "",
              "## Связи после гипотетического переназначения", "",
              "| Сценарий | Сохранено записей | Уникальных направленных типизированных троек | Дубли записей | Петли (записей) | Изолированных при удалении петель |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in scenarios.items():
        lines.append(f"| {name} | {s['retained_relation_records']} | {s['unique_directed_typed_triples']} | {s['duplicate_triple_records']} | {s['self_loop_records']} | {s['isolated_nodes_if_self_loops_removed']} |")
    lines += ["", "Основной сценарий сохраняет все 1971 записи отношений и петли; дедупликация — лишь отдельная оценка по (документ, head, relation_type, tail). Нет записи нового графа. Представитель — первое упоминание по offsets; все IDs, provenance и свойства остаются в preview.", "",
              "## Артефакты и воспроизведение", "",
              "- exact_groups.csv: одна строка на повторяющуюся группу, все IDs/offsets/типы/confidence/контексты/связи; массивы — JSON в ячейках.",
              "- ambiguous_groups.csv: подмножество exact_groups.csv с decision=review, включая недостаточно подтверждённые группы.",
              "- exact_merge_preview.jsonl: metadata со статистикой и SHA-256; затем одна запись на гипотетическую сущность полного exact merge, включая одиночные узлы. members сохраняет исходные упоминания, source_mentions и incident_relations; mappings описывают оба сценария. Это план, не импортируемый граф.",
              "- scripts/exact_matching_audit.py: один скрипт, только стандартная библиотека Python.", "",
              "Запуск из корня: `python scripts/exact_matching_audit.py`; проверки: `python scripts/exact_matching_audit.py --self-test`.", "",
              "SHA-256 источников (также проверяются после записи аудита):", ""]
    for item in manifest:
        lines.append(f"- `{item['path']}`: `{item['sha256']}`")
    return "\n".join(lines) + "\n"


def self_test():
    assert normalize("  E\u0301NERGY\u00a0\tBalance  ") == "énergy balance"
    assert normalize("Straße") == normalize("STRASSE")
    assert normalize("dark soliton") != normalize("combined soliton")
    assert normalize("breast cancer risk") != normalize("breast cancer")
    assert normalize("Ａ") != normalize("A")  # NFC, not NFKC
    es = [{"entity_id": i} for i in "abcde"]
    rs = [{"document_id": "d", "head_id": "a", "tail_id": "b", "relation_type": "R"},
          {"document_id": "d", "head_id": "b", "tail_id": "a", "relation_type": "R"}]
    s = simulate(es, rs, dict(zip("abcde", "aacdd")))
    assert s["canonical_entities"] == 3 and s["isolated_nodes"] == 2
    assert s["self_loop_records"] == 2 and s["duplicate_triple_records"] == 1
    assert s["isolated_nodes_if_self_loops_removed"] == 3
    text = "order order"
    def entity(eid, a, t):
        return dict(entity_id=eid, document_id="d", start=a, end=a+5, surface_text="order",
                    entity_types=[t], confidence=0.9, source_mentions=[])
    records, _ = audit([entity("a", 0, "OBJECT"), entity("b", 6, "CONCEPT")], [],
                       {"d": dict(text=text, title="test", revision_id="rev")})
    assert records[0]["type_conflict"] and records[0]["decision"] == "review"
    # Same surface and same type remain unproved; never merge across documents.
    third = {**entity("c", 0, "OBJECT"), "document_id": "other"}
    docs = {d: dict(text=text, title="test", revision_id="rev") for d in ("d", "other")}
    records, s = audit([entity("a", 0, "OBJECT"), entity("b", 6, "OBJECT"), third],
                       [dict(triple_id="r", document_id="d", head_id="a", tail_id="a",
                             relation_type="R", confidence=0.9)], docs)
    assert len(records) == 2 and records[0]["mixed_linked_isolated"]
    assert records[0]["decision"] == "review" and not records[0]["mixed_same_object_supported"]
    assert s["baseline"]["isolated_nodes"] == 2 and s["full_exact"]["isolated_nodes"] == 1
    assert s["conservative"] == s["baseline"]
    try:
        audit([{**entity("bad", 0, "OBJECT"), "surface_text": "wrong"}], [], docs)
    except ValueError:
        pass
    else:
        raise AssertionError("Offset mismatch was not rejected")
    print("Self-test passed: normalization/modifiers, loops, isolation, duplicates, type conflicts, document scope, unproved identity, mixed groups, offsets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    paths = [SOURCE / "entities.jsonl", SOURCE / "relations.jsonl", DOCUMENTS, SOURCE / "summary.json", SOURCE / "experiment_config.json", SOURCE / "report.md"]
    manifest = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)} for p in paths]
    for p, item in zip(paths, manifest):
        if p.name in REVIEWED_HASHES and item["sha256"] != REVIEWED_HASHES[p.name]:
            raise ValueError(f"Reviewed snapshot changed: {p}; context decisions must be re-audited")
    entities, relations = read_jsonl(paths[0]), read_jsonl(paths[1])
    docs_list = read_jsonl(DOCUMENTS)
    documents = {d["document_id"]: d for d in docs_list}
    summary = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    if (len(entities), len(relations), len(documents)) != (1211, 1971, 30):
        raise ValueError("Wrong experiment snapshot")
    if len({r["triple_id"] for r in relations}) != len(relations) or len(documents) != len(docs_list):
        raise ValueError("Duplicate relation or document IDs")
    if summary["neo4j"]["entities"] != len(entities) or summary["neo4j"]["relations"] != len(relations):
        raise ValueError("Import summary counts disagree")
    records, scenarios = audit(entities, relations, documents)
    groups = [g for g in records if g["record_type"] == "exact_group"]
    if not set(SUPPORTED).issubset({(g["document_id"], g["normalized_name"]) for g in groups}):
        raise ValueError("Missing reviewed groups")
    for scenario, selected in [("full_exact", groups), ("conservative", [g for g in groups if g["decision"] == "conservative_candidate"])]:
        reduction = sum(g["potential_reductions"] for g in selected)
        isolation_drop = sum(g["isolated_mentions"] if g["linked_mentions"] else g["mention_count"]-1 for g in selected)
        assert scenarios[scenario]["canonical_entities"] == len(entities) - reduction
        assert scenarios[scenario]["isolated_nodes"] == scenarios["baseline"]["isolated_nodes"] - isolation_drop
    out = args.output.resolve()
    # Guard against placing any generated artifact in the original experiment.
    protected = [SOURCE.resolve(), DOCUMENTS.parent.resolve()]
    if any(out == p or out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        raise ValueError("Output must be separate from original experiment directories")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "exact_groups.csv", groups)
    write_csv(out / "ambiguous_groups.csv", [g for g in groups if g["decision"] == "review"])
    with (out / "exact_merge_preview.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        metadata = dict(record_type="metadata", schema_version=1, namespace=NAMESPACE, dry_run=True,
                        source_mode="original_jsonl", live_neo4j_verified=False, source_manifest=manifest,
                        normalization="NFC/casefold/NFC/Unicode whitespace", scenarios=scenarios,
                        reviewed_policy="explicit document-context evidence; single consistent type; snapshot-bound")
        for row in [metadata, *records]:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "report.md").write_text(report(groups, scenarios, manifest), encoding="utf-8")
    if any(sha256(p) != m["sha256"] for p, m in zip(paths, manifest)):
        raise RuntimeError("A source file changed during audit")
    print(json.dumps({"output": str(out), "exact_groups": len(groups),
                      "conservative_groups": sum(g["decision"] == "conservative_candidate" for g in groups),
                      "scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
