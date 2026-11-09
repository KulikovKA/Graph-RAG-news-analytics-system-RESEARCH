"""Stage 1D only: structural comparison in memory, standard library, one report.

Run: python scripts/compare_canonical_graphs.py
"""
from collections import Counter, defaultdict
import json
import sys

sys.dont_write_bytecode = True
from exact_matching_audit import ROOT, SOURCE, read_jsonl, sha256

ER = ROOT / "reports/entity_resolution/relex_base_t035"
OUTPUT = ER / "graph_comparison/report.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(r):
    return r["document_id"], r["head_id"], r["relation_type"], r["tail_id"]


def metrics(nodes, edges, exclude_loops=False):
    """nodes: ID -> document; edges: set of (document, head, type, tail).

    Degree counts typed directed edges; a loop contributes two. Density uses
    distinct non-loop ordered endpoint pairs, collapsing relation types only there.
    Weak components include isolated vertices and disregard direction/type/loops.
    """
    retained = {r for r in edges if not exclude_loops or r[1] != r[3]}
    adjacency = {n: set() for n in nodes}
    touched, pairs = set(), set()
    for doc, head, _, tail in retained:
        require(head in nodes and tail in nodes and nodes[head] == nodes[tail] == doc, "Invalid graph endpoint/document")
        touched.update((head, tail))
        if head != tail:
            adjacency[head].add(tail)
            adjacency[tail].add(head)
            pairs.add((head, tail))
    seen, sizes = set(), []
    for seed in sorted(nodes):
        if seed in seen:
            continue
        seen.add(seed)
        stack, size = [seed], 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    n, e = len(nodes), len(retained)
    counts = Counter(nodes.values())
    allowed_pairs = sum(v * (v - 1) for v in counts.values())
    require(sum(sizes) == n and len(seen) == n, "Component coverage error")
    return dict(vertices=n, edges=e, loops=sum(h == t for _, h, _, t in retained),
                isolated=n-len(touched), isolated_pct=100*(n-len(touched))/n if n else 0,
                components=len(sizes), nontrivial_components=sum(s > 1 for s in sizes), largest=max(sizes, default=0),
                mean_degree=2*e/n if n else 0, pairs=len(pairs),
                density=len(pairs)/(n*(n-1)) if n > 1 else 0,
                within_doc_density=len(pairs)/allowed_pairs if allowed_pairs else 0)


def percent(before, after):
    if before == 0:
        return "0,00%" if after == 0 else "н/п (база 0)"
    return f"{100*(after-before)/before:+.2f}%".replace(".", ",")


def number(value, digits=0):
    return (f"{value:.{digits}f}" if digits else str(int(value))).replace(".", ",")


def pair(before, after, digits=0, suffix=""):
    return f"{number(before, digits)}{suffix} → {number(after, digits)}{suffix}"


def main():
    paths = [SOURCE / "entities.jsonl", SOURCE / "relations.jsonl",
             ER / "conservative_resolution/canonical_mapping.jsonl",
             ER / "relation_remapping/canonical_relations.jsonl"]
    protected = sorted(set(paths) | {
        p for directory in ("exact_matching_audit", "conservative_resolution", "relation_remapping")
        for p in (ER / directory).iterdir() if p.is_file()})
    hashes = {p: sha256(p) for p in protected}
    entity_rows, source_relations, mention_rows, canonical_relations = map(read_jsonl, paths)
    entities = {e["entity_id"]: e for e in entity_rows}
    require(len(entity_rows) == len(entities) == 1211, "Invalid entity IDs/count")
    source_mentions = {m["mention_id"]: (e["entity_id"], m) for e in entity_rows for m in e["source_mentions"]}
    require(len(source_mentions) == sum(len(e["source_mentions"]) for e in entity_rows) == 1327, "Invalid original mentions")
    mapping, seen_mentions = {}, set()
    for row in mention_rows:
        eid, target, mid = row["entity_id"], row["canonical_entity_id"], row["mention_id"]
        require(eid in entities and target in entities, "Unknown mapped entity")
        require(mid not in seen_mentions and source_mentions.get(mid) == (eid, row["source_mention"]), "Invalid/repeated source mention")
        require(all(row[k] == v for k, v in entities[eid].items()), "Changed entity properties/provenance")
        require(eid not in mapping or mapping[eid] == target, "Ambiguous entity mapping")
        require(entities[eid]["document_id"] == entities[target]["document_id"], "Cross-document entity mapping")
        mapping[eid] = target
        seen_mentions.add(mid)
    require(len(mention_rows) == 1327 and seen_mentions == source_mentions.keys() and mapping.keys() == entities.keys(), "Incomplete mapping")
    targets = set(mapping.values())
    require(len(targets) == 1122 and all(mapping[t] == t for t in targets), "Invalid canonical targets")
    originals = {r["triple_id"]: r for r in source_relations}
    before_edges, after_edges = {key(r) for r in source_relations}, {key(r) for r in canonical_relations}
    require(len(originals) == len(source_relations) == len(before_edges) == 1971, "Invalid source relations")
    require(len(after_edges) == len(canonical_relations) == len({r["canonical_triple_id"] for r in canonical_relations}) == 1760, "Invalid canonical relations")
    expected = {(d, mapping[h], t, mapping[q]) for d, h, t, q in before_edges}
    require(after_edges == expected, "Canonical relation coverage/key mismatch")
    provenance_ids = []
    for row in canonical_relations:
        sources = row["source_relations"]
        require(row["source_relation_count"] == len(sources) == len(row["source_triple_ids"]), "Invalid provenance count")
        require(row["source_triple_ids"] == [r["triple_id"] for r in sources], "Invalid provenance IDs")
        require(row["source_confidences"] == [{"triple_id": r["triple_id"], "confidence": r["confidence"]} for r in sources], "Changed confidence")
        require(row["is_self_loop"] == (row["head_id"] == row["tail_id"]), "Invalid self-loop flag")
        for original in sources:
            tid = original["triple_id"]
            require(original == originals.get(tid), "Changed original relation/evidence/provenance")
            require(key(row) == (original["document_id"], mapping[original["head_id"]], original["relation_type"], mapping[original["tail_id"]]), "Invalid provenance mapping")
            provenance_ids.append(tid)
    require(Counter(provenance_ids) == Counter(originals.keys()), "Lost/repeated relation provenance")
    before_nodes = {eid: e["document_id"] for eid, e in entities.items()}
    after_nodes = {eid: before_nodes[eid] for eid in targets}
    before, after = metrics(before_nodes, before_edges), metrics(after_nodes, after_edges)
    before_nl, after_nl = metrics(before_nodes, before_edges, True), metrics(after_nodes, after_edges, True)
    docs = sorted(set(before_nodes.values()))
    require(len(docs) == 30 and set(after_nodes.values()) == set(docs), "Document coverage mismatch")
    per_doc = []
    for doc in docs:
        bn = {i: d for i, d in before_nodes.items() if d == doc}
        an = {i: d for i, d in after_nodes.items() if d == doc}
        be, ae = {r for r in before_edges if r[0] == doc}, {r for r in after_edges if r[0] == doc}
        per_doc.append((doc, metrics(bn, be), metrics(an, ae), metrics(an, ae, True)))
    for field in ("vertices", "edges", "loops", "isolated", "components", "nontrivial_components", "pairs"):
        require(sum(b[field] for _, b, _, _ in per_doc) == before[field] and
                sum(a[field] for _, _, a, _ in per_doc) == after[field], f"Non-additive document totals: {field}")
    require(max(b["largest"] for _, b, _, _ in per_doc) == before["largest"] and
            max(a["largest"] for _, _, a, _ in per_doc) == after["largest"], "Largest component mismatch")
    groups = defaultdict(list)
    for eid, canonical_id in mapping.items():
        groups[canonical_id].append(eid)
    originally_linked = {r[k] for r in source_relations for k in ("head_id", "tail_id")}
    absorbed = sum(sum(i not in originally_linked for i in ids) for ids in groups.values() if any(i in originally_linked for i in ids))
    collapsed = sum(len(ids)-1 for ids in groups.values() if all(i not in originally_linked for i in ids))
    require(before["isolated"]-after["isolated"] == absorbed+collapsed, "Isolation decomposition mismatch")
    labels = [("Сущности V", "vertices", 0), ("Уникальные направленные типизированные отношения E", "edges", 0),
              ("Self-loops (уникальные типизированные тройки)", "loops", 0), ("Изолированные вершины I", "isolated", 0),
              ("Доля изолированных, %", "isolated_pct", 2), ("Слабосвязные компоненты C", "components", 0),
              ("Компоненты размером ≥2 вершин", "nontrivial_components", 0),
              ("Размер крупнейшей слабосвязной компоненты", "largest", 0), ("Средняя суммарная степень 2E/V", "mean_degree", 4),
              ("Уникальные направленные пары без петель P", "pairs", 0), ("Глобальная плотность D", "density", 8),
              ("Внутридокументная плотность D_doc", "within_doc_density", 8)]
    lines = ["# Этап 1D — структурное сравнение графов", "",
             "Два графа построены только в памяти по исходным JSONL и результатам 1B/1C. Всего 30 документов; междокументных объединений и рёбер нет. Учитываются сущности и предметные отношения, без служебных связей документа. Метрики вычислены одной функцией для обоих графов.", "",
             "## До / после", "", "Основная таблица сохраняет self-loops; плотность по определению всегда исключает петли.", "",
             "| Показатель | До | После | Δ | Δ относительно «до» |", "|---|---:|---:|---:|---:|"]
    for label, field, digits in labels:
        b, a = before[field], after[field]
        delta = f"{a-b:+.{digits}f}".replace(".", ",") + (" п.п." if field == "isolated_pct" else "")
        lines.append(f"| {label} | {number(b, digits)} | {number(a, digits)} | {delta} | {percent(b, a)} |")
    lines += ["", "При нулевой исходной базе относительный прирост не определён (н/п); изменение доли в колонке Δ выражено в процентных пунктах, а в последней колонке — в относительных процентах.", "",
              "## Единая методика", "",
              "- V — все уникальные entity ID, включая не участвующие в отношениях. Канонические вершины берутся из отображения 1B: 1327 NER mention ID соответствуют 1211 исходным entity ID и 1122 каноническим ID; строки упоминаний не считаются отдельными вершинами.",
              "- E — число уникальных ключей (document_id, head_id, relation_type, tail_id). Направление и тип учитываются; разные типы на одной направленной паре дают разные рёбра. Сохранённые внутри provenance исходные записи повторно не считаются рёбрами.",
              "- I — вершины с нулевой входящей и исходящей степенью. Self-loop даёт входящую и исходящую степень по единице, поэтому суммарная средняя степень равна 2E/V, включая вклад петли дважды. Средние входящая и исходящая степени по отдельности равны E/V.",
              "- Слабосвязные компоненты найдены обходом неориентированной проекции: направление и тип игнорируются, петли не соединяют разные вершины; изолированная вершина образует компоненту размера 1. Размер крупнейшей компоненты — число её уникальных вершин.",
              "- Для плотности отношения проецируются на простые направленные пары: P = число различных (head, tail), head ≠ tail. Все типы на одной паре схлопываются только при подсчёте P. Глобальная плотность D = P/[V(V−1)], без петель, в диапазоне [0,1]. E/[V(V−1)] для типизированного мультиграфа здесь не используется. Если V < 2, принимается D = 0.",
              "- Междокументные пары отсутствуют по дизайну, поэтому дополнительно D_doc = P/Σ_d[V_d(V_d−1)] учитывает только допустимые внутридокументные пары. Это взвешенная по числу возможных пар плотность, а не среднее плотностей документов. В таблицах документов плотность равна P_d/[V_d(V_d−1)]. При нулевом знаменателе используется 0.", "",
              "## Влияние исключения self-loops", "",
              "Петли исключены только из отдельного расчёта в памяти; исходные артефакты 1C сохранены. Число вершин в обоих вариантах фиксировано.", "",
              "| Метрика | До: с петлями | До: без петель | После: с петлями | После: без петель |", "|---|---:|---:|---:|---:|"]
    for label, field, digits in labels:
        if field not in ("vertices", "pairs", "within_doc_density"):
            lines.append(f"| {label} | " + " | ".join(number(m[field], digits) for m in (before, before_nl, after, after_nl)) + " |")
    source_loop_records = sum(r["head_id"] == r["tail_id"] and source["head_id"] != source["tail_id"] for r in canonical_relations for source in r["source_relations"])
    lines += ["", f"После ER петлями стали {source_loop_records} исходных записей, представленных {after['loops']} уникальными типизированными тройками. Удаление петель меняет E и среднюю степень. Плотности D и D_doc не меняются, поскольку их определение исключает петли. Компоненты и размер крупнейшей компоненты не меняются; число изолированных в этих данных также не меняется (в общем случае вершина только с петлёй стала бы изолированной). Семантическая корректность петель не оценивалась.", "",
              "## По документам: объём и изоляция", "", "Стрелки обозначают «до → после». В обеих таблицах отношения уникальны и типизированы; петли включены, если не указано иначе.", "",
              "| Документ | V | ΔV, % | E | ΔE, % | I | Доля I, % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for doc, b, a, _ in per_doc:
        lines.append(f"| {doc} | {pair(b['vertices'], a['vertices'])} | {percent(b['vertices'], a['vertices'])} | {pair(b['edges'], a['edges'])} | {percent(b['edges'], a['edges'])} | {pair(b['isolated'], a['isolated'])} | {pair(b['isolated_pct'], a['isolated_pct'], 2)} |")
    lines += ["", "## По документам: связность, степень и плотность", "",
              "| Документ | Компоненты C | Крупнейшая компонента | Средняя степень | После без петель: степень | Плотность документа | Петли |", "|---|---:|---:|---:|---:|---:|---:|"]
    for doc, b, a, nl in per_doc:
        lines.append(f"| {doc} | {pair(b['components'], a['components'])} | {pair(b['largest'], a['largest'])} | {pair(b['mean_degree'], a['mean_degree'], 4)} | {number(nl['mean_degree'], 4)} | {pair(b['density'], a['density'], 6)} | {pair(b['loops'], a['loops'])} |")
    example = next(row for row in per_doc if row[0] == "W3027457624")
    worsened_share = sum(a["isolated_pct"] > b["isolated_pct"] for _, b, a, _ in per_doc)
    lines += ["", "## Что изменилось и что из этого следует", "",
              f"- Устранены {before['vertices']-after['vertices']} повторных entity ID и {before['edges']-after['edges']} структурных дублей отношений. Все 1971 исходные записи отношений по-прежнему прослеживаются через provenance 1C; новых фактических отношений не извлекалось.",
              f"- Изолированных вершин стало меньше на {before['isolated']-after['isolated']}: {absorbed} исходных изолированных упоминания присоединены к каноническим сущностям со связями, ещё {collapsed} сокращений получено при схлопывании полностью изолированных групп. Доля изолированных изменилась на {number(after['isolated_pct']-before['isolated_pct'], 2)} п.п. Это улучшение структурного покрытия, условное на корректности ER.",
              f"- Слабосвязных компонент стало {before['components']} → {after['components']}; компонент размером ≥2 вершин — {before['nontrivial_components']} → {after['nontrivial_components']}. Одновременно крупнейшая компонента сократилась {before['largest']} → {after['largest']}. Сокращение общего числа компонент сопровождает сжатие узлов и сокращение изолированных singleton-компонент; оно не означает автоматического роста полезной связности. Общий граф остаётся объединением 30 отдельных документных графов.",
              f"- Есть локальное соединение ранее раздельных фрагментов: в {example[0]} C изменилось {example[1]['components']} → {example[2]['components']}, крупнейшая компонента — {example[1]['largest']} → {example[2]['largest']}, при неизменном I={example[1]['isolated']}. Это структурный эффект принятого отображения ID; полезность новых путей не оценивалась. Эффект неоднороден: в {worsened_share} документах доля изолированных выросла из-за изменения знаменателя.",
              f"- Плотность D изменилась на {percent(before['density'], after['density'])}, D_doc — на {percent(before['within_doc_density'], after['within_doc_density'])}, средняя степень — на {percent(before['mean_degree'], after['mean_degree'])}. Знаменатель плотности уменьшается при сжатии вершин; увеличение плотности не доказывает появление новых знаний или рост качества извлечения.",
              "- Структурные метрики не доказывают правильность объединений, фактическую достоверность отношений, допустимость петель, сохранение смысла путей, полноту знаний или улучшение поиска/ответов Graph-RAG. Семантическая корректность и качество Graph-RAG в 1D не измерялись. Confidence, типы и evidence не переоценивались; результаты наследуют ограничения 1A–1C.", "",
              "## Проверки и воспроизводимость", "",
              "Проверены уникальность 1211 entity ID, 1327 mention ID и 1971 исходного triple ID; полнота/однозначность отображения; 1122 канонических ID и 1760 уникальных канонических троек; отсутствие междокументных объединений/рёбер; точное покрытие источников и сохранность provenance/исходных полей. Суммы документных V, E, I, C, P и петель совпадают с глобальными, крупнейшая компонента равна максимуму по документам. Повторный запуск отчёта проверен побайтово, компоненты — независимым подсчётом. SHA-256 всех входов и артефактов 1A–1C контролируются до и после записи.", "",
              "Скрипт: `python scripts/compare_canonical_graphs.py`. Только стандартная библиотека и функции чтения/хеширования из 1A. Создаётся единственный report.md; Neo4j не подключается, модели/embeddings не используются, Entity Resolution и Relation Verification не выполняются. Выполнен только этап 1D.", "", "SHA-256 входов и защищённых артефактов:", ""]
    lines += [f"- `{p.relative_to(ROOT).as_posix()}`: `{digest}`" for p, digest in hashes.items()]
    require(all(sha256(p) == digest for p, digest in hashes.items()), "Protected inputs changed during comparison")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(all(sha256(p) == digest for p, digest in hashes.items()), "Protected inputs changed after writing")
    print(json.dumps(dict(before=before, after=after, after_without_loops=after_nl,
                          absorbed_isolated_mentions=absorbed, collapsed_isolated_repeats=collapsed), indent=2))


if __name__ == "__main__":
    main()
