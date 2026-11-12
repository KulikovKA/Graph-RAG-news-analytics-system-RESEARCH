"""Analyze saved 2F.4 predictions and an explicit, context-based human review.

No models, embeddings, database writes, or additional candidate discovery.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict
from itertools import combinations

sys.dont_write_bytecode = True
import analyze_cross_document_re_2f4 as run

OUT, core, base = run.OUT, run.core, run.base


def format_candidate_report():
    path = OUT / 'candidate_report.md'
    lines = path.read_text(encoding='utf-8').splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('| Категория |'))
    first_row = next(i for i in range(header + 2, len(lines)) if lines[i].startswith('|'))
    if first_row > header + 2:
        note = lines[header + 2:first_row]
        lines = lines[:header] + note + lines[header:header + 2] + lines[first_row:]
    archive_note = 'Полный архив всех 3236 обнаруженных путей: all_evidence_paths.jsonl. Файл evidence_paths.jsonl содержит представительные пути, выбранные до inference; его байты после запуска модели не менялись.'
    if archive_note not in lines:
        lines += ['', archive_note]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def archive_all_paths(mapping, facts, semantic, concepts, profiles, candidates, representatives):
    """Archive every enumerated path, including those not supplied to ReLEx.

    This is lossless bookkeeping of the frozen discovery, not a new experiment.
    The smaller inference path file remains byte-for-byte unchanged.
    """
    bridges = []
    for s in semantic:
        if s['original_cross_document']:
            bridges.append(dict(bridge_id=s['semantic_link_id'], kind='SIMILAR_TO', x1=s['head_id'], x2=s['tail_id'],
                source_ids=[s['semantic_link_id']], decision_2f2=s['decision'],
                similarity=float(s['original_link']['similarity']), identity=False))
    grouped = defaultdict(list)
    for c in concepts:
        grouped[c['tail_id']].append(c)
    for cid, links in sorted(grouped.items()):
        for a, b in combinations(links, 2):
            if profiles[a['head_id']]['document_id'] != profiles[b['head_id']]['document_id']:
                bridges.append(dict(bridge_id=run.identity('bridge:', [cid, a['concept_link_id'], b['concept_link_id']]),
                    kind='HAS_SHARED_CONCEPT', x1=a['head_id'], x2=b['head_id'], concept_id=cid,
                    source_ids=[a['concept_link_id'], b['concept_link_id']], identity=False))
    neighbors = defaultdict(list)
    for f in facts:
        h, t, fid = mapping[f['head_id']], mapping[f['tail_id']], f['canonical_triple_id']
        if h != t:
            neighbors[h].append((t, fid, 'forward'))
            neighbors[t].append((h, fid, 'reverse'))
    archive = {}
    counts = Counter()
    for bridge in run.progress(bridges, desc='Archiving all frozen discovery paths', unit='bridge'):
        x1, x2 = bridge['x1'], bridge['x2']
        bridge['generic_intermediate'] = any(profiles[x]['name'].casefold() in run.GENERIC for x in (x1, x2))
        for a, f1, d1 in sorted(neighbors[x1]):
            for b, f2, d2 in sorted(neighbors[x2]):
                if a == b or profiles[a]['document_id'] == profiles[b]['document_id']:
                    continue
                pid = run.identity('pair:', sorted((a, b)))
                path_id = run.identity('path:', [a, f1, bridge['bridge_id'], f2, b])
                path = dict(path_id=path_id, left_id=a, right_id=b, intermediate_entity_ids=[x1, x2],
                    factual_relation_ids=[f1, f2], bridge=bridge,
                    factual_traversals=[dict(from_id=a, to_id=x1, direction='reverse' if d1 == 'forward' else 'forward'),
                                        dict(from_id=x2, to_id=b, direction=d2)],
                    document_ids=[profiles[a]['document_id'], profiles[b]['document_id']],
                    factual_evidence_refs=[f1, f2], identity_assumed=False, entails_endpoint_relation=False)
                if path_id in representatives:
                    run.require(path == representatives[path_id], 'Archive differs from frozen inference path')
                run.require(path_id not in archive, 'Duplicate archived path')
                archive[path_id] = dict(path, pair_id=pid)
                counts[pid] += 1
    run.require(all(counts[p] == row['evidence_path_count'] for p, row in candidates.items()), 'Full discovery path counts differ')
    run.require(set(representatives) <= set(archive), 'Missing representative path')
    run.jsonl('all_evidence_paths.jsonl', sorted(archive.values(), key=lambda p: p['path_id']))
    return len(archive)


def main():
    run.banner('НАЧИНАЮ ПРОВЕРКУ КАЧЕСТВА ПРЕДСКАЗАНИЙ')
    format_candidate_report()
    original, mapping, entities, docs, facts, semantic, concepts, profiles = run.inputs()
    candidates = {r['pair_id']: r for r in core.read_jsonl(OUT / 'candidate_pairs.jsonl')}
    results = core.read_jsonl(OUT / 'relex_predictions.jsonl')
    selected = {k for k, r in candidates.items() if r['selected_for_inference']}
    run.require({r['pair_id'] for r in results} == selected and len(results) == len(selected), 'Inference incomplete')
    manifest = run.load('manifest.json')
    coverage = []
    for source in 'ABCD':
        for scope in ('discovered', 'selected_for_inference'):
            pool = [r for r in candidates.values() if source in r['sources'] and
                    (scope == 'discovered' or r['selected_for_inference'])]
            document_ids = sorted({r[k] for r in pool for k in ('left_document_id', 'right_document_id')})
            coverage.append(dict(source=source, scope=scope, pairs=len(pool), documents=len(document_ids),
                document_ids=json.dumps(document_ids), disciplines=json.dumps(sorted({docs[d]['discipline'] for d in document_ids}))))
    run.csvout('candidate_source_coverage.csv', coverage)
    run.require(all(r['inference_signature'] == manifest['inference']['signature'] for r in results), 'Mixed inference signatures')
    predictions = []
    path_index = {r['path_id']: r for r in core.read_jsonl(OUT / 'evidence_paths.jsonl')}
    fact_index = {r['canonical_triple_id']: r for r in core.read_jsonl(OUT / 'source_factual_evidence.jsonl')}
    all_path_count = archive_all_paths(mapping, facts, semantic, concepts, profiles, candidates, path_index)
    for result in run.progress(results, desc='Validating saved pair results', unit='pair'):
        pair = candidates[result['pair_id']]
        text = result['model_input']['text']
        run.require(result['model_input']['synthetic'] and not result['model_input']['is_source_evidence'], 'Synthetic input mislabeled')
        run.require(result['audit']['spans'] == 2 and result['audit']['pairs'] == 2 and result['audit']['tokens'] <= 512, 'Bad adapter audit')
        for m in result['fixed_spans']:
            run.require(text[m['start']:m['end']] == m['surface_text'], 'Fixed span changed')
        for fragment in result['source_fragments'] + result['path_fragments']:
            run.require(docs[fragment['document_id']]['text'][fragment['start']:fragment['end']] == fragment['text'], 'Source fragment changed')
        if result['included_path_id']:
            path = path_index[result['included_path_id']]
            run.require(set(path['document_ids']) == {pair['left_document_id'], pair['right_document_id']}, 'Path document mismatch')
            for fid in path['factual_relation_ids']:
                run.require(fid in fact_index, 'Unknown factual evidence ID')
        for relation in result['relations']:
            run.require(relation['confidence'] > .35 and relation['relation_type'] in core.PREDICATES, 'Bad prediction')
            run.require({relation['head_id'], relation['tail_id']} == {pair['left_id'], pair['right_id']}, 'Wrong prediction endpoint')
            run.require(profiles[relation['head_id']]['document_id'] != profiles[relation['tail_id']]['document_id'], 'Not cross-document')
            predictions.append(dict(relation, pair_id=pair['pair_id']))
    run.require(len({r['candidate_id'] for r in predictions}) == len(predictions), 'Duplicate directed typed triple')
    run.require(run.load('pilot_validation.json')['passed'], 'Pilot gate not passed')
    run.require(base.sha(OUT / 'pilot_review.md') == run.load('pilot_validation.json')['content_review_sha256'], 'Pilot review changed after gate')
    with (OUT / 'quality_review.csv').open(encoding='utf-8-sig', newline='') as f:
        review = list(csv.DictReader(f))
    by_id = {r['candidate_id']: r for r in predictions}
    run.require(len({r['candidate_id'] for r in review}) == len(review), 'Repeated review row')
    for row in run.progress(review, desc='Checking explicit quality annotations', unit='review'):
        relation = by_id[row['candidate_id']]
        run.require(row['classification'] in ('SUPPORTED_INFERENCE', 'HYPOTHESIS', 'REJECTED'), 'Unknown review classification')
        run.require(row['reason'] and row['evidence_ids'] and row['review_method'] == 'contextual_manual_audit', 'Review lacks reasoning/evidence')
        run.require(row['head_id'] == relation['head_id'] and row['tail_id'] == relation['tail_id'] and row['relation_type'] == relation['relation_type'], 'Review triple mismatch')
        allowed = {profiles[x]['document_id'] for x in (relation['head_id'], relation['tail_id'])}
        allowed.update(relation['evidence_path_ids'])
        for path_id in relation['evidence_path_ids']:
            allowed.update(path_index[path_id]['factual_relation_ids'])
            allowed.update(path_index[path_id]['bridge']['source_ids'])
        run.require(set(json.loads(row['evidence_ids'])) <= allowed, 'Unresolvable review evidence ID')
    reviewed = {r['candidate_id']: r for r in review}
    run.require(set(reviewed) == set(by_id), 'This run audits every prediction; missing annotation')
    baseline_pairs = run.graph_pairs(mapping, facts, semantic, concepts)
    concept_ids = {c['tail_id'] for c in concepts}
    baseline, membership = run.components(profiles, baseline_pairs, concept_ids)
    baseline_digest = core.digest([[(mapping[f['head_id']], f['relation_type'], mapping[f['tail_id']], f['canonical_triple_id']) for f in facts],
                                 semantic, concepts])
    run.banner('НАЧИНАЮ АНАЛИЗ 12 ПОРОГОВ: 0.35–0.90')
    table = []
    for threshold in run.progress(run.THRESHOLDS, desc='Threshold analysis', unit='threshold'):
        kept = [r for r in predictions if r['confidence'] >= threshold]
        graph, _ = run.components(profiles, baseline_pairs + [(r['head_id'], r['tail_id']) for r in kept], concept_ids)
        audited = [reviewed[r['candidate_id']] for r in kept if r['candidate_id'] in reviewed]
        counts = Counter(r['classification'] for r in audited)
        row = dict(threshold=threshold, predictions=len(kept),
            unique_directed_triples=len({(r['head_id'], r['relation_type'], r['tail_id']) for r in kept}),
            positive_candidate_pairs=len({r['pair_id'] for r in kept}),
            documents=len({profiles[r[s]]['document_id'] for r in kept for s in ('head_id', 'tail_id')}),
            with_evidence_paths=sum(bool(r['evidence_path_ids']) for r in kept),
            without_evidence_paths=sum(not r['evidence_path_ids'] for r in kept),
            cross_baseline_components=sum(membership[r['head_id']] != membership[r['tail_id']] for r in kept),
            virtual_wcc=graph['weak_components'], virtual_largest_component=graph['largest_component_vertices'],
            virtual_largest_component_documents=graph['largest_component_documents'],
            baseline_relex_preserved=1307, baseline_similar_preserved=492, baseline_concept_links_preserved=118,
            reviewed=len(audited), reviewed_supported=counts['SUPPORTED_INFERENCE'], reviewed_hypothesis=counts['HYPOTHESIS'],
            reviewed_rejected=counts['REJECTED'],
            reviewed_rejected_fraction=counts['REJECTED'] / len(audited) if audited else '',
            reviewed_doubtful_or_rejected_fraction=(counts['HYPOTHESIS'] + counts['REJECTED']) / len(audited) if audited else '')
        row.update({'type_' + code: sum(r['relation_type'] == code for r in kept) for code in core.PREDICATES})
        table.append(row)
    run.csvout('threshold_comparison.csv', table)
    run.require(all(a['predictions'] >= b['predictions'] for a, b in zip(table, table[1:])), 'Non-monotonic thresholds')
    # Read-only before/after verifies all source properties, all namespaces and old files.
    after = run.live_snapshot()
    run.require(after == manifest['before_neo4j'], 'Neo4j changed during research')
    actual_protected = run.protected_files()
    run.require(actual_protected == manifest['source_files_sha256'], 'Protected source artifact changed')
    run.require(baseline_digest == core.digest([[(mapping[f['head_id']], f['relation_type'], mapping[f['tail_id']], f['canonical_triple_id']) for f in facts], semantic, concepts]), 'Baseline mutated')
    previous = core.read_jsonl(base.ER / 'cross_document_re/cross_document_relations.jsonl')
    previous = [p for p in previous if p.get('record_type') == 'pair_result']
    previous_predictions = [r for p in previous for r in p['relations']]
    with (OUT / 'previous_quality_review.csv').open(encoding='utf-8-sig', newline='') as f:
        previous_review = list(csv.DictReader(f))
    run.require({r['candidate_id'] for r in previous_review} == {r['candidate_id'] for r in previous_predictions}, 'Old audit incomplete')
    old_types = Counter(r['relation_type'] for r in previous_predictions)
    new_types = Counter(r['relation_type'] for r in predictions)
    quality = Counter(r['classification'] for r in review)
    comparison = dict(previous_pairs=len(previous), previous_predictions=len(previous_predictions),
        previous_types=dict(old_types), current_types=dict(new_types), new_relation_types=sorted(new_types.keys() - old_types.keys()),
        previous_review_counts=dict(Counter(r['classification'] for r in previous_review)),
        current_review_counts=dict(quality), current_reviewed=len(review),
        reviewed_with_paths=sum(bool(by_id[r['candidate_id']]['evidence_path_ids']) for r in review),
        current_predictions_with_paths=sum(bool(r['evidence_path_ids']) for r in predictions),
        current_results_has_result_fraction=sum(profiles[r['head_id']]['name'].casefold() == 'results' and
            profiles[r['tail_id']]['name'].casefold() == 'results' and r['relation_type'] == 'HAS_RESULT' for r in predictions) / len(predictions),
        previous_results_has_result_fraction=old_types['HAS_RESULT'] / len(previous_predictions),
        previous_rejected_fraction=sum(r['classification'] == 'REJECTED' for r in previous_review) / len(previous_review),
        current_rejected_fraction=sum(r['classification'] == 'REJECTED' for r in review) / len(review),
        current_predicted_disciplines=sorted({profiles[r[s]]['discipline'] for r in predictions for s in ('head_id', 'tail_id')}),
        previous_predicted_disciplines=sorted({docs[r[s + '_document_id']]['discipline'] for r in previous_predictions for s in ('head', 'tail')}),
        pair_overlap_after_2f3_mapping=len({tuple(sorted((mapping[r['left']['canonical_entity_id']], mapping[r['right']['canonical_entity_id']])))
            for r in previous if r['left']['canonical_entity_id'] in mapping and r['right']['canonical_entity_id'] in mapping}
            & {tuple(sorted((p['left_id'], p['right_id']))) for p in candidates.values() if p['selected_for_inference']}),
        comparison_is_paired_experiment=False)
    run.save('experiment_comparison.json', comparison)
    validation = dict(passed=True, namespace=run.NS, baseline=baseline, candidate_inventory=len(candidates),
        selected_pairs=len(selected), processed_pairs=len(results), negative_pairs=sum(not r['relations'] for r in results),
        predictions=len(predictions), unique_directed_triples=len(by_id),
        all_discovered_paths_archived=all_path_count, representative_paths=len(path_index),
        all_offsets_exact=True, fixed_spans_only=True, both_directions_evaluated=True,
        no_token_truncation=True, all_source_fragments_exact=True, negative_results_saved=True,
        thresholds=run.THRESHOLDS, threshold_rounding_before_filter=False, inference_repeated_for_thresholds=False,
        baseline_multiset_sha256=baseline_digest, all_1307_facts_preserved=True, all_492_similar_to_preserved=True,
        all_118_concept_links_preserved=True, neo4j_before=manifest['before_neo4j'], neo4j_after=after,
        all_neo4j_namespaces_unchanged=True, protected_source_files_unchanged=True,
        protected_source_file_count=len(actual_protected), quality_audit_counts=dict(quality),
        review_is_targeted_not_unbiased_precision=True, NER_rerun=False, embeddings_rerun=False,
        all_predictions_audited=True, checkpoint_pair_ids_unique=True,
        entity_resolution_changed=False, other_LLMs_launched=False, neo4j_import=False, stage_2f4d_started=False)
    run.save('validation.json', validation)
    report(manifest, validation, table, comparison, candidates, profiles, results, review)
    artifacts = {p.name: base.sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'}
    manifest['artifacts_sha256'] = artifacts
    manifest['analysis_runner_sha256'] = base.sha(Path(__file__))
    manifest['model_execution_accounting'] = dict(preliminary_pilot_unique_pairs=16,
        preliminary_smoke_repeat_calls=1, final_pilot_unique_pairs=16, final_smoke_repeat_calls=1,
        final_full_run_new_pairs=20, final_unique_pairs=36, total_forward_calls_this_research=54,
        threshold_analysis_forward_calls=0,
        note='Preliminary selection was revised during stage A/pilot; its outputs are retained separately, never mixed into the final 36 pairs')
    manifest['finished_utc'] = run.datetime.now(run.timezone.utc).isoformat()
    run.save('manifest.json', manifest)
    print(json.dumps(dict(verified=True, predictions=len(predictions), reviewed=len(review), quality=dict(quality)), ensure_ascii=False), flush=True)


def report(manifest, validation, table, comparison, candidates, profiles, results, review):
    q = validation['quality_audit_counts']
    lines = ['# 2F.4A–C — Evidence-grounded cross-document ReLEx', '',
        '## Результат', '',
        f"Обработано {len(results)} отобранных пар из {len(candidates)} обнаруженных; сохранено {validation['predictions']} предсказаний "
        f"и {validation['negative_pairs']} отрицательных результатов. Содержательно проверено {len(review)} предсказаний: "
        f"SUPPORTED_INFERENCE {q.get('SUPPORTED_INFERENCE', 0)}, HYPOTHESIS {q.get('HYPOTHESIS', 0)}, REJECTED {q.get('REJECTED', 0)}.", '',
        'Neo4j не изменён. Все 1307 RELEX_REL, 492 SIMILAR_TO, 118 HAS_SHARED_CONCEPT, 811 контекстных сущностей, '
        '32 концепта и 30 документов сохранены. Сверены SHA всех свойств и концов рёбер текущего namespace и остальных namespace до/после. '
        '2F.4D и 2F.5 не выполнялись.', '',
        '## Методика и ограничения', '',
        'Полная методика и примеры до inference — candidate_report.md. Не все 682 discovery-кандидата подавались в модель: '
        'исследовательская выборка заранее ограничена по источникам и парам документов; selected_for_inference и причины сохранены. '
        'Счёт источников перекрывается. Все 121 исходное междокументное SIMILAR_TO учтены, после ER они соответствуют 113 парам. '
        'Покрытие документов и дисциплин отдельно по каждому источнику и отбору — candidate_source_coverage.csv. '
        'Сходство не интерпретируется как тождество. Общие endpoints и пути только через общие промежуточные слова выделены в ограниченные диагностические контроли.', '',
        'C использует один SIMILAR_TO либо две HAS_SHARED_CONCEPT через один концепт и по одному factual ребру с каждой стороны. '
        'Все 3236 обнаруженных путей сохранены в all_evidence_paths.jsonl; отдельный evidence_paths.jsonl содержит до трёх представительных путей на пару для входа модели. Оригинальные factual IDs, направления, старые/новые endpoint IDs, '
        'контексты, offsets, mention IDs, revision IDs и source URLs разрешаются через evidence_paths.jsonl, source_factual_evidence.jsonl, entity_profiles.jsonl '
        'и неизменённые исходные артефакты. D использует только сохранённые имена/контексты, типы и локальные окрестности; найдена одна exploratory пара. '
        'Узкий lexical поиск не покрывает парафразы и далёкие дисциплины.', '',
        'Ограничение отбора существенно: из 554 C-пар основной проверке соответствуют только 11 пар с необщими endpoints и '
        'хотя бы одним необщим промежуточным мостом; ещё шесть включены как generic-path controls. '
        'GENERIC — явный список неоднозначных названий, включая genes/genetics, thickness и grain; это не утверждение, '
        'что генетические понятия, толщина или зерно бесполезны. Скрининг отдельных названий способен пропустить полезные отношения, '
        'поэтому 36 пар не представляют весь пул и отсутствие подтверждений не опровергает общую исследовательскую гипотезу. '
        'Названия вроде demand line и Limonene, прошедшие screening, также не гарантируют качества. В исходном KG ничего не удалено.', '',
        'Использован исходный локальный knowledgator/gliner-relex-base-v1.0 без обучения и NER. '
        'Fixed-span adapter scripts/build_relex_graphs.py не изменён: два exact spans → две направленные пары × 20 relation labels. '
        'CPU, 2 threads, seed 0, deterministic algorithms, token limit 512, offline model loading. '
        'Модель получает два исходных фрагмента с документными границами и один путь, если он есть; в ней не появляется отдельный механизм логического доказательства. '
        'Маркер NOT AN ENTAILED CLAIM также не гарантирует соблюдения запрета: это вход для классификатора, а не инструкция reasoning-модели. '
        'Тексты могут обрезаться по символьному радиусу; полные оригинальные контексты сохранены для аудита. '
        'ReLEx принимает формат технически корректно, но acceptance входа не означает научную валидность предсказаний.', '',
        'Smoke/pilot и содержательный gate — pilot_validation.json и pilot_review.md. '
        'Каждая завершённая пара checkpoint/fsync сохранена, включая пустые relations; повторный запуск пропускает её только при совпадении signature. '
        'Сбой последней незавершённой строки восстанавливается с диагностикой, законченные повреждённые строки отклоняются. '
        'Предварительный пилот раннего отбора сохранён отдельно, не смешивается с итоговым результатом и не участвует в таблице порогов.', '',
        'Учёт запусков: предварительный пилот 16 пар + один exact-repeat smoke; итоговый пилот 16 пар + один exact-repeat; '
        'полный запуск обработал оставшиеся 20 пар, переиспользовав 16 checkpoint-результатов итогового пилота. '
        'Итого 54 forward calls с учётом предварительного пилота и повторов smoke, 36 уникальных итоговых пар; '
        'при анализе порогов forward calls = 0.', '',
        '## Все 12 порогов', '',
        'Фильтрация использует исходный float confidence >= t без округления. Адаптер сохраняет только scores > 0.35. '
        'Повторного inference для порогов нет. Изменения WCC — только структурный эффект экспериментальных кандидатов, не новые доказанные факты. '
        'Документы и MENTIONS не входят в WCC; входят 811 сущностей и 32 концепта. Baseline: 223 WCC, крупнейшая 369 вершин / 17 документов.', '',
        '| t | Предсказания / triples | Positive pairs | Docs | С путём / без | Межкомпонентные | WCC | Крупнейшая / docs | Аудит S/H/R |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['threshold']:.2f} | {r['predictions']} / {r['unique_directed_triples']} | {r['positive_candidate_pairs']} | {r['documents']} | "
            f"{r['with_evidence_paths']} / {r['without_evidence_paths']} | {r['cross_baseline_components']} | {r['virtual_wcc']} | "
            f"{r['virtual_largest_component']} / {r['virtual_largest_component_documents']} | {r['reviewed_supported']}/{r['reviewed_hypothesis']}/{r['reviewed_rejected']} |")
    lines += ['', 'Полные распределения по всем 20 типам, сохранность baseline и доли H/R в проверенной выборке — threshold_comparison.csv. '
        'Пустая доля означает отсутствие проверенных предсказаний при этом пороге, а не нулевой риск ошибки.', '',
        '## Содержательный аудит', '',
        'quality_review.csv содержит конкретные directed triples, confidence, классификацию, аргументацию, проверенные source contexts и evidence IDs. '
        'Проверены все девять итоговых предсказаний, а также пять отрицательных пар (negative_pair_audit.csv), включая две независимые patients-пары. '
        'Это целевая выборка: высокий confidence, разные relation types, пути/отсутствие пути, общие слова, причинность и направление. '
        'Её доля ошибок не является несмещённой оценкой precision всего корпуса. HYPOTHESIS обозначает также REVIEW: '
        'недостаточно данных для подтверждения или уверенного отрицания; не все такие строки представляют ценную научную гипотезу.', '',
        'Отдельная общность концепта Patients не переносит результаты между независимыми ESRD/TKA, CRSwNP, GDD/PK, '
        'онкологическими RT и rHuEPO популяциями. Аналогично results обозначает разные локальные итоги. '
        'Пути через эти слова полезны как диагностические отрицательные контроли, но не как свидетельство общего референта.', '',
        'Конкретные ошибки: results лунного реголита → HAS_RESULT → results air-rail и конвергенции; максимальный ошибочный score 0.829874575138092. '
        'rHuEPO → PRODUCES/MEASURES → mammographic density через cancer; military power supply → HAS_PART → demand line '
        'через омонимию электрической линии и экономической supply line; Limonene → IS_A → WSN через общий energy consumption. '
        'Последний пример также показывает, что исходный локальный факт может чрезмерно упростить процесс: высокая энергоёмкость относится к химическому синтезу limonene, '
        'а не к отождествлению вещества и сети. Существующие factual рёбра не редактировались.', '',
        '## Сравнение с предыдущим экспериментом', '', '```json', json.dumps(comparison, ensure_ascii=False, indent=2), '```', '',
        'Предыдущие 40 предсказаний пересмотрены по сохранённым исходным фрагментам (previous_quality_review.csv). '
        'Все 40 старых и все 9 новых предсказаний отклонены в данном контекстном аудите; HAS_RESULT сократился с 32/40 до 4/9, '
        'но ошибки остались и среди четырёх предсказаний с путями. Обе выборки затрагивают пять дисциплин, новых подтверждённых научных relation types нет. '
        'Корпуса и candidate selection различаются, формат контекста тоже: это не парная оценка прироста precision. '
        'Новые relation labels и меньшее число generic-предсказаний сами по себе не доказывают улучшения. '
        'Evidence paths повышают трассируемость и делают видимым запрещённый перенос через абстракцию; '
        'подтверждение научного вывода требует совместимых референтов, условий и семантики, а не наличия пути.', '',
        '## Ответы и решение перед 2F.4D', '',
        f"1. В проверенной выборке получено {q.get('SUPPORTED_INFERENCE', 0)} поддержанных междокументных выводов. "
        'Остальные предсказания нельзя превращать в научные факты; непроверенные строки остаются UNVERIFIED_PREDICTION.', '',
        '2. Evidence paths помогли проследить происхождение кандидатов и обнаруживать ошибки, но сами не обеспечили надёжный междокументный reasoning. '
        'Без контролируемого сопоставимого эксперимента нельзя приписать им количественный прирост качества.', '',
        '3. Таблица показывает реальную зависимость объёма и структуры от confidence. Увеличение порога сокращает выход, '
        'но не является проверкой научной причинности, направления и совместимости когорт.', '',
        '4. Для 2F.4D можно использовать discovery/provenance, отрицательные результаты и аудит как исследовательские артефакты. '
        'Автоматический импорт отношений не рекомендован: сначала отдельная проверка конкретных SUPPORTED_INFERENCE, '
        'если такие появятся. HYPOTHESIS и UNVERIFIED_PREDICTION должны оставаться гипотезами; REJECTED исключаются из претендентов на факты.', '',
        'Надёжно назначить порог для импорта по этим данным нельзя. 0.35 сохраняется как исследовательский cutoff для воспроизводимости; '
        'выбирать 0.70/0.90 только ради меньшего числа рёбер научно необоснованно. Все исходные графовые рёбра сохранены независимо от этих выводов.', '',
        '## Воспроизводимость', '',
        'Запуски: scripts/analyze_cross_document_re_2f4.py --stage discover; --stage pilot; '
        'после содержательного pilot gate --stage infer; scripts/report_cross_document_re_2f4.py. '
        'Для discover/report требуются существующие NEO4J_* credentials, используется READ_ACCESS. '
        'Зафиксированы model/adapter/runner SHA, версии пакетов, source SHA и signature checkpoint. '
        'Повторный inference не нужен для пересчёта таблицы. JSON provenance и offsets — исходные, не восстановленные моделью.', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
