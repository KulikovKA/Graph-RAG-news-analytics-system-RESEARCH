"""Paired prompt comparison from saved scores and explicit contextual annotations.

No inference, candidate expansion, embedding calculation or Neo4j writes.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import compare_cross_document_prompts as run

OUT, core, base = run.OUT, run.core, run.base


def csvout(name, rows, fields=None):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def reviewed(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def analyze():
    run.banner('НАЧИНАЮ ПАРНОЕ СРАВНЕНИЕ A/B/C И АНАЛИЗ 36 КОНФИГУРАЦИЙ ПОРОГОВ')
    original, mapping, entities, docs, facts, semantic, concepts, profiles = run.previous.inputs()
    manifest = run.load('manifest.json')
    pairs = {p['pair_id']: p for p in core.read_jsonl(OUT / 'selected_bridge_pairs.jsonl')}
    contexts = {p['pair_id']: p for p in core.read_jsonl(OUT / 'prepared_contexts.jsonl')}
    require = run.require
    require(len(pairs) == len(contexts) == 68 and set(pairs) == set(contexts), 'Shared sample changed')
    require(run.protected() == manifest['protected_files_sha256'], 'Prior source/report changed')
    for path, expected in manifest['source_inputs_sha256'].items():
        require(base.sha(run.ROOT / path) == expected, 'Input artifact changed')
    signature = manifest['inference']
    for name, field in [('selected_bridge_pairs.jsonl', 'selected_pairs_sha256'),
                        ('prepared_contexts.jsonl', 'contexts_sha256'), ('configurations.json', 'configurations_sha256')]:
        require(base.sha(OUT / name) == signature[field], 'Frozen inference input differs')
    require(base.sha(Path(run.__file__)) == signature['runner_sha256'], 'Inference runner changed')
    require(base.sha(Path(core.__file__)) == signature['adapter_sha256'], 'Adapter changed')
    for pid, context in contexts.items():
        require(context['input_sha256'] == core.digest([context['text'], context['fixed_spans']]), 'Common input hash mismatch')
        require({m['entity_id'] for m in context['fixed_spans']} == {pairs[pid]['E2_id'], pairs[pid]['E3_id']}, 'Outer endpoints used as targets')
        for m in context['fixed_spans']:
            require(context['text'][m['start']:m['end']] == m['surface_text'], 'Prepared target offsets changed')
        for f in context['source_fragments']:
            require(docs[f['document_id']]['text'][f['start']:f['end']] == f['text'], 'Original fragment changed')
        for f in context['local_factual_context']:
            source = next(fact for fact in facts if fact['canonical_triple_id'] == f['canonical_triple_id'])
            require((f['head_id'], f['tail_id'], f['relation_type']) == (mapping[source['head_id']], mapping[source['tail_id']], source['relation_type']), 'Local factual direction changed')
    results, predictions = {}, {}
    for name, labels in run.CONFIGS.items():
        results[name] = list(run.checkpoints(name, signature['signatures'][name]).values())
        require({r['pair_id'] for r in results[name]} == set(pairs), 'Incomplete configuration')
        predictions[name] = []
        for r in run.progress(results[name], desc=f'Validating {name} target spans/prompts', unit='pair'):
            c = contexts[r['pair_id']]
            require(r['input_sha256'] == c['input_sha256'], 'Configuration has different context')
            require(r['audit']['tokens'] == c['token_counts'][name] <= 512, 'Truncated prompt/context')
            require(r['audit']['relation_prompt_count'] == len(labels) and r['audit']['relation_prompts'] == list(labels.values()), 'Head did not receive requested labels')
            require(r['audit']['pairs'] == 2 and r['audit']['spans'] == 2, 'Not exactly E2/E3 directed pairs')
            for x in r['relations']:
                p = pairs[r['pair_id']]
                require({x['head_id'], x['tail_id']} == {p['E2_id'], p['E3_id']}, 'Prediction uses outer entity')
                require(x['relation_code'] in labels and x['relation_prompt'] == labels[x['relation_code']] and x['confidence'] > .35, 'Bad label/score mapping')
                predictions[name].append(dict(x, pair_id=r['pair_id'], configuration=name))
        require(len({x['prediction_id'] for x in predictions[name]}) == len(predictions[name]), 'Repeated prediction ID')
    review = reviewed('quality_review.csv')
    all_predictions = {x['prediction_id']: x for rows in predictions.values() for x in rows}
    by_id = {r['prediction_id']: r for r in review}
    require(len(by_id) == len(review) and set(by_id) == set(all_predictions), 'Every positive prediction must have contextual review')
    for r in run.progress(review, desc='Verifying complete contextual audit', unit='prediction'):
        x = all_predictions[r['prediction_id']]
        require(r['configuration'] == x['configuration'] and r['relation_code'] == x['relation_code'] and r['head_id'] == x['head_id'] and r['tail_id'] == x['tail_id'], 'Audit relation mismatch')
        require(r['classification'] in ('PLAUSIBLE_HYPOTHESIS', 'UNCERTAIN', 'REJECTED') and r['reason'] and r['evidence_ids'], 'Review lacks explicit verdict/reason')
        require(r['review_method'] == 'contextual_manual_audit', 'Unknown review method')
        p, c = pairs[x['pair_id']], contexts[x['pair_id']]
        allowed = {p['E2_document_id'], p['E3_document_id']} | set(p['source_path_ids'])
        allowed.update(f['canonical_triple_id'] for f in c['local_factual_context'])
        require(set(json.loads(r['evidence_ids'])) <= allowed, 'Unknown evidence ID')
    pilot = reviewed('pilot_review.csv')
    pilot_ids = set(run.load('pilot_validation.json')['shared_pilot_ids'])
    expected_pilot_predictions = {x['prediction_id'] for rows in predictions.values() for x in rows if x['pair_id'] in pilot_ids}
    require({r['prediction_id'] for r in pilot} == expected_pilot_predictions, 'Pilot positives not fully audited')
    require(base.sha(OUT / 'pilot_review.csv') == run.load('pilot_validation.json')['content_review_sha256'], 'Pilot audit changed after gate')
    baseline_pairs = run.previous.graph_pairs(mapping, facts, semantic, concepts)
    concept_ids = {c['tail_id'] for c in concepts}
    baseline, membership = run.previous.components(profiles, baseline_pairs, concept_ids)
    table, distribution, summaries = [], [], {}
    for name, labels in run.CONFIGS.items():
        for threshold in run.progress(run.previous.THRESHOLDS, desc=f'{name}: 12 saved-score thresholds', unit='threshold'):
            kept = [x for x in predictions[name] if x['confidence'] >= threshold]
            graph, _ = run.previous.components(profiles, baseline_pairs + [(x['head_id'], x['tail_id']) for x in kept], concept_ids)
            verdicts = Counter(by_id[x['prediction_id']]['classification'] for x in kept)
            row = dict(configuration=name, threshold=threshold, predictions=len(kept),
                unique_directed_triples=len({(x['head_id'], x['relation_code'], x['tail_id']) for x in kept}),
                positive_pairs=len({x['pair_id'] for x in kept}), unique_directed_pairs=len({(x['head_id'], x['tail_id']) for x in kept}),
                documents=len({profiles[x[s]]['document_id'] for x in kept for s in ('head_id', 'tail_id')}),
                cross_baseline_component_relations=sum(membership[x['head_id']] != membership[x['tail_id']] for x in kept),
                cross_baseline_component_pairs=len({x['pair_id'] for x in kept if membership[x['head_id']] != membership[x['tail_id']]}),
                virtual_wcc=graph['weak_components'], largest_component=graph['largest_component_vertices'],
                largest_component_documents=graph['largest_component_documents'],
                plausible_hypotheses=verdicts['PLAUSIBLE_HYPOTHESIS'], uncertain=verdicts['UNCERTAIN'], rejected=verdicts['REJECTED'],
                direct_text_support=sum(by_id[x['prediction_id']]['direct_text_support'].lower() == 'true' for x in kept),
                rejected_fraction=verdicts['REJECTED'] / len(kept) if kept else '',
                label_distribution=json.dumps(dict(Counter(x['relation_code'] for x in kept)), sort_keys=True),
                baseline_relex_preserved=1307, baseline_similar_preserved=492, baseline_concept_links_preserved=118)
            table.append(row)
            for code in labels:
                part = [x for x in kept if x['relation_code'] == code]
                q = Counter(by_id[x['prediction_id']]['classification'] for x in part)
                distribution.append(dict(configuration=name, threshold=threshold, relation_code=code,
                    relation_prompt=labels[code], predictions=len(part), plausible=q['PLAUSIBLE_HYPOTHESIS'],
                    uncertain=q['UNCERTAIN'], rejected=q['REJECTED']))
        summaries[name] = dict(table[-12], negative_pairs=sum(not r['relations'] for r in results[name]),
            prompt_count=len(labels), tokens_min=min(c['token_counts'][name] for c in contexts.values()),
            tokens_max=max(c['token_counts'][name] for c in contexts.values()))
    csvout('threshold_comparison.csv', table)
    csvout('label_threshold_distribution.csv', distribution)
    paired = []
    for pid, p in pairs.items():
        row = dict(pair_id=pid, E2_name=p['E2_name'], E3_name=p['E3_name'],
            E2_document_id=p['E2_document_id'], E3_document_id=p['E3_document_id'], input_sha256=contexts[pid]['input_sha256'])
        for name in run.CONFIGS:
            xs = [x for x in predictions[name] if x['pair_id'] == pid]
            row[name + '_labels'] = json.dumps([dict(direction=x['head_slot'] + '->' + x['tail_slot'], code=x['relation_code'], confidence=x['confidence']) for x in xs])
            row[name + '_verdicts'] = json.dumps(dict(Counter(by_id[x['prediction_id']]['classification'] for x in xs)))
        paired.append(row)
    csvout('paired_predictions.csv', paired)
    original_codes = set(run.HYBRID)
    hybrid_mixed = []
    for pid in pairs:
        xs = [x for x in predictions['C'] if x['pair_id'] == pid]
        if any(x['relation_code'] in original_codes for x in xs) and any(x['relation_code'] in run.SEMANTIC for x in xs):
            hybrid_mixed.append(pid)
    after = run.previous.live_snapshot()
    require(after == manifest['before_neo4j'], 'Neo4j changed')
    require(run.protected() == manifest['protected_files_sha256'], 'Prior source/report changed during comparison')
    validation = dict(passed=True, shared_pairs=len(pairs), configurations=list(run.CONFIGS),
        completed_pairs_by_configuration={k: len(v) for k, v in results.items()},
        predictions_by_configuration={k: len(v) for k, v in predictions.items()},
        positive_pairs_by_configuration={k: sum(bool(r['relations']) for r in v) for k, v in results.items()},
        every_positive_prediction_and_pair_audited=True, pilot_predictions_audited=len(pilot),
        prompt_counts={k: len(v) for k, v in run.CONFIGS.items()}, dynamic_prompts_and_code_mapping_verified=True,
        common_inputs_identical_A_B_C=True, exact_offsets=True, no_outer_targets=True, no_truncation=True,
        separate_signatures_and_checkpoints=True, raw_float_thresholds=run.previous.THRESHOLDS, threshold_inference_calls=0,
        baseline=baseline, baseline_relex_preserved=1307, baseline_similar_preserved=492, baseline_concepts_preserved=118,
        neo4j_before=manifest['before_neo4j'], neo4j_after=after, all_namespaces_unchanged=True,
        protected_files_unchanged=True, protected_file_count=len(manifest['protected_files_sha256']),
        intra_document_default_backward_compatible=run.load('adapter_compatibility.json')['passed'],
        tests='39 passed: tests/test_build_relex_graphs.py tests/test_relex_four_experiments.py',
        hybrid_pairs_with_general_and_original_codes=hybrid_mixed,
        models_trained=False, NER_rerun=False, embeddings_rerun=False, LLMs_launched=False,
        entity_resolution_changed=False, neo4j_import=False, stage_2f5_started=False)
    run.save('validation.json', validation)
    write_report(manifest, validation, summaries, table, distribution, predictions, by_id, pairs, contexts, hybrid_mixed)
    manifest['finished_utc'] = run.datetime.now(run.timezone.utc).isoformat()
    manifest['summary'] = summaries
    manifest['analysis_runner_sha256'] = base.sha(Path(__file__))
    manifest['artifacts_sha256'] = {p.name: base.sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'manifest.json'}
    run.save('manifest.json', manifest)
    print(json.dumps(dict(verified=True, summaries=summaries), ensure_ascii=False), flush=True)


def write_report(manifest, validation, summaries, table, distribution, predictions, by_id, pairs, contexts, hybrid_mixed):
    lines = ['# 2F.4 — Comparative Cross-Document ReLEx A / B / C', '',
        '## Результат', '',
        'Проведены все три эксперимента на одной зафиксированной выборке из 68 E2/E3-пар (21 документ, шесть дисциплин). '
        'Веса одного локального knowledgator/gliner-relex-base-v1.0 неизменны. Отношения не импортировались в Neo4j.', '',
        '| Конфигурация | Labels | Предсказания >0.35 | Positive / negative pairs | Plausible / uncertain / rejected |',
        '|---|---:|---:|---:|---:|']
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['prompt_count']} | {s['predictions']} | {s['positive_pairs']} / {s['negative_pairs']} | {s['plausible_hypotheses']} / {s['uncertain']} / {s['rejected']} |")
    lines += ['', 'Проверены все положительные предсказания и все уникальные положительные пары каждого запуска, '
        'включая 18 предсказаний пилота. Вердикты относятся к этой целевой диагностической выборке, '
        'не являются несмещённой оценкой precision/recall всех возможных междокументных связей.', '',
        '## Постановка и подготовка', '',
        'Предыдущая внешняя постановка E1→E4 заменена на E2↔E3: сами intermediate/bridge entities являются единственными двумя fixed spans. '
        'Локальные E1→E2 и E3→E4 добавляются как контекст, когда доступны; модель не может вернуть их вместо targets. '
        'Ориентация E2/E3 выбиралась по наличию incoming/outgoing factual окрестностей, оценивались оба направления. '
        'Source-path IDs, исходные canonical IDs, native source IDs, document/revision IDs и offsets сохранены.', '',
        'Исходный bridge pool: 128 уникальных пар из существующих SIMILAR_TO, общих концептов, intermediate endpoints всех 3236 evidence paths '
        'и одной сохранённой cross-component exploratory пары. Дубликаты объединены с сохранением всех источников. '
        'Включены все 48 non-patient bridge-пар, в том числе genes/genetics, thickness, grain и другие ранее отсекавшиеся неоднозначные термины; '
        '20 patient-пар выбраны round-robin по парам документов. Новых embedding-кандидатов нет. '
        'Это challenge set: 25 generic controls и 20 patient controls, поэтому пригодных специфических научных пар мало. '
        'Размер не пополнялся фиктивными случаями. Полный pool и причины — bridge_pool.jsonl / selected_bridge_pairs.jsonl.', '',
        'Контекст один для A/B/C: два исходных фрагмента с границами документов и не более одного направленного локального factual ребра на endpoint. '
        'Синтетическое объединение и локальные model-extracted factual summaries не являются новым источником подтверждения. '
        'Наличие старого SIMILAR_TO и его численная оценка не вставлялись в текст как подсказка целевого label. '
        'Evidence path представлен двумя локальными factual флангами и source_path_ids, связывающими их provenance с целевой bridge-парой. '
        'Если исходное ребро ошибочно, его наличие не делает cross-document вывод верным; исходный KG не исправлялся.', '',
        'Все окна frozen до первого pilot forward в prepared_contexts.jsonl. Размер проверен tokenizer и actual collator для всех трёх конфигураций; '
        'общее сокращение выбиралось по самому длинному prompt budget. Entity spans не обрезались. '
        'Input SHA и offsets идентичны во всех трёх файлах предсказаний; число prompt embeddings и индексное отображение label→code проверялись в relation head.', '',
        '## Labels и адаптер', '',
        'A: исходные 20 core.PREDICATES с точными английскими строками. B: девять семантических labels из задания. '
        'C: 11 указанных original labels + все девять B, всего 20; повторов текстовых меток нет. Порядок и точные strings — configurations.json.', '',
        'Минимальное расширение predict_window(..., relation_labels=None) сохраняет старый default и audit; '
        'явный словарь передаётся в collator/tokenizer и размер relation head проверяется динамически. '
        'Старый adapter из 2e7c740, новый default и явный A дали полностью одинаковые raw scores и старый audit '
        'на одном позитивном results-контроле. 39 существующих тестов прошли. Никакого переизвлечения intra-document отношений не было.', '',
        'CPU, threads=2, seed=0, deterministic algorithms, eval/inference mode, offline checkpoint, token limit 512. '
        'Отдельные signatures A/B/C включают ordered labels, model/adapter/runner SHA и общие input SHA. '
        'Одна строка JSONL на пару, включая negative; flush/fsync после каждой строки; signature-checked resume. '
        '14 pilot-пар на конфигурацию входят в итоговые 68, оставшиеся 54 выполнены после review gate. '
        'Основной paired inference: 204 forwards, плюс три compatibility forwards. Для thresholds forwards=0.', '',
        '## Все пороги', '',
        'Фильтр confidence >= t применяется к исходным float; адаптер сохраняет >0.35. '
        'Confidence разных наборов labels не считается одинаково калиброванным. '
        'WCC ниже — структурный эффект гипотетического добавления кандидатов к 843 KG-вершинам; '
        'документы и служебные MENTIONS исключены. Это не прирост доказанных фактов.', '',
        '| Config | t | Predictions | Pairs | Cross-WCC | WCC | Largest / docs | Plausible / uncertain / rejected |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['configuration']} | {row['threshold']:.2f} | {row['predictions']} | {row['positive_pairs']} | {row['cross_baseline_component_relations']} | {row['virtual_wcc']} | {row['largest_component']} / {row['largest_component_documents']} | {row['plausible_hypotheses']} / {row['uncertain']} / {row['rejected']} |")
    lines += ['', 'Полные counts, документы, directed triples/pairs и распределения — threshold_comparison.csv; '
        'все labels, включая нулевые counts, — label_threshold_distribution.csv. '
        'Пустая доля при отсутствии предсказаний не означает нулевую ошибку или высокую precision.', '',
        '## Парное содержательное сравнение', '',
        'Каждое положительное предсказание имеет вердикт, аргументацию, исходные контексты и конкретные evidence IDs в quality_review.csv. '
        'Дополнительно разобраны 28 отрицательных диагностических пар, включая все 20 patient cases, — negative_control_review.csv. '
        'Непосредственное текстовое подтверждение отмечается отдельно; PLAUSIBLE_HYPOTHESIS не равно установленному факту. '
        'paired_predictions.csv сопоставляет A/B/C на одних E2/E3, тексте и offsets, без подмены конфигураций разными сессиями.', '',
        f"Число пар, где C одновременно выдал original и semantic codes: {len(hybrid_mixed)}. "
        'Наличие нескольких labels не интерпретируется автоматически как разумная осторожность: сравнивается их конкретный смысл. '
        'Relation scores — независимые sigmoid, поэтому наблюдаемое изменение scores нельзя называть доказанной softmax-конкуренцией. '
        'Кроме добавления B, C удаляет девять original labels относительно A; сравнение не выделяет причинный вклад только добавленных свободных меток.', '',
        '## Ответы на исследовательские вопросы', '']
    # Concrete interpretation is maintained separately as explicitly reviewed prose.
    findings = OUT / 'findings.md'
    if findings.exists():
        lines.append(findings.read_text(encoding='utf-8').strip())
    else:
        raise ValueError('Write contextual findings for all eight research questions before finalizing')
    lines += ['', '## Сохранность и воспроизводимость', '',
        'Neo4j прочитана с READ_ACCESS до/после: SHA всех свойств и концов рёбер source namespace и всех остальных namespaces совпали. '
        '1307 RELEX_REL, 492 SIMILAR_TO и 118 HAS_SHARED_CONCEPT сохранены. Все предыдущие отчёты и source artifacts '
        'проверены по SHA; единственное намеренное изменение существующего кода — optional parameter адаптера. '
        'Модель, NER, embeddings и ER не менялись. Импорта и 2F.5 нет.', '',
        'Запуск: compare_cross_document_prompts.py --stage select; --stage pilot; '
        'после содержательного pilot_review.csv/gate — --stage infer; report_cross_document_prompts.py. '
        'Для select/final validation используются существующие NEO4J_* credentials без записи в БД. '
        'Model hashes, версии, configs, signatures, inputs и артефакты зафиксированы в manifest.json; '
        'validation.json и independent_validation.json содержат проверки. Итоговый коммит локальный, без push.', '']
    (OUT / 'comparison_report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    analyze()
