"""Saved-score review and threshold analysis for ontology experiment, no inference."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import experiment_cross_document_ontology as run

OUT,core=run.OUT,run.core


def csvout(name,rows,fields=None):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def assessment(category,reason,scope,conditions='',direct=False,informative=False,
               type_correct='False',direction_correct='not_established',referents='False'):
    return dict(classification=category,reason=reason,abstraction_scope=scope,
        additional_conditions=conditions,direct_text_support=direct,scientifically_informative=informative,
        relation_type_correct=type_correct,direction_correct=direction_correct,referents_compatible=referents)


def supported(reason,scope='source-supported local concept relation',direct=True):
    return assessment('SUPPORTED_CONCEPT_RELATION',reason,scope,direct=direct,informative=True,
        type_correct='True',direction_correct='True',referents='True')


def judge(pid,head,tail,code):
    """Explicit source-read annotations, not labels copied from P/N/W status."""
    direction=head+'->'+tail
    if pid=='N07' and code=='USES_SAME_METHOD_AS':
        return assessment('REJECTED',
            'Protein computational design/HLA-hotspot screening и yeast transcriptomics/tolerance/evolutionary engineering — разные методы. Общее study не означает использование одного метода.',
            'two distinct research studies',direction_correct='True (symmetric predicate)',referents='False')
    if pid=='N07' and direction=='E2->E3' and code=='USES':
        return assessment('REJECTED','Нет данных, что rHuEPO study использует независимое limonene study; два исходных контекста описывают собственные методы.',
            'independent research studies',direction_correct='False',referents='False')
    if pid=='N05' and code in ('HAS_RESULT','PRODUCES'):
        return assessment('REJECTED',
            'E2 results относятся к fare optimization/cooperation авиации и HSR, E3 results — к radio observations Луны/regolith. Это два локальных результата: ни один не является результатом или продуктом другого.',
            'two source-specific empirical result referents',direction_correct='False in either direction',referents='False')
    if pid=='N06' and code=='HAS_RESULT':
        return assessment('REJECTED',
            'E2 results относятся к mammographic density/risk regression, E3 RESULTS — к IOP и corneal graft survival. Заголовок/анафора results не превращает один набор исходов в результат другого.',
            'two unrelated clinical research result referents',direction_correct='False in either direction',referents='False')
    # Sanity relations were checked against original text before forwards.
    exact={
        'S01':{'IS_A','SUBTYPE_OF'},'S02':{'HAS_ATTRIBUTE','HAS_PROPERTY'},
        'S03':{'HAS_ATTRIBUTE','HAS_PROPERTY'},'S04':{'DERIVED_FROM','BUILDS_ON'},
        'S05':{'APPLIED_TO'},'S06':{'USED_FOR'}}
    if pid in exact and direction=='E2->E3' and code in exact[pid]:
        reasons={
            'S01':'Первое предложение явно определяет recombinant human erythropoietin как biopharmaceutical drug; class-level inclusion, не identity.',
            'S02':'Источник характеризует fiber-reinforced composites через favorable strength-to-weight ratio; свойство носителя, не измерительное действие.',
            'S03':'Energy consumption of WSN прямо названо и измерено; WSN является носителем характеристики.',
            'S04':'Источник говорит GPU parallel algorithm based on optimized blockwise non-local means: derivation/method construction, не два независимых study.',
            'S05':'Multilayers investigated by XMCD прямо перечислено среди методов; направление method -> studied object соответствует actual application.',
            'S06':'Предлагается use of GPUs для вычислительно дорогих шагов e-health/real-time diagnosis. USED_FOR передаёт назначение, не автономный clinical performance.'}
        return supported(reasons[pid])
    if pid in ('S02','S06') and direction=='E2->E3' and code in ('MEASURES','PERFORMS','IS_A'):
        reasons={('S02','MEASURES'):'Композит имеет strength-to-weight ratio, но не является измерительным методом или прибором.',
                 ('S02','IS_A'):'Fiber-reinforced composite не является подтипом strength-to-weight ratio; перепутаны носитель и свойство.',
                 ('S06','PERFORMS'):'GPU поддерживает вычислительные шаги диагностики, но источник не утверждает, что GPU самостоятельно выполняет diagnosis как субъект.'}
        if (pid,code) in reasons:return assessment('REJECTED',reasons[pid,code],'local source roles',referents='False')
    if pid=='S04' and direction=='E2->E3' and code in ('DEPENDS_ON','COMPLEMENTS','USES_SAME_METHOD_AS'):
        if code=='USES_SAME_METHOD_AS':
            return assessment('REJECTED','Алгоритм построен на OB-NLM; сравнение алгоритма и его метода как двух исследований, использующих один метод, не соответствует ролям.',
                'implementation and underlying method',direction_correct='True (symmetric predicate)',referents='False')
        reason=('Based on подтверждает происхождение реализации, но не задаёт отдельную parameter/condition dependency или её ограничения.' if code=='DEPENDS_ON' else
                'GPU-реализация основана на методе; комплементарная функция двух самостоятельных ресурсов прямо не описана.')
        return assessment('UNCERTAIN',reason,'implementation and underlying method',
            'Нужно определить смысл технической зависимости/комплементарности.',type_correct='Unclear',direction_correct='Unclear',referents='True')
    raise ValueError(f'UNREVIEWED OUTPUT: {pid} {direction} {code}')


def items():
    return {r['pair_id']:r for filename in ['experiment_pairs.jsonl','sanity_pairs.jsonl'] for r in core.read_jsonl(OUT/filename)}


def review_rows(results,contexts):
    coverage={(r['configuration'],r['pair_id']):r for r in csv.DictReader((OUT/'ontology_coverage.csv').open(encoding='utf-8',newline=''))}
    rows=[]
    for result in results:
        c=contexts[result['pair_id']]
        for p in result['relations']:
            a=judge(c['pair_id'],p['head_slot'],p['tail_slot'],p['relation_code'])
            cv=coverage.get((result['configuration'],c['pair_id']))
            direction=p['head_slot']+'_TO_'+p['tail_slot']
            hit=bool(cv and cv['coverage']=='EXACT' and cv['relation_code']==p['relation_code'] and cv['expected_direction']==direction
                     and a['classification'] in ['SUPPORTED_CONCEPT_RELATION','PLAUSIBLE_HYPOTHESIS'])
            hp,tp=[c['endpoints'][s] for s in [p['head_slot'],p['tail_slot']]]
            rows.append(dict(configuration=result['configuration'],pool=c['pool'],pair_id=c['pair_id'],
                prediction_id=p['prediction_id'],head_id=p['head_id'],head_name=hp['name'],tail_id=p['tail_id'],tail_name=tp['name'],
                head_slot=p['head_slot'],tail_slot=p['tail_slot'],relation_code=p['relation_code'],relation_prompt=p['relation_prompt'],
                confidence=p['confidence'],**a,exact_candidate_hit=hit,evidence_ids=json.dumps(c['evidence_ids']),
                head_context=hp['source_context']['text'],tail_context=tp['source_context']['text'],
                source_revision_ids=json.dumps([f['revision_id'] for f in c['source_fragments']]),
                review_method='explicit_single_reviewer_source_context_annotation'))
    return rows


def results():
    return [r for k in run.configs() for r in core.read_jsonl(OUT/f'predictions_{k}.jsonl')]


def pilot():
    manifest=run.checked_manifest();contexts=items();main=results()
    run.require(all({r['pair_id'] for r in main if r['configuration']==k}==set(run.PILOT) for k in run.configs()),'Incomplete pilot')
    sanity=core.read_jsonl(OUT/'sanity_predictions.jsonl')
    run.require(len(sanity)==24,'Incomplete sanity set')
    rows=review_rows(main+sanity,contexts)
    csvout('pilot_review.csv',rows)
    validation=run.load('validation.json')
    validation['pilot']=dict(technical_passed=validation['pilot_technical_passed'],content_reviewed=True,
        reviewed_predictions=len(rows),main_positive_predictions=sum(r['pool']=='main' for r in rows),
        pilot_ids=run.PILOT,signatures=manifest['signatures'],content_review_sha256=run.prior.base.sha(OUT/'pilot_review.csv'),
        continue_reason='No technical errors or changed protected data; low/zero output is not a stopping condition.')
    run.save('validation.json',validation)
    print(json.dumps(validation['pilot'],ensure_ascii=False))


def analyze():
    run.banner('НАЧИНАЮ АНАЛИЗ ПОРОГОВ 0.35–0.90 ПО СОХРАНЁННЫМ RAW SCORES')
    manifest=run.checked_manifest();contexts=items();labels=run.configs()
    checkpoint_hashes={n:run.prior.base.sha(run.prior.old.MODEL_PATH/n)
                       for n in manifest['common_inference']['checkpoint_sha256']}
    run.require(checkpoint_hashes==manifest['common_inference']['checkpoint_sha256'],'Checkpoint files changed after inference')
    main=results();sanity=core.read_jsonl(OUT/'sanity_predictions.jsonl')
    require=run.require
    for name in labels:
        require({r['pair_id'] for r in main if r['configuration']==name}=={k for k in contexts if k.startswith(('P','W','N'))},'Incomplete main results')
    all_results=main+sanity
    require(len(main)==112 and len(sanity)==24,'Incorrect completed sample size')
    require(len({(r['configuration'],r['pair_id']) for r in all_results})==136,'Duplicate results')
    for r in run.progress(all_results,desc='Checking saved raw-score mapping and exact targets',unit='result'):
        c=contexts[r['pair_id']];rels=labels[r['configuration']]
        require(r['input_sha256']==c['input_sha256']==core.digest([c['text'],c['fixed_spans']]),'Context hash changed')
        require(r['inference_signature']==manifest['signatures'][r['configuration']],'Inference signature differs')
        require(r['audit']['tokens']==c['token_counts'][r['configuration']]<=512,'Truncation')
        require(r['audit']['spans']==r['audit']['pairs']==2,'Wrong target pairs')
        require(r['audit']['relation_prompt_count']==len(rels) and r['audit']['relation_prompts']==list(rels.values()),'Dynamic head mapping')
        expected={(a,b,code) for a,b in [('E2','E3'),('E3','E2')] for code in rels}
        require({(p['head_slot'],p['tail_slot'],p['relation_code']) for p in r['raw_scores']}==expected,'Missing raw scores')
        require(len(r['raw_scores'])==len(expected),'Duplicate raw scores')
        for p in r['raw_scores']:
            require(p['head_id']==c[p['head_slot']+'_id'] and p['tail_id']==c[p['tail_slot']+'_id'],'Wrong endpoints')
            require(p['relation_prompt']==rels[p['relation_code']],'Label text mismatch')
        require(r['relations']==[p for p in r['raw_scores'] if p['confidence']>.35],'Positive inventory mismatch')
        require(not any(p['confidence']==.35 for p in r['raw_scores']),'Boundary score needs additional review')
    docs,profiles,facts=run.data()
    for c in contexts.values():
        for sp in c['fixed_spans']:
            require(c['text'][sp['start']:sp['end']]==sp['surface_text'],'Saved span changed')
        for f in c['source_fragments']:
            require(docs[f['document_id']]['text'][f['start']:f['end']]==f['text'],'Saved source changed')
            require(docs[f['document_id']]['revision_id']==f['revision_id'],'Saved revision changed')
        require(all(word not in c['text'] for word in ['STRONG_CANDIDATE','PLAUSIBLE_CANDIDATE','REJECTED','[LOCAL RELATION]']),'Answer leakage')
    rows=review_rows(all_results,contexts)
    csvout('quality_review.csv',rows)
    reviewed={r['prediction_id']:r for r in rows}
    require(len(reviewed)==sum(len(r['relations']) for r in all_results),'Review coverage incomplete')
    pilot_rows=list(csv.DictReader((OUT/'pilot_review.csv').open(encoding='utf-8',newline='')))
    for row in pilot_rows:
        now=reviewed[row['prediction_id']]
        require(all(str(now[k])==v for k,v in row.items()),'Pilot judgement changed')
    sem=core.read_jsonl(run.previous.SOURCE/'preview_semantic_links.jsonl')
    concepts=core.read_jsonl(run.previous.SOURCE/'preview_concept_links.jsonl')
    edges=[(f['head_id'],f['tail_id']) for f in facts.values()]+[(r['head_id'],r['tail_id']) for r in sem+concepts]
    concept_ids={c['tail_id'] for c in concepts}
    baseline,membership=run.previous.components(profiles,edges,concept_ids)
    require(baseline==dict(weak_components=223,largest_component_vertices=369,largest_component_documents=17),'Baseline mismatch')
    coverage=list(csv.DictReader((OUT/'ontology_coverage.csv').open(encoding='utf-8',newline='')))
    table=[]
    accepted_categories={'SUPPORTED_CONCEPT_RELATION','PLAUSIBLE_HYPOTHESIS'}
    for name,rels in labels.items():
        preds=[dict(p,pair_id=r['pair_id']) for r in main if r['configuration']==name for p in r['raw_scores']]
        exact_available=sum(r['configuration']==name and r['coverage']=='EXACT' for r in coverage)
        for t in run.progress(run.previous.THRESHOLDS,desc=f'{name} saved-score thresholds',unit='threshold'):
            kept=[p for p in preds if p['confidence']>=t]
            accepted=[p for p in kept if reviewed[p['prediction_id']]['classification'] in accepted_categories and reviewed[p['prediction_id']]['scientifically_informative']]
            supported=[p for p in accepted if reviewed[p['prediction_id']]['classification']=='SUPPORTED_CONCEPT_RELATION']
            virtuals=[]
            for subset in [kept,supported,accepted]:
                graph,_=run.previous.components(profiles,edges+[(p['head_id'],p['tail_id']) for p in subset],concept_ids)
                virtuals.append(graph)
            q=Counter(reviewed[p['prediction_id']]['classification'] for p in kept)
            hits={p['pair_id'] for p in kept if reviewed[p['prediction_id']]['exact_candidate_hit']}
            controls=[p for p in kept if p['pair_id'].startswith('N')]
            distribution={code:dict(predictions=0,supported=0,plausible=0,uncertain=0,rejected=0) for code in rels}
            category_key={'SUPPORTED_CONCEPT_RELATION':'supported','PLAUSIBLE_HYPOTHESIS':'plausible','UNCERTAIN':'uncertain','REJECTED':'rejected'}
            for p in kept:
                d=distribution[p['relation_code']];d['predictions']+=1;d[category_key[reviewed[p['prediction_id']]['classification']]]+=1
            row=dict(configuration=name,threshold=t,predictions=len(kept),unique_pairs=len({p['pair_id'] for p in kept}),
                supported_concept_relations=q['SUPPORTED_CONCEPT_RELATION'],plausible_hypotheses=q['PLAUSIBLE_HYPOTHESIS'],
                uncertain=q['UNCERTAIN'],rejected=q['REJECTED'],scientifically_informative=len(accepted),
                candidate_pairs_with_any_output=len({p['pair_id'] for p in kept if p['pair_id'].startswith('P')}),
                exact_candidate_hits=len(hits),exact_candidate_hit_ids=json.dumps(sorted(hits)),exact_ontology_covered_candidates=exact_available,
                coverage_conditioned_hit_fraction=len(hits)/exact_available if exact_available else '',
                negative_control_predictions=len(controls),negative_control_wrong_predictions=sum(reviewed[p['prediction_id']]['classification']=='REJECTED' for p in controls),
                negative_control_wrong_pairs=len({p['pair_id'] for p in controls if reviewed[p['prediction_id']]['classification']=='REJECTED'}),
                weak_case_predictions=sum(p['pair_id'].startswith('W') for p in kept),
                raw_cross_baseline_pairs=len({p['pair_id'] for p in kept if membership[p['head_id']]!=membership[p['tail_id']]}),
                accepted_cross_baseline_pairs=len({p['pair_id'] for p in accepted if membership[p['head_id']]!=membership[p['tail_id']]}),
                label_distribution=json.dumps(distribution,sort_keys=True))
            for prefix,graph in zip(['virtual_raw','virtual_supported','virtual_accepted_including_hypotheses'],virtuals):
                for field,value in graph.items():row[prefix+'_'+field]=value
            table.append(row)
    csvout('threshold_comparison.csv',table)
    validation=run.load('validation.json')
    validation.update(passed=True,main_results=112,sanity_results=24,
        configurations=list(labels),main_pairs_per_configuration=28,all_raw_scores_saved=True,
        every_positive_prediction_audited=True,reviewed_main_predictions=sum(r['pool']=='main' for r in rows),
        reviewed_sanity_predictions=sum(r['pool']=='sanity' for r in rows),
        identical_inputs_and_exact_fixed_spans=True,no_truncation=True,
        checkpoint_files_unchanged_after_inference=True,
        protected_file_count=len(manifest['protected_files_sha256']),protected_files_unchanged=True,
        ontology_and_contexts_frozen=True,model_forwards=manifest['model_forwards'],
        threshold_forwards=0,baseline=baseline,thresholds=run.previous.THRESHOLDS,
        tests='42 passed: test_build_relex_graphs, test_relex_four_experiments, test_cross_document_ontology_experiment',
        resume_duplicate_and_foreign_signature_rejected=True,incomplete_tail_recovery_tested=True,
        neo4j_accessed=False,neo4j_import=False,NER_rerun=False,embeddings_rerun=False,other_LLMs_launched=False,
        entity_resolution_changed=False,stage_2f5_started=False)
    run.save('validation.json',validation)
    write_report(manifest,validation,table,rows,main,sanity,contexts,coverage)
    run.checked_manifest()
    manifest['analysis_runner_sha256']=run.prior.base.sha(Path(__file__))
    manifest['artifacts_sha256']={p.name:run.prior.base.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
    run.save('manifest.json',manifest)
    print(json.dumps(dict(passed=True,main_predictions={k:sum(len(r['relations']) for r in main if r['configuration']==k) for k in labels},
        model_forwards=manifest['model_forwards'],reviewed_predictions=len(rows),threshold_rows=len(table)),ensure_ascii=False),flush=True)


def write_report(manifest,validation,table,rows,main,sanity,contexts,coverage):
    summaries={r['configuration']:r for r in table if r['threshold']==.35}
    text=['# Stage 2F.4 — Cross-Document Ontology Experiment A/B/C/D','',
        'Дата: 2026-11-12. Source HEAD `97b8196`, ветка `design-v2-sequential-kg`. В этом опыте разработана одна новая онтология D, выполнен весь pilot/main/sanity protocol. E не создавалась.','',
        '## Результат','',
        '**Содержательной междокументной связи выше 0.35 не распознано ни одной конфигурацией.** D содержит 9 типизированных направленных отношений и повышает проектное точное ontology coverage шести P-смыслов с A=2, B=0, C=2 до D=6. Однако на P01–P06 у всех A/B/C/D нет ни одного выхода >0.35. Это development-set diagnostic, не независимый gold и не полный recall корпуса.','',
        '| Конфигурация | Labels | Предсказания >0.35 | Пары с выходом / 28 | Supported / plausible / uncertain / rejected | Exact P hits / covered | Ошибочные N-пары / 12 |',
        '|---|---:|---:|---:|---|---|---:|']
    for name,s in summaries.items():
        text.append(f"| {name} | {len(run.configs()[name])} | {s['predictions']} | {s['unique_pairs']} | {s['supported_concept_relations']} / {s['plausible_hypotheses']} / {s['uncertain']} / {s['rejected']} | {s['exact_candidate_hits']} / {s['exact_ontology_covered_candidates']} | {s['negative_control_wrong_pairs']} |")
    text += ['',
        'D не дала ложноположительных выходов на N/W, но полностью воздержалась и на P. Это уменьшение ложных срабатываний при нулевом candidate yield, а не доказанная более высокая precision: для пустого выхода precision не определена. Новых эмпирических фактов и новых потенциальных межкомпонентных мостов на заданных порогах нет. Никаких рёбер в Neo4j не добавлялось.','',
        '## Что именно разработано','',
        'В `cross_document_ontology.json` и `ontology_design.md` зафиксированы definition, head/tail roles, direction, positive/boundary examples и evidence requirements для девяти codes. В модель передавались только короткие prompts:', '',
        '| Code | Точный английский prompt |','|---|---|']
    for r in run.load('cross_document_ontology.json')['relations']:text.append(f"| {r['code']} | {r['label']} |")
    text += ['',
        'Различаются subtype и stage, component и participant, property и outcome, function и actual application, actual и potential applicability, derivation и causal influence. Все D-relations направленные. PART_OF является inverse старого HAS_PART. Симметричные untyped association/similarity, causal и автоматические dependency links из D исключены. Это проектное ограничение смысла, не автоматический postfilter выходов модели. Исходная intra-document ontology не менялась.','',
        'Дизайн опирается на разграничение hierarchy/association в [SKOS Reference](https://www.w3.org/TR/skos-reference/#semantic-relations), на process/participant/part distinctions в [OBO RO](https://oborel.github.io/obo-relations/process-relations/) и на научную типизацию relations/coreference в [SciERC/SciIE](https://aclanthology.org/D18-1360/). Наши class-level shortcuts не объявляются эквивалентными формальным RO properties. [Model card ReLEx](https://huggingface.co/knowledgator/gliner-relex-base-v1.0) описывает текстовые relation labels; определения в отчёте не добавлялись в model prompt.','',
        'D не состоит из шести отдельных answer labels: property, function, provenance, participation, structure, subtype и stage применимы к научным областям всего корпуса. Potential applicability охватывает методы, инструменты и вмешательства с явно указанными условиями. E без самостоятельного научного вопроса повторила бы broad semantic B и потому не запускалась.','',
        '## Постановка и контроль эксперимента','',
        'Основной набор: ровно 28 уникальных unordered pairs — P01–P06, N01–N12, W01–W10. Дубликатов нет, исходная идентичность записей и native/canonical provenance сохранены. Sanity set из 6 intra-document pairs отделён в `sanity_pairs.jsonl` и `sanity_predictions.jsonl`; он не входит в основную пороговую таблицу или виртуальную междокументную связность.','',
        'Модель одна: локальный `knowledgator/gliner-relex-base-v1.0`, CPU, 2 threads, seed=0, deterministic algorithms, eval/inference mode, offline checkpoint, token limit 512. A/B/C exact labels/order взяты из прежнего configurations.json без изменений, D frozen до первого forward. Контексты и fixed spans общие для всех четырёх конфигураций, выбран общий radius по самому длинному prompt. Реальные максимумы tokens: '+json.dumps(validation['max_tokens'])+'. Проверялись и tokenizer, и actual collator; relation head получил ожидаемые embeddings и ordered label mapping.','',
        'Во входе только точные source fragments и нейтральные `[DOCUMENT ...]` / `[END DOCUMENT ...]` boundaries. Исходные source contexts каждого endpoint и полные IDs лежат в metadata. Локальные 1307 model-extracted RELEX_REL доступны по refs, но summaries не вставлены, чтобы неверный локальный label не подсказывал cross-document ответ. Expected labels, статусы STRONG/PLAUSIBLE/REJECTED, внешние определения и оценки аудитора в model text не вставлялись. Новые исходные документы не создавались.','',
        'Для каждой directed pair сохранён полный вектор sigmoid scores по всем labels, включая значения <0.35 (`raw_scores`), и отдельный positive inventory >0.35 (`relations`). Это scores, не logits и не калиброванные вероятности научной корректности. Ровно 0.35 среди raw scores не встретилось; поэтому ручной inventory >0.35 полностью покрывает threshold >=0.35.','',
        'Пилот: P01/P03/P06/N01/N07/W04 во всех конфигурациях, затем sanity set. Все 30 положительных pilot/sanity выходов проверены до main; labels/contexts не менялись. Технических ошибок не было, нулевой выход D не использовался как stop gate. Main дополнил checkpoint только оставшимися 22 парами на конфигурацию. Итого '+str(manifest['model_forwards'])+' forwards: 112 main (включая 24 pilot), 24 sanity, 3 compatibility. Threshold forwards=0. JSONL fsync после каждой пары, negative rows сохранены, signature-checked resume и восстановление оборванного tail реализованы.','',
        'Совместимость: исторический adapter из `2e7c740`, текущий default и explicit A дали полностью идентичные raw scores на всех 40 directed/label scores P01, default audit также идентичен. Адаптер не изменялся. Выполнены 39 существующих тестов адаптера/экспериментов и 3 новых теста: неполный checkpoint tail, rejection чужой signature/duplicate, одинаковые поверхности mentions из разных документов. Все 42 прошли.','',
        '## Ontology coverage отдельно от результатов модели','',
        'Покрытие зафиксировано до inference в `ontology_coverage.csv`. EXACT — сохранён конкретный смысл и направление; APPROXIMATE — существенный аспект потерян. Здесь NONE не понадобился: у B всегда есть broad association, но она не является точным выражением stage/participant/conditional application. APPROXIMATE не засчитывается exact hit.','',
        '| P-пара | A | B | C | D |','|---|---|---|---|---|']
    for pid in [f'P{i:02}' for i in range(1,7)]:
        cv=[next(r for r in coverage if r['configuration']==k and r['pair_id']==pid) for k in 'ABCD']
        text.append('| '+pid+' | '+' | '.join(r['coverage']+' / '+r['relation_code']+' / '+r['expected_direction'] for r in cv)+' |')
    text += ['',
        'P01 STAGE_OF сохраняет стадию; IS_A допускает более грубую классификацию. P03 HAS_PARTICIPANT не предполагает специальной agency. P04 SUBTYPE_OF/IS_A описывают specialization процесса, не одинаковые гены или события. P05 PART_OF требует **E3→E2** и component-type scope; волокна из разных работ не становятся тем же оборудованием. P02/P06 требуют modality и дополнительных клинических/изобразительных условий, не actual application. Все эти смыслы использовались в дизайне и потому не являются holdout gold.','',
        'Exact candidate coverage-conditioned hits: A 0/2, B denominator=0 (не 0% recall), C 0/2, D 0/6. Это счётчик заданных candidate meanings, не полный corpus recall. Все шесть P-пар дополнительно имеют zero any-label output во всех конфигурациях. Ошибочный approximate label не может заменить exact hit, а отсутствие exact label не штрафуется как model miss.','',
        '| Кандидат | Frozen D target / direction | Raw score D для ожидаемого смысла | Максимальный D score на паре |','|---|---|---:|---|']
    for r in main:
        if r['configuration']!='D' or not r['pair_id'].startswith('P'):continue
        cv=next(c for c in coverage if c['configuration']=='D' and c['pair_id']==r['pair_id'])
        target=next(p for p in r['raw_scores'] if p['relation_code']==cv['relation_code'] and p['head_slot']+'_TO_'+p['tail_slot']==cv['expected_direction'])
        top=max(r['raw_scores'],key=lambda p:p['confidence'])
        text.append(f"| {r['pair_id']} | {cv['relation_code']} / {cv['expected_direction']} | {target['confidence']:.8f} | {top['relation_code']} {top['head_slot']}→{top['tail_slot']} {top['confidence']:.8f} |")
    text += ['',
        'Показываются raw scores для диагностики, а не основание понизить порог постфактум. Например, более высокий score APPLIED_TO на P02 не является доказательством actual drug exposure. Малые scores concept/stage relations могут отражать отсутствие нужного утверждения в двух локальных текстах и/или трудность новых формулировок; причинно отделить эти факторы этот опыт не позволяет.','',
        '## Все положительные междокументные предсказания','',
        '| Config / pair | Head → code → tail | Score | Вердикт |','|---|---|---:|---|']
    for r in rows:
        if r['pool']=='main':text.append(f"| {r['configuration']} / {r['pair_id']} | {r['head_slot']} → {r['relation_code']} → {r['tail_slot']} | {r['confidence']:.8f} | {r['classification']} |")
    text += ['',
        'Все 11 main outputs отклонены по источникам, а не из-за принадлежности к N. N05 связывает тарифные results и lunar/regolith results как HAS_RESULT/PRODUCES. N06 смешивает mammographic-density outcomes и glaucoma/graft outcomes. N07 USES_SAME_METHOD_AS связывает protein computational design и yeast transcriptomics/evolutionary engineering; USES не имеет evidence использования одной работы другой. Независимые локальные results/study нельзя превращать в происхождение, причинность или method identity. Максимальный rejected score A достигает примерно 0.762.','',
        'На W01–W10 у всех конфигураций нулевой выход >0.35. Это сохранённое наблюдение, не доказательство невозможности любого концептуального отношения. N01 supply-line homonym и N02 grains/grain в данном контексте не дали ошибок, но их отсутствие не подтверждает освоение контекстной семантики вообще. Все N оцениваются по конкретному type/direction/scope; common-category или корректная modal hypothesis не отклонялись бы автоматически.','',
        '## Sanity set: модель реагирует на часть новых labels','',
        '| Config | Предсказания | Supported / plausible / uncertain / rejected | Source-checked ожидаемые смыслы, распознанные при >0.35 |','|---|---:|---|']
    expected_sanity={'S01':{'IS_A','SUBTYPE_OF'},'S02':{'HAS_ATTRIBUTE','HAS_PROPERTY'},'S03':{'HAS_ATTRIBUTE','HAS_PROPERTY'},
                     'S04':{'DERIVED_FROM','BUILDS_ON'},'S05':{'APPLIED_TO'},'S06':{'USED_FOR'}}
    for name in run.configs():
        rr=[r for r in rows if r['configuration']==name and r['pool']=='sanity'];q=Counter(r['classification'] for r in rr)
        hitids={r['pair_id'] for r in rr if r['classification']=='SUPPORTED_CONCEPT_RELATION' and r['relation_code'] in expected_sanity[r['pair_id']] and r['head_slot']=='E2'}
        text.append(f"| {name} | {len(rr)} | {q['SUPPORTED_CONCEPT_RELATION']} / {q['PLAUSIBLE_HYPOTHESIS']} / {q['UNCERTAIN']} / {q['REJECTED']} | {', '.join(sorted(hitids)) or 'нет'} |")
    text += ['',
        'D выдаёт ровно по одному ожидаемому отношению на всех S01–S06: subtype, property (два примера), derivation, actual application, intended use. Следовательно, динамические prompts работают, и полный нулевой cross-document выход D не является отсутствием любого отклика на D ontology. Это не проверка всех девяти labels: STAGE_OF, PART_OF, HAS_PARTICIPANT и POTENTIALLY_APPLICABLE_TO не имеют positive sanity в этом наборе.','',
        'Sanity также выявляет ошибки и неоднозначности: A даёт MEASURES между composite и strength-to-weight ratio, хотя материал является носителем свойства, а не прибором; C даёт IS_A на той же паре. Для GPU algorithm/OB-NLM DERIVED_FROM/BUILDS_ON согласуются с based-on construction; USES_SAME_METHOD_AS смешивает реализацию с методом и отклонён. DEPENDS_ON и COMPLEMENTS оставлены UNCERTAIN: текст подтверждает derivation, но не формализует отдельную parameter dependency или complementary function. Роль GPU в intended diagnosis workflow не превращает его в самостоятельного агента PERFORMS diagnosis.','',
        '## Все 12 порогов и виртуальная связность','',
        'Фильтр >=t применяется к сохранённым float scores; модель не запускается заново. Распределение **всех** labels, в том числе нулевых, и все счётчики P/N/W находятся в `threshold_comparison.csv`. Sanity исключён из таблицы ниже.','',
        '| Config | t | Predictions / pairs | Informative / rejected | Exact P hits | Raw WCC / largest / docs | Accepted cross-WCC pairs |','|---|---:|---|---|---:|---|---:|']
    for r in table:
        text.append(f"| {r['configuration']} | {r['threshold']:.2f} | {r['predictions']} / {r['unique_pairs']} | {r['scientifically_informative']} / {r['rejected']} | {r['exact_candidate_hits']} | {r['virtual_raw_weak_components']} / {r['virtual_raw_largest_component_vertices']} / {r['virtual_raw_largest_component_documents']} | {r['accepted_cross_baseline_pairs']} |")
    text += ['',
        'Baseline восстановлен offline из 811 contextual entities, 32 concepts, 1307 RELEX_REL, 492 SIMILAR_TO и 118 HAS_SHARED_CONCEPT. Document/MENTIONS исключены. Сохраняются 223 WCC, largest 369 vertices / 17 documents во всех 48 config/threshold combinations. Нет даже raw cross-WCC outputs. Отдельные virtual_raw, virtual_supported и virtual_accepted_including_hypotheses поля совпали с baseline. В общем случае raw structural merges не являлись бы научным улучшением; hypotheses и concept relations не становятся empirical KG facts.','',
        'A становится пустым при 0.80, B при 0.70, C при 0.55; D пуст на всех заданных порогах. Повышение порога здесь удаляет проверенные ошибки, но не восстанавливает P-покрытие и не выбирает рабочего победителя. Ошибка A остаётся даже при 0.75.','',
        '## Ответы на исследовательские вопросы','',
        '1. **Новая онтология:** D из девяти коротких labels, отделяющая taxonomy/stage, structure/participation, property/function и actual/potential applicability. Определения, роли, examples и основания frozen до inference.','',
        '2. **Отличие от A/B/C:** отсутствуют untyped relatedness и необоснованные causal shortcuts, добавлены точные stage/participant/modal смыслы. Меняется и состав, и формулировка labels; это не изолированная prompt-wording ablation.','',
        '3. **Более точная представимость:** стадия заболевания P01, клеточный участник P03 и modality P02/P06. P04/P05 A/C уже могли выражать точно при class/component-type scope.','',
        '4. **Распознана ли содержательная междокументная связь?** Нет при всех заданных порогах. Sanity-success относится к внутридокументным примерам и не подменяет main result.','',
        '5. **Улучшение D?** Улучшено проектное coverage и исчезли наблюдаемые false positives на этом наборе. Прирост meaningful extraction/coverage-conditioned candidate hits не показан; нулевой выход не доказывает высокую precision или superiority.','',
        '6. **Ошибки слабых пар:** main false positives остались на N05/N06/N07 у A/B/C; D не дала их. W cases все negative-output, но универсальный gold NO_RELATION из этого не следует.','',
        '7. **Сохраняющиеся классы ошибок:** lexical results/study shortcuts, неверные bearer/measurement roles и implementation/method conflation. У D главная наблюдаемая проблема main — abstention/missing candidate relations; его sanity показывает реакцию лишь на часть labels.','',
        '8. **Пригодность D для второго прохода:** как исследовательская схема аннотирования и разделения meaning/modality — пригодна; как подтверждённая рабочая ontology для извлечения этим checkpoint — пока не обоснована. Предсказания импортировать нельзя.','',
        '9. **Главное ограничение:** старый candidate discovery действительно пропускал P-смыслы, а A/B/C имели ontology gaps. В текущем диагностическом наборе человечески мотивированные кандидаты и D coverage уже есть, но выходов нет: непосредственно ограничивает связка модель + два локальных контекста + задача неявного междокументного вывода. Нельзя приписать всё только весам модели: отсутствует context/label-wording ablation, часть P требует внешнего знания, а четыре D labels не имеют отдельного positive sanity.','',
        '10. **Следующий шаг:** прекратить подбор labels на этих же шести парах как поиск рабочего решения. Предпочтительно разделить concept linking и evidence-grounded relation extraction: типизированные conceptual links проверять по определениям/таксономии, empirical relations требовать из источников с совместимыми условиями. До смены модели можно спроектировать один независимый context/scope diagnostic (например, те же значения в явном локальном утверждении против двух раздельных источников), с новым holdout и без post-hoc tuning. Для оценки научной генерализации нужен тематически более плотный независимо размеченный корпус. Эти варианты только предложены; новые LLM, модели или Stage 2F.5 не запускались.','',
        '## Воспроизводимость и ограничения','',
        '`manifest.json` содержит source commit, все 1183 protected tracked-file hashes, checkpoint/config/tokenizer hashes, точные ordered labels, runner/adapter hashes, frozen input hashes, signatures, runtime и accounting forwards. `validation.json` фиксирует technical preparation, compatibility, pilot gate, завершённость и итоговые проверки. Каждый prediction row имеет signature, input SHA, raw scores и head audit. `quality_review.csv` покрывает все 11 main и 26 sanity positive predictions; `pilot_review.csv` неизменён после gate.','',
        'Воспроизвести score analysis из сохранённых результатов:', '',
        '```powershell',
        '.\.venv\Scripts\python.exe scripts/report_cross_document_ontology.py --stage analyze',
        '```','',
        'Inference runner имеет этапы `prepare`, `pilot`, `infer`. Prepare отказывается перезаписывать уже frozen inputs. На завершённом наборе `infer` проверяет hashes/signatures и использует checkpoints; повторять модель для thresholds не требуется. Для независимого повторения можно скопировать frozen ontology/design/coverage/contexts/manifest/validation/pilot review в отдельный каталог, использовать тот же runner с OUT, направленным на копию, и отдельные пустые output checkpoints. Изменение runner source или labels требует новой experimental signature, а не смешения в текущий набор.','',
        'Все оценки сделаны одним рецензентом по оригинальным абстрактам. Санити и P-смыслы не независимы от известного корпуса/проектирования. Human adjudication использует контексты и явные дополнительные знания из прошлого аудита; эти знания не добавлены модели. Label scores между A/B/C/D не считаются одинаково калиброванными. Нельзя считать 0/6 полным recall или разность false-positive counts статистическим превосходством. Пустые и непустые результаты сохранены без подбора labels после pilot.','',
        'В сравнении со старым опытом 68 пар изменились и выборка, и source-only context recipe. Поэтому меньшие A/B/C counts здесь нельзя приписать одному улучшению discovery или отсутствию summaries. Внутри нового опыта контексты и вычислительные настройки одинаковы для всех конфигураций.','',
        'Модельные веса, NER 0.46, ER, все 1307 factual relations, 492 semantic links, 118 concept links, исходные документы и старые отчёты сохранены; byte hashes проверены до и после inference/analysis. База Neo4j не открывалась: отсутствие записи обеспечивается отсутствием graph calls, а актуальное состояние внешнего сервера отдельно не проверялось. Никаких новых empirical edges, inference сверх описанного, push или перехода к Stage 2F.5 в этом задании нет.','']
    (OUT/'comparison_report.md').write_text('\n'.join(text),encoding='utf-8',newline='\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pilot','analyze'],required=True)
    args=parser.parse_args()
    if args.stage=='pilot':pilot()
    else:analyze()
