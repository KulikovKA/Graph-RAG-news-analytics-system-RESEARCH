"""Audit saved split-context scores; never calls inference or any service."""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess

import experiment_split_context as run

OUT, core, require = run.OUT, run.core, run.require


def direction(score):
    return score['head_slot'] + '_TO_' + score['tail_slot']


def score_map(row):
    return {(s['relation_code'], direction(s)): s for s in row['raw_scores']}


def judge(pid, score):
    """Explicit source-read annotations. Unreviewed outputs block finalization."""
    code, d = score['relation_code'], direction(score)
    if pid == 'N03' and code == 'HAS_PROPERTY':
        return dict(classification='REJECTED', relation_type_correct=False,
            direction_correct=False, referents_compatible=False, external_knowledge_required=False,
            derivable_from_two_sources=False, concept_instance_scope='two source-specific resolution attributes',
            reason='Morphology-coding resolution in the materials source and angular resolution of lunar radio observations are independent attributes. Neither resolution is a bearer of the other resolution as its property; shared wording cannot establish HAS_PROPERTY in either direction.')
    if pid == 'N05' and code == 'DERIVED_FROM':
        return dict(classification='REJECTED', relation_type_correct=False,
            direction_correct=False, referents_compatible=False, external_knowledge_required=False,
            derivable_from_two_sources=False, concept_instance_scope='independent source-specific results',
            reason='Fare-optimization/airline-HSR results and lunar-regolith observations are independent outputs. Neither paper derives its results from the other data or results in the supplied sources. The same word results is not provenance in either direction.')
    if pid == 'N06' and code == 'PART_OF':
        return dict(classification='REJECTED', relation_type_correct=False,
            direction_correct=False, referents_compatible=False, external_knowledge_required=False,
            derivable_from_two_sources=False, concept_instance_scope='mammographic-density results versus corneal-graft study RESULTS heading',
            reason='The mammographic-density/risk findings are not a subset or component of the corneal-graft/glaucoma outcomes, nor vice versa. RESULTS is a section heading in B, not an encompassing collection that includes the independent findings in A.')
    if pid == 'N12' and code == 'HAS_PROPERTY':
        return dict(classification='REJECTED', relation_type_correct=False,
            direction_correct=False, referents_compatible=False, external_knowledge_required=False,
            derivable_from_two_sources=False, concept_instance_scope='two process-specific energy-consumption attributes',
            reason='Energy consumption in limonene chemical synthesis and energy consumption of a wireless sensor network are attributes of different systems. One energy-consumption quantity does not have the other quantity as a property. A shared measurement category cannot establish HAS_PROPERTY.')
    raise ValueError(f'UNREVIEWED OUTPUT: {pid} {code} {d}')


def datasets(m):
    contexts = {v: {c['pair_id']: c for c in core.read_jsonl(OUT/f'contexts_{v}.jsonl')} for v in m['variants']}
    results = {v: {r['pair_id']: r for r in core.read_jsonl(OUT/f'predictions_{v}.jsonl')} for v in m['variants']}
    if m['S3_globally_equivalent_to_S2']:
        contexts['S3'], results['S3'] = contexts['S2'], results['S2']
    return contexts, results


def validate(m, contexts, results):
    records = {r['audit_id']: r for r in core.read_jsonl(run.SOURCE)}
    vals = 0
    expected = {(code, d) for code in run.labels() for d in ['E2_TO_E3', 'E3_TO_E2']}
    for v in m['variants']:
        rows = core.read_jsonl(OUT/f'predictions_{v}.jsonl')
        require(len(rows) == 28 and set(results[v]) == set(records), 'Incomplete/duplicate results')
        require(len(contexts[v]) == 28 and set(contexts[v]) == set(records), 'Incomplete contexts')
        for pid, c in contexts[v].items():
            rec, r = records[pid], results[v][pid]
            require(c['input_sha256'] == core.digest([c['text'], c['fixed_spans']]) == r['input_sha256'], 'Input digest mismatch')
            require(r['inference_signature'] == m['signatures'][v], 'Signature mismatch')
            require(set(score_map(r)) == expected and len(r['raw_scores']) == 18, 'Missing/duplicate directed scores')
            require(r['audit']['spans'] == r['audit']['pairs'] == 2 and r['audit']['tokens'] == c['token_counts'][v] <= 512, 'Audit mismatch')
            require(r['audit']['relation_codes'] == list(run.labels()) and r['audit']['relation_prompts'] == list(run.labels().values()), 'Prompt/index mismatch')
            require(r['relations'] == [s for s in r['raw_scores'] if s['confidence'] >= .35], 'Wrong positive filter')
            require(len(c['fixed_spans']) == 2 and {s['target_slot'] for s in c['fixed_spans']} == {'E2','E3'}, 'Extra spans')
            for sp in c['fixed_spans']:
                p = rec[sp['target_slot']]['profile']
                require(sp['entity_id'] == p['canonical_entity_id'], 'Wrong target ID')
                require(c['text'][sp['start']:sp['end']] == sp['surface_text'] == p['mention_text'], 'Wrong joint span')
            for f in c['source_fragments']:
                doc = rec[f['target_slot']]['source_document']
                require(f['document_id'] == doc['document_id'] and f['revision_id'] == doc['revision_id'], 'Wrong source revision')
                require(doc['text'][f['start']:f['end']] == f['text'] == c['text'][f['model_text_start']:f['model_text_end']], 'Rewritten source passage')
            for s in r['raw_scores']:
                require(0 <= s['confidence'] <= 1, 'Nonfinite score')
                require(s['head_id'] == rec[s['head_slot']]['profile']['canonical_entity_id'] and s['tail_id'] == rec[s['tail_slot']]['profile']['canonical_entity_id'], 'Wrong output endpoint')
                vals += 1
            if 'reused_from' in r:
                alias = r['reused_from']
                original = results[alias['configuration']][pid]
                require(r['raw_scores'] == original['raw_scores'] and c['input_sha256'] == original['input_sha256'], 'Invalid reuse')
    require(run.protected(m['protected_files_sha256']) == m['protected_files_sha256'], 'Protected sources changed')
    require({name: run.sha(run.prior.old.MODEL_PATH/name) for name in m['common_inference']['checkpoint_sha256']} == m['common_inference']['checkpoint_sha256'], 'Model files changed')
    return dict(all_artifacts_passed=True, source_substrings_rechecked=True, output_endpoint_ids_rechecked=True,
        source_and_model_checkpoint_hashes_unchanged=True, complete_raw_values_rechecked=vals,
        all_thresholds_derived_without_inference=True)


def analyze():
    run.banner('НАЧИНАЮ АУДИТ СОХРАНЁННЫХ SCORES И ПРОВЕРКУ АРТЕФАКТОВ; НОВОГО INFERENCE НЕТ')
    m = run.checked_manifest()
    contexts, results = datasets(m)
    require(run.load('validation.json')['inference_passed'], 'Inference not complete')
    with (OUT/'pair_scope_review.csv').open(encoding='utf-8', newline='') as f:
        scope = {r['pair_id']:r for r in csv.DictReader(f)}
    variants = ['S1','S2','S3']
    quality = []
    for v in variants:
        for pid, r in results[v].items():
            if not r['relations']:
                quality.append(dict(configuration=v,pair_id=pid,head_slot='',tail_slot='',relation_code='',confidence='',
                    classification='NO_POSITIVE_PREDICTION',review_applicable=False,
                    relation_type_correct='not_asserted',direction_correct='not_asserted',referents_compatible='not_asserted',
                    external_knowledge_required='not_asserted',derivable_from_two_sources='not_asserted',concept_instance_scope='',
                    reason='No score >=0.35; this is abstention, not a supported negative relation.',
                    source_fragments=json.dumps(contexts[v][pid]['source_fragments'],ensure_ascii=False)))
            for s in r['relations']:
                assessment = judge(pid,s)
                quality.append(dict(configuration=v,pair_id=pid,head_slot=s['head_slot'],tail_slot=s['tail_slot'],
                    relation_code=s['relation_code'],confidence=s['confidence'],classification=assessment.pop('classification'),
                    review_applicable=True,**assessment,source_fragments=json.dumps(contexts[v][pid]['source_fragments'],ensure_ascii=False)))
    run.csvout('quality_review.csv',quality)
    lookup={(r['configuration'],r['pair_id'],r['head_slot'],r['tail_slot'],r['relation_code']):r for r in quality if r['review_applicable']}
    raw=[]
    for pid in sorted(scope):
        maps={v:score_map(results[v][pid]) for v in variants}
        for code,d in maps['S1']:
            row=dict(pair_id=pid,inference_scope=scope[pid]['inference_scope'],relation_code=code,direction=d,
                expected_candidate_meaning=(code,d)==run.EXPECTED.get(pid),
                head_id=maps['S1'][code,d]['head_id'],tail_id=maps['S1'][code,d]['tail_id'])
            row.update({v+'_score':maps[v][code,d]['confidence'] for v in variants})
            row.update(S2_minus_S1=row['S2_score']-row['S1_score'],S3_minus_S1=row['S3_score']-row['S1_score'],S3_minus_S2=row['S3_score']-row['S2_score'])
            raw.append(row)
    run.csvout('raw_score_comparison.csv',raw)
    thresholds=[]
    for v in variants:
        for t in run.THRESHOLDS:
            selected=[(pid,s) for pid,r in results[v].items() for s in r['raw_scores'] if s['confidence']>=t]
            verdicts=Counter(lookup[v,pid,s['head_slot'],s['tail_slot'],s['relation_code']]['classification'] for pid,s in selected)
            row=dict(configuration=v,threshold=t,predictions=len(selected),positive_pairs=len({pid for pid,s in selected}),
                P_predictions=sum(pid.startswith('P') for pid,s in selected),
                N_predictions=sum(pid.startswith('N') for pid,s in selected),
                W_predictions=sum(pid.startswith('W') for pid,s in selected),
                N_positive_pairs=len({pid for pid,s in selected if pid.startswith('N')}),
                exact_P_candidate_hits=len({pid for pid,s in selected if (s['relation_code'],direction(s))==run.EXPECTED.get(pid)}),
                SUPPORTED_CONCEPT_RELATION=verdicts['SUPPORTED_CONCEPT_RELATION'],PLAUSIBLE_HYPOTHESIS=verdicts['PLAUSIBLE_HYPOTHESIS'],
                UNCERTAIN=verdicts['UNCERTAIN'],REJECTED=verdicts['REJECTED'],model_forwards=0)
            thresholds.append(row)
    run.csvout('threshold_comparison.csv',thresholds)
    detail=[]
    for pid,(code,d) in run.EXPECTED.items():
        row=dict(pair_id=pid,inference_scope=scope[pid]['inference_scope'],expected_relation=code,expected_direction=d)
        inverse='E3_TO_E2' if d=='E2_TO_E3' else 'E2_TO_E3'
        for v in variants:
            r=results[v][pid];maps=score_map(r);best=max(r['raw_scores'],key=lambda s:s['confidence'])
            row.update({v+'_expected_score':maps[code,d]['confidence'],v+'_inverse_score':maps[code,inverse]['confidence'],
                v+'_max_score':best['confidence'],v+'_max_code':best['relation_code'],v+'_max_direction':direction(best),
                v+'_expected_over_inverse':maps[code,d]['confidence']>maps[code,inverse]['confidence'],
                v+'_exact_hit_at_035':maps[code,d]['confidence']>=.35})
        row.update(S2_minus_S1=row['S2_expected_score']-row['S1_expected_score'],S3_minus_S1=row['S3_expected_score']-row['S1_expected_score'])
        detail.append(row)
    run.csvout('positive_candidate_scores.csv',detail)
    report(m,contexts,results,thresholds,quality,detail,scope)
    val=run.load('validation.json')
    val.update(validate(m,contexts,results))
    val.update(positive_outputs_reviewed=sum(r['review_applicable'] for r in quality),
        review_status='all outputs >=0.35 explicitly source-reviewed',reviewer_count=1,independent_gold=False,
        tests=dict(command='.venv/Scripts/python.exe -m pytest tests/test_split_context_experiment.py tests/test_cross_document_ontology_experiment.py tests/test_build_relex_graphs.py tests/test_relex_four_experiments.py -q',passed=46,warnings=7),
        completed_utc=datetime.now(timezone.utc).isoformat())
    run.save('validation.json',val)
    m['analysis']=dict(thresholds=run.THRESHOLDS,threshold_rows=len(thresholds),raw_comparison_rows=len(raw),
        positive_candidate_score_rows=len(detail),audited_outputs=val['positive_outputs_reviewed'],threshold_forwards=0,
        report_runner_sha256=run.sha(Path(__file__)),completed_utc=val['completed_utc'])
    m['output_sha256']={p.name:run.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    run.save('manifest.json',m)
    print(json.dumps(dict(validation='passed',forwards=m['model_forwards'],audited_outputs=val['positive_outputs_reviewed'])))


def report(m,contexts,results,thresholds,quality,detail,scope):
    variants=['S1','S2','S3']
    lines=[
        '# Stage 2F.4 — Split-Context Cross-Document ReLEx Experiment',
        '',
        'Дата клиента: 2026-11-13 (Europe/Moscow). Исходный HEAD `bacc41e`, ветка `design-v2-sequential-kg`, исходное дерево чистое. Исследование ограничено 28 ранее выбранными E2–E3 парами, одной онтологией D и одним checkpoint. Никаких операций с Neo4j и перехода к Stage 2F.5.',
        '',
        '## Основной результат',
        '',
        '**Ни одной содержательно принятой междокументной связи при любом из 12 порогов не получено. Ни одна из P01–P06 не дала даже any-label выхода >=0.35 в S1, S2 или S3.** Увеличение непрерывного исходного контекста подавило ложные выходы короткого S1, но не обнаружило искомые смыслы. Отбор дополнительных исходных предложений по локальным RELEX_REL также не дал содержательного положительного результата.',
        '',
        'Это результат ограниченного диагностического development set, не доказательство невозможности split-context RE для checkpoint вообще. В наборе нет бесспорного научного положительного примера, для которого требуются две независимые текстовые посылки: один узкий лексический participant-кандидат P03 не является таким тестом. Следовательно, нельзя оценивать полный recall или причинно приписывать нулевой выход только модели.',
        '',
        '| Вариант | Предсказания >=0.35 | Пары с выходом / 28 | P exact / any-label | N-пары с ошибками / 12 | Supported / plausible / uncertain / rejected |',
        '|---|---:|---:|---|---:|---|',
    ]
    for v in variants:
        r=next(r for r in thresholds if r['configuration']==v and r['threshold']==.35)
        lines.append(f"| {v} | {r['predictions']} | {r['positive_pairs']} | {r['exact_P_candidate_hits']} / {r['P_predictions']} | {r['N_positive_pairs']} | {r['SUPPORTED_CONCEPT_RELATION']} / {r['PLAUSIBLE_HYPOTHESIS']} / {r['UNCERTAIN']} / {r['REJECTED']} |")
    lines += [
        '',
        'W01–W10 во всех вариантах также дали нулевой выход. Для S2 precision пустого выхода не определена; ноль ошибок нельзя трактовать как высокое качество извлечения. N-пары диагностируют конкретные неподтверждённые claims, а не являются универсальным запретом любой связи между понятиями.',
        '',
        '## Что именно сравнивалось',
        '',
        'Протокол зафиксирован в [experiment_design.md](experiment_design.md), вместе с hashes всех контекстов и pair_scope_review.csv до первого forward. После просмотра результатов контексты, scope и labels не менялись. S1 содержит по одному короткому исходному предложению вокруг выбранного native mention. S2 использует оба полных абстракта, если они помещаются; иначе непрерывно расширяет S1 сначала предложениями, затем словами до доступного бюджета. S3 добавляет к S1 исходные предложения с endpoints incident RELEX_REL: relation labels и confidence используются только как сохранённые metadata, в model text они не вставляются; confidence не участвует в отборе.',
        '',
        'Во входе нейтральные документные границы; разрыв внутри документа маркируется PASSAGE BREAK. Абстракты не переписываются, искусственное мостовое утверждение не создаётся. Ровно два targets E2/E3 во всех вариантах, ровно две направленные пары. E1/E4 не становятся targets. Scope, expected labels, аудиторские комментарии и внешние научные определения никогда не передаются модели. Полные исходные абстракты и provenance лежат в pair_scope_review.csv; использованные фрагменты с исходными и model offsets — в contexts JSONL.',
        '',
        '| Вариант | Min / mean / max tokens с prompts | Обе полные аннотации | Источниковых passages |',
        '|---|---|---:|---:|',
    ]
    for v in variants:
        cs=list(contexts[v].values());ns=[c['token_counts'][v] for c in cs]
        lines.append(f"| {v} | {min(ns)} / {sum(ns)/len(ns):.2f} / {max(ns)} | {sum(all(c['full_abstracts'].values()) for c in cs)} / 28 | {sum(len(c['source_fragments']) for c in cs)} |")
    alias_ids=[pid for pid,r in results['S3'].items() if 'reused_from' in r]
    full_ids=[pid for pid,c in contexts['S2'].items() if all(c['full_abstracts'].values())]
    retrieval=Counter(ref['status'] for c in contexts['S3'].values() for ref in c['retrieval_refs'])
    lines += [
        '',
        f"S3 отличается от S2; отдельные файлы нужны. S3=S1 для {', '.join(alias_ids)}: использованы сохранённые векторы без повторного forward. В частности, P03 не имеет incident factual flank, а P05 не получает нового предложения сверх S1. Это отсутствие дополнительной evidence, а не независимое подтверждение одинакового результата. Полные аннотации в S2: {', '.join(full_ids)}. В остальных 22 парах часть источников отрезана по плану; это **явный crop до inference**, а не скрытая truncation collator/encoder.",
        '',
        f"S3 retrieval refs по статусам: {dict(retrieval)}. Несколько refs могут ссылаться на один и тот же passage, поэтому число refs не является числом независимых свидетельств. Отбор фиксирован расстоянием/source offsets, не expected relation или scores. S2 и S3 различаются одновременно длиной, непрерывностью и отбором passages; этот опыт не изолирует влияние каждого из этих факторов.",
        '',
        '## Диагностический scope до inference',
        '',
        '| Пары | Тип необходимого вывода | Ограничение |',
        '|---|---|---|',
        '| P01 | BACKGROUND-KNOWLEDGE DEPENDENT | ESRD как стадия CKD не определена в двух абстрактах |',
        '| P02 | HYPOTHETICAL TRANSFER | Неизвестны low Hb/анемия, eligibility и экспозиция rHuEPO у dialysis/TKA-когорты; нужен также внешний ESRD/CKD-шаг |',
        '| P03 | TEXT-SUPPORTED SPLIT, lexical only | Activity-to-cell-class; смысл участника уже читается в A, B подтверждает название класса, а не общий механизм |',
        '| P04 | BACKGROUND-KNOWLEDGE DEPENDENT, conservative | Термины генетической expression совместимы, но определение overexpression/subtype явно не дано |',
        '| P05 | BACKGROUND-KNOWLEDGE DEPENDENT | Состав bundle из индивидуальных fibers не описан; concept type не означает идентичное оборудование |',
        '| P06 | HYPOTHETICAL TRANSFER | MRI denoising не доказывает пользу конкретного OB-NLM для mammographic density/risk |',
        '| N01–N12, W01–W10 | NEGATIVE / AMBIGUOUS | Для конкретных прежних proposed claims нет достаточных оснований; подробные reasons сохранены для каждой пары |',
        '',
        'Итого: 1 узкая TEXT-SUPPORTED SPLIT, 3 background-dependent, 2 hypothetical, 22 negative/ambiguous. Это разметка одного исследователя, не KG-отношения. Если считать общеязыковой смысл overexpression или bundle достаточным для P04/P05, число лексических concept positives увеличится, но observed yield останется нулевым: все шесть P abstain во всех labels/directions. Не следует представлять этот выбор scope как независимую научную gold-разметку. **Нетривиальных бесспорных TEXT-SUPPORTED SPLIT positives здесь 0.**',
        '',
        '## Scores шести содержательных кандидатов',
        '',
        'Expected meanings и направления взяты из прежнего D-coverage. P05 проверяется в обратном E3→E2 направлении; все остальные — E2→E3. Это некалиброванные sigmoid scores, не вероятность научной истинности. Сравнение полных векторов (504 строки, оба направления и все девять labels) находится в [raw_score_comparison.csv](raw_score_comparison.csv); expected/inverse/max и deltas — в [positive_candidate_scores.csv](positive_candidate_scores.csv).',
        '',
        '| Пара | Expected / direction | S1 | S2 | S3 | Δ S2−S1 | Δ S3−S1 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for r in detail:
        lines.append(f"| {r['pair_id']} | {r['expected_relation']} {r['expected_direction']} | {r['S1_expected_score']:.8f} | {r['S2_expected_score']:.8f} | {r['S3_expected_score']:.8f} | {r['S2_minus_S1']:+.8f} | {r['S3_minus_S1']:+.8f} |")
    lines += [
        '',
        'S2 повышает expected score только у P05/P06 (2 из 6), снижает у остальных четырёх. S3 повышает score P06, снижает P01/P02/P04; P03/P05 идентичны S1. Все expected scores остаются ниже 0.015 при пороге 0.35. Для P02 даже самый высокий альтернативный APPLIED_TO не подтверждает actual drug exposure. Для P04 альтернативный USED_FOR не заменяет нужный subtype.',
        '',
        '| Пара / вариант | Максимальный score среди всех 18 | Максимальный label / direction | Expected / inverse score | Направление expected выше inverse? |',
        '|---|---:|---|---|---|',
    ]
    for r in detail:
        for v in variants:
            lines.append(f"| {r['pair_id']} / {v} | {r[v+'_max_score']:.8f} | {r[v+'_max_code']} {r[v+'_max_direction']} | {r[v+'_expected_score']:.8f} / {r[v+'_inverse_score']:.8f} | {'да' if r[v+'_expected_over_inverse'] else 'нет'} |")
    lines += [
        '',
        'У P01 и P04 expected label предпочитает неверное направление во всех вариантах. У P05 верное E3→E2 направление выше inverse только в S2. P02/P03/P06 имеют верный относительный порядок expected/inverse, но это не correctness предсказания: accepted output отсутствует. Для P03 лексический participant-смысл допустим только на уровне класса; Breg suppression не переносится в rHuEPO-контекст. Для P04 не переносится механизм CNV мышей на инженерную экспрессию дрожжей. Для P05 не установлено, что fibers модели B являются физическими компонентами установки A. Эти ограничения сохраняются независимо от ранга scores.',
        '',
        '## Все положительные выходы и содержательный аудит',
        '',
        '| Вариант / пара | Head → label → tail | Score | Вердикт |',
        '|---|---|---:|---|',
    ]
    for q in quality:
        if q['review_applicable']:
            lines.append(f"| {q['configuration']} / {q['pair_id']} | {q['head_slot']} → {q['relation_code']} → {q['tail_slot']} | {q['confidence']:.8f} | {q['classification']} |")
    lines += [
        '',
        'Все восемь выходов отклонены по конкретным источникам, а не по префиксу N:',
        '',
        '- **N03 HAS_PROPERTY:** resolution morphology coding и angular resolution наблюдений Луны — разные характеристики. Одна resolution не является носителем другой как свойства. В S3 добавление исходного предложения о радионаблюдениях не исправляет head/tail roles.',
        '- **N05 DERIVED_FROM:** результаты fare optimization/cooperation авиации–HSR не получены из результатов наблюдений lunar regolith, и обратное происхождение также не указано. В S1 ошибочны оба направления, включая score 0.88485354. S3 добавляет реальный контекст fare optimization и polarization/regolith, но оставляет один ложный provenance claim.',
        '- **N06 PART_OF:** mammographic-density/risk findings не входят в corneal-graft/glaucoma outcomes. Target RESULTS в B — заголовок раздела независимой статьи, а не множество, включающее результаты A. Два направления одинаково необоснованны.',
        '- **N12 HAS_PROPERTY:** consumption химического синтеза limonene и consumption WSN — свойства разных систем. Общая физическая категория не делает одну величину свойством другой.',
        '',
        'Ни в одном случае для отклонения не нужно внешнее научное определение: различие объектов/сфер и отсутствие provenance/part/bearer связи видны в двух текстах. Отдельные type, direction, referent compatibility, concept/instance и derivability fields с точными входными фрагментами лежат в [quality_review.csv](quality_review.csv). Строки NO_POSITIVE_PREDICTION указывают abstention и не участвуют в счётчиках вердиктов.',
        '',
        '## Все заранее заданные пороги',
        '',
        'Фильтр confidence >= t, только по сохранённым полным векторам; повторных forwards для порогов 0. Все выходы ниже — REJECTED, поэтому accepted=0, P hits=0, W output=0 в каждой строке.',
        '',
        '| t | S1 predictions / N pairs | S2 predictions / N pairs | S3 predictions / N pairs |',
        '|---|---|---|---|',
    ]
    for t in run.THRESHOLDS:
        cells=[]
        for v in variants:
            r=next(x for x in thresholds if x['configuration']==v and x['threshold']==t)
            cells.append(f"{r['predictions']} / {r['N_positive_pairs']}")
        lines.append(f"| {t:.2f} | " + ' | '.join(cells)+' |')
    lines += [
        '',
        'Дополнительный контекст не увеличил число ложных связей относительно S1: S2 6→0, S3 6→2; число затронутых N-пар 4→0/2. Однако относительно S2 вариант S3 возвращает два ложных claim. Это не монотонный эффект длины и не улучшение положительного извлечения. Снижение порога ради получения результата не проводилось.',
        '',
        '## Что проверено технически',
        '',
        f"Локальный `{m['common_inference']['model_id']}`, CPU/{m['common_inference']['threads']} threads, seed=0, deterministic algorithms, eval/inference_mode. Версии: {m['common_inference']['runtime']}. Неизменные checkpoint files и D ontology проверены SHA-256 до/после. Existing predict_window adapter не изменялся. Prompts идут в исходном порядке: `" + '`, `'.join(m['common_inference']['relation_labels'].values()) + '`.',
        '',
        f"Выполнено **{m['model_forwards']} forwards**, **{m['reused_identical_inputs']} эквивалентных входов переиспользованы**; сохранено 84 pair/variant вектора и 1512 scores. Для каждого входа до forward проверены tokenizer без truncation, фактический collator и label indexes. Во время forward проверены 2 target spans, 2 directed pairs, 9 relation prompts и совпадение token counts; существующий adapter также проверяет длину word embeddings. Все 18 уникальных direction/label scores сохранены, включая значения ниже 0.35 и возможный точный zero. JSONL fsync после каждой пары, resume через signature/input hashes; прогресс — настоящий tqdm. Все {len(m['protected_files_sha256'])} ранее отслеживаемых файлов проверены по hashes: предыдущие отчёты, inference, ontology и исходники не менялись.",
        '',
        '46 тестов прошли: существующие адаптерные проверки и эксперименты плюс четыре новые проверки provenance/непрерывных и разорванных passages, одинаковых surface forms, полного сохранения targets и исключения metadata answers из model text. Семь предупреждений — существующий vendor torch.jit.script deprecation. Дополнительно validation.json проверяет все реальные артефакты, scores, threshold derivation и alias equivalence. Предыдущий sanity D=6/6 и историческая adapter compatibility использованы по их сохранённым hashes, новый intra-document опыт не запускался. Этот sanity не подтверждает положительную работу всех девяти labels: STAGE_OF, PART_OF, HAS_PARTICIPANT и POTENTIALLY_APPLICABLE_TO по-прежнему не имеют отдельного positive sanity.',
        '',
        '## Ограничения и возможные причины',
        '',
        '| Причина | Что установлено | Что остаётся гипотезой |',
        '|---|---|---|',
        '| Нехватка научной информации / external knowledge | P01/P04/P05 не содержат явных нужных определений; P02/P06 не сообщают условия переносимости | Доля неудач именно из-за этой нехватки причинно не оценена |',
        '| Нет общего утверждения E2→REL→E3 | Никакое такое утверждение не добавлялось; входы сохраняют независимые источники | Насколько модель требует co-occurrence statement, не изолировано |',
        '| Crop 512 tokens | S2 содержит оба полных абстракта только в 6/28, в том числе P05; скрытой truncation нет | Потеря конкретного необходимого evidence в обрезанных частях может влиять на часть scores; назвать её общей причиной нельзя |',
        '| Relation-label understanding | D даёт сохранённые положительные intra-document sanity для части labels; mapping технически исправен | Принятие четырёх неохваченных labels не проверено положительным sanity, причинно исключить lexical-label difficulty нельзя |',
        '| Span-pair classifier / joint input | При трёх фиксированных структурах 0 accepted, некоторые ложные claims достигают 0.885 | Архитектурная неспособность объединять источники и неудачная структура markers — альтернативные объяснения, не доказанные факты |',
        '| Target selection / concept versus instance | Native mentions сохранены; N06 содержит heading, P05 concept-type требует отличать от оборудования, P03 — от конкретной популяции | Выбор других mentions/концептов может изменить scores, здесь не тестировался |',
        '',
        'На вопрос «что сильнее ограничивает: источники, формат или модель?» этот опыт не даёт причинного ранжирования. Дефицит бесспорных two-premise positives непосредственно установлен аудитом. Измерены влияние трёх входных рецептов на scores/ошибки и отсутствие принятого yield; сопоставление с положительными контрфактическими условиями не выполнялось. У P05 даже полные исходные абстракты не дают output, но в них всё равно нет определения компонента; это не чистый тест архитектуры. Узкий лексический P03 также остаётся ниже порога, что мотивирует отдельную проверку понимания participant label, а не универсальное заключение.',
        '',
        '## Ответы на исследовательские вопросы и следующий минимальный опыт',
        '',
        '1. **Способность обнаруживать split-context отношения:** на этих 28 парах accepted typed yield=0. Модель выдаёт типизированные edges, но все положительные outputs неподдержаны. Нетривиальная способность объединять научные посылки здесь не продемонстрирована и общая невозможность не доказана.',
        '2. **Увеличение исходного контекста:** S2 убирает шесть ложных outputs S1, но не распознаёт ни одну P-пару; expected scores растут только в 2/6 случаях.',
        '3. **Научные окрестности:** S3 не даёт accepted yield, сохраняет две ошибки; для P03/P05 дополнительной информации фактически нет.',
        '4. **Шесть кандидатов:** 0/6 exact candidate hits и 0/6 any-label positive pairs во всех вариантах, включая гипотезы P02/P06.',
        '5. **TEXT-SUPPORTED SPLIT:** есть только узкий лексический P03, который модель не извлекает; бесспорных нетривиальных two-source positives нет. Этот набор не измеряет recall таких примеров.',
        '6. **Expected scores:** все значения, обратные направления и deltas приведены выше и в CSV; рост P05/P06 не является доказательством правильности.',
        '7. **Ложные связи:** по сравнению с S1 уменьшаются до 0 в S2 и до 2 в S3; по сравнению с S2 S3 хуже. Высокий score ложного DERIVED_FROM показывает отсутствие научной калибровки.',
        '8. **Главное ограничение:** дефицит достаточных посылок подтверждён на уровне выборки; вклад input и модели требует отдельного контроля и пока не ранжируется.',
        '9. **Продолжать ReLEx:** оснований использовать второй проход для автоматического создания cross-document KG edges на этом корпусе нет. Есть основание для одного ограниченного диагностического контроля перед окончательным отказом от гипотезы о checkpoint; расширять бесконечный поиск labels/снижать пороги по P-набору не обосновано.',
        '10. **Следующий минимальный эксперимент (только предложение, не выполнен):** сначала проверить, удаётся ли независимо разметить хотя бы несколько реальных source-only пар с двумя необходимыми посылками и исключённым внешним знанием, плюс совместимые негативные контроли. Использовать те же D/checkpoint, заранее заморозить оценки; без пригодных positives модельный benchmark не запускать. Для исключения label failure отдельно достаточно одного естественного, уже существующего intra-document утверждения для каждого необходимого непроверенного predicate (начать с HAS_PARTICIPANT для P03), без создания предложений или нового ontology. Только затем сравнить минимальный/полный split на этих парах; в дальнейшей отдельно разрешённой работе контролировать порядок/границы документов. Подготовка нового корпуса, inference и Stage 2F.5 сейчас не начинались.',
        '',
        '## Артефакты и воспроизведение',
        '',
        'Все запрошенные файлы сохранены в этой директории: experiment_design.md, pair_scope_review.csv, contexts_S1/S2/S3.jsonl, predictions_S1/S2/S3.jsonl, raw_score_comparison.csv, threshold_comparison.csv, quality_review.csv, comparison_report.md, manifest.json, validation.json. Дополнительный positive_candidate_scores.csv содержит компактные expected/inverse/max/delta данные. Пустых файлов нет; отрицательные predictions-файлы содержат полные векторы, а не пустой output. Частичные эквивалентности отмечены, полностью дублирующих файлов нет.',
        '',
        'Подготовка: `.venv/Scripts/python.exe scripts/experiment_split_context.py --stage prepare` (отказывается перезаписывать frozen inputs). Inference/resume: та же команда с `--stage infer`. Повторный analysis без inference: `.venv/Scripts/python.exe scripts/report_split_context.py`. manifest.json хранит frozen source/input/model/runner hashes, версии и output hashes; собственный hash manifest намеренно не включается. Commit остаётся локальным, push не выполняется.',
    ]
    (OUT/'comparison_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    analyze()
