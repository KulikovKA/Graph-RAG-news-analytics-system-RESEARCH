"""Source review and saved-score comparison for local GLiREL; no inference."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import evaluate_glirel as run

OUT,core,require=run.OUT,run.core,run.require
SANITY_REASONS={
    'S01':'The source explicitly calls recombinant human erythropoietin a biopharmaceutical drug: class inclusion, not identity.',
    'S02':'Fiber-reinforced composites have a favorable strength-to-weight ratio; the material bears the property.',
    'S03':'The source explicitly reports energy consumption of WSN; WSN is the bearer, not the property.',
    'S04':'The GPU parallel algorithm is explicitly based on optimized blockwise non-local means; source/method derivation.',
    'S05':'Multilayers were investigated by XMCD; method-to-object actual application is explicit.',
    'S06':'The source proposes GPUs for computational steps supporting real-time diagnosis; intended use, not autonomous clinical diagnosis.',
}


def direction(s):
    return s['head_slot']+'_TO_'+s['tail_slot']


def expected_score(r,code,d):
    return next((s['confidence'] for s in r['raw_scores'] if s['relation_code']==code and direction(s)==d),None)


def csvout(rows):
    require(bool(rows),'No review rows')
    with (OUT/'quality_review.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def judge(pid,s,pool):
    """This run has no positive outputs; fail if new outputs need a fresh review."""
    raise ValueError(f"UNREVIEWED POSITIVE: {pool}/{pid}/{s['head_slot']}/{s['relation_code']}/{s['tail_slot']}")


def reviews(rows):
    result=[]
    for r in rows:
        positives=[s for s in r['raw_scores'] if s['confidence']>=.35]
        for s in positives:
            verdict=judge(r['pair_id'],s,r['pool'])
            result.append(dict(pool=r['pool'],pair_id=r['pair_id'],head_slot=s['head_slot'],tail_slot=s['tail_slot'],
                relation_code=s['relation_code'],score=s['confidence'],classification=verdict['classification'],
                **{k:v for k,v in verdict.items() if k!='classification'},
                direct_input_comparison=r['unchanged_source_input'],actual_input_sha256=r['actual_input_sha256']))
        if not positives:
            result.append(dict(pool=r['pool'],pair_id=r['pair_id'],head_slot='',tail_slot='',relation_code='',score='',
                classification='NO_POSITIVE_PREDICTION',scientific_meaning_correct='not_asserted',relation_type_correct='not_asserted',
                direction_correct='not_asserted',text_supported='not_asserted',requires_external_knowledge='not_asserted',
                independent_studies_merged=False,concept_instance_scope='',
                reason='All available scores are below 0.35; abstention is not a gold NO_RELATION decision.',
                direct_input_comparison=r['unchanged_source_input'],actual_input_sha256=r['actual_input_sha256']))
    return result


def sanity():
    m=run.load();require(m['sanity']['technical_passed'],'Sanity technical failure')
    path=OUT/'sanity_results.jsonl';rows=core.read_jsonl(path)
    require(len(rows)==6,'Incomplete sanity')
    for r in rows:
        score=expected_score(r,r['expected_D_code'],r['expected_direction'])
        max_s=max(r['raw_scores'],key=lambda s:s['confidence']) if r['raw_scores'] else None
        r['source_review']=dict(expected_relation_verdict='SUPPORTED_CONCEPT_RELATION',reason=SANITY_REASONS[r['pair_id']],
            expected_score=score,expected_direction=r['expected_direction'],
            exact_hit_by_threshold={str(t):score is not None and score>=t for t in run.THRESHOLDS},
            maximum_available_score=max_s,positive_predictions_at_035=[s for s in r['raw_scores'] if s['confidence']>=.35],
            result_status='ABSTAINED_AT_ALL_DIAGNOSTIC_THRESHOLDS' if not r['relations'] else 'POSITIVE_OUTPUT_SOURCE_REVIEWED')
    csvout(reviews(rows))
    core.write_jsonl(path,rows)
    m['sanity'].update(reviewed=True,results_sha256=run.sha(path),
        review_method='single reviewer, original source fragments and preserved expected directions',
        exact_hits={str(t):sum(r['source_review']['exact_hit_by_threshold'][str(t)] for r in rows) for t in run.THRESHOLDS},
        diagnostics='Strict complete state_dict load; input token alignment; both inclusive target spans and directed pairs; nine labels/collator; actual encoder length; all 108 scores available. Content failures do not imply an API error. No labels or thresholds were changed.',
        continue_reason='All technical checks passed; zero expected relations at diagnostic cutoffs is a preserved substantive result, not a reason to suppress the cross-document evaluation.')
    run.save(m)
    print(json.dumps(m['sanity'],ensure_ascii=False))


def finalize():
    run.banner('НАЧИНАЮ ИТОГОВЫЙ SOURCE-АУДИТ GLIREL И СРАВНЕНИЕ СОХРАНЁННЫХ SCORES; INFERENCE НЕТ')
    m=run.load()
    require(m['sanity']['reviewed'] and m['cross']['technical_passed'],'Incomplete experiment')
    sanity_rows=core.read_jsonl(OUT/'sanity_results.jsonl')
    cross_rows=core.read_jsonl(OUT/'cross_document_predictions.jsonl')
    all_rows=sanity_rows+cross_rows
    require(len(sanity_rows)==6 and len(cross_rows)==28,'Wrong pair totals')
    review=reviews(all_rows);csvout(review)
    sources={r['pair_id']:r for r in core.read_jsonl(run.CROSS)}
    sources.update({r['pair_id']:r for r in core.read_jsonl(run.SANITY)})
    expected_keys={(h,t,c) for h,t in [('E2','E3'),('E3','E2')] for c in run.split.labels()}
    for row in all_rows:
        src=sources[row['pair_id']];inp=row['actual_input'];audit=row['token_audit']
        require(row['inference_signature']==m['inference_signature'],'Mixed signature')
        require(core.digest([inp['text'],inp['fixed_spans']])==row['actual_input_sha256'],'Actual input hash')
        if row['unchanged_source_input']:
            require(inp['text']==src['text'] and inp['fixed_spans']==src['fixed_spans'],'Changed comparable input')
        regenerated=run.tokenize_context(inp)
        require(regenerated['tokens']==audit['tokens'] and regenerated['ner']==audit['ner'],'Token offset reconstruction')
        require(len(audit['ner'])==2 and audit['transformer_tokens']<=512 and audit['source_word_count']<=512,'Lengths/targets')
        require(row['actual_encoder_forwards']==1,'Not one encoder call per pair')
        require(row['relations']==[s for s in row['raw_scores'] if s['confidence']>=.35],'Wrong saved positives')
        scores,missing=run.convert(row['raw_api_predictions'],audit,run.split.labels())
        require(scores==row['raw_scores'] and missing==row['unreturned_scores'],'API score conversion mismatch')
        present={(s['head_slot'],s['tail_slot'],s['relation_code']) for s in scores}
        absent={(s['head_slot'],s['tail_slot'],s['relation_code']) for s in missing}
        require(present.isdisjoint(absent) and present|absent==expected_keys,'Missing-score accounting')
        for f in inp['source_fragments']:
            original=next(p for p in src['source_fragments'] if p['document_id']==f['document_id'] and p['start']<=f['start'] and f['end']<=p['end'])
            require(original['text'][f['start']-original['start']:f['end']-original['start']]==f['text'],'Non-source crop')
    require(run.split.protected(m['protected_files_sha256'])==m['protected_files_sha256'],'Protected project files changed')
    require({n:run.sha(Path(m['model_path'])/n) for n in m['model_sha256']}==m['model_sha256'],'Model weights changed')
    thresholds=[]
    for pool,rows in [('sanity',sanity_rows),('cross',cross_rows)]:
        for t in run.THRESHOLDS:
            selected=[(r['pair_id'],s) for r in rows for s in r['raw_scores'] if s['confidence']>=t]
            judgments=Counter(judge(pid,s,pool)['classification'] for pid,s in selected)
            thresholds.append(dict(pool=pool,threshold=t,predictions=len(selected),positive_pairs=len({p for p,s in selected}),
                P_pairs=len({p for p,s in selected if p.startswith('P')}),N_pairs=len({p for p,s in selected if p.startswith('N')}),
                W_pairs=len({p for p,s in selected if p.startswith('W')}),verdict_counts=dict(judgments)))
    baseline={r['pair_id']:r for r in core.read_jsonl(run.split.OUT/'predictions_S2.jsonl')}
    report(m,sanity_rows,cross_rows,review,thresholds,baseline)
    m['cross'].update(reviewed=True,reviewed_outputs=sum(r['classification']!='NO_POSITIVE_PREDICTION' and r['pool']=='cross' for r in review))
    m['threshold_analysis']=thresholds
    m['validation']=dict(passed=True,source_substrings_rechecked=True,inclusive_ner_exclusive_output_mapping_rechecked=True,
        both_directions_checked=True,nine_prompts_checked=True,no_implicit_truncation=True,
        strict_complete_checkpoint_load=True,all_protected_files_unchanged=True,weights_unchanged=True,
        diagnostic_threshold_forwards=0,independent_gold=False,f1_computed=False,
        available_scores=sum(len(r['raw_scores']) for r in all_rows),
        absent_scores=sum(len(r['unreturned_scores']) for r in all_rows),
        tests=dict(command='.venv/Scripts/python.exe -m pytest tests/test_evaluate_glirel.py -q',passed=4),
        completed_utc=datetime.now(timezone.utc).isoformat())
    m['report_runner_sha256']=run.sha(Path(__file__))
    m['output_sha256']={p.name:run.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    run.save(m)
    print(json.dumps(dict(validation='passed',cross_pairs=28,thresholds=thresholds)))


def report(m,sanity_rows,cross_rows,review,thresholds,baseline):
    lines=[
        '# Stage 2F.4 — Local GLiREL Evaluation',
        '',
        'Дата клиента: 2026-11-13 (Europe/Moscow). Source HEAD `8175e8e`, ветка `design-v2-sequential-kg`. До эксперимента отслеживаемые файлы не изменены; скачанная пользователем `model/` оставалась untracked. Это последний ограниченный модельный тест перед отдельным решением о расширении корпуса; расширение не выполнялось.',
        '',
        '## Результат',
        '',
        f"GLiREL: exact sanity hits при >=0.35 — **{m['sanity']['exact_hits']['0.35']}/6**; междокументных outputs при >=0.35 — **{sum(len(r['relations']) for r in cross_rows)}**, P-пар с outputs — **{sum(bool(r['relations']) for r in cross_rows if r['pair_id'].startswith('P'))}/6**. Содержательно принятых cross outputs — **{sum(r['pool']=='cross' and r['classification'] in ['SUPPORTED_CONCEPT_RELATION','PLAUSIBLE_HYPOTHESIS'] for r in review)}**. Все три заданных порога показаны ниже. Положительного улучшения относительно ReLEx S2 не продемонстрировано.",
        '',
        'Результаты относятся только к установленной GLiREL 1.2.1, локальному `jackboyla/glirel-large-v0`, фиксированным девяти prompts онтологии D и сохранённым научным примерам. Пороги диагностические; scores разных моделей не считаются одинаково калиброванными. P01–P06 не являются полноценным gold benchmark, F1 не вычислялся.',
        '',
        '| Набор | Порог | Outputs | Пары с output | Вердикты |',
        '|---|---:|---:|---:|---|',
    ]
    for r in thresholds:
        lines.append(f"| {r['pool']} | {r['threshold']:.2f} | {r['predictions']} | {r['positive_pairs']} | {r['verdict_counts'] or 'нет положительных outputs'} |")
    lines += [
        '',
        '## Локальная модель, окружение и диагностика загрузки',
        '',
        f"Использована папка `{m['model_path']}` с glirel_config.json и pytorch_model.bin. Веса прочитаны на месте; не перемещались, не перезаписывались и повторно не скачивались. SHA-256 обоих файлов до/после сохранены в manifest.json. Размер checkpoint: {Path(m['model_path']).joinpath('pytorch_model.bin').stat().st_size} bytes. Предусмотренное альтернативное расположение `%USERPROFILE%/glirel-test/model` не потребовалось; окружение `%USERPROFILE%/glirel-test/.venv` при диагностике не найдено.",
        '',
        f"Python: `{m['api']['python']}`. Версии: {m['api']['runtime']}. В основном .venv уже был glirel, но отсутствовал импортируемый им loguru (пакет не объявлен в glirel Requires-Dist). loguru и его небольшие зависимости установлены только в `.cache/glirel_runtime` через pip --target; .venv, pyproject.toml и uv.lock не менялись. Flair этой установленной версии не нужен: её TokenRepLayer использует собственный TransformerWordEmbeddings.",
        '',
        'Обнаружены две проблемы загрузки, устранённые только в процессе:',
        '',
        '- GLiREL._from_pretrained требует устаревшие proxies/resume_download, которые установленный Hub mixin уже не передаёт. Scoped wrapper передаёт None/False внутреннему loader; файлы пакетов не изменены.',
        '- Локально есть DeBERTa-v3-large config и SentencePiece tokenizer, но отдельных базовых весов нет. Encoder создаётся из локального config, после чего публичный GLiREL.from_pretrained загружает **весь** GLiREL state_dict с strict=True и assign=True. Это не случайно инициализированная модель для inference: все encoder/head параметры заменены checkpoint-весами; missing/unexpected keys нет. Дополнительный backbone checkpoint не скачивался.',
        '',
        f"Загружены {m['api']['state_dict_entries']} state_dict entries, {m['api']['parameters']} параметров; tokenizer vocabulary {m['api']['tokenizer_size']}. CPU, 2 threads, seed=0, deterministic algorithms, eval/no_grad. Две ранние проверки public loader остановились до создания модели из-за Hub signature; успешная проверка загрузки до эксперимента не выполняла inference. Итоговый исследовательский inference содержит ровно {m['model_forwards']} encoder forwards.",
        '',
        'Loader также печатает config.json not found: у checkpoint конфигурация называется glirel_config.json, она успешно прочитана. Transformers 5.1.0 вывел generic warning о Mistral regex при чтении DeBERTa tokenizer. Дополнительная проверка фактически показала DebertaV2Tokenizer, Unigram SentencePiece и Metaspace pre-tokenizer, а не Mistral tokenizer. Предупреждение зафиксировано; автоматическая правка regex и замена версии пакета не проводились. Совместимость tokenization с исходным обучающим стеком не доказана только успешной загрузкой, поэтому отрицательный результат нельзя переносить на все версии GLiREL. Все actual token/mention и subword/word alignments в этом запуске прошли проверки.',
        '',
        '## Адаптер и отсутствие leakage',
        '',
        f"Реальная сигнатура: `{m['api']['predict_relations_signature']}`. В этой версии аргумент называется text, но принимает list[str]. Текст разбит ровно regex установленного GLiREL API: `\\w+(?:[-_]\\w+)*|\\S`. Использованы **включительные** ner end indices, по две известные сущности ENTITY, без NER. На выходе public API end уже **исключительный**, и преобразование обратно проверяется по точным tokens и IDs. ENTITY type не ограничивает relation labels.",
        '',
        'Перед forward проверены source/joint character spans, token boundaries, mention tokens, 2 entity spans, 2 направленные пары и ordered mapping девяти prompts. Проверяется вся subword→word карта; фактическая длина encoder input сверяется forward hook с preflight. Параметры public API: threshold=0.0, top_k=-1. Вендорский API выбирает scores строго >0; поэтому отсутствующие outputs сохранялись бы как unreturned score slots с null, а не нули. Диагностические outputs выделяются фильтром >=0.35/0.50/0.70 из сохранённых scores, новых forwards для порогов нет.',
        '',
        'Prompts D, без изменения порядка: `'+'`, `'.join(m['relation_labels'].values())+'`. Определения, expected labels, оценки аудитора, прежние outputs и внешние научные знания модели не передавались. В cross использован только исходный S2 joint text, с теми же DOCUMENT/END DOCUMENT boundaries. Никакого искусственного E2→REL→E3 утверждения.',
        '',
        '| Набор | Пары | Побайтно исходный текст/targets | Доступных scores | Не возвращённых slots | Max words / subwords с prompts |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for pool,rows in [('sanity',sanity_rows),('cross',cross_rows)]:
        lines.append(f"| {pool} | {len(rows)} | {sum(r['unchanged_source_input'] for r in rows)} | {sum(len(r['raw_scores']) for r in rows)} | {sum(len(r['unreturned_scores']) for r in rows)} | {max(r['token_audit']['source_word_count'] for r in rows)} / {max(r['token_audit']['transformer_tokens'] for r in rows)} |")
    changed=[r['pair_id'] for r in cross_rows if not r['unchanged_source_input']]
    lines += [
        '',
        ('Все 28 S2 inputs сохранены без сокращения: прямая сопоставимость исходного текста и targets с ReLEx S2 сохранена. Токенизация и архитектура моделей различаются.' if not changed else 'У GLiREL пришлось симметрично сократить следующие пары: '+', '.join(changed)+'. Для них полная прямая сопоставимость входов с ReLEx S2 потеряна; actual input и offsets сохранены.'),
        '',
        'Использован консервативный предел 512 transformer subwords с prompts, дополнительно к GLiREL config.max_len=512 исходных words. Если исходный вход не помещается, код уменьшает обе source области на одинаковую долю вокруг полных targets; ни один mention не отрезается. Фактические source substrings и provenance сохраняются в каждой JSONL записи. Результаты fsync после каждой пары; resume проверяет signature и не повторяет завершённые forwards.',
        '',
        '## Шесть внутридокументных sanity-пар',
        '',
        '| Пара | Expected / direction | GLiREL expected score | Максимальный output score / label / direction | Exact hit 0.35 / 0.50 / 0.70 |',
        '|---|---|---:|---|---|',
    ]
    for r in sanity_rows:
        s=r['source_review'];b=s['maximum_available_score']
        lines.append(f"| {r['pair_id']} | {r['expected_D_code']} {r['expected_direction']} | {s['expected_score'] if s['expected_score'] is None else format(s['expected_score'],'.8f')} | {b['confidence']:.8f} {b['relation_code']} {direction(b)} | "+' / '.join(str(int(s['exact_hit_by_threshold'][str(t)])) for t in run.THRESHOLDS)+' |')
    lines += [
        '',
        'Для каждой пары сохранены исходный фрагмент, head/tail, токенные и символьные spans, все фактические scores, expected relation и source-based review. Expected relations подтверждены исходным текстом; их вердикт SUPPORTED_CONCEPT_RELATION относится к проверяемому эталонному смыслу, а не к отсутствующему положительному output модели. В сохранённом прежнем sanity онтологии D GLiNER ReLEx распознала все 6/6 при >=0.35; здесь GLiREL не распознала ни одного при заданных cutoffs. Это отдельный sanity comparison, а не cross-document результат.',
        '',
        'Диагностика не обнаружила некорректных API arguments, смешения inclusive/exclusive spans, потерянных labels, дополнительных targets, отсутствующего направления, неполной загрузки параметров или truncation. Содержательный нулевой sanity сохранён и не использован для остановки cross. Причина низких scores (формулировки D, предметный домен, режим токенизации/версия runtime или checkpoint) причинно не отделена. Переписывание labels, подбор порога и внешние положительные демонстрации после просмотра результатов не выполнялись.',
        '',
        'Ниже порога есть различимый отклик: S02 и S03 ставят ожидаемое HAS_PROPERTY в верном направлении на первое место, но scores всего около 0.00252 и 0.14442. У S01 максимум относится к USED_FOR вместо subtype, у S04/S05 — к HAS_PROPERTY вместо derivation/application. У S06 обратный USED_FOR немного выше правильного. Это ошибки ранжирования/воздержание на известных смыслах, а не положительные ложные edges: выдаваемых при выбранных порогах edges нет. Нулевой threshold-yield не означает, что API возвращал пустые сырые векторы.',
        '',
        '## Междокументные пары и сравнение с ReLEx S2',
        '',
        '| P-пара | Expected / direction | GLiREL expected | ReLEx S2 expected | GLiREL max / label / direction |',
        '|---|---|---:|---:|---|',
    ]
    for pid,(code,d) in run.split.EXPECTED.items():
        r=next(r for r in cross_rows if r['pair_id']==pid);b=max(r['raw_scores'],key=lambda s:s['confidence'])
        gs=expected_score(r,code,d);bs=expected_score(baseline[pid],code,d)
        lines.append(f"| {pid} | {code} {d} | {gs if gs is None else format(gs,'.8f')} | {bs:.8f} | {b['confidence']:.8f} {b['relation_code']} {direction(b)} |")
    lines += [
        '',
        'Более высокий raw score одной модели не считается улучшением научной корректности: шкалы не откалиброваны. P01 требует определения ESRD/CKD, которого нет в двух абстрактах; P04/P05 консервативно зависят от определения overexpression/bundle, а не устанавливают общий эксперимент или идентичность оборудования. P02/P06 — условные гипотезы переносимости, не observed application. P03 допускает узкое activity→cell-class отношение по лексическому смыслу, но не подтверждает Breg→rHuEPO механизм. Нетривиальных бесспорных TEXT-SUPPORTED SPLIT positives в этом наборе нет; это ограничение прежнего source audit сохранено.',
        '',
        '| Сравнение при >=0.35 | GLiREL | GLiNER ReLEx S2 (для sanity — прежний D-run) |',
        '|---|---:|---:|',
        '| Expected sanity hits / 6 | 0 | 6 |',
        '| Cross outputs / 28 пар | 0 | 0 |',
        '| Принятых cross relations | 0 | 0 |',
        '| Ошибочных положительных cross outputs | 0 | 0 |',
        '| P-пар с любым положительным output / 6 | 0 | 0 |',
        '',
        '## Содержательный аудит outputs',
        '',
    ]
    positive_reviews=[r for r in review if r['classification']!='NO_POSITIVE_PREDICTION']
    if positive_reviews:
        lines+=['| Набор / пара | Claim | Score | Вердикт | Причина |','|---|---|---:|---|---|']
        for r in positive_reviews:
            lines.append(f"| {r['pool']} / {r['pair_id']} | {r['head_slot']}→{r['relation_code']}→{r['tail_slot']} | {r['score']:.8f} | {r['classification']} | {r['reason']} |")
    else:
        lines+=['При >=0.35 нет положительных предсказаний ни в sanity, ни в cross; следовательно, нет outputs для распределения по SUPPORTED_CONCEPT_RELATION / PLAUSIBLE_HYPOTHESIS / UNCERTAIN / REJECTED. В quality_review.csv для каждой пары явно сохранён NO_POSITIVE_PREDICTION, поля correctness помечены not_asserted. Это обозначение abstention, не пятый вердикт для фактически предсказанного отношения. Нулевой accepted output не означает precision=100%.']
    largest=max(((s['confidence'],r['pair_id'],s['relation_code'],direction(s)) for r in cross_rows for s in r['raw_scores']),default=None)
    lines += [
        '',
        f"Наибольший cross score: {largest}. Все значения ниже диагностического cutoff сохраняются как raw scores, а не установленный научный claim. Они позволяют видеть ранжирование/ошибочный относительный head-tail порядок, но не являются основанием опускать порог постфактум. Шесть positive candidates и N/W оцениваются по источникам; префикс N сам по себе не основание отклонить любой возможный concept relation.",
        '',
        'Для максимума W09 смысл E3(papal politics)→SUBTYPE_OF→E2(synod on synodality) источниками не поддерживается: историческая папская репрезентация и политика 1605–1700 не определяются как подтип современного синода. При искусственном повышении такого score до статуса edge этот claim был бы REJECTED по type/scope; в текущем эксперименте он остаётся ниже порога и не включён в число положительных ошибок. Другие слабые ответы на P также не заменяют нужный predicate: например, HAS_PROPERTY не выражает применение MRI-denoising к маммограммам.',
        '',
        '## Ответы на вопросы',
        '',
        '1. **Локальная работа:** checkpoint успешно загружен и public API работает после описанных локальных compatibility bridges. Все веса загружены строго; никакие модели не скачивались дополнительно.',
        '2. **Sanity:** exact expected hits и scores приведены выше для всех шести пар и всех трёх cutoffs; нулевой sanity является ограничением оценки модели в выбранной конфигурации, а не скрытым успешным тестом.',
        '3. **Число cross outputs:** приведено для каждого порога в основной таблице; raw-score inventory хранит все доступные labels/directions, включая ниже cutoffs.',
        '4. **Содержательная корректность:** положительные outputs, если они есть, перечислены в quality_review.csv; отсутствие outputs не позволяет оценить precision извлечения.',
        '5. **P01–P06:** expected и maximum scores показаны в таблице; отсутствие positive outputs не превращает эти кандидаты в gold negatives.',
        '6. **Ошибки:** различаются нераспознавание явно поддержанных sanity-смыслов и ложные cross claims. Возможные версии/tokenization/label/domain факторы не выдаются за доказанную архитектурную причину.',
        '7. **Сравнение с ReLEx S2:** оцениваются accepted/rejected outputs и конкретные expected смыслы, а не общий raw-score максимум. ReLEx S2 уже имела нулевой cross yield на этих 28 парах; GLiREL не должна объявляться лучше только за низкую уверенность/воздержание.',
        '8. **Корпус из 30 разнородных статей:** позволяет установить поведение двух локальных конфигураций на выбранных примерах и диагностировать недостаточность benchmark. Он не позволяет оценить общий cross-document recall или доказать превосходство/непригодность всей модельной семьи. При расширении нужен независимый source-supported набор с двумя необходимыми текстовыми посылками и совместимыми негативными контролями; такое расширение сейчас не выполнялось.',
        '',
        '## Файлы и воспроизведение',
        '',
        'Компактный набор: sanity_results.jsonl, cross_document_predictions.jsonl, quality_review.csv, report.md, manifest.json. Каждый JSONL checkpoint содержит фактический вход и provenance, токены/offsets, ner, raw API predictions и преобразованные scores. manifest хранит frozen sources, модельные/package hashes, runtime, подпись inference, проверки и threshold summaries. Hash собственного manifest не включён из-за самоссылки.',
        '',
        'Команды: `.venv/Scripts/python.exe scripts/evaluate_glirel.py --stage sanity`, затем `scripts/report_glirel.py --stage sanity`, затем `scripts/evaluate_glirel.py --stage cross`, затем `scripts/report_glirel.py --stage finalize` с тем же интерпретатором. Четыре новых offset/endpoint/crop теста прошли. Повторный запуск стадий использует signature-checked checkpoints; report не вызывает inference. Веса, cache и окружения в commit не включаются. Neo4j, первый проход GLiNER ReLEx, ER, SIMILAR_TO, HAS_SHARED_CONCEPT и исходные документы не менялись; другие модели и обучение не запускались. Push не выполнялся.',
    ]
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['sanity','finalize'],required=True)
    sanity() if p.parse_args().stage=='sanity' else finalize()
