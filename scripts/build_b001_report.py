"""Build the initial report from already-scored aggregates, never from Gold files."""
from pathlib import Path
from cgraph.common import read_json

metrics = read_json('reports/b001/metrics.json')
baseline = read_json('reports/b001/c_diag_002_control_baseline.json')
b, c = metrics['micro'], baseline['micro']
perf, cp = metrics['performance'], baseline['performance']


def fmt(v):
    return '—' if v is None else f'{v:.4f}'


def score(v):
    return '/'.join(fmt(v[k]) for k in ('precision', 'recall', 'f1'))


def counts(v):
    return '/'.join(str(v[k]) for k in ('tp', 'fp', 'fn'))


lines = [
    '# B-001 — Improved Source-Grounded Hybrid Graph Extraction',
    '',
    'Exploratory DEV: те же 12 уже исследованных документов, что в финальном C-DIAG-002. '
    'Это не независимый финальный тест. D024 имеет неполную relation-разметку: '
    'unmatched prediction не обязательно семантически ложно. Gold использован только offline evaluator; '
    'CALIBRATION/HOLDOUT не открывались, обучения и fine-tuning нет.',
    '',
    'Источник, настройки, диагностические технические перезапуски и ограничения: [протокол](B001_PROTOCOL.md). '
    'Модель `qwen3.5:4b-q4_K_M`, digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`; '
    'Ollama 0.35.0, CPU, seed 42, temperature 0, top-k 1, repeat penalty 1.1/256, '
    'context 8192, max output 4096. Исходная ontology v1.0 и её 12 kinds/23 relations неизменны; NLI не используется.',
    '',
    '## Качество по общему C evaluator',
    '',
    'Точное совпадение Unicode-span и типа, направленное отношение, one-to-one multiset matching; '
    'P/R/F1 и TP/FP/FN. Raw по общему evaluator включает предложения с привязанными endpoints и evidence. '
    'Все результаты без независимой проверки научного смысла.',
    '',
    '| Метрика | B P/R/F1 | B TP/FP/FN | C P/R/F1 | C TP/FP/FN |',
    '|---|---|---|---|---|',
]
for key, title in [('entity_typed','Typed entities'), ('relation_raw_end_to_end','Raw relations'),
                   ('relation_end_to_end','Final relations'), ('qualified_relation','Qualified relations')]:
    lines.append(f'| {title} | {score(b[key])} | {counts(b[key])} | {score(c[key])} | {counts(c[key])} |')
lines += [
    '',
    f"Candidate recall при одинаковом C single-sentence generator: B {b['candidate_pair_recall']['covered']}/{b['candidate_pair_recall']['gold']} = {fmt(b['candidate_pair_recall']['recall'])}; "
    f"C {c['candidate_pair_recall']['covered']}/{c['candidate_pair_recall']['gold']} = {fmt(c['candidate_pair_recall']['recall'])}. "
    'Это diagnostic endpoint coverage, а не качество отношений; candidate pairs не ограничивают B extraction.',
    '',
    f"Raw-all, где неразрешимые relation proposals дополнительно считаются FP: {score(metrics['raw_all_proposals'])}, "
    f"TP/FP/FN {counts(metrics['raw_all_proposals'])}; таких proposals {metrics['raw_all_proposals']['unmappable_counted_as_fp']}. "
    'Необработанные из-за усечённого JSON окна отдельно учитываются в coverage; их количество предложений неизвестно.',
    '',
    f"Явный model ABSTAIN: {metrics['model_abstention']['abstained']}/{metrics['model_abstention']['structurally_validated_relation_records']} "
    f"= {fmt(metrics['model_abstention']['rate'])} от структурно разобранных relation records. "
    f"Pair abstention по диагностике C: B {b['abstention']['abstained']}/{b['abstention']['candidate_pairs']} = {fmt(b['abstention']['rate'])}; "
    f"C {c['abstention']['abstained']}/{c['abstention']['candidate_pairs']} = {fmt(c['abstention']['rate'])}. "
    'Эти знаменатели различаются: отсутствие принятого отношения на паре не обязательно является явным решением LLM.',
    '',
    'Qualifier completeness всегда UNKNOWN: без подтверждения полноты строгий qualified scorer не присваивает TP. '
    'Числа и условия, не привязанные к конкретной связи, сохраняются как UNKNOWN/UNSUPPORTED. '
    'Точная цитата подтверждает provenance, но не семантическую поддержку.',
    '',
    '## Ошибки по всем отношениям',
    '',
    'Final TP/FP/FN относительно annotations; C — тот же GLiNER control из C-DIAG-002. '
    'Нулевой Gold support не доказывает capability.',
    '',
    '| Relation | B TP/FP/FN | C TP/FP/FN |',
    '|---|---|---|',
]
for relation, v in metrics['per_relation'].items():
    lines.append(f"| {relation} | {counts(v['final'])} | {counts(baseline['per_relation'][relation]['final'])} |")
lines += [
    '',
    '## Стоимость и покрытие',
    '',
    f"B: wall {perf['cost']['wall_seconds']:.1f} s; CPU {perf['cost']['sampled_cpu_seconds']:.1f} s; "
    f"peak sampled aggregate RSS {perf['cost']['peak_sampled_rss_bytes'] / 2**30:.3f} GiB; "
    f"LLM calls {perf['llm_calls']}, input tokens {perf['input_tokens']}, output tokens {perf['output_tokens']}, "
    f"retries {perf['retries']}, JSON/truncation errors {perf['json_errors']}, request errors {perf['request_errors']}. "
    f"Успешные окна {perf['windows_successful']}/{perf['windows_total']}; "
    f"failed {len(perf['failed_windows'])}.",
    '',
    f"C control: wall {cp['wall_ms']/1000:.1f} s; CPU {cp['cpu_ms']/1000:.1f} s; "
    f"peak process RSS {cp['peak_ram_bytes']/2**30:.3f} GiB; LLM calls 0. "
    'B включает Python client, Ollama server и его потомков, суммируя RSS с шагом 0.25 s; '
    'C — peak одного процесса. Общий сервер и совместно отображённые страницы ограничивают прямую сравнимость RAM и CPU. '
    'Первичные измерения исключают предварительные пробы, загрузки, тесты и evaluator; '
    'они не являются полным временем исследования.',
    '',
    f"Причины rejection: `{metrics['rejection_reasons']}`.",
    '',
    '## Примеры',
    '',
    'Ниже exact-сопоставление с owner-reviewed DEV annotations; несовпадение с Gold '
    'не является независимым доказательством семантической ошибки.',
    '',
]
for name, title in [('matched', 'Совпадения с Gold'), ('unmatched', 'Unmatched к Gold')]:
    lines.extend([f'**{title}**', ''])
    for x in metrics['examples'][name][:5]:
        lines.append(f"- {x['document']}: `{x['head']}` ({x['head_kind']}) → **{x['relation']}** → "
                     f"`{x['tail']}` ({x['tail_kind']}), {x['polarity']}/{x['modality']}. "
                     f"Evidence: “{x['evidence']}”")
    if not metrics['examples'][name]:
        lines.append('- Нет примеров в этом запуске.')
    lines.append('')
lines += [
    '## Ограничения и проверка',
    '',
    'Один владелец проверял AI-origin разметку; независимой второй разметки нет. '
    'Окна ограничены одним предложением, поэтому межпредложенческие связи и coreference не оцениваются как поддержанные. '
    'Repeat penalty выбран до Gold по техническому повторению на одном DEV-окне, и DEV уже исследован; '
    'статистическая независимость отсутствует. Потери boundary/type и unsupported qualifiers сохраняются в итоговых метриках. '
    'Никакие результаты B не подменяются модельными или синтетическими примерами.',
    '',
    'Воспроизведение и границы доступа приведены в [протоколе](B001_PROTOCOL.md); '
    'полные агрегаты и per-relation counts — [metrics.json](b001/metrics.json), '
    'фиксированный C baseline — [c_diag_002_control_baseline.json](b001/c_diag_002_control_baseline.json). '
    'Локальные raw traces, predictions и источники не коммитятся.',
    '',
]
Path('reports/B001_REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
