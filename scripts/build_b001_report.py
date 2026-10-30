"""Build the initial report from already-scored aggregates, never from Gold files."""
from pathlib import Path
from cgraph.common import read_json

metrics = read_json('reports/b001/metrics.json')
baseline = read_json('reports/b001/c_diag_002_control_baseline.json')
analysis = read_json('reports/b001/error_analysis.json')
b, c = metrics['micro'], baseline['micro']
perf, cp = metrics['performance'], baseline['performance']
# Curated comments and the loss discussion describe the archived B-001 run.
# Refuse to silently publish those fixed observations for another experiment.
assert b['relation_end_to_end']['tp'] == 16 and b['relation_end_to_end']['predicted'] == 64
assert analysis['correct_gold_edges_remaining'] == dict(gold=390, untyped_endpoints=72, typed_endpoints=51, candidate_pairs=50, raw=16, final=16)


def fmt(v):
    return '—' if v is None else f'{v:.4f}'


def score(v):
    return '/'.join(fmt(v[k]) for k in ('precision', 'recall', 'f1'))


def counts(v):
    return '/'.join(str(v[k]) for k in ('tp', 'fp', 'fn'))


lines = [
    '# B-001 — Improved Source-Grounded Hybrid Graph Extraction',
    '',
    'Реальный end-to-end запуск завершён на 12 DEV, 78 LLM-вызовов. **Текущий B-001 не пригоден для '
    'автоматической публикации научного графа**: final precision 25%, recall 4.10%, всего 16 TP. '
    'Relation F1 выше C control, но typed entity F1, candidate recall и число найденных правильных связей ниже; '
    'wall time выше примерно в 22.8 раза. Это не доказательство общего превосходства архитектуры B.',
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
    f"До mapping: {analysis['counts']['llm_explicit_abstain']}/{analysis['counts']['llm_relation_records']} явных model ABSTAIN. "
    f"Pair abstention по диагностике C: B {b['abstention']['abstained']}/{b['abstention']['candidate_pairs']} = {fmt(b['abstention']['rate'])}; "
    f"C {c['abstention']['abstained']}/{c['abstention']['candidate_pairs']} = {fmt(c['abstention']['rate'])}. "
    'Эти знаменатели различаются: отсутствие принятого отношения на паре не обязательно является явным решением LLM.',
    '',
    'Qualifier completeness всегда UNKNOWN: без подтверждения полноты строгий qualified scorer не присваивает TP. '
    'Числа и условия, не привязанные к конкретной связи, сохраняются как UNKNOWN/UNSUPPORTED. '
    f"В данном запуске принято **{analysis['counts']['accepted_qualifiers']} qualifiers**, сохранено "
    f"{analysis['counts']['unresolved_qualifiers']} unresolved records. Полное qualified extraction не реализовано как доказанная capability. "
    'Точная цитата подтверждает provenance, но не семантическую поддержку.',
    '',
    '## Стадии и потери recall',
    '',
    f"Raw parsed LLM: {analysis['counts']['llm_entity_records']} entity records и {analysis['counts']['llm_relation_records']} relation records. "
    f"Exact mapped entities: {analysis['counts']['entities']} (включая новые endpoints из relations); "
    f"структурно адресуемые relations: {analysis['counts']['structural_records']}; final accepted: {analysis['counts']['accepted']}. "
    f"Отклонено {analysis['rejected_stages']['entity_mapping']} entity records на mapping, "
    f"{analysis['rejected_stages']['relation_mapping']} relation records на mapping и "
    f"{analysis['rejected_stages']['ontology_scope']} relations на ontology/scope. "
    'Raw ответы, структурные решения, accepted, rejected и unresolved хранятся отдельно.',
    '',
    'Точные Gold edges, остающиеся доступными: **390 → 72** untyped endpoints → **51** typed endpoints '
    '→ **50** candidate pairs → **16** raw → **16** final. Последовательные FN: '
    '318 boundary/mapping/failed-window losses + 21 kind losses + 1 candidate/context loss + 34 model losses + 0 final-gate losses = 374. '
    'Final validation удалил 119 unmatched-to-Gold proposals и не удалил ни один из 16 raw TP; '
    'это accounting по annotation, а не доказательство семантической ложности каждого удалённого proposal. '
    'Рост precision не компенсирует низкий upstream recall.',
    '',
    f"Final FP по первой применимой категории: {analysis['final_fp_categories']}. "
    'Ошибки границ включают усечённые химические названия и слишком короткие носители свойств. '
    'Одна потеря candidate context — межпредложенческий HAS_PERFORMANCE в D026.',
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
    'Первичные измерения включают локальную загрузку модели, но исключают предварительные пробы, скачивание weights, тесты и evaluator; '
    'они не являются полным временем исследования.',
    '',
    'Оборудование: AMD Ryzen 7 8845HS, 8 физических/16 логических CPU, 32 GB RAM; Python 3.11.9, '
    'четыре inference threads, GPU offload 0. Основной inference выполнен 30 октября 2026, около '
    '12:30–14:46 Europe/Moscow. Input tokens включают кешируемый системный prompt; '
    'это не число заново вычисленных prompt tokens. Одно failed окно D078 (sentence 4 первого chunk) '
    'достигло 4096 output tokens, его partial JSON не принят, все Gold relations оставлены в знаменателе. '
    'PowerShell-обёртка сообщила exit 1 при progress в native stderr; traceback в логе отсутствует, '
    'все 12 файлов и финальный manifest записаны и затем успешно проверены offline evaluator. '
    'Чистый exit оболочки не заявляется как свидетельство завершения.',
    '',
    'Модель использовалась локально; внешней API-тарификации не было. Энергопотребление и денежная стоимость '
    'локальной машины не измерялись. Peak RSS — sampled aggregate, не точный общий объём уникальных физических страниц.',
    '',
    f"Причины rejection: `{metrics['rejection_reasons']}`.",
    '',
    '## Примеры',
    '',
    'Ниже exact-сопоставление с owner-reviewed DEV annotations; несовпадение с Gold '
    'не является независимым доказательством семантической ошибки.',
    '',
]
examples = analysis['final_examples']
choices = [
    ('D001','USES_MATERIAL','gas sensor','cotton/elastic threads',
     'Корректная базовая связь: текст прямо описывает материал устройства; exact typed directed Gold TP. Полнота qualifiers и modality этим совпадением не подтверждены.'),
    ('D026','DETECTS','Ti3C2Tx MXene gas sensors','VOC gases',
     'Корректная базовая detection-связь и Gold TP, но указанные в тексте 50–100 ppb и room temperature не сохранены как accepted qualifiers. Это пример неполного qualified assertion.'),
    ('D010','HAS_MEMBER','Ti','C',
     'Семантически неверное утверждение при точной цитате: Ti и C вырваны из химической записи, не заданы как analyte collection и member. Ошибка chemical span + contextual kind + predicate.'),
    ('D010','HAS_MEMBER','GO','CuO',
     'Семантически неподдержанная group-связь: формула перечисляет компоненты nanocomposite, но не утверждает, что GO является коллекцией с member CuO. Domain/range и exact evidence сами этого не обнаружили.'),
    ('D092','HAS_MEMBER','CO','gas molecules',
     'Неверное направление: source перечисляет CO среди gas molecules; модель направила HAS_MEMBER от отдельного газа к группе. Типы формально допустимы, смысл — нет.'),
]
for doc,rel,h,t,comment in choices:
    x = next(x for x in examples if (x['document'],x['relation'],x['head'],x['tail'])==(doc,rel,h,t))
    lines += [f"**{doc}: `{h}` → {rel} → `{t}`**", '', comment, '', f"> {x['evidence']}", '']
lines += ['**Реальные FN**', '']
for stage,x in analysis['fn_examples'].items():
    lines.append(f"- {stage}: {x['document']} `{x['head']}` ({x['head_kind']}) → {x['relation']} → "
                 f"`{x['tail']}` ({x['tail_kind']}).")
lines += ['', 'Например, `E-textiles` → HAS_PROPERTY → `low cost` потерян из-за kind, '
          'а `fiber gas sensor` → HAS_PROPERTY → `conductive` — на model stage при доступных typed endpoints. '
          'Эти примеры выбраны после scoring для объяснения ошибок и не использованы для изменения prompt или правил.', '']
lines += [
    '## Ограничения и проверка',
    '',
    'Один владелец проверял AI-origin разметку; независимой второй разметки нет. '
    'Окна ограничены одним предложением, поэтому межпредложенческие связи и coreference не оцениваются как поддержанные. '
    'Repeat penalty выбран до Gold по техническому повторению на одном DEV-окне, и DEV уже исследован; '
    'статистическая независимость отсутствует. Потери boundary/type и unsupported qualifiers сохраняются в итоговых метриках. '
    'Никакие результаты B не подменяются модельными или синтетическими примерами.',
    '',
    '`python -m pytest -q`: **46 passed, 1 deselected**, 3.75 s. Исключён сохранённый historical_training test; '
    'обучение не запускалось. Проверены repeated/nested Unicode spans, chemical NO и отрицание, '
    'направление/ontology, unsupported qualifiers, новые exact endpoints, Gold/network access guard, '
    'model digest mismatch и явное сохранение усечённой генерации как failure.',
    '',
    'Нормативные contracts, весь `src/cgraph/`, main и C-DIAG-002 сохранены без изменений. '
    'CLI: `python -m bgraph.cli extract --output <new-directory>` и отдельный '
    '`python -m bgraph.cli evaluate --predictions <directory> --output <metrics.json>`. '
    'Аудит hashes проверил исходные predictions и raw traces; source-only access guard запрещал Gold/closed paths '
    'и разрешал только loopback Ollama. Синтетические тесты не входят в DEV metrics. '
    'Файл metrics содержит старую общую формулировку про wider windows; фактический primary config — '
    'одно предложение, что уточнено в error_analysis.json и данном отчёте.',
    '',
    'Primary decoding/config закреплены до Gold в `144d63e`, exact-surface endpoint schema/rules — в `5a90136`; '
    'base worktree — `b368a63`, ветка `b-001-source-grounded-hybrid`. '
    'После scoring менялись только отчёты, offline анализ и дополнительные runtime-тесты; '
    'frozen extraction, prompt и predictions сохранены.',
    '',
    'Воспроизведение и границы доступа приведены в [протоколе](B001_PROTOCOL.md); '
    'полные агрегаты и per-relation counts — [metrics.json](b001/metrics.json), '
    'фиксированный C baseline — [c_diag_002_control_baseline.json](b001/c_diag_002_control_baseline.json). '
    'Run manifest с hashes — [run.json](b001/run.json); разложение потерь и примеры — '
    '[error_analysis.json](b001/error_analysis.json). Локальные raw traces, predictions и источники не коммитятся.',
    '',
    'Документация runtime: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B), '
    '[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs), '
    '[Ollama generate API](https://docs.ollama.com/api/generate). Реальные параметры и blob hashes '
    'из локального запуска имеют приоритет над advertised maximum context.',
    '',
]
Path('reports/B001_REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
