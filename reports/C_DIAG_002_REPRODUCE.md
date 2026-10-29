# C-DIAG-002 reproduction

Из корня отдельного worktree, Python 3.11; измеренный runtime — native Windows.
Нужны только 12 DEV inputs/labels и pretrained weights. `data/`/`artifacts/`
исключены из Git. Для каждого повторения нужны новые/пустые output-каталоги.

```powershell
uv sync --frozen --extra test
uv run --no-sync cgraph fetch-dev
uv run --no-sync cgraph-zero download-models
uv run --no-sync python scripts/prepare_c_diag_002.py
uv run --no-sync python -m pytest -q
uv run --no-sync python scripts/run_c_diag_002.py --variant chem --output artifacts/c_diag_002/chem
uv run --no-sync python scripts/run_c_diag_002.py --variant battery --output artifacts/c_diag_002/battery
uv run --no-sync python scripts/run_c_diag_002.py --variant control --output artifacts/c_diag_002/control
uv run --no-sync python scripts/run_c_diag_002.py --variant union --output artifacts/c_diag_002/union
uv run --no-sync python scripts/export_c_diag_002_entities.py
uv run --no-sync python scripts/run_c_diag_002.py --variant gold --output artifacts/c_diag_002/gold
uv run --no-sync python scripts/prepare_c_diag_002_ner_inventories.py
uv run --no-sync python scripts/benchmark_c_diag_002_ner.py --variant control
uv run --no-sync python scripts/benchmark_c_diag_002_ner.py --variant chem
uv run --no-sync python scripts/benchmark_c_diag_002_ner.py --variant battery
uv run --no-sync python scripts/benchmark_c_diag_002_ner.py --variant union
uv run --no-sync python scripts/evaluate_c_diag_002.py
uv run --no-sync python scripts/build_c_diag_002_report.py
```

Gold exporter/`--variant gold` — **только диагностический режим**. Source-only
прогоны не могут читать даже этот entity-only export. Не запускать evaluator,
пока не сохранены все пять immutable runs. Он проверяет prediction/config/code
hashes, audits и воспроизведение control C-RESTART-001 перед итоговым comparison.
32 GiB budget проверяется по peak working set; на измеренной машине все прогоны
значительно ниже лимита. Scoring допускает только DEV через прежний loader.

Измеренные runs использовали тот же неизменный locked environment, уже находящийся
в `../Graph-RAG-C-RESTART-001/.venv`, и явно установленный `PYTHONPATH=<new-worktree>/src`.
Pinned assets были hardlinked в новый worktree, только 12 allowlisted DEV files
скопированы по явным именам. Этот способ не нужен для чистого воспроизведения.

Для повторения GLiREL runtime control на чистой машине сначала нужен его pinned
asset bundle. Это **не** команда основного NLI pipeline:

```powershell
uv run --no-sync cgraph-zero --config config/c_restart_001_glirel.json download-models
uv run --no-sync python scripts/prepare_c_diag_002_glirel_runtime.py artifacts/c_diag_002/backbone_export
uv run --no-sync python scripts/probe_c_diag_002_glirel.py artifacts/c_diag_002/backbone_export
```

Подготовка лишь сериализует encoder из строго загруженного полного checkpoint.
Измеренный повторный probe использовал byte-identical готовый backbone export от
C-RESTART-001 и официальный loader с local path; менялся только `model_name` в
локальной runtime config, чтобы не скачивать mutable backbone. Probe запускался
через subprocess с `timeout=600`; после FAIL не было дальнейших попыток и DEV
прогонов GLiREL. Точные runtime/file hashes и output — в `glirel_probe.json`.
Описанная helper preparation добавлена для воспроизведения без старого worktree;
во время измеренной повторной попытки она не запускалась.

Одна неудачная попытка chem завершилась до первого prediction из-за счётчика
encoder calls (`KeyError: input_ids`). Исправлен только measurement hook,
учитывающий positional inputs; модель, правила и protocol не менялись. Лог:
локальный `artifacts/c_diag_002/chem_instrumentation_failure.log`. Эта попытка,
downloads, GLiREL probe, tests и evaluator не входят во времена основных runs.

Сохранённый contract diagnostic test читает исходную ontology только для проверки
её hash и равенства нормативных полей example-free runtime projection. Поля с
Gold examples не используются в моделях или mapping. DEV label-файлы исторический
training-тест не читает, потому что он исключён marker `historical_training`;
он не выполнялся. Обычный inference не может открыть исходную ontology или labels.
