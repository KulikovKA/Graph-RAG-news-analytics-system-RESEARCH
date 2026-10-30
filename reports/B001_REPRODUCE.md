# B-001 reproduction

Primary source/config freeze: `5a90136` and `144d63e`, base `b368a63`, branch `b-001-source-grounded-hybrid`. The primary run is exploratory on the same 12 DEV as C-DIAG-002. Do not substitute a newer model, ontology or split and call the result B-001.

Use Python 3.11, the existing `uv.lock`, and Ollama **0.35.0** with local `qwen3.5:4b-q4_K_M` manifest digest `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`. All required model blob hashes are in `contracts/b001_model.json`. A mutable model tag alone is insufficient: the extraction command verifies both the API digest and local blobs and refuses a mismatch. The tested model is the same one already installed for the architecture-A plan. The backend currently expects the standard local Ollama blob directory under the user's `.ollama/models/blobs` and the loopback endpoint at port 11434.

From the new worktree/repository root:

```powershell
uv sync --frozen --extra test
uv run --no-sync cgraph fetch-dev
uv run --no-sync python -m bgraph.cli extract --output artifacts/b001/dev_primary
uv run --no-sync python -m bgraph.cli evaluate --predictions artifacts/b001/dev_primary --output reports/b001/metrics.json
uv run --no-sync python scripts/analyze_b001_evaluation.py --predictions artifacts/b001/dev_primary
uv run --no-sync python -m pytest -q
```

Use a new/empty extraction output directory; recorded runs are not overwritten. `fetch-dev` uses the existing pinned upstream manifest and only DEV inputs/labels. Inference is a separate process whose audit hook denies Gold, original ontology examples, CALIBRATION/HOLDOUT paths, fitted RE modules and non-loopback network connections. The labels are only opened by the offline evaluation commands. For the measured local run, the verified DEV inputs were copied from the existing C-RESTART worktree; the ignored `data/labels` junction was created only after inference for offline evaluation. Neither source worktree was modified.

The measured commands used the existing C-RESTART Python 3.11.9 environment with `PYTHONPATH=src`; module paths were checked to resolve to this B worktree. Pytest: 46 passed, 1 historical training test deselected. No training test was run. The primary run is stored locally in `artifacts/b001/dev_primary`, raw per-window responses in its `raw/` subdirectory, and progress in `artifacts/b001/primary_inference.log`. The manifest and evaluator verified all 12 prediction files and 78 raw trace files. The one D078 length failure is retained, not repaired into a fictitious complete output.

The PowerShell redirection used during measurement yielded native-stderr formatting and a shell exit status of 1 despite complete artifacts and no traceback. Completion is established by the final manifest and successful offline integrity/scoring checks, not a claimed clean shell exit. For a new logged run, capture stdout/stderr inside a Python `subprocess.run` wrapper and preserve its child return code, rather than relying on PowerShell's formatting of native stderr.

The additional analysis script is an **offline evaluator supplement**, not part of extraction. It reads Gold only after predictions and computes entity kinds, FP categories, FN funnel and examples. It never updates prompts, settings or accepted assertions. `scripts/build_b001_report.py` rebuilds the curated report from the archived B-001 aggregates and intentionally refuses different counts; a new run with different outputs needs a separately reviewed report. Full inputs, Gold, weights and predictions remain ignored; the published `reports/b001/run.json` contains the immutable hashes needed to audit them.
