# Night launcher implementation checks

Date: 2026-11-05, Europe/Moscow. Baseline: a3e780d, branch design-v2-sequential-kg.

- Default dataset is full in config, Python CLI and both PowerShell launchers. Pilot remains an explicit quick-check option.
- PowerShell AST validation passed for run_all_night.ps1 and scripts/run_entity_typing_v2.ps1.
- Main suite: 48 tests passed in 91.10 seconds; all HTTP is forbidden in tests and all model outputs are mocks (night_technical_tests.xml).
- Final archive checks: 2 focused tests passed in 20.59 seconds, including completed manifest status and all ZIP SHA-256 checksums (night_export_tests.xml).
- Scenarios: sequential stages; completed OFF skipped before pending ON; completed modes skipped without metadata/API access; one irrecoverable candidate does not stop others; OFF failures do not stop ON; unavailable server still produces explicit pending rows and partial ZIP; Ctrl+C finalizes offline outputs; duplicate/missing IDs and stale hybrid inputs are rejected.
- Actual root launcher command checked: .\run_all_night.ps1 -DryRun. Result: exit 0, full dataset 1211, planned batches 165, model available True, estimated num_ctx 131072.
- Model: qwen3.5:4b-q4_K_M, Q4_K_M, Ollama 0.35.0, native context 262144.
- Dry-run model loading calls: 0. Dry-run generation calls: 0. Runtime allocation was not performed. Actual selected context is verified only by the user's real run.
- Completeness CLI before experiment correctly returned PARTIAL/nonzero; it checked 19 method/policy entries against all 1211 IDs and did not invent Qwen results.
- Full dry-run artifacts and technical logs remain locally in reports/entity_typing_v2/run1/full/ (ignored generated data). No real T21/T22 outputs or attempt journals were created.
- Git whitespace checks passed. Existing Ontology v2, NER/N3 source, architecture draft and historical NER reports were not changed.

Only source code, mocks, offline checks and metadata-only dry-run were executed. Neither a night experiment nor any real model generation/loading was started. The Windows sleep-inhibition branch is syntax-checked but deliberately not exercised by -DryRun. The normal launcher will keep the foreground process attached, request temporary system activity, restore it on exit and write PowerShell/Python technical logs.

User launch: .\run_all_night.ps1
Resume: repeat the same command.
Results: reports/entity_typing_v2/run1/full/
ZIP: reports/entity_typing_v2/run1/full_results.zip
Success exit 0 requires complete predictions for every frozen ID across T1, T21, T22 and all 16 hybrid policies, plus successful stages and export. Explicit NOT_PROCESSED rows preserve coverage but yield PARTIAL/exit 3. Ctrl+C yields exit 130 with best-effort offline finalization.
