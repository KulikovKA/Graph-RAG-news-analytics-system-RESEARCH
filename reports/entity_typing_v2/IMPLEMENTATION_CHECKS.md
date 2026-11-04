# ENTITY-TYPING-V2-001 implementation checks

Date: 2026-11-05 (Europe/Moscow). Baseline: 99cc18f, branch design-v2-sequential-kg.

- New unit tests: 39 passed in 46.01 seconds (technical_tests.xml). Network access is blocked in the test module; all model responses are mocks.
- Existing NER/N3 regression tests: 60 passed in 15.43 seconds.
- Saved E3 loading: 30 documents, 1327 retained hints, 1211 source-exact spans; saved N3 reused.
- Frozen pilot: 240 candidates, seed 4401, discipline/type strata with score/ambiguity coverage.
- Actual offline commands checked: prepare, gliner, summarize, export-results; PowerShell launcher checked with -DryRun.
- Actual metadata-only dry-run: model available, 45 batches, selected estimated num_ctx=131072; runtime allocation NOT tested.
- Discovered model: qwen3.5:4b-q4_K_M, Q4_K_M, digest 2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd; Ollama 0.35.0.
- Native checkpoint context: 262144. RAM selection uses f16 KV metadata and explicit runtime reserve; dry-run selection can change with free RAM.
- Dry-run generation calls: 0; model load calls: 0. No T21/T22 output files or real Qwen attempt journals were created.
- ZIP CRC and all 32 archived file checksums verified; model weights excluded.
- CLI import isolation verified: torch, GLiNER, transformers and spaCy are not imported.
- git diff --check passed. Historical NER/N3 reports/code and Ontology v2 unchanged.

No real Qwen generation, model download, GLiNER inference, full benchmark, relation extraction, semantic verification or entity resolution was performed. Real runtime context allocation, effective inference timings and ON/OFF results remain for the user's authorized local experiment. Public Ollama API cannot echo effective sampling settings; this limitation is explicit in diagnostics and README.
