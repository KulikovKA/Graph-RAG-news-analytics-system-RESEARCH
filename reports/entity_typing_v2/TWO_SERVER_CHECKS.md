# Two independent Qwen servers: implementation checks

Date: 2026-11-05, Europe/Moscow. Baseline: 97f73fc.

The user selected two independent Ollama servers after reviewing the qwen35 single-slot restriction in Ollama v0.35.0 (server/sched.go lines 481-485). Model and checkpoint were not substituted.

- Default endpoints: http://localhost:11434 and http://127.0.0.1:11435. Each has one compute slot and an independent model/context. Thinking OFF and ON remain separate sequential stages.
- New output root: reports/entity_typing_v2/run2_two_servers/. The pre-existing run1 attempt is preserved; its frozen runtime/results were not modified or migrated.
- Main suite: 60 tests passed in 100.83 seconds (two_server_tests.xml).
- Final focused server suite: 13 tests passed in 17.89 seconds (two_server_focused_tests.xml), including the added guard against freezing an allocated context too small for full prompts.
- Mocks verify actual two-request overlap using a barrier, distinct per-attempt server provenance, complete offline resume, checkpoint mismatch rejection, one-worker failures, cancellation of the peer worker on Ctrl+C and recovery of a response received during cancellation.
- Process tests use mocked Popen: reuse an existing server, start only a missing server with OLLAMA_NUM_PARALLEL=1, hide its Windows console and stop only owned processes.
- Windows simultaneous journal read/replace sharing failure found by a test was fixed with synchronized file reads/writes and bounded replacement retries. A dedicated concurrent read/write regression passed.
- HTTP cancellation is tested without a real connection, including a Connection-close response whose body retains the underlying socket.
- All HTTP/network connections are blocked in unit tests. No real model inference or server startup was used for tests.
- Both PowerShell scripts passed AST syntax validation. git diff --check passed; CLI imports no torch, GLiNER, transformers or spaCy.
- Actual root launcher -DryRun passed: full 1211 candidates, 169 batches, estimated num_ctx 16384 PER INSTANCE. This is two independent windows, not a shared context.
- Dry-run: inference calls 0, model load calls 0, server start calls 0. The first server was available; the absent second server used explicitly marked metadata-only fallback (ready=false). Runtime allocation/startup of the second instance remains for the user's launch.
- RAM selection sums two full model sizes, two f16 KV caches and two runtime reserves. Reclaimable loaded memory is counted once per observed endpoint, including when fallback metadata is reused. Physical/native context is checked on both servers before typing; no text is truncated.
- No new T21/T22 result files or real attempt journals were created by this development work.

Launch/resume: .\run_all_night.ps1
Metadata-only preflight: .\run_all_night.ps1 -DryRun
Results: reports/entity_typing_v2/run2_two_servers/full/
Experiment/T1 log: full/night_runs/ (START T1 through END T1)
Owned server log: full/server_logs/
ZIP: reports/entity_typing_v2/run2_two_servers/full_results.zip

Future real startup reuses existing servers, starts only missing local endpoints, checks identical checkpoint/tag/quantization/version on both, records worker address and timings per attempt, and verifies selected num_ctx via both /api/ps endpoints. When resources or checkpoint consistency fail it does not silently shrink a frozen context or claim a complete experiment. Per-model num_thread remains 4 (up to 8 compute threads total); CPU speedup is not asserted without a real experiment.
