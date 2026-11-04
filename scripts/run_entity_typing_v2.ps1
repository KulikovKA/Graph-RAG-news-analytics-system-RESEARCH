param(
    [ValidateSet('pilot', 'full')][string]$Dataset = 'pilot',
    [string]$Python = '.\.venv-ner-v2\Scripts\python.exe',
    [int]$NumThread = 4,
    [switch]$DryRun,
    [switch]$StrictMaxContext
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoPath
try {
    $commonArgs = @('--dataset', $Dataset, '--num-thread', $NumThread)
    if ($StrictMaxContext) { $commonArgs += '--strict-max-context' }
    & $Python scripts/run_entity_typing_v2.py prepare @commonArgs
    if ($LASTEXITCODE -ne 0) { throw 'Prepare failed' }
    & $Python scripts/run_entity_typing_v2.py qwen --dry-run @commonArgs
    if ($LASTEXITCODE -ne 0) { throw 'Dry-run failed; inspect diagnostics' }
    if ($DryRun) { return }
    & $Python scripts/run_entity_typing_v2.py gliner @commonArgs
    if ($LASTEXITCODE -ne 0) { throw 'T1 failed' }
    & $Python scripts/run_entity_typing_v2.py qwen --mode no-thinking --resume @commonArgs
    if ($LASTEXITCODE -ne 0) { throw 'T2.1 incomplete; rerun launcher to resume' }
    & $Python scripts/run_entity_typing_v2.py qwen --mode thinking --resume @commonArgs
    if ($LASTEXITCODE -ne 0) { throw 'T2.2 incomplete; rerun launcher to resume' }
    & $Python scripts/run_entity_typing_v2.py hybrid @commonArgs
    if ($LASTEXITCODE -ne 0) { throw 'T3 failed' }
    & $Python scripts/run_entity_typing_v2.py summarize @commonArgs
    if ($LASTEXITCODE -ne 0) { throw 'Summary failed' }
} finally {
    Pop-Location
}
