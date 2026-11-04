param(
    [ValidateSet('full', 'pilot')][string]$Dataset = 'full',
    [string]$Python = '.\.venv-ner-v2\Scripts\python.exe',
    [int]$NumThread = 4,
    [switch]$DryRun,
    [switch]$StrictMaxContext
)
# Compatibility entry point; the night launcher owns logging, wakefulness and resume.
$launcher = Join-Path (Split-Path -Parent $PSScriptRoot) 'run_all_night.ps1'
& $launcher @PSBoundParameters
exit $LASTEXITCODE
