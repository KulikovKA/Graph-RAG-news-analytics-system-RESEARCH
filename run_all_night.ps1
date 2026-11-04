param(
    [ValidateSet('full', 'pilot')][string]$Dataset = 'full',
    [string]$Python = '.\.venv-ner-v2\Scripts\python.exe',
    [string]$Config = 'config/entity_typing_v2.json',
    [string]$Output = '',
    [int]$NumThread = 4,
    [string[]]$ServerUrls = @(),
    [switch]$DryRun,
    [switch]$StrictMaxContext
)
$ErrorActionPreference = 'Stop'
$exitCode = 1
$previousPowerState = [uint32]0
$transcriptStarted = $false
Push-Location -LiteralPath $PSScriptRoot
try {
    $configuration = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
    $runBase = if ($Output) { $Output } else { $configuration.output }
    $runPath = if ([System.IO.Path]::IsPathRooted($runBase)) {
        [System.IO.Path]::GetFullPath($runBase)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $runBase))
    }
    if ($Dataset -eq 'full') { $runPath = Join-Path $runPath 'full' }
    $logDirectory = Join-Path $runPath 'night_runs'
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    $logPath = Join-Path $logDirectory ('powershell_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N') + '.log')
    Start-Transcript -LiteralPath $logPath -Append | Out-Null
    $transcriptStarted = $true
    Write-Host "Night launcher: dataset=$Dataset, threads=$NumThread, dry-run=$DryRun"
    Write-Host "Results: $runPath"
    $workerUrls = if ($ServerUrls.Count -gt 0) { $ServerUrls } else { $configuration.server_urls }
    Write-Host ("Independent Ollama workers: " + ($workerUrls -join ', '))
    Write-Host "PowerShell log: $logPath"
    if (-not $DryRun) {
        # Keep the system active; the display can switch off. No persistent power-plan changes.
        if (-not ('EntityTypingNightPower' -as [type])) {
            Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class EntityTypingNightPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
        }
        $previousPowerState = [EntityTypingNightPower]::SetThreadExecutionState([uint32]2147483649)
        if ($previousPowerState -eq 0) { throw 'Cannot keep Windows awake; no experiment was started' }
        Write-Host 'Windows system sleep is inhibited until this launcher exits.'
    }
    $cliArgs = @('scripts/run_entity_typing_v2.py', 'night', '--dataset', $Dataset, '--config', $Config, '--num-thread', $NumThread)
    if ($Output) { $cliArgs += @('--output', $Output) }
    if ($ServerUrls.Count -gt 0) { $cliArgs += @('--server-urls') + $ServerUrls }
    if ($DryRun) { $cliArgs += '--dry-run' }
    if ($StrictMaxContext) { $cliArgs += '--strict-max-context' }
    # Remain attached to the foreground child; no jobs or confirmation prompts.
    & $Python -X utf8 -u @cliArgs
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        if ($DryRun) { Write-Host 'Dry-run passed; no model load or inference was performed.' }
        else { Write-Host 'Night experiment complete; all candidate results validated and ZIP exported.' }
    } else {
        Write-Host "Night launcher returned $exitCode. Inspect completeness.json and night_runs logs; rerun the same command to resume."
    }
} catch {
    Write-Host ("Launcher failed: " + $_.Exception.ToString())
    $exitCode = 1
} finally {
    if ($previousPowerState -ne 0) {
        [void][EntityTypingNightPower]::SetThreadExecutionState($previousPowerState)
    }
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
    Pop-Location
}
exit $exitCode
