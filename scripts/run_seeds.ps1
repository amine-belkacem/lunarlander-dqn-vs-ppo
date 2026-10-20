<#
.SYNOPSIS
Train and evaluate several (algorithm, seed) runs, a few at a time.

.DESCRIPTION
Measured on my laptop (Ryzen 3 7320U, 4 cores, 7.3 GB RAM): one torch thread
per run is twice as fast as four, and three DQN runs in parallel each keep
about 139 steps/s with a peak of ~520 MB committed memory. Default: 3 runs in
parallel with 1 thread each, leaving one core free.

- Runs whose meta.json already says "done" are skipped, so the script can be
  stopped and started again.
- DQN runs are queued first because they take about 10x longer than PPO.
- Starts are staggered: two Python processes importing torch at the same
  instant failed once on this machine.
- The PC is kept awake while the script runs (same mechanism a video player
  uses; no power setting is changed).

.EXAMPLE
.\scripts\run_seeds.ps1 -Seeds 0,1,2
#>
param(
    [string[]]$Algos = @("dqn", "ppo"),
    [int[]]$Seeds = @(0, 1, 2),
    [int]$MaxParallel = 3,
    [int]$StaggerSeconds = 15,
    [string]$OutDir = "runs"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python venv not found at $python" }

Add-Type -Namespace Power -Name Awake -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
"@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"

function Test-RunDone([string]$algo, [int]$seed) {
    $meta = Join-Path $OutDir "$algo\seed_$seed\meta.json"
    if (-not (Test-Path $meta)) { return $false }
    return ((Get-Content $meta -Raw | ConvertFrom-Json).status -eq "done")
}

$logDir = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

$queue = New-Object System.Collections.Queue
foreach ($algo in $Algos | Sort-Object { if ($_ -eq "dqn") { 0 } else { 1 } }) {
    foreach ($seed in $Seeds) {
        if (Test-RunDone $algo $seed) { Write-Host "skip $algo seed $seed (already done)" }
        else { $queue.Enqueue(@{ Algo = $algo; Seed = $seed }) }
    }
}

[Power.Awake]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
$running = @()
$failed = @()
try {
    while ($queue.Count -gt 0 -or $running.Count -gt 0) {
        foreach ($job in @($running | Where-Object { $_.Process.HasExited })) {
            $status = if ($job.Process.ExitCode -eq 0) { "done" } else { "FAILED (exit $($job.Process.ExitCode))" }
            Write-Host ("{0:HH:mm} {1} seed {2}: {3}" -f (Get-Date), $job.Algo, $job.Seed, $status)
            if ($job.Process.ExitCode -ne 0) { $failed += "$($job.Algo) seed $($job.Seed)" }
            $running = @($running | Where-Object { $_ -ne $job })
        }

        if ($queue.Count -gt 0 -and $running.Count -lt $MaxParallel) {
            $next = $queue.Dequeue()
            $name = "$($next.Algo)_seed$($next.Seed)"
            $process = Start-Process -FilePath $python -NoNewWindow -PassThru `
                -RedirectStandardOutput (Join-Path $logDir "$name.out.log") `
                -RedirectStandardError (Join-Path $logDir "$name.err.log") `
                -ArgumentList @("-W", "ignore", "-m", "lunarlander_rl.train",
                                "--config", "configs/$($next.Algo).yaml",
                                "--seed", $next.Seed, "--out-dir", $OutDir, "--threads", 1)
            # Keeps the process handle so ExitCode is still readable after exit.
            $null = $process.Handle
            $running += @{ Algo = $next.Algo; Seed = $next.Seed; Process = $process }
            Write-Host ("{0:HH:mm} started {1}" -f (Get-Date), $name)
            Start-Sleep -Seconds $StaggerSeconds
        }
        else {
            Start-Sleep -Seconds 10
        }
    }

    foreach ($algo in $Algos) {
        foreach ($seed in $Seeds) {
            $runDir = Join-Path $OutDir "$algo\seed_$seed"
            if ((Test-RunDone $algo $seed) -and -not (Test-Path (Join-Path $runDir "final_eval.csv"))) {
                & $python -W ignore -m lunarlander_rl.evaluate --run-dir $runDir
            }
        }
    }
}
finally {
    [Power.Awake]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

if ($failed.Count -gt 0) {
    Write-Host "Failed runs: $($failed -join ', '). See $logDir."
    exit 1
}
