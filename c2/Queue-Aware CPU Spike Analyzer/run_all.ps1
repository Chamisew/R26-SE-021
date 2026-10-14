# Run the full Queue-Aware CPU Spike Analyzer stack with one command:
#   1. Starts the backend + web console (realtime_experiment.py) in a new window
#   2. Waits for it to come online on http://localhost:8081
#   3. Opens the web console in your default browser
#
# Usage:  .\run_all.ps1
# Stop:   close the backend window.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "research_framework\realtime_experiment.py"
$Requirements = Join-Path $Root "research_framework\requirements.txt"

function Test-PythonWorks($Exe) {
    try {
        & $Exe --version 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

# Prefer venv python ONLY if it actually runs (venv may point to a deleted interpreter).
$Python = Join-Path $Root "research_framework\venv\Scripts\python.exe"
if (-not (Test-Path $Python) -or -not (Test-PythonWorks $Python)) {
    if (-not (Test-PythonWorks $Python)) {
        Write-Host "[run_all] Venv python is broken/missing; falling back to system python." -ForegroundColor Yellow
    }
    $Python = $null
    foreach ($cand in @("python", "py")) {
        # 'py -3' needs different arg handling; just probe plain candidates
        if (Test-PythonWorks $cand) { $Python = $cand; break }
    }
    if (-not $Python) { throw "No working Python found. Install Python 3.11+ and retry." }
}
Write-Host "[run_all] Using python: $Python" -ForegroundColor Cyan

# Ensure backend deps are present (aiohttp-cors is the usual missing one).
try {
    & $Python -c "import aiohttp, aiohttp_cors, numpy" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "missing deps" }
} catch {
    Write-Host "[run_all] Installing missing deps from requirements.txt..." -ForegroundColor Yellow
    & $Python -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install failed. Run manually: $Python -m pip install -r $Requirements" }
}

Write-Host "[run_all] Starting backend + web console..." -ForegroundColor Cyan
# -NoExit keeps the backend window open so startup errors stay visible instead of flashing shut.
Start-Process -FilePath $Python -ArgumentList "`"$Backend`"" `
    -WorkingDirectory (Join-Path $Root "research_framework")

Write-Host "[run_all] Waiting for backend on http://localhost:8081 ..."
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8081/api/health" -TimeoutSec 1 -UseBasicParsing
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}

if (-not $ok) {
    Write-Host "[run_all] WARNING: backend did not respond in time; opening UI anyway." -ForegroundColor Yellow
} else {
    Write-Host "[run_all] Backend is up." -ForegroundColor Green
}

Write-Host "[run_all] Opening web console..." -ForegroundColor Cyan
Start-Process "http://localhost:8081"
