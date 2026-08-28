@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

set ROOT=%~dp0
echo ============================================================
echo   SRE Command Center - Complete Deployment & Runner Stack
echo   Root: %ROOT%
echo ============================================================
echo.

REM 1. Check Python and Node
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
)
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js not found in PATH.
    pause
    exit /b 1
)

echo [1/6] Starting Component 1: Log & Metrics Ingestion API (port 8001)...
start "COMP1 - Log Ingestion API :8001" cmd /k "cd /d "%ROOT%component1" && python -m uvicorn app:app --host 0.0.0.0 --port 8001"

echo [2/6] Starting Component 4: CPU Spike Predictor API (port 8000)...
start "COMP4 - CPU Predictor API :8000" cmd /k "cd /d "%ROOT%cpu_spike_predictor" && python -m uvicorn src.api:app --host 0.0.0.0 --port 8000"

echo [3/6] Starting Component 2: Local Azure Microservices (ports 5001-5005)...
start "COMP2 - Local Microservices :5001-5005" cmd /k "cd /d "%ROOT%c2\Queue-Aware CPU Spike Analyzer\research_framework" && python run_local_services.py"

echo [4/6] Starting Dashboard Aggregation Gateway API (port 8766)...
start "DASHBOARD - Gateway API :8766" cmd /k "cd /d "%ROOT%Dashboard" && python server.py"

echo [5/6] Starting Dashboard Frontend UI (React / Vite on port 5173)...
start "DASHBOARD - UI React :5173" cmd /k "cd /d "%ROOT%Dashboard\dashboard-ui" && npx vite --host 127.0.0.1"

timeout /t 3 >nul

echo [6/6] Launching Dashboard in default browser...
start "" "http://localhost:5173"

echo.
echo ============================================================
echo   ALL SERVICES ACTIVE AND OPERATIONAL
echo ============================================================
echo   - SRE React Dashboard     :  http://localhost:5173
echo   - Dashboard Gateway API   :  http://localhost:8766/api/dashboard
echo   - Comp 1 Ingestion API    :  http://localhost:8001/docs
echo   - Comp 4 CPU Predictor    :  http://localhost:8000/docs
echo   - Comp 2 Microservices    :  http://localhost:5001 - 5005
echo ============================================================
echo.
pause
