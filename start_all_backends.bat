@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

set ROOT=%~dp0
echo ============================================================
echo   SRE Command Center - Full Backend Stack (Windows)
echo   Root: %ROOT%
echo ============================================================
echo.

REM =========================================================================
REM STEP 0: Check that Python exists globally and report versions
REM =========================================================================
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 'python' not found in PATH. Install Python 3.10+ first.
    pause
    exit /b 1
)
echo [Check] Python found:
python --version
echo.

REM =========================================================================
REM STEP 1: Component 1 - Log Metrics Pipeline (FastAPI on :8001)
REM =========================================================================
echo.
echo [1/7] Checking Component 1 (Log/Metrics Ingestion Pipeline)...
set COMP1=%ROOT%component1
if not exist "%COMP1%\app.py" (
    echo       SKIP: component1\app.py not found at %COMP1%\app.py
) else (
    set VENV=%COMP1%\venv311
    if not exist "!VENV!\Scripts\python.exe" set "VENV=%COMP1%\venv"
    if not exist "!VENV!\Scripts\python.exe" (
        echo       Creating venv for component1...
        python -m venv "%COMP1%\venv311"
        set "VENV=%COMP1%\venv311"
        echo       Installing component1 requirements...
        "!VENV!\Scripts\python.exe" -m pip install -r "%COMP1%\requirements.txt" --quiet
    )
    echo       Starting Component 1 API (port 8001) in new window...
    start "COMP1 - Ingestion API :8001" cmd /k "cd /d "%COMP1%" && "%COMP1%\venv311\Scripts\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload"
)

REM =========================================================================
REM STEP 2: Component 3 - MUP Memory Predictor Dashboard (Flask on :5003)
REM =========================================================================
echo.
echo [2/7] Checking Component 3 (MUP / Memory Predictor)...
set MUP=%ROOT%MUP
if not exist "%MUP%\dashboard\app.py" (
    echo       SKIP: MUP\dashboard\app.py not found
) else (
    if not exist "%MUP%\models\memory_leak_rf_model.pkl" (
        echo       MUP model not found - running compare_models.py first...
    )
    set "VENV3=%MUP%\venv"
    if not exist "!VENV3!\Scripts\python.exe" (
        echo       Creating venv for MUP...
        python -m venv "%MUP%\venv"
        set "VENV3=%MUP%\venv"
        echo       Installing MUP requirements (this takes a while)...
        "!VENV3!\Scripts\python.exe" -m pip install -r "%MUP%\requirements.txt" --quiet
        "!VENV3!\Scripts\python.exe" -m pip install win10toast plyer --quiet
    )
    REM Train/Compare model
    echo       Training / selecting best MUP model if not already done...
    start /W "MUP - model training" cmd /c "cd /d "%MUP%" && "%MUP%\venv\Scripts\python.exe" scripts\compare_models.py"
    REM Live simulator (writes memory_predictions.csv live)
    echo       Starting MUP live simulator...
    start "COMP3 - Live Simulator" cmd /k "cd /d "%MUP%" && "%MUP%\venv\Scripts\python.exe" scripts\simulate_live.py"
    REM Flask dashboard
    echo       Starting MUP dashboard on port 5003...
    start "COMP3 - Flask Dashboard :5003" cmd /k "cd /d "%MUP%" && "%MUP%\venv\Scripts\python.exe" dashboard\app.py"
)

REM =========================================================================
REM STEP 3: Component 4 - CPU Spike Predictor (FastAPI on :8000 + inference)
REM =========================================================================
echo.
echo [3/7] Checking Component 4 (CPU Spike Predictor)...
set CPU=%ROOT%cpu_spike_predictor
if not exist "%CPU%\src\api.py" (
    echo       SKIP: cpu_spike_predictor\src\api.py not found
) else (
    set "VENV4=%CPU%\.venv"
    if not exist "!VENV4!\Scripts\python.exe" (
        echo       Creating venv for cpu_spike_predictor...
        python -m venv "%CPU%\.venv"
        set "VENV4=%CPU%\.venv"
        echo       Installing cpu_spike_predictor requirements...
        "!VENV4!\Scripts\python.exe" -m pip install -r "%CPU%\requirements.txt" --quiet
    )
    REM Train production model if not present
    if not exist "%CPU%\outputs\cpu_rf_model.joblib" (
        echo       Training production CPU RF model...
        start /W "COMP4 - model training" cmd /c "cd /d "%CPU%" && "%CPU%\.venv\Scripts\python.exe" -m src.cli train"
    )
    REM FastAPI
    echo       Starting COMP4 FastAPI (port 8000)...
    start "COMP4 - CPU Predictor API :8000" cmd /k "cd /d "%CPU%" && "%CPU%\.venv\Scripts\python.exe" -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload"
    REM Live inference loop (reads CSVs from comp2+comp3, predicts, writes live_status.json)
    echo       Starting COMP4 live inference loop...
    start "COMP4 - Live Inference" cmd /k "cd /d "%CPU%" && "%CPU%\.venv\Scripts\python.exe" -m src.live_inference"
)

REM =========================================================================
REM STEP 4: Component 2 - Queue-Aware CPU Spike Analyzer & Simulated Services
REM =========================================================================
echo.
echo [4/7] Starting Component 2 (Local Azure Microservices ports 5001-5005)...
set C2=%ROOT%c2\Queue-Aware CPU Spike Analyzer\research_framework
if exist "%C2%\run_local_services.py" (
    echo       Starting 5 Local Microservices (go-azure, node-azure, python-azure, ruby-azure, php-azure)...
    start "COMP2 - Local Services :5001-5005" cmd /k "cd /d "%C2%" && python run_local_services.py"
) else (
    echo       SKIP: run_local_services.py not found at %C2%
)

REM =========================================================================
REM STEP 5: Unified SRE Command Center HTML Dashboard (served via Python simple server :8080)
REM =========================================================================
echo.
echo [5/7] Starting unified SRE Command Center HTML dashboard...
set DASH=%ROOT%dashboard
if exist "%DASH%\sre_command_center_live.html" (
    start "SRE Dashboard Live" cmd /k "cd /d "%DASH%" && python -m http.server 8080"
    timeout /t 2 >nul
    start "" "http://localhost:8080/sre_command_center_live.html"
) else if exist "%DASH%\sre_command_center_light_theme (2).html" (
    start "SRE Dashboard Static" cmd /k "cd /d "%DASH%" && python -m http.server 8080"
    timeout /t 2 >nul
    start "" "http://localhost:8080/sre_command_center_light_theme (2).html"
) else (
    echo       SKIP: No dashboard HTML found in dashboard\ folder
)

REM =========================================================================
REM STEP 6: Show service URL map
REM =========================================================================
echo.
echo ============================================================
echo   ALL SERVICES LAUNCHED - Endpoint map:
echo ============================================================
echo   Component 1 (Ingestion API)      :  http://localhost:8001/docs
echo   Component 4 (CPU Predictor API)  :  http://localhost:8000/docs
echo   Component 4 (Live Status JSON)   :  http://localhost:8000/metrics
echo   Component 3 (Memory Dashboard)   :  http://localhost:5003
echo   Component 3 (Live API JSON)      :  http://localhost:5003/api/live
echo   SRE Command Center (unified UI)  :  http://localhost:8080/sre_command_center_live.html
echo ============================================================
echo.
echo   To stop: close each terminal window, or press Ctrl+C in this window.
echo.
pause
endlocal
