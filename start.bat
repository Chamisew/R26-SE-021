@echo off
echo ============================================================
echo   CPU SPIKE PREDICTOR - Production System
echo ============================================================

:: Step 1 - Install dependencies
echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

:: Step 2 - Train model
echo.
echo [2/3] Training model on your dataset...
python scripts\train_model.py
if errorlevel 1 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
)

:: Step 3 - Launch dashboard
echo.
echo [3/3] Starting dashboard...
echo.
echo    Dashboard : http://localhost:5000
echo    Ingest    : POST http://localhost:5000/api/ingest
echo.
echo    To simulate live data in a SECOND terminal, run:
echo      python scripts\simulate_live.py
echo.
python dashboard\app.py
pause
