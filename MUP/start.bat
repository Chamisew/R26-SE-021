@echo off
echo ============================================================
echo   Component 3 -- Memory/Failure Predictor
echo ============================================================

echo [1/4] Installing dependencies...
python -m pip install -r requirements.txt --quiet
python -m pip install win10toast plyer --quiet

echo [2/4] Running model comparison and training best model...
python scripts/compare_models.py

echo [3/4] Starting live simulator in new window...
start "Live Simulator" cmd /k python scripts/simulate_live.py

echo [4/4] Starting dashboard...
start "" http://localhost:5003
python dashboard/app.py
pause
