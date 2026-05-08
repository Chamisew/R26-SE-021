#!/bin/bash
set -e

echo "============================================================"
echo "  CPU SPIKE PREDICTOR - Production System"
echo "============================================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[2/3] Training model..."
python3 scripts/train_model.py

echo ""
echo "[3/3] Starting dashboard..."
echo ""
echo "   Dashboard : http://localhost:5000"
echo "   Ingest    : POST http://localhost:5000/api/ingest"
echo ""
echo "   To simulate live data, open a SECOND terminal and run:"
echo "     python3 scripts/simulate_live.py"
echo ""
python3 dashboard/app.py
