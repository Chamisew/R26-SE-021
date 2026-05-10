# Memory Failure Detection Pipeline
 
Stack-agnostic log parsing and hybrid feature extraction pipeline for
predicting memory leaks and CPU spikes in microservice systems.
 
## Components
- `pipeline.py`  – Core data pipeline (Drain3 + hybrid classifier + sliding window)
- `app.py`       – FastAPI REST + WebSocket server
- `_verify.py`   – Post-upgrade verification checklist
- `fix_csv_ids.py` – Utility: fix sequential project IDs in output CSV
- `test_trends.py` – Unit test for failure trend analysis
 
## Quick start
```bash
pip install -r requirements.txt
python pipeline.py          # CSV mode (default)
# or
uvicorn app:app --reload    # API server
```