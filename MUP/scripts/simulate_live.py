"""
Component 3 -- Live Feed Simulator
Replays ml_ready_dataset.csv row-by-row into live_feed.csv to simulate
real-time data from Component 1, calculates live memory predictions using
the trained Random Forest model, and sends predictions to Component 4 API.
"""
import sys, os, time
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import pandas as pd
from datetime import datetime
import joblib
import urllib.request
import urllib.error
import json

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP1_DATA = os.path.abspath(os.path.join(BASE, "..", "component1", "output", "ml_ready_dataset.csv"))
HPMFD_DATA = os.path.abspath(os.path.join(BASE, "..", "..", "..", "HPMFD", "output", "ml_ready_dataset.csv"))
LOCAL_DATA = os.path.join(BASE, "data", "ml_ready_dataset.csv")

def resolve_input_csv() -> str:
    for path in (COMP1_DATA, HPMFD_DATA, LOCAL_DATA):
        if os.path.exists(path):
            return path
    return LOCAL_DATA

INPUT_CSV = resolve_input_csv()
LIVE_CSV  = os.path.join(BASE, "data", "live_feed.csv")
PRED_CSV  = os.path.join(BASE, "data", "memory_predictions.csv")
MODEL_PATH = os.path.join(BASE, "models", "memory_leak_rf_model.pkl")
DELAY     = 0.3  # seconds between rows

FEATURES = [
    "memory_change_10min", "memory_change_5min", "heap_rate", "gc_spike_count",
    "ram_mean", "ram_max", "ram_std", "ram_std_trend",
    "heap_max", "gc_count", "ram_percent", "heap_mb_used",
    "incident_phase_1", "incident_phase_2"
]

# Load model bundle
model = None
scaler = None
ALARM_THRESHOLD = 0.6
failure_idx = 1

if os.path.exists(MODEL_PATH):
    try:
        print(f"[simulate_live] Loading model bundle from {MODEL_PATH}...")
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        scaler = bundle["scaler"]
        FEATURES = bundle.get("features", FEATURES)
        ALARM_THRESHOLD = bundle.get("threshold", 0.6)
        if hasattr(model, "classes_"):
            try:
                failure_idx = list(model.classes_).index("FAILURE")
            except ValueError:
                failure_idx = 1
        print(f"[simulate_live] Model loaded successfully. Alarm threshold: {ALARM_THRESHOLD}")
    except Exception as e:
        print(f"[WARNING] Failed to load model bundle: {e}. Running in dummy prediction mode.")
else:
    print(f"[WARNING] Model bundle not found at {MODEL_PATH}. Running in dummy prediction mode.")

def send_prediction_to_comp4(service_name, project_id, prob, alert, label):
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "service_name": service_name,
        "project_id": project_id,
        "comp2_cpu": {
            "cpu_percent": 0.0,
            "cpu_velocity": 0.0,
            "cpu_trend_5min": 0.0,
            "cpu_trend_10min": 0.0,
            "in_flight_queue": 0.0,
            "incoming_rate": 0.0,
            "processing_rate": 0.0,
            "queue_growth_rate": 0.0,
            "overload_flag": 0,
            "queue_pressure_index": 0.0
        },
        "comp3_memory": {
            "memory_prob": round(prob, 4),
            "alert": str(alert).upper(),
            "pred_label": str(label).upper()
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=1.0) as response:
            return True, response.read().decode('utf-8')
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError) or "timed out" in str(e.reason).lower():
            return False, "timed out"
        return False, "connection refused"
    except Exception as e:
        return False, str(e)

def main():
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"[simulate_live] Streaming {total} rows from ml_ready_dataset.csv")
    print(f"[simulate_live] Writing live data to live_feed.csv  (delay={DELAY}s)")
    print(f"[simulate_live] Sending live predictions to Component 4 API...")
    print(f"[simulate_live] Press Ctrl+C to stop")
    print()

    # Write headers once
    header_written = False
    pred_header_written = False

    while True:  # loop forever to keep simulator running
        for idx, row in df.iterrows():
            # Get predictions
            if model is not None and scaler is not None:
                try:
                    X = pd.DataFrame([row[FEATURES]])
                    X_scaled = scaler.transform(X)
                    probs = model.predict_proba(X_scaled)
                    prob_failure = float(probs[0][failure_idx])
                    alert = "TRUE" if prob_failure >= ALARM_THRESHOLD else "FALSE"
                    pred_label = "FAILURE" if prob_failure >= ALARM_THRESHOLD else "NORMAL"
                except Exception as e:
                    print(f"  [ERROR] Model prediction failed: {e}")
                    prob_failure = 0.0
                    alert = "FALSE"
                    pred_label = "NORMAL"
            else:
                prob_failure = 0.0
                alert = "FALSE"
                pred_label = "NORMAL"

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            svc   = str(row.get("service_name", "unknown"))
            proj  = str(row.get("project_id", "Proj_01"))

            # 1. Update live_feed.csv
            live_row = row.copy()
            live_row["timestamp"] = timestamp_str
            row_df = pd.DataFrame([live_row])
            if not header_written:
                row_df.to_csv(LIVE_CSV, index=False, mode='w')
                header_written = True
            else:
                try:
                    existing = pd.read_csv(LIVE_CSV)
                    combined = pd.concat([existing, row_df], ignore_index=True)
                    if len(combined) > 200:
                        combined = combined.tail(200)
                    combined.to_csv(LIVE_CSV, index=False)
                except Exception:
                    row_df.to_csv(LIVE_CSV, index=False, mode='w')

            # 2. Update memory_predictions.csv live (so Dashboard stays in sync)
            pred_row = {
                "timestamp": timestamp_str,
                "service_name": svc,
                "project_id": proj,
                "memory_prob": round(prob_failure, 4),
                "alert": alert,
                "pred_label": pred_label
            }
            pred_df = pd.DataFrame([pred_row])
            if not pred_header_written:
                pred_df.to_csv(PRED_CSV, index=False, mode='w')
                pred_header_written = True
            else:
                try:
                    existing_preds = pd.read_csv(PRED_CSV)
                    combined_preds = pd.concat([existing_preds, pred_df], ignore_index=True)
                    if len(combined_preds) > 200:
                        combined_preds = combined_preds.tail(200)
                    combined_preds.to_csv(PRED_CSV, index=False)
                except Exception:
                    pred_df.to_csv(PRED_CSV, index=False, mode='w')

            # 3. Send to Component 4 API
            success, msg = send_prediction_to_comp4(svc, proj, prob_failure, alert, pred_label)
            api_status = "OK" if success else f"FAIL ({msg})"

            print(f"  -> [{proj}] {svc[:20]:<22}  mem_prob={prob_failure:.4f}  alert={alert}  API={api_status}")

            time.sleep(DELAY)

        print(f"\n[simulate_live] Restarting stream...\n")

if __name__ == "__main__":
    main()
