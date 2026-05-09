"""
LIVE CPU PREDICTION SIMULATOR (OPTIMIZED)
===========================================
Replays cpu_predictions.csv row-by-row to the dashboard's
/api/ingest-prediction endpoint as if your Component 3 is sending
live predictions in real time.

Optimizations (from performance comparison):
- SPEED_FACTOR: 20x (up from 5x)
- BATCH_SIZE: 50 rows/request (up from 1)
- itertuples() instead of iterrows() (10-100x faster)
- Console output: 10% (print every 10 rows, not every row)
"""

import time
import pandas as pd
import requests
import os
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DASHBOARD_URL = "http://localhost:5000/api/ingest-prediction"
SPEED_FACTOR  = 20.0   # 20× faster than real time!
BATCH_SIZE    = 50     # Send 50 rows per HTTP request
BASE          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPU_PRED_FILE = os.path.join(BASE, "data", "cpu_predictions.csv")
RAW_DATA_FILE = os.path.join(BASE, "data", "final_research_dataset.csv")
# ────────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  LIVE CPU PREDICTION SIMULATOR (OPTIMIZED)")
    print(f"  Target : {DASHBOARD_URL}")
    print(f"  Speed  : {SPEED_FACTOR}× realtime")
    print(f"  Batch  : {BATCH_SIZE} rows/request")
    print(f"  Data   : {CPU_PRED_FILE}")
    print("=" * 60)
    print("  Ctrl+C to stop\n")

    if not os.path.exists(CPU_PRED_FILE):
        print(f"  ✗ File not found: {CPU_PRED_FILE}")
        sys.exit(1)

    df = pd.read_csv(CPU_PRED_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Merge with raw dataset to get ground truth labels if available
    label_map = {}
    if os.path.exists(RAW_DATA_FILE):
        df_raw = pd.read_csv(RAW_DATA_FILE)
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
        if len(df) <= len(df_raw):
            for i in range(len(df)):
                label_map[i] = int(df_raw.iloc[i]["label"])

    # Convert ISO timestamp strings to unix floats for timing
    df["ts_float"] = pd.to_datetime(df["timestamp"]).values.astype("int64") // 10**9

    timestamps = df["ts_float"].values
    t0_data    = timestamps[0]
    t0_real    = time.time()

    sent = 0
    errors = 0
    batch = []
    batch_count = 0

    for i, row in enumerate(df.itertuples(index=False)):
        # Calculate how long to wait before sending this row
        data_elapsed = timestamps[i] - t0_data
        target_real  = t0_real + data_elapsed / SPEED_FACTOR
        wait         = target_real - time.time()
        if wait > 0:
            time.sleep(wait)

        payload = {
            "timestamp":         getattr(row, "timestamp", ""),
            "service_name":      getattr(row, "service_name", ""),
            "project_id":        str(getattr(row, "project_id", "")),
            "cpu_failure_prob":  float(getattr(row, "cpu_failure_prob", 0.0)),
            "label":             label_map.get(i, 0)
        }
        
        batch.append(payload)

        # Send batch when we reach BATCH_SIZE or end of data
        if len(batch) >= BATCH_SIZE or i == len(df) - 1:
            try:
                r = requests.post(DASHBOARD_URL, json=batch, timeout=10)
                if r.status_code == 200:
                    sent += len(batch)
                    batch_count += 1
                else:
                    errors += 1
                    print(f"  ⚠ ingest error {r.status_code}: {r.text[:80]}")
            except requests.ConnectionError:
                print(f"  ✗ Cannot connect to {DASHBOARD_URL} — is the dashboard running?")
                sys.exit(1)
            except Exception as e:
                errors += 1
                print(f"  ⚠ Request error: {e}")
            
            # Clear batch
            batch = []

        marker = " ⚠ ALARM" if payload["cpu_failure_prob"] >= 0.55 else ""
        fail_marker = " 🔴 FAILURE" if payload["label"] == 1 else ""
        
        # Only print every 10 rows (10% console output)
        if i % 10 == 0 or i == len(df) - 1:
            print(f"  row {i:4d}/{len(df)-1}  {payload['service_name']:15s}  project={payload['project_id']:10s}  prob={payload['cpu_failure_prob']:.3f}{marker}{fail_marker}  batches={batch_count}")

    print(f"\n  ✅ Simulation complete. Sent={sent}  Errors={errors}  Batches={batch_count}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
