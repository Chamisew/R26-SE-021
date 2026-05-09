"""
LIVE METRIC SIMULATOR (OPTIMIZED)
===================================
Replays final_research_dataset.csv row-by-row to the dashboard's
/api/ingest endpoint as if your two external components are
sending live metrics in real time.

Optimizations:
- SPEED_FACTOR: 20x
- BATCH_SIZE: 50 rows/request
- itertuples() instead of iterrows()
- Pre-converted UNIX timestamps (no per-row datetime calculations!)
- Reduced console output
"""

import time
import pandas as pd
import requests
import os
import sys

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DASHBOARD_URL = "http://localhost:5000/api/ingest"

SPEED_FACTOR = 20.0
BATCH_SIZE = 50

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE,
    "data",
    "final_research_dataset.csv"
)

SERVICE_MAP = {
    "Python": "python-api",
    "Php": "php-api",
    "Go": "go-service",
    "Node.js": "node-service",
    "Ruby": "ruby-service",
}

DEFAULT_SERVICE = "system-global"
# ──────────────────────────────────────────────────────────────────────────────


def build_payload(row) -> list[dict]:
    base = {
        "cpu_percent": float(getattr(row, "cpu_percent", 0)),
        "in_flight_queue": float(getattr(row, "in_flight_queue", 0)),
        "incoming_rate": float(getattr(row, "incoming_rate", 0)),
        "processing_rate": float(getattr(row, "processing_rate", 0)),
        "queue_growth_rate": float(getattr(row, "queue_growth_rate", 0)),
        "cpu_trend_5min_ma": float(getattr(row, "cpu_trend_5min_ma", 0)),
        "cpu_trend_10min_ma": float(getattr(row, "cpu_trend_10min_ma", 0)),
        "timestamp": float(getattr(row, "ts_float", time.time())),
        "project_id": "project_1",
        "label": int(getattr(row, "label", 0)),
    }

    fs = getattr(row, "failing_service", None)

    # NORMAL STATE
    if pd.isna(fs) or not fs:
        base["service"] = DEFAULT_SERVICE
        return [base]

    # INCIDENT STATE
    svc_name = SERVICE_MAP.get(
        str(fs),
        str(fs).lower().replace(".", "-")
    )

    svc_payload = {
        **base,
        "service": svc_name,
    }

    global_payload = {
        "service": DEFAULT_SERVICE,
        "project_id": "project_1",
        "cpu_percent": float(getattr(row, "cpu_percent", 0)) * 0.3,
        "in_flight_queue": 0,
        "incoming_rate": float(getattr(row, "incoming_rate", 0)),
        "processing_rate": float(getattr(row, "incoming_rate", 0)),
        "queue_growth_rate": 0,
        "cpu_trend_5min_ma": 0,
        "cpu_trend_10min_ma": 0,
        "timestamp": float(getattr(row, "ts_float", time.time())),
        "label": int(getattr(row, "label", 0)),
    }

    return [svc_payload, global_payload]


def main():
    print("=" * 60)
    print("  LIVE METRIC SIMULATOR (OPTIMIZED)")
    print(f"  Target : {DASHBOARD_URL}")
    print(f"  Speed  : {SPEED_FACTOR}× realtime")
    print(f"  Batch  : {BATCH_SIZE} rows/request")
    print(f"  Data   : {DATA_FILE}")
    print("=" * 60)
    print("  Ctrl+C to stop\n")

    # LOAD DATASET
    df = pd.read_csv(DATA_FILE)

    # PRE-CONVERT TIMESTAMPS TO UNIX FLOATS (SPEED CRITICAL!)
    df["ts_float"] = pd.to_datetime(df["timestamp"]).values.astype("int64") // 10**9 # type: ignore

    # SORT
    df = df.sort_values("timestamp").reset_index(drop=True)

    timestamps = df["ts_float"].values
    t0_data = timestamps[0]
    t0_real = time.time()

    sent = 0
    errors = 0
    batch = []
    batch_count = 0

    # FAST ITERATION
    for i, row in enumerate(df.itertuples(index=False)):
        # REALTIME REPLAY (USING PRE-CONVERTED FLOATS - NO DATETIME OPS!)
        data_elapsed = timestamps[i] - t0_data
        target_real = t0_real + data_elapsed / SPEED_FACTOR
        wait = target_real - time.time()

        if wait > 0:
            time.sleep(wait)

        # BUILD PAYLOADS
        payloads = build_payload(row)
        batch.extend(payloads)

        # SEND BATCH
        if len(batch) >= BATCH_SIZE or i == len(df) - 1:
            try:
                r = requests.post(
                    DASHBOARD_URL,
                    json=batch,
                    timeout=10
                )
                if r.status_code == 200:
                    sent += len(batch)
                    batch_count += 1
                else:
                    errors += 1
                    print(f"⚠ ingest error {r.status_code}: {r.text[:80]}")
            except requests.ConnectionError:
                print(f"✗ Cannot connect to {DASHBOARD_URL}")
                print("Is the dashboard running?")
                sys.exit(1)
            except Exception as e:
                errors += 1
                print(f"⚠ Request error: {e}")
            
            # CLEAR BATCH
            batch = []

        # CONSOLE OUTPUT (10% only)
        if i % 10 == 0 or i == len(df) - 1:
            state = getattr(row, "system_state", "")
            svc = getattr(row, "failing_service", "")
            cpu = getattr(row, "cpu_percent", 0)
            label = getattr(row, "label", 0)
            marker = " 🔴 FAILURE" if label == 1 else ""
            svc_str = f"[{svc}]" if pd.notna(svc) and svc else "[normal]"
            
            print(f"row {i:4d}/{len(df)-1}  {state:10s} {svc_str:15s}  cpu={cpu:6.1f}%{marker}  batches={batch_count}")

    print("\n" + "=" * 60)
    print(f"✅ Simulation complete\nSent={sent}\nErrors={errors}\nBatches={batch_count}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
