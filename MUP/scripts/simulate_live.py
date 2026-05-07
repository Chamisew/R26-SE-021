"""
Component 3 -- Live Feed Simulator
Replays ml_ready_dataset.csv row-by-row into live_feed.csv to simulate
real-time data from Component 1.
"""
import sys, os, time
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

import pandas as pd
from datetime import datetime

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE, "data", "ml_ready_dataset.csv")
LIVE_CSV  = os.path.join(BASE, "data", "live_feed.csv")
DELAY     = 0.3  # seconds between rows

FEATURES = [
    "memory_change_10min", "memory_change_5min", "heap_rate", "gc_spike_count",
    "ram_mean", "ram_max", "ram_std", "ram_std_trend",
    "heap_max", "gc_count", "ram_percent", "heap_mb_used",
    "incident_phase_1", "incident_phase_2"
]

def main():
    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"[simulate_live] Streaming {total} rows from ml_ready_dataset.csv")
    print(f"[simulate_live] Writing live data to live_feed.csv  (delay={DELAY}s)")
    print(f"[simulate_live] Press Ctrl+C to stop")
    print()

    # Write header once
    header_written = False

    while True:  # loop forever to keep simulator running
        for idx, row in df.iterrows():
            live_row = row.copy()
            live_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Append row to live_feed.csv
            row_df = pd.DataFrame([live_row])
            if not header_written:
                row_df.to_csv(LIVE_CSV, index=False, mode='w')
                header_written = True
            else:
                # Keep only last 200 rows to avoid huge file
                try:
                    existing = pd.read_csv(LIVE_CSV)
                    combined = pd.concat([existing, row_df], ignore_index=True)
                    if len(combined) > 200:
                        combined = combined.tail(200)
                    combined.to_csv(LIVE_CSV, index=False)
                except Exception:
                    row_df.to_csv(LIVE_CSV, index=False, mode='w')

            label = row.get("label", "?")
            svc   = str(row.get("service_name", "unknown"))[:20]
            proj  = str(row.get("project_id", "?"))
            mg    = row.get("memory_change_5min", 0)
            print(f"  -> [{proj}] {svc:<22}  mem_growth={mg:.4f}  label={label}")

            time.sleep(DELAY)

        print(f"\n[simulate_live] Restarting stream...\n")

if __name__ == "__main__":
    main()
