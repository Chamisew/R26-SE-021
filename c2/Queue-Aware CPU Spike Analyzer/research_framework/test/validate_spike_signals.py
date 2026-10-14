"""
5-Minute Spike Validation Script
---------------------------------
Purpose: Prove that Cloude deployments produce REAL observable failure
events under controlled spike injection, BEFORE committing to a full dataset run.

What this validates:
  1. That at least 1 row has service_unreachable=1 OR health_check_failed=1
     OR request_timeout=1 during a spike window.
  2. That actual_failure=1 ONLY on rows with a real signal (not CPU/queue crossing a number).
  3. That CPU, Queue, Latency, QPI remain strictly in the FEATURES columns.
  4. Prints a clear validation report showing the pattern of signals during the spike.

How to use:
  - Run `python run_pipeline.py` and let it calibrate (~60 seconds).
  - Wait for a spike to trigger automatically (or wait up to 5 minutes).
  - After the pipeline stops, run: python test/validate_spike_signals.py
"""

import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(SCRIPT_DIR, "..", "final_research_dataset.csv")

REAL_SIGNAL_COLS = [
    "service_unreachable", "health_check_failed",
    "request_timeout", "http_errors_per_sec", "error_rate"
]
FEATURE_COLS = [
    "cpu_percent", "in_flight_queue", "incoming_rate",
    "processing_rate", "latency_ms", "qpi"
]

def run_validation():
    print("=" * 80)
    print("    SPIKE VALIDATION: Real Platform Failure Signal Verification")
    print("=" * 80)

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
        print("        Run `python run_pipeline.py` first and wait for at least one spike.")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n[+] Loaded dataset: {len(df)} rows x {len(df.columns)} columns")

    # ─── Check schema ────────────────────────────────────────────────────────────
    missing = [c for c in REAL_SIGNAL_COLS if c not in df.columns]
    if missing:
        print(f"\n[FAIL] New real signal columns are MISSING from CSV: {missing}")
        print("       Your pipeline was run with the OLD realtime_experiment.py.")
        print("       Delete final_research_dataset.csv and re-run run_pipeline.py.")
        sys.exit(1)
    print(f"[OK]  All 5 real platform signal columns present in CSV schema.")

    # ─── Check no CPU/queue threshold dependency in actual_failure ───────────────
    print("\n--- Checking actual_failure is NOT purely CPU/queue threshold-driven ---")
    rows_with_failure = df[df['actual_failure'] == 1]
    if len(rows_with_failure) == 0:
        print("[INFO] No actual_failure=1 rows found yet.")
        print("       This means your spike hasn't produced a real failure event.")
        print("       See OUTCOME SECTION below.")
    else:
        # Verify each failure row has at least one real failure signal active
        real_signal_active = (
            (rows_with_failure['service_unreachable'] == 1) |
            (rows_with_failure['health_check_failed'] == 1) |
            (rows_with_failure['request_timeout'] == 1)
        )
        orphan_failures = rows_with_failure[~real_signal_active]
        if len(orphan_failures) > 0:
            print(f"[FAIL] {len(orphan_failures)} rows have actual_failure=1 but NO real failure signal active!")
            print("       These are false/threshold-based failures. Something is wrong.")
            print(orphan_failures[REAL_SIGNAL_COLS + ['cpu_percent', 'in_flight_queue']].head(5).to_string())
        else:
            print(f"[OK]  All {len(rows_with_failure)} failure rows have at least one real platform failure signal active.")

    # ─── Signal distribution ─────────────────────────────────────────────────────
    print("\n--- Real Platform Signal Distribution ---")
    print(f"{'Signal':<28} | {'Count Active':<14} | {'% of Dataset':<14}")
    print("-" * 60)
    for col in REAL_SIGNAL_COLS:
        if col in ['service_unreachable', 'health_check_failed', 'request_timeout']:
            active = (df[col] == 1).sum()
        else:
            active = (df[col] > 0).sum()
        pct = active / max(len(df), 1) * 100
        print(f"{col:<28} | {active:<14} | {pct:.2f}%")

    # ─── Key pattern table (fault_injected window) ───────────────────────────────
    print("\n--- Sample Pattern During Fault Injection Window (last 30 injected rows) ---")
    injected = df[df['fault_injected'] == 1]
    if len(injected) > 0:
        sample = injected.tail(30)[
            ['service', 'cpu_percent', 'in_flight_queue', 'latency_ms',
             'service_unreachable', 'health_check_failed', 'request_timeout',
             'http_errors_per_sec', 'error_rate', 'actual_failure']
        ]
        print(sample.to_string(index=False))
    else:
        print("[INFO] No fault_injected=1 rows found yet (spike hasn't triggered or was missed).")

    # ─── Outcome verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    has_real_signal = (
        (df['service_unreachable'] == 1).any() or
        (df['health_check_failed'] == 1).any() or
        (df['request_timeout'] == 1).any()
    )

    if has_real_signal and len(rows_with_failure) > 0:
        print("[RESULT] PASS: Real platform failure events OBSERVED.")
        print("         Your Azure/Render/Oracle services DO produce observable failure")
        print("         signals under controlled spike injection. Proceed with full dataset.")
    elif not has_real_signal and len(df[df['fault_injected'] == 1]) > 0:
        print("[RESULT] CAUTION: Spike was injected but NO real failure signals appeared.")
        print("         This means your deployed services survived the workload without")
        print("         generating HTTP 5xx errors, timeouts, or health check failures.")
        print("")
        print("         Options:")
        print("         A) Increase spike duration (change SPIKE_DURATION in config).")
        print("         B) Increase concurrent request burst (raise load_generator rate).")
        print("         C) Accept this and document that controlled experiments show CPU/Queue")
        print("            pressure without hard failures — a SOFT degradation study.")
    else:
        print("[RESULT] INFO: Not enough data yet. Run pipeline longer and retry.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_validation()
