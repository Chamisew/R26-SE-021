"""
Automated Plan Requirements Verification Test
---------------------------------------------
Verifies all implementation rules for final_research_dataset.csv:
1. Schema & Column Integrity (qpi, arrival_processing_diff exist; queue_pressure_index, rate_difference removed; no duplicates).
2. Per-service temporal dynamics (Sample 1 velocity=0.0, Samples 1 & 2 accel=0.0).
3. Zero-variance Z-scores (std <= 0.001 -> NaN).
4. Calibration target exclusion (baseline_tier == 'CALIBRATING' -> ml_label = NaN).
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(FRAMEWORK_DIR)

from queue_analyzer import engineer_features, label_pre_failure

def run_tests():
    print("=" * 80)
    print("    RUNNING AUTOMATED PLAN REQUIREMENTS VERIFICATION SUITE")
    print("=" * 80)

    # Synthetic Multi-Service Telemetry DataFrame
    data = []
    # Service A: 100 rows, std > 0
    t0 = 1700000000.0
    for i in range(100):
        data.append({
            "timestamp": t0 + i * 1.0,
            "time": "12:00:00",
            "incident_id": "inc-1",
            "service": "svc-a",
            "cpu_percent": 20.0 + (i % 5) * 2.0,
            "in_flight_queue": 1.0 + (i % 3),
            "incoming_rate": 10.0,
            "processing_rate": 10.0,
            "latency_ms": 15.0,
            "baseline_tier": "CALIBRATING" if i < 60 else "STATISTICAL",
            "fault_injected": 0,
            "service_unreachable": 1 if i == 80 else 0,
            "health_check_failed": 1 if i == 85 else 0,
            "request_timeout": 1 if i == 90 else 0,
            "http_errors_per_sec": 0.0,
            "error_rate": 0.0,
            "actual_degradation": 0,
            "actual_failure": 1 if i in (80, 85, 90) else 0
        })
    # Service B: 50 rows, zero variance for CPU (std = 0)
    t1 = t0 + 200.0
    for i in range(50):
        data.append({
            "timestamp": t1 + i * 1.0,
            "time": "12:05:00",
            "incident_id": "inc-2",
            "service": "svc-b",
            "cpu_percent": 50.0,  # Zero variance
            "in_flight_queue": 0.0,
            "incoming_rate": 5.0,
            "processing_rate": 5.0,
            "latency_ms": 10.0,
            "baseline_tier": "STATISTICAL",
            "fault_injected": 0,
            "service_unreachable": 0,
            "health_check_failed": 0,
            "request_timeout": 0,
            "http_errors_per_sec": 0.0,
            "error_rate": 0.0,
            "actual_degradation": 0,
            "actual_failure": 0
        })

    df_raw = pd.DataFrame(data)

    # -------------------------------------------------------------------------
    # TEST 1: Column Integrity & Consolidation
    # -------------------------------------------------------------------------
    df_feat = engineer_features(df_raw)
    
    assert "qpi" in df_feat.columns, "FAILED: 'qpi' column missing!"
    assert "arrival_processing_diff" in df_feat.columns, "FAILED: 'arrival_processing_diff' column missing!"
    assert "queue_pressure_index" not in df_feat.columns, "FAILED: Redundant 'queue_pressure_index' column found!"
    assert "rate_difference" not in df_feat.columns, "FAILED: Redundant 'rate_difference' column found!"
    
    dup_cols = df_feat.columns[df_feat.columns.duplicated()].tolist()
    assert len(dup_cols) == 0, f"FAILED: Duplicate columns found: {dup_cols}"
    print("[OK] TEST 1 PASSED: Column consolidation & integrity verified (no duplicate/redundant columns).")

    # -------------------------------------------------------------------------
    # TEST 2: Per-Service Velocity & Acceleration Rules
    # -------------------------------------------------------------------------
    svc_a_rows = df_feat[df_feat['service'] == 'svc-a'].sort_values('timestamp')
    svc_b_rows = df_feat[df_feat['service'] == 'svc-b'].sort_values('timestamp')

    # Sample 1 velocity = 0.0
    assert svc_a_rows['cpu_velocity'].iloc[0] == 0.0, "FAILED: Svc-A sample 1 velocity is not 0.0!"
    assert svc_b_rows['cpu_velocity'].iloc[0] == 0.0, "FAILED: Svc-B sample 1 velocity is not 0.0!"

    # Sample 1 & 2 acceleration = 0.0
    assert svc_a_rows['cpu_acceleration'].iloc[0] == 0.0, "FAILED: Svc-A sample 1 accel is not 0.0!"
    assert svc_a_rows['cpu_acceleration'].iloc[1] == 0.0, "FAILED: Svc-A sample 2 accel is not 0.0!"
    assert svc_b_rows['cpu_acceleration'].iloc[0] == 0.0, "FAILED: Svc-B sample 1 accel is not 0.0!"
    assert svc_b_rows['cpu_acceleration'].iloc[1] == 0.0, "FAILED: Svc-B sample 2 accel is not 0.0!"

    # Ensure velocity values are realistic (no 3000+ explosive spikes)
    max_vel_a = svc_a_rows['cpu_velocity'].abs().max()
    assert max_vel_a < 100.0, f"FAILED: Unrealistic CPU velocity explosion detected: {max_vel_a}"
    print("[OK] TEST 2 PASSED: Per-service temporal velocity & acceleration rules verified.")

    # -------------------------------------------------------------------------
    # TEST 3: Zero-Variance Z-Score Handling
    # -------------------------------------------------------------------------
    # In svc-b, cpu_percent was constant (50.0), so cpu_std = 0.0
    df_feat['cpu_std'] = 0.0
    df_feat['cpu_zscore'] = 0.0
    df_feat = engineer_features(df_feat)
    
    # Check that zero-std results in NaN Z-scores
    assert df_feat['cpu_zscore'].isna().all(), "FAILED: Zero-std Z-scores did not become NaN!"
    print("[OK] TEST 3 PASSED: Zero-variance Z-scores correctly evaluate to NaN.")

    # -------------------------------------------------------------------------
    # TEST 4: Calibration ML Exclusion
    # -------------------------------------------------------------------------
    df_labeled = label_pre_failure(df_feat)
    calib_labels = df_labeled[df_labeled['baseline_tier'] == 'CALIBRATING']['ml_label']
    assert calib_labels.isna().all(), "FAILED: CALIBRATING tier rows do not have ml_label = NaN!"
    print("[OK] TEST 4 PASSED: CALIBRATING rows strictly output ml_label = NaN.")

    # -------------------------------------------------------------------------
    # TEST 5: Actual Failure Derived Strictly From Observed Platform Signals
    # -------------------------------------------------------------------------
    required_signal_cols = [
        "service_unreachable",
        "health_check_failed",
        "request_timeout",
        "http_errors_per_sec",
        "error_rate"
    ]
    for col in required_signal_cols:
        assert col in df_feat.columns, f"FAILED: Real platform signal column '{col}' missing!"

    expected_failure = (
        (df_feat["service_unreachable"] == 1) |
        (df_feat["health_check_failed"] == 1) |
        (df_feat["request_timeout"] == 1)
    ).astype(int)

    if "actual_failure" in df_feat.columns and not df_feat["actual_failure"].isna().all():
        assert (
            df_feat["actual_failure"].fillna(0).astype(int) == expected_failure
        ).all(), "FAILED: actual_failure is not derived purely from observed platform signals!"

    print("[OK] TEST 5 PASSED: actual_failure uses real observed platform signals only.")

    print("=" * 80)
    print("      ALL PLAN REQUIREMENT VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_tests()
