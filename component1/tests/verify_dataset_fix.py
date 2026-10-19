"""
Verification test confirming:
(a) ml_ready_dataset.csv header equals the column list in step 1 exactly;
(b) confidence is not blank for any row;
(c) incident_phase_1/2 do not change when hybrid_label and ground_truth_label are shuffled,
    and the value at time T is identical after deleting all rows later than T;
(d) pipeline_output.csv still contains label_phase_1 and label_phase_2.
"""
import os
import sys
import numpy as np
import pandas as pd

import pipeline

EXPECTED_ML_COLS = [
    "timestamp", "project_id", "service_name", "window_id",
    "label", "failure_type", "incident_phase_1", "incident_phase_2",
    "ram_mean", "ram_max", "ram_std", "ram_std_trend", "ram_percent",
    "memory_change_5min", "memory_change_10min",
    "heap_mb_used", "heap_max", "heap_rate", "gc_count", "gc_spike_count",
    "label_3state", "confidence",
]


def test_a_ml_ready_header():
    # 1. In-code definition check
    assert pipeline.FINAL_ML_COLS == EXPECTED_ML_COLS, (
        f"pipeline.FINAL_ML_COLS does not match step 1:\nExpected: {EXPECTED_ML_COLS}\nActual:   {pipeline.FINAL_ML_COLS}"
    )

    # 2. On-disk CSV check
    ml_path = os.path.join(pipeline.CONFIG["output_dir"], "ml_ready_dataset.csv")
    assert os.path.exists(ml_path), f"{ml_path} does not exist"
    df = pd.read_csv(ml_path)
    csv_cols = list(df.columns)
    assert csv_cols == EXPECTED_ML_COLS, (
        f"ml_ready_dataset.csv header mismatch:\nExpected: {EXPECTED_ML_COLS}\nActual:   {csv_cols}"
    )
    print("  [PASS] (a) ml_ready_dataset.csv header equals step 1 column list exactly.")


def test_b_confidence_not_blank():
    ml_path = os.path.join(pipeline.CONFIG["output_dir"], "ml_ready_dataset.csv")
    df = pd.read_csv(ml_path)
    assert not df.empty, "ml_ready_dataset.csv is empty"
    blank_count = df["confidence"].isna().sum()
    assert blank_count == 0, f"Found {blank_count} blank confidence values out of {len(df)} rows"
    print(f"  [PASS] (b) confidence is not blank for any row ({len(df)} rows checked, 0 blanks).")


def test_c_incident_phases_invariance_and_causality():
    # Generate synthetic time series data across 2 services
    n = 40
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=n, freq="1min")
    rng = np.random.default_rng(42)

    base_df = pd.DataFrame({
        "timestamp": timestamps,
        "service_name": ["order-service"] * (n // 2) + ["payment-service"] * (n // 2),
        "ram_percent": np.linspace(40, 95, n),
        "heap_mb_used": np.linspace(500, 3500, n),
        "heap_max": [3000.0] * n,
        "hybrid_label": rng.choice(["NORMAL", "FAILURE"], size=n),
        "ground_truth_label": rng.choice(["NORMAL", "FAILURE"], size=n),
    })

    # Part 1: Shuffle hybrid_label and ground_truth_label
    shuffled_df = base_df.copy()
    shuffled_df["hybrid_label"] = rng.permutation(base_df["hybrid_label"].values)
    shuffled_df["ground_truth_label"] = rng.permutation(base_df["ground_truth_label"].values)

    res_base = pipeline.add_incident_phase_columns(base_df)
    res_shuffled = pipeline.add_incident_phase_columns(shuffled_df)

    assert (res_base["incident_phase_1"] == res_shuffled["incident_phase_1"]).all(), (
        "incident_phase_1 changed when labels were shuffled!"
    )
    assert (res_base["incident_phase_2"] == res_shuffled["incident_phase_2"]).all(), (
        "incident_phase_2 changed when labels were shuffled!"
    )

    # Part 2: Causality check at time T - deleting rows after T leaves row at T unchanged
    for t_idx in [5, 12, 19, 25, 35]:
        time_t = timestamps[t_idx]
        val_t_p1 = res_base.loc[t_idx, "incident_phase_1"]
        val_t_p2 = res_base.loc[t_idx, "incident_phase_2"]

        # Truncate dataset to only rows <= time_t
        truncated_df = base_df[base_df["timestamp"] <= time_t].copy()
        res_truncated = pipeline.add_incident_phase_columns(truncated_df)

        val_trunc_p1 = res_truncated.loc[t_idx, "incident_phase_1"]
        val_trunc_p2 = res_truncated.loc[t_idx, "incident_phase_2"]

        assert val_t_p1 == val_trunc_p1, (
            f"incident_phase_1 at index {t_idx} differed after deleting future rows! ({val_t_p1} vs {val_trunc_p1})"
        )
        assert val_t_p2 == val_trunc_p2, (
            f"incident_phase_2 at index {t_idx} differed after deleting future rows! ({val_t_p2} vs {val_trunc_p2})"
        )

    print("  [PASS] (c) incident_phase_1/2 are invariant to label shuffling and strictly causal at time T.")


def test_d_pipeline_output_contains_label_phases():
    pipe_path = os.path.join(pipeline.CONFIG["output_dir"], "pipeline_output.csv")
    assert os.path.exists(pipe_path), f"{pipe_path} does not exist"
    df = pd.read_csv(pipe_path)
    assert "label_phase_1" in df.columns, "label_phase_1 missing from pipeline_output.csv"
    assert "label_phase_2" in df.columns, "label_phase_2 missing from pipeline_output.csv"
    assert "risk_phase_early" not in df.columns, "risk_phase_early should not be in pipeline_output.csv"
    assert "risk_phase_critical" not in df.columns, "risk_phase_critical should not be in pipeline_output.csv"
    print("  [PASS] (d) pipeline_output.csv contains label_phase_1 and label_phase_2 (and no risk_phase_*).")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING CONFIRMATION TEST SUITE")
    print("=" * 60)
    test_a_ml_ready_header()
    test_b_confidence_not_blank()
    test_c_incident_phases_invariance_and_causality()
    test_d_pipeline_output_contains_label_phases()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
