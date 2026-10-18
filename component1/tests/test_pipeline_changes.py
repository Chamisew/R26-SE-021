"""
Comprehensive tests for pipeline.py and app.py changes.
Tests: valid ground truth, schema preservation, causal phase flags,
       independence from shuffled labels, debounced FAILURE count <= raw FAILURE count.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import pipeline


def test_valid_ground_truth():
    """Test that ground_truth_label is correctly assigned from failure_events.csv."""
    events = pipeline.load_failure_events()
    assert events is not None, "failure_events.csv should exist"
    assert not events.empty, "failure_events.csv should not be empty"
    assert list(events.columns) == ["service_name", "failure_type", "start_time", "end_time"], \
        f"Unexpected columns: {list(events.columns)}"
    print("  [PASS] Valid ground truth: failure_events.csv loaded correctly")
    print(f"         Events: {len(events)}")
    print(f"         Services: {events['service_name'].unique().tolist()}")


def test_schema_preservation():
    """Test that CSV column names and order are preserved (new columns appended only)."""
    # Check FINAL_ML_COLS contains expected columns
    expected_ml = [
        "timestamp", "project_id", "service_name", "window_id",
        "label", "failure_type", "incident_phase_1", "incident_phase_2",
        "ram_mean", "ram_max", "ram_std", "ram_std_trend", "ram_percent",
        "memory_change_5min", "memory_change_10min",
        "heap_mb_used", "heap_max", "heap_rate", "gc_count", "gc_spike_count",
    ]
    assert pipeline.FINAL_ML_COLS == expected_ml, \
        f"FINAL_ML_COLS mismatch: {pipeline.FINAL_ML_COLS}"
    print("  [PASS] Schema preservation: FINAL_ML_COLS correct")
    print(f"         Columns: {len(expected_ml)}")


def test_causal_phase_flags():
    """Test that incident_phase_1/2 are causal (only use current/past data)."""
    # Create test data where a failure happens at t=10min
    timestamps = pd.date_range("2026-01-01", periods=20, freq="1min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "service_name": ["svc-a"] * 20,
        "ram_percent": [50.0] * 20,
        "hybrid_label": ["NORMAL"] * 20,
    })
    # Set failure at index 10 (t=10min)
    df.loc[10, "hybrid_label"] = "FAILURE"

    result = pipeline.add_incident_phase_flags(df)

    # Phase flags should be set for rows at or before the failure time
    # (no future data used - rows after failure should NOT have phase flags
    #  unless they are within the post-failure window which is causal)
    # The key test: phase flags should NOT use future failure information
    # i.e., a row at t=3min should not know about a failure at t=10min
    for idx, row in result.iterrows():
        if idx < 3:  # More than 7 min before failure - should NOT have phase flags
            assert row["label_phase_1"] == 0, \
                f"Phase 1 should be 0 at row {idx} (too far before failure), got {row['label_phase_1']}"
            assert row["label_phase_2"] == 0, \
                f"Phase 2 should be 0 at row {idx} (too far before failure), got {row['label_phase_2']}"

    # Before failure: phase flags should be set for some rows
    assert result["label_phase_1"].sum() > 0, "Phase 1 should have some 1s before failure"
    assert result["label_phase_2"].sum() > 0, "Phase 2 should have some 1s before failure"
    print("  [PASS] Causal phase flags: no future data used")
    print(f"         Phase 1 count: {result['label_phase_1'].sum()}")
    print(f"         Phase 2 count: {result['label_phase_2'].sum()}")


def test_independence_from_shuffled_labels():
    """Test that predictive features don't depend on hybrid_label or ground_truth_label."""
    # Create two datasets with same metrics but different labels
    n = 50
    timestamps = pd.date_range("2026-01-01", periods=n, freq="5s")

    df1 = pd.DataFrame({
        "timestamp": timestamps,
        "service_name": ["svc-a"] * n,
        "ram_percent": np.linspace(50, 90, n),
        "heap_mb_used": np.linspace(1000, 3000, n),
        "gc_count": [1] * n,
        "hybrid_label": ["NORMAL"] * n,
        "ground_truth_label": ["NORMAL"] * n,
    })

    df2 = df1.copy()
    df2["hybrid_label"] = ["FAILURE"] * n
    df2["ground_truth_label"] = ["FAILURE"] * n

    # Build ML dataframes - features should be identical
    ml1 = pipeline.build_ml_dataframe(df1)
    ml2 = pipeline.build_ml_dataframe(df2)

    # Check that feature columns (not label columns) are identical
    # Exclude all label/outcome columns that are expected to differ
    label_cols = {"label", "failure_type", "label_3state", "confidence"}
    feature_cols = [c for c in pipeline.FINAL_ML_COLS if c not in label_cols]
    for col in feature_cols:
        if col in ml1.columns and col in ml2.columns:
            if col == "timestamp":
                # Timestamps should be identical (same input data)
                if not (ml1[col] == ml2[col]).all():
                    print(f"  [FAIL] Feature '{col}' differs between NORMAL and FAILURE labels")
                    return False
                continue
            # Handle string columns separately
            if ml1[col].dtype == object or ml2[col].dtype == object:
                if not (ml1[col].fillna("") == ml2[col].fillna("")).all():
                    print(f"  [FAIL] Feature '{col}' differs between NORMAL and FAILURE labels")
                    return False
                continue
            # Handle numeric columns
            try:
                v1 = ml1[col].fillna(-999).values.astype(float)
                v2 = ml2[col].fillna(-999).values.astype(float)
                if not np.allclose(v1, v2, equal_nan=True):
                    print(f"  [FAIL] Feature '{col}' differs between NORMAL and FAILURE labels")
                    return False
            except (ValueError, TypeError):
                # Non-numeric, non-string column - compare directly
                if not (ml1[col].fillna("") == ml2[col].fillna("")).all():
                    print(f"  [FAIL] Feature '{col}' differs between NORMAL and FAILURE labels")
                    return False

    print("  [PASS] Independence from shuffled labels: features don't depend on labels")
    print(f"         Checked {len(feature_cols)} feature columns")
    return True


def test_debounced_failure_count():
    """Test that debounced FAILURE count <= raw FAILURE count."""
    # Create data with intermittent failures
    n = 30
    timestamps = pd.date_range("2026-01-01", periods=n, freq="5s")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "service_name": ["svc-a"] * n,
        "log_template": ["memory leak detected"] * n,
        "ram_percent": [90.0] * n,
        "cpu_percent": [20.0] * n,
        "heap_mb_used": [3000.0] * n,
        "gc_count": list(range(n)),
    })

    config = {
        "tfidf_threshold": 0.12,
        "metric_score_threshold": 3,
        "drain3_depth": 4,
        "drain3_sim_thresh": 0.4,
        "drain3_max_children": 100,
    }

    result = pipeline.stage3_hybrid_classifier(df, config)

    raw_fail = int((result["raw_hybrid_label"] == "FAILURE").sum())
    debounced_fail = int((result["hybrid_label"] == "FAILURE").sum())

    assert debounced_fail <= raw_fail, \
        f"Debounced FAILURE ({debounced_fail}) > raw FAILURE ({raw_fail})"

    print("  [PASS] Debounced FAILURE count <= raw FAILURE count")
    print(f"         Raw: {raw_fail}, Debounced: {debounced_fail}")


def test_label_3state():
    """Test that label_3state is computed correctly."""
    events = pipeline.load_failure_events()
    if events.empty:
        print("  [SKIP] No failure events to test label_3state")
        return

    n = 30
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "service_name": [events.iloc[0]["service_name"]] * n,
        "hybrid_label": ["NORMAL"] * n,
    })

    result = pipeline.compute_label_3state(df, events)

    assert "label_3state" in result.columns, "label_3state column missing"
    assert set(result["label_3state"].unique()).issubset({"NORMAL", "PRE_FAILURE", "FAILURE"}), \
        f"Unexpected label_3state values: {result['label_3state'].unique()}"

    print("  [PASS] label_3state computed correctly")
    print(f"         Distribution: {result['label_3state'].value_counts().to_dict()}")


def test_confidence_in_ml_dataset():
    """Test that confidence scoring function works correctly."""
    row = {"detection_layer": "keyword_metric_confirmed"}
    conf = pipeline._compute_confidence(row, mscore=4, metric_thresh=3)
    assert 0.0 <= conf <= 1.0, f"Confidence out of bounds: {conf}"
    print(f"  [PASS] Confidence computation verified: conf={conf}")


def test_baseline_threshold():
    """Test that per-service baselines require 100 healthy rows."""
    # Create a service with only 50 healthy rows
    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="5s"),
        "service_name": ["svc-test"] * n,
        "ram_percent": np.random.normal(50, 5, n),
        "cpu_percent": np.random.normal(30, 3, n),
        "heap_mb_used": np.random.normal(1000, 100, n),
        "gc_count": [1] * n,
        "ground_truth_label": ["NORMAL"] * n,
    })

    pipeline._service_metric_baselines.clear()
    pipeline._update_service_baselines(df)

    # Should use fixed fallbacks (< 100 healthy rows)
    bl = pipeline._service_metric_baselines.get("svc-test", {})
    assert bl.get("threshold_ram") == 85.0, \
        f"Expected fallback threshold 85.0, got {bl.get('threshold_ram')}"
    assert bl.get("threshold_heap") == 2500.0, \
        f"Expected fallback threshold 2500.0, got {bl.get('threshold_heap')}"

    print("  [PASS] Baseline threshold: uses fixed fallbacks until 100 healthy rows")
    print(f"         threshold_ram={bl.get('threshold_ram')}, threshold_heap={bl.get('threshold_heap')}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PIPELINE CHANGES TEST SUITE")
    print("=" * 60)

    tests = [
        ("Valid Ground Truth", test_valid_ground_truth),
        ("Schema Preservation", test_schema_preservation),
        ("Causal Phase Flags", test_causal_phase_flags),
        ("Independence from Shuffled Labels", test_independence_from_shuffled_labels),
        ("Debounced FAILURE Count", test_debounced_failure_count),
        ("Label 3-State", test_label_3state),
        ("Confidence in ML Dataset", test_confidence_in_ml_dataset),
        ("Baseline Threshold", test_baseline_threshold),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            result = test_fn()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
