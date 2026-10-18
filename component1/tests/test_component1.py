"""
Acceptance test suite for Component 1 fixes (pipeline.py and app.py).
Tests:
1. Multi-service live simulation (no duplicate (service_name, timestamp), strictly increasing window_id, rows from every cycle).
2. Trend and memory change features (15+ min rising ram_percent -> memory_change_5min and 10min > 0 in saved CSV).
3. Contract schema validation (exactly 20 columns, 14 numeric features have no NaN).
4. API NaN robustness (/api/windows and /api/component3/feed return HTTP 200 and valid JSON when data has NaNs).
5. Restart persistence (re-running with existing CSV produces no duplicates and preserves project_id mapping).
6. Memory-only failure labeling (cpu_percent=95 with normal RAM/heap/GC must not be labeled FAILURE).
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import pipeline
import app


class TestComponent1Fixes(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="comp1_test_")
        self.cfg = dict(pipeline.CONFIG)
        self.cfg["output_dir"] = self.test_dir
        pipeline.init_persistent_state(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        # Restore default state
        pipeline.init_persistent_state(pipeline.CONFIG.get("output_dir", "output/"))

    def test_1_multiservice_live_cycles_no_duplicates_monotonic_ids(self):
        """Simulate 3 services and 5 consecutive 30s processing cycles."""
        services = ["svc-order", "svc-payment", "svc-catalog"]
        base_time = datetime(2026, 1, 1, 10, 0, 0)
        
        # In a live system, raw data accumulates over time (cumulative snapshot)
        all_raw_rows = []
        for cycle in range(1, 6):
            cycle_start = base_time + timedelta(seconds=(cycle - 1) * 30)
            # 6 points per cycle at 5s intervals
            for step in range(6):
                ts = cycle_start + timedelta(seconds=step * 5)
                for svc in services:
                    all_raw_rows.append({
                        "timestamp": ts,
                        "service_name": svc,
                        "stack": "python",
                        "ram_percent": 50.0 + (cycle * 2) + step,
                        "cpu_percent": 25.0,
                        "heap_mb_used": 600.0,
                        "gc_count": cycle * 5 + step,
                        "log_message": "info heartbeat tick",
                        "log_template": "info heartbeat tick",
                        "hybrid_label": "NORMAL",
                        "ground_truth_label": "NORMAL",
                        "failure_type": "none",
                        "detection_layer": "none",
                    })

            raw_df = pd.DataFrame(all_raw_rows)
            # Stage 4 sliding window runs on cumulative history
            windows_df = pipeline.stage4_sliding_window(raw_df, self.cfg)
            pipeline.continuous_save(windows_df, self.cfg)

        ml_csv = os.path.join(self.test_dir, "ml_ready_dataset.csv")
        self.assertTrue(os.path.exists(ml_csv), "ml_ready_dataset.csv should exist")
        df_ml = pd.read_csv(ml_csv)
        self.assertFalse(df_ml.empty, "ml_ready_dataset.csv should not be empty")

        # 1. No duplicate (service_name, timestamp)
        key_tuples = list(zip(df_ml["service_name"], df_ml["timestamp"].astype(str)))
        self.assertEqual(len(key_tuples), len(set(key_tuples)), "Duplicate (service_name, timestamp) found in ml_ready_dataset.csv")

        # 2. window_id is unique and strictly increasing
        window_ids = df_ml["window_id"].tolist()
        self.assertEqual(len(window_ids), len(set(window_ids)), "window_ids are not unique")
        self.assertTrue(all(x < y for x, y in zip(window_ids, window_ids[1:])), "window_id is not strictly increasing")

        # 3. Every service has rows from every cycle
        for svc in services:
            svc_df = df_ml[df_ml["service_name"] == svc]
            self.assertGreaterEqual(len(svc_df), 5, f"Service {svc} has too few rows across 5 cycles")

    def test_2_rising_ram_change_features(self):
        """Feed 15+ minutes of rising ram_percent for one service."""
        base_time = datetime(2026, 1, 1, 12, 0, 0)
        # 18 minutes of data at 5s interval = 216 rows
        rows = []
        for i in range(216):
            ts = base_time + timedelta(seconds=i * 5)
            # Gradually rising ram_percent from 40.0 to 80.0
            ram = 40.0 + (i / 216.0) * 40.0
            rows.append({
                "timestamp": ts,
                "service_name": "rising-ram-svc",
                "stack": "java",
                "ram_percent": ram,
                "cpu_percent": 30.0,
                "heap_mb_used": 1000.0 + i * 2,
                "gc_count": i // 2,
                "log_message": "processing job",
                "log_template": "processing job",
                "hybrid_label": "NORMAL",
                "ground_truth_label": "NORMAL",
                "failure_type": "none",
                "detection_layer": "none",
            })

        raw_df = pd.DataFrame(rows)
        windows_df = pipeline.stage4_sliding_window(raw_df, self.cfg)
        pipeline.continuous_save(windows_df, self.cfg)

        ml_csv = os.path.join(self.test_dir, "ml_ready_dataset.csv")
        df_ml = pd.read_csv(ml_csv)
        self.assertFalse(df_ml.empty)

        # After 12 minutes (720s), memory_change_5min and memory_change_10min should be > 0
        df_ml["timestamp_dt"] = pd.to_datetime(df_ml["timestamp"])
        twelve_mins_in = base_time + timedelta(minutes=12)
        late_rows = df_ml[df_ml["timestamp_dt"] >= twelve_mins_in]

        self.assertFalse(late_rows.empty, "Should have rows after 12 minutes")
        self.assertTrue((late_rows["memory_change_5min"] > 0).all(), "memory_change_5min should be > 0 for rising RAM")
        self.assertTrue((late_rows["memory_change_10min"] > 0).all(), "memory_change_10min should be > 0 for rising RAM")

    def test_3_saved_csv_header_and_no_nan_in_numeric_columns(self):
        """Assert header has exact 20 columns in order, and 14 numeric feature columns contain no NaN."""
        base_time = datetime(2026, 1, 1, 8, 0, 0)
        # Include single-point window situations where ram_std would naturally be NaN
        rows = [
            {
                "timestamp": base_time + timedelta(seconds=0),
                "service_name": "single-point-svc",
                "stack": "python",
                "ram_percent": 45.0,
                "cpu_percent": 20.0,
                "heap_mb_used": 500.0,
                "gc_count": 1,
                "log_message": "start",
                "log_template": "start",
                "hybrid_label": "NORMAL",
                "ground_truth_label": "NORMAL",
                "failure_type": "none",
                "detection_layer": "none",
            },
            {
                "timestamp": base_time + timedelta(seconds=5),
                "service_name": "single-point-svc",
                "stack": "python",
                "ram_percent": 46.0,
                "cpu_percent": 21.0,
                "heap_mb_used": 510.0,
                "gc_count": 2,
                "log_message": "step 1",
                "log_template": "step 1",
                "hybrid_label": "NORMAL",
                "ground_truth_label": "NORMAL",
                "failure_type": "none",
                "detection_layer": "none",
            },
        ]
        raw_df = pd.DataFrame(rows)
        windows_df = pipeline.stage4_sliding_window(raw_df, self.cfg)
        pipeline.continuous_save(windows_df, self.cfg)

        ml_csv = os.path.join(self.test_dir, "ml_ready_dataset.csv")
        df_ml = pd.read_csv(ml_csv)

        expected_columns = [
            "timestamp", "project_id", "service_name", "window_id",
            "label", "failure_type", "incident_phase_1", "incident_phase_2",
            "ram_mean", "ram_max", "ram_std", "ram_std_trend", "ram_percent",
            "memory_change_5min", "memory_change_10min", "heap_mb_used", "heap_max",
            "heap_rate", "gc_count", "gc_spike_count"
        ]
        self.assertEqual(list(df_ml.columns), expected_columns, "ml_ready_dataset.csv column headers mismatch")

        numeric_feature_cols = [
            "incident_phase_1", "incident_phase_2", "ram_mean", "ram_max",
            "ram_std", "ram_std_trend", "ram_percent", "memory_change_5min",
            "memory_change_10min", "heap_mb_used", "heap_max", "heap_rate",
            "gc_count", "gc_spike_count"
        ]
        self.assertEqual(len(numeric_feature_cols), 14)
        for col in numeric_feature_cols:
            nan_count = df_ml[col].isna().sum()
            self.assertEqual(nan_count, 0, f"Numeric feature column {col} contains {nan_count} NaN values")

    def test_4_api_nan_robustness(self):
        """Call /api/windows and /api/component3/feed when data contains NaN/inf."""
        client = TestClient(app.app)

        # Inject rows with NaNs, infs, NaT into STATE.windows_rows
        with app.STATE.lock:
            app.STATE.windows_rows = [
                {
                    "window_id": 101,
                    "timestamp": pd.Timestamp("2026-01-01 10:00:00"),
                    "service_name": "nan-test-svc",
                    "stack": "python",
                    "ram_percent": float("nan"),
                    "cpu_percent": float("inf"),
                    "ram_std": float("nan"),
                    "heap_mb_used": None,
                    "optional_field": np.nan,
                    "hybrid_label": "NORMAL",
                    "ground_truth_label": "NORMAL",
                },
                {
                    "window_id": 102,
                    "timestamp": "2026-01-01T10:01:00",
                    "service_name": "nan-test-svc",
                    "stack": "python",
                    "ram_percent": 55.0,
                    "cpu_percent": -float("inf"),
                    "ram_std": float("nan"),
                    "heap_mb_used": 800.0,
                    "hybrid_label": "FAILURE",
                    "failure_type": "oom",
                }
            ]

        # Call /api/windows
        resp_windows = client.get("/api/windows")
        self.assertEqual(resp_windows.status_code, 200, f"/api/windows failed with {resp_windows.status_code}")
        data_windows = resp_windows.json()
        self.assertIsInstance(data_windows, list)
        self.assertEqual(len(data_windows), 2)
        # Verify no raw NaN or inf strings
        self.assertIsNone(data_windows[0]["ram_percent"])
        self.assertIsNone(data_windows[0]["cpu_percent"])

        # Call /api/component3/feed
        resp_feed = client.get("/api/component3/feed?since_window_id=100")
        self.assertEqual(resp_feed.status_code, 200, f"/api/component3/feed failed with {resp_feed.status_code}")
        data_feed = resp_feed.json()
        self.assertIn("windows", data_feed)
        self.assertEqual(len(data_feed["windows"]), 2)
        self.assertIsNone(data_feed["windows"][0]["ram_percent"])
        self.assertIsNone(data_feed["windows"][0]["cpu_percent"])

    def test_5_restart_simulation_no_duplicates_and_same_project_id(self):
        """Restart simulation: re-run with existing CSV and assert no duplicates and stable project_id."""
        base_time = datetime(2026, 1, 1, 14, 0, 0)
        svc_a = "service-alpha"
        svc_b = "service-beta"

        # Initial run: map services and save first batch
        pid_a1 = pipeline.get_project_id(svc_a)
        pid_b1 = pipeline.get_project_id(svc_b)

        rows = []
        for i in range(5):
            for svc in [svc_a, svc_b]:
                rows.append({
                    "timestamp": base_time + timedelta(seconds=i * 5),
                    "service_name": svc,
                    "stack": "dotnet",
                    "ram_percent": 50.0 + i,
                    "cpu_percent": 20.0,
                    "heap_mb_used": 600.0,
                    "gc_count": i,
                    "log_message": "ok",
                    "log_template": "ok",
                    "hybrid_label": "NORMAL",
                    "ground_truth_label": "NORMAL",
                    "failure_type": "none",
                    "detection_layer": "none",
                })
        raw_df = pd.DataFrame(rows)
        windows_df = pipeline.stage4_sliding_window(raw_df, self.cfg)
        pipeline.continuous_save(windows_df, self.cfg)

        ml_csv = os.path.join(self.test_dir, "ml_ready_dataset.csv")
        df_before = pd.read_csv(ml_csv)
        count_before = len(df_before)
        max_id_before = int(df_before["window_id"].max())

        # Simulate restart: reload saved keys and counter from test_dir
        pipeline.init_persistent_state(self.test_dir)

        # Assert project_id is identical after reload
        pid_a2 = pipeline.get_project_id(svc_a)
        pid_b2 = pipeline.get_project_id(svc_b)
        self.assertEqual(pid_a1, pid_a2, "project_id changed after reload for service-alpha")
        self.assertEqual(pid_b1, pid_b2, "project_id changed after reload for service-beta")

        # Re-run same windows_df (should NOT add duplicate rows)
        pipeline.continuous_save(windows_df, self.cfg)
        df_after = pd.read_csv(ml_csv)
        self.assertEqual(len(df_after), count_before, "Re-running with existing data created duplicate rows!")

        # Now append 1 truly new row
        new_row = pd.DataFrame([{
            "timestamp": base_time + timedelta(seconds=100),
            "service_name": svc_a,
            "stack": "dotnet",
            "ram_percent": 55.0,
            "cpu_percent": 20.0,
            "heap_mb_used": 600.0,
            "gc_count": 10,
            "log_message": "ok",
            "log_template": "ok",
            "hybrid_label": "NORMAL",
            "ground_truth_label": "NORMAL",
            "failure_type": "none",
            "detection_layer": "none",
        }])
        new_window = pipeline.stage4_sliding_window(pd.concat([raw_df, new_row], ignore_index=True), self.cfg)
        pipeline.continuous_save(new_window, self.cfg)

        df_final = pd.read_csv(ml_csv)
        self.assertEqual(len(df_final), count_before + 1, "Expected exactly 1 new row appended")
        new_max_id = int(df_final["window_id"].max())
        self.assertEqual(new_max_id, max_id_before + 1, "New window_id must be max_id + 1 across restart")

    def test_6_cpu_95_not_labeled_failure(self):
        """A row with cpu_percent = 95 and normal RAM/heap/GC must NOT be labeled FAILURE by hybrid_label."""
        row_df = pd.DataFrame([{
            "timestamp": "2026-01-01 10:00:00",
            "service_name": "cpu-test-svc",
            "stack": "python",
            "ram_percent": 45.0,
            "cpu_percent": 95.0,
            "heap_mb_used": 400.0,
            "gc_count": 5,
            "log_message": "processing batch computations with high cpu load",
            "log_template": "processing batch computations with high cpu load",
            "ground_truth_label": "UNKNOWN",
        }])

        classified = pipeline.stage3_hybrid_classifier(row_df, self.cfg)
        self.assertEqual(classified["raw_hybrid_label"].iloc[0], "NORMAL",
                         f"cpu_percent=95 with normal RAM/heap was incorrectly classified as FAILURE: {classified['raw_hybrid_label'].iloc[0]}")
        self.assertEqual(classified["hybrid_label"].iloc[0], "NORMAL",
                         f"cpu_percent=95 with normal RAM/heap was debounced as FAILURE: {classified['hybrid_label'].iloc[0]}")


if __name__ == "__main__":
    unittest.main()
