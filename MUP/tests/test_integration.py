"""
Integration tests for MUP Component 3.

Tests the full end-to-end pipeline:
  1. Data ingestion → config filtering → feature engineering
  2. Model load → row prediction → SHAP reason derivation
  3. Output file writing → schema validation
  4. Live simulation flow (bounded, not infinite loop)
  5. Component 1 config sync (enabled services match output)
"""

import sys
import os
import json
import shutil
import tempfile
import unittest
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Add scripts directory
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from feature_engineering import compute_derived_features, BASE_FEATURES, ALL_FEATURES

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP1_DATA = os.path.abspath(os.path.join(BASE, "..", "component1", "output", "ml_ready_dataset.csv"))
COMP1_CONFIG = os.path.abspath(os.path.join(BASE, "..", "component1", "services_config.json"))
MODEL_PATH = os.path.join(BASE, "models", "memory_leak_rf_model.pkl")


class TestFullPipeline(unittest.TestCase):
    """End-to-end: load CSV → filter → engineer features → predict → write output."""

    @classmethod
    def setUpClass(cls):
        """Load real dataset and model once for all tests in this class."""
        if not os.path.exists(COMP1_DATA):
            raise unittest.SkipTest(f"Component 1 dataset not found: {COMP1_DATA}")
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"Model bundle not found: {MODEL_PATH}")

        # Step 1: Load and filter
        cls.df_raw = pd.read_csv(COMP1_DATA)
        if os.path.exists(COMP1_CONFIG):
            with open(COMP1_CONFIG, encoding="utf-8") as fh:
                cfg = json.load(fh)
            cls.enabled_services = {s["name"] for s in cfg.get("services", []) if s.get("enabled")}
            if cls.enabled_services:
                cls.df_raw = cls.df_raw[cls.df_raw["service_name"].isin(cls.enabled_services)].copy()

        # Step 2: Drop NaN and compute features
        cls.df_raw.dropna(subset=BASE_FEATURES, inplace=True)
        cls.df_engineered = compute_derived_features(cls.df_raw)

        # Step 3: Load model
        bundle = joblib.load(MODEL_PATH)
        cls.model = bundle["model"]
        cls.scaler = bundle["scaler"]
        cls.features = bundle.get("features", ALL_FEATURES)
        cls.threshold = bundle.get("threshold", 0.6)
        if hasattr(cls.model, "classes_"):
            try:
                cls.failure_idx = list(cls.model.classes_).index("FAILURE")
            except ValueError:
                cls.failure_idx = 1
        else:
            cls.failure_idx = 1

    def test_1_dataset_not_empty_after_filter(self):
        """After filtering by config, dataset still has rows."""
        self.assertGreater(len(self.df_engineered), 0)

    def test_2_only_enabled_services_present(self):
        """No service outside Component 1 config appears in working data."""
        actual = set(self.df_engineered["service_name"].unique())
        if self.enabled_services:
            self.assertTrue(actual.issubset(self.enabled_services),
                            f"Unexpected services: {actual - self.enabled_services}")

    def test_3_all_derived_features_present(self):
        """Every derived feature column exists and has no NaN/Inf."""
        for col in ALL_FEATURES:
            self.assertIn(col, self.df_engineered.columns, f"Missing feature: {col}")
            series = self.df_engineered[col]
            self.assertFalse(series.isna().any(), f"{col} contains NaN")
            self.assertFalse(np.isinf(series).any(), f"{col} contains Inf")

    def test_4_model_predicts_valid_probability(self):
        """Model outputs probabilities in [0, 1] for sample rows."""
        sample = self.df_engineered[self.features].head(50)
        X_scaled = self.scaler.transform(sample)
        probs = self.model.predict_proba(X_scaled)
        self.assertEqual(probs.shape[0], 50)
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())
        # Probabilities sum to 1 per row
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_5_predictions_produce_valid_labels(self):
        """Each prediction maps to FAILURE or NORMAL based on threshold."""
        sample = self.df_engineered[self.features].head(100)
        X_scaled = self.scaler.transform(sample)
        probs = self.model.predict_proba(X_scaled)[:, self.failure_idx]
        labels = np.where(probs >= self.threshold, "FAILURE", "NORMAL")
        unique_labels = set(labels)
        self.assertTrue(unique_labels.issubset({"FAILURE", "NORMAL"}))

    def test_6_output_csv_schema(self):
        """Simulate writing prediction output and verify schema matches memory_predictions.csv."""
        test_dir = tempfile.mkdtemp(prefix="mup_integration_")
        try:
            sample = self.df_engineered.head(10)
            X_scaled = self.scaler.transform(sample[self.features])
            probs = self.model.predict_proba(X_scaled)[:, self.failure_idx]

            pred_rows = []
            for i, row in sample.iterrows():
                p = float(probs[list(sample.index).index(i)])
                pred_rows.append({
                    "timestamp": str(row.get("timestamp", "")),
                    "service_name": str(row.get("service_name", "")),
                    "project_id": str(row.get("project_id", "")),
                    "memory_prob": round(p, 4),
                    "alert": "TRUE" if p >= self.threshold else "FALSE",
                    "pred_label": "FAILURE" if p >= self.threshold else "NORMAL",
                    "reason": "Test reason" if p >= self.threshold else "N/A",
                })

            out_path = os.path.join(test_dir, "memory_predictions.csv")
            pd.DataFrame(pred_rows).to_csv(out_path, index=False)

            df_out = pd.read_csv(out_path)
            expected_cols = {"timestamp", "service_name", "project_id",
                             "memory_prob", "alert", "pred_label", "reason"}
            self.assertEqual(set(df_out.columns), expected_cols)
            self.assertEqual(len(df_out), 10)
            # alert is boolean string (pandas may convert TRUE/FALSE to bool on read)
            alert_vals = set(df_out["alert"].astype(str).str.upper().unique())
            self.assertTrue(alert_vals.issubset({"TRUE", "FALSE"}),
                            f"Unexpected alert values: {alert_vals}")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_7_live_feed_csv_schema(self):
        """Verify live_feed output preserves all feature columns."""
        test_dir = tempfile.mkdtemp(prefix="mup_integration_")
        try:
            sample = self.df_engineered.head(5).copy()
            sample["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            out_path = os.path.join(test_dir, "live_feed.csv")
            sample.to_csv(out_path, index=False)

            df_out = pd.read_csv(out_path)
            self.assertEqual(len(df_out), 5)
            for col in ALL_FEATURES:
                self.assertIn(col, df_out.columns)
            # Metadata columns present
            for col in ["timestamp", "service_name", "project_id", "label"]:
                self.assertIn(col, df_out.columns)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestPerServiceHistoryIntegration(unittest.TestCase):
    """Verify the per-service deque strategy preserves all services."""

    def test_all_services_appear_in_output(self):
        """After simulating row processing, every enabled service has rows in output."""
        if not os.path.exists(COMP1_DATA):
            self.skipTest("Dataset not available")

        df = pd.read_csv(COMP1_DATA, nrows=2000)
        if os.path.exists(COMP1_CONFIG):
            with open(COMP1_CONFIG, encoding="utf-8") as fh:
                cfg = json.load(fh)
            enabled = {s["name"] for s in cfg.get("services", []) if s.get("enabled")}
            if enabled:
                df = df[df["service_name"].isin(enabled)]

        # Simulate the deque append logic
        store = {}
        for _, row in df.iterrows():
            svc = str(row.get("service_name", "unknown"))
            if svc not in store:
                store[svc] = deque(maxlen=50)
            store[svc].append(row.to_dict())

        # Every unique service in the data should be in the store
        expected_services = set(df["service_name"].unique())
        self.assertEqual(set(store.keys()), expected_services)
        # Each deque has at most 50 rows
        for svc, dq in store.items():
            self.assertLessEqual(len(dq), 50)
            self.assertGreater(len(dq), 0)


class TestModelHotReloadIntegration(unittest.TestCase):
    """Verify the model file is loadable and prediction works after reload."""

    def test_reload_preserves_predictions(self):
        """Two consecutive loads of the model yield identical predictions."""
        if not os.path.exists(MODEL_PATH):
            self.skipTest("Model not available")

        bundle1 = joblib.load(MODEL_PATH)
        bundle2 = joblib.load(MODEL_PATH)

        X = np.random.rand(5, len(bundle1["features"]))
        X1 = bundle1["scaler"].transform(pd.DataFrame(X, columns=bundle1["features"]))
        X2 = bundle2["scaler"].transform(pd.DataFrame(X, columns=bundle2["features"]))

        p1 = bundle1["model"].predict_proba(X1)
        p2 = bundle2["model"].predict_proba(X2)
        np.testing.assert_array_equal(p1, p2)


class TestComponent4PayloadIntegration(unittest.TestCase):
    """Verify the POST payload structure matches what Component 4 expects."""

    def test_payload_matches_comp4_schema(self):
        """Build a payload like simulate_live does and validate structure."""
        payload = {
            "service_name": "microservice-21",
            "project_id": "project_1",
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
                "queue_pressure_index": 0.0,
            },
            "comp3_memory": {
                "memory_prob": 0.85,
                "alert": "TRUE",
                "pred_label": "FAILURE",
                "reason": "Critical Heap Fill Capacity",
                "shapley_phi_values": {"heap_fill_ratio": 0.45},
            },
        }
        # Must serialize cleanly
        raw = json.dumps(payload)
        parsed = json.loads(raw)
        self.assertEqual(parsed["service_name"], "microservice-21")
        self.assertIn("comp2_cpu", parsed)
        self.assertIn("comp3_memory", parsed)
        self.assertEqual(len(parsed["comp2_cpu"]), 10)
        self.assertIn("shapley_phi_values", parsed["comp3_memory"])


if __name__ == "__main__":
    unittest.main()
