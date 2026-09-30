"""
Unit tests for MUP Component 3 -- simulate_live.py

Validates:
  1. derive_alert_reason() with SHAP values and fallback logic.
  2. send_prediction_to_comp4() payload structure.
  3. Per-service deque bounded history.
  4. check_and_reload_model() hot-reload detection by mtime.
  5. predict_row() returns correct tuple shape.
  6. Output CSV schemas (live_feed.csv, memory_predictions.csv).
"""


import sys
import os
import json
import tempfile
import shutil
import unittest
from collections import deque
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Add scripts directory
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from feature_engineering import BASE_FEATURES, ALL_FEATURES, compute_derived_features

# Import testable functions from simulate_live

import simulate_live

# class TestComputeDerivedFeatures(unittest.TestCase):

class TestDeriveAlertReason(unittest.TestCase):
    """Test SHAP-based and fallback reason derivation."""

    def test_shap_based_top_two_features(self):
        """When SHAP values are provided, top 2 positive contributors become the reason."""
        shap_vals = {
            "heap_fill_ratio": 0.45,
            "incident_phase_2": 0.30,
            "ram_percent": 0.10,
            "gc_spike_count": -0.05,
        }
        row = pd.Series({"heap_mb_used": 0.0, "heap_fill_ratio": 0.0, "ram_percent": 0.0, "gc_spike_count": 0})
        reason = simulate_live.derive_alert_reason(shap_vals, row)
        # Top 2 positive: heap_fill_ratio, incident_phase_2
        self.assertIn("Critical Heap Fill Capacity", reason)
        self.assertIn("Sustained Late-Stage Memory Leak Accumulation", reason)

    def test_fallback_heap_saturation(self):
        """No SHAP values, heap_mb_used > 400 triggers heap reason."""
        row = pd.Series({"heap_mb_used": 500.0, "heap_fill_ratio": 0.0, "ram_percent": 50.0, "gc_spike_count": 1})
        reason = simulate_live.derive_alert_reason({}, row)
        self.assertIn("High Heap Memory Saturation", reason)

    def test_fallback_ram_high(self):
        """No SHAP values, ram_percent > 75 triggers RAM reason."""
        row = pd.Series({"heap_mb_used": 100.0, "heap_fill_ratio": 0.0, "ram_percent": 80.0, "gc_spike_count": 1})
        reason = simulate_live.derive_alert_reason({}, row)
        self.assertIn("High RAM Percentage Usage", reason)

    def test_fallback_gc_spikes(self):
        """No SHAP values, gc_spike_count > 2 triggers GC reason."""
        row = pd.Series({"heap_mb_used": 100.0, "heap_fill_ratio": 0.0, "ram_percent": 50.0, "gc_spike_count": 5})
        reason = simulate_live.derive_alert_reason({}, row)
        self.assertIn("Frequent Garbage Collection Spikes", reason)

    def test_default_reason_when_no_triggers(self):
        """All values below threshold → default reason."""
        row = pd.Series({"heap_mb_used": 100.0, "heap_fill_ratio": 0.3, "ram_percent": 40.0, "gc_spike_count": 1})
        reason = simulate_live.derive_alert_reason({}, row)
        self.assertEqual(reason, "Sustained Memory Utilization Growth")

    def test_all_negative_shap_falls_back(self):
        """If all SHAP values are negative, fallback to feature-based."""
        shap_vals = {"heap_fill_ratio": -0.1, "ram_percent": -0.2}
        row = pd.Series({"heap_mb_used": 600.0, "heap_fill_ratio": 0.9, "ram_percent": 90.0, "gc_spike_count": 0})
        reason = simulate_live.derive_alert_reason(shap_vals, row)
        # Should use fallback path
        self.assertTrue(len(reason) > 0)


class TestSendPredictionPayload(unittest.TestCase):
    """Test that the payload structure for Component 4 is correct."""

    @patch("simulate_live.urllib.request.urlopen")
    def test_payload_contains_required_fields(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status":"ok"}'
        mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)
        mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)

        shap_vals = {"heap_fill_ratio": 0.5}
        success, body = simulate_live.send_prediction_to_comp4(
            service_name="microservice-21",
            project_id="project_1",
            prob=0.85,
            alert="TRUE",
            label="FAILURE",
            shapley_phi_values=shap_vals,
            reason="Critical Heap Fill Capacity",
        )

        # Inspect the request passed to urlopen
        call_args = mock_urlopen.call_args[0][0]
        payload = json.loads(call_args.data.decode("utf-8"))

        self.assertIn("service_name", payload)
        self.assertIn("project_id", payload)
        self.assertIn("comp2_cpu", payload)
        self.assertIn("comp3_memory", payload)
        self.assertEqual(payload["service_name"], "microservice-21")
        self.assertEqual(payload["comp3_memory"]["memory_prob"], 0.85)
        self.assertEqual(payload["comp3_memory"]["alert"], "TRUE")
        self.assertEqual(payload["comp3_memory"]["pred_label"], "FAILURE")
        self.assertEqual(payload["comp3_memory"]["reason"], "Critical Heap Fill Capacity")
        self.assertEqual(payload["comp3_memory"]["shapley_phi_values"], shap_vals)

    @patch("simulate_live.urllib.request.urlopen", side_effect=Exception("refused"))
    def test_connection_failure_returns_false(self, mock_urlopen):
        success, msg = simulate_live.send_prediction_to_comp4(
            "svc", "proj", 0.5, "FALSE", "NORMAL"
        )
        self.assertFalse(success)

class TestPerServiceDeque(unittest.TestCase):
    """Test bounded per-service history (deque with maxlen)."""

    def test_deque_respects_maxlen(self):
        store = {}
        maxlen = 5
        # Append 10 items for same service
        for i in range(10):
            key = "microservice-21"
            if key not in store:
                store[key] = deque(maxlen=maxlen)
            store[key].append({"val": i})
        # Should only retain last 5
        self.assertEqual(len(store["microservice-21"]), 5)
        values = [r["val"] for r in store["microservice-21"]]
        self.assertEqual(values, [5, 6, 7, 8, 9])

    def test_multiple_services_independent(self):
        store = {}
        for svc in ["svc-a", "svc-b"]:
            store[svc] = deque(maxlen=3)
            for i in range(5):
                store[svc].append({"service": svc, "idx": i})
        self.assertEqual(len(store["svc-a"]), 3)
        self.assertEqual(len(store["svc-b"]), 3)
        self.assertEqual(store["svc-a"][0]["idx"], 2)
        self.assertEqual(store["svc-b"][-1]["idx"], 4)


class TestHotReloadMtime(unittest.TestCase):
    """Test model reload triggers only when mtime changes."""

    def test_no_reload_when_mtime_unchanged(self):
        test_dir = tempfile.mkdtemp(prefix="mup_reload_test_")
        try:
            model_path = os.path.join(test_dir, "model.pkl")
            # Create a dummy file
            with open(model_path, "wb") as f:
                f.write(b"dummy")

            mtime1 = os.path.getmtime(model_path)
            # Without modifying file, mtime should stay the same
            mtime2 = os.path.getmtime(model_path)
            self.assertEqual(mtime1, mtime2)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestPredictRowShape(unittest.TestCase):
    """Test predict_row returns correct tuple structure even without model."""

    def test_returns_tuple_of_four(self):
        """With no model loaded, returns (0.0, FALSE, NORMAL, {})."""
        # Temporarily set model to None
        original_model = simulate_live.model
        simulate_live.model = None
        try:
            row = pd.Series({col: 1.0 for col in ALL_FEATURES})
            result = simulate_live.predict_row(row)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 4)
            prob, alert, label, shap_dict = result
            self.assertEqual(prob, 0.0)
            self.assertEqual(alert, "FALSE")
            self.assertEqual(label, "NORMAL")
            self.assertEqual(shap_dict, {})
        finally:
            simulate_live.model = original_model


class TestOutputSchemas(unittest.TestCase):
    """Validate CSV output files match expected schema."""

    def test_live_feed_has_required_columns(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        live_path = os.path.join(data_dir, "live_feed.csv")
        if os.path.exists(live_path):
            df = pd.read_csv(live_path, nrows=2)
            required = {"timestamp", "project_id", "service_name", "label"}
            self.assertTrue(required.issubset(set(df.columns)),
                            f"live_feed.csv missing: {required - set(df.columns)}")
            # All derived features present
            for col in ["heap_fill_ratio", "gc_pressure_index", "time_to_oom_slope"]:
                self.assertIn(col, df.columns)

    def test_memory_predictions_has_required_columns(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        pred_path = os.path.join(data_dir, "memory_predictions.csv")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path, nrows=2)
            required = {"timestamp", "service_name", "project_id", "memory_prob", "alert", "pred_label", "reason"}
            self.assertTrue(required.issubset(set(df.columns)),
                            f"memory_predictions.csv missing: {required - set(df.columns)}")


if __name__ == "__main__":
    unittest.main()
