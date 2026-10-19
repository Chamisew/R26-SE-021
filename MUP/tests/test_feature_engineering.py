"""
Unit tests for MUP Component 3 -- feature_engineering.py

Validates:
  1. All derived features compute correctly with known inputs.
  2. Division-by-zero safety (epsilon fallback).
  3. NaN/Inf cleanup in derived columns.
  4. prepare_features() returns correct column list.
  5. Missing base columns default to zero without raising errors.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from feature_engineering import (
    compute_derived_features,
    prepare_features,
    BASE_FEATURES,
    DERIVED_FEATURES,
    ALL_FEATURES,
)


class TestComputeDerivedFeatures(unittest.TestCase):
    """Test each derived formula matches the documented specification."""

    def _make_df(self, **overrides):
        """Return a single-row DataFrame with sensible defaults, then apply overrides."""
        defaults = {
            "heap_mb_used": 600.0,
            "heap_max": 1024.0,
            "heap_rate": 5.0,
            "memory_change_5min": 50.0,
            "memory_change_10min": 80.0,
            "gc_spike_count": 10,
            "gc_count": 100,
            "ram_std": 0.05,
            "ram_mean": 0.70,
            "ram_std_trend": 0.02,
            "ram_percent": 65.0,
        }
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def test_heap_fill_ratio(self):
        df = self._make_df(heap_mb_used=512.0, heap_max=1024.0)
        result = compute_derived_features(df)
        expected = 512.0 / (1024.0 + 1e-6)
        self.assertAlmostEqual(result["heap_fill_ratio"].iloc[0], expected, places=6)

    def test_memory_acceleration(self):
        df = self._make_df(memory_change_5min=30.0, memory_change_10min=80.0)
        result = compute_derived_features(df)
        self.assertAlmostEqual(result["memory_acceleration"].iloc[0], -50.0, places=6)

    def test_gc_efficiency(self):
        df = self._make_df(gc_spike_count=5, gc_count=50)
        result = compute_derived_features(df)
        expected = 5.0 / (50.0 + 1e-6)
        self.assertAlmostEqual(result["gc_efficiency"].iloc[0], expected, places=6)

    def test_ram_std_ratio(self):
        df = self._make_df(ram_std=0.1, ram_mean=0.5)
        result = compute_derived_features(df)
        expected = 0.1 / (0.5 + 1e-6)
        self.assertAlmostEqual(result["ram_std_ratio"].iloc[0], expected, places=6)

    def test_memory_growth_velocity(self):
        df = self._make_df(memory_change_5min=20.0, ram_mean=0.8)
        result = compute_derived_features(df)
        expected = 20.0 / (0.8 + 1e-6)
        self.assertAlmostEqual(result["memory_growth_velocity"].iloc[0], expected, places=6)

    def test_time_to_oom_slope(self):
        df = self._make_df(heap_rate=10.0, heap_max=1000.0, heap_mb_used=900.0)
        result = compute_derived_features(df)
        # remaining_headroom = max(1000-900, 0) = 100
        expected = 10.0 / (100.0 + 1e-6)
        self.assertAlmostEqual(result["time_to_oom_slope"].iloc[0], expected, places=6)

    def test_gc_pressure_index(self):
        df = self._make_df(gc_spike_count=3, heap_mb_used=800.0, heap_max=1000.0)
        result = compute_derived_features(df)
        fill = 800.0 / (1000.0 + 1e-6)
        expected = 3.0 * fill
        self.assertAlmostEqual(result["gc_pressure_index"].iloc[0], expected, places=6)

    def test_ram_leak_momentum(self):
        df = self._make_df(ram_std_trend=0.04, ram_percent=50.0)
        result = compute_derived_features(df)
        expected = 0.04 * (50.0 / 100.0)
        self.assertAlmostEqual(result["ram_leak_momentum"].iloc[0], expected, places=6)

# class TestComputeDerivedFeatures(unittest.TestCase):
class TestSafetyGuards(unittest.TestCase):
    """Test edge cases and safety mechanisms."""

    def test_heap_max_zero_epsilon_safety(self):
        """Division by zero guarded by 1e-6 epsilon."""
        df = pd.DataFrame([{
            "heap_mb_used": 500.0, "heap_max": 0.0, "heap_rate": 1.0,
            "memory_change_5min": 0.0, "memory_change_10min": 0.0,
            "gc_spike_count": 0, "gc_count": 0,
            "ram_std": 0.0, "ram_mean": 0.0, "ram_std_trend": 0.0,
            "ram_percent": 0.0,
        }])
        result = compute_derived_features(df)
        # heap_fill_ratio = 500 / (0 + 1e-6) = 500,000,000
        self.assertTrue(np.isfinite(result["heap_fill_ratio"].iloc[0]))

    def test_no_infinite_values_in_output(self):
        """All Inf replaced with NaN then filled with 0."""
        df = pd.DataFrame([{
            "heap_mb_used": 0.0, "heap_max": 0.0, "heap_rate": 0.0,
            "memory_change_5min": 0.0, "memory_change_10min": 0.0,
            "gc_spike_count": 0, "gc_count": 0,
            "ram_std": 0.0, "ram_mean": 0.0, "ram_std_trend": 0.0,
            "ram_percent": 0.0,
        }])
        result = compute_derived_features(df)
        for col in DERIVED_FEATURES:
            self.assertFalse(np.isinf(result[col].iloc[0]), f"{col} is inf")
            self.assertFalse(np.isnan(result[col].iloc[0]), f"{col} is NaN")

    def test_missing_columns_default_to_zero(self):
        """A DataFrame without all base columns should still produce derived features."""
        df = pd.DataFrame([{"heap_mb_used": 100.0}])
        result = compute_derived_features(df)
        # Should not raise; all derived columns present
        for col in DERIVED_FEATURES:
            self.assertIn(col, result.columns)

    def test_output_preserves_original_columns(self):
        """compute_derived_features should keep all input columns."""
        df = pd.DataFrame([{"heap_mb_used": 100.0, "heap_max": 200.0, "custom_col": 42}])
        result = compute_derived_features(df)
        self.assertIn("custom_col", result.columns)
        self.assertEqual(result["custom_col"].iloc[0], 42)


class TestPrepareFeatures(unittest.TestCase):
    """Test the prepare_features() convenience wrapper."""

    def test_returns_df_and_all_features_list(self):
        df = pd.DataFrame([{col: 1.0 for col in BASE_FEATURES}])
        result_df, feature_list = prepare_features(df)
        self.assertEqual(feature_list, ALL_FEATURES)
        self.assertEqual(len(ALL_FEATURES), 22)
        for col in DERIVED_FEATURES:
            self.assertIn(col, result_df.columns)


if __name__ == "__main__":
    unittest.main()
