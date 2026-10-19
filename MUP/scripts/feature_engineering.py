"""
Component 3 -- Feature Engineering Module

Modular feature engineering utility to consistently compute derived domain metrics
across training, model comparison, and live inference workflows.
"""

import pandas as pd
import numpy as np

BASE_FEATURES = [
    "memory_change_10min", "memory_change_5min", "heap_rate", "gc_spike_count",
    "ram_mean", "ram_max", "ram_std", "ram_std_trend",
    "heap_max", "gc_count", "ram_percent", "heap_mb_used",
    "incident_phase_1", "incident_phase_2",
]

DERIVED_FEATURES = [
    "heap_fill_ratio", "memory_acceleration", "gc_efficiency", "ram_std_ratio",
    "memory_growth_velocity", "time_to_oom_slope", "gc_pressure_index", "ram_leak_momentum"
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived domain features on raw telemetry DataFrame.

    Derived metrics:
      1. heap_fill_ratio        = heap_mb_used / (heap_max + 1e-6)
      2. memory_acceleration    = memory_change_5min - memory_change_10min
      3. gc_efficiency          = gc_spike_count / (gc_count + 1e-6)
      4. ram_std_ratio          = ram_std / (ram_mean + 1e-6)
      5. memory_growth_velocity = memory_change_5min / (ram_mean + 1e-6)
      6. time_to_oom_slope      = heap_rate / (max(heap_max - heap_mb_used, 0) + 1e-6)
      7. gc_pressure_index      = gc_spike_count * (heap_mb_used / (heap_max + 1e-6))
      8. ram_leak_momentum      = ram_std_trend * (ram_percent / 100.0)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing base metric columns.

    Returns
    -------
    pd.DataFrame
        Updated dataframe containing all base and derived feature columns.
    """
    df = df.copy()

    # Safe column extraction with fallbacks to zero if missing
    heap_mb_used = df.get("heap_mb_used", 0.0)
    heap_max = df.get("heap_max", 0.0)
    heap_rate = df.get("heap_rate", 0.0)
    mem_5m = df.get("memory_change_5min", 0.0)
    mem_10m = df.get("memory_change_10min", 0.0)
    gc_spikes = df.get("gc_spike_count", 0.0)
    gc_cnt = df.get("gc_count", 0.0)
    r_std = df.get("ram_std", 0.0)
    r_mean = df.get("ram_mean", 0.0)
    r_trend = df.get("ram_std_trend", 0.0)
    r_pct = df.get("ram_percent", 0.0)

    # Computations
    df["heap_fill_ratio"] = heap_mb_used / (heap_max + 1e-6)
    df["memory_acceleration"] = mem_5m - mem_10m
    df["gc_efficiency"] = gc_spikes / (gc_cnt + 1e-6)
    df["ram_std_ratio"] = r_std / (r_mean + 1e-6)

    # New Domain-Specific ML Engineered Features
    df["memory_growth_velocity"] = mem_5m / (r_mean + 1e-6)
    remaining_headroom = (heap_max - heap_mb_used).clip(lower=0.0)
    df["time_to_oom_slope"] = heap_rate / (remaining_headroom + 1e-6)
    df["gc_pressure_index"] = gc_spikes * (heap_mb_used / (heap_max + 1e-6))
    df["ram_leak_momentum"] = r_trend * (r_pct / 100.0)

    # Clean infinite/NaN values
    for col in DERIVED_FEATURES:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ensure all derived metrics are present and return (df, ALL_FEATURES)."""
    df_engineered = compute_derived_features(df)
    return df_engineered, ALL_FEATURES
