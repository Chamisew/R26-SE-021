"""
Sentinel AIOps — Decision Advisor package
=========================================

An advanced, operator-facing decision layer that sits ON TOP of the existing
Component 4 Balanced-Random-Forest + dual-threshold gating engine. It never
replaces the guaranteed fallback path; it only *recommends*.

Modules
-------
forecast    Time-to-failure regression (XGBoost quantile / sklearn fallback).
calibration In-fold isotonic calibration + ECE/Brier + split conformal bands.
policy      Cost-sensitive action recommender + off-policy evaluation.
prioritise  Fleet urgency ranking + cascade (correlated-alert) grouping.
drift       Per-service data/concept drift (PSI/KS + ADWIN/Page-Hinkley).
copilot     Deterministic <=60-word decision brief + real TreeExplainer.
feedback    Append-only, atomic operator-decision log (CheckpointManager style).

CLEAN-BOUNDARY RULE
-------------------
Component 3 memory outputs (P(Mem), alerts, labels) must NEVER enter a model
training feature vector. Every module here routes its feature list through
:func:`assert_no_memory_leakage`, and a dedicated pytest enforces it.
"""

from __future__ import annotations

import re

# The ten Component 2 CPU/queue signals — the ONLY raw inputs allowed into the
# temporal forecaster. Kept here so forecast/policy/drift share one source.
COMP2_CPU_FEATURE_COLS = [
    "cpu_percent",
    "cpu_velocity",
    "cpu_trend_5min",
    "cpu_trend_10min",
    "in_flight_queue",
    "incoming_rate",
    "processing_rate",
    "queue_growth_rate",
    "queue_pressure_index",
    "overload_flag",
]

RANDOM_STATE = 42

# Anything matching these patterns is a Component 3 / memory-derived column and
# is forbidden inside a training feature vector.
_MEMORY_LEAK_PATTERN = re.compile(
    r"(mem|memory|leak|ram|pred_label|oom|heap|gc_)", re.IGNORECASE
)


def is_memory_derived(column: str) -> bool:
    """True when a column name looks memory/Component-3 derived."""
    return bool(_MEMORY_LEAK_PATTERN.search(str(column)))


def assert_no_memory_leakage(feature_cols) -> None:
    """Raise if any feature column is memory/Component-3 derived.

    This is the automated guard the spec requires: it is called on every
    feature vector handed to a forecaster/policy model, and a pytest asserts it
    fires for a deliberately-poisoned column list.
    """
    offending = [str(c) for c in feature_cols if is_memory_derived(c)]
    if offending:
        raise ValueError(
            "Clean-boundary violation: memory-derived column(s) reached the "
            f"model input: {offending}. Component 3 outputs must never enter "
            "the Component 4 training feature vector."
        )


def allowed_comp2_only(feature_cols) -> list[str]:
    """Filter a column list down to pure Comp2 signal names (raises on leak)."""
    assert_no_memory_leakage(feature_cols)
    return [c for c in feature_cols if not is_memory_derived(c)]
