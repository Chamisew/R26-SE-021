import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Data path configuration (resolved relative to the workspace root) ────────
# These absolute paths are computed from this file's location so the pipeline
# works regardless of the current working directory — it can be launched from
# either the workspace root or the cpu_spike_predictor/ directory.
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CPU_SPIKE_PREDICTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Component 2 (CPU/queue) dataset — produced by the Queue-Aware CPU Spike Analyzer.
# Sourced directly from the c2 module instead of a local data/raw/ copy.
DEFAULT_COMP2_CSV = os.path.join(
    _WORKSPACE_ROOT, "c2", "Queue-Aware CPU Spike Analyzer",
    "research_framework", "final_research_dataset.csv",
)
# Component 3 (memory) dataset — produced by MUP (Memory Usage Predictor).
# Sourced directly from the MUP module instead of a local data/raw/ copy.
DEFAULT_COMP3_CSV = os.path.join(
    _WORKSPACE_ROOT, "MUP", "data", "memory_predictions.csv",
)

COMP2_REQUIRED_COLUMNS = {
    "incident_id", "timestamp", "time", "system_state", "incident_phase",
    "failing_service", "patient_zero", "cpu_percent", "cpu_velocity",
    "cpu_trend_5min", "cpu_trend_10min", "in_flight_queue", "incoming_rate",
    "processing_rate", "queue_growth_rate", "overload_flag",
    "queue_pressure_index", "incident_duration", "label"
}

COMP2_CPU_FEATURE_COLUMNS = [
    "cpu_percent",
    "cpu_velocity",
    "cpu_trend_5min",
    "cpu_trend_10min",
    "in_flight_queue",
    "incoming_rate",
    "processing_rate",
    "queue_growth_rate",
    "overload_flag",
    "queue_pressure_index",
]

COMP3_REQUIRED_COLUMNS = {
    "timestamp", "service_name", "project_id", "memory_prob", "alert", "pred_label"
}
