"""
=============================================================================
pipeline.py
Stack-Agnostic Log Parsing and Hybrid Feature Extraction Pipeline
for Memory Failure Detection

Research: "Predicting Memory Leaks and CPU Spikes in Microservice Systems"
Component 1: SLIIT Final Year Dissertation
=============================================================================
"""

import os
import sys
import pickle
import warnings
import threading
import time
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# SHARED PIPELINE STATE (thread-safe)
# =============================================================================
pipeline_state_lock = threading.Lock()
pipeline_state = {
    "raw_df": None,           # latest raw collected DataFrame
    "windows_df": None,       # latest windows DataFrame
    "last_window_id": 0,      # track last saved window_id
    "status": "stopped",      # "collecting", "processing", "ready"
    "rows_collected": 0,
    "windows_generated": 0,
    "active_services": [],
    "started_at": None,
    "last_updated": None,
    "failure_count": 0,
    "normal_count": 0,
}

# =============================================================================
# CONFIG — edit these values to tune the pipeline
# =============================================================================
CONFIG = {
    "input_csv"              : "data/raw_logs_metrics.csv",
    "output_dir"             : "output/",
    "models_dir"             : "models/",
    "drain3_depth"           : 4,
    "drain3_sim_thresh"      : 0.4,
    "drain3_max_children"    : 100,
    "window_size"            : 12,
    "window_minutes"         : 5,
    "tfidf_threshold"        : 0.12,   # lowered: catches more semantic failures
    "metric_score_threshold" : 3,      # lowered: metric fusion upgrades sooner
    # Live collection settings
    "live_duration_minutes"  : -1,
    "live_output_csv"        : "output/live_raw_collection.csv",
}

# =============================================================================
# KEYWORD LISTS — memory and CPU failure signals
# =============================================================================
MEMORY_KEYWORDS = [
    "MEMORY LEAK", "MEMORY PRESSURE", "OUT OF MEMORY", "OOM",
    "HEAP GROWING", "HEAP EXHAUSTION", "HEAP SPACE",
    "OUTOFMEMORYERROR", "ALLOCATION FAILED", "ALLOCATION FAILURE",
    "GC OVERHEAD LIMIT", "HEAP USED CRITICAL", "MEMORY THRESHOLD",
    "HEAP OUT OF MEMORY", "CANNOT ALLOCATE", "MEMORY CRITICAL",
    "HEAP EXHAUSTED", "MEMORY LIMIT", "HEAP PRESSURE",
]

CPU_KEYWORDS = [
    "CPU SPIKE", "HIGH COMPUTATION", "CPU USAGE CRITICAL",
    "THREAD POOL EXHAUSTED", "COMPUTATION STARTED",
]

ALL_KEYWORDS = MEMORY_KEYWORDS + CPU_KEYWORDS

# =============================================================================
# TF-IDF FAILURE REFERENCE CORPUS
# =============================================================================
FAILURE_REFERENCE_CORPUS = [
    "memory leak heap allocation increasing",
    "out of memory error oom condition triggered",
    "heap growing allocated chunks memory pressure",
    "memory pressure critical heap exhaustion",
    "garbage collection overhead limit exceeded",
    "allocation failed heap used memory critical",
    "cpu spike high computation thread started",
    "memory threshold exceeded heap growing fast",
    "fatal heap exhaustion process memory critical",
    "java heap space outofmemoryerror gc overhead",
]


# =============================================================================
# STAGE 1 — DATA LOADER  (CSV mode)
# =============================================================================
def stage1_load_data(config):
    """
    Load raw_logs_metrics.csv, sort by timestamp, and report distributions.
    Returns a single sorted DataFrame.
    """
    print("\n" + "="*60)
    print("STAGE 1 — DATA LOADER (CSV mode)")
    print("="*60)

    try:
        csv_path = config["input_csv"]
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"  Total rows loaded : {len(df):,}")
        print(f"  Columns           : {list(df.columns)}")

        # Rows per service
        print("\n  Rows per service:")
        for svc, cnt in df.groupby("service_name").size().items():
            print(f"    {svc:<40} {cnt:>5} rows")

        # Label distribution
        label_dist = df["ground_truth_label"].value_counts()
        print("\n  Ground-truth label distribution:")
        for lbl, cnt in label_dist.items():
            pct = cnt / len(df) * 100
            print(f"    {lbl:<10} {cnt:>6} ({pct:.1f}%)")

        print("\n  [STAGE 1 COMPLETE]")
        return df

    except Exception as exc:
        print(f"\n  [STAGE 1 ERROR] {exc}")
        sys.exit(1)
