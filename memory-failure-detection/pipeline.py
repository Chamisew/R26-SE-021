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

# =============================================================================
# STAGE 1 — LIVE DOCKER COLLECTOR  (live mode)
# =============================================================================

# ---- Helper: detect stack from image tag ------------------------------------
def _detect_stack(image_tag: str) -> str:
    """
    Map a Docker image tag string to a human-readable technology stack name.
    """
    tag = (image_tag or "").lower()
    if "python" in tag or "flask" in tag:
        return "Python/Flask"
    if "node" in tag:
        return "Node.js/Express"
    if "java" in tag or "spring" in tag:
        return "Java/SpringBoot"
    if "dotnet" in tag or "aspnet" in tag or "microsoft/dotnet" in tag:
        return ".NET/Core"
    if "golang" in tag or "/go" in tag or tag.startswith("go:"):
        return "Go/net_http"
    return "Unknown"


# ---- Helper: detect log level from a log line --------------------------------
def _detect_log_level(text: str) -> str:
    upper = text.upper()
    if any(k in upper for k in ("CRITICAL", "FATAL")):
        return "CRITICAL"
    if any(k in upper for k in ("ERROR", "EXCEPTION", "OOM")):
        return "ERROR"
    if any(k in upper for k in ("WARN", "WARNING")):
        return "WARNING"
    return "INFO"


# ---- Helper: check GC keyword presence --------------------------------------
GC_KEYWORDS = ("GC", "GARBAGE", "COLLECTION")

def _has_gc_keyword(text: str) -> bool:
    upper = text.upper()
    return any(k in upper for k in GC_KEYWORDS)


# ---- Per-container collector (runs in its own thread) -----------------------
def _collect_container(
    container,
    duration_seconds: int,
    interval_seconds: int,
    results_list: list,
    results_lock: threading.Lock,
    error_counter: list,
    warning_counter: list,
) -> None:
    """
    Collect stats + logs from a single container every `interval_seconds`.
    Appends row dicts to `results_list` (thread-safe via lock).
    Runs until `duration_seconds` have elapsed.
    """
    gc_cumulative = 0          # cumulative GC event counter for this container
    start_time    = time.monotonic()

    # Determine stack once from the image tag
    try:
        image_tag = container.image.tags[0] if container.image.tags else ""
    except Exception:
        image_tag = ""
    stack = _detect_stack(image_tag)

    while duration_seconds <= 0 or (time.monotonic() - start_time) < duration_seconds:
        tick_start = time.monotonic()
        row = {}

        # -- Refresh container state (may have stopped) -----------------------
        try:
            container.reload()
            if container.status != "running":
                print(f"  [WARN] Container '{container.name}' is no longer running — stopping its collector.")
                with results_lock:
                    warning_counter[0] += 1
                break
        except Exception as exc:
            print(f"  [WARN] Could not reload container '{container.name}': {exc}")
            with results_lock:
                warning_counter[0] += 1
            break

        # -- Docker STATS (stream=False → single snapshot, non-blocking) ------
        try:
            stats = container.stats(stream=False)

            # CPU %
            cpu_delta    = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"].get("system_cpu_usage", 0)
                - stats["precpu_stats"].get("system_cpu_usage", 0)
            )
            num_cpus = len(
                stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1])
            ) or 1
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0
            else:
                cpu_percent = 0.0

            # RAM
            mem_usage = stats["memory_stats"]["usage"]
            mem_limit = stats["memory_stats"]["limit"]
            ram_percent  = (mem_usage / mem_limit * 100.0) if mem_limit > 0 else 0.0
            heap_mb_used = mem_usage / (1024 * 1024)

        except (KeyError, TypeError, ZeroDivisionError) as exc:
            print(f"  [WARN] Incomplete stats for '{container.name}': {exc}")
            with results_lock:
                warning_counter[0] += 1
            # Sleep remainder of interval and retry
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0, interval_seconds - elapsed))
            continue
        except Exception as exc:
            print(f"  [ERROR] Stats API failure for '{container.name}': {exc}")
            with results_lock:
                error_counter[0] += 1
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0, interval_seconds - elapsed))
            continue

        # -- Docker LOGS (last 3 lines) ----------------------------------------
        try:
            raw_logs = container.logs(
                stdout=True, stderr=True,
                tail=3, timestamps=False
            )
            log_lines = raw_logs.decode("utf-8", errors="replace").strip().splitlines()
            log_message = " | ".join(log_lines[-3:]) if log_lines else ""
        except Exception as exc:
            print(f"  [WARN] Log decoding failed for '{container.name}': {exc}")
            with results_lock:
                warning_counter[0] += 1
            log_message = ""

        # -- Derived fields ---------------------------------------------------
        log_level = _detect_log_level(log_message)

        if _has_gc_keyword(log_message):
            gc_cumulative += 1

        row = {
            "timestamp"          : datetime.now(),
            "service_name"       : container.name,
            "stack"              : stack,
            "log_level"          : log_level,
            "log_message"        : log_message,
            "ram_percent"        : round(ram_percent, 4),
            "cpu_percent"        : round(cpu_percent, 4),
            "heap_mb_used"       : round(heap_mb_used, 4),
            "gc_count"           : gc_cumulative,
            "ground_truth_label" : "UNKNOWN",
            "failure_type"       : "none",
        }

        with results_lock:
            results_list.append(row)

        # Sleep for the remainder of the interval
        elapsed = time.monotonic() - tick_start
        time.sleep(max(0, interval_seconds - elapsed))
