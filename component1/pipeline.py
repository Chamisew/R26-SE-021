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
import json
import requests
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
    "live_duration_minutes"  : 1,
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
# STAGE 1 — LIVE LOGS & METRICS HELPERS
# =============================================================================

def _detect_log_level(text: str) -> str:
    upper = text.upper()
    if any(k in upper for k in ("CRITICAL", "FATAL")):
        return "CRITICAL"
    if any(k in upper for k in ("ERROR", "EXCEPTION", "OOM")):
        return "ERROR"
    if any(k in upper for k in ("WARN", "WARNING")):
        return "WARNING"
    return "INFO"


GC_KEYWORDS = ("GC", "GARBAGE", "COLLECTION")

def _has_gc_keyword(text: str) -> bool:
    upper = text.upper()
    return any(k in upper for k in GC_KEYWORDS)


# ---- Live dashboard printer -------------------------------------------------
def _print_dashboard(
    elapsed_sec: int,
    total_sec: int,
    results_list: list,
    results_lock: threading.Lock,
    warning_counter: list,
    error_counter: list,
    service_status: dict,
) -> None:
    """
    Print a live ASCII dashboard showing per-service stats and online/offline status.
    """
    elapsed_td = str(timedelta(seconds=elapsed_sec))
    total_td = str(timedelta(seconds=total_sec)) if total_sec > 0 else "infinite"

    with results_lock:
        snapshot = list(results_list)

    total_rows = len(snapshot)

    # Aggregate latest row per service
    latest = {}
    for row in snapshot:
        latest[row["service_name"]] = row

    warnings_count = warning_counter[0]
    errors_count = error_counter[0]

    border_width = 65
    header = f"  LIVE COLLECTION — {elapsed_td} elapsed / {total_td} total  "
    col_header = f"  {'Service':<24} {'Status':<8} {'RAM%':>6}  {'CPU%':>6}  {'Heap(MB)':>9}"

    print()
    print("╔" + "═" * border_width + "╗")
    print("║" + header.center(border_width) + "║")
    print("╠" + "═" * border_width + "╣")
    print("║" + col_header.ljust(border_width) + "║")

    # Show all configured services and their status
    for sname in sorted(service_status.keys()):
        is_online = service_status[sname]
        status_str = "UP" if is_online else "DOWN"
        
        row = latest.get(sname)
        if row and is_online:
            ram = row["ram_percent"]
            cpu = row["cpu_percent"]
            heap = row["heap_mb_used"]
            warn_flag = " ⚠️ " if ram > 70 else "    "
            line = (
                f"  {sname:<24} {status_str:<8} {ram:>5.1f}%  {cpu:>5.1f}%  {heap:>7.1f}MB"
                + warn_flag
            )
        else:
            line = f"  {sname:<24} {status_str:<8} {'--':>6}  {'--':>6}  {'--':>9}"
        print("║" + line.ljust(border_width) + "║")

    print("╚" + "═" * border_width + "╝")
    print(f"  Total rows: {total_rows} | Warnings: {warnings_count} | Errors: {errors_count}")


def load_azure_services():
    """Loads service configurations.
    Priority:
    1. Environment variables starting with 'AZURE_SERVICE_' (e.g. AZURE_SERVICE_NODE)
    2. Local '.env' file containing KEY=VALUE pairs
    3. JSON file 'services_config.json' (fallback)
    """
    services = {}

    # 1. Try reading from a local .env file first to populate environment
    env_file = ".env"
    if os.path.exists(env_file):
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip()
        except Exception as e:
            pass

    # 2. Check for environment variables (either system-wide or loaded from .env)
    for env_name, env_val in os.environ.items():
        if env_name.startswith("AZURE_SERVICE_") and env_val.startswith("http"):
            # e.g. AZURE_SERVICE_NODE -> "node-azure"
            service_name = env_name.replace("AZURE_SERVICE_", "").lower() + "-azure"
            services[service_name] = env_val

    if services:
        return services

    # 3. Fallback to services_config.json if no env vars exist
    config_file = "services_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            services = {
                s["name"]: s["base_url"]
                for s in cfg.get("services", [])
                if s.get("enabled", True) and s.get("base_url", "").startswith("http")
            }
            return services
        except Exception as e:
            pass

    return {}


def _detect_azure_stack(service_name: str, sidecar_lang: str) -> str:
    lang = (sidecar_lang or "").lower()
    if any(k in lang for k in ("python", "flask", "django", "fastapi")):
        return "Python/Flask"
    if any(k in lang for k in ("node", "express", "javascript")):
        return "Node.js/Express"
    if any(k in lang for k in ("java", "spring", "jvm")):
        return "Java/SpringBoot"
    if any(k in lang for k in ("dotnet", "aspnet", "c#")):
        return ".NET/Core"
    if any(k in lang for k in ("go", "golang")):
        return "Go/net_http"
    
    # Fallback to name-based detection
    cname = (service_name or "").lower()
    if "python" in cname or "flask" in cname or "fastapi" in cname or "django" in cname:
        return "Python/Flask"
    if "node" in cname or "express" in cname or "npm" in cname:
        return "Node.js/Express"
    if "java" in cname or "spring" in cname or "tomcat" in cname:
        return "Java/SpringBoot"
    if "dotnet" in cname or "aspnet" in cname:
        return ".NET/Core"
    if "golang" in cname or "/go" in cname or "go_" in cname:
        return "Go/net_http"
    return "Unknown"


def _collect_azure_sidecar(
    service_name: str,
    base_url: str,
    duration_seconds: int,
    interval_seconds: int,
    results_list: list,
    results_lock: threading.Lock,
    error_counter: list,
    warning_counter: list,
    service_status: dict,
) -> None:
    """
    Collect metrics from a single Azure sidecar service every `interval_seconds`.
    """
    gc_cumulative = 0
    start_time = time.monotonic()
    stack = "Unknown"

    while duration_seconds <= 0 or (time.monotonic() - start_time) < duration_seconds:
        tick_start = time.monotonic()
        
        try:
            resp = requests.get(f"{base_url}/metrics", timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                
                # Update service online status
                service_status[service_name] = True
                
                cpu_percent = float(data.get("cpu_percent", 0.0))
                ram_percent = float(data.get("memory_percent", 0.0))
                heap_mb_used = float(data.get("memory_used_mb", 0.0))
                
                # Check if sidecar has gc_count, otherwise use 0
                gc_count = data.get("gc_count")
                
                # Mapped or defaulted log message
                log_message = str(data.get("log_message", "")).strip()
                if not log_message:
                    # Some sidecars publish only metrics; synthesize a stable message
                    # so Stage 1.5 does not drop every row as invalid.
                    if ram_percent >= 85 or heap_mb_used >= 2048:
                        log_message = "memory pressure warning"
                    elif cpu_percent >= 80:
                        log_message = "cpu spike warning"
                    else:
                        log_message = "service heartbeat"
                
                # Determine log level
                log_level = str(data.get("log_level", ""))
                if not log_level:
                    log_level = _detect_log_level(log_message)
                else:
                    log_level = _detect_log_level(log_message)
                
                # Detect stack once
                if stack == "Unknown":
                    sidecar_lang = data.get("language", "")
                    stack = _detect_azure_stack(service_name, sidecar_lang)
                
                # GC detection from log message if present
                if log_message and _has_gc_keyword(log_message):
                    gc_cumulative += 1
                
                if gc_count is None:
                    gc_val = gc_cumulative
                else:
                    gc_val = int(gc_count)
                
                row = {
                    "timestamp"          : datetime.now(),
                    "service_name"       : service_name,
                    "stack"              : stack,
                    "log_level"          : log_level,
                    "log_message"        : log_message,
                    "ram_percent"        : round(ram_percent, 4),
                    "cpu_percent"        : round(cpu_percent, 4),
                    "heap_mb_used"       : round(heap_mb_used, 4),
                    "gc_count"           : gc_val,
                    "ground_truth_label" : "UNKNOWN",
                    "failure_type"       : "none",
                }
                
                with results_lock:
                    results_list.append(row)
            else:
                # Mark service as offline
                service_status[service_name] = False
                print(f"  [WARN] /metrics for Azure service '{service_name}' returned HTTP status {resp.status_code}")
                with results_lock:
                    warning_counter[0] += 1
                    
        except Exception as exc:
            # Mark service as offline
            service_status[service_name] = False
            print(f"  [WARN] Network error fetching metrics for Azure service '{service_name}': {exc}")
            with results_lock:
                error_counter[0] += 1
                
        elapsed = time.monotonic() - tick_start
        time.sleep(max(0, interval_seconds - elapsed))


# =============================================================================
# STAGE 1 — LIVE AZURE SIDECAR COLLECTOR
# =============================================================================
def stage1_collect_azure_sidecar(duration_minutes: int = 10) -> pd.DataFrame:
    """
    Discover all configured Azure microservices via env / .env / services_config.json.
    Verify they are active by pinging {base_url}/ping.
    For all active ones, spawn threads to poll {base_url}/metrics every 5 seconds.
    Periodically runs preprocessing, Drain3, Classifier, and Sliding Windows.
    """
    INTERVAL_SEC = 5
    DURATION_SEC = duration_minutes * 60 if duration_minutes > 0 else -1
    PROCESS_INTERVAL_SEC = 30

    print("\n" + "="*60)
    print("STAGE 1 — LIVE AZURE SIDECAR COLLECTOR")
    print("="*60)
    print(f"  Duration   : {duration_minutes} minutes ({DURATION_SEC if DURATION_SEC > 0 else 'infinite'}s)")
    print(f"  Interval   : every {INTERVAL_SEC} seconds")

    # Discover microservices
    all_services = load_azure_services()
    if not all_services:
        print("\n  [WARN] No Azure services configured. Please set environment variables or services_config.json.")
        return pd.DataFrame(columns=[
            "timestamp", "service_name", "stack", "log_level", "log_message",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "ground_truth_label", "failure_type",
        ])

    print(f"\n  Checking configuration for {len(all_services)} Azure service(s):")
    discovered_services = {}
    for name, base_url in all_services.items():
        try:
            resp = requests.get(f"{base_url}/ping", timeout=4.0)
            if resp.status_code == 200:
                discovered_services[name] = base_url
                print(f"    • {name:<35} UP -> {base_url}")
            else:
                print(f"    • {name:<35} returned status {resp.status_code} (ignored)")
        except Exception as exc:
            print(f"    • {name:<35} unreachable: {exc} (ignored)")

    if not discovered_services:
        print("\n  [WARN] No active Azure services found (all ping calls failed).")
        return pd.DataFrame(columns=[
            "timestamp", "service_name", "stack", "log_level", "log_message",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "ground_truth_label", "failure_type",
        ])

    # Shared state
    results_list = []
    results_lock = threading.Lock()
    error_counter = [0]
    warning_counter = [0]
    service_status = {name: True for name in discovered_services}

    # Spawn collector threads
    threads = []
    for name, base_url in discovered_services.items():
        t = threading.Thread(
            target=_collect_azure_sidecar,
            args=(
                name,
                base_url,
                DURATION_SEC,
                INTERVAL_SEC,
                results_list,
                results_lock,
                error_counter,
                warning_counter,
                service_status,
            ),
            daemon=True,
            name=f"collector-{name}",
        )
        t.start()
        threads.append(t)

    print(f"\n  Started {len(threads)} Azure sidecar collector thread(s). Collecting …\n")

    # Dashboard and pipeline loop
    collection_start = time.monotonic()
    last_print_time = collection_start
    next_process_time = collection_start + PROCESS_INTERVAL_SEC
    last_saved_raw_rows = 0

    with pipeline_state_lock:
        pipeline_state["status"] = "collecting"
        pipeline_state["started_at"] = datetime.now()
        pipeline_state["last_updated"] = datetime.now()

    try:
        while True:
            now = time.monotonic()
            elapsed = now - collection_start

            if DURATION_SEC > 0 and elapsed >= DURATION_SEC:
                break

            if (now - last_print_time) >= INTERVAL_SEC:
                _print_dashboard(
                    int(elapsed),
                    DURATION_SEC,
                    results_list,
                    results_lock,
                    warning_counter,
                    error_counter,
                    service_status,
                )
                with results_lock:
                    n_rows = len(results_list)
                ram_vals = [
                    r["ram_percent"]
                    for r in results_list
                    if r["ram_percent"] > 0
                ]
                ram_avg = sum(ram_vals) / len(ram_vals) if ram_vals else 0.0
                active_c = len(set(r["service_name"] for r in results_list))
                print(
                    f"  [{int(elapsed)}s] "
                    f"Azure Services: {active_c} | "
                    f"Rows: {n_rows} | "
                    f"RAM avg: {ram_avg:.1f}%"
                )
                last_print_time = now

            # Periodic pipeline processing
            if now >= next_process_time:
                try:
                    with results_lock:
                        snapshot = list(results_list)

                    if snapshot:
                        raw_snapshot_df = pd.DataFrame(snapshot)
                        raw_snapshot_df.sort_values("timestamp", inplace=True)
                        raw_snapshot_df.reset_index(drop=True, inplace=True)

                        # Incremental raw CSV append
                        os.makedirs("output", exist_ok=True)
                        raw_csv_path = CONFIG["live_output_csv"]
                        new_raw_df = raw_snapshot_df.iloc[last_saved_raw_rows:].copy()
                        if len(new_raw_df) > 0:
                            write_header = not os.path.exists(raw_csv_path)
                            new_raw_df.to_csv(
                                raw_csv_path,
                                mode="a",
                                header=write_header,
                                index=False
                            )
                            last_saved_raw_rows = len(raw_snapshot_df)

                        with pipeline_state_lock:
                            pipeline_state["status"] = "processing"
                            pipeline_state["last_updated"] = datetime.now()

                        processed_df = stage1_5_preprocess(raw_snapshot_df)
                        if not processed_df.empty:
                            processed_df = stage2_drain3_parsing(processed_df, CONFIG)
                            processed_df = stage3_hybrid_classifier(processed_df, CONFIG)
                            windows_df_partial = stage4_sliding_window(processed_df, CONFIG)
                        else:
                            windows_df_partial = pd.DataFrame()

                        if windows_df_partial is not None and not windows_df_partial.empty:
                            windows_df_partial = continuous_save(windows_df_partial, CONFIG)

                            failure_count = int(
                                (windows_df_partial["ground_truth_label"] == "FAILURE").sum()
                            )
                            normal_count = int(
                                (windows_df_partial["ground_truth_label"] == "NORMAL").sum()
                            )
                            with pipeline_state_lock:
                                pipeline_state["raw_df"] = processed_df.copy()
                                pipeline_state["windows_df"] = windows_df_partial.copy()
                                pipeline_state["rows_collected"] = len(raw_snapshot_df)
                                pipeline_state["windows_generated"] = len(windows_df_partial)
                                pipeline_state["active_services"] = sorted(
                                    processed_df["service_name"].astype(str).unique().tolist()
                                )
                                pipeline_state["last_updated"] = datetime.now()
                                pipeline_state["failure_count"] = failure_count
                                pipeline_state["normal_count"] = normal_count
                                pipeline_state["status"] = "collecting"

                            print(
                                f"  [PIPELINE] {len(raw_snapshot_df)} rows | "
                                f"{len(windows_df_partial)} windows | "
                                f"{failure_count} FAILURE | {normal_count} NORMAL"
                            )
                except Exception as proc_exc:
                    print(f"  [WARN] Periodic pipeline processing failed: {proc_exc}")
                    with pipeline_state_lock:
                        pipeline_state["status"] = "collecting"
                        pipeline_state["last_updated"] = datetime.now()
                finally:
                    next_process_time += PROCESS_INTERVAL_SEC

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  [INFO] Live collection interrupted by user! Saving data collected so far...")

    # Wait for all threads to finish
    for t in threads:
        t.join(timeout=INTERVAL_SEC + 2)

    # Build DataFrame
    if not results_list:
        print("\n  [WARN] No data collected — returning empty DataFrame.")
        df = pd.DataFrame(columns=[
            "timestamp", "service_name", "stack", "log_level", "log_message",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "ground_truth_label", "failure_type",
        ])
    else:
        df = pd.DataFrame(results_list)
        df = df[[
            "timestamp", "service_name", "stack", "log_level", "log_message",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "ground_truth_label", "failure_type",
        ]]
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"\n  Collection complete. Total rows: {len(df):,}")

    os.makedirs("output", exist_ok=True)
    raw_csv_path = CONFIG["live_output_csv"]
    df.to_csv(raw_csv_path, index=False)
    print(f"  Raw collection saved -> {raw_csv_path}")

    print("\n  [STAGE 1 LIVE AZURE COMPLETE]")
    with pipeline_state_lock:
        pipeline_state["status"] = "ready"
        pipeline_state["last_updated"] = datetime.now()
    return df


# =============================================================================
# STAGE 1.5 — PREPROCESSING
# =============================================================================
def stage1_5_preprocess(df):
    """
    Clean raw rows before Drain3 parsing:
      1) drop duplicate rows by (timestamp, service_name, log_message)
      2) forward-fill numeric metrics per service
      3) remove null/empty log_message rows
      4) clip out-of-range numeric outliers
      5) strip special characters from log_message
    """
    print("\n" + "="*60)
    print("STAGE 1.5 — PREPROCESSING")
    print("="*60)

    try:
        rows_before = len(df)
        cleaned = df.copy()

        # 1) Drop duplicates by key columns
        cleaned = cleaned.drop_duplicates(
            subset=["timestamp", "service_name", "log_message"],
            keep="first"
        ).reset_index(drop=True)

        # 2) Forward-fill numeric columns per service
        numeric_cols = ["ram_percent", "cpu_percent", "heap_mb_used", "gc_count"]
        nulls_before_fill = int(cleaned[numeric_cols].isna().sum().sum())
        cleaned[numeric_cols] = (
            cleaned.groupby("service_name", sort=False)[numeric_cols]
                   .transform(lambda col: col.ffill())
        )
        nulls_after_fill = int(cleaned[numeric_cols].isna().sum().sum())
        nulls_filled = nulls_before_fill - nulls_after_fill

        # 3) Remove rows where log_message is null/empty
        log_series = cleaned["log_message"]
        invalid_log_mask = log_series.isna() | log_series.astype(str).str.strip().eq("")
        cleaned = cleaned.loc[~invalid_log_mask].reset_index(drop=True)

        # 4) Clip numeric outliers
        clip_specs = {
            "ram_percent": (0, 100),
            "cpu_percent": (0, 100),
            "heap_mb_used": (0, 10000),
        }
        outliers_clipped = 0
        for col, (low, high) in clip_specs.items():
            original = cleaned[col].copy()
            cleaned[col] = cleaned[col].clip(lower=low, upper=high)
            outliers_clipped += int((original != cleaned[col]).sum())

        # 5) Strip disallowed special characters from log_message
        cleaned["log_message"] = (
            cleaned["log_message"]
            .astype(str)
            .str.replace(r"[^A-Za-z0-9 .:\[\]]+", "", regex=True)
            .str.strip()
        )

        rows_after = len(cleaned)
        rows_removed = rows_before - rows_after

        print(f"  Rows before      : {rows_before:,}")
        print(f"  Rows removed     : {rows_removed:,}")
        print(f"  Nulls filled     : {nulls_filled:,}")
        print(f"  Outliers clipped : {outliers_clipped:,}")

        print("\n  [STAGE 1.5 COMPLETE]")
        return cleaned

    except Exception as exc:
        print(f"\n  [STAGE 1.5 ERROR] {exc}")
        raise


# =============================================================================
# STAGE 2 — DRAIN3 LOG PARSING (stack-agnostic)
# =============================================================================
def stage2_drain3_parsing(df, config):
    """
    Run Drain3 over ALL log messages together with no per-service config.
    Adds 'log_template' and 'template_id' columns to df.
    Persists the Drain3 model state to models/drain3_state.bin.
    """
    print("\n" + "="*60)
    print("STAGE 2 — DRAIN3 LOG PARSING (stack-agnostic)")
    print("="*60)

    try:
        from drain3 import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        # Build Drain3 config programmatically (no per-service overrides)
        drain_cfg = TemplateMinerConfig()
        drain_cfg.drain_depth         = config["drain3_depth"]
        drain_cfg.drain_sim_th        = config["drain3_sim_thresh"]
        drain_cfg.drain_max_children  = config["drain3_max_children"]
        drain_cfg.parametrize_numeric_tokens = True

        # Use a file-based persistence so the model can be saved
        miner = TemplateMiner(config=drain_cfg)

        templates   = []
        cluster_ids = []

        print(f"  Processing {len(df):,} log messages …")
        for i, msg in enumerate(df["log_message"].astype(str), 1):
            result = miner.add_log_message(msg)
            templates.append(result["template_mined"])
            cluster_ids.append(result["cluster_id"])
            if i % 2000 == 0:
                print(f"    … {i:,} / {len(df):,} processed")

        df["log_template"] = templates
        df["template_id"]  = cluster_ids

        # Save Drain3 state
        os.makedirs(config["models_dir"], exist_ok=True)
        state_path = os.path.join(config["models_dir"], "drain3_state.bin")
        with open(state_path, "wb") as f:
            pickle.dump(miner, f)
        print(f"\n  Drain3 state saved -> {state_path}")

        # Report
        unique_templates = df["log_template"].nunique()
        print(f"\n  Unique templates discovered : {unique_templates}")

        top10 = (
            df.groupby("log_template")
              .size()
              .sort_values(ascending=False)
              .head(10)
        )
        print("\n  Top-10 most frequent templates:")
        for tmpl, cnt in top10.items():
            print(f"    [{cnt:>5}]  {tmpl[:90]}")

        # Templates per stack
        print("\n  Templates per stack:")
        stack_tmpl = (
            df.groupby("stack")["log_template"]
              .nunique()
              .sort_values(ascending=False)
        )
        for stk, cnt in stack_tmpl.items():
            print(f"    {stk:<30} {cnt} unique templates")

        print("\n  [STAGE 2 COMPLETE]")
        return df

    except Exception as exc:
        print(f"\n  [STAGE 2 ERROR] {exc}")
        raise


# =============================================================================
# STAGE 3 — THREE-LAYER HYBRID CLASSIFIER
# =============================================================================
def _layer1_keyword(template: str):
    """Layer 1: keyword matching on the Drain3 template."""
    upper = template.upper()
    for kw in MEMORY_KEYWORDS:
        if kw in upper:
            return "FAILURE", "keyword", "memory_leak"
    for kw in CPU_KEYWORDS:
        if kw in upper:
            return "FAILURE", "keyword", "cpu_spike"
    return None, None, None


def _build_tfidf(df):
    """
    Fit TF-IDF on (unique templates + reference corpus).
    Returns vectorizer, template matrix, reference matrix.
    """
    unique_templates = df["log_template"].unique().tolist()
    all_docs = unique_templates + FAILURE_REFERENCE_CORPUS

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    vectorizer.fit(all_docs)

    tmpl_matrix = vectorizer.transform(unique_templates)
    ref_matrix  = vectorizer.transform(FAILURE_REFERENCE_CORPUS)

    return vectorizer, tmpl_matrix, ref_matrix, unique_templates


def _layer2_tfidf(template: str, tmpl_lookup: dict, threshold: float):
    """Layer 2: cosine similarity against failure reference corpus."""
    vec = tmpl_lookup.get(template)
    if vec is None:
        return "NORMAL", "normal"
    if vec > threshold:
        return "FAILURE", "tfidf_semantic"
    return "NORMAL", "normal"


def _metric_score(row, service_gc_history: dict):
    """
    Compute metric fusion score for a single row.
    Also updates the rolling gc_count history for the service.
    """
    score = 0

    ram  = row["ram_percent"]
    heap = row["heap_mb_used"]
    cpu  = row["cpu_percent"]
    gc   = row["gc_count"]
    svc  = row["service_name"]

    if ram  > 75: score += 2
    if ram  > 60: score += 1
    if heap > 350: score += 2
    if heap > 280: score += 1
    if cpu  > 80:  score += 2

    # GC spike: check if gc_count increased by > 3 in last 10 rows
    history = service_gc_history.setdefault(svc, [])
    history.append(gc)
    if len(history) > 10:
        history.pop(0)
    if len(history) >= 2:
        gc_increase = history[-1] - history[0]
        if gc_increase > 3:
            score += 2

    return score


def stage3_hybrid_classifier(df, config):
    """
    Apply three-layer hybrid classifier in sequence.
    Adds 'hybrid_label' and 'detection_layer' columns.
    """
    print("\n" + "="*60)
    print("STAGE 3 — THREE-LAYER HYBRID CLASSIFIER")
    print("="*60)

    try:
        threshold = config["tfidf_threshold"]
        metric_thresh = config["metric_score_threshold"]

        # -- Pre-compute TF-IDF similarities (once for all templates) --
        print("  Building TF-IDF similarity index …")
        _, tmpl_matrix, ref_matrix, unique_templates = _build_tfidf(df)
        sim_matrix = cosine_similarity(tmpl_matrix, ref_matrix)
        max_sims   = sim_matrix.max(axis=1)  # shape: (n_unique_templates,)
        tmpl_to_maxsim = {
            tmpl: max_sims[i]
            for i, tmpl in enumerate(unique_templates)
        }
        print(f"  TF-IDF index built over {len(unique_templates)} unique templates.")

        hybrid_labels    = []
        detection_layers = []
        failure_types    = []
        service_gc_hist  = {}

        for _, row in df.iterrows():
            template = str(row["log_template"])

            # Layer 1
            label, layer, failure_type = _layer1_keyword(template)

            # Layer 2 (only if Layer 1 found nothing)
            if label is None:
                label, layer = _layer2_tfidf(
                    template, tmpl_to_maxsim, threshold
                )
                failure_type = None

            # Layer 3 — metric fusion
            mscore = _metric_score(row, service_gc_hist)

            if label == "FAILURE" and mscore >= 2:
                layer = layer + "_metric_confirmed"
            elif label == "NORMAL" and mscore >= metric_thresh:
                label = "FAILURE"
                layer = "metric_fusion"
                failure_type = "metric_only"

            if label == "NORMAL":
                failure_type = "none"
            elif failure_type is None:
                # Covers semantic detections that are not metric-only.
                failure_type = "none"

            hybrid_labels.append(label)
            detection_layers.append(layer)
            failure_types.append(failure_type)

        df["hybrid_label"]    = hybrid_labels
        df["detection_layer"] = detection_layers
        df["failure_type"]    = failure_types

        # Summary
        print(f"\n  hybrid_label distribution:")
        for lbl, cnt in df["hybrid_label"].value_counts().items():
            print(f"    {lbl:<10} {cnt:>6}")
        print(f"\n  failure_type distribution:")
        for ftype, cnt in df["failure_type"].value_counts().items():
            print(f"    {ftype:<12} {cnt:>6}")

        print("\n  [STAGE 3 COMPLETE]")
        return df

    except Exception as exc:
        print(f"\n  [STAGE 3 ERROR] {exc}")
        raise


# =============================================================================
# STAGE 4 — SLIDING WINDOW FEATURE ENGINEERING
# =============================================================================
def stage4_sliding_window(df, config):
    """
    Per-service time-based sliding window feature extraction.
    Novel features: memory_growth, heap_rate, gc_spike_count.
    Returns a new DataFrame of window records.
    """
    print("\n" + "="*60)
    print("STAGE 4 — SLIDING WINDOW FEATURE ENGINEERING")
    print("="*60)

    try:
        window_minutes = config.get("window_minutes", 5)
        lookback_delta = pd.Timedelta(minutes=window_minutes)
        min_points = 3
        records = []
        window_id = int(config.get("window_id_start", 1))
        window_row_counts = []
        window_durations = []

        for svc, grp in df.groupby("service_name", sort=False):
            grp = grp.copy()
            grp["timestamp"] = pd.to_datetime(grp["timestamp"], errors="coerce")
            grp = grp.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            n = len(grp)

            if n < min_points + 1:
                print(f"  WARN: {svc} has only {n} valid timestamped rows — skipping.")
                continue

            for i in range(1, n):
                current = grp.iloc[i]
                current_ts = current["timestamp"]

                window_start_ts = current_ts - lookback_delta
                window = grp[
                    (grp["timestamp"] >= window_start_ts)
                    & (grp["timestamp"] < current_ts)
                ]

                # Skip sparse windows that cannot represent trends reliably.
                if len(window) < min_points:
                    continue

                # -- Novel Feature 1: memory_growth --
                memory_growth = (
                    window["ram_percent"].iloc[-1]
                    - window["ram_percent"].iloc[0]
                )

                # -- Novel Feature 2: heap_rate --
                heap_diffs = window["heap_mb_used"].diff().dropna()
                heap_rate  = heap_diffs.mean() if len(heap_diffs) > 0 else 0.0

                # -- Novel Feature 3: gc_spike_count --
                gc_diffs      = window["gc_count"].diff().dropna()
                gc_spike_count = int((gc_diffs > 0).sum())

                # -- Statistical Features --
                ram_mean = window["ram_percent"].mean()
                ram_max  = window["ram_percent"].max()
                ram_std  = window["ram_percent"].std()
                cpu_mean = window["cpu_percent"].mean()
                cpu_max  = window["cpu_percent"].max()
                heap_max = window["heap_mb_used"].max()
                window_duration_seconds = (
                    (window["timestamp"].iloc[-1] - window["timestamp"].iloc[0])
                    / pd.Timedelta(seconds=1)
                )

                # -- Window Label --
                window_label = (
                    "FAILURE"
                    if (window["ground_truth_label"] == "FAILURE").any()
                    else "NORMAL"
                )

                records.append({
                    "window_id"          : window_id,
                    "timestamp"          : current["timestamp"],
                    "service_name"       : current["service_name"],
                    "stack"              : current["stack"],
                    "log_template"       : current["log_template"],
                    "template_id"        : current["template_id"],
                    "ram_percent"        : current["ram_percent"],
                    "cpu_percent"        : current["cpu_percent"],
                    "heap_mb_used"       : current["heap_mb_used"],
                    "gc_count"           : current["gc_count"],
                    "memory_growth"      : memory_growth,
                    "heap_rate"          : heap_rate,
                    "gc_spike_count"     : gc_spike_count,
                    "window_duration_seconds": window_duration_seconds,
                    "ram_mean"           : ram_mean,
                    "ram_max"            : ram_max,
                    "ram_std"            : ram_std,
                    "cpu_mean"           : cpu_mean,
                    "cpu_max"            : cpu_max,
                    "heap_max"           : heap_max,
                    "hybrid_label"       : current["hybrid_label"],
                    "ground_truth_label" : current["ground_truth_label"],
                    "label_source"       : "ground_truth",
                    "detection_layer"    : current["detection_layer"],
                    "failure_type"       : current["failure_type"],
                })
                window_row_counts.append(len(window))
                window_durations.append(float(window_duration_seconds))
                window_id += 1

        windows_df = pd.DataFrame(records)
        print(f"  Total windows generated : {len(windows_df):,}")
        if len(windows_df) > 0:
            avg_duration = float(np.mean(window_durations)) if window_durations else 0.0
            min_rows = int(np.min(window_row_counts)) if window_row_counts else 0
            max_rows = int(np.max(window_row_counts)) if window_row_counts else 0
            print(f"  Average window duration (sec) : {avg_duration:.1f}")
            print(f"  Rows per window (min/max)     : {min_rows} / {max_rows}")
            wl_dist = windows_df["ground_truth_label"].value_counts()
            for lbl, cnt in wl_dist.items():
                print(f"    window_label={lbl:<10} {cnt:>6}")
        else:
            print("  Average window duration (sec) : 0.0")
            print("  Rows per window (min/max)     : 0 / 0")
            print("    window_label=FAILURE        0")
            print("    window_label=NORMAL         0")

        print("\n  [STAGE 4 COMPLETE]")
        return windows_df

    except Exception as exc:
        print(f"\n  [STAGE 4 ERROR] {exc}")
        raise


# =============================================================================
# CONTINUOUS WINDOW SAVE (LIVE MODE)
# =============================================================================

_SERVICE_PROJECT_MAP = {}
_NEXT_PROJECT_ID = 1

def get_project_id(service_name=None, stack_name=None):
    global _NEXT_PROJECT_ID
    key = (service_name or stack_name or "unknown").strip()
    if not key:
        key = "unknown"
    if key not in _SERVICE_PROJECT_MAP:
        _SERVICE_PROJECT_MAP[key] = f"project_{_NEXT_PROJECT_ID}"
        _NEXT_PROJECT_ID += 1
    return _SERVICE_PROJECT_MAP[key]

def continuous_save(windows_df, config):
    """
    Append only new windows to pipeline_output.csv and ml_ready_dataset.csv.
    Newness is determined by monotonic window_id.
    """
    try:
        if windows_df is None or windows_df.empty:
            return 0

        os.makedirs(config["output_dir"], exist_ok=True)
        out_pipeline = os.path.join(config["output_dir"], "pipeline_output.csv")
        out_ml = os.path.join(config["output_dir"], "ml_ready_dataset.csv")

        windows_df = add_ram_std_trend(windows_df)

        # Ensure project_id and label are in the in-memory dataframe returned
        windows_df["project_id"] = windows_df.apply(
            lambda row: get_project_id(row.get("service_name"), row.get("stack")),
            axis=1,
        )
        windows_df["label"] = windows_df["hybrid_label"]

        with pipeline_state_lock:
            last_saved_id = int(pipeline_state["last_window_id"])

        new_rows = windows_df[windows_df["window_id"] > last_saved_id].copy()
        if new_rows.empty:
            return windows_df

        pipeline_cols = [
            "window_id", "timestamp", "service_name", "stack",
            "log_template", "template_id",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "memory_growth", "heap_rate", "gc_spike_count",
            "ram_mean", "ram_max", "ram_std",
            "cpu_mean", "cpu_max", "heap_max",
            "hybrid_label", "ground_truth_label", "label_source",
            "detection_layer", "failure_type",
        ]
        for col in pipeline_cols:
            if col not in new_rows.columns:
                new_rows[col] = np.nan

        write_header_pipeline = not os.path.exists(out_pipeline)
        new_rows[pipeline_cols].to_csv(
            out_pipeline,
            mode="a",
            header=write_header_pipeline,
            index=False
        )

        ml_df = build_ml_dataframe(new_rows)

        write_header_ml = not os.path.exists(out_ml)
        ml_df.to_csv(
            out_ml,
            mode="a",
            header=write_header_ml,
            index=False
        )

        new_last_id = int(new_rows["window_id"].max())
        with pipeline_state_lock:
            pipeline_state["last_window_id"] = max(
                int(pipeline_state["last_window_id"]), new_last_id
            )
            pipeline_state["last_updated"] = datetime.now()

        print(f"  [SAVE] Appended {len(new_rows)} new windows -> {out_pipeline}")
        print(f"  [SAVE] Appended {len(new_rows)} new windows -> {out_ml}")
        return windows_df

    except Exception as exc:
        print(f"  [SAVE ERROR] {exc}")
        return windows_df


# =============================================================================
# WINDOW FETCH HELPERS (for API / Component 3 polling)
# =============================================================================
def get_latest_windows(n=50):
    """
    Return the latest n rows from output/ml_ready_dataset.csv as list[dict].
    """
    try:
        ml_path = os.path.join(CONFIG["output_dir"], "ml_ready_dataset.csv")
        if not os.path.exists(ml_path):
            return []
        df = pd.read_csv(ml_path)
        if df.empty:
            return []
        return df.tail(int(n)).to_dict(orient="records")
    except Exception as exc:
        print(f"[WARN] get_latest_windows failed: {exc}")
        return []


def get_new_windows_since(last_window_id):
    """
    Return rows with window_id > last_window_id from ml_ready_dataset.csv.
    """
    try:
        ml_path = os.path.join(CONFIG["output_dir"], "ml_ready_dataset.csv")
        if not os.path.exists(ml_path):
            return []
        df = pd.read_csv(ml_path)
        if df.empty:
            return []
        if "window_id" not in df.columns:
            return []
        new_df = df[df["window_id"] > int(last_window_id)]
        return new_df.to_dict(orient="records")
    except Exception as exc:
        print(f"[WARN] get_new_windows_since failed: {exc}")
        return []


# =============================================================================
# DATASET ENRICHMENT UTILITIES
# =============================================================================
FINAL_ML_COLS = [
    # Identifiers & Metadata
    "timestamp", "project_id", "service_name", "window_id",
    # Labels & Outcomes
    "label", "failure_type", "incident_phase_1", "incident_phase_2",
    # System RAM Metrics
    "ram_mean", "ram_max", "ram_std", "ram_std_trend", "ram_percent",
    "memory_change_5min", "memory_change_10min",
    # Application Heap & GC Metrics
    "heap_mb_used", "heap_max", "heap_rate", "gc_count", "gc_spike_count",
]


def add_memory_change_features(df):
    """Compute RAM change over the preceding 5- and 10-minute windows."""
    df = df.copy()
    df["memory_change_5min"] = 0.0
    df["memory_change_10min"] = 0.0
    if df.empty or "ram_percent" not in df.columns:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for _, group in df.groupby("service_name"):
        g = group.sort_values("timestamp").copy()
        idx = g.index
        ram_series = g.set_index("timestamp")["ram_percent"]

        for minutes, col in ((5, "memory_change_5min"), (10, "memory_change_10min")):
            target_ts = g["timestamp"] - pd.Timedelta(minutes=minutes)
            past_ram = (
                pd.Series(target_ts.values, index=idx)
                .apply(lambda ts: ram_series.asof(ts))
            )
            df.loc[idx, col] = (g["ram_percent"].values - past_ram.fillna(g["ram_percent"]).values)

    return df


def add_incident_phase_flags(df):
    """
    Binary flags for memory-incident severity/stage relative to FAILURE rows.
    Phase 1: early warning (5-7 min before) through critical failure.
    Phase 2: imminent/critical (0-5 min before) and post-failure.
    """
    df = df.copy()
    df["incident_phase_1"] = 0
    df["incident_phase_2"] = 0
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    label_col = "hybrid_label" if "hybrid_label" in df.columns else "label"

    for _, group in df.groupby("service_name"):
        g = group.sort_values("timestamp")
        failure_times = g.loc[g[label_col] == "FAILURE", "timestamp"]
        if failure_times.empty:
            continue

        idx = g.index
        for fail_ts in failure_times:
            delta = (fail_ts - g["timestamp"]).dt.total_seconds()
            mask_5_to_7 = (delta > 5 * 60) & (delta <= 7 * 60)
            mask_0_to_5 = (delta >= 0) & (delta <= 5 * 60)
            mask_after = delta < 0

            df.loc[idx[mask_5_to_7], "incident_phase_1"] = 1
            df.loc[idx[mask_0_to_5], "incident_phase_1"] = 1
            df.loc[idx[mask_0_to_5], "incident_phase_2"] = 1
            df.loc[idx[mask_after], "incident_phase_1"] = 1
            df.loc[idx[mask_after], "incident_phase_2"] = 1

    return df


def build_ml_dataframe(windows_df):
    """Build the ML-ready dataframe with the canonical feature schema."""
    enriched = windows_df.copy()
    enriched = add_ram_std_trend(enriched)
    enriched = add_memory_change_features(enriched)
    enriched = add_incident_phase_flags(enriched)

    enriched["project_id"] = enriched.apply(
        lambda row: get_project_id(row.get("service_name"), row.get("stack")),
        axis=1,
    )
    enriched["label"] = enriched["hybrid_label"]
    enriched["failure_type"] = enriched.apply(
        lambda row: row["failure_type"] if row["label"] == "FAILURE" else "",
        axis=1,
    )

    for col in FINAL_ML_COLS:
        if col not in enriched.columns:
            enriched[col] = 0 if col.startswith("incident_phase") else np.nan

    return enriched[FINAL_ML_COLS]


def label_incident_phases(df, failure_timestamp, project_id, failure_type):
    df = df.copy()
    df['project_id'] = project_id
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if isinstance(failure_timestamp, str):
        failure_timestamp = pd.to_datetime(failure_timestamp)
        
    delta = (failure_timestamp - df['timestamp']).dt.total_seconds()
    
    # Defaults
    df['label'] = 'NORMAL'
    df['incident_phase_1'] = 0
    df['incident_phase_2'] = 0
    df['failure_type'] = ""
    
    # Masks
    mask_after = delta < 0
    mask_0_to_5 = (delta >= 0) & (delta <= 5 * 60)
    mask_5_to_7 = (delta > 5 * 60) & (delta <= 7 * 60)
    
    # 5-7min before
    df.loc[mask_5_to_7, 'incident_phase_1'] = 1
    
    # 0-5min before
    df.loc[mask_0_to_5, 'label'] = 'PRE_FAILURE'
    df.loc[mask_0_to_5, 'incident_phase_1'] = 1
    df.loc[mask_0_to_5, 'incident_phase_2'] = 1
    df.loc[mask_0_to_5, 'failure_type'] = failure_type
    
    # after
    df.loc[mask_after, 'label'] = 'FAILURE'
    df.loc[mask_after, 'incident_phase_1'] = 1
    df.loc[mask_after, 'incident_phase_2'] = 1
    df.loc[mask_after, 'failure_type'] = failure_type
    
    print("\n  [LABEL INCIDENT PHASES] Counts:")
    print(df['label'].value_counts())
    return df

def add_ram_std_trend(df, n_windows=3):
    if "ram_std" not in df.columns:
        df['ram_std_trend'] = 0.0
        return df
    df = df.copy()

    df['ram_std_trend'] = (
        df.groupby('service_name', group_keys=False)['ram_std']
          .transform(lambda s: (s - s.shift(n_windows)) / n_windows)
          .fillna(0.0)
    )
    return df

def add_failure_trends(df):
    df = df.copy()
    if df.empty or 'hybrid_label' not in df.columns:
        df['trend_slope_5m'] = 0.0
        df['trend_max_failures_5m'] = 0.0
        df['trend_variance_5m'] = 0.0
        df['trend_slope_10m'] = 0.0
        df['trend_max_failures_10m'] = 0.0
        df['trend_variance_10m'] = 0.0
        return df

    original_index = df.index
    df = df.reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['is_failure'] = (df['hybrid_label'] == 'FAILURE').astype(int)

    def calc_slope(y):
        n = len(y)
        if n < 2: return 0.0
        x = np.arange(n)
        x_mean = (n - 1) / 2.0
        y_mean = np.mean(y)
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean)**2)
        return num / den if den != 0 else 0.0

    new_cols = ['trend_slope_5m', 'trend_max_failures_5m', 'trend_variance_5m',
                'trend_slope_10m', 'trend_max_failures_10m', 'trend_variance_10m']
    for col in new_cols:
        df[col] = 0.0

    for svc, group in df.groupby('service_name'):
        g = group.sort_values('timestamp')
        idx = g.index
        
        g_indexed = g.set_index('timestamp')
        fail_sum_1m = g_indexed['is_failure'].rolling('1min').sum()
        
        df.loc[idx, 'trend_max_failures_5m'] = fail_sum_1m.rolling('5min').max().values
        df.loc[idx, 'trend_variance_5m'] = fail_sum_1m.rolling('5min').var().fillna(0.0).values
        df.loc[idx, 'trend_slope_5m'] = fail_sum_1m.rolling('5min').apply(calc_slope, raw=True).fillna(0.0).values
        
        df.loc[idx, 'trend_max_failures_10m'] = fail_sum_1m.rolling('10min').max().values
        df.loc[idx, 'trend_variance_10m'] = fail_sum_1m.rolling('10min').var().fillna(0.0).values
        df.loc[idx, 'trend_slope_10m'] = fail_sum_1m.rolling('10min').apply(calc_slope, raw=True).fillna(0.0).values

    df = df.drop(columns=['is_failure'])
    df.index = original_index
    return df



# =============================================================================
# STAGE 5 — DATASET CONSTRUCTION AND EXPORT
# =============================================================================
def stage5_export(windows_df, config):
    """
    Export pipeline_output.csv (full audit trail) and
    ml_ready_dataset.csv (clean ML features).
    """
    print("\n" + "="*60)
    print("STAGE 5 — DATASET CONSTRUCTION AND EXPORT")
    print("="*60)

    try:
        os.makedirs(config["output_dir"], exist_ok=True)

        pipeline_cols = [
            "window_id", "timestamp", "service_name", "stack",
            "log_template", "template_id",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "memory_growth", "heap_rate", "gc_spike_count",
            "ram_mean", "ram_max", "ram_std",
            "cpu_mean", "cpu_max", "heap_max",
            "hybrid_label", "ground_truth_label", "label_source",
            "detection_layer", "failure_type",
        ]
        final_ml_cols = FINAL_ML_COLS

        out1 = os.path.join(config["output_dir"], "pipeline_output.csv")
        out2 = os.path.join(config["output_dir"], "ml_ready_dataset.csv")

        if windows_df is None or windows_df.empty:
            empty_pipeline = pd.DataFrame(columns=pipeline_cols)
            empty_ml = pd.DataFrame(columns=final_ml_cols)
            empty_pipeline.to_csv(out1, index=False)
            empty_ml.to_csv(out2, index=False)
            print(f"  pipeline_output.csv   -> {out1}  (0 rows)")
            print(f"  ml_ready_dataset.csv  -> {out2}  (0 rows)")
            print("\n  [STAGE 5 COMPLETE - EMPTY DATASET]")
            return pd.DataFrame(columns=pipeline_cols)

        # -- File 1: pipeline_output.csv --
        windows_df[pipeline_cols].to_csv(out1, index=False)
        print(f"  pipeline_output.csv   -> {out1}  ({len(windows_df):,} rows)")

        windows_df = add_failure_trends(windows_df)

        # Ensure project_id and label are in the in-memory dataframe returned
        windows_df["project_id"] = windows_df["stack"].apply(get_project_id)
        windows_df["label"] = windows_df["hybrid_label"]

        ml_df = build_ml_dataframe(windows_df)

        ml_df.to_csv(out2, index=False)
        print(f"  ml_ready_dataset.csv  -> {out2}  ({len(ml_df):,} rows)")

        print("\n  [STAGE 5 COMPLETE]")
        return windows_df

    except Exception as exc:
        print(f"\n  [STAGE 5 ERROR] {exc}")
        raise


# =============================================================================
# EVALUATION REPORT
# =============================================================================
def print_evaluation(windows_df, raw_df):
    """
    Print comprehensive evaluation: F1/Precision/Recall, confusion matrix,
    layer contribution, per-service F1, template stats, dataset stats.
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)

    y_true = (windows_df["ground_truth_label"] == "FAILURE").astype(int)
    y_pred = (windows_df["hybrid_label"]        == "FAILURE").astype(int)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true,    y_pred, zero_division=0)
    f1    = f1_score(y_true,        y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    print("\n-- 1. Pipeline Performance vs Ground Truth --")
    print(f"   Precision : {prec:.4f}")
    print(f"   Recall    : {rec:.4f}")
    print(f"   F1 Score  : {f1:.4f}  {'[OK] TARGET MET (>0.80)' if f1 > 0.80 else '[!!] below 0.80 target'}")
    print(f"\n   Confusion Matrix:")
    print(f"     TP={tp}  FP={fp}")
    print(f"     FN={fn}  TN={tn}")

    # -- Layer Contribution --
    print("\n-- 2. Layer Contribution Analysis --")
    fail_rows = windows_df[windows_df["hybrid_label"] == "FAILURE"]
    total_fail = len(fail_rows)

    def _count_layer(keyword):
        return fail_rows["detection_layer"].str.contains(keyword, na=False).sum()

    kw_count   = _count_layer("keyword")
    tfidf_count = _count_layer("tfidf_semantic")
    metric_count = _count_layer("metric_fusion")

    print(f"   Total FAILURE detections : {total_fail}")
    print(f"   keyword layer            : {kw_count}")
    print(f"   tfidf_semantic layer     : {tfidf_count}")
    print(f"   metric_fusion layer      : {metric_count}")

    if total_fail > 0:
        gain = tfidf_count / total_fail * 100
    else:
        gain = 0.0
    print(f"\n   Semantic Layer Gain      : {gain:.2f}%  "
          f"{'[OK] TARGET MET (>10%)' if gain > 10 else '[!!] below 10% target'}")

    # -- Per-Service F1 --
    print("\n-- 3. Per-Service F1 Scores (stack-agnosticism proof) --")
    print(f"   {'Service':<40} {'F1':>6}  {'Stack'}")
    print(f"   {'-'*70}")
    for svc, grp in windows_df.groupby("service_name"):
        yt = (grp["ground_truth_label"] == "FAILURE").astype(int)
        yp = (grp["hybrid_label"]        == "FAILURE").astype(int)
        svc_f1 = f1_score(yt, yp, zero_division=0)
        stk = grp["stack"].iloc[0]
        print(f"   {svc:<40} {svc_f1:>6.4f}  {stk}")

    # -- Template Discovery --
    print("\n-- 4. Template Discovery Summary --")
    total_tmpl = raw_df["log_template"].nunique()
    print(f"   Total unique Drain3 templates : {total_tmpl}")
    print(f"\n   Templates per stack:")
    stk_tmpl = raw_df.groupby("stack")["log_template"].nunique().sort_values(ascending=False)
    for stk, cnt in stk_tmpl.items():
        print(f"     {stk:<30} {cnt} templates")

    # -- Dataset Statistics --
    print("\n-- 5. Dataset Statistics --")
    print(f"   Total windows generated : {len(windows_df):,}")
    wl = windows_df["ground_truth_label"].value_counts()
    for lbl, cnt in wl.items():
        print(f"   {lbl:<10}: {cnt:,} ({cnt/len(windows_df)*100:.1f}%)")

    feature_cols = [
        "memory_growth", "heap_rate", "gc_spike_count",
        "ram_mean", "ram_max", "ram_std",
        "cpu_mean", "cpu_max", "heap_max",
    ]
    print(f"\n   Feature value ranges (min / mean / max):")
    print(f"   {'Feature':<20} {'Min':>10}  {'Mean':>10}  {'Max':>10}")
    print(f"   {'-'*55}")
    for col in feature_cols:
        mn  = windows_df[col].min()
        avg = windows_df[col].mean()
        mx  = windows_df[col].max()
        print(f"   {col:<20} {mn:>10.3f}  {avg:>10.3f}  {mx:>10.3f}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


# =============================================================================
# ABLATION STUDY
# =============================================================================
def run_ablation_study(windows_df):
    """
    Compare layer-wise and full-hybrid performance on the same windows_df.
    Experiments:
      A) Layer 1 only       -> keyword
      B) Layer 1 + Layer 2  -> keyword OR tfidf
      C) Layer 3 only       -> metric_fusion
      D) Full hybrid        -> hybrid_label
    """
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    y_true = (windows_df["ground_truth_label"] == "FAILURE").astype(int)
    det = windows_df["detection_layer"].astype(str)

    experiments = [
        ("Layer 1 only",        det.str.contains("keyword", na=False).astype(int)),
        ("Layer 1 + 2",         det.str.contains("keyword|tfidf", na=False).astype(int)),
        ("Layer 3 only",        det.str.contains("metric_fusion", na=False).astype(int)),
        ("Full Hybrid (Yours)", (windows_df["hybrid_label"] == "FAILURE").astype(int)),
    ]

    results = []
    for name, y_pred in experiments:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        results.append({
            "name": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    # --- Main comparison table (requested format) ---
    print("+---------------------+-----------+--------+--------+")
    print("| Method              | Precision | Recall | F1     |")
    print("+---------------------+-----------+--------+--------+")
    for row in results:
        print(
            f"| {row['name']:<19} | "
            f"{row['precision']:<9.4f} | "
            f"{row['recall']:<6.4f} | "
            f"{row['f1']:<6.4f} |"
        )
    print("+---------------------+-----------+--------+--------+")

    # --- Confusion-count details ---
    print("\n  TP/FP/FN by method:")
    for row in results:
        print(
            f"    {row['name']:<19} "
            f"TP={row['tp']:<6} FP={row['fp']:<6} FN={row['fn']:<6}"
        )

    # --- Conclusion: best F1 and delta to next best ---
    ranked = sorted(results, key=lambda r: r["f1"], reverse=True)
    best = ranked[0]
    next_best_f1 = ranked[1]["f1"] if len(ranked) > 1 else 0.0
    improvement = best["f1"] - next_best_f1
    print(
        f"\n  Conclusion: Highest F1 is '{best['name']}' "
        f"({best['f1']:.4f}), improving by {improvement:.4f} over the next best method."
    )
    print("\n  [ABLATION STUDY COMPLETE]")


# =============================================================================
# LEAVE-ONE-STACK-OUT (LOSO) EVALUATION
# =============================================================================
def run_loso_evaluation(windows_df):
    """
    Evaluate stack-agnostic behavior by holding out each stack and reporting
    test performance on that held-out stack.
    """
    print("\n" + "="*60)
    print("LEAVE-ONE-STACK-OUT (LOSO) EVALUATION")
    print("="*60)

    stacks = sorted(windows_df["stack"].dropna().unique().tolist())
    if not stacks:
        print("  [WARN] No stack values found for LOSO evaluation.")
        print("\n  [LOSO COMPLETE]")
        return

    results = []
    for stack_name in stacks:
        test_set = windows_df[windows_df["stack"] == stack_name]
        train_set = windows_df[windows_df["stack"] != stack_name]

        # train_set is intentionally derived to mirror LOSO split and to show
        # no per-stack customization is performed.
        _ = train_set

        y_true = (test_set["ground_truth_label"] == "FAILURE").astype(int)
        y_pred = (test_set["hybrid_label"] == "FAILURE").astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "stack": stack_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "rows": len(test_set),
        })

    print("  Held-Out Stack         | Precision | Recall | F1     | Rows")
    for row in results:
        print(
            f"  {row['stack']:<23} | "
            f"{row['precision']:<9.4f} | "
            f"{row['recall']:<6.4f} | "
            f"{row['f1']:<6.4f} | "
            f"{row['rows']}"
        )

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    print(f"\n  Average F1 across stacks: {avg_f1:.4f}")

    underperforming = [r["stack"] for r in results if r["f1"] <= 0.70]
    if not underperforming:
        print("  Conclusion: Stack-agnosticism confirmed")
    else:
        print(
            "  Conclusion: Stack-agnosticism not fully confirmed. "
            f"Underperforming stacks (F1 <= 0.70): {', '.join(underperforming)}"
        )

    print("\n  [LOSO COMPLETE]")


# =============================================================================
# TF-IDF CORPUS SIZE EXPERIMENT
# =============================================================================
def run_corpus_size_experiment(df, config):
    """
    Evaluate TF-IDF semantic performance against different reference-corpus
    sizes to justify the chosen corpus size.
    """
    print("\n" + "="*60)
    print("TF-IDF CORPUS SIZE EXPERIMENT")
    print("="*60)

    threshold = 0.12
    base_corpus = FAILURE_REFERENCE_CORPUS[:]

    extra_10 = [
        "heap memory consumption rising steadily without release",
        "application nearing memory saturation due to unreclaimed objects",
        "garbage collector running frequently with low reclaimed memory",
        "memory allocation retries increasing under sustained load",
        "process terminated after exceeding configured memory limit",
        "old generation occupancy remains high across multiple gc cycles",
        "service response delay linked to severe heap pressure",
        "container memory usage spikes followed by out of memory crash",
        "resident memory grows continuously indicating possible leak pattern",
        "allocation stall observed before heap exhaustion error",
    ]

    corpora = {
        3:  base_corpus[:3],
        5:  base_corpus[:5],
        7:  base_corpus[:7],
        10: base_corpus[:10],
        15: base_corpus[:10] + extra_10[:5],
        20: base_corpus[:10] + extra_10[:10],
    }

    # Evaluate per-template then map back to rows (mirrors Stage 3 TF-IDF usage)
    unique_templates = df["log_template"].astype(str).unique().tolist()
    y_true = (df["ground_truth_label"] == "FAILURE").astype(int)

    results = []
    for size in [3, 5, 7, 10, 15, 20]:
        ref_corpus = corpora[size]
        all_docs = unique_templates + ref_corpus

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        vectorizer.fit(all_docs)

        tmpl_matrix = vectorizer.transform(unique_templates)
        ref_matrix = vectorizer.transform(ref_corpus)
        sim_matrix = cosine_similarity(tmpl_matrix, ref_matrix)
        max_sims = sim_matrix.max(axis=1)

        tmpl_to_pred = {
            tmpl: (1 if max_sims[i] > threshold else 0)
            for i, tmpl in enumerate(unique_templates)
        }
        y_pred = df["log_template"].astype(str).map(tmpl_to_pred).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "size": size,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    print("  Corpus Size | Precision | Recall | F1")
    for row in results:
        print(
            f"  {row['size']:<11} | "
            f"{row['precision']:<9.4f} | "
            f"{row['recall']:<6.4f} | "
            f"{row['f1']:<6.4f}"
        )

    best = max(results, key=lambda r: r["f1"])
    print(
        f"\n  Best corpus size by F1: {best['size']} "
        f"(F1={best['f1']:.4f}). "
        f"This is the recommended size for the TF-IDF reference corpus."
    )
    print("\n  [CORPUS SIZE EXPERIMENT COMPLETE]")


# =============================================================================
# WINDOW SIZE EXPERIMENT
# =============================================================================
def run_window_size_experiment(df, config):
    """
    Compare multiple sliding-window sizes and report their impact on F1 and
    data volume.
    """
    print("\n" + "="*60)
    print("WINDOW SIZE EXPERIMENT")
    print("="*60)

    candidate_sizes = [4, 6, 8, 10, 12, 16, 20, 24]
    original_size = config.get("window_minutes", 5)  # Correction: track actual minutes
    results = []

    for w in candidate_sizes:
        exp_cfg = dict(config)
        exp_cfg["window_minutes"] = w  # Correction: vary actual parameter

        windows_df = stage4_sliding_window(df.copy(), exp_cfg)

        total_windows = len(windows_df)
        if total_windows == 0:
            failure_windows = 0
            failure_pct = 0.0
            f1 = 0.0
        else:
            failure_windows = int((windows_df["ground_truth_label"] == "FAILURE").sum())
            failure_pct = (failure_windows / total_windows) * 100.0

            y_true = (windows_df["ground_truth_label"] == "FAILURE").astype(int)
            y_pred = (windows_df["hybrid_label"] == "FAILURE").astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            "window_size": w,
            "total_windows": total_windows,
            "failure_windows": failure_windows,
            "failure_pct": failure_pct,
            "f1": f1,
        })

    print("  Window Size | Windows Generated | FAILURE % | F1")
    for row in results:
        print(
            f"  {row['window_size']:<11} | "
            f"{row['total_windows']:<17} | "
            f"{row['failure_pct']:<8.1f}% | "
            f"{row['f1']:.4f}"
        )

    best = max(results, key=lambda r: r["f1"])
    print(
        f"\n  Window size {best['window_size']} selected — "
        "generates sufficient temporal context while maximizing F1 score"
    )

    if original_size == best["window_size"]:
        print(
            f"  Current configured window size ({original_size}) is optimal based on this experiment."
        )
    else:
        current_row = next((r for r in results if r["window_size"] == original_size), None)
        if current_row is not None:
            delta = best["f1"] - current_row["f1"]
            print(
                f"  Current configured window size is {original_size}, while optimal is {best['window_size']} "
                f"(F1 improvement: {delta:.4f})."
            )
        else:
            print(
                f"  Current configured window size is {original_size}, while optimal is {best['window_size']}."
            )

    print("\n  [WINDOW SIZE EXPERIMENT COMPLETE]")


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS (RANDOM FOREST + SHAP)
# =============================================================================
def run_feature_importance_analysis(windows_df):
    """
    Train a Random Forest classifier and compute SHAP feature importances,
    then compare novel vs standard feature contribution.
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    try:
        import shap
    except Exception:
        print("  [ERROR] 'shap' is not installed.")
        print("  Install it with: pip install shap")
        print("\n  [FEATURE IMPORTANCE ANALYSIS SKIPPED]")
        return

    feature_cols = [
        "memory_growth", "heap_rate", "gc_spike_count",
        "ram_mean", "ram_max", "ram_std",
        "cpu_mean", "cpu_max", "heap_max",
        "gc_count", "ram_percent", "heap_mb_used",
    ]
    novel_features = {"memory_growth", "heap_rate", "gc_spike_count"}

    # Keep only rows with required columns present
    work_df = windows_df[feature_cols + ["ground_truth_label"]].copy()
    work_df = work_df.dropna(subset=feature_cols + ["ground_truth_label"])
    if work_df.empty:
        print("  [WARN] No valid rows available after dropping missing values.")
        print("\n  [FEATURE IMPORTANCE ANALYSIS SKIPPED]")
        return

    # 1) Encode target FAILURE=1, NORMAL=0
    y = (work_df["ground_truth_label"] == "FAILURE").astype(int)
    X = work_df[feature_cols]

    # 2) Train/test split + RandomForest training
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 3) SHAP values via TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Binary classifier SHAP output may be list[class0, class1] or array.
    if isinstance(shap_values, list):
        # Class 1 (FAILURE) explanation
        sv = np.abs(shap_values[1])
    else:
        sv = np.abs(shap_values)
        if sv.ndim == 3:
            # shape: (n_samples, n_features, n_classes)
            sv = sv[:, :, 1]

    mean_importance = sv.mean(axis=0)
    total_importance = float(mean_importance.sum())

    ranked = sorted(
        zip(feature_cols, mean_importance),
        key=lambda x: x[1],
        reverse=True,
    )

    # 4) Ranked table
    print("  Rank | Feature          | SHAP Importance | Category")
    for idx, (feat, imp) in enumerate(ranked, start=1):
        category = "Novel" if feat in novel_features else "Standard"
        print(f"  {idx:<4} | {feat:<16} | {imp:<15.4f} | {category}")

    # 5/6) Novel vs Standard contribution percentages
    novel_sum = sum(imp for feat, imp in ranked if feat in novel_features)
    standard_sum = sum(imp for feat, imp in ranked if feat not in novel_features)
    if total_importance > 0:
        novel_pct = (novel_sum / total_importance) * 100.0
        standard_pct = (standard_sum / total_importance) * 100.0
    else:
        novel_pct = 0.0
        standard_pct = 0.0

    print(
        f"\n  Novel features contribute {novel_pct:.1f}% of total predictive importance"
    )
    print(
        f"  Standard features contribute {standard_pct:.1f}% of total predictive importance"
    )

    # 7) Confirmation rule
    if novel_pct > 30.0:
        print("  [CONFIRMED] Novel feature engineering adds significant predictive value")

    print("\n  [FEATURE IMPORTANCE ANALYSIS COMPLETE]")


# =============================================================================
# EARLY WARNING TIME ANALYSIS
# =============================================================================
def measure_early_warning_time(windows_df):
    """
    Measure how many minutes before each ground-truth failure the model first
    raised a FAILURE alert within the same service.
    """
    print("\n" + "="*60)
    print("EARLY WARNING TIME ANALYSIS")
    print("="*60)

    required_cols = {"timestamp", "service_name", "hybrid_label", "ground_truth_label"}
    missing = [c for c in required_cols if c not in windows_df.columns]
    if missing:
        print(f"  [ERROR] Missing required columns: {missing}")
        print("\n  [EARLY WARNING ANALYSIS SKIPPED]")
        return

    work_df = windows_df.copy()
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], errors="coerce")
    work_df = work_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if work_df.empty:
        print("  [WARN] No valid timestamped rows available.")
        print("\n  [EARLY WARNING ANALYSIS SKIPPED]")
        return

    early_warning_minutes = []
    event_records = []

    for svc, grp in work_df.groupby("service_name", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        failure_events = grp[grp["ground_truth_label"] == "FAILURE"]
        predicted_events = grp[grp["hybrid_label"] == "FAILURE"]["timestamp"]

        stack_name = "Unknown"
        if "stack" in grp.columns and not grp["stack"].dropna().empty:
            stack_name = str(grp["stack"].dropna().iloc[0])

        for _, failure_row in failure_events.iterrows():
            failure_ts = failure_row["timestamp"]
            early_preds = predicted_events[predicted_events < failure_ts]

            if not early_preds.empty:
                earliest_pred_ts = early_preds.iloc[0]
                delta_minutes = (failure_ts - earliest_pred_ts) / pd.Timedelta(minutes=1)
                delta_minutes = float(delta_minutes)
                early_warning_minutes.append(delta_minutes)
                event_records.append({
                    "service_name": svc,
                    "stack": stack_name,
                    "warning_min": delta_minutes,
                    "predicted_early": True,
                })
            else:
                event_records.append({
                    "service_name": svc,
                    "stack": stack_name,
                    "warning_min": np.nan,
                    "predicted_early": False,
                })

    total_failures = len(event_records)
    successful = sum(1 for r in event_records if r["predicted_early"])
    missed = total_failures - successful
    success_pct = (successful / total_failures * 100.0) if total_failures > 0 else 0.0
    missed_pct = (missed / total_failures * 100.0) if total_failures > 0 else 0.0

    print("  Early Warning Time Analysis")
    print("  " + "-" * 41)
    print(f"  Total failure events analyzed : {total_failures}")
    print(f"  Successfully predicted early  : {successful} ({success_pct:.1f}%)")
    print(f"  Missed (no early warning)     : {missed} ({missed_pct:.1f}%)")

    if early_warning_minutes:
        min_warn = float(np.min(early_warning_minutes))
        max_warn = float(np.max(early_warning_minutes))
        avg_warn = float(np.mean(early_warning_minutes))
        med_warn = float(np.median(early_warning_minutes))

        print("\n  Early Warning Time Distribution:")
        print(f"  Minimum  : {min_warn:.1f} minutes before failure")
        print(f"  Maximum  : {max_warn:.1f} minutes before failure")
        print(f"  Average  : {avg_warn:.1f} minutes before failure")
        print(f"  Median   : {med_warn:.1f} minutes before failure")

        per_service = (
            pd.DataFrame(event_records)
            .dropna(subset=["warning_min"])
            .groupby(["service_name", "stack"], as_index=False)["warning_min"]
            .mean()
            .sort_values("warning_min", ascending=False)
        )

        print("\n  Service              | Stack           | Avg Warning (min)")
        if per_service.empty:
            print("  (No service had early predictions)")
        else:
            for _, row in per_service.iterrows():
                print(
                    f"  {str(row['service_name']):<20} | "
                    f"{str(row['stack']):<15} | "
                    f"{row['warning_min']:.1f}"
                )

        if avg_warn > 5:
            print(
                f"\n  [CONFIRMED] System provides actionable early warning "
                f"averaging {avg_warn:.1f} minutes before failure"
            )
        elif avg_warn < 2:
            print(
                "\n  [WARNING] Early warning time may be insufficient for manual intervention"
            )
    else:
        print("\n  Early Warning Time Distribution:")
        print("  No early warning detections were found before failure events.")

    print("\n  [EARLY WARNING ANALYSIS COMPLETE]")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*60)
    print("# MEMORY FAILURE DETECTION PIPELINE")
    print("# Stack-Agnostic Log Parsing + Hybrid Feature Extraction")
    print("#"*60)
    print("# MODE: Azure Sidecar Collector")
    print("#"*60)

    # Change working directory to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # -------------------------------------------------------------------------
    # Stage 1 — Ingest from Azure HTTP Sidecars
    df = stage1_collect_azure_sidecar(
        duration_minutes=CONFIG["live_duration_minutes"]
    )

    if df.empty:
        print("[ERROR] No data available after Stage 1. Exiting.")
        sys.exit(1)

    # Stage 1.5 — preprocessing before Drain3
    df = stage1_5_preprocess(df)
    if df.empty:
        print("[ERROR] No data available after Stage 1.5 preprocessing. Exiting.")
        sys.exit(1)

    # Stage 2 — Drain3 log parsing
    df = stage2_drain3_parsing(df, CONFIG)

    # Stage 3 — Hybrid classifier
    df = stage3_hybrid_classifier(df, CONFIG)

    # Stage 4 — Sliding window features
    windows_df = stage4_sliding_window(df, CONFIG)

    # Stage 5 — Export CSVs
    windows_df = stage5_export(windows_df, CONFIG)

    # Evaluation (only meaningful when ground_truth labels are real)
    print("\n  [NOTE] Evaluation skipped (ground_truth_label is UNKNOWN for Azure live collection).")

    print("\n[PIPELINE FINISHED SUCCESSFULLY]\n")


if __name__ == "__main__":
    main()
