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

# ---- Live dashboard printer -------------------------------------------------
def _print_dashboard(
    elapsed_sec: int,
    total_sec: int,
    results_list: list,
    results_lock: threading.Lock,
    warning_counter: list,
    error_counter: list,
) -> None:
    """
    Print a live ASCII dashboard showing per-container stats.
    Called every INTERVAL seconds from the main thread.
    """
    elapsed_td = str(timedelta(seconds=elapsed_sec))
    total_td   = str(timedelta(seconds=total_sec))

    with results_lock:
        snapshot = list(results_list)  # shallow copy while holding lock

    total_rows = len(snapshot)

    # Aggregate latest row per container
    latest: dict[str, dict] = {}
    for row in snapshot:
        latest[row["service_name"]] = row  # last row wins

    warnings_count = warning_counter[0]
    errors_count   = error_counter[0]

    # Build dashboard lines
    border_width = 56
    header = f"  LIVE COLLECTION — {elapsed_td} elapsed / {total_td} total  "
    col_header = f"  {'Container':<24} {'RAM%':>6}  {'CPU%':>6}  {'Heap(MB)':>9}"

    print()
    print("╔" + "═" * border_width + "╗")
    print("║" + header.center(border_width) + "║")
    print("╠" + "═" * border_width + "╣")
    print("║" + col_header.ljust(border_width) + "║")

    if not latest:
        empty_line = "  (no data yet)"
        print("║" + empty_line.ljust(border_width) + "║")
    else:
        for cname, row in sorted(latest.items()):
            ram  = row["ram_percent"]
            cpu  = row["cpu_percent"]
            heap = row["heap_mb_used"]
            warn_flag = " ⚠️ " if ram > 70 else "    "
            line = (
                f"  {cname:<24} {ram:>5.1f}%  {cpu:>5.1f}%  {heap:>7.1f}MB"
                + warn_flag
            )
            print("║" + line.ljust(border_width) + "║")

    print("╚" + "═" * border_width + "╝")
    print(f"  Total rows: {total_rows} | Warnings: {warnings_count} | Errors: {errors_count}")

# ---- Main live-collection entry point ---------------------------------------
def stage1_collect_live(duration_minutes: int = 10) -> pd.DataFrame:
    """
    Connect to Docker Desktop, discover all running containers, and collect
    hardware stats + log lines every 5 seconds for `duration_minutes`.

    Returns a DataFrame with columns matching the Stage 2–5 schema:
        timestamp, service_name, stack, log_level, log_message,
        ram_percent, cpu_percent, heap_mb_used, gc_count,
        ground_truth_label, failure_type
    """
    INTERVAL_SEC  = 5
    DURATION_SEC  = duration_minutes * 60
    PROCESS_INTERVAL_SEC = 30

    print("\n" + "="*60)
    print("STAGE 1 — LIVE DOCKER COLLECTOR")
    print("="*60)
    print(f"  Duration   : {duration_minutes} minutes ({DURATION_SEC}s)")
    print(f"  Interval   : every {INTERVAL_SEC} seconds")

    # -- Connect to Docker Desktop -------------------------------------------
    try:
        import docker  # imported here so CSV mode works without docker installed
        try:
            client = docker.from_env()          # handles Windows named-pipe automatically
            client.ping()                        # raises if Docker is not reachable
        except Exception:
            # Explicit named-pipe fallback for Windows
            client = docker.DockerClient(base_url="npipe:////./pipe/docker_engine")
            client.ping()
    except Exception as exc:
        print(
            "\nERROR: Docker Desktop is not running.\n"
            "Please start Docker Desktop and try again.\n"
            f"(Detail: {exc})"
        )
        sys.exit(1)

    # -- Discover running containers -----------------------------------------
    containers = client.containers.list()
    if not containers:
        print(
            "\n  [WARN] No running containers found.\n"
            "  Start at least one Docker container and rerun the pipeline."
        )
        return pd.DataFrame(columns=[
            "timestamp", "service_name", "stack", "log_level", "log_message",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "ground_truth_label", "failure_type",
        ])

    print(f"\n  Discovered {len(containers)} running container(s):")
    for c in containers:
        img = c.image.tags[0] if c.image.tags else "<no tag>"
        print(f"    • {c.name:<35} image: {img}")

    # -- Shared state (thread-safe) ------------------------------------------
    results_list    : list  = []
    results_lock            = threading.Lock()
    error_counter   : list  = [0]   # mutable wrapper for thread mutation
    warning_counter : list  = [0]

    # -- Spawn one thread per container --------------------------------------
    threads = []
    for container in containers:
        t = threading.Thread(
            target=_collect_container,
            args=(
                container,
                DURATION_SEC,
                INTERVAL_SEC,
                results_list,
                results_lock,
                error_counter,
                warning_counter,
            ),
            daemon=True,
            name=f"collector-{container.name}",
        )
        t.start()
        threads.append(t)

    print(f"\n  Started {len(threads)} collector thread(s). Collecting …\n")

    # -- Dashboard loop (main thread) ----------------------------------------
    collection_start = time.monotonic()
    last_print_time  = collection_start
    next_process_time = collection_start + PROCESS_INTERVAL_SEC
    last_saved_raw_rows = 0

    with pipeline_state_lock:
        pipeline_state["status"] = "collecting"
        pipeline_state["started_at"] = datetime.now()
        pipeline_state["last_updated"] = datetime.now()

    try:
        while True:
            now     = time.monotonic()
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
                )
                # Brief inline progress line
                with results_lock:
                    n_rows = len(results_list)
                ram_vals = [
                    r["ram_percent"]
                    for r in results_list
                    if r["ram_percent"] > 0
                ]
                ram_avg = sum(ram_vals) / len(ram_vals) if ram_vals else 0.0
                active_containers = len(set(r["service_name"] for r in results_list))
                print(
                    f"  [{int(elapsed)}s] "
                    f"Containers: {active_containers} | "
                    f"Rows: {n_rows} | "
                    f"RAM avg: {ram_avg:.1f}%"
                )
                last_print_time = now

            # -- Periodic Stage 2-4 processing every 30s ----------------------
            if now >= next_process_time:
                try:
                    with results_lock:
                        snapshot = list(results_list)

                    if snapshot:
                        raw_snapshot_df = pd.DataFrame(snapshot)
                        raw_snapshot_df.sort_values("timestamp", inplace=True)
                        raw_snapshot_df.reset_index(drop=True, inplace=True)

                        # Incremental raw CSV append (live_raw_collection.csv)
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

                        # Run Stage 1.5 -> 2 -> 3 -> 4 on collected-so-far data
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
                            continuous_save(windows_df_partial, CONFIG)

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

            time.sleep(0.5)  # tight sleep so we don't overshoot much
    except KeyboardInterrupt:
        print("\n  [INFO] Live collection interrupted by user! Saving data collected so far...")

    # -- Wait for all threads to finish --------------------------------------
    for t in threads:
        t.join(timeout=INTERVAL_SEC + 2)

    # -- Build DataFrame -----------------------------------------------------
    if not results_list:
        print("\n  [WARN] No data collected — returning empty DataFrame.")
        df = pd.DataFrame(columns=[
            "timestamp", "service_name", "stack", "log_level", "log_message",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "ground_truth_label", "failure_type",
        ])
    else:
        df = pd.DataFrame(results_list)
        # Ensure exact column order expected by downstream stages
        df = df[[
            "timestamp", "service_name", "stack", "log_level", "log_message",
            "ram_percent", "cpu_percent", "heap_mb_used", "gc_count",
            "ground_truth_label", "failure_type",
        ]]
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"\n  Collection complete. Total rows: {len(df):,}")

    # -- Save raw CSV --------------------------------------------------------
    os.makedirs("output", exist_ok=True)
    raw_csv_path = CONFIG["live_output_csv"]
    df.to_csv(raw_csv_path, index=False)
    print(f"  Raw collection saved -> {raw_csv_path}")

    print("\n  [STAGE 1 LIVE COMPLETE]")
    with pipeline_state_lock:
        pipeline_state["status"] = "ready"
        pipeline_state["last_updated"] = datetime.now()
    return df

