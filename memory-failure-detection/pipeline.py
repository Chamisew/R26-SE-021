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

def get_project_id(stack_name):
    global _NEXT_PROJECT_ID
    if stack_name not in _SERVICE_PROJECT_MAP:
        _SERVICE_PROJECT_MAP[stack_name] = f"project_{_NEXT_PROJECT_ID}"
        _NEXT_PROJECT_ID += 1
    return _SERVICE_PROJECT_MAP[stack_name]

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

        windows_df = add_failure_trends(windows_df)
        windows_df = add_ram_std_trend(windows_df)

        with pipeline_state_lock:
            last_saved_id = int(pipeline_state["last_window_id"])

        new_rows = windows_df[windows_df["window_id"] > last_saved_id].copy()
        if new_rows.empty:
            return 0

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
        ml_cols_raw = [
            "timestamp", "service_name", "window_id",
            "memory_growth", "heap_rate", "gc_spike_count",
            "ram_mean", "ram_max", "ram_std",
            "cpu_mean", "cpu_max", "heap_max",
            "gc_count", "ram_percent", "heap_mb_used",
            "failure_type", "hybrid_label",
            "ram_std_trend",
            "trend_slope_5m", "trend_max_failures_5m", "trend_variance_5m",
            "trend_slope_10m", "trend_max_failures_10m", "trend_variance_10m"
        ]
        for col in pipeline_cols + ml_cols_raw:
            if col not in new_rows.columns:
                new_rows[col] = np.nan

        write_header_pipeline = not os.path.exists(out_pipeline)
        new_rows[pipeline_cols].to_csv(
            out_pipeline,
            mode="a",
            header=write_header_pipeline,
            index=False
        )

        ml_df = new_rows[ml_cols_raw].copy()
        ml_df["project_id"] = new_rows["stack"].apply(get_project_id)
        ml_df["label"] = ml_df["hybrid_label"]
        ml_df["failure_type"] = ml_df.apply(lambda row: row["failure_type"] if row["label"] == "FAILURE" else "", axis=1)
        ml_df["incident_phase_1"] = 0
        ml_df["incident_phase_2"] = 0
        ml_df.drop(columns=["hybrid_label"], inplace=True)

        final_ml_cols = [
            "timestamp", "project_id", "service_name", "window_id",
            "memory_growth", "heap_rate", "gc_spike_count",
            "ram_mean", "ram_max", "ram_std", "ram_std_trend",
            "trend_slope_5m", "trend_max_failures_5m", "trend_variance_5m",
            "trend_slope_10m", "trend_max_failures_10m", "trend_variance_10m",
            "heap_max", "gc_count", "ram_percent", "heap_mb_used", 
            "label", "failure_type", "incident_phase_1", "incident_phase_2"
        ]
        ml_df = ml_df[final_ml_cols]

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
        return len(new_rows)

    except Exception as exc:
        print(f"  [SAVE ERROR] {exc}")
        return 0


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
    
    def calc_trend(group):
        return (group['ram_std'] - group['ram_std'].shift(n_windows)) / n_windows
        
    df['ram_std_trend'] = df.groupby('service_name', group_keys=False).apply(calc_trend)
    df['ram_std_trend'] = df['ram_std_trend'].fillna(0.0)
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

