"""
FastAPI wrapper for Component 1 memory-failure pipeline.
Provides REST + WebSocket APIs for dashboard and Component 3 ingestion.
"""

import os
import threading
import time
import asyncio
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import f1_score

import pipeline

PIPELINE_MODE = os.getenv("PIPELINE_MODE", "live")

try:
    PIPELINE_DURATION_MINUTES = int(os.getenv("PIPELINE_DURATION_MINUTES", "-1"))
except ValueError:
    PIPELINE_DURATION_MINUTES = -1


class SharedState:
    """Thread-safe in-memory state used by all endpoints."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.status = "stopped"          # stopped | running | collecting
        self.mode = PIPELINE_MODE

        self.raw_rows: list[dict[str, Any]] = []
        self.windows_rows: list[dict[str, Any]] = []

        self.total_rows_collected = 0
        self.total_windows_generated = 0
        self.last_global_window_id = 0
        self.active_containers: list[str] = []
        self.last_error: str | None = None


STATE = SharedState()

app = FastAPI(title="Memory Failure Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize_value(v: Any) -> Any:
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if pd.isna(v):
        return None
    return v


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    records = []
    for rec in df.to_dict(orient="records"):
        records.append({k: _sanitize_value(v) for k, v in rec.items()})
    return records


def _run_pipeline_once(mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stage sequence once and return (raw_df, windows_df)."""
    cfg = dict(pipeline.CONFIG)
    if mode == "csv":
        if hasattr(pipeline, "stage1_load_data"):
            raw_df = pipeline.stage1_load_data(cfg)
        else:
            raise RuntimeError("CSV mode is not supported in the Azure refactor. Set PIPELINE_MODE=live.")
    else:
        duration_minutes = cfg.get("live_duration_minutes", PIPELINE_DURATION_MINUTES)
        if hasattr(pipeline, "stage1_collect_azure_sidecar"):
            raw_df = pipeline.stage1_collect_azure_sidecar(
                duration_minutes=duration_minutes
            )
        elif hasattr(pipeline, "stage1_collect_live"):
            raw_df = pipeline.stage1_collect_live(
                duration_minutes=duration_minutes
            )
        else:
            raise AttributeError("No Azure or legacy live collector is available in pipeline.py")

    if raw_df.empty:
        return raw_df, pd.DataFrame()

    raw_df = pipeline.stage1_5_preprocess(raw_df)
    if raw_df.empty:
        return raw_df, pd.DataFrame()

    raw_df = pipeline.stage2_drain3_parsing(raw_df, cfg)
    raw_df = pipeline.stage3_hybrid_classifier(raw_df, cfg)
    windows_df = pipeline.stage4_sliding_window(raw_df, cfg)
    if mode == "csv" and windows_df is not None and not windows_df.empty:
        windows_df = pipeline.stage5_export(windows_df, cfg)
    return raw_df, windows_df


def _pipeline_worker() -> None:
    """Background thread that continuously runs data collection + stages."""
    with STATE.lock:
        STATE.status = "running"
        STATE.started_at = time.time()
        mode = STATE.mode

    if mode == "live":
        # Start pipeline collector in a separate daemon thread with exception handling
        def run_collector():
            try:
                duration_minutes = PIPELINE_DURATION_MINUTES
                if hasattr(pipeline, "stage1_collect_azure_sidecar"):
                    pipeline.CONFIG["live_duration_minutes"] = duration_minutes
                    pipeline.stage1_collect_azure_sidecar(duration_minutes=duration_minutes)
                elif hasattr(pipeline, "stage1_collect_live"):
                    cfg = dict(pipeline.CONFIG)
                    cfg["live_duration_minutes"] = duration_minutes
                    pipeline.stage1_collect_live(duration_minutes=duration_minutes)
                else:
                    raise AttributeError("No live collector exists in pipeline.py")
            except SystemExit as se:
                with STATE.lock:
                    STATE.status = "stopped"
                    STATE.last_error = f"Collector thread exited (SystemExit: {se.code})"
            except BaseException as exc:
                with STATE.lock:
                    STATE.status = "stopped"
                    STATE.last_error = f"Collector thread failed: {str(exc)}"

        collector_thread = threading.Thread(target=run_collector, daemon=True)
        collector_thread.start()

        # Continuously sync pipeline state into app.py STATE
        while True:
            with pipeline.pipeline_state_lock:
                raw_df = pipeline.pipeline_state.get("raw_df")
                win_df = pipeline.pipeline_state.get("windows_df")
                
            with STATE.lock:
                if raw_df is not None and not raw_df.empty:
                    STATE.raw_rows = _df_to_records(raw_df)
                    STATE.total_rows_collected = len(STATE.raw_rows)
                    STATE.active_containers = sorted(
                        list({r.get("service_name") for r in STATE.raw_rows if r.get("service_name")})
                    )
                if win_df is not None and not win_df.empty:
                    STATE.windows_rows = _df_to_records(win_df)
                    STATE.total_windows_generated = len(STATE.windows_rows)
                    if "window_id" in win_df.columns:
                        STATE.last_global_window_id = int(win_df["window_id"].max())
                # Keep status as running if no collector thread error has been recorded
                if STATE.status != "stopped":
                    STATE.status = "running"
            time.sleep(2)
    else:
        # CSV Mode: run once for static datasets, re-run only when the file changes
        last_mtime = None
        has_run_once = False
        while True:
            try:
                csv_path = pipeline.CONFIG.get("input_csv", "data/raw_logs_metrics.csv")
                current_mtime = None
                if os.path.exists(csv_path):
                    current_mtime = os.path.getmtime(csv_path)

                should_run = not has_run_once or (current_mtime is not None and current_mtime != last_mtime)

                if should_run:
                    with STATE.lock:
                        STATE.status = "collecting"

                    raw_df, windows_df = _run_pipeline_once("csv")

                    with STATE.lock:
                        raw_records = _df_to_records(raw_df)
                        if raw_records:
                            STATE.raw_rows = raw_records
                            STATE.total_rows_collected = len(raw_records)
                            STATE.active_containers = sorted(
                                list({r.get("service_name") for r in raw_records if r.get("service_name")})
                            )

                        if windows_df is not None and not windows_df.empty:
                            window_records = _df_to_records(windows_df)
                            STATE.windows_rows = window_records
                            STATE.total_windows_generated = len(window_records)
                            if "window_id" in windows_df.columns:
                                STATE.last_global_window_id = int(windows_df["window_id"].max())

                        STATE.status = "running"
                        STATE.last_error = None
                    last_mtime = current_mtime
                    has_run_once = True
            except Exception as exc:
                with STATE.lock:
                    STATE.status = "running"
                    STATE.last_error = str(exc)
                time.sleep(5)
            
            time.sleep(10)


@app.on_event("startup")
def on_startup() -> None:
    worker = threading.Thread(target=_pipeline_worker, daemon=True, name="pipeline-worker")
    worker.start()


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    with STATE.lock:
        return {
            "status": STATE.status,
            "mode": STATE.mode,
            "uptime_seconds": int(time.time() - STATE.started_at),
            "total_rows_collected": STATE.total_rows_collected,
            "total_windows_generated": STATE.total_windows_generated,
            "last_global_window_id": STATE.last_global_window_id,
            "active_containers": STATE.active_containers,
            "last_error": STATE.last_error,
        }


@app.get("/api/live-metrics")
def api_live_metrics() -> list[dict[str, Any]]:
    with STATE.lock:
        if not STATE.raw_rows:
            return []
        raw_df = pd.DataFrame(STATE.raw_rows)

    if raw_df.empty:
        return []

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    raw_df = raw_df.sort_values("timestamp")
    latest = raw_df.groupby("service_name", as_index=False).tail(1)

    cols = [
        "service_name", "stack", "timestamp", "ram_percent", "cpu_percent",
        "heap_mb_used", "gc_count", "log_level", "log_message",
        "hybrid_label", "failure_type",
    ]
    for c in cols:
        if c not in latest.columns:
            latest[c] = None

    return _df_to_records(latest[cols])


@app.get("/api/windows")
def api_windows(limit: int = Query(default=50, ge=1, le=1000)) -> list[dict[str, Any]]:
    with STATE.lock:
        windows = STATE.windows_rows[-limit:]
    return windows


@app.get("/api/templates")
def api_templates() -> list[dict[str, Any]]:
    with STATE.lock:
        if not STATE.raw_rows:
            return []
        raw_df = pd.DataFrame(STATE.raw_rows)

    if raw_df.empty or "log_template" not in raw_df.columns:
        return []

    grouped = (
        raw_df.groupby("log_template")
        .agg(
            count=("log_template", "size"),
            failure_related=("hybrid_label", lambda s: bool((s == "FAILURE").any())),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(20)
    )
    grouped.rename(columns={"log_template": "template"}, inplace=True)
    return _df_to_records(grouped[["template", "count", "failure_related"]])


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    with STATE.lock:
        if not STATE.windows_rows:
            return {
                "total_windows": 0,
                "failure_windows": 0,
                "normal_windows": 0,
                "failure_percent": 0.0,
                "failure_rate": 0.0,
                "layer_contribution": {"keyword": 0, "tfidf_semantic": 0, "metric_fusion": 0},
                "per_service_f1": {},
                "overall_f1": None,
                "trend_5m": [],
            }
        wdf = pd.DataFrame(STATE.windows_rows)

    total_windows = len(wdf)
    failure_windows = int((wdf["hybrid_label"] == "FAILURE").sum())
    normal_windows = total_windows - failure_windows
    failure_percent = (failure_windows / total_windows * 100.0) if total_windows > 0 else 0.0

    layers = wdf["detection_layer"].astype(str)
    layer_contribution = {
        "keyword": int(layers.str.contains("keyword", na=False).sum()),
        "tfidf_semantic": int(layers.str.contains("tfidf_semantic", na=False).sum()),
        "metric_fusion": int(layers.str.contains("metric_fusion", na=False).sum()),
    }

    # Issue 7: Check if we have any valid ground truth labels before calculating F1
    has_gt = wdf["ground_truth_label"].isin(["FAILURE", "NORMAL"]).any()

    per_service_f1: dict[str, float | None] = {}
    for svc, grp in wdf.groupby("service_name"):
        svc_has_gt = grp["ground_truth_label"].isin(["FAILURE", "NORMAL"]).any()
        if not svc_has_gt:
            per_service_f1[str(svc)] = None
        else:
            y_true = (grp["ground_truth_label"] == "FAILURE").astype(int)
            y_pred = (grp["hybrid_label"] == "FAILURE").astype(int)
            per_service_f1[str(svc)] = float(f1_score(y_true, y_pred, zero_division=0))

    if not has_gt:
        overall_f1 = None
    else:
        y_true_all = (wdf["ground_truth_label"] == "FAILURE").astype(int)
        y_pred_all = (wdf["hybrid_label"] == "FAILURE").astype(int)
        overall_f1 = float(f1_score(y_true_all, y_pred_all, zero_division=0))

    # 5 Minute Trend Analyze Feature
    trend_5m = []
    if "timestamp" in wdf.columns:
        wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], errors="coerce")
        now = pd.Timestamp.now()
        five_mins_ago = now - pd.Timedelta(minutes=5)
        recent_wdf = wdf[wdf["timestamp"] >= five_mins_ago].copy()
        if not recent_wdf.empty:
            recent_wdf["minute"] = recent_wdf["timestamp"].dt.floor("Min")
            recent_wdf["is_fail"] = (recent_wdf["hybrid_label"] == "FAILURE").astype(int)
            trend_df = recent_wdf.groupby("minute")["is_fail"].sum().reset_index()
            for _, r in trend_df.iterrows():
                trend_5m.append({"time": r["minute"].strftime("%H:%M"), "failures": int(r["is_fail"])})

    return {
        "total_windows": total_windows,
        "failure_windows": failure_windows,
        "normal_windows": normal_windows,
        "failure_percent": round(failure_percent, 2),
        "failure_rate": round(failure_percent, 2),  # Added failure_rate for schema consistency
        "layer_contribution": layer_contribution,
        "per_service_f1": per_service_f1,
        "overall_f1": overall_f1,
        "trend_5m": trend_5m,
    }


@app.get("/api/component3/feed")
def api_component3_feed(since_window_id: int = Query(default=0, ge=0)) -> dict[str, Any]:
    # 1. Prefer using pipeline.get_new_windows_since()
    windows = pipeline.get_new_windows_since(since_window_id)
    
    # 2. Fallback to STATE.windows_rows if file returns nothing
    if not windows:
        with STATE.lock:
            if not STATE.windows_rows:
                return {"last_window_id": since_window_id, "new_windows_count": 0, "windows": []}
            wdf = pd.DataFrame(STATE.windows_rows)
        
        new_df = wdf[wdf["window_id"] > since_window_id].sort_values("window_id")
        if new_df.empty:
            return {"last_window_id": since_window_id, "new_windows_count": 0, "windows": []}
            
        # Ensure we add ML enrichment columns to the fallback dataframe if missing
        if "memory_change_5min" not in new_df.columns:
            new_df = pipeline.build_ml_dataframe(new_df)
        elif "project_id" not in new_df.columns:
            new_df["project_id"] = new_df["stack"].apply(pipeline.get_project_id)
        if "label" not in new_df.columns and "hybrid_label" in new_df.columns:
            new_df["label"] = new_df["hybrid_label"]
            
        windows = _df_to_records(new_df)

    # 3. Standardize output windows to include all ML, trend columns, and requested aliases
    enriched_windows = []
    for w in windows:
        ew = dict(w)
        
        # Populate project_id and label/hybrid_label
        if "project_id" not in ew and "stack" in ew:
            ew["project_id"] = pipeline.get_project_id(ew["stack"])
        if "label" not in ew and "hybrid_label" in ew:
            ew["label"] = ew["hybrid_label"]
        elif "hybrid_label" not in ew and "label" in ew:
            ew["hybrid_label"] = ew["label"]
            
        # Standard trend columns aliases/names
        # Ensure we support both exact spellings requested (e.g. trend_slope_5m and trend_5m_slope)
        for min_str in ["5m", "10m"]:
            slope_key = f"trend_slope_{min_str}"
            alias_slope_key = f"trend_{min_str}_slope"
            
            # Map slope
            if slope_key in ew:
                ew[alias_slope_key] = ew[slope_key]
            elif alias_slope_key in ew:
                ew[slope_key] = ew[alias_slope_key]
            else:
                ew[slope_key] = 0.0
                ew[alias_slope_key] = 0.0

        # General aliases
        if "trend_max_failures_5m" in ew:
            ew["trend_max_failures"] = ew["trend_max_failures_5m"]
        elif "trend_max_failures" not in ew:
            ew["trend_max_failures"] = None

        if "trend_variance_5m" in ew:
            ew["trend_variance"] = ew["trend_variance_5m"]
        elif "trend_variance" not in ew:
            ew["trend_variance"] = None
            
        enriched_windows.append(ew)

    last_window_id = since_window_id
    if enriched_windows:
        last_window_id = int(max(w.get("window_id", since_window_id) for w in enriched_windows))

    return {
        "last_window_id": last_window_id,
        "new_windows_count": len(enriched_windows),
        "windows": enriched_windows,
    }


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            metrics = api_live_metrics()
            await websocket.send_json({"type": "metrics_update", "data": metrics})

            for row in metrics:
                if row.get("hybrid_label") == "FAILURE":
                    alert = {
                        "type": "failure_alert",
                        "service_name": row.get("service_name"),
                        "failure_type": row.get("failure_type"),
                        "hybrid_label": row.get("hybrid_label"),
                        "ram_percent": row.get("ram_percent"),
                        "timestamp": row.get("timestamp"),
                    }
                    await websocket.send_json(alert)

            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
