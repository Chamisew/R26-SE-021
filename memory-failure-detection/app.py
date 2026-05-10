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


class SharedState:
    """Thread-safe in-memory state used by all endpoints."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.status = "stopped"          # stopped | running | collecting
        self.mode = os.getenv("PIPELINE_MODE", pipeline.MODE)

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
        raw_df = pipeline.stage1_load_data(cfg)
    else:
        raw_df = pipeline.stage1_collect_live(
            duration_minutes=cfg.get("live_duration_minutes", -1)
        )

    if raw_df.empty:
        return raw_df, pd.DataFrame()

    raw_df = pipeline.stage1_5_preprocess(raw_df)
    if raw_df.empty:
        return raw_df, pd.DataFrame()

    raw_df = pipeline.stage2_drain3_parsing(raw_df, cfg)
    raw_df = pipeline.stage3_hybrid_classifier(raw_df, cfg)
    windows_df = pipeline.stage4_sliding_window(raw_df, cfg)
    return raw_df, windows_df


def _pipeline_worker() -> None:
    """Background thread that continuously runs data collection + stages."""
    with STATE.lock:
        STATE.status = "running"
        STATE.started_at = time.time()
        mode = STATE.mode

    if mode == "live":
        # Start pipeline collector in a separate daemon thread
        def run_collector():
            cfg = dict(pipeline.CONFIG)
            cfg["live_duration_minutes"] = -1
            pipeline.stage1_collect_live(duration_minutes=-1)
            
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
                STATE.status = "running"
            time.sleep(2)
    else:
        # CSV Mode
        while True:
            try:
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

                    STATE.status = "running"
                    STATE.last_error = None
            except Exception as exc:
                with STATE.lock:
                    STATE.status = "running"
                    STATE.last_error = str(exc)
                time.sleep(5)
            
            time.sleep(60)


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
            "active_containers": STATE.active_containers,
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
                "layer_contribution": {"keyword": 0, "tfidf_semantic": 0, "metric_fusion": 0},
                "per_service_f1": {},
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

    per_service_f1: dict[str, float] = {}
    for svc, grp in wdf.groupby("service_name"):
        y_true = (grp["ground_truth_label"] == "FAILURE").astype(int)
        y_pred = (grp["hybrid_label"] == "FAILURE").astype(int)
        per_service_f1[str(svc)] = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.sum() > 0 else 0.0

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
        "layer_contribution": layer_contribution,
        "per_service_f1": per_service_f1,
        "trend_5m": trend_5m,
    }


@app.get("/api/component3/feed")
def api_component3_feed(since_window_id: int = Query(default=0, ge=0)) -> dict[str, Any]:
    with STATE.lock:
        if not STATE.windows_rows:
            return {"last_window_id": since_window_id, "new_windows_count": 0, "windows": []}
        wdf = pd.DataFrame(STATE.windows_rows)

    new_df = wdf[wdf["window_id"] > since_window_id].sort_values("window_id")
    if new_df.empty:
        return {"last_window_id": since_window_id, "new_windows_count": 0, "windows": []}

    feed_cols = [
        "window_id", "service_name",
        "memory_growth", "heap_rate", "gc_spike_count",
        "ram_mean", "ram_max", "ram_std",
        "cpu_mean", "cpu_max", "heap_max",
        "gc_count", "ram_percent", "heap_mb_used",
        "failure_type", "timestamp",
    ]
    for c in feed_cols:
        if c not in new_df.columns:
            new_df[c] = None

    payload_df = new_df[feed_cols].copy()
    payload_df["label"] = new_df["hybrid_label"]
    payload_cols = [
        "window_id", "service_name",
        "memory_growth", "heap_rate", "gc_spike_count",
        "ram_mean", "ram_max", "ram_std",
        "cpu_mean", "cpu_max", "heap_max",
        "gc_count", "ram_percent", "heap_mb_used",
        "label", "failure_type", "timestamp",
    ]
    windows = _df_to_records(payload_df[payload_cols])
    return {
        "last_window_id": int(new_df["window_id"].max()),
        "new_windows_count": len(windows),
        "windows": windows,
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

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
