app = FastAPI(
    title="Memory Failure Detection API",
    description="FastAPI wrapper providing REST and WebSocket APIs for microservice memory-failure pipeline monitoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SharedState:
    """Thread-safe in-memory state used across all API endpoints and background workers."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.status = "stopped"          # Options: stopped | running | collecting
        self.mode = PIPELINE_MODE

        self.raw_rows: list[dict[str, Any]] = []
        self.windows_rows: list[dict[str, Any]] = []

        self.total_rows_collected = 0
        self.total_windows_generated = 0
        self.last_global_window_id = 0
        self.active_containers: list[str] = []
        self.last_error: str | None = None


STATE = SharedState()

SERVICES_CONFIG_PATH = Path(__file__).resolve().parent / "services_config.json"


def _read_services_config() -> dict[str, Any]:
    """Safely load configured services from services_config.json."""
    try:
        with SERVICES_CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        return {"services": []}
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Cannot read services configuration: {exc}"
        ) from exc

    if not isinstance(config, dict) or not isinstance(config.get("services", []), list):
        raise HTTPException(
            status_code=500, detail="services_config.json has an invalid format"
        )
    return config

    def _write_services_config(config: dict[str, Any]) -> None:
    """Safely write updated service configuration back to services_config.json."""
    try:
        with SERVICES_CONFIG_PATH.open("w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2)
            config_file.write("\n")
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Cannot save services configuration: {exc}"
        ) from exc

SERVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AzureServiceInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Service container name")
    base_url: HttpUrl = Field(..., description="HTTP base endpoint URL of sidecar")
    enabled: bool = Field(default=True, description="Service enabled flag")


    @app.get("/api/azure-services")
def api_azure_services() -> list[dict[str, Any]]:
    """Return all configured Azure microservice sidecars."""
    return _read_services_config()["services"]


@app.get("/api/services-config")
def api_services_config() -> dict[str, Any]:
    """Read-only view of services_config.json for configuration tab."""
    return _read_services_config()

    @app.post("/api/azure-services")
def api_add_azure_service(service: AzureServiceInput) -> dict[str, Any]:
    """Validate, health check, and register a new Azure sidecar service."""
    name = service.name.strip()
    if not SERVICE_NAME_PATTERN.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail="Service name may contain only letters, numbers, dots, underscores, and hyphens.",
        )

    base_url = str(service.base_url).rstrip("/")
    verified = False
    verification_notes = []

    try:
        metrics_resp = requests.get(f"{base_url}/metrics", timeout=6.0)
        if metrics_resp.status_code == 200:
            verified = True
            verification_notes.append("/metrics returned HTTP 200")
        else:
            verification_notes.append(f"/metrics returned HTTP {metrics_resp.status_code}")
    except requests.RequestException as exc:
        verification_notes.append(f"/metrics unreachable: {exc}")

    config = _read_services_config()
    config["services"] = [
        item for item in config.get("services", [])
        if isinstance(item, dict) and item.get("name") != name
    ]
    config["services"].append({"name": name, "base_url": base_url, "enabled": service.enabled})
    _write_services_config(config)

    with STATE.lock:
        if name not in STATE.active_containers and service.enabled:
            STATE.active_containers = sorted(STATE.active_containers + [name])

    msg = f"Service '{name}' saved to services_config.json."
    if verified:
        msg += " Endpoint verified and active."
    else:
        msg += f" Note: remote check details: {'; '.join(verification_notes)}"

    return {
        "status": "saved",
        "verified": verified,
        "service": {"name": name, "base_url": base_url, "enabled": service.enabled},
        "message": msg,
    }

    @app.delete("/api/azure-services/{service_name}")
def api_delete_azure_service(service_name: str) -> dict[str, Any]:
    """Delete a service entry from services_config.json."""
    name = service_name.strip()
    config = _read_services_config()
    services_list = config.get("services", [])
    before_len = len(services_list)
    config["services"] = [s for s in services_list if isinstance(s, dict) and s.get("name") != name]

    if len(config["services"]) == before_len:
        raise HTTPException(
            status_code=404, detail=f"Service '{name}' not found in services_config.json."
        )

    _write_services_config(config)
    with STATE.lock:
        STATE.active_containers = [s for s in STATE.active_containers if s != name]

    return {
        "status": "deleted",
        "service_name": name,
        "message": f"Service '{name}' removed from services_config.json.",
    }

    @app.get("/api/status")
def api_status() -> dict[str, Any]:
    """Fetch current API, pipeline state, and active containers."""
    try:
        configured = [
            s["name"] for s in _read_services_config().get("services", [])
            if isinstance(s, dict) and s.get("name") and s.get("enabled", True)
        ]
    except Exception:
        configured = []

    with STATE.lock:
        data_seen = {r.get("service_name") for r in STATE.raw_rows if r.get("service_name")}
        merged_services = sorted(set(configured) | set(STATE.active_containers) | data_seen)
        return {
            "status": STATE.status,
            "mode": STATE.mode,
            "uptime_seconds": int(time.time() - STATE.started_at),
            "total_rows_collected": STATE.total_rows_collected,
            "total_windows_generated": STATE.total_windows_generated,
            "last_global_window_id": STATE.last_global_window_id,
            "active_containers": merged_services,
            "configured_services": configured,
            "last_error": STATE.last_error,
        }

        @app.get("/api/live-metrics")
def api_live_metrics(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    """Retrieve recent raw telemetry and log records."""
    if not isinstance(limit, int):
        limit = 50

    with STATE.lock:
        if not STATE.raw_rows:
            return []
        return list(STATE.raw_rows[:limit])

        @app.get("/api/windows")
def api_windows(limit: int = Query(default=50, ge=1, le=1000)) -> list[dict[str, Any]]:
    """Retrieve processed feature window records with historical fallback."""
    with STATE.lock:
        if STATE.windows_rows:
            return [{k: _sanitize_value(v) for k, v in row.items()} for row in STATE.windows_rows[-limit:]]

    try:
        out_csv = os.path.join(pipeline.CONFIG.get("output_dir", "output/"), "pipeline_output.csv")
        if os.path.exists(out_csv):
            df = pd.read_csv(out_csv)
            if not df.empty:
                recent_df = df.tail(limit)
                records = _df_to_records(recent_df)
                with STATE.lock:
                    if not STATE.windows_rows:
                        STATE.windows_rows = records
                        STATE.total_windows_generated = len(df)
                        if "window_id" in df.columns:
                            valid_ids = pd.to_numeric(df["window_id"], errors="coerce").dropna()
                            if not valid_ids.empty:
                                STATE.last_global_window_id = int(valid_ids.max())
                return records
    except Exception:
        pass
    return []
@app.get("/api/templates")
def api_templates() -> list[dict[str, Any]]:
    """Retrieve top mined log templates and failure flag association."""
    with STATE.lock:
        if not STATE.raw_rows:
            return []
        raw_df = pd.DataFrame(STATE.raw_rows)

    if raw_df.empty:
        return []

    if "log_template" not in raw_df.columns or raw_df["log_template"].dropna().empty:
        if "log_message" in raw_df.columns:
            raw_df["log_template"] = raw_df["log_message"].apply(
                lambda m: re.sub(r"\d+", "<*>", str(m))
            )
        else:
            return []

    if "hybrid_label" in raw_df.columns:
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
    else:
        grouped = (
            raw_df.groupby("log_template")
            .size()
            .rename("count")
            .reset_index()
            .assign(failure_related=False)
            .sort_values("count", ascending=False)
            .head(20)
        )
    grouped.rename(columns={"log_template": "template"}, inplace=True)
    return _df_to_records(grouped[["template", "count", "failure_related"]])

@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    """Calculate summary pipeline metrics, per-service F1 scores, and layer distributions."""
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
    if "ground_truth_label" in wdf.columns and wdf["ground_truth_label"].isin(["FAILURE", "NORMAL"]).any():
        failure_windows = int((wdf["ground_truth_label"] == "FAILURE").sum())
        normal_windows = int((wdf["ground_truth_label"] == "NORMAL").sum())
    else:
        failure_windows = int((wdf["hybrid_label"] == "FAILURE").sum())
        normal_windows = total_windows - failure_windows
    failure_percent = (failure_windows / total_windows * 100.0) if total_windows > 0 else 0.0

    layers = wdf["detection_layer"].astype(str)
    layer_contribution = {
        "keyword": int(layers.str.contains("keyword", na=False).sum()),
        "tfidf_semantic": int(layers.str.contains("tfidf_semantic", na=False).sum()),
        "metric_fusion": int(layers.str.contains("metric_fusion", na=False).sum()),
    }

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

    overall_f1 = float(f1_score(
        (wdf["ground_truth_label"] == "FAILURE").astype(int),
        (wdf["hybrid_label"] == "FAILURE").astype(int),
        zero_division=0
    )) if has_gt else None

    trend_5m = []
    if "timestamp" in wdf.columns:
        wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], errors="coerce")
        five_mins_ago = pd.Timestamp.now() - pd.Timedelta(minutes=5)
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
        "failure_rate": round(failure_percent, 2),
        "layer_contribution": layer_contribution,
        "per_service_f1": per_service_f1,
        "overall_f1": overall_f1,
        "trend_5m": trend_5m,
    }

    @app.get("/api/component3/feed")
def api_component3_feed(since_window_id: int = Query(default=0, ge=0)) -> dict[str, Any]:
    """Provide feature feed and dataset stream for Component 3 ingestion."""
    windows = pipeline.get_new_windows_since(since_window_id)

    if not windows:
        with STATE.lock:
            if not STATE.windows_rows:
                return {"last_window_id": since_window_id, "new_windows_count": 0, "windows": []}
            wdf = pd.DataFrame(STATE.windows_rows)

        new_df = wdf[wdf["window_id"] > since_window_id].sort_values("window_id")
        if new_df.empty:
            return {"last_window_id": since_window_id, "new_windows_count": 0, "windows": []}

        if "memory_change_5min" not in new_df.columns:
            new_df = pipeline.build_ml_dataframe(new_df)
        elif "project_id" not in new_df.columns:
            new_df["project_id"] = new_df["service_name"].apply(pipeline.get_project_id)
        if "label" not in new_df.columns and "hybrid_label" in new_df.columns:
            new_df["label"] = new_df["hybrid_label"]

        windows = _df_to_records(new_df)

    enriched_windows = []
    for w in windows:
        ew = dict(w)

        if "project_id" not in ew or not ew["project_id"] or pd.isna(ew["project_id"]):
            svc = ew.get("service_name") or ew.get("stack")
            if svc:
                ew["project_id"] = pipeline.get_project_id(svc)
        if "label" not in ew and "hybrid_label" in ew:
            ew["label"] = ew["hybrid_label"]
        elif "hybrid_label" not in ew and "label" in ew:
            ew["hybrid_label"] = ew["label"]

        for min_str in ["5m", "10m"]:
            slope_key = f"trend_slope_{min_str}"
            alias_slope_key = f"trend_{min_str}_slope"

            if slope_key in ew:
                ew[alias_slope_key] = ew[slope_key]
            elif alias_slope_key in ew:
                ew[slope_key] = ew[alias_slope_key]
            else:
                ew[slope_key] = 0.0
                ew[alias_slope_key] = 0.0

        if "trend_max_failures_5m" in ew:
            ew["trend_max_failures"] = ew["trend_max_failures_5m"]
        elif "trend_max_failures" not in ew:
            ew["trend_max_failures"] = None

        if "trend_variance_5m" in ew:
            ew["trend_variance"] = ew["trend_variance_5m"]
        elif "trend_variance" not in ew:
            ew["trend_variance"] = None

        sanitized_ew = {k: _sanitize_value(v) for k, v in ew.items()}
        enriched_windows.append(sanitized_ew)

    last_window_id = since_window_id
    if enriched_windows:
        last_window_id = int(max((w.get("window_id") or since_window_id) for w in enriched_windows))

    return {
        "last_window_id": last_window_id,
        "new_windows_count": len(enriched_windows),
        "windows": enriched_windows,
    }

    @app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Stream live telemetry updates and trigger immediate failure alerts."""
    await websocket.accept()
    try:
        while True:
            metrics = api_live_metrics(50)
            with STATE.lock:
                active_svcs = list(STATE.active_containers)
            await websocket.send_json({
                "type": "metrics_update",
                "data": metrics,
                "active_services": active_svcs,
            })

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

            await asyncio.sleep(3)
    except (WebSocketDisconnect, Exception):
        return

        def _sanitize_value(v: Any) -> Any:
    """Convert NaN, Infinity, and non-serializable pandas/numpy types to JSON safe values."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.floating, float)):
        if np.isinf(v) or np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v
# Metrics Route Query Parameter Bounds: limit: int = Query(default=50, ge=1, le=500)
# Windows Route Query Parameter Bounds: limit: int = Query(default=50, ge=1, le=1000)
# Feed Route Query Parameter Bounds: since_window_id: int = Query(default=0, ge=0)

def _pipeline_worker() -> None:
    """Execute live collection or CSV batch execution asynchronously in background."""
    with STATE.lock:
        STATE.status = "running"
        STATE.started_at = time.time()
        mode = STATE.mode

    if mode == "live":
        def run_collector():
            while True:
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
                    break
                except BaseException as exc:
                    with STATE.lock:
                        STATE.status = "stopped"
                        STATE.last_error = f"Collector thread failed: {str(exc)}"

                if PIPELINE_DURATION_MINUTES > 0:
                    time.sleep(2)
                else:
                    break

        collector_thread = threading.Thread(target=run_collector, daemon=True)
        collector_thread.start()

        while True:
            with pipeline.pipeline_state_lock:
                raw_df = pipeline.pipeline_state.get("raw_df")
                win_df = pipeline.pipeline_state.get("windows_df")
                pipe_configured = pipeline.pipeline_state.get("configured_services", [])

            try:
                file_configured = [
                    s["name"] for s in _read_services_config().get("services", [])
                    if isinstance(s, dict) and s.get("name") and s.get("enabled", True)
                ]
            except Exception:
                file_configured = []

            all_configured = sorted(set(pipe_configured) | set(file_configured))

            with STATE.lock:
                if raw_df is not None and not raw_df.empty:
                    STATE.raw_rows = _prepare_metric_records(raw_df, limit=500)
                    STATE.total_rows_collected = len(raw_df)
                    data_seen = {r.get("service_name") for r in STATE.raw_rows if r.get("service_name")}
                    STATE.active_containers = sorted(set(all_configured) | data_seen)
                elif all_configured:
                    STATE.active_containers = all_configured

                if win_df is not None and not win_df.empty:
                    STATE.windows_rows = _df_to_records(win_df.tail(200))
                    STATE.total_windows_generated = len(win_df)
                    if "window_id" in win_df.columns:
                        try:
                            valid_ids = pd.to_numeric(win_df["window_id"], errors="coerce").dropna()
                            if not valid_ids.empty:
                                STATE.last_global_window_id = int(valid_ids.max())
                        except Exception:
                            pass
                if STATE.status != "stopped":
                    STATE.status = "running"
            time.sleep(2)