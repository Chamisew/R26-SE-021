"""
CPU SPIKE PREDICTOR — PRODUCTION DASHBOARD
==========================================
Flask + Flask-SocketIO application.

Architecture:
  - metric_adapter.py  : receives live metrics from your 2 components
  - predictor.py       : ML inference on rolling metric windows
  - outcome_logger.py  : SQLite feedback loop (alarm outcomes)
  - This file          : HTTP API + WebSocket push

WebSocket events pushed to clients:
  "service_update"   — new prediction for a service  (every PREDICT_INTERVAL_SEC)
  "alarm"            — when a service crosses the threshold
  "alarm_resolved"   — when an alarm is dismissed
"""

import os, time, threading, logging
from datetime import datetime
import pandas as pd

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(BASE, ".env"))
except ImportError:
    pass

from metric_adapter import metric_store
from predictor      import predict_latest, MODEL_NAME, THRESHOLD
from outcome_logger import init_db, record_alarm, resolve_alarm, list_alarms, drift_summary
from notifier       import send_alarm_notifications

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
PREDICT_INTERVAL_SEC  = 5    # how often the background loop runs predictions
ALARM_COOLDOWN_SEC    = 60   # seconds before the same service can re-alarm
# ────────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-16s  %(levelname)s  %(message)s"
)
logger = logging.getLogger("app")

BASE    = os.path.dirname(os.path.abspath(__file__))
PARENT  = os.path.dirname(BASE)

app = Flask(__name__,
            template_folder=os.path.join(BASE, "templates"),
            static_folder=os.path.join(BASE, "static"))
app.config["SECRET_KEY"] = "cpu-spike-predictor-secret"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# In-memory alarm state: service -> last alarm timestamp
_alarm_state: dict[str, float] = {}
_alarm_ids:   dict[str, int]   = {}   # service -> latest DB alarm id

# MTTA Live Tracking
# service -> { "first_alarm_ts": float | None, "in_failure": bool }
_mtta_tracker: dict[str, dict] = {}
_mtta_events:  list[dict]      = []   # list of calculated MTTA events

_state_lock   = threading.Lock()

# --- Initialise subsystems ---
init_db()
metric_store.start_puller()


# ─── BACKGROUND PREDICTION LOOP ───────────────────────────────────────────────
def prediction_loop():
    """
    Runs every PREDICT_INTERVAL_SEC in a background thread.
    For each registered service:
      1. Grab its metric buffer
      2. Run predict_latest()
      3. Emit "service_update" via WebSocket to all connected browsers
      4. If alarm and cooldown passed → record in DB + emit "alarm"
    """
    logger.info("Prediction loop started")
    while True:
        services = metric_store.all_services()
        for svc_name in services:
            buf  = metric_store.get_buffer(svc_name)
            if buf is None:
                continue
            rows = buf.snapshot()
            pred = predict_latest(rows)
            pred["service"] = svc_name
            pred["ts"]      = datetime.now().isoformat()

            # --- Live MTTA Calculation ---
            # For this internal prediction loop, we don't have ground truth labels
            # We'll rely on external CPU predictions from component 3 for MTTA
            # So we skip live MTTA here and handle it in /api/ingest-prediction

            # Push update to all connected dashboards
            socketio.emit("service_update", pred)

            # Alarm handling with cooldown
            if pred["alarm"]:
                now = time.time()
                with _state_lock:
                    last = _alarm_state.get(svc_name, 0)
                    if now - last >= ALARM_COOLDOWN_SEC:
                        _alarm_state[svc_name] = now
                        alarm_id = record_alarm(
                            service=svc_name,
                            prob=pred["prob"],
                            cpu_percent=pred["latest_cpu"]
                        )
                        _alarm_ids[svc_name] = alarm_id

                        alarm_payload = {
                            "alarm_id":  alarm_id,
                            "service":   svc_name,
                            "prob":      pred["prob"],
                            "cpu":       pred["latest_cpu"],
                            "threshold": THRESHOLD,
                            "ts":        datetime.now().isoformat(),
                        }
                        socketio.emit("alarm", alarm_payload)
                        send_alarm_notifications(alarm_payload)
                        logger.warning(f"ALARM fired: {svc_name}  prob={pred['prob']:.3f}")

        time.sleep(PREDICT_INTERVAL_SEC)


_pred_thread = threading.Thread(target=prediction_loop, daemon=True)
_pred_thread.start()


# ─── HTTP ROUTES ──────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html",
                           model_name=MODEL_NAME,
                           threshold=THRESHOLD)


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    Push endpoint for your two external components.

    POST /api/ingest
    Content-Type: application/json

    Body (single metric):
    {
      "service":           "python-api",
      "cpu_percent":       43.7,
      "in_flight_queue":   5.0,
      "incoming_rate":     40.0,
      "processing_rate":   35.0,
      "queue_growth_rate": 5.0,
      "cpu_trend_5min_ma": 12.3,    # optional
      "cpu_trend_10min_ma": 8.1,    # optional
      "timestamp":         1778081050.0  # optional
    }

    Or a list of the above for batch push.
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    rows = data if isinstance(data, list) else [data]
    errors = []
    for row in rows:
        ok, err = metric_store.ingest(row)
        if not ok:
            errors.append(err)

    if errors:
        return jsonify({"error": errors}), 422

    return jsonify({"ok": True, "ingested": len(rows)}), 200


@app.route("/api/services")
def get_services():
    """
    Snapshot of all services with their latest prediction.
    Used for initial page load; live updates come via WebSocket.
    """
    services = metric_store.all_services()
    result   = []
    for name in services:
        buf  = metric_store.get_buffer(name)
        rows = buf.snapshot() if buf else []
        pred = predict_latest(rows)
        pred["service"] = name
        pred["ts"]      = datetime.now().isoformat()
        result.append(pred)

    result.sort(key=lambda x: {"alarm": 0, "watch": 1, "healthy": 2, "warmup": 3}.get(x["status"], 9))
    return jsonify({"services": result, "timestamp": datetime.now().isoformat()})


@app.route("/api/timeline/<service>")
def get_timeline(service):
    """Last 100 raw metric rows for a service (for the sparkline chart)."""
    buf = metric_store.get_buffer(service)
    if not buf:
        return jsonify({"error": f"Service '{service}' not found"}), 404
    rows  = buf.snapshot()[-100:]
    return jsonify({"service": service, "rows": rows, "count": len(rows)})


@app.route("/api/outcomes")
def get_outcomes():
    service = request.args.get("service", "")
    return jsonify({"alarms": list_alarms(service or None)})


@app.route("/api/ingest-prediction", methods=["POST"])
def ingest_prediction():
    """
    Push endpoint for your Component 3 to send CPU predictions.
    
    POST /api/ingest-prediction
    Content-Type: application/json
    
    Body (single prediction):
    {
      "timestamp":         "2026-01-01T00:00:00",
      "service_name":      "python-api",
      "project_id":        "project_1",
      "cpu_failure_prob":  0.67,
      "label":             0  # optional ground-truth label for MTTA
    }
    
    Or a list of the above for batch push.
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    rows = data if isinstance(data, list) else [data]
    
    for row in rows:
        svc_name = row.get("service_name", "unknown")
        project_id = row.get("project_id", "unknown")
        prob = float(row.get("cpu_failure_prob", 0.0))
        ts_str = row.get("timestamp", datetime.now().isoformat())
        ground_truth_label = int(row.get("label", 0))
        
        # Parse timestamp to float
        try:
            ts_dt = pd.to_datetime(ts_str)
            timestamp = ts_dt.timestamp()
        except Exception:
            timestamp = time.time()
        ts_iso = datetime.fromtimestamp(timestamp).isoformat()
        
        with _state_lock:
            if svc_name not in _mtta_tracker:
                _mtta_tracker[svc_name] = {"first_alarm_ts": None, "in_failure": False}
            
            tracker = _mtta_tracker[svc_name]
            
            # 1. Track first alarm
            if prob >= THRESHOLD:
                if tracker["first_alarm_ts"] is None:
                    tracker["first_alarm_ts"] = timestamp
                    logger.info(f"MTTA: First alarm for {svc_name} at {ts_iso}")
            
            # 2. Detect failure start (ground truth label 0 -> 1)
            if ground_truth_label == 1 and not tracker["in_failure"]:
                tracker["in_failure"] = True
                t_fail = timestamp
                t_alarm = tracker["first_alarm_ts"]
                
                mtta = 0.0
                if t_alarm is not None:
                    mtta = max(0.0, t_fail - t_alarm)
                
                event = {
                    "event_id": f"{svc_name}-{len(_mtta_events)}",
                    "service_name": svc_name,
                    "project_id": project_id,
                    "t_alarm": datetime.fromtimestamp(t_alarm).isoformat() if t_alarm else "",
                    "t_fail": ts_iso,
                    "mtta_seconds": mtta,
                    "met_target": bool(mtta >= 120.0),
                    "timestamp": timestamp
                }
                _mtta_events.append(event)
                socketio.emit("mtta_update", event)
                logger.info(f"MTTA: Failure detected for {svc_name}. MTTA = {mtta:.1f}s")
            
            # 3. Detect failure end (label 1 -> 0)
            if ground_truth_label == 0 and tracker["in_failure"]:
                tracker["in_failure"] = False
                tracker["first_alarm_ts"] = None
                logger.info(f"MTTA: Failure resolved for {svc_name}. Resetting tracker.")
    
    return jsonify({"ok": True, "ingested": len(rows)}), 200


@app.route("/api/outcomes/<int:alarm_id>/confirm", methods=["POST"])
def confirm_outcome(alarm_id):
    """Mark alarm as a real failure (true positive)."""
    resolve_alarm(alarm_id, "true_positive")
    socketio.emit("alarm_resolved", {"alarm_id": alarm_id, "outcome": "true_positive"})
    return jsonify({"ok": True})


@app.route("/api/outcomes/<int:alarm_id>/dismiss", methods=["POST"])
def dismiss_outcome(alarm_id):
    """Mark alarm as a false positive."""
    resolve_alarm(alarm_id, "false_positive")
    socketio.emit("alarm_resolved", {"alarm_id": alarm_id, "outcome": "false_positive"})
    return jsonify({"ok": True})


@app.route("/api/drift")
def get_drift():
    """Model drift summary based on logged outcomes."""
    return jsonify(drift_summary())


@app.route("/api/mtta")
def get_mtta():
    """Return MTTA results (combined offline and live)."""
    mtta_csv = os.path.join(PARENT, "results", "cpu_mtta_results.csv")
    
    results = []
    if os.path.exists(mtta_csv):
        try:
            import pandas as pd
            df = pd.read_csv(mtta_csv)
            results = df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error reading MTTA CSV: {e}")

    # Merge with live events
    with _state_lock:
        combined = results + _mtta_events
        
    if not combined:
        return jsonify({"results": [], "summary": {"avg_mtta": 0, "target_met_rate": 0, "total_events": 0}})

    import pandas as pd
    df_combined = pd.DataFrame(combined)
    summary = {
        "avg_mtta": float(df_combined["mtta_seconds"].mean()),
        "target_met_rate": float(df_combined["met_target"].mean()),
        "total_events": len(df_combined)
    }
    return jsonify({"results": combined, "summary": summary})


@app.route("/api/mtta/chart")
def get_mtta_chart():
    """Serve the MTTA chart image."""
    from flask import send_from_directory
    results_dir = os.path.join(PARENT, "results")
    return send_from_directory(results_dir, "cpu_mtta_chart.png")


@app.route("/api/status")
def get_status():
    """Health check + system info."""
    services = metric_store.all_services()
    return jsonify({
        "healthy":         True,
        "model":           MODEL_NAME,
        "threshold":       THRESHOLD,
        "services_count":  len(services),
        "services":        services,
        "predict_every_s": PREDICT_INTERVAL_SEC,
        "timestamp":       datetime.now().isoformat(),
    })


# ─── WEBSOCKET EVENTS ─────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    logger.info(f"Client connected: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CPU SPIKE PREDICTOR — PRODUCTION DASHBOARD")
    print("=" * 60)
    print(f"  Dashboard:   http://localhost:5000")
    print(f"  Ingest API:  POST http://localhost:5000/api/ingest")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Threshold:   {THRESHOLD}")
    print(f"  Predict every {PREDICT_INTERVAL_SEC}s via background thread")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
