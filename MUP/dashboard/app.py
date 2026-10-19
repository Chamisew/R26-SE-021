"""
Component 3 -- Dashboard (Flask)
Run: python dashboard/app.py
Open: http://localhost:5003
"""
import sys, os, json, logging
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, send_file
import pandas as pd

from metric_adapter  import get_live_rows
from predictor       import predict, model_loaded, get_feature_importance
from outcome_logger  import log_predictions
from notifier        import send_alert

# ── Logging ──────────────────────────────────────────────────────────────────
LOGS = os.path.join(BASE, "logs")
os.makedirs(LOGS, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS, "dashboard.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── State ────────────────────────────────────────────────────────────────────
_seen_timestamps = set()
_recent_alerts   = []   # list of dicts for the alerts panel
MAX_ALERTS = 50

# ── Helper ───────────────────────────────────────────────────────────────────
def _load_lopo():
    path = os.path.join(BASE, "logs", "lopo_results.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

def _load_comparison():
    path = os.path.join(BASE, "models", "comparison_results.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    # strip folds key for JSON response
    return [{k: v for k, v in r.items() if k != "folds"} for r in data]

def _load_metadata():
    path = os.path.join(BASE, "models", "model_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/live")
def api_live():
    global _seen_timestamps, _recent_alerts

    if not model_loaded():
        return jsonify({"error": "Model not found. Run compare_models.py first.", "rows": []})

    df_live = get_live_rows(50)
    if df_live.empty:
        return jsonify({"rows": [], "alerts_count": 0, "rows_seen": 0})

    df_pred = predict(df_live)
    log_predictions(df_pred)

    # Detect new alerts
    for _, row in df_pred.iterrows():
        ts = str(row.get("timestamp", ""))
        svc = str(row.get("service_name", ""))
        key = f"{ts}_{svc}"
        if row.get("alert") and key not in _seen_timestamps:
            _seen_timestamps.add(key)
            alert_info = {
                "service_name": svc,
                "project_id":   str(row.get("project_id", "")),
                "memory_prob":  float(row.get("memory_prob", 0)),
                "mem_growth":   float(row.get("memory_change_5min", 0)),
                "ram_mean":     float(row.get("ram_mean", 0)),
                "timestamp":    ts
            }
            _recent_alerts.insert(0, alert_info)
            if len(_recent_alerts) > MAX_ALERTS:
                _recent_alerts.pop()
            # Desktop notification
            try:
                send_alert(**alert_info)
            except Exception:
                pass

    # Build response
    rows_out = []
    for _, row in df_pred.iterrows():
        rows_out.append({
            "timestamp":    str(row.get("timestamp", "")),
            "service_name": str(row.get("service_name", "")),
            "project_id":   str(row.get("project_id", "")),
            "memory_prob":  float(row.get("memory_prob", 0)),
            "alert":        bool(row.get("alert", False)),
            "pred_label":   str(row.get("pred_label", "NORMAL")),
            "memory_change_5min":float(row.get("memory_change_5min", 0)),
            "ram_mean":     float(row.get("ram_mean", 0)),
        })

    meta = _load_metadata()
    return jsonify({
        "rows":         rows_out,
        "rows_seen":    len(_seen_timestamps),
        "alerts_count": sum(1 for r in rows_out if r["alert"]),
        "mean_f1":      meta.get("mean_f1", 0),
        "f1_variance":  meta.get("f1_variance", 0),
        "best_model":   meta.get("best_model", "N/A"),
        "model_ready":  True
    })

@app.route("/api/feature_importance")
def api_feature_importance():
    return jsonify(get_feature_importance())

@app.route("/api/lopo")
def api_lopo():
    return jsonify(_load_lopo())

@app.route("/api/comparison")
def api_comparison():
    return jsonify(_load_comparison())

@app.route("/api/alerts")
def api_alerts():
    return jsonify({"alerts": _recent_alerts[:MAX_ALERTS]})

@app.route("/api/download")
def api_download():
    path = os.path.join(BASE, "data", "memory_predictions.csv")
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name="memory_predictions.csv")

if __name__ == "__main__":
    logger.info("Starting Component 3 Dashboard --> http://localhost:5003")
    app.run(host="0.0.0.0", port=5003, debug=False, use_reloader=False)
