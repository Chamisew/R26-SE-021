"""
TRAIN MODEL — Production System
================================
Trains on final_research_dataset.csv which has columns:
  incident_id, timestamp, time, system_state, incident_phase,
  failing_service, cpu_percent, in_flight_queue, incoming_rate,
  processing_rate, queue_growth_rate, cpu_trend_5min_ma,
  cpu_trend_10min_ma, label

Feature engineering is designed to work on a rolling window of
recent rows so it can run identically at prediction time on live data.

Output: models/  (best_model.pkl, scaler.pkl, features.pkl,
                   threshold.pkl, smooth_window.pkl, model_name.pkl)
"""

import os, warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.ensemble      import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import f1_score, precision_score, recall_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(BASE, "data", "final_research_dataset.csv")
MODELS  = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# --- Tunable constants ---
ALARM_THRESHOLD = 0.55   # probability above which we fire an alarm
SMOOTH_WINDOW   = 3      # rolling average window for smoothing predictions
F1_THRESHOLD    = 0.50   # probability threshold used for F1 evaluation

FEATURES = [
    "cpu_zscore",           # CPU relative to its own rolling mean
    "cpu_trend_5min_ma",    # 5-min moving average trend (from dataset)
    "cpu_trend_10min_ma",   # 10-min moving average trend
    "cpu_accel",            # rate of change of the 5-min trend
    "queue_growth_rate",    # queue growth (from dataset)
    "queue_ratio",          # in_flight / (incoming_rate + 1)
    "processing_gap",       # incoming_rate - processing_rate
    "cpu_volatility",       # rolling std of cpu_percent (5-row window)
]

LABEL = "label"


def ensure_project_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "service_name" not in d.columns:
        fs = d.get("failing_service")
        if fs is None:
            d["service_name"] = "system-global"
        else:
            d["service_name"] = fs.fillna("system-global").astype(str)
    if "project_id" not in d.columns:
        d["project_id"] = pd.Categorical(d["service_name"]).codes
    return d


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features from raw columns.
    Uses only backward-looking operations — safe for live streaming.
    All operations use min_periods so they work even on warm-up rows.
    """
    d = df.copy().reset_index(drop=True)
    d = d.sort_values("timestamp").reset_index(drop=True)

    # --- CPU z-score vs its own 20-row rolling baseline ---
    roll_mean = d["cpu_percent"].rolling(20, min_periods=3).mean().fillna(d["cpu_percent"].mean())
    roll_std  = d["cpu_percent"].rolling(20, min_periods=3).std().fillna(1.0).clip(lower=0.5)
    d["cpu_zscore"] = (d["cpu_percent"] - roll_mean) / roll_std

    # --- CPU acceleration (2nd derivative of the 5-min trend) ---
    d["cpu_accel"] = d["cpu_trend_5min_ma"].diff(2).fillna(0)

    # --- Queue ratio: how full is the in-flight queue relative to inflow ---
    d["queue_ratio"] = d["in_flight_queue"] / (d["incoming_rate"] + 1.0)

    # --- Processing gap: incoming - processing (positive = building backlog) ---
    d["processing_gap"] = d["incoming_rate"] - d["processing_rate"]

    # --- CPU volatility: 5-row rolling std ---
    d["cpu_volatility"] = d["cpu_percent"].rolling(5, min_periods=2).std().fillna(0)

    # Fill any NaN in dataset-provided features
    for col in ["cpu_trend_5min_ma", "cpu_trend_10min_ma", "queue_growth_rate"]:
        d[col] = d[col].fillna(0)

    return d


def _fit_and_eval(model, X_train, y_train, X_test, y_test):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    if hasattr(model, "set_params"):
        pos = y_train.sum(); neg = len(y_train) - pos
        if pos > 0 and hasattr(model, "predict_proba") and model.__class__.__name__ == "XGBClassifier":
            model.set_params(scale_pos_weight=neg / max(pos, 1))
    model.fit(Xtr, y_train)
    probs = model.predict_proba(Xte)[:, 1]
    preds = (probs >= F1_THRESHOLD).astype(int)
    f1  = f1_score(y_test, preds, zero_division=0)
    pre = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    return f1, pre, rec, model, sc


def lopo_cross_validation(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = ensure_project_columns(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    projects = sorted(df["project_id"].unique().tolist())
    rows = []
    summary = {}
    for name in list(get_models().keys()):
        fold_metrics = []
        for pid in projects:
            train_df = df[df["project_id"] != pid]
            test_df  = df[df["project_id"] == pid]
            trX = train_df[FEATURES].values
            trY = train_df[LABEL].astype(int).values
            teX = test_df[FEATURES].values
            teY = test_df[LABEL].astype(int).values
            model = get_models()[name]   # fresh unfitted instance each fold
            f1, pre, rec, fitted, scaler = _fit_and_eval(model, trX, trY, teX, teY)
            rows.append({
                "model": name,
                "project_id": int(pid),
                "f1": f1,
                "precision": pre,
                "recall": rec,
            })
            fold_metrics.append(f1)
        mean_f1 = float(np.mean(fold_metrics)) if fold_metrics else 0.0
        summary[name] = {"mean_f1": mean_f1}
    res_df = pd.DataFrame(rows)
    return summary, res_df


def _failure_events(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_project_columns(df)
    d = df.sort_values("timestamp").reset_index(drop=True)
    events = []
    event_counter = {}
    
    peak_rows = d[d["incident_phase"] == "PEAK"]
    grouped = peak_rows.groupby(["incident_id", "service_name"], as_index=False).agg({
        "timestamp": "min",
        "project_id": "first"
    })
    
    for _, row in grouped.iterrows():
        svc = str(row["service_name"])
        incident_id = str(row["incident_id"])
        
        if svc not in event_counter:
            event_counter[svc] = 0
        
        t_fail = float(row["timestamp"])
        pid = str(row["project_id"])
        events.append({
            "event_id": f"{svc}-{event_counter[svc]}",
            "service_name": svc,
            "project_id": pid,
            "t_fail": t_fail,
            "incident_id": incident_id
        })
        event_counter[svc] += 1
    
    return pd.DataFrame(events)


def compute_mtta(df_raw: pd.DataFrame, cpu_preds: pd.DataFrame, memory_path: str | None, out_csv: str, out_png: str):
    mem_df = None
    if memory_path and os.path.exists(memory_path):
        try:
            mem_df = pd.read_csv(memory_path)
            if "timestamp" in mem_df.columns:
                mem_df["timestamp_dt"] = pd.to_datetime(mem_df["timestamp"])
                mem_df["ts_float"]     = mem_df["timestamp_dt"].values.astype(np.int64) // 10**9 # type: ignore
        except Exception as e:
            print(f"  ⚠ Could not read memory_predictions.csv: {e}")
            mem_df = None

    events = _failure_events(df_raw)
    rows = []
    
    # Convert cpu_preds timestamps to float for math
    cpu_preds = cpu_preds.copy()
    cpu_preds["ts_float"] = pd.to_datetime(cpu_preds["timestamp"]).values.astype(np.int64) // 10**9 # type: ignore

    for _, ev in events.iterrows():
        pid = str(ev["project_id"])
        svc = str(ev["service_name"])
        t_fail = float(ev["t_fail"])
        
        # CPU Alarms
        cpu_g = cpu_preds[(cpu_preds["service_name"] == svc)]
        cpu_g = cpu_g.sort_values("ts_float")
        cpu_alarm_rows = cpu_g[(cpu_g["cpu_failure_prob"] >= ALARM_THRESHOLD) & (cpu_g["ts_float"] <= t_fail)]
        t_alarm_cpu = float(cpu_alarm_rows["ts_float"].min()) if not cpu_alarm_rows.empty else np.nan
        
        # Memory Alarms (Component 3 integration)
        t_alarm_mem = np.nan
        if mem_df is not None:
            # Columns to look for: memory_prob (user's sample), memory_failure_prob, failure_prob
            prob_col = None
            for col in ["memory_prob", "memory_failure_prob", "failure_prob"]:
                if col in mem_df.columns:
                    prob_col = col
                    break
            
            if prob_col is not None:
                mg = mem_df[mem_df["service_name"] == svc].sort_values("ts_float")
                mem_alarm_rows = mg[(mg[prob_col] >= ALARM_THRESHOLD) & (mg["ts_float"] <= t_fail)]
                if not mem_alarm_rows.empty:
                    t_alarm_mem = float(mem_alarm_rows["ts_float"].min())
        
        # MTTA = T_fail - Earliest Alarm
        t_alarm = t_alarm_cpu
        if not np.isnan(t_alarm_mem):
            t_alarm = np.nanmin([t_alarm, t_alarm_mem]) if not np.isnan(t_alarm) else t_alarm_mem
        
        mtta = 0.0
        if not np.isnan(t_alarm):
            mtta = max(0.0, t_fail - t_alarm)
            
        rows.append({
            "event_id": ev["event_id"],
            "service_name": svc,
            "project_id": pid,
            "t_alarm": datetime.fromtimestamp(t_alarm).isoformat() if not np.isnan(t_alarm) else "",
            "t_fail": datetime.fromtimestamp(t_fail).isoformat(),
            "mtta_seconds": mtta,
            "met_target": bool(mtta >= 120.0),
        })
    
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False)
    if not res.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(res["event_id"], res["mtta_seconds"], color="#2ca02c")
        plt.axhline(120.0, color="red", linestyle="--", label="Target (120s)")
        plt.title("MTTA per Failure Event (CPU + Memory Alarms)")
        plt.ylabel("MTTA (seconds)")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
    return res


def get_models():
    _gb = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
    _rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 max_depth=10, random_state=42)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest":       RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                     max_depth=10, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                          learning_rate=0.05, subsample=0.8,
                                                          random_state=42),
        "GB+RF Ensemble":     VotingClassifier(estimators=[("gb", _gb), ("rf", _rf)],
                                               voting="soft"),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                           eval_metric="logloss", random_state=42,
                                           verbosity=0)
    return models


def train_and_evaluate():
    print("=" * 55)
    print("  TRAINING PRODUCTION MODEL")
    print("=" * 55)

    df_raw = pd.read_csv(DATA)
    print(f"  Loaded {len(df_raw)} rows, {df_raw['label'].sum()} failure rows")

    df0 = ensure_project_columns(df_raw)
    df = engineer_features(df0)

    X = df[FEATURES].values
    y = df[LABEL].astype(int).values

    print(f"  Features: {FEATURES}\n")

    summary, lopo_df = lopo_cross_validation(df)
    lopo_path = os.path.join(MODELS, "lopo_results.csv")
    lopo_df.to_csv(lopo_path, index=False)
    for name in summary:
        print(f"  {name:22s}  LOPO mean F1={summary[name]['mean_f1']:.3f}")
    best_name = max(summary.items(), key=lambda x: x[1]["mean_f1"])[0]
    best_f1   = summary[best_name]["mean_f1"]
    print(f"\n  Best model by LOPO: {best_name}  (mean F1 = {best_f1:.4f})")

    # Re-fit best model on full data with its own scaler
    # Use get_models() to get a fresh instance — avoids deep-copy issues with VotingClassifier
    final_model = get_models()[best_name]
    final_scaler = StandardScaler()
    Xs_final = final_scaler.fit_transform(X)
    if best_name == "XGBoost":
        pos = y.sum(); neg = len(y) - pos # type: ignore
        final_model.set_params(scale_pos_weight=neg / max(pos, 1))
    final_model.fit(Xs_final, y)

    # Save artefacts
    joblib.dump(final_model,     os.path.join(MODELS, "best_model.pkl"))
    joblib.dump(final_scaler,    os.path.join(MODELS, "scaler.pkl"))
    joblib.dump(FEATURES,        os.path.join(MODELS, "features.pkl"))
    joblib.dump(ALARM_THRESHOLD, os.path.join(MODELS, "threshold.pkl"))
    joblib.dump(SMOOTH_WINDOW,   os.path.join(MODELS, "smooth_window.pkl"))
    joblib.dump(best_name,       os.path.join(MODELS, "model_name.pkl"))

    # MTTA Calculation using existing cpu_predictions.csv as input
    cpu_pred_path = os.path.join(os.path.dirname(DATA), "cpu_predictions.csv")
    if not os.path.exists(cpu_pred_path):
        print(f"  ⚠ Skipping MTTA: {cpu_pred_path} not found (required as input)")
    else:
        cpu_preds = pd.read_csv(cpu_pred_path)
        mtta_csv = os.path.join(RESULTS, "cpu_mtta_results.csv")
        mtta_png = os.path.join(RESULTS, "cpu_mtta_chart.png")
        mem_path = os.path.join(os.path.dirname(DATA), "memory_predictions.csv")
        compute_mtta(df_raw, cpu_preds, mem_path, mtta_csv, mtta_png)
        print("  ✅  MTTA CSV:", mtta_csv)
        print("  ✅  MTTA chart:", mtta_png)

    print(f"\n  Saved to models/ and results/")
    print("  ✅  LOPO results:", lopo_path)
    print("  ✅  Training complete.\n")
    return best_name, best_f1


if __name__ == "__main__":
    train_and_evaluate()
