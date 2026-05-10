"""
PREDICTION ENGINE
=================
Takes a list of raw metric rows for one service,
applies the same feature engineering as training,
and returns a smoothed failure probability.

Warmup guard: if fewer than WARMUP_ROWS rows are available,
returns None so the dashboard shows "warming up" instead of a
potentially misleading prediction.
"""

import os
import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger("predictor")

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")

WARMUP_ROWS = 10   # minimum rows before we predict


def _load_artefacts():
    try:
        model     = joblib.load(os.path.join(MODELS, "best_model.pkl"))
        scaler    = joblib.load(os.path.join(MODELS, "scaler.pkl"))
        features  = joblib.load(os.path.join(MODELS, "features.pkl"))
        threshold = joblib.load(os.path.join(MODELS, "threshold.pkl"))
        smooth_w  = joblib.load(os.path.join(MODELS, "smooth_window.pkl"))
        name      = joblib.load(os.path.join(MODELS, "model_name.pkl"))
        logger.info(f"Model loaded: {name} | threshold={threshold} | smooth={smooth_w}")
        return model, scaler, features, threshold, smooth_w, name
    except Exception as e:
        logger.error(f"Failed to load model artefacts: {e}")
        raise


MODEL, SCALER, FEATURES, THRESHOLD, SMOOTH_W, MODEL_NAME = _load_artefacts()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identical to training feature engineering.
    Uses only backward-looking rolling ops — no leakage.
    """
    d = df.copy().reset_index(drop=True)

    # CPU z-score vs its own 20-row rolling baseline
    roll_mean = d["cpu_percent"].rolling(20, min_periods=3).mean().fillna(d["cpu_percent"].mean())
    roll_std  = d["cpu_percent"].rolling(20, min_periods=3).std().fillna(1.0).clip(lower=0.5)
    d["cpu_zscore"] = (d["cpu_percent"] - roll_mean) / roll_std

    # CPU acceleration (2nd derivative of 5-min trend)
    d["cpu_accel"] = d["cpu_trend_5min_ma"].diff(2).fillna(0)

    # Queue ratio
    d["queue_ratio"] = d["in_flight_queue"] / (d["incoming_rate"] + 1.0)

    # Processing gap
    d["processing_gap"] = d["incoming_rate"] - d["processing_rate"]

    # CPU volatility
    d["cpu_volatility"] = d["cpu_percent"].rolling(5, min_periods=2).std().fillna(0)

    for col in ["cpu_trend_5min_ma", "cpu_trend_10min_ma", "queue_growth_rate"]:
        d[col] = d[col].fillna(0)

    return d


def predict_latest(rows: list[dict]) -> dict:
    """
    Given a list of raw metric dicts (oldest → newest) for ONE service,
    return a prediction dict:
    {
        "prob":          float | None,   # smoothed failure probability
        "alarm":         bool,
        "status":        "alarm" | "watch" | "healthy" | "warmup",
        "row_count":     int,
        "threshold":     float,
        "model":         str,
        "latest_cpu":    float,
        "latest_queue":  float,
        "latest_incoming": float,
        "latest_processing": float,
    }
    """
    n = len(rows)

    latest = rows[-1] if rows else {}
    base = {
        "prob":               None,
        "alarm":              False,
        "status":             "warmup",
        "row_count":          n,
        "threshold":          THRESHOLD,
        "model":              MODEL_NAME,
        "latest_cpu":         float(latest.get("cpu_percent", 0)),
        "latest_queue":       float(latest.get("in_flight_queue", 0)),
        "latest_incoming":    float(latest.get("incoming_rate", 0)),
        "latest_processing":  float(latest.get("processing_rate", 0)),
        "latest_queue_growth":float(latest.get("queue_growth_rate", 0)),
    }

    if n < WARMUP_ROWS:
        return base   # not enough data yet

    df = pd.DataFrame(rows)
    df = engineer_features(df)

    # Guard: ensure all features exist
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    X = df[FEATURES].values
    try:
        Xs    = SCALER.transform(X)
        probs = MODEL.predict_proba(Xs)[:, 1]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return base

    # Smooth the last SMOOTH_W predictions
    smoothed = pd.Series(probs).rolling(SMOOTH_W, min_periods=1).mean().values
    prob = float(smoothed[-1])

    if prob >= THRESHOLD:
        status = "alarm"
    elif prob >= 0.35:
        status = "watch"
    else:
        status = "healthy"

    base.update({
        "prob":   round(prob, 4),
        "alarm":  prob >= THRESHOLD,
        "status": status,
    })
    return base
