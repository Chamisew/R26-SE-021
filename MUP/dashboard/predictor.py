"""
Component 3 -- Predictor

Loads the trained model bundle (produced by compare_models.py) and runs
inference on live rows fetched from the metric adapter.

Public API
----------
    predict(df_live)          -> pd.DataFrame with memory_prob, alert, pred_label
    model_loaded()            -> bool
    get_feature_importance()  -> dict of feature -> RF importance score
    get_shap_importance(df)   -> dict of feature -> mean |SHAP value|
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PKL = os.path.join(BASE, "models", "memory_leak_rf_model.pkl")

FEATURES = [
    "memory_change_10min", "memory_change_5min", "heap_rate", "gc_spike_count",
    "ram_mean", "ram_max", "ram_std", "ram_std_trend",
    "heap_max", "gc_count", "ram_percent", "heap_mb_used",
    "incident_phase_1", "incident_phase_2",
]

_bundle = None


def _load() -> dict:
    """Load the model bundle from disk, caching it in memory after first load.

    Returns
    -------
    dict
        Bundle containing keys: model, scaler, features, threshold, label_pos.

    Raises
    ------
    FileNotFoundError
        If the model pickle file does not exist (compare_models.py not yet run).
    """
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PKL):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PKL}. Run compare_models.py first."
            )
        _bundle = joblib.load(MODEL_PKL)
        logger.info("[predictor] Model loaded.")
    return _bundle


def predict(df_live: pd.DataFrame) -> pd.DataFrame:
    """Run failure-probability inference on a DataFrame of live telemetry rows.

    Missing feature columns are filled with 0.0 to allow partial rows from the
    live feed simulator. The returned DataFrame includes all original columns
    plus three new ones: memory_prob, alert, and pred_label.

    Parameters
    ----------
    df_live : pd.DataFrame
        Input rows from the metric adapter. Must contain at least some of the
        14 feature columns; missing ones are zero-filled.

    Returns
    -------
    pd.DataFrame
        Copy of df_live with additional columns:
          memory_prob (float [0,1]) -- predicted probability of FAILURE.
          alert       (bool)        -- True when memory_prob >= alarm threshold.
          pred_label  (str)         -- "FAILURE" or "NORMAL".
    """
    bundle = _load()
    model = bundle["model"]
    scaler = bundle["scaler"]
    thresh = bundle.get("threshold", 0.6)

    df = df_live.copy()
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0
    df[FEATURES] = df[FEATURES].fillna(0.0)

    X_scaled = scaler.transform(df[FEATURES])
    probs = model.predict_proba(X_scaled)[:, 1]

    df["memory_prob"] = np.round(probs, 4)
    df["alert"] = df["memory_prob"] >= thresh
    df["pred_label"] = df["memory_prob"].apply(
        lambda p: "FAILURE" if p >= thresh else "NORMAL"
    )
    return df


def model_loaded() -> bool:
    """Return True if the model pickle file exists on disk.

    Returns
    -------
    bool
        True when models/memory_leak_rf_model.pkl is present.
    """
    return os.path.exists(MODEL_PKL)


def get_feature_importance() -> dict:
    """Return the Random Forest built-in feature importance scores.

    Uses the mean decrease in impurity (Gini) from the trained forest.
    Returns an empty dict if the model is not loaded or does not support
    feature_importances_.

    Returns
    -------
    dict
        Feature name -> importance score (float), sorted descending by value.
    """
    try:
        bundle = _load()
        model = bundle["model"]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            scored = {f: round(float(v), 4) for f, v in zip(FEATURES, imp)}
            return dict(sorted(scored.items(), key=lambda x: x[1], reverse=True))
    except Exception:
        pass
    return {}


def get_shap_importance(df_sample: pd.DataFrame) -> dict:
    """Compute mean absolute SHAP values for a sample DataFrame using TreeExplainer.

    This function provides model-agnostic, theoretically grounded feature
    attribution (Lundberg & Lee, 2017) as an alternative to the RF built-in
    importance. Use a small representative sample (e.g. 100-200 rows) to keep
    computation fast in the dashboard context.

    Parameters
    ----------
    df_sample : pd.DataFrame
        A sample of live or historical rows. Feature columns will be scaled
        using the stored scaler before passing to the explainer.

    Returns
    -------
    dict
        Feature name -> mean |SHAP value| (float), sorted descending.
        Returns an empty dict if SHAP computation fails.
    """
    try:
        import shap as shap_lib

        bundle = _load()
        model = bundle["model"]
        scaler = bundle["scaler"]

        df = df_sample.copy()
        for feat in FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0
        df[FEATURES] = df[FEATURES].fillna(0.0)
        X_scaled = scaler.transform(df[FEATURES])

        explainer = shap_lib.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled)
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]

        mean_abs = np.abs(sv).mean(axis=0)
        scored = {f: round(float(v), 4) for f, v in zip(FEATURES, mean_abs)}
        return dict(sorted(scored.items(), key=lambda x: x[1], reverse=True))
    except Exception as exc:
        logger.warning(f"[predictor] SHAP importance failed: {exc}")
        return {}
