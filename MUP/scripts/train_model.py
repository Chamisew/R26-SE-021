"""
Component 3 -- Memory Leak Predictor  (Train Only)

Trains a Random Forest classifier on ml_ready_dataset.csv using Leave-One-Project-Out
(LOPO) cross-validation across all microservice projects.

Pipeline stages
---------------
  Stage 1 : Data ingestion and NaN removal
  Stage 2 : Class imbalance handling (class_weight='balanced')
  Stage 3 : LOPO cross-validation loop (train on N-1, test on 1)
  Stage 4 : SHAP TreeExplainer -- cross-fold feature importance consistency
  Stage 5 : Final model training on all projects + CSV export

Run
---
    python scripts/train_model.py

Outputs
-------
    models/memory_leak_rf_model.pkl   -- serialised model + scaler bundle
    models/model_metadata.json        -- mean F1, variance, feature list
    logs/lopo_results.csv             -- per-fold metrics
    logs/shap_fold_<project>.png      -- SHAP bar chart per LOPO fold
    logs/shap_consistency.png         -- top-feature consistency summary
    data/memory_predictions.csv       -- failure probability per row (Component 4 input)
"""

import sys
import os
import warnings
import json
import logging

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "ml_ready_dataset.csv")
MODELS_DIR = os.path.join(BASE, "models")
LOGS_DIR = os.path.join(BASE, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "train.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES = [
    "memory_change_10min", "memory_change_5min", "heap_rate", "gc_spike_count",
    "ram_mean", "ram_max", "ram_std", "ram_std_trend",
    "heap_max", "gc_count", "ram_percent", "heap_mb_used",
    "incident_phase_1", "incident_phase_2",
]
TARGET = "label"
RANDOM_SEED = 42
ALARM_THRESHOLD = 0.6
TOP_N_SHAP = 5          # number of top features tracked for consistency
CONSISTENCY_MIN = 7     # a feature is "universal" if it appears in >= 7/10 folds


# ---------------------------------------------------------------------------
# Stage 1 helper
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load ml_ready_dataset.csv, validate required columns, and drop NaN rows.

    Parameters
    ----------
    path : str
        Absolute path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with all required feature and target columns present.

    Raises
    ------
    SystemExit
        If the file is missing or required columns are absent.
    """
    if not os.path.exists(path):
        logger.error(f"Input file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    required = FEATURES + [TARGET, "timestamp", "service_name", "project_id"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in input CSV: {missing_cols}")
        sys.exit(1)

    before = len(df)
    df.dropna(subset=FEATURES + [TARGET], inplace=True)
    after = len(df)
    logger.info(f"  Rows loaded  : {before}  |  after dropna: {after}")
    logger.info(f"  Label dist   : {dict(df[TARGET].value_counts())}")
    return df


# ---------------------------------------------------------------------------
# Stage 3 helper -- one LOPO fold
# ---------------------------------------------------------------------------
def run_lopo_fold(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    projects: pd.Series,
    proj: str,
) -> dict | None:
    """Train on all projects except *proj*, evaluate on *proj*.

    Parameters
    ----------
    model : RandomForestClassifier
        Unfitted classifier instance (will be cloned each fold via fit()).
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full label series.
    projects : pd.Series
        Series of project_id values aligned to X and y.
    proj : str
        Project ID to hold out as the test set for this fold.

    Returns
    -------
    dict or None
        Dictionary with keys project, f1, precision, recall, roc_auc,
        shap_top_features, trained_model, scaler.
        Returns None if the test fold contains only one class (skipped).
    """
    mask_test = projects == proj
    mask_train = ~mask_test
    y_test = y[mask_test]

    if len(y_test.unique()) < 2:
        logger.info(f"  [{proj}] Skipped -- only one class in test fold.")
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[mask_train])
    X_test_s = scaler.transform(X[mask_test])

    model.fit(X_train_s, y[mask_train])

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # Validate AUC label orientation
    y_test_bin = (y_test == "FAILURE").astype(int)
    auc = roc_auc_score(y_test_bin, y_prob)
    # If AUC < 0.5, probabilities are inverted -- flip to correct
    if auc < 0.5:
        y_prob = 1.0 - y_prob
        auc = roc_auc_score(y_test_bin, y_prob)

    f1 = f1_score(y_test, y_pred, pos_label="FAILURE", zero_division=0)
    prec = precision_score(y_test, y_pred, pos_label="FAILURE", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="FAILURE", zero_division=0)

    # ---- Stage 4: SHAP for this fold ------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_s)

    # shap_values shape varies by sklearn version:
    #   list of 2 arrays [n_samples, n_features]  (older sklearn)
    #   single 3-D array [n_samples, n_features, n_classes]  (newer sklearn)
    if isinstance(shap_values, list):
        sv = shap_values[1]          # index 1 = FAILURE class
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]    # last axis = class index
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_importance = dict(zip(FEATURES, mean_abs_shap.tolist()))
    top_features = sorted(shap_importance, key=shap_importance.get, reverse=True)[:TOP_N_SHAP]

    # Save per-fold SHAP bar chart
    _save_shap_bar(shap_importance, proj)

    return {
        "project": proj,
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "roc_auc": round(auc, 4),
        "shap_top_features": top_features,
        "trained_model": model,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Stage 4 helpers -- SHAP plots
# ---------------------------------------------------------------------------
def _save_shap_bar(shap_importance: dict, proj: str) -> None:
    """Save a horizontal bar chart of SHAP feature importances for one fold.

    Parameters
    ----------
    shap_importance : dict
        Mapping of feature name -> mean absolute SHAP value.
    proj : str
        Project ID used in the filename and chart title.
    """
    sorted_items = sorted(shap_importance.items(), key=lambda x: x[1])
    features = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#2563eb" if v >= sorted(values)[-TOP_N_SHAP] else "#94a3b8" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, values, color=colors)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance -- {proj}", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(LOGS_DIR, f"shap_fold_{proj}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"  SHAP chart saved --> logs/shap_fold_{proj}.png")


def save_shap_consistency(fold_results: list) -> dict:
    """Analyse SHAP top-feature consistency across all LOPO folds.

    A feature is considered "universal" if it appears in the top-N list of
    at least CONSISTENCY_MIN folds (default 7 out of 10).

    Parameters
    ----------
    fold_results : list of dict
        List of dicts returned by run_lopo_fold (None entries already removed).

    Returns
    -------
    dict
        Mapping of feature name -> number of folds in which it ranked top-N.
        Also saves logs/shap_consistency.png.
    """
    from collections import Counter
    counter = Counter()
    for fold in fold_results:
        for feat in fold["shap_top_features"]:
            counter[feat] += 1

    # Sort by frequency descending
    sorted_counts = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    universal = [f for f, cnt in sorted_counts.items() if cnt >= CONSISTENCY_MIN]

    logger.info("")
    logger.info("STAGE 4 -- SHAP Cross-Fold Consistency")
    logger.info("=" * 60)
    logger.info(f"  Top-{TOP_N_SHAP} feature appearance across {len(fold_results)} folds:")
    for feat, cnt in sorted_counts.items():
        tag = "  <-- UNIVERSAL" if cnt >= CONSISTENCY_MIN else ""
        logger.info(f"    {feat:<30} appeared in {cnt}/{len(fold_results)} folds{tag}")
    logger.info(f"  Universal features (>= {CONSISTENCY_MIN} folds): {universal}")

    # Plot consistency bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    fnames = list(sorted_counts.keys())
    fcounts = list(sorted_counts.values())
    colors = ["#2563eb" if c >= CONSISTENCY_MIN else "#94a3b8" for c in fcounts]
    ax.barh(fnames[::-1], fcounts[::-1], color=colors[::-1])
    ax.axvline(CONSISTENCY_MIN, color="red", linestyle="--", linewidth=1.5,
               label=f"Consistency threshold ({CONSISTENCY_MIN} folds)")
    ax.set_xlabel("Number of folds in top-N", fontsize=11)
    ax.set_title("SHAP Feature Consistency Across LOPO Folds", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "shap_consistency.png"), dpi=120)
    plt.close()
    logger.info("  Consistency chart saved --> logs/shap_consistency.png")
    logger.info("=" * 60)

    return sorted_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full 5-stage Memory Leak Predictor training pipeline."""
    logger.info("=" * 60)
    logger.info("  Component 3 -- Memory Leak Predictor  (Train)")
    logger.info("=" * 60)

    # ---- Stage 1: Data ingestion ----------------------------------------
    logger.info("")
    logger.info("STAGE 1 -- Data Ingestion")
    df = load_data(DATA)
    X = df[FEATURES]
    y = df[TARGET]
    projects = df["project_id"]

    # ---- Stage 2: Class imbalance handling (embedded in model config) ----
    logger.info("")
    logger.info("STAGE 2 -- Class Imbalance Handling")
    logger.info("  Using class_weight='balanced' in RandomForestClassifier.")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # ---- Stage 3: LOPO cross-validation ---------------------------------
    logger.info("")
    logger.info("STAGE 3 -- Leave-One-Project-Out Cross-Validation")
    logger.info("=" * 60)

    unique_projects = sorted(projects.unique())
    fold_results = []

    for proj in unique_projects:
        result = run_lopo_fold(model, X, y, projects, proj)
        if result is None:
            continue
        fold_results.append(result)
        logger.info(
            f"  Fold [{result['project']:<12}] | "
            f"F1={result['f1']:.4f} | "
            f"Prec={result['precision']:.4f} | "
            f"Rec={result['recall']:.4f} | "
            f"AUC={result['roc_auc']:.4f} | "
            f"Top SHAP: {result['shap_top_features'][:3]}"
        )

    logger.info("-" * 60)
    f1_values = [r["f1"] for r in fold_results]
    mean_f1 = float(np.mean(f1_values))
    var_f1 = float(np.var(f1_values))
    mean_auc = float(np.mean([r["roc_auc"] for r in fold_results]))
    logger.info(f"  Mean F1     : {mean_f1:.4f}  (target >= 0.75)")
    logger.info(f"  F1 Variance : {var_f1:.6f}  (target < 0.10)")
    logger.info(f"  Mean AUC    : {mean_auc:.4f}")
    logger.info(f"  Validation  : {'[TARGET MET]' if mean_f1 >= 0.75 else '[BELOW TARGET]'}")
    logger.info("=" * 60)

    # Save per-fold metrics CSV
    fold_rows = [
        {k: v for k, v in r.items() if k not in ("trained_model", "scaler", "shap_top_features")}
        | {"shap_top3": ", ".join(r["shap_top_features"][:3])}
        for r in fold_results
    ]
    pd.DataFrame(fold_rows).to_csv(os.path.join(LOGS_DIR, "lopo_results.csv"), index=False)

    # ---- Stage 4: SHAP consistency analysis -----------------------------
    shap_counts = save_shap_consistency(fold_results)

    # ---- Stage 5: Final model on all data + export ----------------------
    logger.info("")
    logger.info("STAGE 5 -- Final Model Training and Export")
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X)
    model.fit(X_all, y)

    # Save model bundle
    joblib.dump(
        {
            "model": model,
            "scaler": scaler_final,
            "features": FEATURES,
            "threshold": ALARM_THRESHOLD,
            "label_pos": "FAILURE",
        },
        os.path.join(MODELS_DIR, "memory_leak_rf_model.pkl"),
    )
    logger.info("  Model saved --> models/memory_leak_rf_model.pkl")

    # Save metadata
    meta = {
        "best_model": "Random Forest",
        "mean_f1": round(mean_f1, 4),
        "f1_variance": round(var_f1, 6),
        "mean_auc": round(mean_auc, 4),
        "alarm_threshold": ALARM_THRESHOLD,
        "features": FEATURES,
        "shap_consistency": shap_counts,
        "rows_trained": len(df),
        "projects": int(projects.nunique()),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Generate memory_predictions.csv for Component 4
    y_prob_all = model.predict_proba(X_all)[:, 1]
    df_out = df[["timestamp", "service_name", "project_id"]].copy()
    df_out["memory_prob"] = np.round(y_prob_all, 4)
    df_out["alert"] = df_out["memory_prob"] >= ALARM_THRESHOLD
    df_out["pred_label"] = df_out["memory_prob"].apply(
        lambda p: "FAILURE" if p >= ALARM_THRESHOLD else "NORMAL"
    )
    out_csv = os.path.join(BASE, "data", "memory_predictions.csv")
    df_out.to_csv(out_csv, index=False)
    logger.info(f"  memory_predictions.csv saved --> data/memory_predictions.csv")
    logger.info("")
    logger.info("[DONE] Training complete.")


if __name__ == "__main__":
    main()
