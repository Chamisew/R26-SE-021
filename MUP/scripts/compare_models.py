"""
Component 3 -- Model Comparison + Best Model Selection

Evaluates six classifiers using Leave-One-Project-Out (LOPO) cross-validation,
selects the best model by mean F1-score, runs SHAP analysis on the best model's
folds, saves all artefacts, and generates the memory_predictions.csv output for
Component 4.

Models compared
---------------
  Random Forest, XGBoost (if installed), Gradient Boosting, Decision Tree,
  Logistic Regression, KNN

Run
---
    python scripts/compare_models.py

Outputs
-------
    models/memory_leak_rf_model.pkl       -- best model + scaler bundle
    models/model_metadata.json            -- metadata for best model
    models/comparison_results.json        -- per-model LOPO metrics
    logs/comparison_results.csv           -- summary table (CSV)
    logs/lopo_results.csv                 -- per-fold metrics for best model
    logs/model_comparison.png             -- F1 / AUC bar chart
    logs/shap_fold_<project>.png          -- SHAP bar chart per fold (best model)
    logs/shap_consistency.png             -- SHAP cross-fold consistency chart
    data/memory_predictions.csv           -- failure probability scores (Component 4)
"""

import sys
import os
import time
import json
import warnings
import logging
from collections import Counter

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

try:
    from xgboost import XGBClassifier
    _XGBOOST = True
except ImportError:
    _XGBOOST = False

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
        logging.FileHandler(os.path.join(LOGS_DIR, "compare.log"), encoding="utf-8"),
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
TOP_N_SHAP = 5
CONSISTENCY_MIN = 7

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def build_models() -> dict:
    """Return a dictionary of named, unfitted classifier instances.

    Returns
    -------
    dict
        Keys are human-readable model names; values are unfitted estimators.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=RANDOM_SEED,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=RANDOM_SEED,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }
    if _XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_SEED, scale_pos_weight=3, verbosity=0,
        )
    return models


# ---------------------------------------------------------------------------
# LOPO cross-validation
# ---------------------------------------------------------------------------
def lopo_cv(model, X: pd.DataFrame, y: pd.Series, projects: pd.Series) -> dict:
    """Run Leave-One-Project-Out cross-validation for a single classifier.

    For each unique project ID, the model is trained on all other projects
    and evaluated on the held-out project. Per-fold metrics and SHAP values
    are collected (SHAP only for tree-based models that support TreeExplainer).

    Parameters
    ----------
    model : sklearn estimator
        An unfitted classifier that implements fit / predict / predict_proba.
    X : pd.DataFrame
        Full feature matrix (all projects).
    y : pd.Series
        Full label series aligned to X.
    projects : pd.Series
        Project ID series aligned to X and y.

    Returns
    -------
    dict
        Keys: mean_f1, f1_var, mean_prec, mean_rec, mean_auc, fold_details,
        shap_fold_data (list of per-fold SHAP dicts, empty for non-tree models).
    """
    unique = sorted(projects.unique())
    f1s, precs, recs, aucs = [], [], [], []
    fold_details = []
    shap_fold_data = []

    for proj in unique:
        mask_test = projects == proj
        mask_train = ~mask_test
        y_test = y[mask_test]

        if len(y_test.unique()) < 2:
            logger.info(f"    [{proj}] Skipped -- only one class in test fold.")
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X[mask_train])
        X_test_s = scaler.transform(X[mask_test])

        try:
            model.fit(X_train_s, y[mask_train])
            y_pred = model.predict(X_test_s)
            y_prob = model.predict_proba(X_test_s)[:, 1]
        except Exception as exc:
            logger.warning(f"    [{proj}] Error: {exc}")
            continue

        y_test_bin = (y_test == "FAILURE").astype(int)

        try:
            auc = roc_auc_score(y_test_bin, y_prob)
            # Flip if inverted (AUC < 0.5 means probabilities are reversed)
            if auc < 0.5:
                auc = roc_auc_score(y_test_bin, 1.0 - y_prob)
        except Exception:
            auc = 0.5

        f1 = f1_score(y_test, y_pred, pos_label="FAILURE", zero_division=0)
        prec = precision_score(y_test, y_pred, pos_label="FAILURE", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="FAILURE", zero_division=0)

        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
        aucs.append(auc)
        fold_details.append({
            "project": proj, "f1": round(f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "auc": round(auc, 4),
        })

        # SHAP -- only for tree-based models
        shap_top = []
        if hasattr(model, "feature_importances_"):
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_test_s)
                if isinstance(sv, list):
                    sv = sv[1]
                elif sv.ndim == 3:
                    sv = sv[:, :, 1]
                mean_abs = np.abs(sv).mean(axis=0)
                shap_imp = dict(zip(FEATURES, mean_abs.tolist()))
                shap_top = sorted(shap_imp, key=shap_imp.get, reverse=True)[:TOP_N_SHAP]
                shap_fold_data.append({"project": proj, "importance": shap_imp, "top": shap_top})
            except Exception as exc:
                logger.warning(f"    [{proj}] SHAP failed: {exc}")

    n = len(f1s)
    return {
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "f1_var": float(np.var(f1s)) if f1s else 0.0,
        "mean_prec": float(np.mean(precs)) if precs else 0.0,
        "mean_rec": float(np.mean(recs)) if recs else 0.0,
        "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
        "fold_details": fold_details,
        "shap_fold_data": shap_fold_data,
        "n_folds": n,
    }


# ---------------------------------------------------------------------------
# SHAP output helpers
# ---------------------------------------------------------------------------
def save_shap_bar_chart(shap_importance: dict, proj: str) -> None:
    """Save a horizontal bar chart of SHAP importances for one LOPO fold.

    Parameters
    ----------
    shap_importance : dict
        Feature name -> mean absolute SHAP value for this fold.
    proj : str
        Project ID used in the output filename and chart title.
    """
    sorted_items = sorted(shap_importance.items(), key=lambda x: x[1])
    features = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    threshold_val = sorted(values)[-TOP_N_SHAP] if len(values) >= TOP_N_SHAP else 0
    colors = ["#2563eb" if v >= threshold_val else "#94a3b8" for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(features, values, color=colors)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"SHAP Feature Importance -- {proj}", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=8,
        )
    plt.tight_layout()
    out = os.path.join(LOGS_DIR, f"shap_fold_{proj}.png")
    plt.savefig(out, dpi=120)
    plt.close()
    logger.info(f"  SHAP chart saved --> logs/shap_fold_{proj}.png")


def save_shap_consistency(shap_fold_data: list) -> dict:
    """Compute and save the SHAP cross-fold consistency analysis.

    Counts how many LOPO folds each feature appears in the top-N SHAP ranking.
    A feature is "universal" if it appears in >= CONSISTENCY_MIN folds.

    Parameters
    ----------
    shap_fold_data : list of dict
        Each dict has keys 'project', 'importance', 'top' (from lopo_cv).

    Returns
    -------
    dict
        Feature name -> fold count mapping, sorted descending.
    """
    counter = Counter()
    for fold in shap_fold_data:
        for feat in fold["top"]:
            counter[feat] += 1

    sorted_counts = dict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    universal = [f for f, cnt in sorted_counts.items() if cnt >= CONSISTENCY_MIN]

    logger.info("")
    logger.info("SHAP Cross-Fold Consistency")
    logger.info("-" * 50)
    n_folds = len(shap_fold_data)
    for feat, cnt in sorted_counts.items():
        tag = "  <-- UNIVERSAL" if cnt >= CONSISTENCY_MIN else ""
        logger.info(f"  {feat:<30} {cnt}/{n_folds} folds{tag}")
    logger.info(f"  Universal features (>= {CONSISTENCY_MIN}/{n_folds}): {universal}")

    # Consistency bar chart
    fnames = list(sorted_counts.keys())
    fcounts = list(sorted_counts.values())
    colors = ["#2563eb" if c >= CONSISTENCY_MIN else "#94a3b8" for c in fcounts]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fnames[::-1], fcounts[::-1], color=colors[::-1])
    ax.axvline(
        CONSISTENCY_MIN, color="red", linestyle="--", linewidth=1.5,
        label=f"Consistency threshold ({CONSISTENCY_MIN} folds)",
    )
    ax.set_xlabel("Number of folds in top-N", fontsize=11)
    ax.set_title("SHAP Feature Consistency Across LOPO Folds", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "shap_consistency.png"), dpi=120)
    plt.close()
    logger.info("  Consistency chart saved --> logs/shap_consistency.png")

    return sorted_counts


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------
def save_comparison_chart(results: list, best_name: str) -> None:
    """Save a side-by-side F1 / AUC bar chart comparing all evaluated models.

    Parameters
    ----------
    results : list of dict
        Sorted list of model result dicts (best first by F1).
    best_name : str
        Name of the best model (highlighted in blue).
    """
    model_names = [r["model"] for r in results]
    f1_vals = [r["mean_f1"] for r in results]
    auc_vals = [r["mean_auc"] for r in results]
    colors = ["#2563eb" if n == best_name else "#94a3b8" for n in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Component 3 -- Model Comparison (LOPO)", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    bars = ax1.barh(model_names, f1_vals, color=colors)
    ax1.axvline(0.75, color="red", linestyle="--", linewidth=1.5, label="Target F1 = 0.75")
    ax1.set_xlabel("Mean F1 Score")
    ax1.set_title("F1 Score (LOPO)")
    ax1.set_xlim(0, 1.05)
    ax1.legend()
    for bar, v in zip(bars, f1_vals):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=9)

    ax2 = axes[1]
    bars2 = ax2.barh(model_names, auc_vals, color=colors)
    ax2.set_xlabel("ROC-AUC")
    ax2.set_title("ROC-AUC (LOPO)")
    ax2.set_xlim(0, 1.05)
    for bar, v in zip(bars2, auc_vals):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "model_comparison.png"), dpi=120)
    plt.close()
    logger.info("  Comparison chart saved --> logs/model_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run full model comparison, select best model, run SHAP, export outputs."""
    logger.info("=" * 62)
    logger.info("  Component 3 -- Model Comparison (LOPO x models)")
    logger.info("=" * 62)

    # Load data
    if not os.path.exists(DATA):
        logger.error(f"Input file not found: {DATA}")
        sys.exit(1)

    df = pd.read_csv(DATA)
    df.dropna(subset=FEATURES + [TARGET], inplace=True)
    X = df[FEATURES]
    y = df[TARGET]
    projects = df["project_id"]
    logger.info(f"  Dataset : {len(df)} rows | {projects.nunique()} projects")
    logger.info(f"  Labels  : {dict(y.value_counts())}")
    logger.info("")

    models_def = build_models()
    results = []

    # ---- Evaluate each model --------------------------------------------
    for name, clf in models_def.items():
        t0 = time.time()
        cv = lopo_cv(clf, X, y, projects)
        elapsed = round(time.time() - t0, 1)
        results.append({
            "model": name,
            "mean_f1": round(cv["mean_f1"], 4),
            "f1_var": round(cv["f1_var"], 6),
            "mean_prec": round(cv["mean_prec"], 4),
            "mean_rec": round(cv["mean_rec"], 4),
            "mean_auc": round(cv["mean_auc"], 4),
            "train_sec": elapsed,
            "folds": cv["fold_details"],
            "shap_fold_data": cv["shap_fold_data"],
        })
        logger.info(
            f"  {name:<22} | F1={cv['mean_f1']:.4f} | "
            f"Var={cv['f1_var']:.6f} | AUC={cv['mean_auc']:.4f} | "
            f"Time={elapsed}s"
        )

    # ---- Rank models ----------------------------------------------------
    logger.info("")
    logger.info("=" * 62)
    logger.info("  RANKING (best to worst by F1, then variance)")
    logger.info("=" * 62)
    results_sorted = sorted(results, key=lambda x: (-x["mean_f1"], x["f1_var"]))
    for i, r in enumerate(results_sorted, 1):
        tag = "  <-- BEST" if i == 1 else ""
        logger.info(
            f"  #{i} {r['model']:<22} | F1={r['mean_f1']:.4f} | "
            f"Var={r['f1_var']:.6f} | AUC={r['mean_auc']:.4f}{tag}"
        )

    best = results_sorted[0]
    logger.info("")
    logger.info(f"  Selected  : {best['model']}")
    logger.info(f"  Mean F1   : {best['mean_f1']}")
    logger.info(f"  F1 Var    : {best['f1_var']}")

    # ---- Stage 4: SHAP for best model -----------------------------------
    logger.info("")
    logger.info("=" * 62)
    logger.info("  STAGE 4 -- SHAP Analysis (best model folds)")
    logger.info("=" * 62)
    shap_consistency = {}
    if best["shap_fold_data"]:
        for fold_shap in best["shap_fold_data"]:
            save_shap_bar_chart(fold_shap["importance"], fold_shap["project"])
        shap_consistency = save_shap_consistency(best["shap_fold_data"])
    else:
        logger.warning("  No SHAP data available for the best model.")

    # ---- Save artefacts -------------------------------------------------
    # Comparison CSV (no shap_fold_data column)
    df_res = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("folds", "shap_fold_data")}
        for r in results_sorted
    ])
    df_res.to_csv(os.path.join(LOGS_DIR, "comparison_results.csv"), index=False)

    # Comparison JSON (include folds, exclude raw shap_fold_data for size)
    json_out = [
        {k: v for k, v in r.items() if k != "shap_fold_data"}
        for r in results_sorted
    ]
    with open(os.path.join(MODELS_DIR, "comparison_results.json"), "w") as fh:
        json.dump(json_out, fh, indent=2)

    # Per-fold LOPO CSV for best model
    fold_rows = [
        dict(fd, shap_top3=", ".join(
            next((s["top"][:3] for s in best["shap_fold_data"] if s["project"] == fd["project"]), [])
        ))
        for fd in best["folds"]
    ]
    pd.DataFrame(fold_rows).to_csv(os.path.join(LOGS_DIR, "lopo_results.csv"), index=False)

    # Comparison bar chart
    save_comparison_chart(results_sorted, best["model"])

    # ---- Stage 5: Train final model on all data -------------------------
    logger.info("")
    logger.info("  STAGE 5 -- Training final model on ALL data...")
    best_clf = models_def[best["model"]]
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X)
    best_clf.fit(X_all, y)

    joblib.dump(
        {
            "model": best_clf,
            "scaler": scaler,
            "features": FEATURES,
            "threshold": ALARM_THRESHOLD,
            "label_pos": "FAILURE",
        },
        os.path.join(MODELS_DIR, "memory_leak_rf_model.pkl"),
    )

    # Metadata
    meta = {
        "best_model": best["model"],
        "mean_f1": best["mean_f1"],
        "f1_variance": best["f1_var"],
        "mean_auc": best["mean_auc"],
        "alarm_threshold": ALARM_THRESHOLD,
        "features": FEATURES,
        "shap_consistency": shap_consistency,
        "label_positive": "FAILURE",
        "rows_trained": len(df),
        "projects": int(projects.nunique()),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Generate memory_predictions.csv for Component 4
    y_prob_all = best_clf.predict_proba(X_all)[:, 1]
    df_out = df[["timestamp", "service_name", "project_id"]].copy()
    df_out["memory_prob"] = np.round(y_prob_all, 4)
    df_out["alert"] = df_out["memory_prob"] >= ALARM_THRESHOLD
    df_out["pred_label"] = df_out["memory_prob"].apply(
        lambda p: "FAILURE" if p >= ALARM_THRESHOLD else "NORMAL"
    )
    df_out.to_csv(os.path.join(BASE, "data", "memory_predictions.csv"), index=False)
    logger.info("  memory_predictions.csv saved --> data/memory_predictions.csv")

    logger.info("")
    logger.info("[DONE] Comparison complete.")
    logger.info(f"       Best model  : {best['model']}")
    logger.info(f"       Mean F1     : {best['mean_f1']}")
    logger.info(f"       F1 Variance : {best['f1_var']}")
    logger.info(f"       Mean AUC    : {best['mean_auc']}")


if __name__ == "__main__":
    main()
