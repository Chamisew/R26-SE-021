"""
advisor.calibration — Probability calibration + uncertainty
===========================================================

Turns the Random Forest's raw score into a *trustworthy* probability and
attaches a defensible confidence statement to every prediction.

* Calibration (isotonic by default, Platt optional) is fit **strictly inside a
  LOPO training fold** via :class:`CalibratedClassifierCV` with internal CV, so
  a held-out project is never used to calibrate the model evaluated on it.
* Quality is reported with Expected Calibration Error (ECE) and Brier score,
  before and after calibration.
* Split conformal prediction gives a distribution-free coverage guarantee: the
  conformal prediction set is ``{0}``, ``{1}`` or ``{0,1}``. When the set contains
  both classes the prediction is ambiguous and flagged
  ``needs_human_review=True`` — the "uncertain, ask a human" behaviour the
  Decision Advisor requires.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.model import (
    train_model, TARGET_COL, get_features_and_target,
)
from src.advisor import assert_no_memory_leakage, RANDOM_STATE

logger = logging.getLogger("advisor.calibration")

ALPHA_DEFAULT = 0.10  # 90% conformal coverage target


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width-binned ECE between predicted probability and empirical rate."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(conf - acc)
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def calibrate_fold(train_df: pd.DataFrame, method: str = "isotonic"):
    """Train RF then wrap it in a calibrated model fit ONLY on ``train_df``.

    Returns (raw_model, calibrated_model). The calibrated model uses internal
    cross-validation on the training fold; the eventual test project is unseen.

    Robust to degenerate/imbalanced folds: the number of calibration folds is
    reduced to what the minority class can support, and if the fold cannot be
    calibrated at all the calibrated model is returned as ``None`` so the
    caller can fall back to the raw probabilities (never crash the LOPO run).
    """
    from sklearn.calibration import CalibratedClassifierCV
    X_train, y_train = get_features_and_target(train_df, TARGET_COL)
    assert_no_memory_leakage(list(X_train.columns))
    raw = train_model(X_train, y_train, random_state=RANDOM_STATE)
    counts = np.bincount(np.asarray(y_train).ravel().astype(int), minlength=2)
    minority = int(counts.min())
    cv = int(max(2, min(3, minority)))
    if minority < 2:
        return raw, None
    cal = CalibratedClassifierCV(raw, method=method, cv=cv)
    try:
        cal.fit(X_train, y_train)
    except ValueError:
        # Too few examples per class for even 2-fold internal CV on this fold.
        return raw, None
    return raw, cal


def _predict_calibrated(cal, df: pd.DataFrame) -> np.ndarray:
    X, _ = get_features_and_target(df, TARGET_COL)
    probs = cal.predict_proba(X)
    classes = list(cal.classes_)
    col = classes.index(1) if 1 in classes else len(classes) - 1
    return probs[:, col]


class SplitConformalClassifier:
    """Top-label conformal predictor with a human-review flag.

    Non-conformity score ``s = 1 - p(true class)`` is computed on a held-in
    calibration split; the ``{0,1}`` prediction rule uses the ``(1-alpha)``
    quantile of those scores.
    """

    def __init__(self, alpha: float = ALPHA_DEFAULT):
        self.alpha = alpha
        self.q_hat: float | None = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "SplitConformalClassifier":
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob, dtype=float)
        p_true = np.where(y_true == 1, y_prob, 1.0 - y_prob)
        scores = 1.0 - p_true
        n = len(scores)
        if n == 0:
            self.q_hat = 1.0
            return self
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_sets(self, y_prob: np.ndarray):
        y_prob = np.asarray(y_prob, dtype=float)
        if self.q_hat is None:
            raise RuntimeError("SplitConformalClassifier.fit must be called first.")
        sets, conf, review = [], [], []
        for p in y_prob:
            include_one = (1.0 - p) <= self.q_hat
            include_zero = p <= self.q_hat
            s = set()
            if include_one:
                s.add(1)
            if include_zero:
                s.add(0)
            if not s:
                s.add(1 if p >= 0.5 else 0)
            ambiguous = (s == {0, 1})
            sets.append(sorted(s))
            conf.append(float(1.0 - self.alpha))
            review.append(bool(ambiguous))
        return sets, conf, review


def run_calibration_lopo(merged_df: pd.DataFrame,
                         method: str = "isotonic",
                         alpha: float = ALPHA_DEFAULT,
                         out_dir: str = "outputs",
                         sample_rows: int = 20000) -> pd.DataFrame:
    """Per-project LOPO calibration + conformal evaluation.

    Reports ECE/Brier before and after calibration and empirical conformal
    coverage per fold.
    """
    df = merged_df.copy()
    projects = sorted(df["project_id"].dropna().unique()) if "project_id" in df.columns else ["Proj_01"]
    rows = []

    for test_proj in projects:
        if len(projects) > 1:
            train_df = df[df["project_id"] != test_proj]
            test_df = df[df["project_id"] == test_proj]
        else:
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE,
                                                 stratify=df.get(TARGET_COL, None))
        if len(train_df) > sample_rows:
            # Manual stratified downsample (pandas .sample has no stratify kwarg)
            # so the rare class survives — isotonic needs both classes present.
            try:
                frac = sample_rows / len(train_df)
                parts = [g.sample(max(2, int(len(g) * frac)), random_state=RANDOM_STATE)
                         for _, g in train_df.groupby(TARGET_COL, dropna=False)]
                train_df = pd.concat(parts).reset_index(drop=True)
            except (ValueError, KeyError):
                train_df = train_df.sample(sample_rows, random_state=RANDOM_STATE)
        if test_df.empty or train_df.empty:
            continue
        if train_df[TARGET_COL].nunique() < 2:
            continue
        # A test fold needs both classes and enough rows to give a meaningful
        # ECE/Brier/coverage estimate; skip the collapsed micro-projects.
        y_true = test_df[TARGET_COL].fillna(0).astype(int).values
        if len(y_true) < 30 or np.unique(y_true).size < 2:
            continue

        raw, cal = calibrate_fold(train_df, method=method)

        X_test, _ = get_features_and_target(test_df, TARGET_COL)
        raw_prob = raw.predict_proba(X_test)[:, list(raw.classes_).index(1)]
        cal_prob = _predict_calibrated(cal, test_df) if cal is not None else raw_prob

        # Conformal fit uses the TRAINING fold only (never the test project).
        train_cal_prob = _predict_calibrated(cal, train_df) if cal is not None else \
            raw.predict_proba(get_features_and_target(train_df, TARGET_COL)[0])[:,
                list(raw.classes_).index(1)]
        train_true = train_df[TARGET_COL].fillna(0).astype(int).values
        conf_model = SplitConformalClassifier(alpha=alpha).fit(train_cal_prob, train_true)
        _sets, _conf, review = conf_model.predict_sets(cal_prob)
        covered = [1 if (y_true[i] in _sets[i]) else 0 for i in range(len(_sets))]

        rows.append({
            "test_project": test_proj,
            "n_test": int(len(test_df)),
            "calibration_applied": int(cal is not None),
            "ece_raw": expected_calibration_error(y_true, raw_prob),
            "ece_calibrated": expected_calibration_error(y_true, cal_prob),
            "brier_raw": brier_score(y_true, raw_prob),
            "brier_calibrated": brier_score(y_true, cal_prob),
            "conformal_coverage": float(np.mean(covered)) if covered else 0.0,
            "human_review_rate": float(np.mean(review)) if review else 0.0,
            "q_hat": conf_model.q_hat,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, "advisor_calibration_lopo.csv"), index=False)
        summary = {
            "method": method,
            "mean_ece_raw": float(out["ece_raw"].mean()),
            "mean_ece_calibrated": float(out["ece_calibrated"].mean()),
            "mean_brier_raw": float(out["brier_raw"].mean()),
            "mean_brier_calibrated": float(out["brier_calibrated"].mean()),
            "mean_conformal_coverage": float(out["conformal_coverage"].mean()),
            "mean_human_review_rate": float(out["human_review_rate"].mean()),
            "folds_total": int(len(out)),
            "folds_calibrated": int(out["calibration_applied"].sum())
            if "calibration_applied" in out else 0,
            "generated": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(out_dir, "advisor_calibration_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Calibration LOPO: ECE %.3f->%.3f Brier %.3f->%.3f coverage %.3f",
                    summary["mean_ece_raw"], summary["mean_ece_calibrated"],
                    summary["mean_brier_raw"], summary["mean_brier_calibrated"],
                    summary["mean_conformal_coverage"])
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    from src.ingestion import load_aligned_telemetry
    run_calibration_lopo(load_aligned_telemetry())
