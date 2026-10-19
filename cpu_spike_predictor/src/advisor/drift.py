"""
advisor.drift — Data & concept drift + retrain advisor
======================================================

Per-service drift detection and an *advisory only* threshold/retrain policy.

Detectors
---------
* Population Stability Index (PSI) on the prediction distribution (data drift).
* Two-sample Kolmogorov-Smirnov (SciPy) as a non-parametric distribution test.
* Page-Hinkley and an adaptive-window (ADWIN-style) test on the model *residual*
  stream to catch concept drift (the error distribution moving).

Advisor output (never auto-applied)
-----------------------------------
``advise_for_service`` returns one of:
  * ``KEEP_THRESHOLDS``
  * ``ADJUST_THRESHOLD``  — with the candidate value and the *projected*
    FAR / recall trade-off so the operator sees the consequence first,
  * ``RETRAIN_RECOMMENDED`` — with a reason, requiring human approval.

Retrain suggestions are recorded in a small versioned registry that supports
rollback and never silently swaps the live model.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger("advisor.drift")

PSI_STABLE = 0.10
PSI_MODERATE = 0.25
KS_ALPHA = 0.01
VERSION = "advisor-drift-v1"


def population_stability_index(reference: np.ndarray, current: np.ndarray,
                               bins: int = 10) -> float:
    """PSI between two 1-D samples on shared reference-derived bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref_counts = np.histogram(reference, edges)[0].astype(float)
    cur_counts = np.histogram(current, edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    from scipy.stats import ks_2samp
    stat, p = ks_2samp(np.asarray(reference, dtype=float),
                       np.asarray(current, dtype=float))
    return {"ks_stat": float(stat), "p_value": float(p)}


class PageHinkley:
    """Sequential change-point detector on a stream of residuals."""

    def __init__(self, min_delta: float = 1.0, threshold: float = 10.0,
                 lambda_: float = 2.0):
        self.min_delta = min_delta
        self.threshold = threshold
        self.lambda_ = lambda_
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.x_mean = 0.0
        self.min_sum = np.inf
        self.alarm = False

    def update(self, x: float):
        self.n += 1
        self.x_mean += (x - self.x_mean) / self.n
        self.sum += x - self.x_mean - self.min_delta
        if self.sum < self.min_sum:
            self.min_sum = self.sum
        ph = self.sum - self.min_sum
        if ph > self.threshold:
            self.alarm = True
        return ph

    def run(self, series) -> dict:
        self.reset()
        for v in np.asarray(series, dtype=float):
            self.update(float(v))
        return {"alarm": bool(self.alarm), "n": int(self.n)}


class ADWINWindow:
    """Adaptive-window mean-shift detector (simplified ADWIN).

    Keeps a growing window and flags a change when some split into two sub-windows
    differs beyond a Hoeffding-bound slack controlled by ``delta``.
    """

    def __init__(self, delta: float = 0.005, max_window: int = 1000,
                 max_scan: int = 200):
        self.delta = delta
        self.max_window = max_window
        self.max_scan = max_scan
        self.window: list[float] = []
        self.alarm = False

    def _hoeffding_bound(self, m: float) -> float:
        if m <= 0:
            return np.inf
        v = 1.0  # residuals assumed range-scaled to unit variance
        return float(np.sqrt((2.0 * v ** 2 * np.log(2.0 / self.delta)) / (2.0 * m)))

    def update(self, x: float) -> bool:
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            self.window = self.window[-self.max_window:]
        n = len(self.window)
        if n < 8:
            return False
        # Precompute prefix sums so any window mean is O(1); scan only the most
        # recent max_scan split points to keep the update cheap for the hot path.
        csum = np.cumsum(self.window)
        start = max(1, n - self.max_scan)
        alarm = False
        for i in range(start, n):
            n_left, n_right = i, n - i
            if n_left < 4 or n_right < 4:
                continue
            mean_left = csum[i - 1] / n_left
            mean_right = (csum[n - 1] - csum[i - 1]) / n_right
            m = 1.0 / (1.0 / n_left + 1.0 / n_right)
            if abs(mean_left - mean_right) - self._hoeffding_bound(m) > 0:
                alarm = True
                self.window = self.window[i:]
                break
        if alarm:
            self.alarm = True
        return alarm

    def run(self, series) -> dict:
        self.window = []
        self.alarm = False
        for v in np.asarray(series, dtype=float):
            self.update(float(v))
        return {"alarm": bool(self.alarm), "window_size": int(len(self.window))}


def project_far_recall(y_true, probs, threshold: float) -> dict:
    """FAR and recall of a single threshold on (probs, labels)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    far = fp / (tn + fp) if (tn + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"threshold": float(threshold), "far": round(far, 4),
            "recall": round(recall, 4), "alarms": int(pred.sum())}


def suggest_thresholds(y_true, probs, grid=None, min_recall: float = 0.85) -> dict:
    """Pick the threshold meeting ``min_recall`` with the lowest FAR."""
    grid = grid if grid is not None else np.round(np.arange(0.30, 0.86, 0.05), 2)
    evaluated = [project_far_recall(y_true, probs, t) for t in grid]
    feasible = [e for e in evaluated if e["recall"] >= min_recall]
    best = min(feasible, key=lambda e: e["far"]) if feasible else min(
        evaluated, key=lambda e: abs(e["recall"] - min_recall))
    return {"candidates": evaluated, "recommended": best,
            "min_recall_target": min_recall}


def advise_for_service(reference_probs, current_probs, residuals,
                       y_true_current=None, current_threshold: float = 0.6) -> dict:
    """Combine detectors into a single advisory decision for one service."""
    reference_probs = np.asarray(reference_probs, dtype=float)
    current_probs = np.asarray(current_probs, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    psi = population_stability_index(reference_probs, current_probs)
    ks = ks_drift(reference_probs, current_probs)
    ph = PageHinkley().run(residuals)
    adwin = ADWINWindow().run(residuals)

    severe = psi > PSI_MODERATE or (ks["p_value"] < KS_ALPHA and abs(
        np.mean(current_probs) - np.mean(reference_probs)) > 0.15)
    concept_drift = ph["alarm"] or adwin["alarm"]
    moderate = (PSI_STABLE < psi <= PSI_MODERATE) or ks["p_value"] < KS_ALPHA

    advice = "KEEP_THRESHOLDS"
    payload = {"psi": round(psi, 4), "ks": ks, "page_hinkley": ph, "adwin": adwin,
               "current_threshold": current_threshold}

    if severe and concept_drift:
        advice = "RETRAIN_RECOMMENDED"
        payload["reason"] = ("Severe data drift (PSI/KS) with concurrent concept "
                             "drift on residuals — live model no longer matches "
                             "production distribution.")
    elif severe or concept_drift or moderate:
        advice = "ADJUST_THRESHOLD"
        if y_true_current is not None and len(np.asarray(y_true_current)) > 0:
            sug = suggest_thresholds(y_true_current, current_probs)
            payload["threshold_suggestion"] = sug["recommended"]
            payload["reason"] = (f"Drift detected; moving threshold to "
                                 f"{sug['recommended']['threshold']:.2f} projects "
                                 f"FAR {sug['recommended']['far']:.2f} / "
                                 f"recall {sug['recommended']['recall']:.2f}.")
        else:
            payload["reason"] = "Drift detected; recompute threshold once labels arrive."
    else:
        payload["reason"] = "Distributions stable; keep current thresholds."

    payload["advice"] = advice
    # Advisory only — always requires an explicit human action to apply.
    payload["requires_human_approval"] = advice != "KEEP_THRESHOLDS"
    return payload


class RetrainRegistry:
    """Versioned, append-only log of drift advice + retrain proposals.

    Supports rollback and — critically — never replaces the live model: it only
    records what SHOULD happen pending human approval.
    """

    def __init__(self, path: str = "outputs/advisor_drift_registry.json"):
        self.path = path
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": VERSION, "entries": [], "active_version": "live-rf-base"}

    def record(self, service: str, advice: dict, approved: bool = False) -> dict:
        entry = {
            "service": service,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "advice": advice.get("advice"),
            "detail": advice,
            "approved": bool(approved),
            "applied": False,  # only a human-gated apply step could flip this
        }
        self.state["entries"].append(entry)
        if len(self.state["entries"]) > 2000:
            self.state["entries"] = self.state["entries"][-2000:]
        self._save()
        return entry

    def rollback(self, to_version: str | None = None) -> dict:
        self.state["active_version"] = to_version or "live-rf-base"
        self._save()
        return {"active_version": self.state["active_version"]}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp, self.path)


def run_drift_report(predictions_df: pd.DataFrame,
                     reference_project: str = "Proj_01",
                     out_dir: str = "outputs") -> pd.DataFrame:
    """Per-service drift report over a predictions frame (project vs rest)."""
    if "cpu_failure_prob" not in predictions_df.columns:
        return pd.DataFrame()
    df = predictions_df.copy()
    ref_mask = df["project_id"] == reference_project if "project_id" in df else pd.Series(True, index=df.index)
    reference = df.loc[ref_mask, "cpu_failure_prob"].to_numpy()
    rows = []
    registry = RetrainRegistry(os.path.join(out_dir, "advisor_drift_registry.json"))
    for svc, g in df.groupby("service_name"):
        current = g["cpu_failure_prob"].to_numpy()
        if len(current) < 10 or len(reference) < 10:
            continue
        resid = (current - g["imminent_failure"].to_numpy()) if "imminent_failure" in g else current - current.mean()
        advice = advise_for_service(reference, current, resid,
                                    y_true_current=g["imminent_failure"].to_numpy()
                                    if "imminent_failure" in g else None)
        registry.record(str(svc), advice)
        rows.append({"service_name": svc, "n": int(len(current)),
                     "advice": advice["advice"], "psi": advice["psi"],
                     "ks_p_value": round(advice["ks"]["p_value"], 5),
                     "page_hinkley_alarm": advice["page_hinkley"]["alarm"],
                     "adwin_alarm": advice["adwin"]["alarm"]})
    report = pd.DataFrame(rows)
    if not report.empty:
        os.makedirs(out_dir, exist_ok=True)
        report.to_csv(os.path.join(out_dir, "advisor_drift_report.csv"), index=False)
    return report


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    ref = rng.normal(0.3, 0.1, 2000).clip(0, 1)
    cur = rng.normal(0.6, 0.15, 2000).clip(0, 1)
    verdict = advise_for_service(ref, cur, cur - ref)["advice"]
    print("advice:", verdict)
