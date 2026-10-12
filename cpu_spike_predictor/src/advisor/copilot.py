"""
advisor.copilot — Grounded operator decision brief
==================================================

Produces a plain-language brief (<=60 words) an SRE can act on in seconds.

* Attribution uses a REAL :class:`shap.TreeExplainer` over the Random Forest when
  Shap is importable, memoised per model so the expensive explainer build happens
  once; the existing fast approximate path in :mod:`src.xai` remains the fallback
  so the hot loop never blocks on SHAP construction.
* The default brief is a DETERMINISTIC template assembled only from structured
  facts we own (calibrated risk, TTF band, ranked action, top SHAP driver,
  runbook id). No network, fully reproducible, always on the hot path.
* An optional LLM backend lives behind the ``SENTINEL_ADVISOR_LLM`` env flag.
  When enabled it is fed *only* those structured facts, must return a strict JSON
  object, runs with a hard timeout, and automatically falls back to the template
  on any error/timeout/bad-schema. It is NOT on the hot prediction path — the API
  calls the template synchronously and may enrich asynchronously + cache.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.model import COMP2_CPU_FEATURE_COLS
from src.xai import FEATURE_DISPLAY_NAMES, compute_shapley_values

logger = logging.getLogger("advisor.copilot")

MAX_BRIEF_WORDS = 60
RUNBOOK_BY_ACTION = {
    "NO_ACTION": "RB-04",
    "LOAD_SHEDDING": "RB-01",
    "SCALE_OUT": "RB-02",
    "PROACTIVE_RESTART": "RB-03",
    "CRITICAL_RESTART_AND_REROUTE": "RB-03",
}

# Operator-facing checklists (Phase 1 Explainability view). Each is a sequence
# of verification steps, not executable commands.
RUNBOOK_CHECKLISTS = {
    "RB-01": [
        "Confirm queue growth and in-flight depth are rising on the target service",
        "Apply Envoy ingress rate limit at 25% drop ratio (Tier 1, reversible)",
        "Watch incoming vs processing rate until the queue drains",
        "Release the rate limit once queue_pressure_index returns to baseline",
        "Record the incident note and keep the service on watch for 180 s",
    ],
    "RB-02": [
        "Verify host/node capacity headroom before scaling",
        "Scale deployment out by +2 replicas (Tier 2)",
        "Wait for new pods to pass readiness probes",
        "Confirm per-pod CPU falls below the alarm threshold",
        "Schedule scale-back after the spike window clears",
    ],
    "RB-03": [
        "Capture a thread/heap snapshot before replacing the pod",
        "Drain ingress traffic from the target pod",
        "Restart / replace the pod gracefully (Tier 2-3)",
        "Verify rollout status and readiness within 60 s",
        "Post-recovery: inspect queue back-pressure and log root cause",
    ],
    "RB-04": [
        "No action required — metrics are within baseline",
        "Continue standard monitoring cadence",
        "Verify thresholds and drift advisories on schedule",
    ],
}


# ---------------------------------------------------------------------------
# Attribution: real TreeExplainer with cached fast path + approximate fallback
# ---------------------------------------------------------------------------
_EXPLAINER_CACHE: dict[int, Any] = {}


def _get_explainer(model):
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        try:
            import shap
            _EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
        except Exception as exc:  # pragma: no cover - env dependent
            logger.info("TreeExplainer unavailable (%s); using approximate path.", exc)
            _EXPLAINER_CACHE[key] = None
    return _EXPLAINER_CACHE[key]


def attributions(model, input_df: pd.DataFrame) -> tuple[dict[str, float], float, str]:
    """Return (phi_by_feature, base_value, method).

    method is 'treeshap' when the real explainer ran, else 'approximate'.
    """
    feature_cols = [c for c in COMP2_CPU_FEATURE_COLS if c in input_df.columns]
    explainer = _get_explainer(model) if model is not None else None
    if explainer is not None:
        try:
            X = input_df[feature_cols].astype(float)
            sv = explainer.shap_values(X)
            # shap returns one of:
            #   ndarray (n, features)            -> already class-1 / single class
            #   ndarray (n, features, n_classes)  -> take the failure class slice
            #   list  [class0, class1]            -> take class1
            if isinstance(sv, list):
                cls_arr = np.asarray(sv[1] if len(sv) > 1 else sv[0], dtype=float)
            else:
                arr = np.asarray(sv, dtype=float)
                if arr.ndim == 3:
                    cls_arr = arr[:, :, 1]
                elif arr.ndim == 2:
                    cls_arr = arr
                else:
                    cls_arr = arr.reshape(1, -1)
            phi_row = np.asarray(cls_arr)[0]
            ev = np.ravel(np.asarray(explainer.expected_value, dtype=float))
            # For binary RF, expected_value may be [class0, class1]; take class1.
            base = float(ev[1] if len(ev) > 1 else ev[0])
            phi = {c: round(float(v), 4) for c, v in zip(feature_cols, phi_row)}
            return phi, base, "treeshap"
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("TreeExplainer call failed (%s); approximating.", exc)

    _base, phi, _pct = compute_shapley_values(model, input_df, feature_cols)
    return phi, _base, "approximate"


def top_driver(phi: dict[str, float]) -> tuple[str, float]:
    if not phi:
        return "none", 0.0
    feat = max(phi.items(), key=lambda kv: abs(kv[1]))
    return feat[0], float(feat[1])


# ---------------------------------------------------------------------------
# Deterministic template brief (default, hot-path safe)
# ---------------------------------------------------------------------------
def _truncate_words(text: str, limit: int = MAX_BRIEF_WORDS) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(",.;") + "."


def build_brief_template(service_name: str, risk: float, confidence: float,
                         action: str, ttf_p50: float | None,
                         ttf_band: tuple[float, float] | None,
                         phi: dict[str, float],
                         needs_human_review: bool = False,
                         attribution_method: str = "approximate") -> str:
    feat, val = top_driver(phi)
    driver = FEATURE_DISPLAY_NAMES.get(feat, feat)
    driver_tag = "SHAP" if attribution_method == "treeshap" else "contribution"
    parts = [
        f"{service_name}: risk {risk*100:.0f}% (conf {confidence*100:.0f}%).",
    ]
    if ttf_p50 is not None and risk >= 0.5:
        if ttf_p50 <= 0.1:
            parts.append("Failure imminent (<1 min).")
        else:
            band = f" (range {ttf_band[0]:.1f}-{ttf_band[1]:.1f} min)" if ttf_band else ""
            parts.append(f"~{ttf_p50:.1f} min to failure{band}.")
    parts.append(f"Recommend {action.replace('_', ' ').lower()} [{RUNBOOK_BY_ACTION.get(action, 'RB-04')}],")
    parts.append(f"driven by {driver} ({driver_tag} {val:+.3f}).")
    if needs_human_review:
        parts.append("Uncertain — verify before acting.")
    else:
        parts.append("Above Tier-1 needs your approval.")
    return _truncate_words(" ".join(parts))


# ---------------------------------------------------------------------------
# Optional LLM backend (off by default, never on the hot path)
# ---------------------------------------------------------------------------
LLM_BRIEF_SCHEMA = {"type": "object",
                    "required": ["brief", "confidence"],
                    "properties": {"brief": {"type": "string"},
                                   "confidence": {"type": "number"}}}


def _llm_enabled() -> bool:
    return os.getenv("SENTINEL_ADVISOR_LLM", "").strip().lower() in ("1", "true", "on", "yes")


def _facts_json(service_name, risk, confidence, action, ttf_p50, ttf_band, phi, review) -> str:
    return json.dumps({
        "service": service_name, "failure_risk": round(risk, 3),
        "confidence": round(confidence, 3), "recommended_action": action,
        "time_to_failure_min": ttf_p50, "ttf_band_min": ttf_band,
        "top_attribution": {k: v for k, v in sorted(phi.items(), key=lambda kv: -abs(kv[1]))[:5]},
        "needs_human_review": bool(review),
    }, sort_keys=True)


@lru_cache(maxsize=256)
def _cached_llm(facts_hash: str) -> str | None:
    return None


def build_brief(service_name: str, risk: float, confidence: float, action: str,
                ttf_p50: float | None, ttf_band: tuple[float, float] | None,
                phi: dict[str, float], needs_human_review: bool = False,
                allow_llm: bool = False, timeout_s: float = 2.5,
                attribution_method: str = "approximate") -> dict:
    """Return {"brief", "method", "facts_hash"}. Template is default and the
    guaranteed output; LLM only when explicitly opted-in AND env flag is set."""
    template = build_brief_template(service_name, risk, confidence, action,
                                    ttf_p50, ttf_band, phi, needs_human_review)
    facts = _facts_json(service_name, risk, confidence, action, ttf_p50, ttf_band, phi, needs_human_review)
    facts_hash = hashlib.sha256(facts.encode()).hexdigest()[:16]
    result = {"brief": template, "method": "template", "facts_hash": facts_hash}

    if not (allow_llm and _llm_enabled()):
        return result

    endpoint = os.getenv("SENTINEL_ADVISOR_LLM_ENDPOINT", "")
    api_key = os.getenv("SENTINEL_ADVISOR_LLM_API_KEY", "")
    if not endpoint:
        return result
    try:
        import requests
        payload = {
            "model": os.getenv("SENTINEL_ADVISOR_LLM_MODEL", "local"),
            "response_format": {"type": "json_object"},
            "max_tokens": 120,
            "messages": [
                {"role": "system",
                 "content": "You are an SRE decision copilot. Reply ONLY with JSON "
                            f"matching schema {json.dumps(LLM_BRIEF_SCHEMA)}. Use ONLY "
                            "the supplied facts; do not invent numbers."},
                {"role": "user", "content": facts},
            ],
        }
        t0 = time.perf_counter()
        resp = requests.post(endpoint, json=payload,
                             headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                             timeout=timeout_s)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, dict) and isinstance(parsed.get("brief"), str) \
                and isinstance(parsed.get("confidence"), (int, float)):
            if time.perf_counter() - t0 <= timeout_s:
                result["brief"] = _truncate_words(parsed["brief"])
                result["method"] = "llm"
    except Exception as exc:  # any failure => deterministic fallback
        logger.info("LLM brief failed (%s); using template.", exc)
    return result


if __name__ == "__main__":  # pragma: no cover
    demo_phi = {"queue_pressure_index": 0.31, "cpu_percent": 0.22, "cpu_velocity": 0.08,
                "in_flight_queue": -0.01, "overload_flag": 0.0}
    print(build_brief("checkout", 0.74, 0.62, "PROACTIVE_RESTART", 4.2, (2.1, 9.0),
                      demo_phi, needs_human_review=False)["brief"])
