"""
advisor.policy — Learned, cost-sensitive action recommender
===========================================================

Replaces pure threshold lookup with an explicit *expected-cost* comparison of
every candidate SRE action, while keeping the existing dual-threshold gating
matrix as a guaranteed fallback.

For an observation with failure probability ``p`` (calibrated) and a time-to-
failure forecast ``ttf`` (minutes, p50 with a p10-p90 band), each action ``a`` is
scored by::

    expected_cost(a) = residual_failure_prob(a) * FAILURE_COST
                     + a.disruption_cost
                     + a.lead_time_penalty(ttf)

``residual_failure_prob`` folds in how effective the action is AND whether it can
finish before the forecast failure (a scale-out that needs 2 min is worthless if
failure is 0.3 min away). The lowest-cost action is recommended; the top three
are returned with the reason each lower-ranked alternative lost.

Safety invariants (never violated here):
* The advisor RECOMMENDS. Actions above Tier 1 require operator approval —
  ``requires_operator_approval`` is set accordingly and the caller gates on the
  180 s circuit breaker in :mod:`src.mitigation`.
* If the model is unavailable or uncertain (conformal ``needs_human_review``),
  the deterministic gating matrix result is returned as the fallback and is
  surfaced alongside the advisor pick.

Offline policy evaluation
--------------------------
``evaluate_offline_policy`` implements Inverse-Propensity Scoring (IPS), self-
normalised IPS (SNIPS) and Doubly-Robust (DR) estimators over a logged bandit of
(context -> action, propensity, reward) so a candidate cost policy can be judged
against the frozen behaviour policy before ever touching the live loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("advisor.policy")

# ---------------------------------------------------------------------------
# Action catalogue
# ---------------------------------------------------------------------------
MITIGATION_TIER = {
    "NO_ACTION": 0,
    "LOAD_SHEDDING": 1,
    "SCALE_OUT": 2,
    "PROACTIVE_RESTART": 2,
    "CRITICAL_RESTART_AND_REROUTE": 3,
}
# Map advisor action id -> the src.mitigation.execute_mitigation_action verb.
MITIGATION_COMMAND = {
    "NO_ACTION": "NO_ACTION",
    "LOAD_SHEDDING": "TRIGGER_LOAD_SHEDDING",
    "SCALE_OUT": "TRIGGER_LOAD_SHEDDING",       # escalates to tier-2 scale verb
    "PROACTIVE_RESTART": "TRIGGER_PROACTIVE_POD_RESTART",
    "CRITICAL_RESTART_AND_REROUTE": "CRITICAL_RESTART_AND_TRAFFIC_REROUTE",
}


@dataclass
class ActionSpec:
    name: str
    tier: int
    disruption_cost: float      # immediate operational cost (user-visible units)
    effectiveness: float        # fraction of failure risk removed if it completes
    lead_time_min: float        # minutes needed to take effect
    description: str = ""


ACTIONS: dict[str, ActionSpec] = {
    a.name: a for a in [
        ActionSpec("NO_ACTION", 0, 0.0, 0.0, 0.0, "Continue monitoring; no intervention."),
        ActionSpec("LOAD_SHEDDING", 1, 8.0, 0.45, 0.2, "Envoy shed 25% traffic; fast, low disruption."),
        ActionSpec("SCALE_OUT", 2, 15.0, 0.60, 1.5, "kubectl add 2 replicas; effective but slower to provision."),
        ActionSpec("PROACTIVE_RESTART", 2, 22.0, 0.75, 2.0, "Graceful pod restart; clears leak/spike but disruptive."),
        ActionSpec("CRITICAL_RESTART_AND_REROUTE", 3, 40.0, 0.92, 3.0, "Isolate, restart, reroute; strongest, highest cost."),
    ]
}

FAILURE_COST = 120.0            # cost of an unmitigated failure (downtime + blast radius)
APPROVAL_TIER = 1              # tier above which operator approval is mandatory


@dataclass
class Decision:
    service_name: str
    recommended_action: str
    expected_cost: float
    confidence: float
    requires_operator_approval: bool
    fallback_matrix_action: str
    ranked_actions: list[dict] = field(default_factory=list)
    needs_human_review: bool = False
    rationale: str = ""


def matrix_fallback(cpu_prob: float, mem_prob: float,
                    mem_alert: bool = False, mem_label: str = "NORMAL",
                    cpu_thresh: float = 0.60, mem_thresh: float = 0.70) -> str:
    """The existing guaranteed dual-threshold decision, expressed as an advisor
    action id (CPU-only spike -> LOAD_SHEDDING, memory -> PROACTIVE_RESTART,
    both -> CRITICAL_RESTART_AND_REROUTE)."""
    cpu_alarm = cpu_prob is not None and cpu_prob >= cpu_thresh
    mem_alarm = mem_alert or (mem_prob is not None and mem_prob >= mem_thresh) \
        or (mem_label in ("WARNING", "FAILURE"))
    if cpu_alarm and mem_alarm:
        return "CRITICAL_RESTART_AND_REROUTE"
    if cpu_alarm:
        return "LOAD_SHEDDING"
    if mem_alarm:
        return "PROACTIVE_RESTART"
    return "NO_ACTION"


def _feasibility(action: ActionSpec, ttf_p50: float | None) -> float:
    """Fraction of the action's benefit realisable given the time available.

    If the action needs ``lead_time_min`` but failure is forecast in ``ttf``,
    a linear ramp penalises actions that cannot finish in time.
    """
    if action.lead_time_min <= 0:
        return 1.0
    if ttf_p50 is None:
        return 1.0
    if ttf_p50 >= action.lead_time_min:
        return 1.0
    return float(np.clip(ttf_p50 / action.lead_time_min, 0.05, 1.0))


def score_actions(cpu_prob: float, mem_prob: float | None,
                  ttf_p50: float | None, lead_window_min: float = 5.0) -> list[dict]:
    """Return every action scored by expected cost (ascending)."""
    # Joint failure probability from the two independent channels.
    p_cpu = float(np.clip(cpu_prob, 0.0, 1.0))
    p_mem = float(np.clip(mem_prob, 0.0, 1.0)) if mem_prob is not None else 0.0
    p_fail = 1.0 - (1.0 - p_cpu) * (1.0 - p_mem)

    scored = []
    for a in ACTIONS.values():
        feas = _feasibility(a, ttf_p50)
        residual = p_fail * (1.0 - a.effectiveness * feas)
        # Time pressure: only meaningful when a failure is likely soon.
        time_pressure = 0.0
        if ttf_p50 is not None and p_fail > 0.5:
            time_pressure = max(0.0, lead_window_min - ttf_p50) * 2.0 * p_fail
        expected_cost = residual * FAILURE_COST + a.disruption_cost + time_pressure
        prevented = p_fail - residual
        scored.append({
            "action": a.name,
            "tier": a.tier,
            "expected_cost": round(float(expected_cost), 3),
            "residual_failure_prob": round(float(residual), 4),
            "risk_reduced": round(float(prevented), 4),
            "feasibility": round(float(feas), 3),
            "description": a.description,
        })
    scored.sort(key=lambda d: d["expected_cost"])
    return scored


def _confidence(scored: list[dict]) -> float:
    """Confidence rises with the cost gap between the best and runner-up."""
    if len(scored) < 2:
        return 0.5
    best, second = scored[0]["expected_cost"], scored[1]["expected_cost"]
    spread = second - best
    denom = abs(second) + 1e-6
    return float(np.clip(0.5 + 0.5 * (spread / denom), 0.5, 0.99))


def decide(service_name: str, cpu_prob: float, mem_prob: float | None,
           ttf_p50: float | None, needs_human_review: bool = False,
           ttf_band: tuple[float, float] | None = None,
           lead_window_min: float = 5.0) -> Decision:
    """Produce the operator-facing recommendation with top-3 ranking + reasons."""
    scored = score_actions(cpu_prob, mem_prob, ttf_p50, lead_window_min)
    best = scored[0]
    fallback = matrix_fallback(cpu_prob, mem_prob)

    ranked = []
    for i, s in enumerate(scored[:3]):
        reason = "lowest expected cost" if i == 0 else \
            f"higher expected cost (+{s['expected_cost'] - scored[0]['expected_cost']:.1f})"
        if s["action"] != scored[0]["action"] and s["feasibility"] < 1.0:
            reason += f"; cannot complete in forecast time (feasibility {s['feasibility']:.0%})"
        ranked.append({**s, "rank": i + 1, "reason": reason})

    confidence = _confidence(scored)
    if needs_human_review:
        confidence = min(confidence, 0.6)

    return Decision(
        service_name=service_name,
        recommended_action=best["action"],
        expected_cost=best["expected_cost"],
        confidence=round(confidence, 3),
        requires_operator_approval=best["tier"] > APPROVAL_TIER,
        fallback_matrix_action=fallback,
        ranked_actions=ranked,
        needs_human_review=needs_human_review,
        rationale=(f"{best['action']} minimises expected cost "
                   f"({best['expected_cost']}) reducing failure risk by "
                   f"{best['risk_reduced']*100:.0f}% at calibrated P(fail)."),
    )


# ---------------------------------------------------------------------------
# Offline policy evaluation (IPS / SNIPS / DR)
# ---------------------------------------------------------------------------
def behaviour_policy_propensities(cpu_prob: float, mem_prob: float | None,
                                  tau: float = 1.0) -> dict[str, float]:
    """A soft-max over the gating matrix costs — the historical logging policy
    whose propensities weight the offline evaluation."""
    scores = {}
    for name, a in ACTIONS.items():
        fb = matrix_fallback(cpu_prob, mem_prob)
        match = 1.0 if name == fb else -0.5
        scores[name] = match / max(tau, 1e-6) - 0.02 * a.tier
    mx = max(scores.values())
    exps = {k: np.exp(v - mx) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: float(v / z) for k, v in exps.items()}


def simulate_incident_log(df: pd.DataFrame, prob_col: str, n: int = 5000,
                          seed: int = 42) -> pd.DataFrame:
    """Build a synthetic logged bandit from real prediction rows.

    Each row: a context (cpu/mem prob), the behaviour-policy action + propensity,
    a simulated reward = -(realised failure cost) reflecting whether that action
    would have prevented the observed outcome. Used only for OPE demonstration.
    """
    rng = np.random.default_rng(seed)
    sample = df.dropna(subset=[prob_col]).sample(min(n, len(df)), random_state=seed)
    rows = []
    for _, r in sample.iterrows():
        cpu = float(r[prob_col])
        mem = float(r["memory_prob"]) if "memory_prob" in r and pd.notna(r["memory_prob"]) else 0.0
        label = int(r["imminent_failure"]) if "imminent_failure" in r else int(cpu >= 0.6)
        props = behaviour_policy_propensities(cpu, mem)
        names = list(props.keys())
        p = np.array([props[k] for k in names])
        a = names[int(rng.choice(len(names), p=p / p.sum()))]
        # Reward: negative cost; failure occurred (label=1) and action ineffective.
        eff = ACTIONS[a].effectiveness
        failed_anyway = label == 1 and rng.random() > eff
        reward = -(FAILURE_COST * int(failed_anyway) + ACTIONS[a].disruption_cost)
        rows.append({"context_cpu": cpu, "context_mem": mem, "label": label,
                     "action": a, "propensity": props[a], "reward": float(reward)})
    return pd.DataFrame(rows)


def _target_policy_action(cpu: float, mem: float) -> str:
    scored = score_actions(cpu, mem, ttf_p50=None)
    return scored[0]["action"]


def evaluate_offline_policy(log_df: pd.DataFrame, estimator: str = "snips") -> dict:
    """Estimate the value of the deterministic cost policy from a logged bandit.

    ``log_df`` columns: context_cpu, context_mem, action, propensity, reward,
    plus an optional model-based ``r_hat`` for the DR estimator.
    """
    if log_df.empty:
        return {"estimator": estimator, "value": None, "ess": 0.0, "n": 0}

    rewards = log_df["reward"].to_numpy(dtype=float)
    props = log_df["propensity"].to_numpy(dtype=float)
    actions = log_df["action"].tolist()
    target = [_target_policy_action(c, m) for c, m in
              zip(log_df["context_cpu"], log_df["context_mem"])]
    # Deterministic target policy -> importance weight is 1 if logged action
    # matches the target action, else 0.
    match = np.array([1.0 if a == t else 0.0 for a, t in zip(actions, target)])
    w = match / np.clip(props, 1e-6, None)

    est = estimator.lower()
    if est == "ips":
        value = float(np.mean(w * rewards))
    elif est == "snips":
        denom = np.sum(w)
        value = float(np.sum(w * rewards) / denom) if denom > 0 else float("nan")
    elif est == "dr":
        r_hat = log_df["r_hat"].to_numpy(dtype=float) if "r_hat" in log_df else rewards
        value = float(np.mean(r_hat + w * (rewards - r_hat)))
    else:
        raise ValueError(f"Unknown estimator '{estimator}' (use ips|snips|dr)")

    ess = float((np.sum(w) ** 2) / np.sum(w ** 2)) if np.any(w) else 0.0
    return {"estimator": est, "value": value, "ess": ess, "n": int(len(log_df)),
            "agreement": float(np.mean(match))}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    d = decide("demo-svc", cpu_prob=0.72, mem_prob=0.15, ttf_p50=4.0)
    print(d.recommended_action, d.expected_cost, d.confidence)
    for r in d.ranked_actions:
        print(" ", r["rank"], r["action"], r["expected_cost"], "|", r["reason"])
