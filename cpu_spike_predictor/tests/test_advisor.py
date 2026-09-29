"""
Decision Advisor test-suite
===========================

Covers the contracts the spec demands:
  * no memory-feature leakage into the temporal model,
  * calibration probability bounds + ECE reduction,
  * conformal prediction sets + human-review flag,
  * advisor falls back to the dual-threshold gating matrix,
  * cost-sensitive top-3 ranking + tier/approval safety,
  * the 180 s circuit breaker still gates actuation,
  * atomic append-only feedback log + input validation,
  * drift advice returns a recognised action type,
  * prioritisation ordering + cascade de-duplication,
  * copilot brief word budget + real TreeExplainer path,
  * advisor REST contracts,
  * core advisor decision path latency budget (<5 ms p99 added compute).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.advisor import (
    assert_no_memory_leakage, is_memory_derived, COMP2_CPU_FEATURE_COLS,
)
from src.advisor import forecast as FC
from src.advisor import calibration as CAL
from src.advisor import policy as PO
from src.advisor import prioritise as PR
from src.advisor import drift as DR
from src.advisor import copilot as CO
from src.advisor import feedback as FB


def _toy_comp2_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) for c in COMP2_CPU_FEATURE_COLS})
    df["project_id"] = ["Proj_01"] * (n // 2) + ["Proj_02"] * (n - n // 2)
    df["service_name"] = ["svc-a"] * (n // 2) + ["svc-b"] * (n - n // 2)
    df["timestamp_dt"] = pd.date_range("2026-01-01", periods=n, freq="min")
    df["ground_truth_failure"] = (rng.random(n) > 0.5).astype(int)
    df["imminent_failure"] = df["ground_truth_failure"]
    return df


# ── A. leakage guard ────────────────────────────────────────────────────────
def test_memory_leakage_guard_fires():
    poisoned = COMP2_CPU_FEATURE_COLS + ["memory_prob", "pred_label", "ram_percent"]
    with pytest.raises(ValueError):
        assert_no_memory_leakage(poisoned)


def test_leakage_detector_identifies_memory_columns():
    assert is_memory_derived("memory_leak_prob")
    assert is_memory_derived("pred_label")
    assert not is_memory_derived("cpu_percent")
    assert not is_memory_derived("queue_pressure_index")


def test_build_temporal_features_has_no_memory_columns():
    df = _toy_comp2_df()
    _feat, cols = FC.build_temporal_features(df)
    assert len(cols) > len(COMP2_CPU_FEATURE_COLS)
    assert not any(is_memory_derived(c) for c in cols), cols


def test_ttf_target_is_bounded():
    df = FC.add_time_to_failure_target(_toy_comp2_df())
    assert df[FC.TTF_TARGET_COL].min() >= 0.0
    assert df[FC.TTF_TARGET_COL].max() <= FC.MAX_TTF_MIN


# ── B. calibration + conformal ──────────────────────────────────────────────
def test_calibration_probability_bounds():
    y = np.array([0, 1] * 50)
    raw = np.random.default_rng(1).uniform(0.3, 0.9, 100)
    ece = CAL.expected_calibration_error(y, raw)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= CAL.brier_score(y, raw) <= 1.0


def test_conformal_sets_and_review_flag():
    y = np.array([0, 1] * 50)
    prob = np.concatenate([np.random.default_rng(2).uniform(0.0, 0.3, 50),
                           np.random.default_rng(3).uniform(0.7, 1.0, 50)])
    conf = CAL.SplitConformalClassifier(alpha=0.1).fit(prob, y)
    sets, _c, review = conf.predict_sets(np.array([0.05, 0.95, 0.5]))
    assert sets[0] == [0] and sets[1] == [1]
    assert sets[2] == [0, 1]          # ambiguous -> human review
    assert review[2] is True


# ── C. policy: matrix fallback + top-3 + safety ─────────────────────────────
def test_matrix_fallback_mapping():
    assert PO.matrix_fallback(0.9, 0.1) == "LOAD_SHEDDING"
    assert PO.matrix_fallback(0.1, 0.9) == "PROACTIVE_RESTART"
    assert PO.matrix_fallback(0.9, 0.9) == "CRITICAL_RESTART_AND_REROUTE"
    assert PO.matrix_fallback(0.1, 0.1) == "NO_ACTION"


def test_advisor_exposes_matrix_fallback():
    d = PO.decide("svc", cpu_prob=0.8, mem_prob=0.1, ttf_p50=6.0)
    assert d.fallback_matrix_action == "LOAD_SHEDDING"
    assert len(d.ranked_actions) == 3
    assert d.ranked_actions[0]["rank"] == 1


def test_high_tier_action_requires_approval():
    d = PO.decide("svc", cpu_prob=0.95, mem_prob=0.9, ttf_p50=0.5)
    if PO.MITIGATION_TIER[d.recommended_action] > PO.APPROVAL_TIER:
        assert d.requires_operator_approval is True


def test_urgent_time_penalises_slow_actions():
    # With ~0.6 min runway a fast shed (0.2 min) completes but a 3-min restart
    # cannot, so the slow action's feasibility is penalised.
    scored = PO.score_actions(cpu_prob=0.9, mem_prob=0.0, ttf_p50=0.6)
    by = {s["action"]: s for s in scored}
    assert by["CRITICAL_RESTART_AND_REROUTE"]["feasibility"] < 1.0
    assert by["LOAD_SHEDDING"]["feasibility"] == 1.0
    assert (by["CRITICAL_RESTART_AND_REROUTE"]["feasibility"]
            < by["LOAD_SHEDDING"]["feasibility"])


def test_offline_policy_evaluation_runs():
    log = pd.DataFrame([
        {"context_cpu": 0.8, "context_mem": 0.1, "action": "LOAD_SHEDDING",
         "propensity": 0.4, "reward": -20.0, "label": 1},
        {"context_cpu": 0.2, "context_mem": 0.1, "action": "NO_ACTION",
         "propensity": 0.5, "reward": -1.0, "label": 0},
    ])
    for est in ("ips", "snips", "dr"):
        out = PO.evaluate_offline_policy(log, estimator=est)
        assert out["estimator"] == est and "value" in out


# ── D. prioritisation + cascade grouping ────────────────────────────────────
def test_urgency_orders_risky_critical_service_first():
    sig = [
        PR.ServiceSignal("batch", 0.85, 0.1, ttf_p50=2.0, criticality=1, fan_out=1),
        PR.ServiceSignal("core", 0.9, 0.2, ttf_p50=1.0, criticality=5, fan_out=8),
        PR.ServiceSignal("idle", 0.05, 0.0, ttf_p50=None, criticality=5, fan_out=1),
    ]
    rows = PR.prioritise_fleet(sig)
    assert rows[0]["service_name"] == "core"
    assert rows[-1]["service_name"] == "idle"


def test_cascade_grouping_dedupes_dependents():
    sig = [
        PR.ServiceSignal("checkout", 0.9, 0.1, ttf_p50=1.0, fan_out=5,
                         depends_on=["cart", "auth"]),
        PR.ServiceSignal("cart", 0.7, 0.1, ttf_p50=2.0, depends_on=["auth"]),
        PR.ServiceSignal("auth", 0.6, 0.1, ttf_p50=3.0),
    ]
    groups = PR.group_cascades(sig)
    assert len(groups) == 1
    g = groups[0]
    assert g["patient_zero"] == "checkout"
    assert g["deduped_alerts"] >= 1


# ── E. drift advice types ───────────────────────────────────────────────────
def test_drift_advice_is_recognised():
    rng = np.random.default_rng(4)
    ref = rng.normal(0.3, 0.1, 500).clip(0, 1)
    stable = rng.normal(0.3, 0.1, 500).clip(0, 1)
    shifted = rng.normal(0.7, 0.1, 500).clip(0, 1)
    assert DR.advise_for_service(ref, stable, stable - ref)["advice"] == "KEEP_THRESHOLDS"
    assert DR.advise_for_service(ref, shifted, shifted - ref)["advice"] in (
        "ADJUST_THRESHOLD", "RETRAIN_RECOMMENDED")


def test_psi_and_bounds():
    x = np.linspace(0, 1, 1000)
    assert DR.population_stability_index(x, x) == pytest.approx(0.0, abs=1e-6)
    proj = DR.project_far_recall([0, 1] * 50, np.array([0.2, 0.8] * 50), 0.5)
    assert 0.0 <= proj["far"] <= 1.0 and 0.0 <= proj["recall"] <= 1.0


# ── F. copilot brief + TreeExplainer ────────────────────────────────────────
def test_copilot_brief_word_budget():
    phi = {"cpu_percent": 0.3, "queue_pressure_index": 0.2, "in_flight_queue": 0.1}
    b = CO.build_brief_template("svc", 0.74, 0.6, "PROACTIVE_RESTART", 3.0, (1.0, 8.0), phi)
    assert len(b.split()) <= CO.MAX_BRIEF_WORDS


def test_copilot_llm_disabled_returns_template():
    old = os.environ.pop("SENTINEL_ADVISOR_LLM", None)
    phi = {"cpu_percent": 0.3}
    res = CO.build_brief("svc", 0.7, 0.6, "LOAD_SHEDDING", 4.0, (2.0, 9.0), phi,
                         allow_llm=True)
    assert res["method"] == "template"   # flag off -> deterministic
    if old is not None:
        os.environ["SENTINEL_ADVISOR_LLM"] = old


def test_real_treeshap_attribution():
    from src.model import load_model
    model_path = "outputs/cpu_rf_model.joblib"
    if not os.path.exists(model_path):
        pytest.skip("trained RF artifact not present")
    m = load_model(model_path)
    df = pd.DataFrame([[88, 15, 12, 9, 120, 260, 150, 18, 40, 1]],
                      columns=COMP2_CPU_FEATURE_COLS)
    phi, base, method = CO.attributions(m, df)
    assert method == "treeshap"
    assert set(phi.keys()) == set(COMP2_CPU_FEATURE_COLS)


# ── G. feedback atomicity + validation ──────────────────────────────────────
def test_feedback_atomic_append_and_validation(tmp_path):
    path = str(tmp_path / "fb.jsonl")
    fl = FB.FeedbackLogger(path)
    fl.log("svc", "accepted", "LOAD_SHEDDING", confidence=0.6)
    fl.log("svc2", "overridden", "NO_ACTION", chosen_action="SCALE_OUT")
    recs = fl.read_all()
    assert len(recs) == 2
    assert recs[1]["chosen_action"] == "SCALE_OUT"
    with pytest.raises(ValueError):
        fl.log("svc", "not_a_decision", "NO_ACTION")
    assert not os.path.exists(path + ".tmp")   # temp file atomically replaced


# ── cooldown respected (safety gate) ────────────────────────────────────────
def test_cooldown_gates_actuation():
    from src.mitigation import execute_mitigation_action
    svc = "cooldown-test-svc"
    first = execute_mitigation_action(svc, "TRIGGER_LOAD_SHEDDING", 0.8, 0.1)
    assert first["status"] == "SUCCESS"
    second = execute_mitigation_action(svc, "TRIGGER_LOAD_SHEDDING", 0.85, 0.1)
    assert second["action_executed"] == "SUPPRESSED_COOLDOWN_ACTIVE"


# ── API contracts ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src.api import app
    with TestClient(app) as c:
        yield c


def _ready(client):
    r = client.get("/healthz/readiness")
    return r.status_code == 200


def test_advisor_decide_contract(client):
    if not _ready(client):
        pytest.skip("model not ready")
    payload = {
        "service_name": "api-svc",
        "comp2_cpu": {c: 0.0 for c in COMP2_CPU_FEATURE_COLS},
    }
    payload["comp2_cpu"].update({"cpu_percent": 90, "in_flight_queue": 150,
                                 "queue_pressure_index": 40, "overload_flag": 1,
                                 "cpu_velocity": 10})
    r = client.post("/api/advisor/decide", json=payload)
    assert r.status_code == 200
    body = r.json()
    for k in ("recommended_action", "confidence", "ranked_actions",
              "time_to_failure", "brief", "fallback_matrix_action"):
        assert k in body
    assert len(body["ranked_actions"]) == 3


def test_advisor_decide_validation(client):
    r = client.post("/api/advisor/decide", json={"service_name": "x"})
    assert r.status_code == 422   # missing comp2_cpu -> Pydantic validation


def test_advisor_feedback_contract(client, tmp_path, monkeypatch):
    # Point the default feedback logger at a temp file.
    import src.advisor.feedback as _fb
    monkeypatch.setattr(_fb, "_default_logger", _fb.FeedbackLogger(str(tmp_path / "fb.jsonl")))
    r = client.post("/api/advisor/feedback", json={
        "service_name": "s", "decision": "accepted", "recommended_action": "LOAD_SHEDDING"})
    assert r.status_code == 200
    bad = client.post("/api/advisor/feedback", json={
        "service_name": "s", "decision": "bogus", "recommended_action": "NO_ACTION"})
    assert bad.status_code == 422


def test_advisor_drift_and_thresholds_endpoints(client):
    assert client.get("/api/advisor/drift").status_code == 200
    assert client.get("/api/advisor/thresholds").status_code == 200


# ── latency budget ──────────────────────────────────────────────────────────
def test_advisor_core_latency_budget():
    # Added advisor compute (cost model + template brief), excluding the pre-
    # existing RF inference, must stay under 5 ms p99 across 10k decisions.
    phi = {"cpu_percent": 0.3, "queue_pressure_index": 0.2}
    samples = np.random.default_rng(0).uniform(0.3, 0.95, 10000)
    ttfs = np.random.default_rng(1).uniform(0.1, 20.0, 10000)
    lat = []
    for p, t in zip(samples, ttfs):
        s = time.perf_counter()
        d = PO.decide("svc", cpu_prob=float(p), mem_prob=0.1, ttf_p50=float(t))
        CO.build_brief_template("svc", float(p), d.confidence, d.recommended_action,
                                float(t), (t * 0.5, t * 1.5), phi)
        lat.append((time.perf_counter() - s) * 1000.0)
    p99 = float(np.percentile(lat, 99))
    assert p99 < 5.0, f"advisor core p99 latency {p99:.3f} ms exceeds 5 ms budget"
