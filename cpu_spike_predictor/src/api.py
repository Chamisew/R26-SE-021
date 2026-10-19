import time
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
import pandas as pd
import numpy as np
from src.model import (
    load_model, predict_probabilities,
    COMP2_CPU_FEATURE_COLS, TARGET_COL
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIVE_STATUS_FILE = "outputs/live_status.json"
LIVE_LOG_FILE    = "outputs/live_log.jsonl"
LIVE_LOG_MAX     = 500

def _cpu_telemetry_present(c2) -> bool:
    """True when at least one of the 10 Comp2 CPU features is non-zero.

    An all-zero feature row means Component 2 never reported telemetry for
    this service; the model then outputs a constant baseline probability
    (~0.149 → "15%"), which must never be presented as a measured risk.
    """
    return any(abs(float(getattr(c2, col, 0.0) or 0.0)) > 0.0 for col in COMP2_CPU_FEATURE_COLS)


# How long a CPU measurement recorded by the Component 2 loop stays quotable by a
# memory-only request. The loop polls every second, so anything older than this has
# stopped receiving Comp2 rows and must not be re-stamped as live.
CPU_CARRY_MAX_AGE_SEC = 30.0


def _read_live_status() -> dict:
    if not os.path.exists(LIVE_STATUS_FILE):
        return {}
    try:
        with open(LIVE_STATUS_FILE, "r") as f:
            status = json.load(f)
    except Exception:
        return {}
    return status if isinstance(status, dict) else {}


def _carried_cpu_measurement(service_name: str) -> dict | None:
    """Return the last genuinely measured CPU record for this service, if still fresh.

    Component 3 (MUP) posts memory-only payloads whose ``comp2_cpu`` block is empty,
    because its dataset has no CPU columns. Scoring that empty row produces the
    model's constant baseline, and persisting it wholesale erased the real CPU
    prediction the Component 2 loop had written seconds earlier. Reusing the stored
    measurement lets a memory update refresh only the memory half of the row.
    """
    existing = _read_live_status().get(service_name)
    if not isinstance(existing, dict) or not existing.get("cpu_telemetry_available"):
        return None

    cpu_prob = existing.get("cpu_failure_prob")
    if not isinstance(cpu_prob, (int, float)) or isinstance(cpu_prob, bool):
        return None

    measured_at = existing.get("prediction_timestamp")
    if not isinstance(measured_at, str):
        return None
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(measured_at)).total_seconds()
    except ValueError:
        return None
    if age > CPU_CARRY_MAX_AGE_SEC:
        return None

    return existing


def _write_live_status(service_name: str, result: dict) -> None:
    os.makedirs("outputs", exist_ok=True)
    status = _read_live_status()
    status[service_name] = result
    with open(LIVE_STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

def _append_live_log(entry: dict) -> None:
    os.makedirs("outputs", exist_ok=True)
    with open(LIVE_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    with open(LIVE_LOG_FILE, "r") as f:
        lines = f.readlines()
    if len(lines) > LIVE_LOG_MAX:
        with open(LIVE_LOG_FILE, "w") as f:
            f.writelines(lines[-LIVE_LOG_MAX:])


class Component2Payload(BaseModel):
    cpu_percent: float = Field(..., description="Component 2: Current CPU utilization %", ge=0, le=100)
    cpu_velocity: float = Field(0.0, description="Component 2: Rate of change of CPU utilization")
    cpu_trend_5min: float = Field(0.0, description="Component 2: CPU trend over 5 minutes")
    cpu_trend_10min: float = Field(0.0, description="Component 2: CPU trend over 10 minutes")
    in_flight_queue: float = Field(0.0, description="Component 2: Work items currently in queue")
    incoming_rate: float = Field(0.0, description="Component 2: Rate of new requests arriving")
    processing_rate: float = Field(0.0, description="Component 2: Rate requests are being processed")
    queue_growth_rate: float = Field(0.0, description="Component 2: Rate queue is growing")
    overload_flag: int = Field(0, description="Component 2: Overload indicator (0 or 1)", ge=0, le=1)
    queue_pressure_index: float = Field(0.0, description="Component 2: Queue pressure build-up measure")


class Component3Payload(BaseModel):
    memory_prob: float = Field(0.0, description="Component 3: Memory failure probability score", ge=0, le=1)
    alert: str = Field("FALSE", description="Component 3: Memory alert flag (TRUE/FALSE)")
    pred_label: str = Field("NORMAL", description="Component 3: Memory prediction label (NORMAL/WARNING/FAILURE)")


class TelemetryPayload(BaseModel):
    service_name: str = Field("srv-generic-backend", description="Microservice identifier")
    project_id: str = Field("Proj_01", description="Associated project identifier")
    comp2_cpu: Component2Payload = Field(..., description="Component 2: CPU, queue, system data")
    comp3_memory: Component3Payload = Field(default_factory=Component3Payload, description="Component 3: Memory prediction data") # type: ignore


class PredictionResponse(BaseModel):
    service_name: str
    project_id: str
    cpu_failure_prob: float
    cpu_alarm: bool
    memory_prob: float
    memory_alert: bool
    memory_pred_label: str
    joint_alarm: bool
    action_recommended: str
    warning_message: str
    rca_narrative: str = ""
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    shapley_phi_values: dict[str, float] = Field(default_factory=dict)
    # 'treeshap' when exact TreeExplainer values, 'approximate' for the
    # importance-weighted deviation path in src.xai (compat field name kept).
    attribution_method: str = "approximate"
    sre_runbook_steps: list[str] = Field(default_factory=list)
    incident_ticket_payload: dict = Field(default_factory=dict)
    mitigation_executed: dict = Field(default_factory=dict)
    latency_ms: float
    timestamp: str


class BatchTelemetryPayload(BaseModel):
    services: list[TelemetryPayload] = Field(..., description="List of telemetry payloads")


# ── Decision Advisor request/response models ───────────────────────────────
class AdvisorHistoryRow(BaseModel):
    """One prior Comp2 telemetry sample (used only when a caller supplies a
    short history so the temporal forecaster can build lag/rolling features)."""
    comp2_cpu: Component2Payload


class AdvisorDecisionRequest(BaseModel):
    service_name: str = Field("srv-generic-backend")
    project_id: str = Field("Proj_01")
    comp2_cpu: Component2Payload
    comp3_memory: Component3Payload = Field(default_factory=Component3Payload)
    history: list[AdvisorHistoryRow] = Field(
        default_factory=list,
        description="Optional recent Comp2 samples for true temporal TTF; empty -> "
                    "fast calibrated risk->TTF proxy on the hot path.")
    allow_llm_brief: bool = Field(False, description="Opt into the async LLM brief enrichment.")


class RankedAction(BaseModel):
    rank: int
    action: str
    expected_cost: float
    residual_failure_prob: float
    risk_reduced: float
    reason: str


class AdvisorDecisionResponse(BaseModel):
    service_name: str
    risk: float
    recommended_action: str
    expected_cost: float
    confidence: float
    confidence_band: list[float]
    requires_operator_approval: bool
    needs_human_review: bool
    fallback_matrix_action: str
    ranked_actions: list[RankedAction]
    time_to_failure: dict
    brief: str
    brief_method: str
    attribution_method: str = "approximate"
    rationale: str
    latency_ms: float
    timestamp: str


class AdvisorFeedbackRequest(BaseModel):
    service_name: str
    decision: str = Field(..., description="accepted | ignored | overridden")
    recommended_action: str
    chosen_action: str | None = None
    outcome: str | None = None
    confidence: float | None = None
    actor: str = "operator"


class ThresholdSuggestionRequest(BaseModel):
    service: str
    threshold: float = Field(..., ge=0.0, le=1.0)
    approved_by: str = Field(..., description="Human approver id (mandatory)")
    reason: str = ""


API_METRICS = {
    "total_predictions": 0,
    "total_alarms_triggered": 0,
    "total_mitigations_executed": 0
}


app = FastAPI(
    title="CPU Spike Prediction Engine - REST API",
    description="""Real-time predictive telemetry engine for microservice failure detection.

    **Component Data Ownership:**
    - Component 2: Sends CPU, queue, system metrics via comp2_cpu field
    - Component 3: Sends memory predictions via comp3_memory field
    - Both components submit to the same /predict endpoint
    """,
    version="2.0.0"
)

model = None
MODEL_PATH = "outputs/cpu_rf_model.joblib"


def _build_model_input_df(payload: TelemetryPayload) -> pd.DataFrame:
    """
    Builds the model input DataFrame from Component 2 CPU fields ONLY.
    Component 3 memory fields are NOT used as model features (leakage prevention).
    """
    c2 = payload.comp2_cpu
    row = {
        "cpu_percent": c2.cpu_percent,
        "cpu_velocity": c2.cpu_velocity,
        "cpu_trend_5min": c2.cpu_trend_5min,
        "cpu_trend_10min": c2.cpu_trend_10min,
        "in_flight_queue": c2.in_flight_queue,
        "incoming_rate": c2.incoming_rate,
        "processing_rate": c2.processing_rate,
        "queue_growth_rate": c2.queue_growth_rate,
        "queue_pressure_index": c2.queue_pressure_index,
        "overload_flag": c2.overload_flag,
    }
    return pd.DataFrame([row], columns=COMP2_CPU_FEATURE_COLS)


@app.on_event("startup")
def startup_event():
    global model
    logger.info("Initializing API and loading models...")

    if not os.path.exists(MODEL_PATH):
        logger.critical(
            f"Trained model NOT found at {MODEL_PATH}. "
            f"Run 'python -m src.cli train' first to produce the model artifact. "
            f"The API will NOT auto-train — model training is a separate pipeline step."
        )
        # model remains None; readiness check will return 503
        return

    try:
        model = load_model(MODEL_PATH)
        logger.info(f"API loaded trained model from {MODEL_PATH} "
                    f"({len(model.feature_importances_)} features, classes={list(model.classes_)}).")
    except Exception as e:
        logger.error(f"Failed to load trained model from {MODEL_PATH}: {e}")


@app.get("/health")
@app.get("/healthz/liveness")
def health_liveness():
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/healthz/readiness")
def health_readiness():
    if model is None:
        raise HTTPException(status_code=503, detail="Model container not ready")
    return {"status": "ready", "model_loaded": True, "timestamp": datetime.utcnow().isoformat()}


@app.get("/metrics")
def get_prometheus_metrics(format: str = "json"):
    if format == "prometheus" or format == "text":
        from fastapi.responses import PlainTextResponse
        prometheus_text = f"""# HELP cpu_predictor_total_predictions Total telemetry predictions served
# TYPE cpu_predictor_total_predictions counter
cpu_predictor_total_predictions {API_METRICS["total_predictions"]}

# HELP cpu_predictor_active_alarms Total failure alarms triggered
# TYPE cpu_predictor_active_alarms counter
cpu_predictor_active_alarms {API_METRICS["total_alarms_triggered"]}

# HELP cpu_predictor_mitigations_executed Total automated SRE self-healing actions executed
# TYPE cpu_predictor_mitigations_executed counter
cpu_predictor_mitigations_executed {API_METRICS["total_mitigations_executed"]}

# HELP cpu_predictor_model_loaded Model readiness state
# TYPE cpu_predictor_model_loaded gauge
cpu_predictor_model_loaded {1 if model is not None else 0}
"""
        return PlainTextResponse(prometheus_text, media_type="text/plain; version=0.0.4")

    return {
        "cpu_predictor_total_predictions": API_METRICS["total_predictions"],
        "cpu_predictor_active_alarms": API_METRICS["total_alarms_triggered"],
        "cpu_predictor_mitigations_executed": API_METRICS["total_mitigations_executed"],
        "cpu_predictor_model_loaded": 1 if model is not None else 0,
        "timestamp": datetime.utcnow().isoformat()
    }


def _make_decision(
    cpu_fail_prob: float, mem_prob: float, mem_alert: bool, mem_pred_label: str, service_name: str
) -> tuple[str, str]:
    """
    Decision Engine: Combines:
      - Component 2 CPU model prediction (cpu_fail_prob)
      - Component 3 Memory prediction signals (mem_prob, mem_alert, mem_pred_label)
    Memory signals are used for decision augmentation, not model training features.
    """
    cpu_alarm = cpu_fail_prob >= 0.6
    mem_alarm_triggered = mem_alert or (mem_prob >= 0.7) or (mem_pred_label in ("WARNING", "FAILURE"))

    if cpu_alarm and mem_alarm_triggered:
        action = "CRITICAL_RESTART_AND_TRAFFIC_REROUTE"
        warning = (f"CRITICAL: Service '{service_name}' faces imminent CPU+Memory failure. "
                   f"(CPU prob={cpu_fail_prob*100:.1f}%, Mem prob={mem_prob*100:.1f}%, Mem label={mem_pred_label}). "
                   f"Restart and reroute traffic immediately.")
    elif cpu_alarm:
        action = "TRIGGER_LOAD_SHEDDING"
        warning = (f"WARNING: Service '{service_name}' CPU spike predicted "
                   f"({cpu_fail_prob*100:.1f}% probability). Shed non-critical traffic.")
    elif mem_alarm_triggered:
        action = "TRIGGER_PROACTIVE_POD_RESTART"
        warning = (f"WARNING: Service '{service_name}' memory issue detected "
                   f"(prob={mem_prob*100:.1f}%, label={mem_pred_label}). Restart pod proactively.")
    else:
        action = "NO_ACTION"
        warning = f"OK: Service '{service_name}' metrics are within normal bounds."

    return action, warning


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TelemetryPayload):
    """
    **Single Service Prediction**

    Called separately by Component 2 and Component 3, or combined by an aggregator.

    **Component 2 provides** (comp2_cpu): cpu_percent, cpu_velocity, cpu_trend_5min,
      cpu_trend_10min, in_flight_queue, incoming_rate, processing_rate,
      queue_growth_rate, overload_flag, queue_pressure_index

    **Component 3 provides** (comp3_memory): memory_prob, alert, pred_label
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction model is not available.")

    start_time = time.perf_counter()

    cpu_telemetry_available = _cpu_telemetry_present(payload.comp2_cpu)
    carried = None if cpu_telemetry_available else _carried_cpu_measurement(payload.service_name)

    if carried is not None:
        cpu_fail_prob = float(carried["cpu_failure_prob"])
    else:
        input_df = _build_model_input_df(payload)
        try:
            cpu_fail_prob = float(predict_probabilities(model, input_df)[0])
        except Exception as e:
            logger.error(f"Prediction inference failure for {payload.service_name}: {e}")
            raise HTTPException(status_code=500, detail="Inference error occurred.")

    mem_prob = payload.comp3_memory.memory_prob
    mem_alert_str = str(payload.comp3_memory.alert).upper()
    mem_alert = mem_alert_str in ("TRUE", "1", "YES", "T")
    mem_pred_label = payload.comp3_memory.pred_label

    cpu_alarm = cpu_fail_prob >= 0.6
    mem_alarm_bool = mem_alert or (mem_prob >= 0.7) or (mem_pred_label in ("WARNING", "FAILURE"))
    # joint_alarm = True only when BOTH CPU AND Memory alarms fire simultaneously.
    # Previously: `cpu_alarm or (mem_prob >= 0.8)` was wrong — it made every CPU alarm
    # show as CRITICAL on the dashboard even when memory was healthy.
    joint_alarm = cpu_alarm and mem_alarm_bool

    action, warning = _make_decision(cpu_fail_prob, mem_prob, mem_alert, mem_pred_label, payload.service_name)

    if carried is not None:
        # Reuse the explanation that describes the carried CPU measurement: the
        # all-zero input row would attribute risk to features that never moved.
        rca_report = {
            "rca_narrative": carried.get("rca_narrative") or "",
            "feature_contributions": carried.get("feature_contributions") or {},
            "shapley_phi_values": carried.get("shapley_phi_values") or {},
            "attribution_method": carried.get("attribution_method") or "approximate",
            "sre_runbook_steps": carried.get("sre_runbook_steps") or [],
            "incident_ticket_payload": carried.get("incident_ticket_payload") or {},
        }
    else:
        from src.xai import generate_enterprise_rca_report
        rca_report = generate_enterprise_rca_report(
            service_name=payload.service_name,
            cpu_failure_prob=cpu_fail_prob,
            memory_leak_prob=mem_prob,
            input_df=input_df,
            model=model
        )

    from src.mitigation import execute_mitigation_action
    mitigation_res = execute_mitigation_action(
        service_name=payload.service_name,
        action=action,
        cpu_prob=cpu_fail_prob,
        mem_leak_prob=mem_prob
    )

    API_METRICS["total_predictions"] += 1
    if joint_alarm:
        API_METRICS["total_alarms_triggered"] += 1
    if mitigation_res.get("status") == "SUCCESS" and action != "NO_ACTION":
        API_METRICS["total_mitigations_executed"] += 1

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0

    logger.info(f"[{payload.service_name}] CPU_prob={cpu_fail_prob:.4f} "
                f"Mem_prob={mem_prob:.4f} Mem_label={mem_pred_label} "
                f"Action={action} | Latency={latency_ms:.2f}ms")

    result = PredictionResponse(
        service_name=payload.service_name,
        project_id=payload.project_id,
        cpu_failure_prob=cpu_fail_prob,
        cpu_alarm=cpu_alarm,
        memory_prob=mem_prob,
        memory_alert=mem_alarm_bool,
        memory_pred_label=mem_pred_label,
        joint_alarm=joint_alarm,
        action_recommended=action,
        warning_message=warning,
        rca_narrative=rca_report["rca_narrative"],
        feature_contributions=rca_report["feature_contributions"],
        shapley_phi_values=rca_report["shapley_phi_values"],
        attribution_method=rca_report.get("attribution_method", "approximate"),
        sre_runbook_steps=rca_report["sre_runbook_steps"],
        incident_ticket_payload=rca_report["incident_ticket_payload"],
        mitigation_executed=mitigation_res,
        latency_ms=latency_ms,
        timestamp=datetime.utcnow().isoformat()
    )

    status_record = result.model_dump()
    # Flag baseline (no-telemetry) predictions so the dashboard suppresses the
    # constant 14.9% model output instead of showing it as a live "15%" risk.
    status_record["cpu_telemetry_available"] = cpu_telemetry_available or carried is not None
    if carried is not None:
        # Carry the Component 2 provenance with the measurement, so the dashboard
        # can still see when the CPU half was actually sampled.
        for field in ("prediction_timestamp", "source_timestamp", "incident_id",
                      "cpu_pct_raw", "cpu_trend_raw", "state_machine"):
            if field in carried:
                status_record[field] = carried[field]
    _write_live_status(payload.service_name, status_record)
    _append_live_log(result.model_dump())

    return result


@app.post("/batch_predict", response_model=list[PredictionResponse])
def batch_predict(batch: BatchTelemetryPayload):
    """
    **Batch Prediction for Multiple Microservices.**

    Each payload includes Component 2 CPU data and Component 3 memory data.
    """
    results = []
    for service_payload in batch.services:
        result = predict(service_payload)
        results.append(result)
    return results


# ══════════════════════════════════════════════════════════════════════════
#  DECISION ADVISOR LAYER
#  Sits on top of the RF + dual-threshold engine. Recommends only; the gating
#  matrix remains the guaranteed fallback and the 180 s breaker still gates any
#  actuation. Heavy work (forecaster build, drift, LLM) is lazy / async.
# ══════════════════════════════════════════════════════════════════════════
_ADVISOR_TTF_PATH = "outputs/advisor_ttf_model.joblib"
_advisor_state = {"forecaster": None, "forecaster_loaded": False}


def _get_forecaster():
    """Lazily load a saved TTF forecaster artifact (never trains on the hot path)."""
    if _advisor_state["forecaster_loaded"]:
        return _advisor_state["forecaster"]
    _advisor_state["forecaster_loaded"] = True
    if os.path.exists(_ADVISOR_TTF_PATH):
        try:
            from src.advisor.forecast import TimeToFailureForecaster
            _advisor_state["forecaster"] = TimeToFailureForecaster.load(_ADVISOR_TTF_PATH)
            logger.info("Advisor loaded TTF forecaster from %s", _ADVISOR_TTF_PATH)
        except Exception as e:
            logger.warning("Advisor forecaster load failed: %s", e)
    return _advisor_state["forecaster"]


def _ttf_proxy_from_risk(risk: float) -> dict:
    """Fast, calibrated risk->lead-time proxy for the single-shot hot path.

    Higher failure risk implies a shorter expected runway. The rigorous temporal
    forecaster is used instead whenever the caller supplies a history buffer and
    a trained forecaster artifact is present.
    """
    from src.advisor.forecast import MAX_TTF_MIN
    p50 = float(np.clip((1.0 - risk) * MAX_TTF_MIN, 0.2, MAX_TTF_MIN))
    return {"p10": round(max(0.0, p50 * 0.45), 2), "p50": round(p50, 2),
            "p90": round(min(MAX_TTF_MIN, p50 * 1.9 + 1.5), 2), "method": "risk-proxy"}


def _ttf_from_history(req: AdvisorDecisionRequest) -> dict | None:
    fc = _get_forecaster()
    if fc is None or not req.history:
        return None
    try:
        from src.advisor.forecast import build_temporal_features
        rows = [h.comp2_cpu.model_dump() for h in req.history]
        rows.append(req.comp2_cpu.model_dump())
        hist_df = pd.DataFrame(rows)
        hist_df["project_id"] = req.project_id
        hist_df["service_name"] = req.service_name
        hist_df["timestamp_dt"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(hist_df), freq="min")
        feat_df, cols = build_temporal_features(hist_df)
        q = fc.predict_quantiles(feat_df)
        return {"p10": round(float(q["p10"][-1]), 2), "p50": round(float(q["p50"][-1]), 2),
                "p90": round(float(q["p90"][-1]), 2), "method": "temporal-gbm"}
    except Exception as e:
        logger.warning("Advisor temporal TTF failed, using proxy: %s", e)
        return None


@app.post("/api/advisor/decide", response_model=AdvisorDecisionResponse)
def advisor_decide(req: AdvisorDecisionRequest):
    """Real-time Decision Advisor: risk, confidence, time-to-failure, and the
    cost-optimal action with ranked alternatives + a plain-language brief."""
    start = time.perf_counter()
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction model is not available.")

    input_df = _build_model_input_df(req)
    cpu_prob = float(predict_probabilities(model, input_df)[0])
    mem_prob = req.comp3_memory.memory_prob

    from src.advisor.policy import decide
    from src.advisor import copilot as _copilot

    ttf = _ttf_from_history(req) or _ttf_proxy_from_risk(cpu_prob)
    needs_review = 0.45 <= cpu_prob <= 0.65  # gray zone -> flag for human review

    decision = decide(req.service_name, cpu_prob, mem_prob,
                      ttf_p50=ttf["p50"], needs_human_review=needs_review,
                      ttf_band=(ttf["p10"], ttf["p90"]))

    _phi, _base, _method = _copilot.attributions(model, input_df)
    brief_res = _copilot.build_brief(
        req.service_name, risk=cpu_prob, confidence=decision.confidence,
        action=decision.recommended_action, ttf_p50=ttf["p50"],
        ttf_band=(ttf["p10"], ttf["p90"]), phi=_phi,
        needs_human_review=needs_review, allow_llm=req.allow_llm_brief,
        attribution_method=_method)

    resp = AdvisorDecisionResponse(
        service_name=req.service_name,
        risk=round(cpu_prob, 4),
        recommended_action=decision.recommended_action,
        expected_cost=decision.expected_cost,
        confidence=decision.confidence,
        confidence_band=[round(max(0.0, decision.confidence - 0.15), 3),
                         round(min(1.0, decision.confidence + 0.15), 3)],
        requires_operator_approval=decision.requires_operator_approval,
        needs_human_review=needs_review,
        fallback_matrix_action=decision.fallback_matrix_action,
        ranked_actions=[RankedAction(**{k: r[k] for k in
                        ("rank", "action", "expected_cost", "residual_failure_prob",
                         "risk_reduced", "reason")}) for r in decision.ranked_actions],
        time_to_failure=ttf,
        brief=brief_res["brief"],
        brief_method=brief_res["method"],
        attribution_method=_method,
        rationale=decision.rationale,
        latency_ms=(time.perf_counter() - start) * 1000.0,
        timestamp=datetime.utcnow().isoformat(),
    )
    return resp


@app.get("/api/advisor/brief/{service}")
def advisor_brief(service: str):
    """Regenerate the decision brief for a service from the latest live status."""
    if not os.path.exists(LIVE_STATUS_FILE):
        raise HTTPException(status_code=404, detail="No live status available.")
    with open(LIVE_STATUS_FILE, "r") as f:
        status = json.load(f)
    rec = status.get(service)
    if not rec:
        raise HTTPException(status_code=404, detail=f"No advisor record for '{service}'.")
    from src.advisor import copilot as _copilot
    from src.advisor.policy import decide
    cpu_prob = float(rec.get("cpu_failure_prob", 0.0) or 0.0)
    mem_prob = float(rec.get("memory_prob", 0.0) or 0.0)
    decision = decide(service, cpu_prob, mem_prob, ttf_p50=_ttf_proxy_from_risk(cpu_prob)["p50"])
    ttf = _ttf_proxy_from_risk(cpu_prob)
    brief = _copilot.build_brief(service, cpu_prob, decision.confidence,
                                 decision.recommended_action, ttf["p50"],
                                 (ttf["p10"], ttf["p90"]), rec.get("shapley_phi_values", {}))
    return {"service": service, "brief": brief["brief"], "method": brief["method"],
            "recommended_action": decision.recommended_action, "time_to_failure": ttf}


@app.post("/api/advisor/feedback")
def advisor_feedback(fb: AdvisorFeedbackRequest):
    """Append an operator decision to the atomic feedback log."""
    from src.advisor.feedback import get_default_logger, VALID_DECISIONS
    if fb.decision not in VALID_DECISIONS:
        raise HTTPException(status_code=422, detail=f"decision must be one of {VALID_DECISIONS}")
    rec = get_default_logger().log(
        service_name=fb.service_name, decision=fb.decision,
        recommended_action=fb.recommended_action, chosen_action=fb.chosen_action,
        outcome=fb.outcome, confidence=fb.confidence, actor=fb.actor)
    return {"status": "recorded", "record": rec}


@app.get("/api/advisor/drift")
def advisor_drift(limit: int = 50):
    """Return the most recent per-service drift advisories (computed offline)."""
    report_path = "outputs/advisor_drift_report.csv"
    if not os.path.exists(report_path):
        return {"status": "not_computed", "services": [],
                "hint": "Run 'python -m src.advisor.drift_report' or the eval pipeline."}
    df = pd.read_csv(report_path)
    return {"status": "ok", "generated_from": report_path,
            "services": df.head(limit).to_dict(orient="records")}


@app.get("/api/advisor/thresholds")
def advisor_thresholds_get():
    registry_path = "outputs/advisor_drift_registry.json"
    if not os.path.exists(registry_path):
        return {"status": "empty", "entries": []}
    with open(registry_path) as f:
        reg = json.load(f)
    adj = [e for e in reg.get("entries", []) if e.get("advice") == "ADJUST_THRESHOLD"]
    return {"active_version": reg.get("active_version"), "adjust_suggestions": adj[-20:]}


@app.post("/api/advisor/thresholds")
def advisor_thresholds_post(ts: ThresholdSuggestionRequest):
    """Record a human-APPROVED threshold change proposal. Never applied silently."""
    from src.advisor.drift import RetrainRegistry
    reg = RetrainRegistry()
    entry = reg.record(ts.service, {"advice": "ADJUST_THRESHOLD",
                                    "detail": {"threshold": ts.threshold,
                                               "approved_by": ts.approved_by,
                                               "reason": ts.reason}}, approved=True)
    entry["applied"] = False  # application is a separate, operator-run step
    return {"status": "recorded_pending_apply", "entry": entry}


# ── Ask Sentinel Chat ─────────────────────────────────────────────────────
class AdvisorChatRequest(BaseModel):
    message: str = Field(..., max_length=500, description="Operator question")
    service: str = Field("", description="Optional service context")
    session_id: str = Field("default", description="Per-operator session id")
    stream: bool = Field(False, description="Enable streaming response")


@app.post("/api/advisor/chat")
def advisor_chat(req: AdvisorChatRequest):
    """Ask Sentinel: natural-language query over the Decision Advisor.

    Uses read-only tools only; never modifies models or executes actions.
    Falls back to deterministic intent router when LLM is unavailable.
    """
    from src.advisor.chat import handle_chat, handle_chat_stream
    if req.stream:
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            handle_chat_stream(req.message, req.service, req.session_id),
            media_type="text/plain")
    resp = handle_chat(req.message, req.service, req.session_id)
    return {"answer": resp.answer, "tools_called": resp.tools_called,
            "method": resp.method, "proposal": resp.proposal,
            "as_of": resp.as_of, "session_id": resp.session_id}


@app.get("/api/advisor/chat/suggestions")
def advisor_chat_suggestions():
    """Return suggested question chips for the chat UI."""
    from src.advisor.chat import SUGGESTED_QUESTIONS
    return {"suggestions": SUGGESTED_QUESTIONS}


# ══════════════════════════════════════════════════════════════════════════
#  PER-SERVICE EXPLAINABILITY & MITIGATION (Phase 1) — READ-ONLY endpoints.
#  None of these may call execute_mitigation_action or mutate model/policy
#  state; they only surface what the prediction/mitigation engines recorded.
# ══════════════════════════════════════════════════════════════════════════

UNAVAILABLE = "unavailable"

# Ladder mirrors src.mitigation exactly: same commands, shown as PREVIEW only.
MITIGATION_TIERS = [
    {"tier": 1, "label": "Tier 1 — Envoy load shedding",
     "detail": "Rate-limit ingress, drop 25% of non-critical traffic (reversible)",
     "command": "envoy-cli rate-limit set --service={svc} --drop-ratio=0.25"},
    {"tier": 2, "label": "Tier 2 — Scale out / proactive replacement",
     "detail": "kubectl scale +2 replicas, or gracefully replace the pod",
     "command": "kubectl scale deployment/{svc} --replicas=+2"},
    {"tier": 3, "label": "Tier 3 — Drain & rolling restart",
     "detail": "Isolate traffic, drain pod, rolling restart deployment",
     "command": "kubectl rollout restart deployment/{svc} && kubectl label pod -l app={svc} status=drain"},
]


def _normalize_circuit_breaker(raw: str | None, tier: int, cooldown_remaining: int) -> str:
    """Collapse src.mitigation's display strings to the three canonical states."""
    s = (raw or "").upper()
    if s.startswith("STANDBY") or tier <= 0:
        return "STANDBY"
    if s.startswith("CLOSED") or (tier > 0 and cooldown_remaining > 0):
        return "CLOSED"
    if "ACTIVE" in s:
        return "ACTIVE"
    return "STANDBY"


def _tier_from_label(raw: str | None) -> int:
    s = raw or ""
    if "TIER_" in s:
        digit = s.split("TIER_", 1)[1][:1]
        if digit.isdigit():
            return int(digit)
    return 0


class CauseFactor(BaseModel):
    feature: str
    display_name: str
    contribution_pct: float | None = None
    phi: float | None = None


class ServiceExplanationResponse(BaseModel):
    service_name: str
    data_available: bool
    reason: str = ""
    attribution_method: str = "approximate"
    severity: str = UNAVAILABLE
    impact_radius: str = UNAVAILABLE
    impact_radius_note: str = ""
    impact_radius_heuristic: str = UNAVAILABLE
    primary_cause: CauseFactor | None = None
    secondary_cause: CauseFactor | None = None
    rca_narrative: str = ""
    feature_contributions: dict[str, float] = Field(default_factory=dict)
    shapley_phi_values: dict[str, float] = Field(default_factory=dict)
    runbook_id: str = "RB-04"
    runbook_checklist: list[str] = Field(default_factory=list)
    runbook_steps: list[str] = Field(default_factory=list)
    incident_ticket_payload: dict = Field(default_factory=dict)
    pagerduty_payload: dict = Field(default_factory=dict)
    servicenow_payload: dict = Field(default_factory=dict)
    source_timestamp: str = ""
    generated_at: str = ""


class ServiceMitigationResponse(BaseModel):
    service_name: str
    data_available: bool
    reason: str = ""
    status: str = UNAVAILABLE
    active_tier: int = 0
    tiers: list[dict] = Field(default_factory=lambda: list(MITIGATION_TIERS))
    circuit_breaker: str = "STANDBY"
    cooldown_remaining_sec: int = 0
    cooldown_total_sec: int = 180
    command_preview: str | None = None
    simulated: bool = True
    source: str = "none"
    generated_at: str = ""


class MitigationHistoryResponse(BaseModel):
    service_name: str
    entries: list[dict] = Field(default_factory=list)
    count: int = 0
    note: str = ""


@app.get("/api/services/{service}/explanation", response_model=ServiceExplanationResponse)
def service_explanation(service: str):
    """Read-only RCA explanation for one service, assembled from the latest
    live-status record. Missing data is rendered as 'unavailable', never zeros."""
    from src.advisor.copilot import RUNBOOK_BY_ACTION, RUNBOOK_CHECKLISTS
    from src.xai import FEATURE_DISPLAY_NAMES

    rec = _read_live_status().get(service)
    if not isinstance(rec, dict):
        raise HTTPException(status_code=404, detail=f"No telemetry record for '{service}'.")

    contributions = rec.get("feature_contributions") or {}
    phi = rec.get("shapley_phi_values") or {}
    telemetry_ok = bool(rec.get("cpu_telemetry_available", True))
    # A flat contribution map (every feature identical) is the placeholder the
    # model emits for all-zero telemetry rows — never present it as an analysis.
    placeholder = bool(contributions) and len({round(float(v), 3) for v in contributions.values()}) <= 1
    data_available = telemetry_ok and not placeholder
    reason = "" if data_available else (
        "no CPU telemetry measured for this service" if not telemetry_ok
        else "feature attribution is placeholder data (uniform contributions)")

    method = rec.get("attribution_method") or "approximate"

    sorted_feats = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True) if data_available else []

    def _cause(idx) -> CauseFactor | None:
        if len(sorted_feats) <= idx:
            return None
        feat, pct = sorted_feats[idx]
        return CauseFactor(feature=feat,
                           display_name=FEATURE_DISPLAY_NAMES.get(feat, feat),
                           contribution_pct=round(float(pct), 1),
                           phi=round(float(phi.get(feat, 0.0)), 4))

    ticket = rec.get("incident_ticket_payload") or {}
    severity = ticket.get("severity") or rec.get("state_machine") or UNAVAILABLE
    heuristic_radius = ticket.get("impact_radius") or UNAVAILABLE

    advisor = rec.get("advisor") or {}
    action = advisor.get("recommended_action") or rec.get("action_recommended") or "NO_ACTION"
    # Gating-matrix names differ from advisor action names; unify them.
    action = {"TRIGGER_LOAD_SHEDDING": "LOAD_SHEDDING",
              "TRIGGER_PROACTIVE_POD_RESTART": "PROACTIVE_RESTART",
              "CRITICAL_RESTART_AND_TRAFFIC_REROUTE": "CRITICAL_RESTART_AND_REROUTE"}.get(action, action)
    runbook_id = RUNBOOK_BY_ACTION.get(action, "RB-04")

    sev_map = {"CRITICAL": "critical", "WARNING": "warning", "NOMINAL": "info",
               "HEALTHY": "info"}
    pd_severity = sev_map.get(str(severity).upper(), "info")
    pagerduty_payload = {
        "event_action": "trigger",
        "dedup_key": ticket.get("ticket_id", ""),
        "payload": {
            "summary": f"[{severity}] {service}: CPU failure risk "
                       f"{rec.get('cpu_failure_prob', UNAVAILABLE)}",
            "severity": pd_severity,
            "source": service,
            "timestamp": rec.get("prediction_timestamp", ""),
            "custom_details": {k: v for k, v in ticket.items() if k != "recommended_runbook_steps"},
        },
        "note": "SIMULATED payload preview — not sent to PagerDuty",
    }
    servicenow_payload = {
        "short_description": f"[{severity}] {service} — predicted CPU spike (Sentinel AIOps)",
        "cmdb_ci": service,
        "urgency": {"critical": "H1", "warning": "H2", "info": "H3"}.get(pd_severity, "H3"),
        "severity": {"critical": "S1", "warning": "S2", "info": "S3"}.get(pd_severity, "S3"),
        "contact_type": "monitoring",
        "description": ticket.get("summary", ""),
        "worknotes": " | ".join(ticket.get("recommended_runbook_steps", [])[:3]),
        "note": "SIMULATED payload preview — not filed in ServiceNow",
    }

    return ServiceExplanationResponse(
        service_name=service,
        data_available=data_available,
        reason=reason,
        attribution_method=method,
        severity=severity,
        impact_radius=UNAVAILABLE,
        impact_radius_note="No real topology/dependency source exists in this deployment; "
                           "impact radius cannot be derived. Not shown rather than invented.",
        impact_radius_heuristic=heuristic_radius,
        primary_cause=_cause(0),
        secondary_cause=_cause(1),
        rca_narrative=rec.get("rca_narrative") or "",
        feature_contributions=contributions if data_available else {},
        shapley_phi_values=phi if data_available else {},
        runbook_id=runbook_id,
        runbook_checklist=RUNBOOK_CHECKLISTS.get(runbook_id, []),
        runbook_steps=rec.get("sre_runbook_steps") or [],
        incident_ticket_payload=ticket,
        pagerduty_payload=pagerduty_payload,
        servicenow_payload=servicenow_payload,
        source_timestamp=rec.get("prediction_timestamp", ""),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/services/{service}/mitigation", response_model=ServiceMitigationResponse)
def service_mitigation(service: str):
    """Read-only view of the REAL mitigation state (tier, cooldown, circuit
    breaker). This endpoint never executes or mutates anything."""
    import time as _time
    from src.mitigation import _SERVICE_STATE, COOLDOWN_SECONDS

    now = _time.time()
    state = _SERVICE_STATE.get(service)
    if isinstance(state, dict):
        tier = int(state.get("tier", 0))
        elapsed = now - float(state.get("last_action_ts", 0.0) or 0.0)
        remaining = max(0, int(COOLDOWN_SECONDS - elapsed)) if state.get("last_action_ts") else 0
        breaker = _normalize_circuit_breaker(None, tier, remaining)
        status = "HEALTHY" if tier == 0 else ("COOLDOWN_ACTIVE" if remaining > 0 else "ACTIVE")
        source = "process_state"
        reason = ""
        data_available = True
    else:
        rec = _read_live_status().get(service) or {}
        snap = rec.get("mitigation_executed") or {}
        if snap:
            tier = _tier_from_label(snap.get("mitigation_tier"))
            remaining = int(snap.get("cooldown_remaining_sec") or 0)
            breaker = _normalize_circuit_breaker(snap.get("circuit_breaker"), tier, remaining)
            status = snap.get("status") or UNAVAILABLE
            source = "status_snapshot"
            reason = "API process has no live state for this service; showing last persisted snapshot"
            data_available = True
        else:
            tier, remaining, breaker, status = 0, 0, "STANDBY", UNAVAILABLE
            source, reason, data_available = "none", "no mitigation record for this service", False

    preview = None
    if tier >= 1:
        preview = MITIGATION_TIERS[min(tier, 3) - 1]["command"].format(svc=service)
        if tier == 2:
            advisor_action = (((_read_live_status().get(service) or {}).get("advisor")
                               ) or {}).get("recommended_action")
            if advisor_action == "PROACTIVE_RESTART":
                preview = f"kubectl delete pod -l app={service} --grace-period=30"

    return ServiceMitigationResponse(
        service_name=service,
        data_available=data_available,
        reason=reason,
        status=status,
        active_tier=tier,
        tiers=list(MITIGATION_TIERS),
        circuit_breaker=breaker,
        cooldown_remaining_sec=remaining,
        cooldown_total_sec=COOLDOWN_SECONDS,
        command_preview=preview,
        simulated=True,
        source=source,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/services/{service}/mitigation/history", response_model=MitigationHistoryResponse)
def service_mitigation_history(service: str):
    """Read-only mitigation history from src.mitigation.get_mitigation_history
    (in-memory, bounded to the last 20 escalations per service)."""
    from src.mitigation import get_mitigation_history
    entries = list(get_mitigation_history(service) or [])
    note = "" if entries else ("History lives in the prediction API process memory; "
                               "it is empty until an escalation occurs (and resets on restart).")
    return MitigationHistoryResponse(service_name=service, entries=entries,
                                     count=len(entries), note=note)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)