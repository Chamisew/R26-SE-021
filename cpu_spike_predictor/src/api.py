import time
import json
from datetime import datetime
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

def _write_live_status(service_name: str, result: dict) -> None:
    os.makedirs("outputs", exist_ok=True)
    status = {}
    if os.path.exists(LIVE_STATUS_FILE):
        try:
            with open(LIVE_STATUS_FILE, "r") as f:
                status = json.load(f)
        except Exception:
            status = {}
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
    sre_runbook_steps: list[str] = Field(default_factory=list)
    incident_ticket_payload: dict = Field(default_factory=dict)
    mitigation_executed: dict = Field(default_factory=dict)
    latency_ms: float
    timestamp: str


class BatchTelemetryPayload(BaseModel):
    services: list[TelemetryPayload] = Field(..., description="List of telemetry payloads")


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
        sre_runbook_steps=rca_report["sre_runbook_steps"],
        incident_ticket_payload=rca_report["incident_ticket_payload"],
        mitigation_executed=mitigation_res,
        latency_ms=latency_ms,
        timestamp=datetime.utcnow().isoformat()
    )

    _write_live_status(payload.service_name, result.model_dump())
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
