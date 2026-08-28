import pytest
import pandas as pd
import numpy as np
from src.model import load_model, COMP2_CPU_FEATURE_COLS
from src.xai import compute_shapley_values, generate_enterprise_rca_report

def test_shapley_values_computation():
    # Construct a sample single-row DataFrame with elevated CPU features
    data = {
        "cpu_percent": [95.0],
        "cpu_velocity": [12.0],
        "cpu_trend_5min": [30.0],
        "cpu_trend_10min": [45.0],
        "in_flight_queue": [400.0],
        "incoming_rate": [350.0],
        "processing_rate": [200.0],
        "queue_growth_rate": [150.0],
        "overload_flag": [1],
        "queue_pressure_index": [500.0],
    }
    df = pd.DataFrame(data)
    model = load_model("outputs/cpu_rf_model.joblib")
    
    base_val, phi_dict, attributions = compute_shapley_values(model, df, COMP2_CPU_FEATURE_COLS)
    
    assert isinstance(base_val, float)
    assert len(phi_dict) == len(COMP2_CPU_FEATURE_COLS)
    assert len(attributions) == len(COMP2_CPU_FEATURE_COLS)
    # Attribution percentages should sum approximately to 100% (allowing for 1% float rounding)
    assert abs(sum(attributions.values()) - 100.0) < 1.0

def test_enterprise_rca_report_generation():
    data = {
        "cpu_percent": [92.0],
        "cpu_velocity": [8.0],
        "cpu_trend_5min": [25.0],
        "cpu_trend_10min": [40.0],
        "in_flight_queue": [300.0],
        "incoming_rate": [280.0],
        "processing_rate": [180.0],
        "queue_growth_rate": [100.0],
        "overload_flag": [1],
        "queue_pressure_index": [350.0],
    }
    df = pd.DataFrame(data)
    model = load_model("outputs/cpu_rf_model.joblib")

    rca_report = generate_enterprise_rca_report(
        model=model,
        input_df=df,
        service_name="payment-gateway",
        cpu_failure_prob=0.88,
        memory_leak_prob=0.20,
        action_recommended="TRIGGER_LOAD_SHEDDING",
        feature_cols=COMP2_CPU_FEATURE_COLS
    )

    assert rca_report["service_name"] == "payment-gateway"
    assert rca_report["severity"] in ["CRITICAL", "WARNING"]
    assert "rca_narrative" in rca_report
    assert len(rca_report["sre_runbook_steps"]) > 0
    
    ticket = rca_report["incident_ticket_payload"]
    assert ticket["service_name"] == "payment-gateway"
    assert "ticket_id" in ticket
    assert len(ticket["top_3_cpu_features_by_importance"]) == 3
