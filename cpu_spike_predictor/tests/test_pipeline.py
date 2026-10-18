import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from src.ingestion import load_cpu_features, load_memory_predictions, merge_datasets, add_lead_time_labels
from src.model import train_model, predict_probabilities, save_model, load_model, get_features_and_target, COMP2_CPU_FEATURE_COLS, TARGET_COL
from src.validation import run_lopo_cross_validation
from src.mtta import compute_mtta_for_alarm_type, analyze_mtta_strategies
from src.api import app

client = TestClient(app)

@pytest.fixture
def sample_cpu_data(tmp_path):
    file_path = tmp_path / "final_research_dataset.csv"
    start_time = datetime(2026, 6, 9, 12, 0)

    rows = []
    for proj in ["P1", "P2"]:
        for t in range(30):
            is_fail = 1.0 if (proj == "P1" and t >= 25) else 0.0
            cpu_val = 20.0 + (t * 2 if proj == "P1" else 5.0)
            cpu_vel = 2.0 if (proj == "P1" and t >= 1) else 0.0
            trend_5 = (t * 2.0) if proj == "P1" and t >= 5 else 0.0
            trend_10 = (t * 1.5) if proj == "P1" and t >= 10 else 0.0
            inc_rate = 100.0 + cpu_val * 2.0
            proc_rate = 100.0 + cpu_val * 1.8
            qgrowth = max(0.0, inc_rate - proc_rate)
            inflight = cpu_val * 0.5
            overload = 1 if cpu_val > 80.0 else 0
            qpressure = round((inflight * qgrowth) / 100.0, 2)
            rows.append({
                "incident_id": 1 if is_fail else 0,
                "timestamp": (start_time + timedelta(minutes=t)).timestamp() + (0 if proj == "P1" else 0.001),
                "time": (start_time + timedelta(minutes=t)).strftime("%H:%M:%S"),
                "system_state": "SPIKE" if is_fail else ("BUILDUP" if proj == "P1" and t >= 20 else "NORMAL"),
                "incident_phase": "SPIKE" if is_fail else ("BUILDUP" if proj == "P1" and t >= 20 else "NONE"),
                "failing_service": f"srv-{proj}" if is_fail else np.nan,
                "patient_zero": f"srv-{proj}" if is_fail else np.nan,
                "project_id": f"Proj_{proj}",
                "service_name": f"srv-{proj}",
                "cpu_percent": cpu_val,
                "cpu_velocity": cpu_vel,
                "cpu_trend_5min": trend_5,
                "cpu_trend_10min": trend_10,
                "in_flight_queue": inflight,
                "incoming_rate": inc_rate,
                "processing_rate": proc_rate,
                "queue_growth_rate": qgrowth,
                "overload_flag": overload,
                "queue_pressure_index": qpressure,
                "incident_duration": (t - 20) if (proj == "P1" and t >= 20) else 0.0,
                "label": int(is_fail),
            })

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_mem_predictions(tmp_path):
    file_path = tmp_path / "memory_predictions.csv"
    start_time = datetime(2026, 6, 9, 12, 0)

    rows = []
    for proj in ["P1", "P2"]:
        for t in range(30):
            if proj == "P1" and t >= 20:
                ratio = (t - 20) / max(1, 10)
                mem_prob = min(1.0, 0.4 + 0.58 * ratio)
                if mem_prob >= 0.9:
                    alert, pred = "TRUE", "FAILURE"
                elif mem_prob >= 0.7:
                    alert, pred = "TRUE", "WARNING"
                else:
                    alert, pred = "FALSE", "NORMAL"
            else:
                mem_prob = 0.05
                alert, pred = "FALSE", "NORMAL"
            td = start_time + timedelta(minutes=t)
            rows.append({
                "timestamp": f"{td.minute:02d}:{td.second:04.1f}",
                "service_name": f"srv-{proj}",
                "project_id": f"Proj_{proj}",
                "memory_prob": round(mem_prob, 4),
                "alert": alert,
                "pred_label": pred,
            })

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_ingestion_and_lead_time_labeling(sample_cpu_data, sample_mem_predictions):
    cpu_df = load_cpu_features(sample_cpu_data, lead_time_minutes=5)
    assert len(cpu_df) == 60
    assert "imminent_failure" in cpu_df.columns
    assert "cpu_percent" in cpu_df.columns
    for col in COMP2_CPU_FEATURE_COLS:
        assert col in cpu_df.columns, f"Comp2 CPU feature '{col}' missing from loaded df"

    p1_df = cpu_df[cpu_df["project_id"] == "Proj_P1"].sort_values("timestamp").reset_index(drop=True)
    assert p1_df.loc[24, "imminent_failure"] == 1
    assert p1_df.loc[20, "imminent_failure"] == 1   # 5 min ahead of failure at t=25
    assert p1_df.loc[19, "imminent_failure"] == 0   # 6 min ahead — outside lead-time window
    assert p1_df.loc[0, "imminent_failure"] == 0

    mem_df = load_memory_predictions(sample_mem_predictions)
    assert len(mem_df) == 60
    assert "memory_prob" in mem_df.columns
    assert "alert" in mem_df.columns
    assert "pred_label" in mem_df.columns
    assert "project_id" in mem_df.columns
    assert "service_name" in mem_df.columns

    merged_df = merge_datasets(cpu_df, mem_df)
    assert len(merged_df) == 60
    assert "memory_prob" in merged_df.columns
    assert "memory_leak_prob" in merged_df.columns
    assert "pred_label" in merged_df.columns
    assert "alert" in merged_df.columns
    for col in COMP2_CPU_FEATURE_COLS:
        assert col in merged_df.columns


def test_model_training_and_saving(sample_cpu_data, sample_mem_predictions, tmp_path):
    cpu_df = load_cpu_features(sample_cpu_data, lead_time_minutes=5)
    mem_df = load_memory_predictions(sample_mem_predictions)
    merged_df = merge_datasets(cpu_df, mem_df)

    X, y = get_features_and_target(merged_df, TARGET_COL)
    assert list(X.columns) == COMP2_CPU_FEATURE_COLS
    assert X.shape[1] == len(COMP2_CPU_FEATURE_COLS)

    model = train_model(X, y)
    assert model is not None

    probs = predict_probabilities(model, X)
    assert len(probs) == len(X)
    assert all(0.0 <= p <= 1.0 for p in probs)

    model_file = str(tmp_path / "model.joblib")
    save_model(model, model_file)
    assert os.path.exists(model_file)

    loaded_model = load_model(model_file)
    probs_loaded = predict_probabilities(loaded_model, X)
    np.testing.assert_array_almost_equal(probs, probs_loaded)


def test_lopo_cross_validation(sample_cpu_data, sample_mem_predictions):
    cpu_df = load_cpu_features(sample_cpu_data, lead_time_minutes=5)
    mem_df = load_memory_predictions(sample_mem_predictions)
    merged_df = merge_datasets(cpu_df, mem_df)

    lopo_results, all_preds = run_lopo_cross_validation(merged_df)
    assert len(lopo_results) == 2
    assert "f1_score" in lopo_results.columns
    assert "cpu_failure_prob" in all_preds.columns
    assert "memory_leak_prob" in all_preds.columns or "memory_prob" in all_preds.columns


def test_mtta_calculation():
    start_time = datetime(2026, 6, 9, 12, 0)
    rows = []
    for t in range(30):
        ts = start_time + timedelta(minutes=t)
        rows.append({
            "timestamp": ts,
            "project_id": "Proj_01",
            "service_name": "srv-proj_01",
            "ground_truth_failure": 1 if t >= 25 else 0,
            "cpu_failure_prob": 0.8 if t >= 20 else 0.1,
            "memory_prob": 0.05,
            "memory_leak_prob": 0.05,
        })

    df = pd.DataFrame(rows)
    df["cpu_alarm"] = (df["cpu_failure_prob"] >= 0.6).astype(int)

    mtta_df = compute_mtta_for_alarm_type(df, "cpu_alarm", window_minutes=20)
    assert len(mtta_df) == 1
    assert mtta_df.loc[0, "anticipated"] == 1
    assert mtta_df.loc[0, "mtta_minutes"] == 5.0
    assert mtta_df.loc[0, "mtta_seconds"] == 300.0


def test_fastapi_endpoints():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert "status" in health.json()

        low_payload = {
            "service_name": "srv-test-low",
            "project_id": "Proj_01",
            "comp2_cpu": {
                "cpu_percent": 20.0,
                "cpu_velocity": 0.5,
                "cpu_trend_5min": 1.0,
                "cpu_trend_10min": 2.0,
                "in_flight_queue": 10.0,
                "incoming_rate": 150.0,
                "processing_rate": 148.0,
                "queue_growth_rate": 2.0,
                "overload_flag": 0,
                "queue_pressure_index": 0.2,
            },
            "comp3_memory": {
                "memory_prob": 0.1,
                "alert": "FALSE",
                "pred_label": "NORMAL",
            },
        }
        r_low = client.post("/predict", json=low_payload)
        assert r_low.status_code == 200, f"status={r_low.status_code}, detail={r_low.json()}"
        d_low = r_low.json()
        assert "cpu_failure_prob" in d_low
        assert d_low["cpu_alarm"] is False
        assert d_low["action_recommended"] == "NO_ACTION"
        assert d_low["latency_ms"] > 0
        assert "memory_prob" in d_low
        assert "memory_pred_label" in d_low

        high_payload = {
            "service_name": "srv-test-high",
            "project_id": "Proj_02",
            "comp2_cpu": {
                "cpu_percent": 95.0,
                "cpu_velocity": 8.0,
                "cpu_trend_5min": 25.0,
                "cpu_trend_10min": 40.0,
                "in_flight_queue": 500.0,
                "incoming_rate": 300.0,
                "processing_rate": 220.0,
                "queue_growth_rate": 80.0,
                "overload_flag": 1,
                "queue_pressure_index": 400.0,
            },
            "comp3_memory": {
                "memory_prob": 0.85,
                "alert": "TRUE",
                "pred_label": "FAILURE",
            },
        }
        r_high = client.post("/predict", json=high_payload)
        assert r_high.status_code == 200
        d_high = r_high.json()
        assert "cpu_failure_prob" in d_high
        assert "rca_narrative" in d_high
        assert len(d_high["sre_runbook_steps"]) >= 1
        assert isinstance(d_high["feature_contributions"], dict)
        assert len(d_high["feature_contributions"]) > 0
