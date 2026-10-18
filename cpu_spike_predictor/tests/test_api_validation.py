import pytest
from fastapi.testclient import TestClient
from src.api import app

def test_api_missing_fields_validation():
    with TestClient(app) as client:
        # 1. Payload with missing comp2_cpu
        incomplete_payload = {
            "service_name": "srv-incomplete",
            "project_id": "Proj_01"
        }
        res = client.post("/predict", json=incomplete_payload)
        # Should return 422 Unprocessable Entity
        assert res.status_code == 422

def test_api_batch_predict():
    with TestClient(app) as client:
        batch_payload = {
            "services": [
                {
                    "service_name": "srv-1",
                    "project_id": "Proj_01",
                    "comp2_cpu": {
                        "cpu_percent": 25.0,
                        "cpu_velocity": 0.0,
                        "cpu_trend_5min": 0.0,
                        "cpu_trend_10min": 0.0,
                        "in_flight_queue": 10.0,
                        "incoming_rate": 150.0,
                        "processing_rate": 150.0,
                        "queue_growth_rate": 0.0,
                        "overload_flag": 0,
                        "queue_pressure_index": 1.0,
                    },
                    "comp3_memory": {
                        "memory_prob": 0.05,
                        "alert": "FALSE",
                        "pred_label": "NORMAL",
                    }
                },
                {
                    "service_name": "srv-2",
                    "project_id": "Proj_01",
                    "comp2_cpu": {
                        "cpu_percent": 95.0,
                        "cpu_velocity": 10.0,
                        "cpu_trend_5min": 30.0,
                        "cpu_trend_10min": 40.0,
                        "in_flight_queue": 300.0,
                        "incoming_rate": 250.0,
                        "processing_rate": 120.0,
                        "queue_growth_rate": 130.0,
                        "overload_flag": 1,
                        "queue_pressure_index": 450.0,
                    },
                    "comp3_memory": {
                        "memory_prob": 0.9,
                        "alert": "TRUE",
                        "pred_label": "FAILURE",
                    }
                }
            ]
        }
        res = client.post("/batch_predict", json=batch_payload)
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["action_recommended"] == "NO_ACTION"
        assert data[1]["action_recommended"] in ["CRITICAL_RESTART_AND_TRAFFIC_REROUTE", "TRIGGER_LOAD_SHEDDING", "TRIGGER_PROACTIVE_POD_RESTART"]
