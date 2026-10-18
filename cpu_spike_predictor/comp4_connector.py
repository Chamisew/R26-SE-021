"""
comp4_connector.py  — Drop-in connector for Component 2 and Component 3 teams
===========================================================================

HOW TO USE:
-----------
Component 2 team (CPU/Queue/System data owner):
    from comp4_connector import send_comp2_cpu_data
    send_comp2_cpu_data(service_name="srv-payment-backend", project_id="Proj_01",
                         cpu_percent=87.5, cpu_velocity=5.2, cpu_trend_5min=12.3,
                         cpu_trend_10min=20.1, in_flight_queue=150.0, incoming_rate=250.0,
                         processing_rate=230.0, queue_growth_rate=20.0, overload_flag=0,
                         queue_pressure_index=3.0)

Component 3 team (Memory data owner):
    from comp4_connector import send_comp3_memory_data
    send_comp3_memory_data(service_name="srv-payment-backend", project_id="Proj_01",
                            memory_prob=0.95, alert="TRUE", pred_label="FAILURE")

Combined send (if aggregator sends both):
    from comp4_connector import send_to_component4
"""

import requests
import json
from datetime import datetime

COMPONENT4_API_URL = "http://localhost:8000/predict"
BATCH_API_URL      = "http://localhost:8000/batch_predict"


def send_to_component4(
    service_name: str,
    project_id: str = "Proj_01",
    cpu_percent: float = 0.0,
    cpu_velocity: float = 0.0,
    cpu_trend_5min: float = 0.0,
    cpu_trend_10min: float = 0.0,
    in_flight_queue: float = 0.0,
    incoming_rate: float = 0.0,
    processing_rate: float = 0.0,
    queue_growth_rate: float = 0.0,
    overload_flag: int = 0,
    queue_pressure_index: float = 0.0,
    memory_prob: float = 0.0,
    memory_alert: str = "FALSE",
    memory_pred_label: str = "NORMAL",
    timeout_sec: int = 5
) -> dict:
    payload = {
        "service_name": service_name,
        "project_id": project_id,
        "comp2_cpu": {
            "cpu_percent": cpu_percent,
            "cpu_velocity": cpu_velocity,
            "cpu_trend_5min": cpu_trend_5min,
            "cpu_trend_10min": cpu_trend_10min,
            "in_flight_queue": in_flight_queue,
            "incoming_rate": incoming_rate,
            "processing_rate": processing_rate,
            "queue_growth_rate": queue_growth_rate,
            "overload_flag": overload_flag,
            "queue_pressure_index": queue_pressure_index,
        },
        "comp3_memory": {
            "memory_prob": memory_prob,
            "alert": memory_alert,
            "pred_label": memory_pred_label,
        },
    }
    try:
        response = requests.post(COMPONENT4_API_URL, json=payload, timeout=timeout_sec)
        response.raise_for_status()
        result = response.json()
        action = result.get("action_recommended", "UNKNOWN")
        prob = result.get("cpu_failure_prob", 0) * 100
        mem_p = result.get("memory_prob", 0) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {service_name:<30} "
              f"CPU_prob={prob:5.1f}%  Mem_prob={mem_p:5.1f}%  → {action}")
        return result
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to Component 4 API at {COMPONENT4_API_URL}")
        print("        Make sure the API is running:  uvicorn src.api:app --port 8000")
        return {}
    except requests.exceptions.Timeout:
        print(f"[ERROR] Component 4 API timed out after {timeout_sec}s for {service_name}")
        return {}
    except Exception as e:
        print(f"[ERROR] Unexpected error calling Component 4 API: {e}")
        return {}


def send_comp2_cpu_data(
    service_name: str,
    project_id: str = "Proj_01",
    cpu_percent: float = 0.0,
    cpu_velocity: float = 0.0,
    cpu_trend_5min: float = 0.0,
    cpu_trend_10min: float = 0.0,
    in_flight_queue: float = 0.0,
    incoming_rate: float = 0.0,
    processing_rate: float = 0.0,
    queue_growth_rate: float = 0.0,
    overload_flag: int = 0,
    queue_pressure_index: float = 0.0,
    timeout_sec: int = 5,
) -> dict:
    """
    Component 2 only: send CPU, queue, system data. Memory fields default to normal.
    """
    return send_to_component4(
        service_name=service_name, project_id=project_id,
        cpu_percent=cpu_percent, cpu_velocity=cpu_velocity,
        cpu_trend_5min=cpu_trend_5min, cpu_trend_10min=cpu_trend_10min,
        in_flight_queue=in_flight_queue, incoming_rate=incoming_rate,
        processing_rate=processing_rate, queue_growth_rate=queue_growth_rate,
        overload_flag=overload_flag, queue_pressure_index=queue_pressure_index,
        timeout_sec=timeout_sec,
    )


def send_comp3_memory_data(
    service_name: str,
    project_id: str = "Proj_01",
    memory_prob: float = 0.0,
    alert: str = "FALSE",
    pred_label: str = "NORMAL",
    timeout_sec: int = 5,
) -> dict:
    """
    Component 3 only: send memory prediction data. CPU fields default to 0.
    """
    return send_to_component4(
        service_name=service_name, project_id=project_id,
        memory_prob=memory_prob, memory_alert=alert, memory_pred_label=pred_label,
        timeout_sec=timeout_sec,
    )


def send_batch_to_component4(services: list[dict], timeout_sec: int = 10) -> list[dict]:
    """
    Batch send: services is a list of dicts matching send_to_component4() kwargs.
    """
    try:
        payloads = []
        for s in services:
            cpu_fields = {
                "cpu_percent": s.get("cpu_percent", 0.0),
                "cpu_velocity": s.get("cpu_velocity", 0.0),
                "cpu_trend_5min": s.get("cpu_trend_5min", 0.0),
                "cpu_trend_10min": s.get("cpu_trend_10min", 0.0),
                "in_flight_queue": s.get("in_flight_queue", 0.0),
                "incoming_rate": s.get("incoming_rate", 0.0),
                "processing_rate": s.get("processing_rate", 0.0),
                "queue_growth_rate": s.get("queue_growth_rate", 0.0),
                "overload_flag": s.get("overload_flag", 0),
                "queue_pressure_index": s.get("queue_pressure_index", 0.0),
            }
            mem_fields = {
                "memory_prob": s.get("memory_prob", 0.0),
                "alert": s.get("memory_alert", s.get("alert", "FALSE")),
                "pred_label": s.get("memory_pred_label", s.get("pred_label", "NORMAL")),
            }
            payloads.append({
                "service_name": s["service_name"],
                "project_id": s.get("project_id", "Proj_01"),
                "comp2_cpu": cpu_fields,
                "comp3_memory": mem_fields,
            })
        response = requests.post(BATCH_API_URL, json={"services": payloads}, timeout=timeout_sec)
        response.raise_for_status()
        results = response.json()
        for r in results:
            action = r.get("action_recommended", "UNKNOWN")
            prob = r.get("cpu_failure_prob", 0) * 100
            name = r.get("service_name", "?")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {name:<30} CPU_prob={prob:5.1f}%  → {action}")
        return results
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Cannot connect to Component 4 API at {BATCH_API_URL}")
        return []
    except Exception as e:
        print(f"[ERROR] Batch call failed: {e}")
        return []


if __name__ == "__main__":
    print("Testing connection to Component 4 API...\n")
    result = send_to_component4(
        service_name="test-service",
        project_id="Proj_01",
        cpu_percent=50.0, cpu_velocity=2.0, cpu_trend_5min=3.0, cpu_trend_10min=5.0,
        in_flight_queue=80.0, incoming_rate=200.0, processing_rate=195.0,
        queue_growth_rate=5.0, overload_flag=0, queue_pressure_index=0.4,
        memory_prob=0.2, memory_alert="FALSE", memory_pred_label="NORMAL",
    )
    if result:
        print("\nConnection successful! Full API response:")
        print(json.dumps(result, indent=2))
    else:
        print("\nConnection failed. Is the API running?")
        print("  Run:  uvicorn src.api:app --host 0.0.0.0 --port 8000")
