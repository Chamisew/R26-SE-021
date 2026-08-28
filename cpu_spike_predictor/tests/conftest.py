import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
