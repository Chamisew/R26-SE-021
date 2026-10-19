import pandas as pd

from src.live_inference import LiveInferenceEngine, preprocess_comp2_for_inference


def test_current_c2_schema_streams_all_services(tmp_path):
    services = [f"service-{index:02d}" for index in range(12)]
    rows = []
    for index, service in enumerate(services):
        rows.append({
            "timestamp": 1_791_898_406.0 + index,
            "incident_id": f"incident-{index}",
            "service": service,
            "cpu_percent": 20.0 + index,
            "cpu_velocity": 1.0,
            "cpu_trend_5min": 2.0,
            "cpu_trend_10min": 3.0,
            "in_flight_queue": 4.0,
            "incoming_rate": 5.0,
            "processing_rate": 4.0,
            "queue_growth_rate": 1.0,
            "overload_flag": 0,
            "qpi": 0.25 + index,
        })

    comp2_path = tmp_path / "final_research_dataset.csv"
    pd.DataFrame(rows).to_csv(comp2_path, index=False)
    engine = LiveInferenceEngine(
        comp2_path=str(comp2_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )

    new_rows = engine._read_new_comp2()
    normalized = preprocess_comp2_for_inference(new_rows)

    assert len(new_rows) == 12
    assert set(normalized["service_name"]) == set(services)
    assert normalized["queue_pressure_index"].tolist() == [row["qpi"] for row in rows]
