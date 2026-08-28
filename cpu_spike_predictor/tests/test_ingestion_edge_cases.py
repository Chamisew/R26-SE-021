import pytest
import pandas as pd
from datetime import datetime
from src.ingestion import (
    load_cpu_features,
    load_memory_predictions,
    merge_datasets,
    _validate_comp2_schema,
    _validate_comp3_schema
)

def test_missing_mandatory_columns_validation(tmp_path):
    # CSV missing critical columns
    invalid_data = pd.DataFrame({
        "timestamp": [1717934400],
        "project_id": ["P1"],
    })
    csv_file = tmp_path / "invalid_c2.csv"
    invalid_data.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        _validate_comp2_schema(invalid_data, str(csv_file))

def test_comp3_missing_mandatory_columns_validation(tmp_path):
    invalid_data = pd.DataFrame({
        "timestamp": ["12:00:00"],
        "project_id": ["P1"],
    })
    csv_file = tmp_path / "invalid_c3.csv"
    invalid_data.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        _validate_comp3_schema(invalid_data, str(csv_file))

def test_corrupted_timestamps_handling(tmp_path):
    corrupted_data = pd.DataFrame({
        "incident_id": [1],
        "timestamp": ["1717934400"],
        "time": ["12:00:00"],
        "system_state": ["NORMAL"],
        "incident_phase": ["NONE"],
        "failing_service": [None],
        "patient_zero": [None],
        "project_id": ["P1"],
        "service_name": ["srv-1"],
        "cpu_percent": [25.0],
        "cpu_velocity": [0.0],
        "cpu_trend_5min": [0.0],
        "cpu_trend_10min": [0.0],
        "in_flight_queue": [10.0],
        "incoming_rate": [150.0],
        "processing_rate": [150.0],
        "queue_growth_rate": [0.0],
        "overload_flag": [0],
        "queue_pressure_index": [1.0],
        "incident_duration": [0.0],
        "label": [0]
    })
    csv_file = tmp_path / "valid_c2.csv"
    corrupted_data.to_csv(csv_file, index=False)

    df = load_cpu_features(str(csv_file), lead_time_minutes=5)
    assert not df.empty
    assert "ground_truth_failure" in df.columns
    assert "imminent_failure" in df.columns
