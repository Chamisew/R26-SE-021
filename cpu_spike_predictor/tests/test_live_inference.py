"""Unit and integration tests for the live inference engine."""
import os
import json
import pytest
import pandas as pd
import numpy as np
from src.live_inference import LiveInferenceEngine, preprocess_comp2_for_inference, preprocess_comp3_for_inference, align_comp2_comp3

@pytest.fixture
def mock_csv_environment(tmp_path, sample_cpu_data, sample_mem_predictions):
    c2_csv = tmp_path / "final_research_dataset.csv"
    c3_csv = tmp_path / "memory_predictions.csv"
    
    # Copy or reference the fixture data
    c2_df = pd.read_csv(sample_cpu_data)
    c2_df.to_csv(c2_csv, index=False)
    
    c3_df = pd.read_csv(sample_mem_predictions)
    c3_df.to_csv(c3_csv, index=False)
    
    return str(c2_csv), str(c3_csv)

def test_live_inference_startup_with_fixtures(mock_csv_environment, tmp_path):
    c2_path, c3_path = mock_csv_environment
    engine = LiveInferenceEngine(
        comp2_path=c2_path,
        comp3_path=c3_path,
        checkpoint_path=str(tmp_path / ".ckpt.json"),
        output_path=str(tmp_path / "preds.jsonl"),
        status_path=str(tmp_path / "status.json"),
        poll_interval=2
    )
    assert engine.startup() is True
    assert engine.model is not None

def test_live_inference_read_and_process(mock_csv_environment, tmp_path):
    c2_path, c3_path = mock_csv_environment
    engine = LiveInferenceEngine(
        comp2_path=c2_path,
        comp3_path=c3_path,
        checkpoint_path=str(tmp_path / ".ckpt.json"),
        output_path=str(tmp_path / "preds.jsonl"),
        status_path=str(tmp_path / "status.json"),
        poll_interval=2
    )
    assert engine.startup() is True
    comp2_new = engine._read_new_comp2()
    assert not comp2_new.empty
    
    comp3_new = engine._read_new_comp3()
    assert not comp3_new.empty
