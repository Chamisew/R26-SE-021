"""
=============================================================================
pipeline.py
Stack-Agnostic Log Parsing and Hybrid Feature Extraction Pipeline
for Memory Failure Detection

Core Responsibilities:
1. Data Ingestion & Normalization: Ingest telemetry and logs from sidecars or
   CSV inputs; handle missing data, scheme validation, and duplicate removal.
2. Log Parsing: Process logs via a single global Drain3 miner to extract templates
   and cluster IDs regardless of service stack.
3. Hybrid Classification: Execute a three-layer detection engine (Layer 1: Regex
   Keyword Matching, Layer 2: TF-IDF / Embedding Semantic Similarity,
   Layer 3: Metric Fusion) with debouncing and confidence scoring.
4. Feature Engineering: Perform per-service multi-scale sliding-window aggregation
   to capture RAM, heap, CPU, and GC trends.
5. Ground-Truth Alignment & Export: Map ground-truth failure windows, label
   pre-failure/failure incident phases, and export ML-ready datasets.
=============================================================================
"""
CONFIG = {
    "data_dir"               : "data/",
    "output_dir"             : "output/",
    "models_dir"             : "models/",
    "failure_events_csv"     : "data/failure_events.csv",
    "drain3_state_file"      : "models/drain3_state.bin",
    "live_output_csv"        : "output/live_raw_collection.csv",
    "pipeline_output_csv"    : "output/pipeline_output.csv",
    "ml_ready_csv"           : "output/ml_ready_dataset.csv",
    "drain3_depth"           : 4,
    "drain3_sim_thresh"      : 0.4,
    "drain3_max_children"    : 100,
    "window_size"            : 12,
    "window_minutes"         : 5,
    "tfidf_threshold"        : 0.12,
    "semantic_margin"        : 0.05,
    "semantic_backend"       : "tfidf",
    "embedding_model"        : "all-MiniLM-L6-v2",
    "embedding_threshold"    : 0.35,
    "metric_score_threshold" : 3,
    "live_duration_minutes"  : 1,
}
def load_failure_events():
    """Load failure events from configuration path.
    Returns DataFrame with validated schema or empty DataFrame if missing/malformed.
    """
    events_path = CONFIG.get("failure_events_csv", os.path.join(os.path.dirname(__file__), "data", "failure_events.csv"))
    expected_cols = ["service_name", "failure_type", "start_time", "end_time"]
    
    if not os.path.exists(events_path):
        return pd.DataFrame(columns=expected_cols)
    try:
        df = pd.read_csv(events_path)
        for col in expected_cols:
            if col not in df.columns:
                return pd.DataFrame(columns=expected_cols)
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
        return df.dropna(subset=["start_time", "end_time"])
    except Exception:
        return pd.DataFrame(columns=expected_cols)

        def normalize_event_timestamps(df, col="timestamp"):
    """Coerce timestamps to ISO UTC-naive datetime objects."""
    if df is None or df.empty or col not in df.columns:
        return df
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    return df