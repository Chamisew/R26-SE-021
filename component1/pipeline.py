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

    def compute_label_3state(df, failure_events):
    """Compute label_3state: NORMAL, PRE_FAILURE (5-10 min prior), and FAILURE."""
    df = df.copy()
    df["label_3state"] = "NORMAL"

    if failure_events is None or failure_events.empty or "timestamp" not in df.columns:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for _, event in failure_events.iterrows():
        svc = event["service_name"]
        start = event["start_time"]
        end = event["end_time"] if "end_time" in event and pd.notna(event["end_time"]) else start

        svc_mask = df["service_name"] == svc
        delta_start = (start - df["timestamp"]).dt.total_seconds()

        # PRE_FAILURE: 5-10 minutes prior to failure event start
        mask_pre = svc_mask & (delta_start > 5 * 60) & (delta_start <= 10 * 60)
        df.loc[mask_pre, "label_3state"] = "PRE_FAILURE"

        # FAILURE: inside the failure duration window
        mask_fail = svc_mask & (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df.loc[mask_fail, "label_3state"] = "FAILURE"

    return df

    def fill_ground_truth(df, failure_events):
    df = df.copy()
    if "ground_truth_label" not in df.columns:
        df["ground_truth_label"] = "UNKNOWN"
    else:
        df["ground_truth_label"] = df["ground_truth_label"].fillna("UNKNOWN")

    if failure_events is None or failure_events.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for _, event in failure_events.iterrows():
        svc = event["service_name"]
        start = event["start_time"]
        end = event["end_time"]
        mask = (
            (df["service_name"] == svc)
            & (df["timestamp"] >= start)
            & (df["timestamp"] <= end)
        )
        df.loc[mask, "ground_truth_label"] = "FAILURE"

    # Only set to NORMAL if the service was actively tracked in failure_events
    services_with_events = set(failure_events["service_name"].unique())
    for svc in services_with_events:
        svc_mask = df["service_name"] == svc
        unknown_mask = svc_mask & (df["ground_truth_label"] == "UNKNOWN")
        df.loc[unknown_mask, "ground_truth_label"] = "NORMAL"

    return df

    def validate_telemetry_schema(df):
    """Validate presence of minimum core schema columns in input telemetry."""
    required_schema = ["timestamp", "service_name", "ram_percent", "cpu_percent", "heap_mb_used"]
    missing = [col for col in required_schema if col not in df.columns]
    if missing:
        raise ValueError(f"Telemetry schema validation failed. Missing columns: {missing}")
    return True

    def normalize_numeric_telemetry(df):
    """Safely coerce RAM, CPU, Heap, and GC fields to numeric types."""
    numeric_fields = ["ram_percent", "cpu_percent", "heap_mb_used", "heap_max", "gc_count"]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0.0)
    return df