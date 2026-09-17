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

    # 1) Deterministic duplicate detection based on service, timestamp, and log message
cleaned = cleaned.drop_duplicates(
    subset=["timestamp", "service_name", "log_message"],
    keep="first"
).reset_index(drop=True)

def _detect_azure_stack(service_name: str, sidecar_lang: str) -> str:
    lang = (sidecar_lang or "").lower()
    cname = (service_name or "").lower()
    
    mapping = [
        (("python", "flask", "django", "fastapi"), "Python/Flask"),
        (("node", "express", "javascript", "npm"), "Node.js/Express"),
        (("java", "spring", "jvm", "tomcat"), "Java/SpringBoot"),
        (("dotnet", "aspnet", "c#"), ".NET/Core"),
        (("go", "golang"), "Go/net_http"),
        (("php", "fpm"), "PHP/FPM"),
        (("ruby", "rails", "puma"), "Ruby/Puma"),
        (("swift", "vapor"), "Swift/Vapor")
    ]
    
    for keywords, stack_label in mapping:
        if any(k in lang or k in cname for k in keywords):
            return stack_label
            
    return "Microservice"

    def _detect_log_level(text: str) -> str:
    if not text:
        return "INFO"
    upper = str(text).upper()
    if any(k in upper for k in ("CRITICAL", "FATAL", "EMERGENCY")):
        return "CRITICAL"
    if any(k in upper for k in ("ERROR", "EXCEPTION", "OOM", "FAIL")):
        return "ERROR"
    if any(k in upper for k in ("WARN", "WARNING")):
        return "WARNING"
    if any(k in upper for k in ("DEBUG", "TRACE")):
        return "DEBUG"
    return "INFO"

    GC_KEYWORDS = ("GC", "GARBAGE", "COLLECTION", "PAUSE", "YOUNG GEN", "OLD GEN")

def _has_gc_keyword(text: str) -> bool:
    if not text:
        return False
    upper = str(text).upper()
    return any(k in upper for k in GC_KEYWORDS)

    # Extracted parsing logic block inside stage2_drain3_parsing:
result = miner.add_log_message(msg)
templates.append(result["template_mined"])
cluster_ids.append(result["cluster_id"])

state_path = CONFIG.get("drain3_state_file", os.path.join(config["models_dir"], "drain3_state.bin"))
os.makedirs(os.path.dirname(state_path), exist_ok=True)
with open(state_path, "wb") as f:
    pickle.dump(miner, f)
print(f"Drain3 state successfully saved -> {state_path}")

def _layer1_keyword(template: str):
    """Layer 1: Deterministic word-boundary keyword matching for memory failures."""
    if not template or not isinstance(template, str):
        return None, None, None
    for pattern, failure_type in LAYER1_PATTERNS:
        if pattern.search(template):
            return "FAILURE", "keyword", failure_type
    return None, None, None

    def _layer2_tfidf(template: str, tmpl_lookup: dict = None, threshold: float = 0.12, margin: float = 0.05):
    """Layer 2: TF-IDF semantic template similarity layer."""
    template_str = str(template) if template is not None else ""
    if tmpl_lookup is not None and template_str in tmpl_lookup:
        entry = tmpl_lookup[template_str]
        sim_failure = entry.get("sim_failure", 0.0)
        sim_normal = entry.get("sim_normal", 0.0)
        ftype = entry.get("failure_type", "oom")
    else:
        # Fallback inline evaluation
        all_docs = [template_str] + FAILURE_REFERENCE_CORPUS + NORMAL_REFERENCE_CORPUS
        v = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
        v.fit(all_docs)
        t_vec = v.transform([template_str])
        f_mat = v.transform(FAILURE_REFERENCE_CORPUS)
        n_mat = v.transform(NORMAL_REFERENCE_CORPUS)
        sim_f_row = cosine_similarity(t_vec, f_mat)[0]
        sim_n_row = cosine_similarity(t_vec, n_mat)[0]
        best_f_idx = int(np.argmax(sim_f_row))
        sim_failure = float(sim_f_row[best_f_idx])
        sim_normal = float(np.max(sim_n_row))
        ftype = FAILURE_REFERENCE_TYPES[best_f_idx]

    if sim_failure > threshold and (sim_failure - sim_normal) > margin:
        return "FAILURE", "tfidf_semantic", ftype
    return "NORMAL", "normal", "none"

    def _layer2_embedding(template: str, tmpl_lookup: dict = None, threshold: float = 0.35, margin: float = 0.05):
    """Layer 2: Dense sentence embedding semantic similarity layer."""
    template_str = str(template) if template is not None else ""
    if tmpl_lookup is not None and template_str in tmpl_lookup:
        entry = tmpl_lookup[template_str]
        sim_failure = entry["sim_failure"]
        sim_normal = entry["sim_normal"]
        ftype = entry["failure_type"]
    else:
        model = _get_embedding_model()
        t_emb = model.encode([template_str], convert_to_numpy=True)
        t_emb = t_emb / max(float(np.linalg.norm(t_emb)), 1e-9)
        f_emb = model.encode(FAILURE_REFERENCE_CORPUS, convert_to_numpy=True)
        f_emb = f_emb / np.maximum(np.linalg.norm(f_emb, axis=1, keepdims=True), 1e-9)
        n_emb = model.encode(NORMAL_REFERENCE_CORPUS, convert_to_numpy=True)
        n_emb = n_emb / np.maximum(np.linalg.norm(n_emb, axis=1, keepdims=True), 1e-9)

        sim_f = np.dot(t_emb, f_emb.T)[0]
        sim_n = np.dot(t_emb, n_emb.T)[0]
        best_idx = int(np.argmax(sim_f))
        sim_failure = float(sim_f[best_idx])
        sim_normal = float(np.max(sim_n))
        ftype = FAILURE_REFERENCE_TYPES[best_idx]

    if sim_failure > threshold and (sim_failure - sim_normal) > margin:
        return "FAILURE", "embedding_semantic", ftype
    return "NORMAL", "normal", "none"

    def tune_semantic_params(df):
    """Grid-search hyperparameter tuning for semantic thresholds and margins."""
    if df is None or df.empty or "ground_truth_label" not in df.columns:
        return 0.12, 0.05

    valid_df = df[df["ground_truth_label"].isin(["FAILURE", "NORMAL"])].copy()
    if valid_df.empty or (valid_df["ground_truth_label"] == "FAILURE").sum() == 0:
        return 0.12, 0.05

    backend = CONFIG.get("semantic_backend", "tfidf")
    # Tuning loop logic as implemented in tune_semantic_params

    def _metric_score(row, service_gc_history: dict):
    """Layer 3: Multi-metric evidence fusion combining RAM, CPU, Heap, and GC."""
    score = 0
    ram  = row["ram_percent"]
    heap = row["heap_mb_used"]
    cpu  = row["cpu_percent"]
    gc   = row["gc_count"]
    svc  = row["service_name"]

    bl = _service_metric_baselines.get(svc, {})
    ram_threshold = bl.get("threshold_ram", 85.0)
    heap_threshold = bl.get("threshold_heap", 2500.0)

    if ram > ram_threshold: score += 2
    if cpu > 85: score += 2
    if heap > heap_threshold: score += 2

    history = service_gc_history.setdefault(svc, [])
    history.append(gc)
    if len(history) > 10:
        history.pop(0)
    if len(history) >= 2 and (history[-1] - history[0]) > 3:
        score += 2

    return score