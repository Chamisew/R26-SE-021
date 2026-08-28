import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Data path configuration (resolved relative to the workspace root) ────────
# These absolute paths are computed from this file's location so the pipeline
# works regardless of the current working directory — it can be launched from
# either the workspace root or the cpu_spike_predictor/ directory.
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CPU_SPIKE_PREDICTOR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Component 2 (CPU/queue) dataset — produced by the Queue-Aware CPU Spike Analyzer.
# Sourced directly from the c2 module instead of a local data/raw/ copy.
DEFAULT_COMP2_CSV = os.path.join(
    _WORKSPACE_ROOT, "c2", "Queue-Aware CPU Spike Analyzer",
    "research_framework", "final_research_dataset.csv",
)
# Component 3 (memory) dataset — produced by MUP (Memory Usage Predictor).
# Sourced directly from the MUP module instead of a local data/raw/ copy.
DEFAULT_COMP3_CSV = os.path.join(
    _WORKSPACE_ROOT, "MUP", "data", "memory_predictions.csv",
)

COMP2_REQUIRED_COLUMNS = {
    "incident_id", "timestamp", "time", "system_state", "incident_phase",
    "failing_service", "patient_zero", "cpu_percent", "cpu_velocity",
    "cpu_trend_5min", "cpu_trend_10min", "in_flight_queue", "incoming_rate",
    "processing_rate", "queue_growth_rate", "overload_flag",
    "queue_pressure_index", "incident_duration", "label"
}

COMP2_CPU_FEATURE_COLUMNS = [
    "cpu_percent",
    "cpu_velocity",
    "cpu_trend_5min",
    "cpu_trend_10min",
    "in_flight_queue",
    "incoming_rate",
    "processing_rate",
    "queue_growth_rate",
    "overload_flag",
    "queue_pressure_index",
]

COMP3_REQUIRED_COLUMNS = {
    "timestamp", "service_name", "project_id", "memory_prob", "alert", "pred_label"
}


def add_lead_time_labels(df: pd.DataFrame, lead_time_minutes: int = 5) -> pd.DataFrame:
    """
    Creates a proactive target label 'imminent_failure' for Component 2 data.
    imminent_failure = 1 if ground_truth_failure == 1 now or within the next lead_time_minutes rows.
    Uses a forward-looking rolling max over the ground_truth_failure column.
    Uses project_id/service_name for grouping if available, otherwise treats as single series.
    """
    group_col = None
    if "project_id" in df.columns and df["project_id"].notna().any():
        group_col = "project_id"
    elif "service_name" in df.columns and df["service_name"].notna().any():
        group_col = "service_name"

    df = df.copy().reset_index(drop=True)
    df["imminent_failure"] = df["ground_truth_failure"].copy().astype(int)

    def _forward_rolling_max(series: pd.Series, window: int) -> pd.Series:
        """Forward-looking rolling max: for each position i, max(series[i : i+window])."""
        reversed_s = series.iloc[::-1].reset_index(drop=True)
        result = reversed_s.rolling(window=window, min_periods=1).max().iloc[::-1]
        return result.reset_index(drop=True)

    if group_col and df[group_col].nunique() > 1:
        df = df.sort_values(by=[group_col, "timestamp"]).reset_index(drop=True)
        for _, proj_df in df.groupby(group_col):
            has_failure_ahead = _forward_rolling_max(
                proj_df["ground_truth_failure"], lead_time_minutes + 1
            )
            df.loc[proj_df.index, "imminent_failure"] = has_failure_ahead.astype(int).values
    else:
        has_failure_ahead = _forward_rolling_max(
            df["ground_truth_failure"], lead_time_minutes + 1
        )
        df["imminent_failure"] = has_failure_ahead.astype(int)

    return df


def _validate_comp2_schema(df: pd.DataFrame, file_path: str) -> None:
    """
    Strict schema validation for Component 2 dataset.
    Raises ValueError with a clear message if required columns are missing.
    Does NOT silently create fake columns.
    """
    missing = COMP2_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Component 2 dataset '{file_path}' is missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(COMP2_REQUIRED_COLUMNS)}. "
            f"This is likely the wrong CSV file — Component 2 requires 'final_research_dataset.csv'."
        )


def _validate_comp3_schema(df: pd.DataFrame, file_path: str) -> None:
    """
    Strict schema validation for Component 3 dataset.
    Raises ValueError with a clear message if required columns are missing.
    Does NOT silently create fake columns.
    """
    missing = COMP3_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Component 3 dataset '{file_path}' is missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(COMP3_REQUIRED_COLUMNS)}. "
            f"This is likely the wrong CSV file — Component 3 requires 'memory_predictions.csv'."
        )


def load_cpu_features(file_path: str = None, lead_time_minutes: int = 5) -> pd.DataFrame:
    """
    Loads Component 2 CSV dataset: final_research_dataset.csv
    Contains CPU, queue, system state, incident data, and label.
    Operates ONLY on Component 2-owned fields. Does NOT inject memory columns.
    Validates schema strictly — fails with a clear error if columns are missing.
    """
    if file_path is None or not os.path.exists(file_path):
        # Fall back to the workspace-root-relative default path so the dataset
        # is found regardless of the current working directory.
        if file_path is None:
            file_path = DEFAULT_COMP2_CSV
        elif os.path.exists(DEFAULT_COMP2_CSV):
            file_path = DEFAULT_COMP2_CSV
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Component 2 CSV not found at: {file_path}. "
                f"Expected 'final_research_dataset.csv' produced by Component 2 "
                f"at c2/Queue-Aware CPU Spike Analyzer/research_framework/."
            )

    logger.info(f"Loading Component 2 (CPU/Queue/System) dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Strict schema validation — do NOT silently create missing columns
    _validate_comp2_schema(df, file_path)

    # Derive ground_truth_failure from label (Component 2 target)
    if "ground_truth_failure" not in df.columns:
        df["ground_truth_failure"] = df["label"].fillna(0).astype(int)
    else:
        df["ground_truth_failure"] = df["ground_truth_failure"].fillna(0).astype(int)

    # Derive project_id and service_name from failing_service if not present
    # (final_research_dataset.csv does not include these columns, but LOPO and merge need them)
    if "project_id" not in df.columns:
        import re
        def _extract_pid(svc):
            if pd.isna(svc):
                return "Proj_01"
            m = re.search(r"proj[_\-]?(\d+)", str(svc), re.IGNORECASE)
            if m:
                return f"Proj_{int(m.group(1)):02d}"
            return "Proj_01"
        df["project_id"] = df["failing_service"].apply(_extract_pid)
        # Assign incrementing project IDs per incident group for non-incident rows
        incident_groups = df[df["incident_id"] > 0]["incident_id"].unique()
        for idx, inc_id in enumerate(sorted(incident_groups)):
            mask = df["incident_id"] == inc_id
            svc_name = df.loc[mask, "failing_service"].dropna().iloc[0] if df.loc[mask, "failing_service"].notna().any() else f"srv_{idx+1:02d}"
            df.loc[mask, "project_id"] = _extract_pid(svc_name)
        # For rows with no incident, assign to a default project based on row position
        if df["project_id"].nunique() <= 1 and len(incident_groups) == 0:
            df["project_id"] = "Proj_01"

    if "service_name" not in df.columns:
        if "failing_service" in df.columns:
            df["service_name"] = df["failing_service"]
        else:
            df["service_name"] = df["project_id"].apply(
                lambda p: f"srv-{str(p).lower()}-backend" if pd.notna(p) else "srv-generic-backend"
            )
        # Fill NaN service names using project_id
        mask_svc = df["service_name"].isna()
        if mask_svc.any():
            df.loc[mask_svc, "service_name"] = df.loc[mask_svc, "project_id"].apply(
                lambda p: f"srv-{str(p).lower()}-backend" if pd.notna(p) else "srv-generic-backend"
            )
    df["project_id"] = df["project_id"].fillna("Proj_01")

    # Parse timestamp (Component 2 uses Unix numeric timestamps)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Fill NaN values in feature columns (not creating columns — they exist per schema)
    for col in COMP2_CPU_FEATURE_COLUMNS:
        if col in df.columns:
            if col == "overload_flag":
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(0.0)

    df["ground_truth_failure"] = df["ground_truth_failure"].fillna(0).astype(int)
    df["label"] = df["label"].fillna(0).astype(int)

    df = add_lead_time_labels(df, lead_time_minutes=lead_time_minutes)

    logger.info(f"Component 2 loaded: {len(df)} rows. "
                f"Features: {[c for c in COMP2_CPU_FEATURE_COLUMNS if c in df.columns]}")
    return df


def load_memory_predictions(file_path: str = DEFAULT_COMP3_CSV) -> pd.DataFrame:
    """
    Loads Component 3 memory dataset: memory_predictions.csv
    Operates ONLY on Component 3-owned fields:
    timestamp, service_name, project_id, memory_prob, alert, pred_label.
    Does NOT inject CPU columns.
    Validates schema strictly — fails with a clear error if columns are missing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Component 3 memory predictions file not found at: {file_path}")

    logger.info(f"Loading Component 3 (Memory) dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Strict schema validation — do NOT silently create missing columns
    _validate_comp3_schema(df, file_path)

    # Alias for backward compatibility
    df["memory_leak_prob"] = df["memory_prob"]

    # Fill NaN values in existing columns
    df["service_name"] = df["service_name"].fillna("srv-generic-backend")
    df["project_id"] = df["project_id"].fillna("Proj_01")
    df["alert"] = df["alert"].fillna("FALSE")
    df["pred_label"] = df["pred_label"].fillna("NORMAL")
    df["memory_prob"] = df["memory_prob"].fillna(0.0)

    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        raw_ts = df["timestamp"].astype(str)
        comp3_like = raw_ts.str.match(r"^\d{1,2}:\d{1,2}(\.\d+)?$")
        if comp3_like.any():
            comp3_idx = comp3_like
            parsed_parts = raw_ts[comp3_idx].str.extract(r"^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")
            minutes = pd.to_numeric(parsed_parts[0], errors="coerce")
            seconds = pd.to_numeric(parsed_parts[1], errors="coerce")
            total_seconds = minutes * 60 + seconds
            base_dt = pd.Timestamp("2026-06-09 00:00:00")
            df.loc[comp3_idx, "timestamp_dt"] = base_dt + pd.to_timedelta(total_seconds, unit="s")
        non_comp3 = ~comp3_like if comp3_like.any() else pd.Series(True, index=df.index)
        if non_comp3.any():
            df.loc[non_comp3, "timestamp_dt"] = pd.to_datetime(raw_ts[non_comp3], errors="coerce")

    df["alert_bool"] = df["alert"].astype(str).str.upper().isin(["TRUE", "1", "YES", "T"])

    logger.info(f"Component 3 loaded: {len(df)} rows. "
                f"memory_prob range: [{df['memory_prob'].min():.3f}, {df['memory_prob'].max():.3f}] "
                f"| TRUE alerts: {df['alert_bool'].sum()}")
    return df


def merge_datasets(cpu_df: pd.DataFrame, mem_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integration Layer: Merges Component 2 (CPU/Queue) and Component 3 (Memory) datasets.

    Timestamp Alignment Strategy:
    - Component 2: Unix numeric timestamp -> timestamp_dt (full datetime)
    - Component 3: May be "MM:SS.s" format, or ISO, or numeric -> normalized to timestamp_dt
    - Both datasets are grouped by project_id + service_name, then aligned by row-position
      within each service group. This preserves the per-service temporal sequence when
      exact timestamp matches are not guaranteed (different formats/offsets).
    - For each service group, rows are matched 1:1 by index within the group (up to the
      minimum length). Unmatched rows are handled safely with memory defaults.

    Source Preservation:
    - All Component 2 CPU/queue/system/incident columns come from cpu_df.
    - All Component 3 memory_prob/alert/pred_label/project_id come from mem_df.
    - A prefix-free schema with clear provenance is used.
    """
    logger.info("=== Integration Layer: Merging Component 2 (CPU) + Component 3 (Memory) ===")

    cpu_df = cpu_df.copy()
    mem_df = mem_df.copy()

    if "service_name" not in cpu_df.columns:
        cpu_df["service_name"] = "srv-generic-backend"
    if "project_id" not in cpu_df.columns:
        cpu_df["project_id"] = "Proj_01"
    if "service_name" not in mem_df.columns:
        mem_df["service_name"] = "srv-generic-backend"
    if "project_id" not in mem_df.columns:
        mem_df["project_id"] = "Proj_01"

    group_keys = ["project_id", "service_name"]
    cpu_groups = list(cpu_df.groupby(group_keys, sort=False))
    mem_groups_dict = {k: g for k, g in mem_df.groupby(group_keys, sort=False)}

    merged_parts = []
    unmatched_services = []

    for key, cpu_g in cpu_groups:
        cpu_g = cpu_g.sort_values(
            by="timestamp_dt" if "timestamp_dt" in cpu_g.columns else "timestamp"
        ).reset_index(drop=True)

        if key in mem_groups_dict:
            mem_group_raw = mem_groups_dict[key]
            mem_g = mem_group_raw.sort_values(
                by="timestamp_dt" if "timestamp_dt" in mem_group_raw.columns else "timestamp"
            ).reset_index(drop=True)
            n = min(len(cpu_g), len(mem_g))
            cpu_slice = cpu_g.iloc[:n].copy()
            mem_slice = mem_g.iloc[:n][["memory_prob", "alert", "pred_label", "memory_leak_prob"]].copy()
            mem_slice.index = cpu_slice.index
            merged_part = pd.concat([cpu_slice, mem_slice], axis=1)

            if len(cpu_g) > n:
                remainder = cpu_g.iloc[n:].copy()
                remainder["memory_prob"] = 0.0
                remainder["memory_leak_prob"] = 0.0
                remainder["alert"] = "FALSE"
                remainder["pred_label"] = "NORMAL"
                merged_part = pd.concat([merged_part, remainder], ignore_index=True)
        else:
            unmatched_services.append(key)
            merged_part = cpu_g.copy()
            merged_part["memory_prob"] = 0.0
            merged_part["memory_leak_prob"] = 0.0
            merged_part["alert"] = "FALSE"
            merged_part["pred_label"] = "NORMAL"

        merged_parts.append(merged_part)

    merged_df = pd.concat(merged_parts, ignore_index=True)

    sort_cols = []
    if "project_id" in merged_df.columns:
        sort_cols.append("project_id")
    if "service_name" in merged_df.columns:
        sort_cols.append("service_name")
    if "timestamp_dt" in merged_df.columns:
        sort_cols.append("timestamp_dt")
    elif "timestamp" in merged_df.columns:
        sort_cols.append("timestamp")
    if sort_cols:
        merged_df = merged_df.sort_values(by=sort_cols).reset_index(drop=True)

    logger.info(f"Integration complete: {len(merged_df)} total rows.")
    logger.info(f"  - Component 2 feature columns present: {[c for c in COMP2_CPU_FEATURE_COLUMNS if c in merged_df.columns]}")
    logger.info(f"  - Component 3 memory columns present: memory_prob, alert, pred_label, memory_leak_prob")
    if unmatched_services:
        logger.warning(f"  - Services in Comp2 not found in Comp3 (memory defaults applied): {unmatched_services[:5]}...")

    return merged_df


def save_aligned_telemetry(merged_df: pd.DataFrame, output_dir: str = "data/processed") -> tuple[str, str]:
    """
    Saves the integrated Component 2+3 dataset in both CSV and Apache Parquet formats.
    Drops intermediate datetime join keys before saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "aligned_telemetry.csv")
    parquet_path = os.path.join(output_dir, "aligned_telemetry.parquet")

    save_df = merged_df.drop(columns=["time_key"], errors="ignore")

    save_df.to_csv(csv_path, index=False)
    try:
        save_df.to_parquet(parquet_path, index=False, compression="snappy")
        logger.info(f"Saved Big Data Parquet file to {parquet_path} (Snappy compressed)")
    except Exception as e:
        logger.warning(f"Could not write parquet format: {e}")

    logger.info(f"Saved aligned integration dataset CSV to {csv_path} "
                f"({len(save_df)} rows, {len(save_df.columns)} cols)")
    return csv_path, parquet_path


def load_aligned_telemetry(processed_dir: str = "data/processed") -> pd.DataFrame:
    """
    High-performance loader: Attempts to load Apache Parquet first (<50ms),
    falling back to CSV if Parquet is not found.
    """
    parquet_path = os.path.join(processed_dir, "aligned_telemetry.parquet")
    csv_path = os.path.join(processed_dir, "aligned_telemetry.csv")

    if os.path.exists(parquet_path):
        logger.info(f"Fast-loading integrated telemetry from Apache Parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        logger.info(f"Loading integrated telemetry from CSV: {csv_path}")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No processed telemetry dataset found in {processed_dir}")


if __name__ == "__main__":
    try:
        cpu = load_cpu_features()
        mem = load_memory_predictions()
        merged = merge_datasets(cpu, mem)
        save_aligned_telemetry(merged)
        print("Ingestion & Parquet conversion complete!")
        print("Comp2 columns:", [c for c in cpu.columns])
        print("Comp3 columns:", [c for c in mem.columns])
        print("Merged shape:", merged.shape)
        print(merged[["cpu_percent", "memory_prob", "alert", "pred_label", "label"]].head())
    except Exception as e:
        logger.error(f"Ingestion self-test failed: {e}", exc_info=True)
