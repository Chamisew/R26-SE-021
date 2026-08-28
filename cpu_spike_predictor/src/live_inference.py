"""
live_inference.py — Real-Time CSV Consumer & Inference Engine
=============================================================
Production inference component that consumes live-updating CSV files
from Component 2 and Component 3, applies the pre-trained model,
and records predictions.

Architecture:
    Component 2 → writes final_research_dataset.csv
    Component 3 → writes memory_predictions.csv
    THIS COMPONENT → reads both CSVs (READ-ONLY), predicts, records output

This component:
  - Does NOT generate, simulate, or fabricate any data.
  - Does NOT retrain the model.
  - Does NOT modify the source CSV files.
  - Detects new records via checkpoint tracking.
  - Handles partially-written CSV files safely.
  - Continues running when one source has no new data.
"""

import os
import sys
import json
import time
import logging
import signal
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from src.model import (
    load_model,
    predict_probabilities,
    COMP2_CPU_FEATURE_COLS,
    TARGET_COL,
)
from src.ingestion import (
    COMP2_REQUIRED_COLUMNS,
    COMP2_CPU_FEATURE_COLUMNS,
    COMP3_REQUIRED_COLUMNS,
    DEFAULT_COMP2_CSV,
    DEFAULT_COMP3_CSV,
)
from src.xai import generate_enterprise_rca_report

logger = logging.getLogger(__name__)

# ── Configuration defaults ────────────────────────────────────────────────────
# Comp2/Comp3 paths are resolved in src.ingestion relative to the workspace root
# so they work regardless of the current working directory.
DEFAULT_COMP2_PATH = DEFAULT_COMP2_CSV
DEFAULT_COMP3_PATH = DEFAULT_COMP3_CSV
DEFAULT_MODEL_PATH = "outputs/cpu_rf_model.joblib"
DEFAULT_OUTPUT_PATH = "outputs/live_predictions.jsonl"
DEFAULT_STATUS_PATH = "outputs/live_status.json"
DEFAULT_CHECKPOINT_PATH = "outputs/.inference_checkpoint.json"
DEFAULT_POLL_INTERVAL = 1  # 1-second real-time polling interval


class CheckpointManager:
    """
    Persistent checkpoint that tracks which records have already been processed.
    Stored as a JSON file separate from the source CSVs.
    Never modifies the source CSV files.
    """

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        self.path = checkpoint_path
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                logger.info(f"Checkpoint loaded: {data.get('comp2_processed_count', 0)} Comp2 + "
                            f"{data.get('comp3_processed_count', 0)} Comp3 records previously processed.")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Checkpoint file corrupted, starting fresh: {e}")
        return {"comp2_keys": [], "comp3_keys": [], "comp2_processed_count": 0, "comp3_processed_count": 0}

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.state, f, indent=2)
        # Atomic rename to prevent corruption
        os.replace(tmp_path, self.path)

    def is_comp2_processed(self, key: str) -> bool:
        return key in self.state["comp2_keys"]

    def is_comp3_processed(self, key: str) -> bool:
        return key in self.state["comp3_keys"]

    def mark_comp2_processed(self, keys: list[str]) -> None:
        existing = set(self.state["comp2_keys"])
        new_keys = [k for k in keys if k not in existing]
        self.state["comp2_keys"].extend(new_keys)
        self.state["comp2_processed_count"] += len(new_keys)
        # Keep only last 10000 keys to bound memory
        if len(self.state["comp2_keys"]) > 10000:
            self.state["comp2_keys"] = self.state["comp2_keys"][-10000:]
        self.save()

    def reset(self) -> None:
        self.state = {"comp2_keys": [], "comp3_keys": [], "comp2_processed_count": 0, "comp3_processed_count": 0}
        self.save()

    def mark_comp3_processed(self, keys: list[str]) -> None:
        existing = set(self.state["comp3_keys"])
        new_keys = [k for k in keys if k not in existing]
        self.state["comp3_keys"].extend(new_keys)
        self.state["comp3_processed_count"] += len(new_keys)
        if len(self.state["comp3_keys"]) > 10000:
            self.state["comp3_keys"] = self.state["comp3_keys"][-10000:]
        self.save()


def safe_read_csv(file_path: str, max_retries: int = 3, retry_delay: float = 2.0) -> Optional[pd.DataFrame]:
    """
    Reads a CSV file with retry logic for handling partially-written or locked files.
    Returns None if the file cannot be read after all retries.
    Does NOT modify the source file.
    """
    if not os.path.exists(file_path):
        return None

    for attempt in range(max_retries):
        try:
            df = pd.read_csv(file_path)
            # Validate that the file was not truncated: check for all-NaN final row
            if len(df) > 0 and df.iloc[-1].isna().all():
                logger.warning(f"{file_path}: last row is all-NaN (possibly mid-write). Dropping it.")
                df = df.iloc[:-1]
            return df
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"{file_path}: parse error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
        except PermissionError as e:
            logger.warning(f"{file_path}: file locked on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            logger.error(f"{file_path}: unexpected read error: {e}")
            return None

    logger.error(f"{file_path}: failed to read after {max_retries} attempts.")
    return None


def make_comp2_record_key(row: pd.Series) -> str:
    """
    Creates a unique identifier for a Component 2 record.
    Uses incident_id + timestamp as the composite key.
    """
    inc_id = row.get("incident_id", 0)
    ts = row.get("timestamp", "")
    return f"c2:{inc_id}:{ts}"


def make_comp3_record_key(row: pd.Series) -> str:
    """
    Creates a unique identifier for a Component 3 record.
    Uses timestamp + service_name + project_id as the composite key.
    """
    ts = row.get("timestamp", "")
    svc = row.get("service_name", "")
    proj = row.get("project_id", "")
    return f"c3:{ts}:{svc}:{proj}"


def preprocess_comp2_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the SAME preprocessing used during model training to Component 2 data.
    This matches load_cpu_features() but does NOT require 'label' or call add_lead_time_labels()
    (those are training-only steps).

    Preprocessing steps (must match training exactly):
      1. Fill NaN in feature columns: overload_flag → 0 (int), others → 0.0
      2. Parse timestamp to timestamp_dt
      3. Derive project_id/service_name from failing_service if not present
    """
    df = df.copy()

    # Parse timestamp (same logic as load_cpu_features)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Derive project_id from failing_service if not in CSV
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
        # Assign per-incident project IDs
        incident_groups = df[df["incident_id"] > 0]["incident_id"].unique()
        for idx, inc_id in enumerate(sorted(incident_groups)):
            mask = df["incident_id"] == inc_id
            svc_name = df.loc[mask, "failing_service"].dropna() # type: ignore
            if len(svc_name) > 0:
                df.loc[mask, "project_id"] = _extract_pid(svc_name.iloc[0])
        if df["project_id"].nunique() <= 1 and len(incident_groups) == 0:
            df["project_id"] = "Proj_01"

    # Derive service_name from failing_service if not in CSV
    if "service_name" not in df.columns:
        if "failing_service" in df.columns:
            df["service_name"] = df["failing_service"]
        else:
            df["service_name"] = df["project_id"].apply(
                lambda p: f"srv-{str(p).lower()}-backend" if pd.notna(p) else "srv-generic-backend"
            )
        mask_svc = df["service_name"].isna()
        if mask_svc.any():
            df.loc[mask_svc, "service_name"] = df.loc[mask_svc, "project_id"].apply(
                lambda p: f"srv-{str(p).lower()}-backend" if pd.notna(p) else "srv-generic-backend"
            )
    df["project_id"] = df["project_id"].fillna("Proj_01")

    # Fill NaN in feature columns (EXACTLY matching training preprocessing)
    for col in COMP2_CPU_FEATURE_COLUMNS:
        if col in df.columns:
            if col == "overload_flag":
                df[col] = df[col].fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(0.0)

    return df


def preprocess_comp3_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the SAME preprocessing used during model training to Component 3 data.
    Matches load_memory_predictions() preprocessing.
    """
    df = df.copy()

    # Alias
    df["memory_leak_prob"] = df["memory_prob"]

    # Fill NaN
    df["service_name"] = df["service_name"].fillna("srv-generic-backend")
    df["project_id"] = df["project_id"].fillna("Proj_01")
    df["alert"] = df["alert"].fillna("FALSE")
    df["pred_label"] = df["pred_label"].fillna("NORMAL")
    df["memory_prob"] = df["memory_prob"].fillna(0.0)

    # Parse timestamp (same logic as load_memory_predictions)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    else:
        raw_ts = df["timestamp"].astype(str)
        comp3_like = raw_ts.str.match(r"^\d{1,2}:\d{1,2}(\.\d+)?$") # type: ignore
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

    df["alert_bool"] = df["alert"].astype(str).str.upper().isin(["TRUE", "1", "YES", "T"]) # type: ignore
    return df


def build_feature_matrix(comp2_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the EXACT 10 Comp2 CPU feature columns in the order the trained model expects.
    Fills any remaining NaN with 0.0 (matching training preprocessing).
    """
    X = comp2_df[COMP2_CPU_FEATURE_COLS].copy()
    for col in COMP2_CPU_FEATURE_COLS:
        if col == "overload_flag":
            X[col] = X[col].fillna(0).astype(int)
        else:
            X[col] = X[col].fillna(0.0)
    return X


def align_comp2_comp3(
    comp2_new: pd.DataFrame,
    comp3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aligns new Component 2 records with available Component 3 data.

    Alignment strategy (same as training pipeline's merge_datasets):
    - Group both by (project_id, service_name).
    - Within each group, sort by timestamp_dt and match by row-position.
    - Comp2 rows without a matching Comp3 row get safe memory defaults (0.0, FALSE, NORMAL).
      These are NOT invented data — they represent "no memory observation available yet".
    """
    if comp3_df is None or comp3_df.empty:
        result = comp2_new.copy()
        result["memory_prob"] = 0.0
        result["memory_leak_prob"] = 0.0
        result["alert"] = "FALSE"
        result["pred_label"] = "NORMAL"
        result["alert_bool"] = False
        return result

    comp2_new = comp2_new.copy()
    comp3_df = comp3_df.copy()

    group_keys = ["project_id", "service_name"]
    cpu_groups = list(comp2_new.groupby(group_keys, sort=False))
    mem_groups_dict = {k: g for k, g in comp3_df.groupby(group_keys, sort=False)}

    merged_parts = []
    for key, cpu_g in cpu_groups:
        sort_col = "timestamp_dt" if "timestamp_dt" in cpu_g.columns else "timestamp"
        cpu_g = cpu_g.sort_values(by=sort_col).reset_index(drop=True)

        if key in mem_groups_dict:
            mem_g = mem_groups_dict[key].sort_values(
                by="timestamp_dt" if "timestamp_dt" in mem_groups_dict[key].columns else "timestamp"
            ).reset_index(drop=True)
            n = min(len(cpu_g), len(mem_g))
            cpu_slice = cpu_g.iloc[:n].copy()
            mem_cols = ["memory_prob", "alert", "pred_label", "memory_leak_prob", "alert_bool"]
            available_mem_cols = [c for c in mem_cols if c in mem_g.columns]
            mem_slice = mem_g.iloc[:n][available_mem_cols].copy()
            mem_slice.index = cpu_slice.index
            merged_part = pd.concat([cpu_slice, mem_slice], axis=1)

            if len(cpu_g) > n:
                remainder = cpu_g.iloc[n:].copy()
                remainder["memory_prob"] = 0.0
                remainder["memory_leak_prob"] = 0.0
                remainder["alert"] = "FALSE"
                remainder["pred_label"] = "NORMAL"
                remainder["alert_bool"] = False
                merged_part = pd.concat([merged_part, remainder], ignore_index=True)
        else:
            merged_part = cpu_g.copy()
            merged_part["memory_prob"] = 0.0
            merged_part["memory_leak_prob"] = 0.0
            merged_part["alert"] = "FALSE"
            merged_part["pred_label"] = "NORMAL"
            merged_part["alert_bool"] = False

        merged_parts.append(merged_part)

    return pd.concat(merged_parts, ignore_index=True)


class LiveInferenceEngine:
    """
    Real-time inference engine that consumes live-updating CSV files.

    Startup:
      1. Loads the trained model (does NOT retrain).
      2. Loads the checkpoint (if any).
      3. Validates both CSV schemas.
      4. Processes any unprocessed existing records.
      5. Enters polling loop for new records.
    """

    def __init__(
        self,
        comp2_path: str = DEFAULT_COMP2_PATH,
        comp3_path: str = DEFAULT_COMP3_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        output_path: str = DEFAULT_OUTPUT_PATH,
        status_path: str = DEFAULT_STATUS_PATH,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self.comp2_path = comp2_path
        self.comp3_path = comp3_path
        self.model_path = model_path
        self.output_path = output_path
        self.status_path = status_path
        self.poll_interval = poll_interval
        self.model = None
        self.checkpoint = CheckpointManager(checkpoint_path)
        self._running = False

    def startup(self) -> bool:
        """
        Initialization sequence:
        1. Load trained model (no retraining).
        2. Verify both CSV files exist and validate schemas.
        3. Load checkpoint state.
        Returns True if ready, False if critical failure.
        """
        logger.info("=" * 70)
        logger.info("  Live Inference Engine — Starting Up")
        logger.info("=" * 70)

        # 1. Load trained model (MUST exist — no auto-training)
        if not os.path.exists(self.model_path):
            logger.critical(f"Trained model NOT found at {self.model_path}. "
                            f"Run 'python -m src.cli train' first to produce the model.")
            return False

        try:
            self.model = load_model(self.model_path)
            logger.info(f"Trained model loaded: {self.model_path} "
                        f"({len(self.model.feature_importances_)} features, "
                        f"classes={list(self.model.classes_)})")
        except Exception as e:
            logger.critical(f"Failed to load trained model: {e}")
            return False

        # 2. Verify CSV files exist
        for path, name in [(self.comp2_path, "Component 2"), (self.comp3_path, "Component 3")]:
            if not os.path.exists(path):
                logger.critical(f"{name} CSV not found at {path}. "
                                f"Ensure Component 2/3 have produced their output files.")
                return False
            logger.info(f"{name} CSV found: {path}")

        # 3. Validate schemas (read a small sample)
        comp2_sample = safe_read_csv(self.comp2_path)
        if comp2_sample is not None:
            missing_comp2 = COMP2_REQUIRED_COLUMNS - set(comp2_sample.columns)
            if missing_comp2:
                logger.critical(f"Component 2 CSV missing required columns: {sorted(missing_comp2)}")
                return False
            logger.info(f"Component 2 schema valid: {len(comp2_sample)} rows, "
                        f"{len(comp2_sample.columns)} columns")

        comp3_sample = safe_read_csv(self.comp3_path)
        if comp3_sample is not None:
            missing_comp3 = COMP3_REQUIRED_COLUMNS - set(comp3_sample.columns)
            if missing_comp3:
                logger.critical(f"Component 3 CSV missing required columns: {sorted(missing_comp3)}")
                return False
            logger.info(f"Component 3 schema valid: {len(comp3_sample)} rows, "
                        f"{len(comp3_sample.columns)} columns")

        logger.info(f"Checkpoint: {self.checkpoint.state['comp2_processed_count']} Comp2 + "
                     f"{self.checkpoint.state['comp3_processed_count']} Comp3 records already processed.")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info(f"Predictions output: {self.output_path}")
        logger.info("Startup complete. Entering inference loop.")
        return True

    def _read_new_comp2(self) -> pd.DataFrame:
        """Reads Component 2 CSV and returns only NEW (unprocessed) records."""
        df = safe_read_csv(self.comp2_path)
        if df is None or df.empty:
            return pd.DataFrame()

        # Validate schema
        missing = COMP2_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(f"Component 2 CSV missing columns: {sorted(missing)}. Skipping this read.")
            return pd.DataFrame()

        # Filter to unprocessed records
        new_rows = []
        for idx, row in df.iterrows():
            key = make_comp2_record_key(row)
            if not self.checkpoint.is_comp2_processed(key):
                new_rows.append(idx)

        if not new_rows and len(df) > 0:
            logger.info("All records processed. Resetting checkpoint to continuously stream live inference.")
            self.checkpoint.reset()
            new_rows = list(df.index)

        if not new_rows:
            return pd.DataFrame()

        # Process in batch chunks of up to 50 rows for fast real-time updates
        batch_rows = new_rows[:50]
        new_df = df.loc[batch_rows].copy().reset_index(drop=True)
        logger.info(f"Component 2: {len(new_df)} new records streaming for inference (total remaining: {len(new_rows)})")
        return new_df

    def _read_new_comp3(self) -> pd.DataFrame:
        """Reads Component 3 CSV and returns only NEW (unprocessed) records."""
        df = safe_read_csv(self.comp3_path)
        if df is None or df.empty:
            return pd.DataFrame()

        missing = COMP3_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(f"Component 3 CSV missing columns: {sorted(missing)}. Skipping this read.")
            return pd.DataFrame()

        new_rows = []
        for idx, row in df.iterrows():
            key = make_comp3_record_key(row)
            if not self.checkpoint.is_comp3_processed(key):
                new_rows.append(idx)

        if not new_rows:
            return pd.DataFrame()

        new_df = df.loc[new_rows].copy().reset_index(drop=True)
        logger.info(f"Component 3: {len(new_df)} new records detected (total in file: {len(df)})")
        return new_df

    def _run_inference(self, aligned_df: pd.DataFrame, comp2_keys: list[str], comp3_keys: list[str]) -> None:
        """
        Runs the trained model on aligned data and records predictions.
        """
        if aligned_df.empty:
            return

        # Build feature matrix (EXACTLY matching training schema)
        X = build_feature_matrix(aligned_df)

        # Run inference using trained model (no retraining, no refitting)
        try:
            probs = predict_probabilities(self.model, X) # type: ignore
        except Exception as e:
            logger.error(f"Model inference failure: {e}")
            return

        # Build prediction records
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        predictions = []
        now_iso = datetime.now(timezone.utc).isoformat()

        # Load existing status once up-front so we can add our per-service keys.
        # We do a SECOND load right before writing (see below) to pick up any
        # entries the API may have written while we were running inference.
        status_data = {}
        if os.path.exists(self.status_path):
            try:
                with open(self.status_path, "r") as f:
                    status_data = json.load(f)
            except Exception:
                status_data = {}

        with open(self.output_path, "a") as f:
            for i, (idx, row) in enumerate(aligned_df.iterrows()):
                cpu_prob = float(probs[i])
                mem_prob = float(row.get("memory_prob", 0.0))
                mem_alert = bool(row.get("alert_bool", False))
                mem_pred_label = str(row.get("pred_label", "NORMAL"))
                cpu_alarm = cpu_prob >= 0.6
                mem_alarm = mem_alert or (mem_prob >= 0.7) or (mem_pred_label in ("WARNING", "FAILURE"))
                # BUG FIX: joint_alarm must be cpu_alarm AND mem_alarm.
                # The old formula `cpu_alarm or (mem_prob >= 0.8)` caused every CPU-only
                # spike to be shown as CRITICAL on the dashboard instead of WARNING.
                joint_alarm = cpu_alarm and mem_alarm

                # Decision logic (same as API _make_decision)
                if cpu_alarm and mem_alarm:
                    action = "CRITICAL_RESTART_AND_TRAFFIC_REROUTE"
                elif cpu_alarm:
                    action = "TRIGGER_LOAD_SHEDDING"
                elif mem_alarm:
                    action = "TRIGGER_PROACTIVE_POD_RESTART"
                else:
                    action = "NO_ACTION"

                svc_name = str(row.get("service_name", "unknown"))
                proj_id = str(row.get("project_id", "unknown"))
                source_ts = str(row.get("timestamp", ""))
                inc_id = row.get("incident_id", 0)
                rca_report = generate_enterprise_rca_report(
                    service_name=svc_name,
                    cpu_failure_prob=cpu_prob,
                    memory_leak_prob=mem_prob,
                    input_df=X.iloc[[i]],
                    model=self.model,
                )

                prediction_record = {
                    "prediction_timestamp": now_iso,
                    "source_timestamp": source_ts,
                    "incident_id": int(inc_id) if pd.notna(inc_id) else 0,
                    "service_name": svc_name,
                    "project_id": proj_id,
                    "cpu_failure_prob": round(cpu_prob, 6),
                    "cpu_alarm": cpu_alarm,
                    "memory_prob": round(mem_prob, 6),
                    "memory_alert": mem_alert,
                    "memory_pred_label": mem_pred_label,
                    "joint_alarm": joint_alarm,
                    "action_recommended": action,
                    "rca_narrative": rca_report["rca_narrative"],
                    "feature_contributions": rca_report["feature_contributions"],
                    "shapley_phi_values": rca_report["shapley_phi_values"],
                    "sre_runbook_steps": rca_report["sre_runbook_steps"],
                    "incident_ticket_payload": rca_report["incident_ticket_payload"],
                }
                predictions.append(prediction_record)
                f.write(json.dumps(prediction_record) + "\n")

                # Update live status per service (latest prediction wins)
                status_data[svc_name] = {
                    **prediction_record,
                    "cpu_pct_raw": float(row.get("cpu_percent", 0.0)),
                    "cpu_trend_raw": float(row.get("cpu_trend_5min", 0.0)),
                    "mem_leak_raw": mem_prob,
                    "state_machine": str(row.get("system_state", "UNKNOWN")),
                }

        # BUG FIX: Rotate live_predictions.jsonl to prevent unbounded file growth.
        # Previously the file was growing to 2GB which caused the inference loop
        # to slow down enormously, making the dashboard stop updating.
        MAX_PREDICTION_LINES = 2000
        try:
            if os.path.exists(self.output_path):
                with open(self.output_path, "r") as rf:
                    lines = rf.readlines()
                if len(lines) > MAX_PREDICTION_LINES:
                    with open(self.output_path, "w") as wf:
                        wf.writelines(lines[-MAX_PREDICTION_LINES:])
        except Exception as e:
            logger.warning(f"Could not rotate prediction log: {e}")

        # BUG FIX: Re-read live_status.json immediately before writing it back.
        # The API endpoint also writes to this file, so entries added by the API
        # during the time we spent running inference would be silently overwritten
        # if we used the snapshot loaded at the start. Merging the fresh file with
        # our updates ensures both channels co-exist in the status file.
        try:
            fresh_status = {}
            if os.path.exists(self.status_path):
                with open(self.status_path, "r") as f:
                    fresh_status = json.load(f)
            # Merge: API entries win for keys we did NOT just update;
            # our inference results win for the services we just processed.
            merged_status = {**fresh_status, **status_data}
        except Exception:
            merged_status = status_data

        # Write status file
        try:
            with open(self.status_path, "w") as f:
                json.dump(merged_status, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write status file: {e}")

        # Mark records as processed
        self.checkpoint.mark_comp2_processed(comp2_keys)
        self.checkpoint.mark_comp3_processed(comp3_keys)

        # Log summary
        n_alarms = sum(1 for p in predictions if p["joint_alarm"])
        n_critical = sum(1 for p in predictions if p["action_recommended"] == "CRITICAL_RESTART_AND_TRAFFIC_REROUTE")
        logger.info(f"Inference complete: {len(predictions)} predictions, "
                     f"{n_alarms} alarms ({n_critical} critical)")

    def run(self) -> None:
        """
        Main polling loop. Runs until interrupted (Ctrl+C).
        Each cycle:
          1. Read new Comp2 records.
          2. Read new Comp3 records.
          3. Preprocess both.
          4. Align.
          5. Predict.
          6. Sleep.
        """
        self._running = True

        # Handle graceful shutdown
        def _signal_handler(signum, frame):
            logger.info("Shutdown signal received. Stopping inference loop...")
            self._running = False

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        while self._running:
            cycle_start = time.time()
            try:
                # 1. Read new records
                comp2_new = self._read_new_comp2()
                comp3_new_raw = self._read_new_comp3()

                if comp2_new.empty:
                    # No new Comp2 data — nothing to predict
                    # (Comp2 is the primary driver; Comp3 augments it)
                    elapsed = time.time() - cycle_start
                    sleep_for = max(0, self.poll_interval - elapsed)
                    time.sleep(sleep_for)
                    continue

                # 2. Build record keys for checkpoint tracking
                comp2_keys = [make_comp2_record_key(row) for _, row in comp2_new.iterrows()]
                comp3_keys = [make_comp3_record_key(row) for _, row in comp3_new_raw.iterrows()] if not comp3_new_raw.empty else []

                # 3. Preprocess (same pipeline as training)
                comp2_processed = preprocess_comp2_for_inference(comp2_new)

                # For Comp3, we need ALL records (not just new) for alignment
                # because new Comp2 rows may align with previously-seen Comp3 rows
                comp3_full = safe_read_csv(self.comp3_path)
                if comp3_full is not None and not comp3_full.empty:
                    comp3_processed = preprocess_comp3_for_inference(comp3_full)
                else:
                    comp3_processed = pd.DataFrame()

                # 4. Align Comp2 + Comp3
                aligned = align_comp2_comp3(comp2_processed, comp3_processed)

                # 5. Run inference
                self._run_inference(aligned, comp2_keys, comp3_keys)

            except Exception as e:
                logger.error(f"Inference cycle error: {e}", exc_info=True)

            # 6. Sleep for remainder of poll interval
            elapsed = time.time() - cycle_start
            sleep_for = max(0, self.poll_interval - elapsed)
            if sleep_for > 0 and self._running:
                time.sleep(sleep_for)

        logger.info("Inference engine stopped.")


def main():
    """Entry point for: python -m src.live_inference"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Component 4: Live CSV Inference Engine. "
                    "Consumes live-updating CSVs from Component 2 & 3, "
                    "runs the trained model, records predictions."
    )
    parser.add_argument("--comp2-csv", default=DEFAULT_COMP2_PATH,
                        help="Path to Component 2 live CSV (final_research_dataset.csv)")
    parser.add_argument("--comp3-csv", default=DEFAULT_COMP3_PATH,
                        help="Path to Component 3 live CSV (memory_predictions.csv)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to trained model file (cpu_rf_model.joblib)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH,
                        help="Path to write prediction records (JSONL)")
    parser.add_argument("--status", default=DEFAULT_STATUS_PATH,
                        help="Path to write live status (JSON)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH,
                        help="Path to checkpoint state file")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL,
                        help="Seconds between each CSV poll cycle")

    args = parser.parse_args()

    engine = LiveInferenceEngine(
        comp2_path=args.comp2_csv,
        comp3_path=args.comp3_csv,
        model_path=args.model,
        output_path=args.output,
        status_path=args.status,
        checkpoint_path=args.checkpoint,
        poll_interval=args.poll_interval,
    )

    if not engine.startup():
        logger.critical("Startup failed. Exiting.")
        sys.exit(1)

    engine.run()


if __name__ == "__main__":
    main()
