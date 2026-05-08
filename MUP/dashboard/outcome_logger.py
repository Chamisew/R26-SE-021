"""
Component 3 -- Outcome Logger
Appends live predictions to memory_predictions.csv for Component 4.
"""
import os, logging
import pandas as pd

logger = logging.getLogger(__name__)

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(BASE, "data", "memory_predictions.csv")

COLS = ["timestamp", "service_name", "project_id", "memory_prob", "alert", "pred_label"]

def log_predictions(df: pd.DataFrame):
    """Append predicted rows to memory_predictions.csv."""
    try:
        rows = df[COLS] if all(c in df.columns for c in COLS) else df
        mode   = "a" if os.path.exists(OUT_CSV) else "w"
        header = not os.path.exists(OUT_CSV)
        rows.to_csv(OUT_CSV, index=False, mode=mode, header=header)
    except Exception as e:
        logger.warning(f"[outcome_logger] Failed to write predictions: {e}")
