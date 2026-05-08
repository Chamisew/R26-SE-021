"""
Component 3 -- Metric Adapter
Reads the live_feed.csv written by simulate_live.py.
"""
import os, logging
import pandas as pd

logger = logging.getLogger(__name__)

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_CSV = os.path.join(BASE, "data", "live_feed.csv")

def get_live_rows(n=50) -> pd.DataFrame:
    """Return the latest n rows from the live feed."""
    if not os.path.exists(LIVE_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LIVE_CSV)
        return df.tail(n).reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[metric_adapter] Could not read live_feed.csv: {e}")
        return pd.DataFrame()
