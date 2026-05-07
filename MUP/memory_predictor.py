import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import shap
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------
# STAGE 1 - Data Ingestion & Validation
# ---------------------------------------------
print("=" * 60)
print("STAGE 1: Data Ingestion & Validation")
print("=" * 60)

df = pd.read_csv("component1.csv")

FEATURE_COLS = [
    "memory_mb",
    "memory_growth_5min",
    "memory_growth_10min",
    "acceleration",
    "heap_rate",
    "gc_spike_count"
]
TARGET_COL = "label"
PROJECT_COL = "project_id"
