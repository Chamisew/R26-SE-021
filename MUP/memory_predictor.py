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

# Validate schema
missing = [c for c in FEATURE_COLS + [TARGET_COL, PROJECT_COL] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns from component1.csv: {missing}")

# Remove NaN rows
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

# Encode label: FAILURE=1, NORMAL=0
df["label_encoded"] = (df[TARGET_COL] == "FAILURE").astype(int)

print(f"  Total rows loaded : {len(df)}")
print(f"  Projects found    : {sorted(df[PROJECT_COL].unique())}")
print(f"  Class distribution:\n{df[TARGET_COL].value_counts().to_string()}")
print(f"  NaN rows dropped  : {df.isnull().sum().sum()}")
