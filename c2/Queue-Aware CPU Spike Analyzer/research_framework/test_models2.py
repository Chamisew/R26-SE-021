import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "test"))
from feature_sets_test import evaluate_5_models
import pandas as pd

# Test on pre_schema dataset
print("=" * 80)
print("TESTING ON PRE_SCHEMA DATASET")
print("=" * 80)
df = pd.read_csv('final_research_dataset.pre_schema_20261009_001347.csv', on_bad_lines='skip', low_memory=False)
print(f"Shape: {df.shape}")
print(f"Actual failure dist: {df['actual_failure'].value_counts().to_dict()}")
print(f"QPI range: {df['qpi'].min()} to {df['qpi'].max()}")
print(f"QPI unique: {df['qpi'].nunique()}")
print(f"Queue range: {df['in_flight_queue'].min()} to {df['in_flight_queue'].max()}")

evaluate_5_models(df)