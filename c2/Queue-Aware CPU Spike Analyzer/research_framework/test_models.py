import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "test"))
from feature_sets_test import evaluate_5_models
import pandas as pd

# Test on empirical validation dataset
print("=" * 80)
print("TESTING ON EMPIRICAL VALIDATION DATASET")
print("=" * 80)
df_emp = pd.read_csv('empirical_validation_dataset.csv')
print(f"Shape: {df_emp.shape}")
print(f"Failure status dist: {df_emp['failure_status'].value_counts().to_dict()}")
print(f"QPI range: {df_emp['qpi'].min()} to {df_emp['qpi'].max()}")
print(f"QPI unique: {df_emp['qpi'].nunique()}")

# Need to add actual_failure column for the test
if 'actual_failure' not in df_emp.columns:
    df_emp['actual_failure'] = (df_emp['failure_status'] > 0).astype(int)

evaluate_5_models(df_emp)

print("\n\n" + "=" * 80)
print("TESTING ON FINAL RESEARCH DATASET")
print("=" * 80)
df_final = pd.read_csv('final_research_dataset.csv', on_bad_lines='skip', low_memory=False)
print(f"Shape: {df_final.shape}")
print(f"Actual failure dist: {df_final['actual_failure'].value_counts().to_dict()}")
print(f"QPI range: {df_final['qpi'].min()} to {df_final['qpi'].max()}")
print(f"QPI unique: {df_final['qpi'].nunique()}")
print(f"Queue range: {df_final['in_flight_queue'].min()} to {df_final['in_flight_queue'].max()}")

# The ml_label is all NaN, so we need to use actual_failure
evaluate_5_models(df_final)