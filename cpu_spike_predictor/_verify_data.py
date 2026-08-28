import sys
sys.path.insert(0, '.')
from src.ingestion import load_cpu_features, load_memory_predictions, DEFAULT_COMP2_CSV, DEFAULT_COMP3_CSV
import os

print('Component 2 path:', DEFAULT_COMP2_CSV)
print('  Exists:', os.path.exists(DEFAULT_COMP2_CSV))
print()
print('Component 3 path:', DEFAULT_COMP3_CSV)
print('  Exists:', os.path.exists(DEFAULT_COMP3_CSV))
print()

cpu_df = load_cpu_features(DEFAULT_COMP2_CSV)
print('Component 2 loaded OK:')
print('  Rows:', len(cpu_df))
print('  Failure rows:', cpu_df['label'].sum())
print('  Projects:', sorted(cpu_df['project_id'].unique()))
print()

mem_df = load_memory_predictions(DEFAULT_COMP3_CSV)
print('Component 3 loaded OK:')
print('  Rows:', len(mem_df))
print('  Memory alerts:', mem_df['alert_bool'].sum())
print('  Services:', sorted(mem_df['service_name'].unique()))
print()
print('Both datasets ready.')
