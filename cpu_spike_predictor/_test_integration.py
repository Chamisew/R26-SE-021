import sys
sys.path.insert(0, '.')
from src.ingestion import load_memory_predictions, DEFAULT_COMP3_CSV
import os

print(f'DEFAULT_COMP3_CSV path: {DEFAULT_COMP3_CSV}')
print(f'File exists: {os.path.exists(DEFAULT_COMP3_CSV)}')

if os.path.exists(DEFAULT_COMP3_CSV):
    df = load_memory_predictions(DEFAULT_COMP3_CSV)
    print(f'Loaded successfully: {len(df)} rows')
    print(f'Columns: {list(df.columns)}')
    alerts = df['alert'].head(10).tolist()
    print(f'Sample alerts: {alerts}')
    alert_bools = df['alert_bool'].head(10).tolist()
    print(f'Sample alert_bool: {alert_bools}')
    mem_min = df['memory_prob'].min()
    mem_max = df['memory_prob'].max()
    print(f'memory_prob range: [{mem_min:.4f}, {mem_max:.4f}]')
    print('SUCCESS: Integration path works correctly!')
else:
    print('ERROR: File not found at the expected path')
