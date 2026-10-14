"""Quick smoke test for the new feature_sets_test.py with a tiny synthetic dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'test'))

from feature_sets_test import SMOTE_AVAILABLE, SKLEARN_AVAILABLE, evaluate_5_models
print('sklearn available:', SKLEARN_AVAILABLE)
print('SMOTE   available:', SMOTE_AVAILABLE)

import numpy as np, pandas as pd

np.random.seed(42)
n = 500
t = np.arange(n, dtype=float)
cpu = 20 + 5 * np.random.randn(n)
q   = np.random.randint(0, 3, n).astype(float)
label = np.zeros(n)
label[160:200] = 1   # two injected PRE_FAILURE windows
label[360:400] = 1

df = pd.DataFrame({
    'timestamp':            t,
    'cpu_percent':          cpu,
    'in_flight_queue':      q,
    'incoming_rate':        10 + np.random.randn(n),
    'processing_rate':      10 + np.random.randn(n),
    'qpi':                  np.random.rand(n) * 0.3,
    'queue_growth_rate':    np.random.randn(n) * 0.1,
    'processing_efficiency':0.9 + np.random.randn(n) * 0.05,
    'qref':                 np.full(n, 5.0),
    'ml_label':             label,
})
df['baseline_tier'] = 'STATISTICAL'

results = evaluate_5_models(df)
print()
print(results[['model_name', 'precision', 'recall', 'f1_score',
               'false_alarms_per_hour', 'accuracy']].to_string(index=False))
print('\nSmoke test PASSED.')
