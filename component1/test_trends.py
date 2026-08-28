import pandas as pd
from pipeline import add_failure_trends

print("Loading data...")
df = pd.read_csv('output/ml_ready_dataset.csv')
print("Original shape:", df.shape)

if 'hybrid_label' not in df.columns:
    df['hybrid_label'] = df['label'] if 'label' in df.columns else 'NORMAL'

print("Calculating trends...")
out_df = add_failure_trends(df)

print("Columns added:")
for col in ['trend_slope_5m', 'trend_max_failures_5m', 'trend_variance_5m',
            'trend_slope_10m', 'trend_max_failures_10m', 'trend_variance_10m']:
    print(f"{col}: mean={out_df[col].mean():.4f}, max={out_df[col].max():.4f}")

print("Test complete.")
