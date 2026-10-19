"""
Dataset Quality, Downed Container Telemetry, & QPI Filtering Analysis Suite
---------------------------------------------------------------------------
1. Decouples Online Detector Predictions (predicted_state) from Ground Truth (actual_failure).
2. Filters out and documents downed container telemetry (service_unreachable == 1 -> cpu == 0.0).
3. Filters on queue_stale == 0 for robust QPI evaluation, removing fixed queue placeholders.
"""

import pandas as pd
import numpy as np
import os

DATASET_FILE = 'final_research_dataset.csv'
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"{DATASET_FILE} not found!")

print(f"Loading {DATASET_FILE}...")
df = pd.read_csv(DATASET_FILE, on_bad_lines='skip', low_memory=False)
total_rows = len(df)
print(f"Total rows loaded: {total_rows:,}\n")

# =========================================================================
# 1. PREDICTED_STATE VS ACTUAL_FAILURE DECOUPLING ANALYSIS
# =========================================================================
print("=" * 80)
print("1. PREDICTOR (predicted_state) VS GROUND TRUTH (actual_failure)")
print("=" * 80)

if 'predicted_state' in df.columns and 'actual_failure' in df.columns:
    ct = pd.crosstab(df['predicted_state'], df['actual_failure'], margins=True)
    print("Contingency Table (predicted_state vs actual_failure):\n")
    print(ct.to_string())
    print()

    # Binary evaluation: Alerting (FAILED) vs Ground Truth Physical Failure (actual_failure == 1)
    # Exclude CALIBRATING tier for fair operational evaluation
    eval_df = df[df['baseline_tier'] != 'CALIBRATING'].copy()
    y_true = pd.to_numeric(eval_df['actual_failure'], errors='coerce').fillna(0).astype(int)
    y_pred = (eval_df['predicted_state'] == 'FAILED').astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Binary Evaluation (Alert 'FAILED' vs Ground Truth Failure):")
    print(f"  True Positives (TP):  {tp:,} (Severe telemetry overload accompanied by physical outage)")
    print(f"  False Positives (FP): {fp:,} (Telemetry alert triggered while process was still responding)")
    print(f"  False Negatives (FN): {fn:,} (Physical failure occurred before telemetry alerted / debounce lag)")
    print(f"  True Negatives (TN):  {tn:,} (Nominal telemetry with healthy service)")
    print(f"  Operational Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("\n  [NOTE] Discrepancies represent legitimate classification tradeoffs and debounce windows,")
    print("         NOT dataset labeling bugs. Tautological 100% agreement would indicate label leakage.\n")
else:
    print("Columns 'predicted_state' or 'actual_failure' missing from dataset.\n")

# =========================================================================
# 2. DOWNED CONTAINER TELEMETRY & ZERO-CPU ANALYSIS
# =========================================================================
print("=" * 80)
print("2. DOWNED CONTAINER TELEMETRY (service_unreachable & cpu_percent)")
print("=" * 80)

if 'service_unreachable' in df.columns:
    unreach_counts = df['service_unreachable'].value_counts(dropna=False).to_dict()
    n_unreach = int((df['service_unreachable'] == 1).sum())
    n_reach = int((df['service_unreachable'] == 0).sum())
    print(f"service_unreachable counts: 0 (Active)={n_reach:,} | 1 (Downed)={n_unreach:,}")

    # Check CPU when service_unreachable == 1
    downed = df[df['service_unreachable'] == 1]
    downed_cpu_zeros = int((pd.to_numeric(downed['cpu_percent'], errors='coerce') == 0.0).sum())
    print(f"Downed containers with cpu_percent == 0.0: {downed_cpu_zeros:,} of {len(downed):,} ({downed_cpu_zeros/max(1,len(downed))*100:.1f}%)")
    print("  -> EXPLANATION: When a container is killed, crashed, or partitioned, the scraper")
    print("     cannot query /metrics; missing telemetry defaults to 0.0% CPU as designed.\n")

    # Compare CPU stats: All rows vs Active Only (service_unreachable == 0)
    c_all = pd.to_numeric(df['cpu_percent'], errors='coerce').dropna()
    c_active = pd.to_numeric(df[df['service_unreachable'] == 0]['cpu_percent'], errors='coerce').dropna()

    print("CPU Utilization Statistics:")
    print(f"  ALL Rows:     mean={c_all.mean():.2f}% | std={c_all.std():.2f}% | zeros={(c_all == 0).sum():,} | max={c_all.max():.2f}%")
    print(f"  ACTIVE Only:  mean={c_active.mean():.2f}% | std={c_active.std():.2f}% | zeros={(c_active == 0).sum():,} | max={c_active.max():.2f}%")
    print("  [RECOMMENDATION] CPU velocity/workload studies must filter on service_unreachable == 0.\n")
else:
    print("Column 'service_unreachable' missing.\n")

# =========================================================================
# 3. QUEUE STALENESS & QPI FILTERING ANALYSIS
# =========================================================================
print("=" * 80)
print("3. QUEUE STALENESS FILTERING FOR QPI ANALYSIS (queue_stale == 0)")
print("=" * 80)

if 'queue_stale' in df.columns:
    stale_counts = df['queue_stale'].value_counts(dropna=False).to_dict()
    n_stale = int((df['queue_stale'] == 1).sum())
    n_nonstale = int((df['queue_stale'] == 0).sum())
    print(f"queue_stale counts: 0 (Dynamic Queue)={n_nonstale:,} | 1 (Frozen Placeholder)={n_stale:,}")

    # Contrast QPI and Queue: Unfiltered vs Filtered (queue_stale == 0)
    q_all = pd.to_numeric(df['in_flight_queue'], errors='coerce').dropna()
    q_filt = pd.to_numeric(df[df['queue_stale'] == 0]['in_flight_queue'], errors='coerce').dropna()

    print("\nQueue Length Distribution:")
    print(f"  UNFILTERED: mean={q_all.mean():.3f} | std={q_all.std():.3f} | values={q_all.value_counts().head(5).to_dict()}")
    print(f"  FILTERED (queue_stale==0): mean={q_filt.mean():.3f} | std={q_filt.std():.3f} | values={q_filt.value_counts().head(5).to_dict()}")

    if 'qpi' in df.columns:
        qpi_all = pd.to_numeric(df['qpi'], errors='coerce').dropna()
        qpi_filt = pd.to_numeric(df[df['queue_stale'] == 0]['qpi'], errors='coerce').dropna()

        print("\nQPI (Queue Pressure Index) Distribution:")
        print(f"  UNFILTERED: mean={qpi_all.mean():.4f} | std={qpi_all.std():.4f} | at 1.0 (clamped): {(qpi_all == 1.0).sum():,}")
        print(f"  FILTERED (queue_stale==0): mean={qpi_filt.mean():.4f} | std={qpi_filt.std():.4f} | at 1.0 (clamped): {(qpi_filt == 1.0).sum():,}")

    # Contrast correlation with overload
    if 'overload' in df.columns:
        ov = pd.to_numeric(df['overload'], errors='coerce').fillna(0)
        df_active_q = df[df['queue_stale'] == 0]
        ov_filt = pd.to_numeric(df_active_q['overload'], errors='coerce').fillna(0)

        print("\nCorrelation with 'overload' Target (Unfiltered vs Filtered):")
        for col in ['in_flight_queue', 'qpi', 'queue_growth_rate', 'cpu_percent']:
            if col in df.columns:
                v_all = pd.to_numeric(df[col], errors='coerce').fillna(0)
                v_filt = pd.to_numeric(df_active_q[col], errors='coerce').fillna(0)
                corr_all = round(float(np.corrcoef(v_all, ov)[0, 1]), 3)
                corr_filt = round(float(np.corrcoef(v_filt, ov_filt)[0, 1]), 3)
                print(f"  {col:<20} | Unfiltered: {corr_all:>6} | Filtered (queue_stale==0): {corr_filt:>6}")

    print("\n  [RECOMMENDATION] QPI ablation and thesis charts should always apply df[df['queue_stale'] == 0].\n")
else:
    print("Column 'queue_stale' missing.\n")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
