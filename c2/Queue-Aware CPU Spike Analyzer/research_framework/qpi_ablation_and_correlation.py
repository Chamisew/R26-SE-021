"""
QPI Ablation Study & Advanced Correlation Suite
Commit 2: feat: operational pressure correlation suite

Adds on top of commit 1:
  - `analyze_correlations(df)`:
      • Spearman and Pearson log correlations for QPI vs Response Latency
      • Correlation between QPI and Queue Growth Rate (dQ/dt)
      • Correlation between QPI and Error Rate
      • Predictive Power Comparison: QPI vs Failure Status vs CPU vs Failure Status
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "empirical_validation_dataset.csv")


def load_or_generate_dataset():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
    else:
        from empirical_validation import generate_empirical_workload
        df = generate_empirical_workload()
        df.to_csv(DATASET_PATH, index=False)
    return df


def compute_qpi_variants(df):
    q = df['in_flight_queue']
    inc = df['incoming_rate']
    proc = np.maximum(df['processing_rate'], 0.001)

    if 'q_ref_frozen' in df.columns:
        q_ref = df['q_ref_frozen']
    elif 'workload_stage' in df.columns:
        healthy_queues = df[df['workload_stage'] == 'Low Load']['in_flight_queue']
        q_ref = max(1.0, float(np.percentile(healthy_queues, 95)))
    else:
        q_ref = max(1.0, float(np.percentile(q.iloc[:max(10, int(len(df) * 0.2))], 95)))

    qpi_queue = np.minimum(1.0, q / q_ref)
    qpi_rate = np.minimum(1.0, inc / proc)
    qpi_mult = np.minimum(1.0, (q / q_ref) * (inc / proc))
    qpi_add = np.minimum(1.0, 0.5 * (q / q_ref) + 0.5 * np.minimum(1.0, inc / proc))

    q_growth = np.gradient(q, df['time_sec'])

    df_var = df.copy()
    df_var['qpi_queue'] = qpi_queue
    df_var['qpi_rate'] = qpi_rate
    df_var['qpi_mult'] = qpi_mult
    df_var['qpi_add'] = qpi_add
    df_var['queue_growth'] = q_growth

    return df_var


# =============================================================================
# OPERATIONAL CORRELATIONS  ← NEW
# =============================================================================

def analyze_correlations(df):
    """
    Calculates exact correlation coefficients proving operational pressure capture:
    - QPI vs Response Latency
    - QPI vs Queue Growth (dQ/dt)
    - QPI vs Error Rate
    - QPI vs Failure Status (compared against CPU vs Failure Status)
    """
    print("\n" + "=" * 80)
    print("      OPERATIONAL PRESSURE CORRELATION SUITE (QPI vs METRICS)")
    print("=" * 80)

    # Use 'qpi' column if present, else fallback to proposed multiplicative 'qpi_mult'
    target_qpi = df['qpi'] if 'qpi' in df.columns else df['qpi_mult']

    # Compute Spearman (rank) and Pearson correlations
    r_lat_spearman = target_qpi.corr(df['response_time_ms'], method='spearman')
    r_lat_pearson  = target_qpi.corr(np.log1p(df['response_time_ms']))
    r_growth       = target_qpi.corr(df['queue_growth'])
    r_errors       = target_qpi.corr(df['interval_errors'])

    # CPU vs Failure Status vs QPI vs Failure Status
    r_qpi_failure = target_qpi.corr(df['failure_status'])
    r_cpu_failure = df['cpu_percent'].corr(df['failure_status'])

    print("-" * 80)
    print(f"1. QPI <--> Response Latency      : Spearman r_s = {r_lat_spearman:.4f} | Pearson log(r) = {r_lat_pearson:.4f}")
    print(f"2. QPI <--> Queue Growth (dQ/dt)   : Pearson r   = {r_growth:.4f}")
    print(f"3. QPI <--> Error Rate            : Pearson r   = {r_errors:.4f}")
    print("-" * 80)
    print("4. PREDICTIVE POWER COMPARISON (Failure Status Correlation):")
    print(f"   - QPI <--> Failure Status       : r = {r_qpi_failure:.4f}  [EARLY SYSTEMIC PROXY]")
    print(f"   - CPU <--> Failure Status       : r = {r_cpu_failure:.4f}  [REACTIVE RESOURCE PROXY]")

    if r_qpi_failure > r_cpu_failure:
        print("   [+] PROOF: QPI demonstrates HIGHER correlation with failure status than CPU alone!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    df = load_or_generate_dataset()
    df_var = compute_qpi_variants(df)
    analyze_correlations(df_var)
