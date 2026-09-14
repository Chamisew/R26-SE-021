"""
QPI Ablation Study & Advanced Correlation Suite
Commit 1: feat: dataset loading and candidate QPI formulation variants

Adds:
  - Dataset loading and fallback generator
  - Dynamic frozen Q_ref derivation from healthy baseline
  - Candidate formulations:
      1. QPI-1 (Queue Only): min(1.0, Q / Q_ref)
      2. QPI-2 (Rate Only): min(1.0, lambda / mu)
      3. QPI-3 (Multiplicative Proposed): min(1.0, (Q / Q_ref) * (lambda / mu))
      4. QPI-4 (Additive Weighted): min(1.0, 0.5 * (Q / Q_ref) + 0.5 * min(1.0, lambda / mu))
  - Queue growth rate dQ/dt calculation via np.gradient
"""

import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "empirical_validation_dataset.csv")


def load_or_generate_dataset():
    """Load existing empirical validation dataset or generate if missing."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
    else:
        from empirical_validation import generate_empirical_workload
        df = generate_empirical_workload()
        df.to_csv(DATASET_PATH, index=False)
    return df


def compute_qpi_variants(df):
    """
    Computes candidate QPI formulations for ablation comparison using
    service-specific frozen Q_ref derived from P95 of Healthy Baseline telemetry.
    """
    q = df['in_flight_queue']
    inc = df['incoming_rate']
    proc = np.maximum(df['processing_rate'], 0.001)

    # Deriving Q_ref from 95th Percentile of Healthy Baseline Data (Low Load)
    if 'q_ref_frozen' in df.columns:
        q_ref = df['q_ref_frozen']
    elif 'workload_stage' in df.columns:
        healthy_queues = df[df['workload_stage'] == 'Low Load']['in_flight_queue']
        q_ref = max(1.0, float(np.percentile(healthy_queues, 95)))
    else:
        q_ref = max(1.0, float(np.percentile(q.iloc[:max(10, int(len(df) * 0.2))], 95)))

    # 1. QPI-1: Queue Only
    qpi_queue = np.minimum(1.0, q / q_ref)

    # 2. QPI-2: Rate Imbalance Only (lambda / mu)
    qpi_rate = np.minimum(1.0, inc / proc)

    # 3. QPI-3: Proposed Multiplicative
    qpi_mult = np.minimum(1.0, (q / q_ref) * (inc / proc))

    # 4. QPI-4: Additive Weighted (0.5 Queue + 0.5 Rate)
    qpi_add = np.minimum(1.0, 0.5 * (q / q_ref) + 0.5 * np.minimum(1.0, inc / proc))

    # Compute Queue Growth Rate dQ/dt
    q_growth = np.gradient(q, df['time_sec'])

    df_var = df.copy()
    df_var['qpi_queue'] = qpi_queue
    df_var['qpi_rate'] = qpi_rate
    df_var['qpi_mult'] = qpi_mult
    df_var['qpi_add'] = qpi_add
    df_var['queue_growth'] = q_growth

    return df_var


if __name__ == "__main__":
    df = load_or_generate_dataset()
    df_var = compute_qpi_variants(df)
    print(f"Loaded {len(df_var)} rows. Computed QPI variants: qpi_queue, qpi_rate, qpi_mult, qpi_add.")
