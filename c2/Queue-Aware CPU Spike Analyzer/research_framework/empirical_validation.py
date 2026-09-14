"""
Empirical Validation Suite: Comparing QPI with Actual System Behavior

Adds:
  - Configuration constants (SCRIPT_DIR, DATASET_PATH, etc.)
  - `generate_empirical_workload(duration_sec=600, seed=42)`:
      • Multi-stage workload cycles (Low Load -> Moderate Load -> High Load -> Overload)
      • Arrival rate (λ) and service capacity (μ) dynamics
      • Queue growth dynamics: dQ/dt = λ - μ with M/M/1 pressure
      • Dynamic CPU contention and response latency explosion
      • Stochastic error sampling
      • Baseline calibration of healthy Q_ref via P95
      • Authoritative calibrated QPI calculation
      • Machine learning derived features (z-scores, efficiency, growth rate, failure labels)
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd

# Default output file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "empirical_validation_dataset.csv")
FINAL_DATASET_PATH = os.path.join(SCRIPT_DIR, "final_research_dataset.csv")
PLOT_PATH = os.path.join(SCRIPT_DIR, "qpi_empirical_validation.png")


def generate_empirical_workload(duration_sec=600, seed=42):
    """
    Generates a multi-cycle time-series simulation of microservice behavior
    under ramping workload conditions:
      Stage 1 (0-60s):   Low load      (λ = 10, μ = 50, CPU ~ 15%, Queue ~ 0)
      Stage 2 (60-120s): Moderate load (λ = 30, μ = 50, CPU ~ 45%, Queue ~ 2)
      Stage 3 (120-210s): High load    (λ = 45 -> 55, μ = 50, CPU ~ 70%, Queue ~ 10 -> 30, QPI rising)
      Stage 4 (210-300s): Overload     (λ = 85, μ = 35 [degraded], CPU ~ 98%, Queue > 50, Latency/Errors spike)
      Cycle 2 repeats for out-of-sample machine learning validation.
    """
    np.random.seed(seed)
    records = []

    t_start = 1700000000.0
    queue = 0.0
    total_requests = 0
    total_errors = 0
    cycle_len = 300

    for t in range(duration_sec):
        timestamp = t_start + t
        t_phase = t % cycle_len
        if t_phase == 0:
            queue = 0.0  # Reset queue depth at cycle boundary

        # 1. Determine Workload Stage & Arrival Rate (λ)
        if t_phase < 60:
            stage = "Low Load"
            lambda_base = 10.0
            mu_capacity = 50.0
            base_cpu = 15.0
            base_latency = 12.0
            error_prob = 0.0
        elif t_phase < 120:
            stage = "Moderate Load"
            lambda_base = 30.0
            mu_capacity = 50.0
            base_cpu = 45.0
            base_latency = 25.0
            error_prob = 0.001
        elif t_phase < 210:
            stage = "High Load"
            # Ramp λ towards and slightly past saturation point (40 -> 55 req/s)
            ramp_factor = (t_phase - 120) / 90.0
            lambda_base = 40.0 + ramp_factor * 15.0  # 40 -> 55 req/s
            mu_capacity = 50.0
            base_cpu = 55.0 + ramp_factor * 25.0      # 55 -> 80% CPU
            base_latency = 35.0 + (ramp_factor ** 2) * 120.0  # Latency starts curving up
            error_prob = 0.005 * ramp_factor
        else:
            stage = "Overload"
            ramp_factor = (t_phase - 210) / 90.0
            lambda_base = 55.0 + ramp_factor * 35.0   # 55 -> 90 req/s
            # Thermal/thread contention degrades service capacity μ
            mu_capacity = 50.0 - ramp_factor * 18.0   # 50 -> 32 req/s
            base_cpu = min(99.5, 80.0 + ramp_factor * 20.0)
            base_latency = 150.0 + (ramp_factor ** 2.5) * 800.0  # Latency explodes
            error_prob = 0.02 + 0.15 * ramp_factor

        # Add natural stochastic noise
        inc_rate = max(0.0, lambda_base + np.random.normal(0, 1.2))
        proc_rate_max = max(1.0, mu_capacity + np.random.normal(0, 1.0))

        # 2. Queue Dynamics (dQ/dt = λ - μ with M/M/1 queuing pressure near saturation)
        rho = min(0.98, inc_rate / proc_rate_max)
        queue_expected = (rho ** 2) / max(1.0 - rho, 0.02) if rho > 0.5 else 0.0

        diff = inc_rate - proc_rate_max
        if diff > 0:
            queue += diff * 1.0
        else:
            # Drain towards equilibrium expected queue
            if queue > queue_expected:
                queue -= min(queue - queue_expected, abs(diff) * 1.0)
            else:
                queue += (queue_expected - queue) * 0.2

        queue = max(0.0, queue)
        actual_proc_rate = min(inc_rate + queue, proc_rate_max)

        # 3. CPU Dynamics
        cpu_noise = np.random.normal(0, 1.2)
        cpu_percent = min(100.0, max(5.0, base_cpu + (queue / 50.0) * 10.0 + cpu_noise))

        # 4. Latency & Error Metrics
        latency_ms = max(5.0, base_latency + (queue / 4.0) ** 1.6 + np.random.exponential(5))
        new_reqs = int(inc_rate)
        total_requests += new_reqs
        new_errors = int(np.random.binomial(new_reqs, min(1.0, error_prob))) if new_reqs > 0 else 0
        total_errors += new_errors

        # 5. QPI placeholder
        qpi = 0.0
        cpu_pi = min(1.0, cpu_percent / 100.0)

        # 6. Failure Status Labeling
        # 0 = NORMAL, 1 = PRE_FAILURE (QPI high, latency curving up, before full crash), 2 = DEGRADED/FAILED
        if t_phase < 150:
            failure_status = 0
        elif t_phase < 210:
            failure_status = 1  # Pre-Failure window
        else:
            failure_status = 2  # Severe degradation / Overload

        records.append({
            "timestamp": timestamp,
            "time_sec": t,
            "workload_stage": stage,
            "cpu_percent": round(cpu_percent, 2),
            "in_flight_queue": round(queue, 2),
            "incoming_rate": round(inc_rate, 2),
            "processing_rate": round(actual_proc_rate, 2),
            "qpi": round(qpi, 4),
            "cpu_pi": round(cpu_pi, 4),
            "response_time_ms": round(latency_ms, 2),
            "errors": total_errors,
            "interval_errors": new_errors,
            "requests": total_requests,
            "failure_status": failure_status,
            "overload_flag": 1 if inc_rate > actual_proc_rate else 0
        })

    df = pd.DataFrame(records)

    # ── Step 2: Calibrate Qref from P95 of Healthy Baseline (Low Load stage only) ──────────────
    try:
        from queue_analyzer import calibrate_healthy_qref
        qref_map = calibrate_healthy_qref(df, percentile=95, print_report=True)
        qref_val = qref_map if isinstance(qref_map, (int, float)) else list(qref_map.values())[0]
    except ImportError:
        healthy_queues = df[df['workload_stage'] == 'Low Load']['in_flight_queue']
        qref_val = max(1.0, float(np.percentile(healthy_queues, 95)))

    # Recompute QPI across the full dataset using the frozen calibrated Qref.
    epsilon = 0.001
    arrival_ratio = df['incoming_rate'] / np.maximum(df['processing_rate'], epsilon)
    df['qref'] = qref_val
    df['q_ref_frozen'] = qref_val
    df['queue_pressure_index'] = np.minimum(
        1.0, (df['in_flight_queue'] / qref_val) * arrival_ratio
    ).round(4)
    df['qpi'] = df['queue_pressure_index']   # authoritative, calibrated QPI

    # Derive complete ML feature representation
    df['latency_ms'] = df['response_time_ms']
    df['queue_growth_rate'] = (df['incoming_rate'] - df['processing_rate']).round(4)
    df['processing_efficiency'] = np.minimum(1.0, df['processing_rate'] / np.maximum(df['incoming_rate'], epsilon)).round(4)

    low_mask = df['workload_stage'] == 'Low Load'
    cpu_mean = df.loc[low_mask, 'cpu_percent'].mean()
    cpu_std = max(0.001, df.loc[low_mask, 'cpu_percent'].std())
    qpi_mean = df.loc[low_mask, 'qpi'].mean()
    qpi_std = max(0.001, df.loc[low_mask, 'qpi'].std())

    df['cpu_zscore'] = ((df['cpu_percent'] - cpu_mean) / cpu_std).round(4)
    df['qpi_zscore'] = ((df['qpi'] - qpi_mean) / qpi_std).round(4)
    df['actual_failure'] = (df['failure_status'] == 2).astype(int)

    try:
        from queue_analyzer import label_pre_failure
        df = label_pre_failure(df, lookahead_sec=30)
    except ImportError:
        # Fallback pre-failure lookahead labeler
        df['ml_label'] = ((df['failure_status'] == 1) | (df['failure_status'] == 2)).astype(int)

    return df


if __name__ == "__main__":
    df = generate_empirical_workload()
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated {len(df)} telemetry rows. Saved to {DATASET_PATH}")
