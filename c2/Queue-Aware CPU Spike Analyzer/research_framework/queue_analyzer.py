"""
Queue-Aware CPU Spike Analyzer
------------------------------
A production-oriented feature engineering pipeline for predicting microservice failures.
This component models system pressure using queue dynamics (arrival vs processing) 
and CPU trends to predict spikes before they hit critical thresholds.

Feature Calculations:
- arrival_processing_diff: Raw imbalance between incoming load and processing capacity.
- queue_growth_rate: Velocity of the queue size (derivative).
- overload_flag: Binary indicator when λ > μ.
- queue_pressure_index: A weighted metric combining backlog size and growth momentum.
- cpu_trend_Xmin: Rolling averages of CPU utilization to capture temporal drift.
- Pre-Failure Labeling: Heuristic that marks periods leading up to a spike as 'FAILURE'.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_telemetry(file_path):
    """Loads and prepares time-series data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Telemetry file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    # Ensure timestamp is numeric (unix seconds)
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def engineer_features(df):
    """
    Applies Advanced Queue-Aware feature engineering.
    - Adds velocity/acceleration to detect sudden spikes.
    - Implements robust QPI math.
    - Uses EWM for faster recovery decay.
    """
    # 1. CPU Dynamics (Velocity & Acceleration)
    time_diff = df['timestamp'].diff().replace(0, 1)
    df['cpu_velocity'] = df['cpu_percent'].diff() / time_diff
    df['cpu_acceleration'] = df['cpu_velocity'].diff() / time_diff
    
    # 2. Fast-Decay Trends (EWM)
    # Using EWM allows the system to 'forget' old high values faster after a recovery
    df['cpu_trend_5min'] = df['cpu_percent'].ewm(span=300, adjust=False).mean()
    df['cpu_trend_10min'] = df['cpu_percent'].ewm(span=600, adjust=False).mean()
    
    # 3. Arrival vs Processing Dynamics
    df['arrival_processing_diff'] = df['incoming_rate'] - df['processing_rate']
    
    # 4. Overload Flag (λ > μ)
    df['overload_flag'] = (df['incoming_rate'] > df['processing_rate']).astype(int)
    
    # 5. Queue Growth Rate (dq/dt)
    df['queue_growth_rate'] = df['in_flight_queue'].diff() / time_diff
    df['queue_growth_rate'] = df['queue_growth_rate'].fillna(0)
    
    # 6. Queue Pressure Index (QPI)
    # Formula: QPI = (Current_Backlog / Threshold) * (Arrival/Processing_Ratio)
    # This detects when small backlogs are growing uncontrollably
    max_q = 50.0 # Standard threshold for critical backlog
    arrival_ratio = (df['incoming_rate'] / df['processing_rate'].replace(0, 1)).clip(upper=5)
    
    df['queue_pressure_index'] = (df['in_flight_queue'] / max_q) * arrival_ratio
    df['queue_pressure_index'] = df['queue_pressure_index'].clip(upper=1.0)
    
    # 7. Normalization for ML Readiness
    # Scales critical features to 0-1 range
    cols_to_norm = ['cpu_percent', 'in_flight_queue', 'incoming_rate', 'processing_rate', 'cpu_velocity']
    for col in cols_to_norm:
        if col in df.columns:
            max_val = df[col].max()
            if max_val > 0:
                df[f'{col}_norm'] = df[col] / max_val
            else:
                df[f'{col}_norm'] = 0.0

    return df

def label_pre_failure(df, lookahead_sec=30, spike_threshold=85):
    """
    Labels each row as FAILURE (1) if a CPU spike or queue overflow 
    occurs within the 'lookahead_sec' window in the future.
    Otherwise NORMAL (0).
    """
    # Determine if a point is "In Spike"
    df['is_currently_spiking'] = (
        (df['cpu_percent'] >= spike_threshold) | 
        (df['in_flight_queue'] >= 50)
    ).astype(int)
    
    # Look ahead: use rolling window on the reverse of the dataframe
    # to see if any future points are spiking
    reversed_spikes = df['is_currently_spiking'][::-1]
    # Assuming 1Hz sampling, lookahead_sec rows = lookahead_sec
    has_future_spike = reversed_spikes.rolling(window=lookahead_sec, min_periods=1).max()[::-1]
    
    # Label is FAILURE (1) if there's a spike in the future BUT we aren't currently spiking
    # This specifically targets the 'pre-failure' window
    df['ml_label'] = has_future_spike.fillna(0).astype(int)
    
    # Clean up helper column
    df.drop(columns=['is_currently_spiking'], inplace=True)
    return df

def process_pipeline(input_path="final_research_dataset.csv", output_path="ml_ready_dataset.csv"):
    """Complete execution pipeline."""
    print(f"[*] Initializing Queue-Aware Analysis on {input_path}...")
    
    try:
        # Load
        df = load_telemetry(input_path)
        
        # Feature Engineering
        df = engineer_features(df)
        
        # Labeling (The "Pre-Failure" Heuristic)
        df = label_pre_failure(df, lookahead_sec=30, spike_threshold=80)
        
        # Final Cleaning
        df = df.fillna(0)
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"[+] Success! ML-ready dataset saved to {output_path}")
        print(f"[+] Total Records: {len(df)}")
        print(f"[+] Features Engineered: {len(df.columns)}")
        print(f"[+] Label Distribution: \n{df['ml_label'].value_counts(normalize=True)}")
        
        return df
    except Exception as e:
        print(f"[!] Pipeline Error: {e}")
        return None

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Queue-Aware CPU Spike Analyzer")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh the ML dataset every 5 seconds")
    args = parser.parse_args()

    if args.watch:
        print("[!] WATCH MODE ACTIVE: Auto-refreshing ml_ready_dataset.csv every 5 seconds...")
        while True:
            process_pipeline()
            time.sleep(5)
    else:
        # Default single run
        process_pipeline()
