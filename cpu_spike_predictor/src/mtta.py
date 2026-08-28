import pandas as pd
import numpy as np
import os
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

def compute_mtta_for_alarm_type(df: pd.DataFrame, alarm_col: str, window_minutes: int = 20) -> pd.DataFrame:
    """
    Computes MTTA for each failure event across all projects.
    For each failure start (T_fail), looks back up to `window_minutes` minutes.
    Finds the earliest alarm (T_alarm) in that window, and calculates:
        MTTA = T_fail - T_alarm (in seconds)
    """
    results = []
    
    # Sort dataset to ensure proper rolling/shifting calculations
    df = df.sort_values(by=["project_id", "timestamp"]).reset_index(drop=True)
    
    for project_id, proj_df in df.groupby("project_id"):
        proj_df = proj_df.copy().reset_index(drop=True)
        
        # Identify start of failure events: ground_truth_failure transitions 0 -> 1
        proj_df["fail_start"] = (proj_df["ground_truth_failure"] == 1) & (proj_df["ground_truth_failure"].shift(1, fill_value=0) == 0)
        
        fail_times = proj_df[proj_df["fail_start"] == True]["timestamp"].tolist()
        
        for t_fail in fail_times:
            # 20 minutes window lookback: [T_fail - 20 min, T_fail)
            start_window = t_fail - timedelta(minutes=window_minutes)
            
            # Filter lookback window
            lookback_df = proj_df[(proj_df["timestamp"] >= start_window) & (proj_df["timestamp"] < t_fail)]
            
            # Find alarms in this window
            alarms_in_window = lookback_df[lookback_df[alarm_col] == 1]
            
            if not alarms_in_window.empty:
                # Earliest alarm in the window
                t_alarm = alarms_in_window["timestamp"].min()
                mtta_seconds = (t_fail - t_alarm).total_seconds()
                
                results.append({
                    "project_id": project_id,
                    "service_name": proj_df["service_name"].iloc[0],
                    "t_fail": t_fail,
                    "t_alarm": t_alarm,
                    "mtta_seconds": mtta_seconds,
                    "mtta_minutes": mtta_seconds / 60.0,
                    "anticipated": 1
                })
            else:
                # Failure occurred but no alarm was raised in lookback window
                results.append({
                    "project_id": project_id,
                    "service_name": proj_df["service_name"].iloc[0],
                    "t_fail": t_fail,
                    "t_alarm": pd.NaT,
                    "mtta_seconds": 0.0,
                    "mtta_minutes": 0.0,
                    "anticipated": 0
                })
                
    return pd.DataFrame(results)

def analyze_mtta_strategies(predictions_path: str = "outputs/cpu_predictions.csv", output_dir: str = "outputs") -> pd.DataFrame:
    """
    Evaluates MTTA using different strategies:
      1. CPU Alarm only (cpu_failure_prob >= 0.6)
      2. Memory Alarm only (memory_leak_prob >= 0.6)
      3. Joint (AND) Alarm: Both CPU and Memory alarms are active (cpu_alarm == 1 AND memory_leak_prob >= 0.6)
      4. Joint (OR) Alarm: Either CPU alarm OR Memory alarm is active
    """
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
        
    df = pd.read_csv(predictions_path)

    if "timestamp_dt" in df.columns:
        ts_col = "timestamp_dt"
    else:
        ts_col = "timestamp"
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")

    if "memory_prob" in df.columns and "memory_leak_prob" not in df.columns:
        df["memory_leak_prob"] = df["memory_prob"]
    elif "memory_leak_prob" not in df.columns:
        df["memory_leak_prob"] = 0.0

    # Define alarm states
    df["cpu_alarm"] = (df["cpu_failure_prob"] >= 0.6).astype(int)
    df["mem_alarm"] = (df["memory_leak_prob"] >= 0.6).astype(int)
    df["joint_and_alarm"] = ((df["cpu_alarm"] == 1) & (df["mem_alarm"] == 1)).astype(int)
    df["joint_or_alarm"] = ((df["cpu_alarm"] == 1) | (df["mem_alarm"] == 1)).astype(int)
    
    strategies = {
        "CPU_Only": "cpu_alarm",
        "Memory_Only": "mem_alarm",
        "Joint_AND": "joint_and_alarm",
        "Joint_OR": "joint_or_alarm"
    }
    
    all_mtta_results = []
    
    logger.info("=== Computing MTTA for different Alarm Strategies ===")
    for strategy_name, alarm_col in strategies.items():
        mtta_df = compute_mtta_for_alarm_type(df, alarm_col)
        mtta_df["strategy"] = strategy_name
        all_mtta_results.append(mtta_df)
        
        # Calculate summary statistics for anticipated failures
        anticipated_df = mtta_df[mtta_df["anticipated"] == 1]
        
        if not anticipated_df.empty:
            mean_mtta = anticipated_df["mtta_minutes"].mean()
            std_mtta = anticipated_df["mtta_minutes"].std()
            min_mtta = anticipated_df["mtta_minutes"].min()
            max_mtta = anticipated_df["mtta_minutes"].max()
        else:
            mean_mtta, std_mtta, min_mtta, max_mtta = 0.0, 0.0, 0.0, 0.0
            
        anticipation_rate = mtta_df["anticipated"].mean() * 100.0
        
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"  Anticipation Rate: {anticipation_rate:.1f}%")
        logger.info(f"  Mean MTTA:         {mean_mtta:.2f} mins ({mean_mtta*60:.1f} secs)")
        logger.info(f"  Std Dev MTTA:      {std_mtta:.2f} mins ({std_mtta*60:.1f} secs)")
        logger.info(f"  MTTA Range:        [{min_mtta:.1f}, {max_mtta:.1f}] mins")
        
    combined_mtta_df = pd.concat(all_mtta_results, ignore_index=True)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    mtta_path = os.path.join(output_dir, "cpu_mtta_results.csv")
    combined_mtta_df.to_csv(mtta_path, index=False)
    logger.info(f"Saved combined MTTA results to {mtta_path}")
    
    return combined_mtta_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        analyze_mtta_strategies()
    except Exception as e:
        logger.error(f"MTTA self-test failed: {e}")
