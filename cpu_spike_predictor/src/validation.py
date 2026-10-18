import pandas as pd
import numpy as np
import os
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.model import train_model, predict_probabilities, get_features_and_target, COMP2_CPU_FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)


def run_lopo_cross_validation(merged_df: pd.DataFrame, alarm_threshold: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-One-Project-Out (LOPO) cross-validation.
    The model is trained on Component 2 CPU features ONLY.
    Component 3 memory columns are preserved in the output for downstream decision analysis
    but are NOT training features (prevents leakage).
    
    For single-project datasets, falls back to 80/20 train/test split.
    """
    if "project_id" not in merged_df.columns:
        merged_df["project_id"] = "Proj_01"
    projects = merged_df["project_id"].unique()
    logger.info(f"Starting LOPO CV across {len(projects)} projects with {len(COMP2_CPU_FEATURE_COLS)} Comp2 CPU features...")

    fold_metrics = []
    all_preds_list = []

    # If only 1 project, use 80/20 split instead of LOPO
    if len(projects) == 1:
        logger.info("Single project detected - using 80/20 train/test split instead of LOPO")
        from sklearn.model_selection import train_test_split
        
        train_df, test_df = train_test_split(merged_df.copy(), test_size=0.2, random_state=42, stratify=merged_df.get(TARGET_COL, None))
        
        X_train, y_train = get_features_and_target(train_df, TARGET_COL)
        X_test, y_test = get_features_and_target(test_df, TARGET_COL)

        model = train_model(X_train, y_train, random_state=42)

        test_probs = predict_probabilities(model, X_test)
        test_df["cpu_failure_prob"] = test_probs
        test_df["cpu_alarm"] = (test_probs >= alarm_threshold).astype(int)

        if "memory_prob" in test_df.columns:
            test_df["memory_leak_prob"] = test_df["memory_prob"]
        elif "memory_leak_prob" not in test_df.columns:
            test_df["memory_leak_prob"] = 0.0

        all_preds_list.append(test_df)

        y_pred = test_df["cpu_alarm"]
        y_true = y_test.astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            far = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        except Exception:
            tn, fp, fn, tp = 0, 0, 0, 0
            far = 0.0

        fold_metrics.append({
            "test_project": "80/20-split",
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_alarm_rate": far,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        })

        logger.info(f"80/20 Split - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, FAR: {far:.4f}")
    else:
        # Multi-project LOPO
        for test_proj in sorted(projects):
            logger.info(f"--- FOLD: Testing on unseen project {test_proj} ---")

            train_df = merged_df[merged_df["project_id"] != test_proj].copy()
            test_df = merged_df[merged_df["project_id"] == test_proj].copy()

            if train_df.empty or test_df.empty:
                logger.warning(f"Skipping fold {test_proj}: empty train/test split")
                continue

            X_train, y_train = get_features_and_target(train_df, TARGET_COL)
            X_test, y_test = get_features_and_target(test_df, TARGET_COL)

            model = train_model(X_train, y_train, random_state=42)

            test_probs = predict_probabilities(model, X_test)
            test_df = test_df.copy()
            test_df["cpu_failure_prob"] = test_probs
            test_df["cpu_alarm"] = (test_probs >= alarm_threshold).astype(int)

            if "memory_prob" in test_df.columns:
                test_df["memory_leak_prob"] = test_df["memory_prob"]
            elif "memory_leak_prob" not in test_df.columns:
                test_df["memory_leak_prob"] = 0.0

            all_preds_list.append(test_df)

            y_pred = test_df["cpu_alarm"]
            y_true = y_test.astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            try:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                far = fp / (tn + fp) if (tn + fp) > 0 else 0.0
            except Exception:
                tn, fp, fn, tp = 0, 0, 0, 0
                far = 0.0

            fold_metrics.append({
                "test_project": test_proj,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "false_alarm_rate": far,
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            })

            logger.info(f"Fold {test_proj} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, FAR: {far:.4f}")

    lopo_results_df = pd.DataFrame(fold_metrics)
    all_preds_df = pd.concat(all_preds_list, ignore_index=True) if all_preds_list else pd.DataFrame()

    sort_cols = [c for c in ["service_name", "project_id", "timestamp", "timestamp_dt"] if c in all_preds_df.columns]
    if sort_cols:
        all_preds_df = all_preds_df.sort_values(by=sort_cols).reset_index(drop=True)

    if len(lopo_results_df) > 0:
        logger.info("=== LOPO CV OVERALL SUMMARY ===")
        logger.info(f"Mean F1:        {lopo_results_df['f1_score'].mean():.4f}")
        logger.info(f"Mean Precision: {lopo_results_df['precision'].mean():.4f}")
        logger.info(f"Mean Recall:    {lopo_results_df['recall'].mean():.4f}")
        logger.info(f"Mean FAR:       {lopo_results_df['false_alarm_rate'].mean():.4f}")
        logger.info(f"F1 Variance:    {lopo_results_df['f1_score'].var():.4f}")
    else:
        logger.warning("LOPO CV produced no folds (single project or empty)")

    return lopo_results_df, all_preds_df


def save_validation_results(lopo_results_df: pd.DataFrame, all_preds_df: pd.DataFrame, output_dir: str = "outputs"):
    """
    Saves fold performance results and predictions to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    lopo_path = os.path.join(output_dir, "lopo_results.csv")
    preds_path = os.path.join(output_dir, "cpu_predictions.csv")
    lopo_results_df.to_csv(lopo_path, index=False)
    all_preds_df.to_csv(preds_path, index=False)
    logger.info(f"Saved LOPO results to {lopo_path}")
    logger.info(f"Saved CPU predictions ({len(all_preds_df)} rows) to {preds_path}")


if __name__ == "__main__":
    from src.ingestion import (
        load_cpu_features, load_memory_predictions, merge_datasets,
        DEFAULT_COMP2_CSV, DEFAULT_COMP3_CSV,
    )

    logging.basicConfig(level=logging.INFO)
    try:
        cpu = load_cpu_features(DEFAULT_COMP2_CSV, lead_time_minutes=5)
        mem = load_memory_predictions(DEFAULT_COMP3_CSV)
        merged = merge_datasets(cpu, mem)
        lopo_results, all_preds = run_lopo_cross_validation(merged)
        save_validation_results(lopo_results, all_preds)
    except Exception as e:
        logger.error(f"Validation self-test failed: {e}", exc_info=True)
