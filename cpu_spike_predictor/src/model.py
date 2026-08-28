import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

COMP2_CPU_FEATURE_COLS = [
    "cpu_percent",
    "cpu_velocity",
    "cpu_trend_5min",
    "cpu_trend_10min",
    "in_flight_queue",
    "incoming_rate",
    "processing_rate",
    "queue_growth_rate",
    "queue_pressure_index",
    "overload_flag",
]

FEATURE_COLS = COMP2_CPU_FEATURE_COLS
TARGET_COL = "imminent_failure"


def get_features_and_target(df: pd.DataFrame, target_col: str = "imminent_failure"):
    """
    Splits dataframe into Component 2 CPU feature matrix X and target vector y.
    Uses ONLY Component 2-owned features to avoid leakage from Component 3 predictions.
    """
    available_features = [c for c in COMP2_CPU_FEATURE_COLS if c in df.columns]
    missing = [c for c in COMP2_CPU_FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Model: Some Comp2 feature columns missing from input df: {missing}")

    X = df[available_features].copy()
    for col in available_features:
        if col == "overload_flag":
            X[col] = X[col].fillna(0).astype(int)
        else:
            X[col] = X[col].fillna(0.0)

    y = df[target_col].fillna(0).astype(int)
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int = 100, random_state: int = 42) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier with class_weight='balanced' for class imbalance.
    Input features: Component 2 CPU/queue/system metrics ONLY.
    Target: imminent_failure (proactive lead-time label from Component 2).
    """
    logger.info(f"Training RandomForestClassifier on {X.shape[0]} samples with {X.shape[1]} Comp2 CPU features...")
    logger.info(f"  Feature columns: {list(X.columns)}")
    logger.info(f"  Target class distribution: {dict(y.value_counts())}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X, y)
    logger.info("Model training complete.")
    return model


def save_model(model: RandomForestClassifier, filepath: str = "outputs/cpu_rf_model.joblib"):
    """
    Serializes the model to a joblib file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    logger.info(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    logger.info("Model saved successfully.")


def load_model(filepath: str = "outputs/cpu_rf_model.joblib") -> RandomForestClassifier:
    """
    Deserializes a model from a joblib file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")
    logger.info(f"Loading model from {filepath}...")
    model = joblib.load(filepath)
    logger.info("Model loaded successfully.")
    return model


def predict_probabilities(model: RandomForestClassifier, X) -> np.ndarray:
    """
    Returns the Component 2 CPU failure probability (class 1) for each sample.
    Accepts DataFrame or NumPy array.
    """
    if isinstance(X, np.ndarray):
        n_features = X.shape[1] if X.ndim > 1 else 1
        expected = len(COMP2_CPU_FEATURE_COLS)
        if n_features != expected and n_features < expected:
            pad_width = expected - n_features
            X = np.hstack([X, np.zeros((X.shape[0], pad_width), dtype=X.dtype)])
        X_input = X
    elif isinstance(X, pd.DataFrame):
        available = [c for c in COMP2_CPU_FEATURE_COLS if c in X.columns]
        if len(available) != len(COMP2_CPU_FEATURE_COLS):
            for col in COMP2_CPU_FEATURE_COLS:
                if col not in X.columns:
                    X[col] = 0.0
        X_input = X[COMP2_CPU_FEATURE_COLS]
    else:
        X_input = X

    if len(model.classes_) == 1:
        single_class = model.classes_[0]
        n = len(X_input) if hasattr(X_input, '__len__') else 1
        if single_class == 0:
            return np.zeros(n)
        else:
            return np.ones(n)

    probs = model.predict_proba(X_input)
    return probs[:, 1]


def get_feature_importances(model: RandomForestClassifier) -> dict:
    """
    Returns a dictionary mapping Component 2 CPU feature names to their relative importance.
    """
    importances = model.feature_importances_
    n = min(len(importances), len(COMP2_CPU_FEATURE_COLS))
    feat_names = COMP2_CPU_FEATURE_COLS[:n]
    result = dict(zip(feat_names, importances[:n]))
    for col in COMP2_CPU_FEATURE_COLS[n:]:
        result[col] = 0.0
    return result
