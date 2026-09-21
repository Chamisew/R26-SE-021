import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.join(os.path.dirname(__file__), "test"))

df = pd.read_csv('final_research_dataset.pre_schema_20261009_001347.csv', on_bad_lines='skip', low_memory=False)

# Exact same split logic as feature_sets_test.py
split_idx = max(1, min(len(df) - 1, int(len(df) * 0.5)))
valid_split = None
for candidate in range(max(1, int(len(df) * 0.25)), min(len(df) - 1, int(len(df) * 0.75)) + 1):
    y_train = df['actual_failure'].iloc[:candidate]
    y_test = df['actual_failure'].iloc[candidate:]
    if y_train.nunique() >= 2 and y_test.nunique() >= 2:
        valid_split = candidate
        break

if valid_split is None:
    positive_idx = np.flatnonzero(df['actual_failure'].to_numpy() == 1)
    split_idx = max(1, min(len(df) - 1, int(positive_idx[0])))
else:
    split_idx = valid_split

print(f"Split index: {split_idx}")

y_train = df['actual_failure'].iloc[:split_idx]
y_test = df['actual_failure'].iloc[split_idx:]

t_seconds = df['time_sec'].iloc[split_idx:] if 'time_sec' in df.columns else pd.Series(range(split_idx, len(df)))
if 'time_sec' in df.columns:
    first_failure_time = df.loc[df['actual_failure'] == 1, 'time_sec'].min()
else:
    first_failure_time = None

def _proba1(clf, X):
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.zeros(0)
    p = clf.predict_proba(X)
    if p.shape[1] == 1:
        return np.ones(p.shape[0]) if clf.classes_[0] == 1 else np.zeros(p.shape[0])
    return p[:, list(clf.classes_).index(1)]

def _make_clf():
    return RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', min_samples_leaf=1)

model_definitions = {
    "Model A (CPU Only)": ['cpu_percent'],
    "Model D (CPU + QPI)": ['cpu_percent', 'qpi'],
}

for name, features in model_definitions.items():
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    X_train = df[features].iloc[:split_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = df[features].iloc[split_idx:].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Time-ordered validation slice (last 20% of train)
    val_n = max(10, min(len(X_train) - 1, int(len(X_train) * 0.2)))
    X_tr, y_tr = X_train.iloc[:-val_n], y_train.iloc[:-val_n]
    X_val, y_val = X_train.iloc[-val_n:], y_train.iloc[-val_n:]
    print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train pos rate: {y_tr.mean():.3f}, Val pos rate: {y_val.mean():.3f}, Test pos rate: {y_test.mean():.3f}")

    # OOF threshold tuning
    oof_proba = np.zeros(len(X_train))
    oof_mask = np.zeros(len(X_train), dtype=bool)
    try:
        for tri, vai in TimeSeriesSplit(n_splits=3).split(X_train):
            fold_clf = _make_clf()
            fold_clf.fit(X_train.iloc[tri], y_train.iloc[tri])
            oof_proba[vai] = _proba1(fold_clf, X_train.iloc[vai])
            oof_mask[vai] = True
    except ValueError:
        pass
    y_oof = y_train.iloc[oof_mask]

    if oof_mask.sum() > 10 and y_oof.nunique() >= 2:
        prec_c, rec_c, thr_c = precision_recall_curve(y_oof, oof_proba[oof_mask])
        f1_c = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-9)
        candidates = [(t, f) for t, f in zip(thr_c, f1_c[:len(thr_c)]) if t < 1.0]
        print(f"OOF candidates: {len(candidates)}")
        if candidates:
            # Show top 5 candidates
            sorted_cand = sorted(candidates, key=lambda tf: (-round(tf[1], 9), tf[0]))[:5]
            for t, f in sorted_cand:
                print(f"  thresh={t:.4f}, F1={f:.4f}")
            best_thr, best_f1 = min(candidates, key=lambda tf: (-round(tf[1], 9), tf[0]))
            threshold = float(best_thr)
            print(f"SELECTED: threshold={threshold:.4f}, OOF F1={best_f1:.4f}")
        else:
            threshold = 0.5
            print(f"No candidates, using 0.5")
    else:
        threshold = 0.5
        print(f"Not enough OOF data, using 0.5")

    # Final model
    clf = _make_clf()
    clf.fit(X_train, y_train)

    proba_test = _proba1(clf, X_test)
    print(f"Test proba: min={proba_test.min():.4f}, max={proba_test.max():.4f}, mean={proba_test.mean():.4f}")
    print(f"Test proba > threshold: {(proba_test >= threshold).sum()} / {len(proba_test)}")
    
    preds = (proba_test >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    fp = cm[0][1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    fn = cm[1][0] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    alert_indices = np.where(preds == 1)[0]
    if len(alert_indices) > 0 and first_failure_time is not None:
        first_alert_time = float(t_seconds.iloc[alert_indices[0]])
        lead_time = max(0, int(round(first_failure_time - first_alert_time)))
    else:
        lead_time = 0

    print(f"RESULT: Precision={prec*100:.1f}%, Recall={rec*100:.1f}%, F1={f1:.3f}, FP={fp}, FN={fn}, Lead=+{lead_time}s, thr={threshold:.3f}")
    
    # Also show what happens with threshold 0.5
    preds_5 = (proba_test >= 0.5).astype(int)
    prec_5, rec_5, f1_5, _ = precision_recall_fscore_support(y_test, preds_5, average='binary', pos_label=1, zero_division=0)
    print(f"@0.5:   Precision={prec_5*100:.1f}%, Recall={rec_5*100:.1f}%, F1={f1_5:.3f}")