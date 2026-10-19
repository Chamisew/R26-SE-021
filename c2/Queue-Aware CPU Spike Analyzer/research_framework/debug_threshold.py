import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.join(os.path.dirname(__file__), "test"))

df = pd.read_csv('final_research_dataset.pre_schema_20261009_001347.csv', on_bad_lines='skip', low_memory=False)

# Replicate the split logic
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

print(f'Split index: {split_idx}')

y_train = df['actual_failure'].iloc[:split_idx]
y_test = df['actual_failure'].iloc[split_idx:]

# Test Model A (CPU Only)
print("\n" + "="*60)
print("MODEL A: CPU Only")
print("="*60)
features_A = ['cpu_percent']
X_train_A = df[features_A].iloc[:split_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_A = df[features_A].iloc[split_idx:].replace([np.inf, -np.inf], np.nan).fillna(0)

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

# OOF for Model A
oof_proba_A = np.zeros(len(X_train_A))
oof_mask_A = np.zeros(len(X_train_A), dtype=bool)
for tri, vai in TimeSeriesSplit(n_splits=3).split(X_train_A):
    fold_clf = _make_clf()
    fold_clf.fit(X_train_A.iloc[tri], y_train.iloc[tri])
    oof_proba_A[vai] = _proba1(fold_clf, X_train_A.iloc[vai])
    oof_mask_A[vai] = True

y_oof_A = y_train.iloc[oof_mask_A]
print(f"OOF mask sum: {oof_mask_A.sum()}")
print(f"OOF y unique: {y_oof_A.nunique()}")
print(f"OOF proba stats: min={oof_proba_A[oof_mask_A].min():.4f}, max={oof_proba_A[oof_mask_A].max():.4f}, mean={oof_proba_A[oof_mask_A].mean():.4f}")
print(f"OOF proba unique values: {len(np.unique(oof_proba_A[oof_mask_A]))}")

if oof_mask_A.sum() > 10 and y_oof_A.nunique() >= 2:
    prec_c, rec_c, thr_c = precision_recall_curve(y_oof_A, oof_proba_A[oof_mask_A])
    f1_c = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-9)
    candidates = [(t, f) for t, f in zip(thr_c, f1_c[:len(thr_c)]) if t < 1.0]
    print(f"Number of threshold candidates: {len(candidates)}")
    if candidates:
        best_thr, best_f1 = min(candidates, key=lambda tf: (-round(tf[1], 9), tf[0]))
        print(f"Best threshold: {best_thr:.4f}, Best F1: {best_f1:.4f}")
    else:
        print("No candidates, using 0.5")
else:
    print("Not enough OOF data, using 0.5")

# Train final Model A
clf_A = _make_clf()
clf_A.fit(X_train_A, y_train)
proba_test_A = _proba1(clf_A, X_test_A)
print(f"Test proba stats: min={proba_test_A.min():.4f}, max={proba_test_A.max():.4f}, mean={proba_test_A.mean():.4f}")

# Test Model D (CPU + QPI)
print("\n" + "="*60)
print("MODEL D: CPU + QPI")
print("="*60)
features_D = ['cpu_percent', 'qpi']
X_train_D = df[features_D].iloc[:split_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_D = df[features_D].iloc[split_idx:].replace([np.inf, -np.inf], np.nan).fillna(0)

# OOF for Model D
oof_proba_D = np.zeros(len(X_train_D))
oof_mask_D = np.zeros(len(X_train_D), dtype=bool)
for tri, vai in TimeSeriesSplit(n_splits=3).split(X_train_D):
    fold_clf = _make_clf()
    fold_clf.fit(X_train_D.iloc[tri], y_train.iloc[tri])
    oof_proba_D[vai] = _proba1(fold_clf, X_train_D.iloc[vai])
    oof_mask_D[vai] = True

y_oof_D = y_train.iloc[oof_mask_D]
print(f"OOF mask sum: {oof_mask_D.sum()}")
print(f"OOF y unique: {y_oof_D.nunique()}")
print(f"OOF proba stats: min={oof_proba_D[oof_mask_D].min():.4f}, max={oof_proba_D[oof_mask_D].max():.4f}, mean={oof_proba_D[oof_mask_D].mean():.4f}")
print(f"OOF proba unique values: {len(np.unique(oof_proba_D[oof_mask_D]))}")

if oof_mask_D.sum() > 10 and y_oof_D.nunique() >= 2:
    prec_c, rec_c, thr_c = precision_recall_curve(y_oof_D, oof_proba_D[oof_mask_D])
    f1_c = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-9)
    candidates = [(t, f) for t, f in zip(thr_c, f1_c[:len(thr_c)]) if t < 1.0]
    print(f"Number of threshold candidates: {len(candidates)}")
    if candidates:
        best_thr, best_f1 = min(candidates, key=lambda tf: (-round(tf[1], 9), tf[0]))
        print(f"Best threshold: {best_thr:.4f}, Best F1: {best_f1:.4f}")
    else:
        print("No candidates, using 0.5")
else:
    print("Not enough OOF data, using 0.5")

# Train final Model D
clf_D = _make_clf()
clf_D.fit(X_train_D, y_train)
proba_test_D = _proba1(clf_D, X_test_D)
print(f"Test proba stats: min={proba_test_D.min():.4f}, max={proba_test_D.max():.4f}, mean={proba_test_D.mean():.4f}")

# Feature importance
print("\nFeature importance Model A:", clf_A.feature_importances_)
print("Feature importance Model D:", clf_D.feature_importances_)