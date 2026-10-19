import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_recall_curve
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

y_train = df['actual_failure'].iloc[:split_idx]
y_test = df['actual_failure'].iloc[split_idx:]

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

# Model A
features_A = ['cpu_percent']
X_train_A = df[features_A].iloc[:split_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_A = df[features_A].iloc[split_idx:].replace([np.inf, -np.inf], np.nan).fillna(0)

# OOF for threshold tuning
oof_proba_A = np.zeros(len(X_train_A))
oof_mask_A = np.zeros(len(X_train_A), dtype=bool)
for tri, vai in TimeSeriesSplit(n_splits=3).split(X_train_A):
    fold_clf = _make_clf()
    fold_clf.fit(X_train_A.iloc[tri], y_train.iloc[tri])
    oof_proba_A[vai] = _proba1(fold_clf, X_train_A.iloc[vai])
    oof_mask_A[vai] = True

y_oof_A = y_train.iloc[oof_mask_A]
prec_c, rec_c, thr_c = precision_recall_curve(y_oof_A, oof_proba_A[oof_mask_A])
f1_c = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-9)
candidates = [(t, f) for t, f in zip(thr_c, f1_c[:len(thr_c)]) if t < 1.0]
best_thr_A, best_f1_A = min(candidates, key=lambda tf: (-round(tf[1], 9), tf[0]))
print(f"Model A best threshold: {best_thr_A:.4f}, OOF F1: {best_f1_A:.4f}")

clf_A = _make_clf()
clf_A.fit(X_train_A, y_train)
proba_test_A = _proba1(clf_A, X_test_A)
preds_A = (proba_test_A >= best_thr_A).astype(int)

prec_A, rec_A, f1_A, _ = precision_recall_fscore_support(y_test, preds_A, average='binary', pos_label=1, zero_division=0)
cm_A = confusion_matrix(y_test, preds_A, labels=[0, 1])
fp_A = cm_A[0][1] if cm_A.shape[0] > 1 and cm_A.shape[1] > 1 else 0
fn_A = cm_A[1][0] if cm_A.shape[0] > 1 and cm_A.shape[1] > 1 else 0
print(f"Model A TEST: Precision={prec_A:.4f}, Recall={rec_A:.4f}, F1={f1_A:.4f}, FP={fp_A}, FN={fn_A}")
print(f"Pred distribution: {np.bincount(preds_A)}")
print(f"True distribution: {np.bincount(y_test)}")

# Model D
features_D = ['cpu_percent', 'qpi']
X_train_D = df[features_D].iloc[:split_idx].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_D = df[features_D].iloc[split_idx:].replace([np.inf, -np.inf], np.nan).fillna(0)

oof_proba_D = np.zeros(len(X_train_D))
oof_mask_D = np.zeros(len(X_train_D), dtype=bool)
for tri, vai in TimeSeriesSplit(n_splits=3).split(X_train_D):
    fold_clf = _make_clf()
    fold_clf.fit(X_train_D.iloc[tri], y_train.iloc[tri])
    oof_proba_D[vai] = _proba1(fold_clf, X_train_D.iloc[vai])
    oof_mask_D[vai] = True

y_oof_D = y_train.iloc[oof_mask_D]
prec_c, rec_c, thr_c = precision_recall_curve(y_oof_D, oof_proba_D[oof_mask_D])
f1_c = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-9)
candidates = [(t, f) for t, f in zip(thr_c, f1_c[:len(thr_c)]) if t < 1.0]
best_thr_D, best_f1_D = min(candidates, key=lambda tf: (-round(tf[1], 9), tf[0]))
print(f"\nModel D best threshold: {best_thr_D:.4f}, OOF F1: {best_f1_D:.4f}")

clf_D = _make_clf()
clf_D.fit(X_train_D, y_train)
proba_test_D = _proba1(clf_D, X_test_D)
preds_D = (proba_test_D >= best_thr_D).astype(int)

prec_D, rec_D, f1_D, _ = precision_recall_fscore_support(y_test, preds_D, average='binary', pos_label=1, zero_division=0)
cm_D = confusion_matrix(y_test, preds_D, labels=[0, 1])
fp_D = cm_D[0][1] if cm_D.shape[0] > 1 and cm_D.shape[1] > 1 else 0
fn_D = cm_D[1][0] if cm_D.shape[0] > 1 and cm_D.shape[1] > 1 else 0
print(f"Model D TEST: Precision={prec_D:.4f}, Recall={rec_D:.4f}, F1={f1_D:.4f}, FP={fp_D}, FN={fn_D}")
print(f"Pred distribution: {np.bincount(preds_D)}")
print(f"True distribution: {np.bincount(y_test)}")

# Also test with threshold 0.5 for comparison
print("\n--- With threshold 0.5 ---")
preds_A_5 = (proba_test_A >= 0.5).astype(int)
prec_A_5, rec_A_5, f1_A_5, _ = precision_recall_fscore_support(y_test, preds_A_5, average='binary', pos_label=1, zero_division=0)
print(f"Model A @0.5: Precision={prec_A_5:.4f}, Recall={rec_A_5:.4f}, F1={f1_A_5:.4f}")

preds_D_5 = (proba_test_D >= 0.5).astype(int)
prec_D_5, rec_D_5, f1_D_5, _ = precision_recall_fscore_support(y_test, preds_D_5, average='binary', pos_label=1, zero_division=0)
print(f"Model D @0.5: Precision={prec_D_5:.4f}, Recall={rec_D_5:.4f}, F1={f1_D_5:.4f}")