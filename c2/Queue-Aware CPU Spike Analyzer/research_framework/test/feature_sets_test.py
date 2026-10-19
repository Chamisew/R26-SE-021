"""
feature_sets_test.py  (v4 — corrected features, GradientBoosting, verified A≠B)
=================================================================================
Research novelty experiment: 5 feature-set representations (Models A – E).

Key fixes over v3
------------------
1. in_flight_queue corr with overload = -0.695 (queue DROPS when service is
   overloaded — it can't accept new requests). Raw queue alone is an anti-signal.
   → Model B now uses latency_ms (positive correlation) instead of raw queue,
     making B genuinely different from A.

2. cpu_zscore has 36k NaNs → filled with 0 previously, destroying its signal.
   → NaNs imputed with the median computed on the training portion only (no leakage).

3. GradientBoostingClassifier used instead of RandomForest. It handles class
   imbalance better and typically gives 10-20% better F1 on tabular data.
   SMOTE is still applied for the OOF threshold selection pass.

4. Feature importance printed for Model E so the paper can show which queue
   signals matter most.

5. Model A–E definitions revised so each model adds a genuinely new signal:
   A: cpu_percent only (baseline)
   B: cpu_percent + latency_ms + overload_flag (network / load signals)
   C: A + queue dynamics (incoming_rate, processing_rate, queue_growth_rate, processing_efficiency)
   D: A + QPI + statistical anomaly scores (cpu_zscore, qpi_zscore)
   E: Full — all of C + D (queue-aware with z-score anomaly context)
"""

import os, sys, json
import time as _time
import numpy as np
import pandas as pd

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
ALT_DATASET_PATH = os.path.join(SCRIPT_DIR, '..', 'final_research_dataset.csv')
DATASET_PATH     = os.path.join(SCRIPT_DIR, '..', 'empirical_validation_dataset.csv')

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (precision_recall_fscore_support,
                                 confusion_matrix,
                                 precision_recall_curve,
                                 accuracy_score)
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

LOOKAHEAD_SEC = 30


# ---------------------------------------------------------------------------
# Per-service overload-based labeling
# ---------------------------------------------------------------------------

def label_per_service(df, lookahead_sec=LOOKAHEAD_SEC):
    """
    Per-service 30-second lookahead on `overload`:
      overload==1 on current row      → target=2 (FAILURE, excluded later)
      overload==1 within lookahead_sec → target=1 (PRE_FAILURE)
      otherwise                        → target=0 (NORMAL)
    If ml_label is already pre-computed in df, uses ml_label directly.
    """
    if 'ml_label' in df.columns and not df['ml_label'].isna().all():
        df = df.copy()
        df['target'] = pd.to_numeric(df['ml_label'], errors='coerce')
        return df

    df = df.copy()
    df['target'] = 0

    for svc, grp in df.groupby('service', sort=False):
        grp_s = grp.sort_values('timestamp')
        idx   = grp_s.index
        ts    = grp_s['timestamp'].values.astype(float)
        ov    = grp_s['overload'].values.astype(int)
        tgt   = np.zeros(len(grp_s), dtype=int)

        for i in range(len(grp_s)):
            if ov[i] == 1:
                tgt[i] = 2
            else:
                window = (ts > ts[i]) & (ts <= ts[i] + lookahead_sec)
                if np.any(ov[window] == 1):
                    tgt[i] = 1

        df.loc[idx, 'target'] = tgt

    return df


def prepare_dataset(df):
    """
    Clean, label, and engineer features.
    NaN imputation is done AFTER the train/test split to prevent leakage.
    """
    if 'baseline_tier' in df.columns:
        df = df[df['baseline_tier'].isin(['PROVISIONAL', 'STATISTICAL'])].copy()

    # Ensure overload column
    if 'overload' not in df.columns:
        cpu = pd.to_numeric(df.get('cpu_percent', 0), errors='coerce').fillna(0)
        af  = pd.to_numeric(df.get('actual_failure', 0), errors='coerce').fillna(0)
        df['overload'] = ((cpu >= 90) | (af == 1)).astype(int)
    else:
        df['overload'] = pd.to_numeric(df['overload'], errors='coerce').fillna(0).astype(int)

    # Timestamp
    if 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df), dtype=float)
    else:
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

    if 'service' not in df.columns:
        df['service'] = 'default'

    # Per-service labeling
    df = label_per_service(df, LOOKAHEAD_SEC)

    # Drop FAILURE rows — features unreliable while service is down
    df = df[df['target'] != 2].copy()
    df = df[df['target'].notna()].copy()

    # Derive helper features
    if 'overload_flag' not in df.columns:
        df['overload_flag'] = df.get('overload', 0)

    if 'latency_ms' not in df.columns:
        df['latency_ms'] = 0.0

    if 'queue_growth_rate' not in df.columns and 'incoming_rate' in df.columns and 'processing_rate' in df.columns:
        inc = pd.to_numeric(df['incoming_rate'], errors='coerce').fillna(0)
        prc = pd.to_numeric(df['processing_rate'], errors='coerce').fillna(0)
        df['queue_growth_rate'] = inc - prc

    if 'processing_efficiency' not in df.columns and 'incoming_rate' in df.columns and 'processing_rate' in df.columns:
        inc = pd.to_numeric(df['incoming_rate'], errors='coerce').fillna(0)
        prc = pd.to_numeric(df['processing_rate'], errors='coerce').fillna(0)
        df['processing_efficiency'] = np.minimum(1.0, prc / np.maximum(inc, 0.001))

    # Coerce all feature columns to float (NaN stays NaN for now — imputed per split)
    all_feature_cols = [
        'cpu_percent', 'latency_ms', 'overload_flag',
        'incoming_rate', 'processing_rate', 'queue_growth_rate', 'processing_efficiency',
        'qpi', 'cpu_zscore', 'qpi_zscore',
    ]
    for col in all_feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df.reset_index(drop=True)


def _impute(X_train, X_test):
    """Fill NaNs with training-set medians (no leakage)."""
    medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        mask_tr = np.isnan(X_train[:, j])
        mask_te = np.isnan(X_test[:, j])
        X_train[mask_tr, j] = medians[j]
        X_test[mask_te, j]  = medians[j]
    return X_train, X_test


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_5_models(df=None):
    print('\n' + '=' * 90)
    print('   RESEARCH NOVELTY EXPERIMENT: FEATURE REPRESENTATIONS (MODELS A - E)')
    print('=' * 90)

    if not SKLEARN_AVAILABLE:
        raise ImportError('scikit-learn required.')

    # ── Load dataset ──────────────────────────────────────────────────────────
    if df is None:
        for path in [DATASET_PATH, ALT_DATASET_PATH]:
            if os.path.exists(path):
                df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
                break
        else:
            raise FileNotFoundError('Dataset not found.')

    df = prepare_dataset(df)
    df = df.sort_values('timestamp').reset_index(drop=True)

    n_pos = int(df['target'].sum())
    n_neg = int((df['target'] == 0).sum())
    prevalence = n_pos / max(1, len(df)) * 100
    print('  Dataset  : %d rows | PRE_FAILURE: %d (%.1f%%) | NORMAL: %d'
          % (len(df), n_pos, prevalence, n_neg))

    if n_pos < 10:
        raise ValueError('Too few PRE_FAILURE rows.')

    # ── Model A–E: corrected feature sets ─────────────────────────────────────
    #
    # A  CPU only — absolute baseline, no network or queue knowledge
    # B  CPU + latency + overload_flag — adds network/load signals (NOT raw queue
    #    depth, which has -0.695 correlation with overload and misleads the model)
    # C  CPU + Queue dynamics — arrival rate, processing rate, growth rate, efficiency
    # D  CPU + QPI + statistical anomaly context (cpu_zscore, qpi_zscore)
    # E  Full Queue-Aware — all of C + D (the proposed model)
    model_defs = {
        'Model A (CPU Only)': [
            'cpu_percent',
        ],
        'Model B (CPU + Load Signals)': [
            'cpu_percent', 'latency_ms', 'overload_flag',
        ],
        'Model C (CPU + Queue Dynamics)': [
            'cpu_percent', 'incoming_rate', 'processing_rate',
            'queue_growth_rate', 'processing_efficiency',
        ],
        'Model D (CPU + QPI)': [
            'cpu_percent', 'qpi', 'cpu_zscore', 'qpi_zscore',
        ],
        'Model E (Full Queue-Aware)': [
            'cpu_percent', 'latency_ms', 'overload_flag',
            'incoming_rate', 'processing_rate',
            'queue_growth_rate', 'processing_efficiency',
            'qpi', 'cpu_zscore', 'qpi_zscore',
        ],
    }

    # Strip features not present in df
    model_defs = {n: [f for f in feats if f in df.columns]
                  for n, feats in model_defs.items()}

    # ── Temporal train / test split ───────────────────────────────────────────
    split_idx = int(len(df) * 0.70)
    for frac in [0.70, 0.65, 0.60, 0.75, 0.80, 0.50]:
        cand = int(len(df) * frac)
        if (df['target'].iloc[:cand].nunique() >= 2
                and df['target'].iloc[cand:].nunique() >= 2):
            split_idx = cand
            break

    y_train_full = df['target'].iloc[:split_idx].values
    y_test       = df['target'].iloc[split_idx:].values
    ts_test      = df['timestamp'].iloc[split_idx:].values

    pf_mask            = df['target'] == 1
    first_failure_time = float(df.loc[pf_mask, 'timestamp'].min()) if pf_mask.any() else None
    test_span_h = max(1.0, (ts_test[-1] - ts_test[0]) / 3600.0) if len(ts_test) > 1 else 1.0

    print('  Train    : %d rows (%d PRE_FAILURE) | Test: %d rows (%d PRE_FAILURE)'
          % (split_idx, int(y_train_full.sum()), len(df) - split_idx, int(y_test.sum())))
    print('  Classifier: GradientBoosting | Balancing: class_weight + SMOTE | Lookahead: %ds'
          % LOOKAHEAD_SEC)
    print('-' * 90)
    print('%-35s | %-9s | %-8s | %-7s | %-5s | %-5s | %-9s | %-7s'
          % ('Model', 'Precision', 'Recall', 'F1', 'FP', 'FN', 'LeadTime', 'FA/hr'))
    print('-' * 90)
    print('NOTE: PRE_FAILURE = overload (CPU>=90%% or platform failure) within 30 s, per-service.')
    print('-' * 90)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _make_clf():
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )

    def _proba1(clf, X):
        X = np.asarray(X)
        if X.shape[0] == 0:
            return np.zeros(0)
        p = clf.predict_proba(X)
        if p.shape[1] == 1:
            return np.ones(p.shape[0]) if clf.classes_[0] == 1 else np.zeros(p.shape[0])
        return p[:, list(clf.classes_).index(1)]

    def _smote(X_tr, y_tr):
        if not SMOTE_AVAILABLE:
            return X_tr, y_tr
        counts = np.bincount(y_tr.astype(int))
        if len(counts) < 2 or counts.min() < 2:
            return X_tr, y_tr
        try:
            k = min(5, counts.min() - 1)
            return SMOTE(random_state=42, k_neighbors=k).fit_resample(X_tr, y_tr)
        except Exception:
            return X_tr, y_tr

    # ── Per-model loop ────────────────────────────────────────────────────────
    results = []
    feature_importances = {}

    for name, features in model_defs.items():
        if not features:
            print('%-35s | SKIPPED — no features in dataset' % name)
            continue

        X_tr_raw = (df[features].iloc[:split_idx].values.copy()).astype(float)
        X_te_raw = (df[features].iloc[split_idx:].values.copy()).astype(float)

        # Impute NaNs with training medians (no leakage)
        X_tr_imp, X_te_imp = _impute(X_tr_raw, X_te_raw)

        # Apply SMOTE to balanced training set for OOF threshold search
        X_bal, y_bal = _smote(X_tr_imp, y_train_full)

        # ── OOF threshold selection ───────────────────────────────────────────
        oof_proba = np.zeros(len(X_bal))
        oof_mask  = np.zeros(len(X_bal), dtype=bool)
        threshold = 0.5
        try:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for tri, vai in skf.split(X_bal, y_bal):
                if len(np.unique(y_bal[tri])) >= 2:
                    fc = _make_clf()
                    fc.fit(X_bal[tri], y_bal[tri])
                    oof_proba[vai] = _proba1(fc, X_bal[vai])
                    oof_mask[vai]  = True
        except Exception:
            pass

        y_oof = y_bal[oof_mask]
        if oof_mask.sum() > 30 and len(np.unique(y_oof)) >= 2:
            prec_c, rec_c, thr_c = precision_recall_curve(y_oof, oof_proba[oof_mask])
            f1_c = 2 * prec_c * rec_c / np.maximum(prec_c + rec_c, 1e-9)
            cands = [(t, f, r) for t, f, r in
                     zip(thr_c, f1_c[:len(thr_c)], rec_c[:len(thr_c)])
                     if 0.05 <= t <= 0.95 and r >= 0.35]
            if not cands:
                cands = [(t, f, r) for t, f, r in
                         zip(thr_c, f1_c[:len(thr_c)], rec_c[:len(thr_c)])
                         if 0.05 <= t <= 0.95]
            if cands:
                best = max(cands, key=lambda x: (round(x[1], 4), x[2], -abs(x[0] - 0.5)))
                threshold = float(best[0])

        # ── Final model on full training set ─────────────────────────────────
        # Train on original (not SMOTE-inflated) data with sample_weight for balance
        pos_w = (y_train_full == 0).sum() / max(1, (y_train_full == 1).sum())
        sample_weight = np.where(y_train_full == 1, pos_w, 1.0)

        clf = _make_clf()
        clf.fit(X_tr_imp, y_train_full, sample_weight=sample_weight)

        # Store feature importances for Model E
        if 'Full Queue-Aware' in name:
            feature_importances = dict(zip(features, clf.feature_importances_))

        proba_test = _proba1(clf, X_te_imp)
        preds      = (proba_test >= threshold).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average='binary', pos_label=1, zero_division=0)
        acc = accuracy_score(y_test, preds)
        cm  = confusion_matrix(y_test, preds, labels=[0, 1])
        fp  = int(cm[0][1]) if cm.shape == (2, 2) else 0
        fn  = int(cm[1][0]) if cm.shape == (2, 2) else 0

        alert_idx = np.where(preds == 1)[0]
        if len(alert_idx) > 0 and first_failure_time is not None:
            lead_time = max(0, int(round(first_failure_time - float(ts_test[alert_idx[0]]))))
        else:
            lead_time = 0

        fa_per_hr = round(fp / test_span_h, 2)

        print('%-35s | %8.1f%% | %7.1f%% | %6.3f | %5d | %5d | %+8ds | %7.2f | thr=%.3f'
              % (name, prec*100, rec*100, f1, fp, fn, lead_time, fa_per_hr, threshold))

        results.append({
            'model_name':            name,
            'model':                 name,
            'features':              features,
            'precision':             float(prec),
            'recall':                float(rec),
            'f1_score':              float(f1),
            'f1':                    float(f1),
            'threshold':             float(threshold),
            'fp':                    fp,
            'fn':                    fn,
            'lead_time':             lead_time,
            'mean_lead_s':           lead_time,
            'false_alarms_per_hour': fa_per_hr,
            'accuracy':              float(acc),
        })

    print('-' * 90)

    # ── Feature importance for Model E ────────────────────────────────────────
    if feature_importances:
        print('\n  Model E - Feature Importance (GradientBoosting):')
        for feat, imp in sorted(feature_importances.items(), key=lambda x: -x[1]):
            bar = '#' * int(imp * 40)
            print('    %-28s %.4f  %s' % (feat, imp, bar))

    print('\n[+] RESEARCH COMPARISON FINDINGS:')
    print('    PRE_FAILURE = overload (CPU>=90%% or platform failure) within 30 s (per service).')
    print('    Classifier: GradientBoosting (200 trees, lr=0.1, depth=4, subsample=0.8).')
    print('    Balancing: sample_weight proportional to class imbalance + SMOTE for OOF.')
    print('    Threshold: OOF 5-fold PR-curve optimisation (recall floor 0.35).')
    print('    Model E adds cpu_zscore, qpi_zscore over Model C.')
    print('    Note: in_flight_queue excluded - corr(overload)=-0.695 (anti-signal).')
    print('=' * 90 + '\n')

    # ── Persist cache ─────────────────────────────────────────────────────────
    try:
        clean = [{k: (v.item() if hasattr(v, 'item') else v)
                  for k, v in r.items() if k != 'features'}
                 for r in results]
        cache = os.path.join(SCRIPT_DIR, 'benchmark_results.json')
        with open(cache, 'w') as fh:
            json.dump({'status': 'success', 'results': clean,
                       'timestamp': _time.time()}, fh, indent=2)
    except Exception:
        pass

    return pd.DataFrame(results)


if __name__ == '__main__':
    evaluate_5_models()
