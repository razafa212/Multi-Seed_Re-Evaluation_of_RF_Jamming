"""
common.py — Shared pipeline functions for the revision experiments.

These functions are copied VERBATIM from the author's notebooks
(AMSTE_25ms.ipynb / AMSTE_15ms.ipynb) so that every revision experiment
uses exactly the same pseudo-episode construction, episode split, windowing,
tabular-feature extraction, and metric computation as the original paper.
Do NOT modify these unless you also re-run the main results.

Two datasets are used by the original code:
    25 m/s  ->  Dataset_1.csv
    15 m/s  ->  Dataset_2.csv
Set the paths in CONFIG below (or pass them to load_dataset()).
"""
import os, json, glob, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

# ─────────────────────────────────────────────────────────────────────────
# CONFIG  (verbatim values from the notebooks)
# ─────────────────────────────────────────────────────────────────────────
SEEDS          = [42, 123, 456, 789, 2024]
FEATURES       = ['SNR', 'RSSI', 'PDR', 'Speed', 'Relative_Speed']
WINDOW_SIZES   = [1, 5, 15]
EPISODE_LENGTH = 60          # seconds per pseudo-episode
WINDOW_SEC     = 15          # window used by RF-Stat / XGBoost-only / LSTM
K_NEIGHBORS    = 5
SPEED_RANGE    = {15: (13, 17), 25: (23, 27)}
SCENARIO_TO_LABEL = {1: 0, 2: 1, 3: 2}   # Interference / Reactive / Constant

# EDIT THESE to point at your local copies of the dataset:
DATASET_PATHS = {
    25: os.environ.get('DATASET_25', 'Dataset_1.csv'),
    15: os.environ.get('DATASET_15', 'Dataset_2.csv'),
}


# ─────────────────────────────────────────────────────────────────────────
# Dataset loading  (mirrors notebook Cell 2)
# ─────────────────────────────────────────────────────────────────────────
def load_dataset(target_speed, path=None):
    """Load CSV, map labels, build pseudo_run_id, filter to one speed band."""
    if path is None:
        path = DATASET_PATHS[target_speed]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset for {target_speed} m/s not found at '{path}'. "
            f"Set DATASET_PATHS[{target_speed}] in common.py or the "
            f"DATASET_{target_speed} environment variable.")
    df = pd.read_csv(path)
    df['label'] = df['Scenario'].map(SCENARIO_TO_LABEL)
    df = df.sort_values(['Speed', 'Scenario', 'Time']).reset_index(drop=True)
    df['pseudo_run_id'] = (
        df['Speed'].round(2).astype(str) + '_' +
        df['Scenario'].astype(str) + '_' +
        (df['Time'] // EPISODE_LENGTH).astype(int).astype(str)
    )
    lo, hi = SPEED_RANGE[target_speed]
    df_speed = df[(df['Speed'] >= lo) & (df['Speed'] <= hi)].copy()
    return df_speed


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# Episode split  (VERBATIM from notebook)
# ─────────────────────────────────────────────────────────────────────────
def split_by_episode(df, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Stratified episode-based split (per-class shuffle)."""
    rng = np.random.RandomState(seed)
    ep_to_label = (df.groupby('pseudo_run_id')['label']
                     .agg(lambda x: x.iloc[0]).to_dict())
    eps_by_class = {}
    for ep, lbl in ep_to_label.items():
        eps_by_class.setdefault(lbl, []).append(ep)

    train_eps, val_eps, test_eps = [], [], []
    for lbl, eps in eps_by_class.items():
        eps = np.array(eps)
        rng.shuffle(eps)
        n = len(eps)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        train_eps.extend(eps[:n_train].tolist())
        val_eps.extend(eps[n_train:n_train + n_val].tolist())
        test_eps.extend(eps[n_train + n_val:].tolist())

    train_df = df[df['pseudo_run_id'].isin(train_eps)].copy()
    val_df   = df[df['pseudo_run_id'].isin(val_eps)].copy()
    test_df  = df[df['pseudo_run_id'].isin(test_eps)].copy()
    assert not (set(train_eps) & set(test_eps)), 'LEAKAGE!'
    assert not (set(train_eps) & set(val_eps)),  'LEAKAGE!'
    assert not (set(val_eps)   & set(test_eps)), 'LEAKAGE!'
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────
# Windowing  (VERBATIM — non-overlapping, step = window length)
# ─────────────────────────────────────────────────────────────────────────
def create_windows(df, window_sec, step_sec=None):
    if step_sec is None:
        step_sec = window_sec
    X_list, y_list, ep_list = [], [], []
    for run_id, group in df.groupby('pseudo_run_id'):
        group = group.sort_values('Time').reset_index(drop=True)
        times = group['Time'].values
        X_raw = group[FEATURES].values
        y_raw = group['label'].values
        t0, t_end = times[0], times[-1]
        while t0 + window_sec <= t_end:
            t1 = t0 + window_sec
            mask = (times >= t0) & (times < t1)
            if mask.sum() > 0:
                X_list.append(X_raw[mask])
                y_list.append(np.bincount(y_raw[mask]).argmax())
                ep_list.append(run_id)
            t0 += step_sec
    return X_list, np.array(y_list), ep_list


def pad_sequences(X_list, max_len=None):
    if max_len is None:
        max_len = max(len(x) for x in X_list)
    n_feat = X_list[0].shape[1]
    out = np.zeros((len(X_list), max_len, n_feat), dtype=np.float32)
    for i, x in enumerate(X_list):
        sl = min(len(x), max_len)
        out[i, :sl, :] = x[:sl]
    return out, max_len


def pad_and_normalize(X_train_list, X_val_list, X_test_list):
    max_len = max(len(x) for x in X_train_list)
    n_features = X_train_list[0].shape[1]
    def pad(X_list):
        out = np.zeros((len(X_list), max_len, n_features), dtype=np.float32)
        for i, x in enumerate(X_list):
            sl = min(len(x), max_len)
            out[i, :sl, :] = x[:sl]
        return out
    Xtr, Xv, Xte = pad(X_train_list), pad(X_val_list), pad(X_test_list)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr.reshape(-1, n_features)).reshape(Xtr.shape)
    Xv  = sc.transform(Xv.reshape(-1, n_features)).reshape(Xv.shape)
    Xte = sc.transform(Xte.reshape(-1, n_features)).reshape(Xte.shape)
    return Xtr, Xv, Xte, max_len


# ─────────────────────────────────────────────────────────────────────────
# Tabular features  (VERBATIM — 7 stats x 5 features = 35 dims)
# ─────────────────────────────────────────────────────────────────────────
def extract_tabular_features(X_padded):
    out = []
    for x in X_padded:
        m = ~(x == 0).all(axis=1)
        xv = x[m] if m.sum() > 0 else x
        f = []
        for c in range(xv.shape[1]):
            v = xv[:, c]
            f.extend([np.mean(v), np.std(v), np.min(v), np.max(v),
                      np.median(v), np.percentile(v, 25), np.percentile(v, 75)])
        out.append(f)
    return np.array(out)


# 7 summary statistics per feature, in the order produced above:
STATS_PER_FEATURE = 7
def feature_block_indices(feature_name):
    """Return the 7 column indices in the 35-dim vector for one raw feature."""
    c = FEATURES.index(feature_name)
    return list(range(c * STATS_PER_FEATURE, (c + 1) * STATS_PER_FEATURE))


def select_tabular_columns(X35, drop_features=()):
    """Return a copy of the 35-dim tabular matrix with the columns belonging
    to `drop_features` removed (used for the no-relative-speed ablations)."""
    drop = set()
    for fn in drop_features:
        drop.update(feature_block_indices(fn))
    keep = [i for i in range(X35.shape[1]) if i not in drop]
    return X35[:, keep], keep


# ─────────────────────────────────────────────────────────────────────────
# Metrics  (VERBATIM)
# ─────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, num_classes=3):
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    labels = list(range(num_classes))
    p_pc, r_pc, f1_pc, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return dict(accuracy=acc, precision_macro=float(p),
                recall_macro=float(r), f1_macro=float(f1),
                precision_per_class=p_pc.tolist(),
                recall_per_class=r_pc.tolist(),
                f1_per_class=f1_pc.tolist(),
                confusion_matrix=cm)


def agg(values, pct=True):
    """mean ± sample-std (ddof=1) helper, matching the notebooks."""
    a = np.asarray(values, dtype=float)
    m, s = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)
    if pct:
        m, s = m * 100, s * 100
    return m, s


CLASS_NAMES = ['Interference', 'Reactive', 'Constant']
