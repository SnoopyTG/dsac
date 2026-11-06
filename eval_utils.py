#!/usr/bin/env python3
"""
Shared evaluation helpers so the CNN and the APLA+DTW baseline are scored with
IDENTICAL splits, thresholds and metrics (otherwise the comparison is meaningless).
"""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def best_threshold(y, p):
    """Threshold maximising F1 on the VALIDATION set (never on test)."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    cands = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    f1s = [f1_score(y, p >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def compute_metrics(y, p, thr, hop_s, kss=None):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    neg_hours = (y == 0).sum() * hop_s / 3600.0          # alert riding time
    m = dict(
        auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        f1=float(f1_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        false_alarms_per_hour=float(fp / neg_hours) if neg_hours > 0 else float("nan"),
        miss_rate=float(fn / max(1, tp + fn)),
        threshold=float(thr), n_windows=int(len(y)), n_riders_note="see run config",
    )
    if kss is not None and np.std(p) > 0 and np.std(kss) > 0:
        # stand-in for the PERCLOS correlation benchmark until real PERCLOS exists
        m["pearson_p_vs_kss"] = float(np.corrcoef(p, kss)[0, 1])
    return m


def cv_splits(riders, k=5, seed=0, val_frac=0.2):
    """Leave-riders-out K-fold. Yields (train_idx, val_idx, test_idx).
    Splitting is by RIDER, so no rider is ever in two sets."""
    riders = np.asarray(riders)
    uniq = np.random.default_rng(seed).permutation(np.unique(riders))
    folds = np.array_split(uniq, k)
    rng = np.random.default_rng(seed + 1)
    for i in range(k):
        test_r = folds[i]
        rest = np.concatenate([folds[j] for j in range(k) if j != i])
        rest = rng.permutation(rest)
        n_val = max(1, int(round(val_frac * len(rest))))
        val_r, train_r = rest[:n_val], rest[n_val:]
        yield (np.flatnonzero(np.isin(riders, train_r)),
               np.flatnonzero(np.isin(riders, val_r)),
               np.flatnonzero(np.isin(riders, test_r)))


def summarize(list_of_metrics):
    keys = [k for k in list_of_metrics[0] if isinstance(list_of_metrics[0][k], float)]
    return {k: (float(np.nanmean([m[k] for m in list_of_metrics])),
                float(np.nanstd([m[k] for m in list_of_metrics]))) for k in keys}
