#!/usr/bin/env python3
"""
Phase 3b: Classical baseline mirroring Li et al. 2017 (reference [1] in the report).

  steering proxy (10 Hz) -> sliding Approximate Entropy (ApEn, m=2, r=0.2*SD)
  -> APLA smoothing of the ApEn series -> DTW distance to a self-learned
  "awake" reference series -> larger distance = more drowsy.

It is UNSUPERVISED apart from one scalar threshold picked on the validation set.
The reference series is the mean APLA-smoothed ApEn sequence of ALERT TRAIN windows.

Usage:
  python apla_dtw_baseline.py --data data --name baseline
  python apla_dtw_baseline.py --data data --name baseline_cv --cv 5
"""
import argparse, json, os
import numpy as np, pandas as pd
import feature_extraction as fx
from eval_utils import best_threshold, compute_metrics, cv_splits, summarize


# ------------------------------------------------------------ algorithms
def apen(x, m=2, r_factor=0.2, r=None):
    """Approximate entropy (Pincus). Higher = less regular.
    r=None -> r = r_factor*SD of THIS window (paper default, scale-invariant).
    r=<float> -> fixed tolerance (e.g. from alert baseline), sensitive to amplitude."""
    x = np.asarray(x, float)
    r = r_factor * x.std() if r is None else r
    if r == 0:
        return 0.0
    def phi(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(-1)
        return np.log((d <= r).mean(1)).mean()
    return float(phi(m) - phi(m + 1))


def apla(y, tol=0.05):
    """Adaptive piecewise linear approximation: greedily extend each linear
    segment (endpoint-to-endpoint) while max deviation <= tol."""
    y = np.asarray(y, float); n = len(y); out = y.copy(); s = 0
    def dev(a, b):
        return np.max(np.abs(y[a:b + 1] - np.linspace(y[a], y[b], b - a + 1)))
    while s < n - 1:
        e = s + 1
        while e + 1 < n and dev(s, e + 1) <= tol:
            e += 1
        out[s:e + 1] = np.linspace(y[s], y[e], e - s + 1)
        s = e
    return out


def dtw(a, b):
    """Classic DTW with |.| cost; returned distance is normalised by (n+m)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


# ------------------------------------------------------------ pipeline
def apen_sequences(csv_path, feats, cfg, sub_s=10, sub_hop_s=2, tol=0.05, fixed_r=False):
    """For each feature window -> smoothed ApEn sequence (K points)."""
    fs, tfs, L = cfg["raw_fs"], cfg["target_fs"], int(cfg["win_s"] * cfg["target_fs"])
    sub, hop = int(sub_s * tfs), int(sub_hop_s * tfs)
    df = pd.read_csv(csv_path)
    proxies = {sid: fx.prep_session(g, fs, cfg["w_lean"])["proxy"] for sid, g in df.groupby("session_id", sort=False)}
    r = None
    if fixed_r:   # r = 0.2 * SD of the steering proxy over ALERT TRAIN sessions
        ok = set(feats[(feats["mode"] == "alert") & (feats.split == "train")].session_id)
        r = 0.2 * float(np.std(np.concatenate([proxies[k] for k in ok])))
        print(f"fixed ApEn tolerance r = {r:.3f} deg")
    seqs = []
    for sid, t_end in zip(feats.session_id, feats.t_end):
        end = int(round(t_end * tfs)) + 1
        w = proxies[sid][end - L:end]
        raw = np.array([apen(w[i:i + sub], r=r) for i in range(0, L - sub + 1, hop)])
        seqs.append(apla(raw, tol))
    return np.array(seqs)


def evaluate_fold(seqs, y, tr, va, te, hop_s, kss):
    ref = seqs[tr][y[tr] == 0].mean(0)                       # self-learned "awake" reference
    score = np.array([dtw(s, ref) for s in seqs])
    scale = np.percentile(score[tr], 95)
    p = np.clip(score / scale, 0, 1.5)                       # monotone map to ~[0,1]
    thr = best_threshold(y[va], p[va])
    return compute_metrics(y[te], p[te], thr, hop_s, kss[te]), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default="baseline"); ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--fixed_r", type=int, default=0, help="1: fixed ApEn tolerance from alert baseline")
    ap.add_argument("--cv", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = json.load(open(f"{a.data}/feature_config.json"))
    feats = pd.read_csv(f"{a.data}/features.csv")
    y, rider, split, kss = feats.label.values, feats.rider_id.values, feats.split.values, feats.kss_mean.values
    print("computing ApEn sequences (a minute or so)...")
    seqs = apen_sequences(f"{a.data}/synthetic_imu.csv", feats, cfg, tol=a.tol, fixed_r=bool(a.fixed_r))
    run_dir = os.path.join(a.out, a.name); os.makedirs(run_dir, exist_ok=True)

    if a.cv:
        ms = []
        for k, (tr, va, te) in enumerate(cv_splits(rider, a.cv, a.seed)):
            m, _ = evaluate_fold(seqs, y, tr, va, te, cfg["hop_s"], kss)
            print(f"fold {k}: AUC={m['auc']:.3f} F1={m['f1']:.3f} FA/h={m['false_alarms_per_hour']:.1f}")
            ms.append(m)
        summ = summarize(ms)
        print("\nCV mean +/- std:")
        for k in ["auc", "f1", "precision", "recall", "false_alarms_per_hour", "pearson_p_vs_kss"]:
            print(f"  {k:24s} {summ[k][0]:.3f} +/- {summ[k][1]:.3f}")
        json.dump(dict(mode="cv", folds=ms, summary=summ), open(f"{run_dir}/metrics.json", "w"), indent=2)
    else:
        tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
        m, _ = evaluate_fold(seqs, y, tr, va, te, cfg["hop_s"], kss)
        print(f"\nBASELINE TEST ({len(np.unique(rider[te]))} riders)")
        for k in ["auc", "f1", "precision", "recall", "false_alarms_per_hour", "miss_rate", "pearson_p_vs_kss"]:
            print(f"  {k:24s} {m[k]:.3f}")
        json.dump(dict(mode="single", metrics=m), open(f"{run_dir}/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
