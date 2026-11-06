#!/usr/bin/env python3
"""
Phase 2 verification:
  (1) reconstruction quality: reconstructed yaw/lean vs ground truth columns
  (2) per-class feature statistics + effect sizes (Cohen's d)
  (3) rider-level ablation: Hs-only vs full features (logistic regression)
Usage: python check_features.py --data data
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score
import steering_entropy as se

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); a = ap.parse_args()
fs = json.load(open(f"{a.data}/metadata.json"))["fs"]

print("=== (1) Reconstruction vs ground truth (3 sessions) ===")
raw = pd.read_csv(f"{a.data}/synthetic_imu.csv")
for sid in raw.session_id.unique()[:3]:
    g = raw[raw.session_id == sid].iloc[int(15 * fs):]            # skip filter warm-up
    full = raw[raw.session_id == sid]
    yaw = se.reconstruct_yaw(full.gz.values, fs)[int(15 * fs):]
    lean = se.reconstruct_lean(full.ay.values, full.az.values, full.gx.values, fs)[int(15 * fs):]
    gt_yaw = se.highpass(g.yaw_deg.values, fs, 0.03)
    gt_lean = se.highpass(g.lean_deg.values, fs, 0.03)
    c = lambda x, y: np.corrcoef(x, y)[0, 1]
    hf = lambda x: se.highpass(x, fs, 0.3)      # band where rider corrections live (>0.3 Hz)
    print(f"{sid} [{g['mode'].iloc[0]:10s}] yaw: all={c(yaw, gt_yaw):.3f} >0.3Hz={c(hf(yaw), hf(gt_yaw)):.3f} | "
          f"lean: all={c(lean, gt_lean):.3f} >0.3Hz={c(hf(lean), hf(gt_lean)):.3f}")

print("\n=== (2) Feature statistics by class ===")
f = pd.read_csv(f"{a.data}/features.csv")
cols = ["Hs", "jerk_mean", "rev_rate", "lean_var", "elapsed_min"]
for c in cols:
    x0, x1 = f[f.label == 0][c], f[f.label == 1][c]
    d = (x1.mean() - x0.mean()) / np.sqrt((x0.var() + x1.var()) / 2)
    print(f"{c:12s} alert={x0.mean():8.3f}  drowsy={x1.mean():8.3f}  Cohen d={d:+.2f}")
print("NaNs:", int(f[cols].isna().sum().sum()))

print("\n=== (3) Rider-level ablation (train->test, logistic regression) ===")
tr, te = f[f.split == "train"], f[f.split == "test"]
print(f"train windows={len(tr)} test windows={len(te)} test riders={te.rider_id.nunique()}")
for name, cs in [("Hs only", ["Hs"]),
                 ("Hs + jerk", ["Hs", "jerk_mean"]),
                 ("all (no elapsed)", ["Hs", "jerk_mean", "rev_rate", "lean_var"]),
                 ("all + elapsed", cols)]:
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    m.fit(tr[cs], tr.label)
    p = m.predict_proba(te[cs])[:, 1]
    print(f"{name:18s} AUC={roc_auc_score(te.label, p):.3f}  F1={f1_score(te.label, p > .5):.3f}")
