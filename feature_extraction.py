#!/usr/bin/env python3
"""
Phase 2b: Windowed feature extraction.

Reads Phase 1 CSV -> reconstructs steering proxy from RAW IMU only -> slides a
window -> computes Hs + complementary features -> writes
  data/features.csv   (tabular: for baselines / ablation / sanity checks)
  data/windows.npz    (tensors for the Phase 3 1D-CNN)
  data/feature_config.json  (alpha, window params - needed again at inference)

Usage:
  python feature_extraction.py --data data/synthetic_imu.csv --out data --win 30 --hop 15 --w_lean 0.0
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
import steering_entropy as se

TARGET_FS = 10   # Hz for steering-proxy analysis


# ------------------------------------------------------------ per-window features
def jerk_features(acc_xyz, fs, fc=8.0):
    """Jerk magnitude = |d(a)/dt| (m/s^3). Accelerometer is low-passed at `fc` Hz
    FIRST: engine vibration (~20 Hz) and sensor noise otherwise dominate the
    derivative and hide the rider's actual corrections."""
    sos = butter(2, fc, btype="lowpass", fs=fs, output="sos")
    a = sosfilt(sos, acc_xyz, axis=0)
    j = np.linalg.norm(np.diff(a, axis=0) * fs, axis=1)
    return float(j.mean()), float(np.percentile(j, 95))


def reversal_rate(sig, fs, gap_deg=0.5):
    """Steering reversals per minute: a reversal is counted when the signal turns
    around after moving more than `gap_deg` from its last extreme (classic SRR;
    ignores sensor jitter)."""
    rev, direction, ext = 0, 0, sig[0]
    for v in sig[1:]:
        if direction == 0:
            if v - ext > gap_deg:
                direction, ext = 1, v
            elif ext - v > gap_deg:
                direction, ext = -1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v > gap_deg:
                rev += 1; direction, ext = -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext > gap_deg:
                rev += 1; direction, ext = 1, v
    return rev / (len(sig) / fs / 60.0)


def window_features(w, alpha):
    """w: dict of aligned arrays for one window."""
    jm, j95 = jerk_features(w["acc"], w["fs_raw"])
    return dict(
        Hs=se.steering_entropy(w["err"], alpha),
        jerk_mean=jm, jerk_p95=j95,
        rev_rate=reversal_rate(w["proxy"], TARGET_FS),
        lean_var=float(np.var(w["lean"])),
        elapsed_min=float(w["elapsed_min"]),
    )


# ------------------------------------------------------------ per-session prep
def prep_session(g, fs, w_lean):
    """Reconstruct signals from raw IMU only."""
    yaw = se.reconstruct_yaw(g["gz"].values, fs)
    lean = se.reconstruct_lean(g["ay"].values, g["az"].values, g["gx"].values, fs)
    proxy = se.steering_proxy(yaw, lean, w_lean)
    q = int(round(fs / TARGET_FS))
    return dict(
        proxy=se.downsample(proxy, fs, TARGET_FS),
        lean=se.downsample(lean, fs, TARGET_FS),
        acc=g[["ax", "ay", "az"]].values,
        q=q,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_imu.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--win", type=float, default=30.0, help="window seconds")
    ap.add_argument("--hop", type=float, default=15.0, help="hop seconds")
    ap.add_argument("--w_lean", type=float, default=0.0, help="lean weight in steering proxy")
    ap.add_argument("--warmup", type=float, default=15.0, help="seconds dropped at session start (filter settling)")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(os.path.dirname(a.data), "metadata.json")))
    fs = int(meta["fs"])
    df = pd.read_csv(a.data)
    L = int(a.win * TARGET_FS)
    H = int(a.hop * TARGET_FS)
    W0 = int(a.warmup * TARGET_FS)

    # ---- pass 1: reconstruct every session; calibrate alpha on ALERT TRAIN only
    sessions = {}
    for sid, g in df.groupby("session_id", sort=False):
        s = prep_session(g, fs, a.w_lean)
        s["err"] = se.prediction_error(s["proxy"])
        s["meta"] = g.iloc[0][["rider_id", "split", "mode"]].to_dict()
        s["label"] = se.downsample(g["label"].values.astype(float), fs, TARGET_FS)
        s["kss"] = se.downsample(g["kss"].values.astype(float), fs, TARGET_FS)
        s["t"] = g["t"].values[::s["q"]][: len(s["proxy"])]
        sessions[sid] = s
    base = [s["err"][W0:] for s in sessions.values()
            if s["meta"]["mode"] == "alert" and s["meta"]["split"] == "train"]
    alpha = se.calibrate_alpha(base)
    print(f"alpha (alert-train 90th pct |e|) = {alpha:.4f} deg")

    # ---- pass 2: windows
    rows, X, F = [], [], []
    for sid, s in sessions.items():
        n = len(s["proxy"])
        yaw_rate = np.gradient(s["proxy"], 1 / TARGET_FS)
        for st in range(W0, n - L + 1, H):
            sl = slice(st, st + L)
            raw = slice(st * s["q"], (st + L) * s["q"])
            w = dict(proxy=s["proxy"][sl], lean=s["lean"][sl], err=s["err"][sl],
                     acc=s["acc"][raw], fs_raw=fs, elapsed_min=s["t"][sl][-1] / 60)
            f = window_features(w, alpha)
            rows.append(dict(rider_id=int(s["meta"]["rider_id"]), session_id=sid,
                             split=s["meta"]["split"], mode=s["meta"]["mode"],
                             t_end=float(s["t"][sl][-1]), **f,
                             label=int(s["label"][sl].mean() >= 0.5),
                             kss_mean=float(s["kss"][sl].mean())))
            # CNN tensor channels: proxy, lean, yaw-rate, prediction error
            X.append(np.stack([w["proxy"], w["lean"], yaw_rate[sl], w["err"]]))
            F.append([f["Hs"], f["jerk_mean"], f["rev_rate"], f["lean_var"], f["elapsed_min"]])

    feats = pd.DataFrame(rows)
    feats.to_csv(os.path.join(a.out, "features.csv"), index=False)
    np.savez_compressed(
        os.path.join(a.out, "windows.npz"),
        X=np.asarray(X, np.float32), F=np.asarray(F, np.float32),
        y=feats["label"].values.astype(np.float32),
        rider=feats["rider_id"].values, split=feats["split"].values.astype(str),
        feature_names=np.array(["Hs", "jerk_mean", "rev_rate", "lean_var", "elapsed_min"]),
        channel_names=np.array(["steer_proxy", "lean", "steer_rate", "pred_error"]),
    )
    json.dump(dict(alpha=alpha, win_s=a.win, hop_s=a.hop, target_fs=TARGET_FS,
                   w_lean=a.w_lean, raw_fs=fs, warmup_s=a.warmup),
              open(os.path.join(a.out, "feature_config.json"), "w"), indent=2)
    print(f"{len(feats)} windows | X shape {np.asarray(X).shape}")
    print(feats.groupby("label")[["Hs", "jerk_mean", "rev_rate", "lean_var"]].mean().round(3))


if __name__ == "__main__":
    main()
