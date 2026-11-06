#!/usr/bin/env python3
"""
Phase 1: Synthetic data & physics engine for two-wheeler fatigue detection.

Generates multi-channel 6-axis IMU data (ax, ay, az, gx, gy, gz) mimicking an
MPU6050 mounted on a handlebar, for ALERT vs FATIGUED riding.

Behavioural model (the hypothesis from Progress Report I, Sec 4.2):
  alert    -> small, tightly distributed, frequent corrections, fast reaction
  fatigued -> slow drift (under-correction) punctuated by sharp, bursty
              over-corrections, micro-sleep lapses, delayed reaction,
              larger lean wobble.
A latent fatigue level f(t) in [0,1] drives every parameter; KSS = 1 + 8 f.

Usage:
  python data_generator.py --out data --riders 20 --fs 50 --duration 300 --validate --plot
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.signal import lfilter

G = 9.81
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


# ----------------------------------------------------------------- helpers
def ou(n, fs, tau, sigma, rng):
    """Ornstein-Uhlenbeck / AR(1) coloured noise with std=sigma, time const tau (s)."""
    a = np.exp(-1.0 / (fs * tau))
    w = rng.standard_normal(n) * sigma * np.sqrt(1 - a * a)
    return lfilter([1.0], [1.0, -a], w)


def burst_kernel(fs, dur=2.0, decay=0.45, freq=1.2):
    t = np.arange(int(dur * fs)) / fs
    return np.exp(-t / decay) * np.sin(2 * np.pi * freq * t)


def add_events(sig, idx, amps, kernel):
    for i, a in zip(idx, amps):
        end = min(len(sig), i + len(kernel))
        sig[i:end] += a * kernel[: end - i]
    return sig


def fatigue_profile(mode, n, fs, rng):
    """Latent fatigue f(t) in [0,1]."""
    t = np.linspace(0, 1, n)
    if mode == "alert":
        f = 0.05 + 0.05 * ou(n, fs, 30, 0.5, rng)
    elif mode == "fatigued":
        f = 0.85 + 0.05 * ou(n, fs, 30, 0.5, rng)
    else:  # transition: alert -> fatigued
        f = 1 / (1 + np.exp(-10 * (t - 0.5))) * 0.85 + 0.05
    return np.clip(f, 0, 1)


def new_rider(rng):
    """Rider-level variability so a model cannot just memorise one style."""
    return dict(
        style=float(rng.lognormal(0, 0.2)),           # steering amplitude scale
        speed=float(rng.uniform(7, 14)),              # m/s (~25-50 km/h)
        vib_freq=float(rng.uniform(18, 24)),          # engine vibration Hz
        vib_amp=float(rng.uniform(0.03, 0.10) * G),
        acc_bias=rng.normal(0, 0.04 * G, 3).tolist(),
        gyro_bias=rng.normal(0, 0.4, 3).tolist(),     # deg/s
        fatigue_sens=float(rng.uniform(0.8, 1.2)),    # how strongly fatigue shows
    )


# ----------------------------------------------------------------- core physics
def simulate_session(rider, mode, duration, fs, rng):
    n = int(duration * fs)
    t = np.arange(n) / fs
    f = np.clip(fatigue_profile(mode, n, fs, rng) * rider["fatigue_sens"], 0, 1)
    s = rider["style"]

    # 1) Intended path: road curvature -> desired handlebar yaw (deg)
    road = ou(n, fs, 8.0, 4.0 * s, rng)

    # 2) Alert corrections: small, frequent, fast. Delay grows with fatigue.
    corr = ou(n, fs, 0.4, 0.35 * s, rng)
    delay = int(round((0.15 + 0.35 * f.mean()) * fs))
    corr = np.roll(corr, delay)

    # 3) Fatigue effects
    drift = f * ou(n, fs, 6.0, 2.5, rng)              # under-correction drift
    kern = burst_kernel(fs)
    lam = (0.5 + 6.0 * f) / 60.0                       # bursts per second
    idx = np.flatnonzero(rng.random(n) < lam / fs)
    amps = rng.choice([-1, 1], len(idx)) * (2.0 + rng.lognormal(0.8, 0.5, len(idx))) * f[idx] * s
    bursts = add_events(np.zeros(n), idx, amps, kern)

    # micro-sleep lapses: corrections suppressed 2-4 s, then a violent recovery
    ms_mask = np.ones(n)
    lam_ms = (f ** 2) / 90.0
    ms_idx = np.flatnonzero(rng.random(n) < lam_ms / fs)
    ms = np.zeros(n)
    for i in ms_idx:
        L = int(rng.uniform(2, 4) * fs)
        ms_mask[i:i + L] = 0.1
        end = min(n - 1, i + L)
        add_events(ms, [end], [rng.choice([-1, 1]) * rng.uniform(8, 14) * s], kern)

    yaw = road + corr * (1 - 0.6 * f) * ms_mask + drift + bursts + ms   # handlebar yaw (deg)

    # 4) Lean dynamics: steering -> heading rate -> lateral accel -> lean
    v = rider["speed"] + ou(n, fs, 10, 0.8, rng)
    delta = np.deg2rad(yaw) * 0.3                      # effective steer angle
    heading_rate = v * np.sin(delta) / 1.3             # wheelbase 1.3 m
    a_lat = v * heading_rate
    lean_ss = np.rad2deg(np.arctan(a_lat / G))
    wobble = ou(n, fs, 0.7, 0.5 * (1 + 1.5 * f) * s, rng)   # balance wobble grows w/ fatigue
    lean = lean_ss + wobble

    # 5) Ideal IMU signals (sensor frame; documented simplification)
    dt = 1.0 / fs
    gz = np.gradient(yaw, dt)                          # handlebar yaw rate  (deg/s)
    gx = np.gradient(lean, dt)                         # roll rate           (deg/s)
    gy = ou(n, fs, 0.3, 1.5, rng) + 0.5 * np.gradient(v, dt)   # pitch from bumps/accel
    a_long = np.gradient(v, dt)
    vib = rider["vib_amp"] * np.sin(2 * np.pi * rider["vib_freq"] * t + rng.uniform(0, 6.28))
    bump = np.zeros(n)
    b_idx = np.flatnonzero(rng.random(n) < 0.15 / fs)
    add_events(bump, b_idx, rng.normal(0, 2.0, len(b_idx)), burst_kernel(fs, 0.4, 0.08, 4))

    ax = a_long + vib * 0.5
    ay = G * np.sin(np.deg2rad(wobble)) + 0.3 * a_lat + vib * 0.5
    az = G * np.cos(np.deg2rad(lean)) + vib + bump

    # 6) MPU6050-like sensor noise + bias
    acc_noise = 0.03   # m/s^2 (~400 ug/rtHz over ~50 Hz BW)
    gyr_noise = 0.04   # deg/s
    ab, gb = rider["acc_bias"], rider["gyro_bias"]
    out = dict(
        ax=ax + ab[0] + rng.normal(0, acc_noise, n),
        ay=ay + ab[1] + rng.normal(0, acc_noise, n),
        az=az + ab[2] + rng.normal(0, acc_noise, n),
        gx=gx + gb[0] + rng.normal(0, gyr_noise, n),
        gy=gy + gb[1] + rng.normal(0, gyr_noise, n),
        gz=gz + gb[2] + rng.normal(0, gyr_noise, n),
    )

    # 7) Labels: KSS (1-9) from latent f; binary label = KSS >= 7 (Report Sec 4.4)
    kss = np.clip(np.round(1 + 8 * f + rng.normal(0, 0.3, n).cumsum() * 0 ), 1, 9)
    df = pd.DataFrame(out)
    df.insert(0, "t", t)
    df["yaw_deg"] = yaw          # ground-truth aux signals (for debugging / baseline)
    df["lean_deg"] = lean
    df["fatigue_level"] = f
    df["kss"] = kss
    df["label"] = (kss >= 7).astype(int)
    df["elapsed_min"] = t / 60.0
    df["mode"] = mode
    return df


# ----------------------------------------------------------------- validation
def steering_entropy(alpha, alpha_ref_err=None, n_bins=9):
    """Nakayama-style steering entropy with 2nd-order Taylor prediction."""
    a = np.asarray(alpha)
    pred = a[2:-1] + (a[2:-1] - a[1:-2]) + 0.5 * ((a[2:-1] - a[1:-2]) - (a[1:-2] - a[:-3]))
    e = a[3:] - pred
    return e


def entropy_from_err(e, alpha90):
    edges = np.array([-5, -2.5, -1, -0.5, 0.5, 1, 2.5, 5]) * alpha90
    p = np.bincount(np.digitize(e, edges), minlength=9) / len(e)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def validate(df, fs):
    step = max(1, fs // 10)                            # decimate to ~10 Hz
    res = {}
    base = df[df["mode"] == "alert"]
    errs = np.concatenate([steering_entropy(g["yaw_deg"].values[::step]) for _, g in base.groupby("session_id")])
    a90 = np.percentile(np.abs(errs), 90)
    w = 30 * 10
    for mode, g in df.groupby("mode"):
        hs = []
        for _, sg in g.groupby("session_id"):
            y = sg["yaw_deg"].values[::step]
            for i in range(0, len(y) - w, w // 2):
                hs.append(entropy_from_err(steering_entropy(y[i:i + w]), a90))
        res[mode] = (np.mean(hs), np.std(hs))
    print("\nSteering-entropy sanity check (30 s windows, 10 Hz):")
    for m, (mu, sd) in res.items():
        print(f"  {m:10s} Hs = {mu:.3f} +/- {sd:.3f}")
    ok = res["fatigued"][0] > res["alert"][0]
    print("  PASS: fatigued > alert" if ok else "  FAIL: retune fatigue parameters")


def make_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
    for j, mode in enumerate(["alert", "fatigued"]):
        g = df[df["mode"] == mode]
        g = g[g["session_id"] == g["session_id"].iloc[0]]
        g = g[g["t"] < 60]
        ax[0, j].plot(g["t"], g["yaw_deg"]); ax[0, j].set_title(f"{mode}: handlebar yaw (deg)")
        ax[1, j].plot(g["t"], g["lean_deg"], color="tab:orange"); ax[1, j].set_title("lean (deg)")
        ax[2, j].plot(g["t"], g["gz"], color="tab:green"); ax[2, j].set_title("gz (deg/s)")
        ax[2, j].set_xlabel("time (s)")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--riders", type=int, default=20)
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--duration", type=float, default=300, help="seconds per session")
    ap.add_argument("--modes", default="alert,transition,fatigued")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    modes = a.modes.split(",")

    # rider-level split (70/15/15) -- never mix a rider across splits
    ids = rng.permutation(a.riders)
    n_tr, n_va = int(0.7 * a.riders), int(0.15 * a.riders)
    split = {int(r): ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, r in enumerate(ids)}

    frames, riders_meta = [], {}
    for r in range(a.riders):
        prof = new_rider(rng)
        riders_meta[r] = prof
        for k, mode in enumerate(modes):
            df = simulate_session(prof, mode, a.duration, a.fs, rng)
            df.insert(0, "session_id", f"R{r:02d}_S{k}")
            df.insert(0, "rider_id", r)
            df["split"] = split[r]
            frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    path = os.path.join(a.out, "synthetic_imu.csv")
    data.round(4).to_csv(path, index=False)
    with open(os.path.join(a.out, "metadata.json"), "w") as fh:
        json.dump(dict(fs=a.fs, duration=a.duration, seed=a.seed, channels=CHANNELS,
                       split=split, riders=riders_meta), fh, indent=2)
    print(f"Wrote {len(data):,} rows -> {path}")
    print(data.groupby("mode")["label"].mean().rename("drowsy_fraction").to_string())

    if a.validate:
        validate(data, a.fs)
    if a.plot:
        make_plot(data, os.path.join(a.out, "sanity_plot.png"))
        print("Saved plot ->", os.path.join(a.out, "sanity_plot.png"))


if __name__ == "__main__":
    main()