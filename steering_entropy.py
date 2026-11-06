#!/usr/bin/env python3
"""
Phase 2a: Steering entropy (Hs) for two-wheelers.

Pipeline (Progress Report I, Sec 4.2):
  1. Reconstruct a steering-angle proxy from raw IMU (NO ground-truth columns):
       yaw  = HP( integral( HP(gz) ) )          handlebar yaw angle (deg)
       lean = complementary filter(gx, ay, az)  lean/roll angle (deg)
       proxy = yaw + w_lean * lean              coupled steering+balance proxy
  2. Predict each sample with a 2nd-order Taylor extrapolation of the previous
     three samples -> prediction error e_t.
  3. Bin e_t into 9 bins whose edges are multiples of alpha (alert-baseline
     90th percentile of |e|)  ->  p(e_i).
  4. Hs = Shannon entropy (bits) of the bin distribution.  Max = log2(9) = 3.17.
"""
import numpy as np
from scipy.signal import butter, sosfilt, decimate

BIN_MULT = np.array([-5, -2.5, -1, -0.5, 0.5, 1, 2.5, 5])   # -> 9 bins
N_BINS = len(BIN_MULT) + 1


# ------------------------------------------------------------ reconstruction
def highpass(x, fs, fc, order=2):
    """Causal Butterworth high-pass (online-faithful: phone/app can run it live)."""
    sos = butter(order, fc, btype="highpass", fs=fs, output="sos")
    return sosfilt(sos, x)


def reconstruct_yaw(gz, fs, fc=0.03):
    """Integrate yaw rate (deg/s) -> yaw (deg). HP before AND after to kill
    gyro bias and integration drift."""
    g = highpass(gz, fs, fc)
    yaw = np.cumsum(g) / fs
    return highpass(yaw, fs, fc)


def reconstruct_lean(ay, az, gx, fs, alpha=0.98):
    """Complementary filter: gyro for short-term, accelerometer gravity vector
    for long-term. Returns lean angle (deg)."""
    roll_acc = np.degrees(np.arctan2(ay, az))
    lean = np.empty_like(roll_acc)
    lean[0] = roll_acc[0]
    dt = 1.0 / fs
    for i in range(1, len(lean)):
        lean[i] = alpha * (lean[i - 1] + gx[i] * dt) + (1 - alpha) * roll_acc[i]
    return highpass(lean, fs, 0.03)


def steering_proxy(yaw, lean, w_lean=0.0):
    return yaw + w_lean * lean


def downsample(x, fs, target=10):
    """Anti-aliased decimation to `target` Hz (steering dynamics are < 5 Hz)."""
    q = int(round(fs / target))
    return x if q <= 1 else decimate(x, q, ftype="fir", zero_phase=False)


# ------------------------------------------------------------ entropy core
def prediction_error(a):
    """e_t = a_t - a_hat_t, a_hat from position/velocity/acceleration of the
    previous 3 samples (Nakayama 2nd-order Taylor).
    Returned array has len(a) with first 3 entries = 0 so windows stay aligned."""
    a = np.asarray(a, float)
    p1, p2, p3 = a[2:-1], a[1:-2], a[:-3]
    pred = p1 + (p1 - p2) + 0.5 * ((p1 - p2) - (p2 - p3))
    e = a[3:] - pred
    return np.concatenate([np.zeros(3), e])


def calibrate_alpha(error_arrays, q=90):
    """Alert-baseline scale alpha = q-th percentile of |e|. Fit on ALERT TRAIN
    data only (or on the first minutes of a rider in deployment)."""
    allerr = np.concatenate([np.abs(e[3:]) for e in error_arrays])
    return float(np.percentile(allerr, q))


def steering_entropy(e, alpha):
    """Shannon entropy (bits) of the binned prediction-error distribution."""
    e = np.asarray(e)
    e = e[np.isfinite(e)]
    if len(e) == 0 or alpha <= 0:
        return np.nan
    idx = np.digitize(e, BIN_MULT * alpha)
    p = np.bincount(idx, minlength=N_BINS) / len(e)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


if __name__ == "__main__":       # tiny self-test
    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 0.1)
    calm = np.sin(0.3 * t) * 3 + rng.normal(0, 0.1, len(t))
    shaky = calm + (rng.random(len(t)) < 0.05) * rng.normal(0, 4, len(t))
    a = calibrate_alpha([prediction_error(calm)])
    print("Hs calm :", round(steering_entropy(prediction_error(calm), a), 3))
    print("Hs shaky:", round(steering_entropy(prediction_error(shaky), a), 3))
