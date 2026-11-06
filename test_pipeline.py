#!/usr/bin/env python3
"""
Automated tests for Phases 1-3.  Run with either:
    pytest -q test_pipeline.py        (pip install pytest)
    python test_pipeline.py           (built-in mini runner, no pytest needed)
Uses a tiny temporary dataset, so it takes well under a minute.
"""
import json, os, tempfile
import numpy as np, pandas as pd

import data_generator as gen
import steering_entropy as se
import feature_extraction as fx
import apla_dtw_baseline as bl
import eval_utils as eu


# ------------------------------------------------------------------ Phase 1
def _small_session(mode, seed=0, dur=120, fs=50):
    rng = np.random.default_rng(seed)
    return gen.simulate_session(gen.new_rider(rng), mode, dur, fs, rng)


def test_generator_schema_and_gravity():
    df = _small_session("alert")
    for c in gen.CHANNELS + ["yaw_deg", "lean_deg", "kss", "label"]:
        assert c in df.columns
    assert not df[gen.CHANNELS].isna().any().any()
    assert abs(df["az"].mean() - 9.81) < 0.8, "az should be ~gravity"


def test_generator_labels_follow_mode():
    assert _small_session("alert")["label"].mean() < 0.05
    assert _small_session("fatigued")["label"].mean() > 0.7


def test_generator_is_reproducible():
    a, b = _small_session("alert", 3), _small_session("alert", 3)
    assert np.allclose(a["gz"].values, b["gz"].values)


def test_fatigued_is_burstier_than_alert():
    kurt = lambda x: float(((x - x.mean()) ** 4).mean() / x.var() ** 2)
    ea = se.prediction_error(_small_session("alert")["yaw_deg"].values[::5])
    ef = se.prediction_error(_small_session("fatigued")["yaw_deg"].values[::5])
    assert kurt(ef) > kurt(ea), "fatigued errors should be heavier-tailed"


# ------------------------------------------------------------------ Phase 2
def test_taylor_predictor_properties():
    """Nakayama predictor (0.5 weight on the acceleration term): exact for linear
    signals; for a quadratic the error is the constant 0.5 * (second difference)."""
    t = np.arange(50) * 0.1
    e_lin = se.prediction_error(3 + 2 * t)
    assert np.abs(e_lin).max() < 1e-9 and len(e_lin) == 50
    e_quad = se.prediction_error(0.7 * t ** 2)[3:]
    assert np.allclose(e_quad, 0.5 * (2 * 0.7 * 0.1 ** 2))


def test_entropy_bounds_and_ordering():
    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 0.1)
    calm = np.sin(0.3 * t) * 3 + rng.normal(0, .1, len(t))
    shaky = calm + (rng.random(len(t)) < .05) * rng.normal(0, 4, len(t))
    a = se.calibrate_alpha([se.prediction_error(calm)])
    hc, hs = (se.steering_entropy(se.prediction_error(x), a) for x in (calm, shaky))
    assert 0 <= hc < hs <= np.log2(9) + 1e-9


def test_reversal_rate_on_known_sine():
    fs, f0 = 10, 0.2                      # 0.2 Hz sine -> 2 reversals / cycle -> 24 per minute
    t = np.arange(0, 60, 1 / fs)
    assert abs(fx.reversal_rate(3 * np.sin(2 * np.pi * f0 * t), fs) - 24) < 3
    assert fx.reversal_rate(0.1 * np.sin(2 * np.pi * f0 * t), fs) == 0, "below gap -> no reversals"


def test_reconstruction_matches_ground_truth_band():
    df, fs = _small_session("fatigued", dur=90), 50
    yaw = se.reconstruct_yaw(df.gz.values, fs)[15 * fs:]
    gt = se.highpass(df.yaw_deg.values, fs, 0.03)[15 * fs:]
    hf = lambda x: se.highpass(x, fs, 0.3)
    assert np.corrcoef(hf(yaw), hf(gt))[0, 1] > 0.9


def test_feature_extraction_end_to_end_no_leakage():
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(1); frames = []
        split = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}
        for r in range(5):
            p = gen.new_rider(rng)
            for k, m in enumerate(["alert", "fatigued"]):
                s = gen.simulate_session(p, m, 90, 50, rng)
                s.insert(0, "session_id", f"R{r}_S{k}"); s.insert(0, "rider_id", r); s["split"] = split[r]
                frames.append(s)
        pd.concat(frames).round(4).to_csv(f"{d}/synthetic_imu.csv", index=False)
        json.dump(dict(fs=50), open(f"{d}/metadata.json", "w"))
        import sys
        sys.argv = ["fx", "--data", f"{d}/synthetic_imu.csv", "--out", d, "--win", "30", "--hop", "15", "--warmup", "10"]
        fx.main()
        z = np.load(f"{d}/windows.npz"); f = pd.read_csv(f"{d}/features.csv")
        assert z["X"].shape[1:] == (4, 300) and len(f) == len(z["X"])
        assert np.isfinite(z["X"]).all() and np.isfinite(z["F"]).all()
        assert not (set(f[f.split == "train"].rider_id) & set(f[f.split == "test"].rider_id))


# ------------------------------------------------------------------ Phase 3
def test_apen_orders_regularity():
    t = np.arange(100) / 10
    assert bl.apen(np.sin(t)) < bl.apen(np.random.default_rng(0).standard_normal(100))


def test_apla_respects_tolerance_and_dtw_identity():
    x = np.cumsum(np.random.default_rng(0).normal(0, .05, 40))
    assert np.abs(bl.apla(x, 0.1) - x).max() <= 0.1 + 1e-9
    assert bl.dtw(x, x) == 0 and bl.dtw(x, x + 1) > 0


def test_cv_splits_have_no_rider_overlap():
    riders = np.repeat(np.arange(20), 10)
    for tr, va, te in eu.cv_splits(riders, k=5, seed=0):
        R = [set(riders[i]) for i in (tr, va, te)]
        assert not (R[0] & R[1]) and not (R[0] & R[2]) and not (R[1] & R[2])


def test_metrics_perfect_and_far():
    y = np.array([0, 0, 0, 0, 1, 1]); p = np.array([.1, .1, .6, .1, .9, .9])
    m = eu.compute_metrics(y, p, 0.5, hop_s=15)
    assert m["recall"] == 1.0 and m["auc"] == 1.0
    assert abs(m["false_alarms_per_hour"] - 1 / (4 * 15 / 3600)) < 1e-6


def test_cnn_shapes_size_and_can_overfit():
    import torch
    from model import FatigueCNN
    m = FatigueCNN(n_aux=5, dropout=0.0)
    x, a = torch.randn(16, 4, 300), torch.randn(16, 5)
    assert m(x, a).shape == (16,) and m.n_params() < 50_000
    y = (torch.rand(16) > .5).float()
    opt = torch.optim.Adam(m.parameters(), 1e-2); lf = torch.nn.BCEWithLogitsLoss()
    first = None
    for _ in range(60):
        opt.zero_grad(); l = lf(m(x, a), y); l.backward(); opt.step()
        first = first if first is not None else l.item()
    assert l.item() < 0.5 * first, "model must be able to overfit 16 samples (sanity check)"


# ------------------------------------------------------------------ mini runner
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:
            failed += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
