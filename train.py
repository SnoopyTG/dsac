#!/usr/bin/env python3
"""
Phase 3c: Train + evaluate the 1D-CNN.

Examples
  python train.py --data data --aux all --name cnn_all                 # single run, uses the 'split' column
  python train.py --data data --aux hs  --name cnn_hs                  # ablation: entropy only as aux
  python train.py --data data --aux none --name cnn_wave               # ablation: waveform only
  python train.py --data data --aux all --cv 5 --name cnn_all_cv       # leave-riders-out 5-fold CV (recommended)

--aux options: none | hs | all_no_time | all
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from model import FatigueCNN
from eval_utils import best_threshold, compute_metrics, cv_splits, summarize

AUX_SETS = {"none": [], "hs": ["Hs"],
            "all_no_time": ["Hs", "jerk_mean", "rev_rate", "lean_var"],
            "all": ["Hs", "jerk_mean", "rev_rate", "lean_var", "elapsed_min"]}


def load(data_dir):
    d = np.load(os.path.join(data_dir, "windows.npz"), allow_pickle=False)
    feats = pd.read_csv(os.path.join(data_dir, "features.csv"))
    assert len(feats) == len(d["X"]) and (feats.rider_id.values == d["rider"]).all(), \
        "features.csv and windows.npz are out of sync - rerun feature_extraction.py"
    cfg = json.load(open(os.path.join(data_dir, "feature_config.json")))
    names = list(d["feature_names"])
    return d["X"], d["F"], d["y"], d["rider"], d["split"], names, feats, cfg


def predict(model, X, A, dev, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev)
            ab = torch.from_numpy(A[i:i + bs]).to(dev) if A.shape[1] else None
            out.append(torch.sigmoid(model(xb, ab)).cpu().numpy())
    return np.concatenate(out)


def fit_and_eval(X, A, y, tr, va, te, args, hop_s, kss, dev, run_dir=None):
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- normalisation statistics from TRAIN windows only (no leakage)
    ch_mu = X[tr].mean(axis=(0, 2), keepdims=True); ch_sd = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = ((X - ch_mu) / ch_sd).astype(np.float32)
    if A.shape[1]:
        a_mu = A[tr].mean(0, keepdims=True); a_sd = A[tr].std(0, keepdims=True) + 1e-6
        An = ((A - a_mu) / a_sd).astype(np.float32)
    else:
        a_mu = a_sd = None; An = A.astype(np.float32)

    model = FatigueCNN(in_ch=X.shape[1], n_aux=A.shape[1], dropout=args.dropout).to(dev)
    n_pos, n_neg = y[tr].sum(), (1 - y[tr]).sum()
    pos_w = torch.tensor(n_neg / max(1, n_pos), dtype=torch.float32, device=dev)   # class weighting
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    Xt, At, yt = (torch.from_numpy(Xn[tr]).to(dev), torch.from_numpy(An[tr]).to(dev),
                  torch.from_numpy(y[tr]).float().to(dev))
    Xv, Av, yv = (torch.from_numpy(Xn[va]).to(dev), torch.from_numpy(An[va]).to(dev),
                  torch.from_numpy(y[va]).float().to(dev))
    best, best_state, bad, hist = 1e9, None, 0, []

    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xt), device=dev); tl = 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = Xt[idx]
            if args.augment:                                       # mild amplitude jitter + noise
                xb = xb * (1 + 0.1 * (torch.rand(len(idx), 1, 1, device=dev) - 0.5) * 2) \
                     + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(model(xb, At[idx] if A.shape[1] else None), yt[idx])
            loss.backward(); opt.step(); tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv, Av if A.shape[1] else None), yv).item()
        sched.step(vl); hist.append(dict(epoch=ep, train_loss=tl / len(Xt), val_loss=vl))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)                                # restore best-val weights

    p_va = predict(model, Xn[va], An[va], dev)
    thr = best_threshold(y[va], p_va)                                # threshold chosen on VAL
    p_te = predict(model, Xn[te], An[te], dev)
    m = compute_metrics(y[te], p_te, thr, hop_s, kss[te])
    m.update(epochs_run=len(hist), best_val_loss=float(best), params=model.n_params())

    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), n_aux=A.shape[1], in_ch=X.shape[1],
                        threshold=thr), os.path.join(run_dir, "model.pt"))
        json.dump(dict(ch_mu=ch_mu.ravel().tolist(), ch_sd=ch_sd.ravel().tolist(),
                       a_mu=None if a_mu is None else a_mu.ravel().tolist(),
                       a_sd=None if a_sd is None else a_sd.ravel().tolist(), threshold=thr),
                  open(os.path.join(run_dir, "norm.json"), "w"), indent=2)
        pd.DataFrame(hist).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        pd.DataFrame(dict(idx=te, y=y[te], p=p_te)).to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default="cnn"); ap.add_argument("--aux", default="all", choices=AUX_SETS)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--augment", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", type=int, default=0, help="K>0: leave-riders-out K-fold CV")
    a = ap.parse_args()

    X, F, y, rider, split, names, feats, cfg = load(a.data)
    A = F[:, [names.index(n) for n in AUX_SETS[a.aux]]] if AUX_SETS[a.aux] else np.zeros((len(X), 0), np.float32)
    kss = feats.kss_mean.values
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(a.out, a.name)
    print(f"device={dev} | windows={len(X)} | aux={a.aux} {AUX_SETS[a.aux]} | drowsy frac={y.mean():.2f}")

    t0 = time.time()
    if a.cv:
        ms = []
        for k, (tr, va, te) in enumerate(cv_splits(rider, a.cv, a.seed)):
            m = fit_and_eval(X, A, y, tr, va, te, a, cfg["hop_s"], kss, dev)
            print(f"fold {k}: test riders={len(np.unique(rider[te]))} AUC={m['auc']:.3f} F1={m['f1']:.3f} "
                  f"FA/h={m['false_alarms_per_hour']:.1f} epochs={m['epochs_run']}")
            ms.append(m)
        summ = summarize(ms)
        print("\nCV mean +/- std:")
        for k in ["auc", "f1", "precision", "recall", "false_alarms_per_hour", "pearson_p_vs_kss"]:
            print(f"  {k:24s} {summ[k][0]:.3f} +/- {summ[k][1]:.3f}")
        os.makedirs(run_dir, exist_ok=True)
        json.dump(dict(mode="cv", aux=a.aux, folds=ms, summary=summ), open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    else:
        tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
        assert not (set(rider[tr]) & set(rider[te])) and not (set(rider[tr]) & set(rider[va])), "rider leakage!"
        m = fit_and_eval(X, A, y, tr, va, te, a, cfg["hop_s"], kss, dev, run_dir)
        print(f"\nTEST ({len(np.unique(rider[te]))} riders, {len(te)} windows), params={m['params']:,}")
        for k in ["auc", "f1", "precision", "recall", "false_alarms_per_hour", "miss_rate", "pearson_p_vs_kss"]:
            print(f"  {k:24s} {m[k]:.3f}")
        json.dump(dict(mode="single", aux=a.aux, metrics=m), open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    print(f"done in {time.time() - t0:.0f}s -> {run_dir}/")


if __name__ == "__main__":
    main()
