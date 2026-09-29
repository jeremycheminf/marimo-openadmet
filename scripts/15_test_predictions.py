"""Per-compound test-set predictions for the notebook's predicted-vs-measured plot.

The sweeps (08/09) logged RMSE per round, not the predictions themselves. This
script recovers them for the *final* round of every run, plus the no-buy
baseline of every stage, so the notebook can show a parity plot driven by two
sliders: internal data (stage) and external compounds bought (budget).

- chemprop: read straight from models/al_chemprop_*/preds.csv, which holds the
  final round's ensemble prediction (verified to reproduce the logged RMSE
  exactly). The no-buy baseline is refit (--chemprop-baseline, GPU env).
- ensemble: rebuilt, not recomputed from scratch. Each run's training set is
  reconstructed exactly as run_buy_simulation built it -- the stage's
  alphabetical slice, minus the deterministic probe carve-out, plus the pool
  compounds in its purchase log -- and the RefinementStack is refit with the
  sweep's seed. The script checks the refit against the logged RMSE.

`random` uses seed 0 only (one draw of five); the others ran once.

    pixi run python scripts/15_test_predictions.py --ensemble            # ~170 fits
    pixi run -e gpu python scripts/15_test_predictions.py --chemprop     # reads preds + 3 GPU baselines
    pixi run python scripts/15_test_predictions.py --one                 # timing / reproduction check

Writes results/metrics/test_predictions_<model>.parquet; script 10 packs them
into assets/test_predictions.parquet.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.marimo_openadmet import reporting  # noqa: E402
from src.marimo_openadmet.active_learning import EnsembleBackend  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "metrics"
STRATEGIES = ["random", "diversity_tanimoto", "uncertainty", "similarity"]
BUDGETS = [100, 500, 1000, 5000]
PROBE_FRAC = 0.1
CONFIG_SEED = 42


def load():
    d = {}
    for name in ("openadmet_train_std", "openadmet_test_std", "rtlogd_pool_std"):
        d[name] = (
            pd.read_parquet(PROC / f"{name}_df.parquet"),
            np.load(PROC / f"{name}_X.npy"),
            np.load(PROC / f"{name}_y.npy"),
        )
    return d


def labeled_slice(df_train, X_train, y_train, fraction):
    """Exactly the labelled set run_buy_simulation trains on at round 0."""
    sorted_idx = df_train.sort_values(by="Molecule Name").index.to_numpy()
    pos = sorted_idx[: int(round(fraction * len(df_train)))]
    X, y = X_train[pos], y_train[pos]
    n_all = len(pos)
    probe_rng = np.random.default_rng(1000 + int(round(fraction * 100)))
    n_probe = max(1, int(round(n_all * PROBE_FRAC)))
    probe_idx = np.sort(probe_rng.choice(n_all, size=n_probe, replace=False))
    label_idx = np.setdiff1d(np.arange(n_all), probe_idx)
    return X[label_idx], y[label_idx]


def fit_predict(Xl, yl, X_test):
    b = EnsembleBackend(n_folds=3, seed=CONFIG_SEED)
    b.fit(None, Xl, yl)
    return np.asarray(b.predict(None, X_test), dtype=np.float32)


def purchased(model, strategy, fraction, seed, budget):
    rid = reporting.run_id(model, strategy, fraction, seed, budget=budget)
    sel = pd.read_csv(OUT / f"{rid}_selected.csv", usecols=["round", "pool_index"])
    return sel["pool_index"].to_numpy(dtype=int)


def logged_rmse(model, strategy, fraction, seed, budget, rnd=None):
    rid = reporting.run_id(model, strategy, fraction, seed, budget=budget)
    m = pd.read_csv(OUT / f"{rid}_metrics.csv")
    row = m.iloc[0] if rnd == 0 else m.loc[m["round"].idxmax()]
    return float(row["rmse"])


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_ensemble(one=False):
    d = load()
    (dtr, Xtr, ytr), (_, Xte, yte), (_, Xpo, ypo) = (
        d["openadmet_train_std"], d["openadmet_test_std"], d["rtlogd_pool_std"])
    out_path = OUT / "test_predictions_ensemble.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    have = set(map(tuple, done[["strategy", "fraction", "budget"]].drop_duplicates().to_numpy())) if len(done) else set()
    frames = [done] if len(done) else []
    fractions = [0.1] if one else [f / 100 for f in range(10, 101, 10)]

    for fr in fractions:
        Xl, yl = labeled_slice(dtr, Xtr, ytr, fr)
        jobs = [("none", 0)] + [(s, b) for s in STRATEGIES for b in BUDGETS]
        if one:
            jobs = [("none", 0), ("random", 1000)]
        for strat, budget in jobs:
            if (strat, fr, budget) in have:
                continue
            t0 = time.time()
            if strat == "none":
                X, y = Xl, yl
                ref = logged_rmse("ensemble", "random", fr, 0, 100, rnd=0)
            else:
                seed = 0 if strat == "random" else CONFIG_SEED
                idx = purchased("ensemble", strat, fr, seed, budget)
                X, y = np.vstack([Xl, Xpo[idx]]), np.concatenate([yl, ypo[idx]])
                ref = logged_rmse("ensemble", strat, fr, seed, budget)
            p = fit_predict(X, y, Xte)
            got = rmse(p, yte)
            print(f"frac {fr:.1f} {strat:18s} budget {budget:5d}: rmse {got:.4f} "
                  f"(logged {ref:.4f}, diff {got - ref:+.4f}) n_train {len(y):5d} "
                  f"{time.time() - t0:5.1f}s", flush=True)
            frames.append(pd.DataFrame({
                "model": "ensemble", "strategy": strat, "fraction": fr, "budget": budget,
                "test_index": np.arange(len(p), dtype=np.int16), "pred": p,
                "n_train": len(y),
            }))
            if not one:  # checkpoint after every fit so an interruption loses one fit
                pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
    if one:
        print("(--one: nothing written)")


def run_chemprop(baseline=True):
    _, _, yte = load()["openadmet_test_std"]
    frames = []
    for d in sorted((ROOT / "models").glob("al_chemprop_*_budget*_seed*")):
        k = re.search(r"al_chemprop_(.+)_frac(\d+)_budget(\d+)_seed(\d+)", d.name)
        st, fr, bu, sd = k.group(1), int(k.group(2)) / 100, int(k.group(3)), int(k.group(4))
        if st == "random" and sd != 0:
            continue
        p = pd.read_csv(d / "preds.csv").iloc[:, 1].to_numpy(dtype=np.float32)
        ref = logged_rmse("chemprop", st, fr, sd, bu)
        assert abs(rmse(p, yte) - ref) < 1e-6, (d.name, rmse(p, yte), ref)
        n_bought = len(purchased("chemprop", st, fr, sd, bu))
        frames.append(pd.DataFrame({
            "model": "chemprop", "strategy": st, "fraction": fr, "budget": bu,
            "test_index": np.arange(len(p), dtype=np.int16), "pred": p, "n_train": -1,
            "n_bought": n_bought,
        }))
    print(f"chemprop: {len(frames)} runs read, all reproduce the logged RMSE")

    if baseline:
        from src.marimo_openadmet.active_learning import ChempropBackend
        d = load()
        dtr, _, ytr = d["openadmet_train_std"]
        dte = d["openadmet_test_std"][0]
        sorted_idx = dtr.sort_values(by="Molecule Name").index.to_numpy()
        for fr in (0.1, 0.5, 1.0):
            pos = sorted_idx[: int(round(fr * len(dtr)))]
            sl = dtr.iloc[pos].reset_index(drop=True)
            n_all = len(sl)
            probe_rng = np.random.default_rng(1000 + int(round(fr * 100)))
            probe_idx = np.sort(probe_rng.choice(n_all, size=max(1, int(round(n_all * PROBE_FRAC))), replace=False))
            lab = sl.iloc[np.setdiff1d(np.arange(n_all), probe_idx)].reset_index(drop=True)
            t0 = time.time()
            b = ChempropBackend(output_dir=ROOT / "models" / f"cp_testpred_baseline_frac{int(fr * 100):03d}",
                                ensemble_size=3, epochs=20, seed=CONFIG_SEED)
            b.fit(lab, None, lab["LogD"].to_numpy())
            p = np.asarray(b.predict(dte, None), dtype=np.float32)
            ref = logged_rmse("chemprop", "random", fr, 0, 100, rnd=0)
            print(f"chemprop baseline frac {fr:.1f}: rmse {rmse(p, yte):.4f} (logged {ref:.4f}) "
                  f"{time.time() - t0:.0f}s", flush=True)
            frames.append(pd.DataFrame({
                "model": "chemprop", "strategy": "none", "fraction": fr, "budget": 0,
                "test_index": np.arange(len(p), dtype=np.int16), "pred": p, "n_train": len(lab),
                "n_bought": 0,
            }))
    pd.concat(frames, ignore_index=True).to_parquet(OUT / "test_predictions_chemprop.parquet", index=False)
    print("wrote test_predictions_chemprop.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--chemprop", action="store_true")
    ap.add_argument("--no-chemprop-baseline", action="store_true")
    ap.add_argument("--one", action="store_true", help="time two ensemble fits and check reproduction")
    a = ap.parse_args()
    if a.one:
        run_ensemble(one=True)
    if a.ensemble:
        run_ensemble()
    if a.chemprop:
        run_chemprop(baseline=not a.no_chemprop_baseline)
    if not (a.one or a.ensemble or a.chemprop):
        ap.print_help()
