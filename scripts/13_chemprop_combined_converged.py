"""The missing cell: chemprop on *everything*, trained to the same budget.

Script 11 showed that chemprop's apparent appetite for external data was
undertraining -- internal-only at 27 matched epochs (0.5535) beats
cp_combined at 5 epochs (0.5901). But that control only proves the 5-epoch
comparison was confounded. It does not answer the actual question:

    does external data help a *converged* D-MPNN?

To answer it, train on all 26,277 compounds for the same 27 epochs and
compare against internal-only-27ep. If combined-27ep wins, external data
genuinely helps once the model is fit properly; if it loses, the external
pool is a net cost for this target regardless of training budget.

This is the expensive arm (27 epochs over 26,277 compounds, ~5x the control),
so by default it waits for the other heavy jobs to drain before starting.

Writes results/metrics/chemprop_combined_converged.json, which script 12
folds into assets/model_capacity.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.marimo_openadmet.models import ChempropModel  # noqa: E402

# Jobs worth yielding to: the AL sweeps and the 2x2 builder.
BUSY_PATTERNS = ("08_active_learning", "09_active_learning", "12_model_capacity")
DEFAULT_EPOCHS = 27
ENSEMBLE_SIZE = 3
SEED = 42


def busy_processes() -> list[str]:
    """Other python processes running one of the heavy scripts (not us)."""
    try:
        import psutil
    except ImportError:
        return []
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me or not p.info["cmdline"]:
                continue
            # Only the actual compute workers. Launcher shells (bash, cmd,
            # pixi) carry the same script name in their cmdline and would
            # otherwise be counted -- noisy, and they linger after the worker.
            if "python" not in (p.info["name"] or "").lower():
                continue
            cmd = " ".join(p.info["cmdline"])
            if any(pat in cmd for pat in BUSY_PATTERNS):
                out.append(f"{p.info['pid']}: {Path(cmd.split()[-1]).name}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def wait_for_quiet(poll: int = 120, max_wait: int = 24 * 3600) -> None:
    waited = 0
    while waited < max_wait:
        busy = busy_processes()
        if not busy:
            print(f"[{time.strftime('%H:%M:%S')}] machine is quiet -- starting", flush=True)
            return
        print(
            f"[{time.strftime('%H:%M:%S')}] waiting on {len(busy)} job(s): "
            + "; ".join(busy),
            flush=True,
        )
        time.sleep(poll)
        waited += poll
    print(f"max wait {max_wait}s reached -- starting anyway", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--no-wait", action="store_true", help="start immediately")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()

    if not args.no_wait:
        wait_for_quiet(poll=args.poll)

    proc = ROOT / "data" / "processed"
    train = pd.read_parquet(proc / "openadmet_train_std_df.parquet")[["SMILES", "LogD"]]
    pool = pd.read_parquet(proc / "rtlogd_pool_std_df.parquet")[["SMILES", "LogD"]]
    test = pd.read_parquet(proc / "openadmet_test_std_df.parquet")
    combined = pd.concat([train, pool], ignore_index=True)

    out_dir = ROOT / "models" / f"cp_combined_{args.epochs}ep"
    print(
        f"training combined ({len(combined)} compounds) for {args.epochs} epochs "
        f"-> {out_dir}",
        flush=True,
    )
    t0 = time.time()
    model = ChempropModel(
        output_dir=out_dir,
        ensemble_size=ENSEMBLE_SIZE,
        seed=SEED,
        epochs=args.epochs,
    )
    model.fit(combined, smiles_col="SMILES", target_col="LogD")
    model.predict(test[["SMILES"]], smiles_col="SMILES")
    mins = (time.time() - t0) / 60

    pr = pd.read_csv(out_dir / "preds.csv")
    col = [c for c in pr.columns if c.lower() not in ("smiles", "id")][0]
    j = pr.merge(
        test[["SMILES", "LogD"]], left_on="smiles", right_on="SMILES", how="inner"
    ).dropna(subset=[col, "LogD"])
    rmse = float(np.sqrt(mean_squared_error(j["LogD"], j[col])))
    r2 = float(r2_score(j["LogD"], j[col]))
    print(
        f"combined @ {args.epochs} epochs: RMSE={rmse:.4f} R2={r2:.4f} "
        f"(n={len(j)}, {mins:.0f} min)",
        flush=True,
    )

    out = ROOT / "results" / "metrics" / "chemprop_combined_converged.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "epochs": args.epochs,
                "n_train": int(len(combined)),
                "ensemble_size": ENSEMBLE_SIZE,
                "seed": SEED,
                "rmse": rmse,
                "r2": r2,
                "n_test": int(len(j)),
                "minutes": round(mins, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
