"""Matched-gradient-update control for the model-capacity comparison.

The cp_train_only / cp_pool_only / cp_combined baselines were each trained for
5 epochs. Because an epoch is one pass over the training set, cp_combined
(26,277 compounds) received ~5.3x more gradient updates than cp_train_only
(4,999). So the combined model's advantage confounds *more data* with *more
optimisation*.

This script retrains the internal-only model for ceil(5 * 26277/4999) = 26
epochs, matching cp_combined's update count on the original 4,999 compounds.
If internal-only-26ep stays well behind cp_combined, the gain is attributable
to the external data rather than to extra training.

Writes results/metrics/chemprop_epoch_control.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.marimo_openadmet.models import ChempropModel  # noqa: E402

BASE_EPOCHS = 5
ENSEMBLE_SIZE = 3
SEED = 42


def rmse_r2(truth: pd.DataFrame, preds_csv: Path) -> tuple[float, float, int]:
    pr = pd.read_csv(preds_csv)
    pred_col = [c for c in pr.columns if c.lower() not in ("smiles", "id")][0]
    j = (
        pr.merge(truth, left_on="smiles", right_on="SMILES", how="inner")
        .dropna(subset=[pred_col, "LogD"])
    )
    return (
        float(np.sqrt(mean_squared_error(j["LogD"], j[pred_col]))),
        float(r2_score(j["LogD"], j[pred_col])),
        len(j),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--epochs", type=int, default=None,
        help="override the matched-epoch count (e.g. 142 to update-match "
             "combined@27 rather than combined@5)",
    )
    args = ap.parse_args()

    proc = ROOT / "data" / "processed"
    train = pd.read_parquet(proc / "openadmet_train_std_df.parquet")
    test = pd.read_parquet(proc / "openadmet_test_std_df.parquet")
    pool = pd.read_parquet(proc / "rtlogd_pool_std_df.parquet")
    truth = test[["SMILES", "LogD"]]

    n_internal, n_combined = len(train), len(train) + len(pool)
    epochs = args.epochs or int(math.ceil(BASE_EPOCHS * n_combined / n_internal))
    print(
        f"internal={n_internal} combined={n_combined} -> "
        f"matched epochs = ceil({BASE_EPOCHS} * {n_combined}/{n_internal}) = {epochs}",
        flush=True,
    )

    out_dir = ROOT / "models" / f"cp_train_only_{epochs}ep"
    model = ChempropModel(
        output_dir=out_dir,
        ensemble_size=ENSEMBLE_SIZE,
        seed=SEED,
        epochs=epochs,
    )
    print(f"training internal-only for {epochs} epochs -> {out_dir}", flush=True)
    model.fit(train[["SMILES", "LogD"]], smiles_col="SMILES", target_col="LogD")
    model.predict(test[["SMILES"]], smiles_col="SMILES")

    rmse, r2, n = rmse_r2(truth, out_dir / "preds.csv")
    print(f"internal-only @ {epochs} epochs: RMSE={rmse:.4f} R2={r2:.4f} (n={n})", flush=True)

    result = {
        "matched_epochs": epochs,
        "base_epochs": BASE_EPOCHS,
        "n_internal": n_internal,
        "n_combined": n_combined,
        "ensemble_size": ENSEMBLE_SIZE,
        "seed": SEED,
        "internal_only_matched": {"rmse": rmse, "r2": r2, "n_test": n},
    }
    # Re-score the existing 5-epoch baselines from their saved predictions so the
    # control sits next to them in one artifact.
    for name in ("cp_train_only", "cp_pool_only", "cp_combined"):
        p = ROOT / "models" / name / "preds.csv"
        if p.exists():
            r, r2b, nb = rmse_r2(truth, p)
            result[name] = {"rmse": r, "r2": r2b, "n_test": nb, "epochs": BASE_EPOCHS}

    suffix = "" if args.epochs is None else f"_{epochs}ep"
    out = ROOT / "results" / "metrics" / f"chemprop_epoch_control{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
