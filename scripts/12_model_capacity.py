"""The model-capacity 2x2: does external data pay off differently per model?

The whole buy-simulation (scripts 08/09) uses the RefinementStack ensemble.
That bakes in an assumption: that the surrogate can actually exploit whatever
it buys. A tree ensemble over fixed ECFP+descriptors saturates quickly; a
D-MPNN keeps absorbing data. If they differ, then "is external data worth
buying?" is partly a question about *which model you intend to train*, not
just about the data.

So: both models x {internal only, external pool only, both}. Chemprop numbers
are re-scored from the saved predictions of scripts 04/06 (models/cp_*); the
ensemble is refit here because it saves no predictions.

Also folds in the matched-epoch control from script 11 when present, so the
"more data vs more gradient updates" confound travels with the comparison.

Writes assets/model_capacity.parquet (notebook asset) and a copy under
results/metrics/. Skips the expensive ensemble refits when the parquet already
holds them unless --recompute is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.marimo_openadmet.models import RefinementStack  # noqa: E402

PROC = ROOT / "data" / "processed"
ASSET = ROOT / "assets" / "model_capacity.parquet"
MIRROR = ROOT / "results" / "metrics" / "model_capacity.parquet"
CONTROL = ROOT / "results" / "metrics" / "chemprop_epoch_control.json"
CONVERGED = ROOT / "results" / "metrics" / "chemprop_combined_converged.json"

CHEMPROP_RUNS = {
    "internal only": "cp_train_only",
    "external pool only": "cp_pool_only",
    "both": "cp_combined",
}


def score_preds(truth: pd.DataFrame, preds_csv: Path) -> tuple[float, float, int]:
    pr = pd.read_csv(preds_csv)
    col = [c for c in pr.columns if c.lower() not in ("smiles", "id")][0]
    j = pr.merge(truth, left_on="smiles", right_on="SMILES", how="inner").dropna(
        subset=[col, "LogD"]
    )
    return (
        float(np.sqrt(mean_squared_error(j["LogD"], j[col]))),
        float(r2_score(j["LogD"], j[col])),
        len(j),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="refit the ensemble arms")
    args = ap.parse_args()

    test_df = pd.read_parquet(PROC / "openadmet_test_std_df.parquet")
    truth = test_df[["SMILES", "LogD"]]
    rows: list[dict] = []

    # ---- chemprop: re-score saved predictions (cheap) ----
    for config, run in CHEMPROP_RUNS.items():
        p = ROOT / "models" / run / "preds.csv"
        if not p.exists():
            print(f"chemprop {config}: {p} missing, skipping")
            continue
        n_train = (
            len(pd.read_csv(ROOT / "models" / run / "train.csv"))
            if (ROOT / "models" / run / "train.csv").exists()
            else None
        )
        rmse, r2, n = score_preds(truth, p)
        rows.append(
            dict(model="chemprop", config=config, n_train=n_train, epochs=5,
                 rmse=rmse, r2=r2, n_test=n, source=run)
        )
        print(f"chemprop {config:20s} n={n_train} RMSE={rmse:.4f} R2={r2:.4f}", flush=True)

    # ---- longer-trained internal-only arms (script 11, any --epochs) ----
    # Labels carry the epoch count rather than the word "matched": which pairs
    # are comparable depends on whether you match epochs or gradient updates,
    # and the two give opposite answers here. The notebook spells that out.
    controls = sorted((ROOT / "results" / "metrics").glob("chemprop_epoch_control*.json"))
    if not controls:
        print("no epoch control yet -- run scripts/11 first")
    for cf in controls:
        c = json.loads(cf.read_text(encoding="utf-8"))
        m = c.get("internal_only_matched")
        if not m:
            continue
        ep = c.get("matched_epochs")
        rows.append(
            dict(model="chemprop", config=f"internal only ({ep} ep)",
                 n_train=c.get("n_internal"), epochs=ep,
                 rmse=m["rmse"], r2=m["r2"], n_test=m.get("n_test"),
                 source=f"cp_train_only_{ep}ep")
        )
        print(f"chemprop internal-only @ {ep} ep: RMSE={m['rmse']:.4f}", flush=True)

    # ---- converged combined arm (script 13) ----
    if CONVERGED.exists():
        c = json.loads(CONVERGED.read_text(encoding="utf-8"))
        rows.append(
            dict(model="chemprop", config=f"both ({c.get('epochs')} ep)",
                 n_train=c.get("n_train"), epochs=c.get("epochs"),
                 rmse=c["rmse"], r2=c["r2"], n_test=c.get("n_test"),
                 source="cp_combined_matched")
        )
        print(f"chemprop converged-combined: RMSE={c['rmse']:.4f}", flush=True)
    else:
        print(f"no converged combined arm yet ({CONVERGED.name}) -- run scripts/13")

    # ---- ensemble: refit (expensive); reuse cached rows unless --recompute ----
    cached = pd.read_parquet(ASSET) if ASSET.exists() else pd.DataFrame()
    have = (
        set(cached[cached["model"] == "ensemble"]["config"])
        if not cached.empty and "model" in cached
        else set()
    )
    if not args.recompute and have >= set(CHEMPROP_RUNS):
        print("ensemble arms cached; use --recompute to refit")
        rows += cached[cached["model"] == "ensemble"].to_dict("records")
    else:
        Xtr = np.load(PROC / "openadmet_train_std_X.npy")
        ytr = np.load(PROC / "openadmet_train_std_y.npy")
        Xpo = np.load(PROC / "rtlogd_pool_std_X.npy")
        ypo = np.load(PROC / "rtlogd_pool_std_y.npy")
        Xte = np.load(PROC / "openadmet_test_std_X.npy")
        yte = np.load(PROC / "openadmet_test_std_y.npy")
        arms = {
            "internal only": (Xtr, ytr),
            "external pool only": (Xpo, ypo),
            "both": (np.vstack([Xtr, Xpo]), np.concatenate([ytr, ypo])),
        }
        for config, (X, y) in arms.items():
            print(f"ensemble {config}: fitting on {len(y)} ...", flush=True)
            pred = RefinementStack(n_folds=3, seed=42).fit(X, y).predict(Xte)
            rmse = float(np.sqrt(mean_squared_error(yte, pred)))
            r2 = float(r2_score(yte, pred))
            rows.append(
                dict(model="ensemble", config=config, n_train=int(len(y)), epochs=None,
                     rmse=rmse, r2=r2, n_test=int(len(yte)), source="refit")
            )
            print(f"ensemble {config:20s} n={len(y)} RMSE={rmse:.4f} R2={r2:.4f}", flush=True)

    out = pd.DataFrame(rows)
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    MIRROR.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ASSET, index=False)
    out.to_parquet(MIRROR, index=False)
    print(f"\nwrote {ASSET} ({out.shape})")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
