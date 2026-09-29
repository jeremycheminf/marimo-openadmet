"""Persistence + simple static visuals for the buy-simulation sweeps.

Per-run artifacts (repo conventions, under results/):
  - metrics  -> results/metrics/al_{model}_{strategy}_frac{pct}_seed{s}_metrics.csv
  - compound -> results/metrics/al_{model}_{strategy}_frac{pct}_seed{s}_selected.csv
  - map cache-> results/metrics/al_map_coords.parquet (shared, computed once)
  - history  -> results/metrics/al_{model}_runs.json (provenance)
  - figures  -> results/figures/al_{model}_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .active_learning import METRICS_COLUMNS, PURCHASE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_METRICS = PROJECT_ROOT / "results" / "metrics"
RESULT_FIGURES = PROJECT_ROOT / "results" / "figures"


def run_id(model, strategy, fraction, seed, budget=0):
    pct = int(round(fraction * 100))
    return f"al_{model}_{strategy}_frac{pct:03d}_budget{budget:05d}_seed{seed}"


def write_run(records, purchases, model, strategy, fraction, seed, budget=0, metrics_dir=None):
    """Persist one run's per-round metrics + compound purchases."""
    metrics_dir = RESULT_METRICS if metrics_dir is None else Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rid = run_id(model, strategy, fraction, seed, budget=budget)
    # Metrics: one row per round, written via the canonical columns.
    mpath = metrics_dir / f"{rid}_metrics.csv"
    pd.DataFrame(records).reindex(columns=METRICS_COLUMNS).to_csv(mpath, index=False)

    # Purchases: one row per bought compound.
    ppath = metrics_dir / f"{rid}_selected.csv"
    if purchases:
        pd.DataFrame(purchases).reindex(columns=PURCHASE_COLUMNS).to_csv(
            ppath, index=False
        )
    else:
        pd.DataFrame(columns=PURCHASE_COLUMNS).to_csv(ppath, index=False)
    return mpath, ppath


def n_rounds_for(purchase_step):
    """Number of buy rounds the engine uses for a given purchase_step.

    Mirrors ``active_learning.run_buy_simulation``: n_rounds = round(1/purchase_step).
    """
    return max(1, int(round(1.0 / purchase_step)))


def run_complete(model, strategy, fraction, seed, budget, purchase_step, metrics_dir=None):
    """True if this run's metrics file already holds a *complete* run.

    A complete run has one baseline row plus ``n_rounds_for(purchase_step)`` buy
    rounds. Used by ``--skip-existing`` so an interrupted sweep can resume
    without recomputing finished runs.
    """
    metrics_dir = RESULT_METRICS if metrics_dir is None else Path(metrics_dir)
    mpath = metrics_dir / f"{run_id(model, strategy, fraction, seed, budget=budget)}_metrics.csv"
    if not mpath.exists() or mpath.stat().st_size == 0:
        return False
    try:
        with mpath.open(encoding="utf-8") as fh:
            n_rows = sum(1 for _ in fh) - 1  # drop header
    except OSError:
        return False
    return n_rows >= n_rounds_for(purchase_step) + 1


def collect_metrics(model, metrics_dir=None):
    """Concat every run's metric file for one model into one tidy DataFrame."""
    metrics_dir = RESULT_METRICS if metrics_dir is None else Path(metrics_dir)
    files = sorted(metrics_dir.glob(f"al_{model}_*_metrics.csv"))
    parts = [pd.read_csv(f) for f in files]
    if not parts:
        return pd.DataFrame(columns=METRICS_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df.reindex(columns=METRICS_COLUMNS)


def update_runs(entry, model, runs_file=None, metrics_dir=None):
    """Append a provenance entry to the model's runs.json."""
    metrics_dir = RESULT_METRICS if metrics_dir is None else Path(metrics_dir)
    runs_file = runs_file or metrics_dir / f"al_{model}_runs.json"
    entries = []
    if runs_file.exists():
        try:
            entries = json.loads(runs_file.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entries.append(entry)
    runs_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")

def ensure_map_cache(
    processed_dir,
    cache_path=None,
    n_pca=50,
    n_neighbors=15,
    min_dist=0.1,
    overwrite=False,
):
    """Compute (once) and cache UMAP coordinates for train + test + pool molecules."""
    cache_path = RESULT_METRICS / "al_map_coords.parquet" if cache_path is None else Path(cache_path)
    if cache_path.exists() and not overwrite:
        return cache_path

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap

    processed_dir = Path(processed_dir)
    specs = [
        ("train", "openadmet_train_std_df", "openadmet_train_std_X"),
        ("test", "openadmet_test_std_df", "openadmet_test_std_X"),
        ("pool", "rtlogd_pool_std_df", "rtlogd_pool_std_X"),
    ]
    parts = []
    X_parts = []
    for src, df_name, x_name in specs:
        df = pd.read_parquet(processed_dir / f"{df_name}.parquet")
        X = np.load(processed_dir / f"{x_name}.npy")
        parts.append((src, df))
        X_parts.append(X)

    X_all = np.vstack(X_parts)
    Xs = StandardScaler().fit_transform(X_all)
    Xp = PCA(n_components=min(n_pca, X_all.shape[1]), random_state=42).fit_transform(Xs)
    emb = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=42, verbose=False
    ).fit_transform(Xp)

    rows = []
    start = 0
    for src, df in parts:
        n = len(df)
        rows.append(
            pd.DataFrame(
                {
                    "source": src,
                    "source_index": np.arange(n),
                    "smiles": df["SMILES"].values,
                    "logd": df["LogD"].values,
                    "umap_x": emb[start : start + n, 0],
                    "umap_y": emb[start : start + n, 1],
                }
            )
        )
        start += n
    out = pd.concat(rows, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_path, index=False)
    return cache_path


def save_learning_curve_figure(metrics_df, model, out_path=None):
    """Per-fraction learning curves (RMSE vs compounds bought) for each budget."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = RESULT_FIGURES / f"al_{model}_learning_curves.png" if out_path is None else Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fracs = sorted(metrics_df["fraction"].unique())
    budgets = sorted(metrics_df["budget"].unique()) if "budget" in metrics_df.columns else [None]
    cols = min(5, len(fracs) or 1)
    rows = (int(np.ceil(len(fracs) / cols)) or 1) * max(1, len(budgets))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows), squeeze=False)
    ai = 0
    for budget in budgets:
        bsub = metrics_df if budget is None else metrics_df[metrics_df["budget"] == budget]
        for fr in fracs:
            ax = axes.ravel()[ai]
            ai += 1
            sub = bsub[bsub["fraction"] == fr]
            for strat, sdf in sub.groupby("strategy"):
                agg = (
                    sdf.groupby("n_bought", as_index=False)
                    .agg(mean=("rmse", "mean"), std=("rmse", "std"))
                    .sort_values("n_bought")
                )
                ax.plot(agg["n_bought"], agg["mean"], label=strat)
                ax.fill_between(
                    agg["n_bought"],
                    agg["mean"] - agg["std"],
                    agg["mean"] + agg["std"],
                    alpha=0.2,
                )
            blab = f" | budget={budget}" if budget is not None else ""
            ax.set_title(f"frac={fr:.0%} train{blab}", fontsize=9)
            ax.set_xlabel("compounds bought")
            ax.set_ylabel("test RMSE")
            ax.legend(fontsize=7)
    for ax in axes.ravel()[ai:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_final_heatmap_figure(final_df, model, out_path=None):
    """End-state RMSE heatmap: initial-train fraction x budget (per strategy)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = RESULT_FIGURES / f"al_{model}_final_heatmap.png" if out_path is None else Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    strategies = sorted(final_df["strategy"].unique())
    cols = min(3, len(strategies) or 1)
    rows = int(np.ceil(len(strategies) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False)
    for ax, strat in zip(axes.ravel(), strategies):
        sub = final_df[final_df["strategy"] == strat]
        agg = sub.groupby(["fraction", "budget"], as_index=False)["rmse"].mean()
        pivot = agg.pivot(index="fraction", columns="budget", values="rmse")
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{int(c)}" for c in pivot.columns], fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:.0f}%" for r in pivot.index], fontsize=7)
        ax.set_title(strat, fontsize=9)
        ax.set_xlabel("budget (compounds bought)")
        ax.set_ylabel("initial train %")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(strategies):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
