import argparse
import sys
from datetime import datetime
from pathlib import Path

# Adds the project root dir and 'src' to the module search path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.marimo_openadmet.active_learning import ChempropBackend, run_buy_simulation  # noqa: E402
from src.marimo_openadmet import reporting  # noqa: E402
from src.marimo_openadmet.reporting import update_runs, write_run  # noqa: E402


def parse_fraction_list(value):
    return [int(part) / 100.0 for part in value.split(",") if part.strip()]


def parse_strategy_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_budget_list(value):
    return [int(part) for part in value.split(",") if part.strip()]


def build_parser():
    ap = argparse.ArgumentParser(
        description="Buy-RTLogD active-learning sweep with Chemprop."
    )
    # Defaults reproduce the shipped sweep: 3 stages x 4 budgets, 3 random
    # seeds, 20 epochs on GPU (`pixi run -e gpu`). The 5-epoch runs it
    # replaced are archived under results/metrics/archive_chemprop_5ep/.
    ap.add_argument("--fractions", default="10,50,100")
    ap.add_argument(
        "--strategies",
        default="random,diversity_tanimoto,uncertainty,similarity",
        help="Comma list; `random` runs --seeds seeds, the others run once at --config-seed.",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--purchase-step", type=float, default=0.1)
    ap.add_argument("--budgets", default="100,500,1000,5000")
    ap.add_argument("--config-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ensemble-size", type=int, default=3)
    ap.add_argument("--error-selector", choices=["uncertainty", "error_similarity"], default="uncertainty")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics CSV already holds a complete run (resume).",
    )
    return ap


def main():
    args = build_parser().parse_args()

    if args.smoke:
        args.fractions = "10,100"
        args.seeds = 2

    processed_dir = root_dir / "data" / "processed"
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(processed_dir / "openadmet_train_std_df.parquet")
    X_train = np.load(processed_dir / "openadmet_train_std_X.npy")
    y_train = np.load(processed_dir / "openadmet_train_std_y.npy")
    df_test = pd.read_parquet(processed_dir / "openadmet_test_std_df.parquet")
    X_test = np.load(processed_dir / "openadmet_test_std_X.npy")
    y_test = np.load(processed_dir / "openadmet_test_std_y.npy")
    df_pool = pd.read_parquet(processed_dir / "rtlogd_pool_std_df.parquet")
    X_pool = np.load(processed_dir / "rtlogd_pool_std_X.npy")
    y_pool = np.load(processed_dir / "rtlogd_pool_std_y.npy")

    name_col = next(
        (
            c
            for c in ["Molecule Name", "molecule_name", "Name", "ID", "id"]
            if c in df_train.columns
        ),
        None,
    )
    sorted_idx = (
        df_train.sort_values(by=name_col).index.to_numpy()
        if name_col
        else np.arange(len(df_train))
    )

    fractions = parse_fraction_list(args.fractions)
    strategies = parse_strategy_list(args.strategies)
    budgets = parse_budget_list(args.budgets)
    if args.error_selector == "error_similarity" and "uncertainty" in strategies:
        strategies = [
            ("error_similarity" if s == "uncertainty" else s) for s in strategies
        ]

    model = "chemprop"
    clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    skipped = 0
    # Cache of per-pool-compound max Tanimoto vs the fraction's *initial* labeled
    # set; reference depends only on the fraction, so it is reused across budgets.
    tc_cache = {}

    for fr in fractions:
        n_f = int(round(fr * len(df_train)))
        pos = sorted_idx[:n_f]
        if n_f == 0:
            continue
        train_df = df_train.iloc[pos].reset_index(drop=True)
        train_X = X_train[pos]
        train_y = y_train[pos]

        for budget in budgets:
            for strat in strategies:
                seeds_run = (
                    list(range(args.seeds)) if strat == "random" else [args.config_seed]
                )
                for s in seeds_run:
                    rid = reporting.run_id(model, strat, fr, s, budget=budget)
                    if args.skip_existing and reporting.run_complete(
                        model, strat, fr, s, budget, args.purchase_step
                    ):
                        skipped += 1
                        continue
                    backend = ChempropBackend(
                        output_dir=models_dir
                        / f"al_chemprop_{strat}_frac{int(round(fr*100)):03d}_budget{budget:05d}_seed{s}",
                        ensemble_size=args.ensemble_size,
                        epochs=args.epochs,
                        seed=args.config_seed,
                    )
                    print(
                        f"[{clock}] {rid}: start "
                        f"(epochs={args.epochs}, ensemble={args.ensemble_size})"
                    )
                    res = run_buy_simulation(
                        backend=backend,
                        strategy=strat,
                        fraction=fr,
                        seed=s,
                        budget=budget,
                        train_df=train_df,
                        train_X=train_X,
                        train_y=train_y,
                        pool_df=df_pool,
                        pool_X=X_pool,
                        pool_y=y_pool,
                        test_df=df_test,
                        test_X=X_test,
                        test_y=y_test,
                        purchase_step=args.purchase_step,
                        pool_max_tc=tc_cache.get(fr),
                    )
                    mpath, ppath = write_run(
                        res["records"], res["purchases"], model, strat, fr, s, budget=budget
                    )
                    if res.get("pool_max_tc") is not None:
                        tc_cache[fr] = res["pool_max_tc"]
                    update_runs(
                        {
                            "run_id": rid,
                            "model": model,
                            "strategy": strat,
                            "fraction": fr,
                            "seed": s,
                            "params": {
                                "purchase_step": args.purchase_step,
                                "probe_frac": 0.1,
                                "epochs": args.epochs,
                                "ensemble_size": args.ensemble_size,
                                "n_initial_labeled": res["n_initial_labeled"],
                                "n_probe": res["n_probe"],
                                "n_rounds": res["n_rounds"],
                            },
                            "created_at": clock,
                            "metrics_file": str(mpath),
                            "selected_file": str(ppath),
                        },
                        model,
                    )
                    last = res["records"][-1]
                    print(
                        f"[{clock}] {rid}: done rmse={last['rmse']:.4f} "
                        f"r2={last['r2']:.4f} n_bought={last['n_bought']}"
                    )

    metrics_df = reporting.collect_metrics(model)
    if (not args.no_figures) and not metrics_df.empty:
        reporting.save_learning_curve_figure(metrics_df, model)
        final = metrics_df.loc[
            metrics_df.groupby(["strategy", "fraction", "budget", "seed"])["round"].idxmax()
        ]
        reporting.save_final_heatmap_figure(final, model)

    if not metrics_df.empty:
        final = metrics_df.loc[
            metrics_df.groupby(["strategy", "fraction", "budget", "seed"])["round"].idxmax()
        ]
        summary = (
            final.groupby(["fraction", "budget", "strategy"], as_index=False)
            .agg(rmse=("rmse", "mean"), r2=("r2", "mean"), n_runs=("seed", "nunique"))
        )
        print("\n" + "=" * 70)
        print("CHEMPROP BUY SWEEP - SUMMARY (final RMSE / R2 per fraction x budget x strategy)")
        print("=" * 70)
        print(
            summary.sort_values(["fraction", "budget", "strategy"]).to_string(
                index=False, float_format=lambda v: f"{v:.4f}"
            )
        )
        print("-" * 70)
        print(f"Metrics written to: {reporting.RESULT_METRICS}")
        if not args.no_figures:
            print(f"Figures written to: {reporting.RESULT_FIGURES}")
        print("=" * 70)

    if args.smoke:
        print("\nSMOKE RUN COMPLETE - tune --fractions/--strategies/--seeds for the full sweep.")

    if args.skip_existing and skipped:
        print(f"Resumed: {skipped} already-complete run(s) skipped.")


if __name__ == "__main__":
    main()
