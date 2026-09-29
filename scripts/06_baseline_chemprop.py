import json
import sys
from datetime import datetime
from pathlib import Path

# Adds the project root directory and 'src' to Python's module search path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402

from src.marimo_openadmet.models import ChempropModel  # noqa: E402


def evaluate_chemprop(name, train_df, test_df, output_dir):
    print(f"--- Training Chemprop: {name} ---")
    model = ChempropModel(output_dir=output_dir, ensemble_size=3, seed=42)

    # Fit on training data dataframe
    model.fit(train_df, smiles_col="SMILES", target_col="LogD")

    # Predict on test data dataframe
    preds = model.predict(test_df, smiles_col="SMILES")
    y_true = test_df["LogD"].values

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2 = r2_score(y_true, preds)
    print(f"{name} -> Test RMSE: {rmse:.4f} | Test R2: {r2:.4f}\n")
    return rmse, r2


def save_and_print_results(results, test_size):
    """Persist the baseline metrics to disk and print a compact summary table."""
    results_dir = root_dir / "results" / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)[["model", "n_train", "rmse", "r2"]]

    csv_path = results_dir / "chemprop_baselines.csv"
    json_path = results_dir / "chemprop_baselines.json"

    results_df.to_csv(csv_path, index=False)
    payload = {
        "experiment": "chemprop_baselines",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_size": test_size,
        "results": results_df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 70)
    print("CHEMPROP BASELINES - SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("-" * 70)
    best_idx = results_df["r2"].idxmax()
    best = results_df.loc[best_idx]
    print(
        f"Best model: {best['model']} (n_train={int(best['n_train'])})"
        f" -> Test RMSE: {best['rmse']:.4f} | Test R2: {best['r2']:.4f}"
    )
    print(f"Metrics saved to:\n  {csv_path}\n  {json_path}")
    print("=" * 70)


def main():
    interim_dir = Path("data/interim")

    # Load interim datasets (which contain raw SMILES and LogD columns)
    train_df = pd.read_parquet(interim_dir / "openadmet_train_std.parquet")
    test_df = pd.read_parquet(interim_dir / "openadmet_test_std.parquet")
    pool_df = pd.read_parquet(interim_dir / "rtlogd_pool_std.parquet")

    # Apply identical LogD filter range [-1, 6]
    train_df = train_df[
        (train_df["LogD"] >= -1.0) & (train_df["LogD"] <= 6.0)
    ].reset_index(drop=True)
    test_df = test_df[(test_df["LogD"] >= -1.0) & (test_df["LogD"] <= 6.0)].reset_index(
        drop=True
    )
    pool_df = pool_df[(pool_df["LogD"] >= -1.0) & (pool_df["LogD"] <= 6.0)].reset_index(
        drop=True
    )

    combined_df = pd.concat([train_df, pool_df], ignore_index=True)
    evaluations = [
        ("Chemprop OpenADMET Train Only", train_df, "models/cp_train_only"),
        ("Chemprop RTlogD Pool Only", pool_df, "models/cp_pool_only"),
        ("Chemprop Combined (OpenADMET + RTlogD)", combined_df, "models/cp_combined"),
    ]

    results = []
    for name, train, output_dir in evaluations:
        rmse, r2 = evaluate_chemprop(name, train, test_df, output_dir)
        results.append({"model": name, "n_train": len(train), "rmse": rmse, "r2": r2})

    save_and_print_results(results, test_size=len(test_df))


if __name__ == "__main__":
    main()
