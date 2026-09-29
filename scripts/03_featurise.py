import sys
from pathlib import Path

# Adds the project root directory and 'src' to Python's module search path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from src.marimo_openadmet.features import smiles_to_combined_features  # noqa: E402


def filter_and_featurize(input_path, output_prefix, processed_dir):
    print(f"Processing {input_path.name}...")
    df = pd.read_parquet(input_path)

    # Filter LogD between -1 and 6
    initial_len = len(df)
    df = df[(df["LogD"] >= -1.0) & (df["LogD"] <= 6.0)].reset_index(drop=True)
    print(f"Filtered LogD [-1, 6]: {initial_len} -> {len(df)} rows")

    # Generate combined ECFP + RDKit 2D descriptor features
    X, valid_idx = smiles_to_combined_features(df["SMILES"].values, skip_invalid=True)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y = df_valid["LogD"].values

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataframe and precomputed numpy arrays
    df_valid.to_parquet(processed_dir / f"{output_prefix}_df.parquet", index=False)
    np.save(processed_dir / f"{output_prefix}_X.npy", X)
    np.save(processed_dir / f"{output_prefix}_y.npy", y)
    print(
        f"Saved combined feature arrays (shape: {X.shape}) for {output_prefix} to {processed_dir}\n"
    )


def main():
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    datasets = ["openadmet_train_std", "openadmet_test_std", "rtlogd_pool_std"]
    for name in datasets:
        path = interim_dir / f"{name}.parquet"
        if path.exists():
            filter_and_featurize(path, name, processed_dir)
        else:
            print(f"Warning: {path} not found, skipping.")


if __name__ == "__main__":
    main()
