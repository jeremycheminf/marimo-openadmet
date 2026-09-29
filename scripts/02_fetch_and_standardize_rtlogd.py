from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_and_filter(smiles, min_mw=150, max_mw=600):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            # 1. Basic cleanup and largest fragment selection (strips salts/solvents)
            cleaned = rdMolStandardize.Cleanup(mol)
            parent = rdMolStandardize.LargestFragmentChooser().choose(cleaned)

            # 2. Re-sanitize to ensure validity after fragment stripping
            Chem.SanitizeMol(parent)

            # 3. Apply molecular weight filter
            mw = Descriptors.MolWt(parent)
            if min_mw < mw < max_mw:
                return Chem.MolToSmiles(parent, isomericSmiles=True)
    except Exception:
        pass
    return None


def main():
    data_dir = Path("data/raw")
    data_dir_processed = Path("data/interim")
    # data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/myzhengSIMM/RTlogD/main/"
    tdata_url = base_url + "T-data_predictions(chembl32_logD).csv"
    orig_base_url = base_url + "original_data/"
    orig_files = ["Lipo_logD.csv", "DB29-data.csv"]

    dfs = []

    print("Downloading T-data predictions...")
    try:
        df_t = pd.read_csv(tdata_url)
        if "smiles" in df_t.columns and "standard_value" in df_t.columns:
            subset = df_t[["smiles", "standard_value"]].rename(
                columns={"smiles": "SMILES", "standard_value": "LogD"}
            )
            dfs.append(subset)
    except Exception as e:
        print(f"Error loading T-data: {e}")

    for f in orig_files:
        url = orig_base_url + f
        try:
            print(f"Downloading {f}...")
            temp_df = pd.read_csv(url)
            s_col = next(
                (
                    c
                    for c in ["SMILES", "smiles", "Canonical_SMILES", "can"]
                    if c in temp_df.columns
                ),
                None,
            )
            l_col = next(
                (
                    c
                    for c in [
                        "LogD",
                        "logD",
                        "logD7.4",
                        "standard_value",
                        "target",
                        "Y",
                        "exp",
                    ]
                    if c in temp_df.columns
                ),
                None,
            )

            if s_col and l_col:
                subset = temp_df[[s_col, l_col]].rename(
                    columns={s_col: "SMILES", l_col: "LogD"}
                )
                dfs.append(subset)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        raise ValueError("No data could be retrieved from RTlogD repository.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["SMILES", "LogD"])
    output_path = data_dir / "rtlogd_pool.parquet"
    combined.to_parquet(output_path, index=False)

    print(
        "Standardizing molecules (salt stripping + largest fragment) and filtering by MW (150 < MW < 600)..."
    )
    combined["Standardized_SMILES"] = combined["SMILES"].apply(standardize_and_filter)
    combined = combined.dropna(subset=["Standardized_SMILES"])

    # Drop duplicates based on the finalized canonical structure
    combined = combined.drop_duplicates(subset=["Standardized_SMILES"]).reset_index(
        drop=True
    )

    output_df = combined[["Standardized_SMILES", "LogD"]].rename(
        columns={"Standardized_SMILES": "SMILES"}
    )
    output_path = data_dir_processed / "rtlogd_pool_std.parquet"
    output_df.to_parquet(output_path, index=False)
    print(f"Saved standardized pool of {len(output_df)} records to {output_path}")


if __name__ == "__main__":
    main()
