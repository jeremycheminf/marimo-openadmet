# scripts/01_fetch_openadmet.py
from pathlib import Path
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_and_filter(smiles, min_mw=150, max_mw=600):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            cleaned = rdMolStandardize.Cleanup(mol)
            parent = rdMolStandardize.LargestFragmentChooser().choose(cleaned)
            Chem.SanitizeMol(parent)
            mw = Descriptors.MolWt(parent)
            if min_mw < mw < max_mw:
                return Chem.MolToSmiles(parent, isomericSmiles=True)
    except Exception:
        pass
    return None


def main():
    print("getting data", flush=True)
    data_dir = Path("data/raw")
    data_dir_interim = Path("data/interim")
    # data_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loading OpenADMET Expansion Therapeutics dataset splits from Hugging Face...",
        flush=True,
    )
    train_dataset = load_dataset(
        "openadmet/openadmet-expansionrx-challenge-data", split="train"
    )
    test_dataset = load_dataset(
        "openadmet/openadmet-expansionrx-challenge-data", split="test"
    )

    df_train = train_dataset.to_pandas()
    df_test = test_dataset.to_pandas()

    print(f"Loaded raw train split: {len(df_train)} rows", flush=True)
    print(f"Loaded raw test split: {len(df_test)} rows", flush=True)

    # Validate required columns
    for name, df in [("train", df_train), ("test", df_test)]:
        if "SMILES" not in df.columns:
            raise ValueError(
                f"Expected 'SMILES' column in {name} split, found columns: {df.columns.tolist()}"
            )
        if "LogD" not in df.columns:
            raise ValueError(
                f"Expected 'LogD' column in {name} split, found columns: {df.columns.tolist()}"
            )
    output_path_train = data_dir / "openadmet_train.parquet"
    output_path_test = data_dir / "openadmet_test.parquet"
    df_train.to_parquet(output_path_train, index=False)
    df_test.to_parquet(output_path_test, index=False)

    print(
        "Standardizing SMILES and filtering train split (150 < MW < 600)...", flush=True
    )
    df_train["Standardized_SMILES"] = df_train["SMILES"].apply(standardize_and_filter)
    df_train = df_train.dropna(subset=["Standardized_SMILES"])
    df_train = df_train.drop(columns=["SMILES"]).rename(
        columns={"Standardized_SMILES": "SMILES"}
    )

    print(
        "Standardizing SMILES and filtering test split (150 < MW < 600)...", flush=True
    )
    df_test["Standardized_SMILES"] = df_test["SMILES"].apply(standardize_and_filter)
    df_test = df_test.dropna(subset=["Standardized_SMILES"])
    df_test = df_test.drop(columns=["SMILES"]).rename(
        columns={"Standardized_SMILES": "SMILES"}
    )

    print(f"Cleaned train split: {len(df_train)} rows", flush=True)
    print(f"Cleaned test split: {len(df_test)} rows", flush=True)

    train_path = data_dir_interim / "openadmet_train_std.parquet"
    df_train.to_parquet(train_path, index=False)
    print(f"Saved processed train data to {train_path}", flush=True)

    test_path = data_dir_interim / "openadmet_test.parquet"
    df_test.to_parquet(test_path, index=False)
    print(f"Saved processed test data to {test_path}", flush=True)


if __name__ == "__main__":
    main()
