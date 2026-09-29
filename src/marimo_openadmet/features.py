# src/marimo_openadmet/features.py
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np


def get_descriptor_names():
    return [x[0] for x in Descriptors.descList]


def smiles_to_ecfp(smiles_list, radius=2, n_bits=2048, skip_invalid=True):
    """Legacy/compatibility function returning only ECFP bit vectors using MorganGenerator."""
    features_list = []
    valid_indices = []

    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            if skip_invalid:
                continue
            else:
                raise ValueError(f"Invalid SMILES at index {i}: {s}")

        fp = gen.GetFingerprintAsNumPy(mol)
        features_list.append(fp.astype(np.float32))
        valid_indices.append(i)

    return np.array(features_list, dtype=np.float32), valid_indices


def smiles_to_combined_features(smiles_list, radius=2, n_bits=2048, skip_invalid=True):
    """Combines modern MorganGenerator ECFP bit vectors with all ~200 RDKit 2D molecular descriptors."""
    desc_names = get_descriptor_names()
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    features_list = []
    valid_indices = []

    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            if skip_invalid:
                continue
            else:
                raise ValueError(f"Invalid SMILES at index {i}: {s}")

        # 1. ECFP Fingerprint (using modern MorganGenerator)
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)

        # 2. RDKit 2D Descriptors
        desc_vals = list(calc.CalcDescriptors(mol))
        desc_vals = np.nan_to_num(desc_vals, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate both feature sets
        combined = np.concatenate([fp, desc_vals])
        features_list.append(combined)
        valid_indices.append(i)

    return np.array(features_list, dtype=np.float32), valid_indices
