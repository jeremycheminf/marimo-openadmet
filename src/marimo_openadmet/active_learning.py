"""Round-based "buy external data" active-learning engine.

Shared by the ensemble and chemprop sweeps (scripts 08/09).

PURCHASE DESIGN (absolute budgets): each run starts from an initial labeled
slice (f% of OpenADMET train, minus a fixed probe) and buys an absolute
number of RTLogD pool compounds (budget = 100 / 500 / 1000 / 5000),
split into chunks of ``purchase_step * budget`` per round. So the purchase
is *on top of* the training slice: e.g. fraction 10% with budget 500
starts from ~450 labeled compounds and buys 500 more.

The engine is backend-agnostic: adapters (``EnsembleBackend`` / ``ChempropBackend``)
expose the same ``fit`` / ``predict`` / ``predict_std`` interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, r2_score

from .models import ChempropModel, RefinementStack
from .selectors import (
    ModelErrorSimilaritySelector,
    RandomSelector,
    TopScoreSelector,
    UncertaintySelector,
    tanimoto_maxmin,
)

STRATEGIES = (
    "random",
    "diversity_tanimoto",
    "uncertainty",
    "similarity",
    "error_similarity",
)

METRICS_COLUMNS = [
    "model",
    "strategy",
    "fraction",
    "budget",
    "seed",
    "round",
    "purchase_pct",
    "n_train",
    "n_bought",
    "rmse",
    "r2",
]

PURCHASE_COLUMNS = [
    "strategy",
    "fraction",
    "budget",
    "seed",
    "round",
    "purchase_pct",
    "pool_index",
    "smiles",
    "oracle_logd",
    "acquisition_score",
    "rank_in_round",
]


class EnsembleBackend:
    """RefinementStack backend: operates on precomputed combined features."""

    model_name = "ensemble"

    def __init__(self, n_folds=3, seed=42):
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, train_df, train_X, train_y):
        model = RefinementStack(n_folds=self.n_folds, seed=self.seed)
        model.fit(np.asarray(train_X), np.asarray(train_y))
        self.model = model

    def predict(self, test_df, test_X):
        return self.model.predict(np.asarray(test_X))

    def predict_std(self, cand_df, cand_X):
        _, std = self.model.predict_with_std(np.asarray(cand_X))
        return np.asarray(std)


class ChempropBackend:
    """Chemprop backend: operates on SMILES dataframes."""

    model_name = "chemprop"

    def __init__(self, output_dir, ensemble_size=3, epochs=5, seed=42):
        self.output_dir = Path(output_dir)
        self.ensemble_size = ensemble_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, train_df, train_X, train_y):
        fit_df = train_df[["SMILES", "LogD"]].copy()
        self.model = ChempropModel(
            output_dir=self.output_dir,
            ensemble_size=self.ensemble_size,
            seed=self.seed,
            epochs=self.epochs,
        )
        self.model.fit(fit_df, smiles_col="SMILES", target_col="LogD")

    def predict(self, test_df, test_X):
        return self.model.predict(test_df[["SMILES"]], smiles_col="SMILES")

    def predict_std(self, cand_df, cand_X):
        _, std = self.model.predict_with_std(cand_df[["SMILES"]], smiles_col="SMILES")
        return np.asarray(std)


# Shared Morgan generator (modern rdFingerprintGenerator API; radius 2, 2048 bits,
# matching the featurisation used everywhere else in this project).
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _smiles_to_fingerprints(smiles_list):
    """Morgan bit-vector fingerprints aligned to the input list (skip invalid)."""
    fps = []
    valid = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        fps.append(MORGAN_GENERATOR.GetFingerprint(mol))
        valid.append(i)
    return fps, valid

def max_tanimoto_to_reference(query_smiles, reference_smiles, return_index=False):
    """Max Tanimoto similarity of every query compound to a fixed reference set.

    Returns an array aligned 1:1 with ``query_smiles`` (invalid SMILES give
    NaN), so callers can always assign the result back to the query frame.
    Used by the ``similarity`` strategy ("buy what is closest to my project's
    chemistry") and by the notebook asset builder (per-compound ``max_tc``).

    With ``return_index=True`` also returns the position *in the reference
    list* of the closest compound (-1 where undefined), which the notebook's
    molecule detail card uses to show a purchase beside its nearest
    in-house neighbour. The argmax rides along on the same BulkTanimoto pass,
    so it costs nothing extra.
    """
    query_smiles = list(query_smiles)
    reference_smiles = list(reference_smiles)
    query_fps, valid = _smiles_to_fingerprints(query_smiles)
    ref_fps, ref_valid = _smiles_to_fingerprints(reference_smiles)
    out = np.full(len(query_smiles), np.nan, dtype=float)
    idx = np.full(len(query_smiles), -1, dtype=np.int32)
    if ref_fps:
        for rank, fp in enumerate(query_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            if not sims:
                continue
            best = int(np.argmax(sims))
            out[valid[rank]] = sims[best]
            # map back through ref_valid: unparseable references were dropped
            idx[valid[rank]] = ref_valid[best]
    return (out, idx) if return_index else out


def run_buy_simulation(
    *,
    backend,
    strategy: str,
    fraction: float,
    seed: int,
    train_df: pd.DataFrame,
    train_X: Optional[np.ndarray],
    train_y: np.ndarray,
    pool_df: pd.DataFrame,
    pool_X: Optional[np.ndarray],
    pool_y: np.ndarray,
    test_df: pd.DataFrame,
    test_X: Optional[np.ndarray],
    test_y: np.ndarray,
    purchase_step: float = 0.25,
    probe_frac: float = 0.1,
    error_frac: float = 0.25,
    budget: int = 1000,
    pool_max_tc: Optional[np.ndarray] = None,
    model_name: Optional[str] = None,
):
    """Run the buy-simulation for one (strategy, fraction, seed) combo."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}; expected one of {STRATEGIES}")

    model_name = model_name or backend.model_name
    train_X = None if train_X is None else np.asarray(train_X)
    pool_X = None if pool_X is None else np.asarray(pool_X)
    train_y = np.asarray(train_y)
    pool_y = np.asarray(pool_y)
    test_y = np.asarray(test_y)

    n_all = len(train_df)

    # ---- carve a deterministic residual-probe set (uniform across strategies) ----
    probe_rng = np.random.default_rng(1000 + int(round(fraction * 100)))
    n_probe = max(1, int(round(n_all * probe_frac)))
    probe_idx = np.sort(probe_rng.choice(n_all, size=n_probe, replace=False))
    label_idx = np.setdiff1d(np.arange(n_all), probe_idx)

    labeled_df = train_df.iloc[label_idx].reset_index(drop=True)
    probe_df = train_df.iloc[probe_idx].reset_index(drop=True)
    labeled_y = train_y[label_idx]
    probe_y = train_y[probe_idx]
    labeled_X = train_X[label_idx] if train_X is not None else None
    probe_X = train_X[probe_idx] if train_X is not None else None

    L = len(labeled_df)
    n_rounds = max(1, int(round(1.0 / purchase_step)))
    step = max(1, int(np.ceil(budget / n_rounds)))
    pool_len = len(pool_df)

    # fingerprint / precomputed-score channels
    pool_fps = None
    error_sel = None
    tanimoto_fps = None
    if strategy in ("error_similarity", "diversity_tanimoto"):
        pool_fps, pool_valid = _smiles_to_fingerprints(pool_df["SMILES"].values)
    if strategy == "diversity_tanimoto":
        tanimoto_fps = [None] * pool_len
        for fp, i in zip(pool_fps, pool_valid):
            tanimoto_fps[i] = fp
    if strategy == "error_similarity":
        probe_fps, _ = _smiles_to_fingerprints(probe_df["SMILES"].values)
        error_sel = ModelErrorSimilaritySelector(error_frac=error_frac)

    # `similarity` targets relevance directly: max Tanimoto of each pool
    # compound to the *original* internal train set, fixed across rounds
    # ("relevance to my project", deliberately not train+bought -- the adaptive
    # form chases clusters).
    max_tc = None if pool_max_tc is None else np.asarray(pool_max_tc, dtype=float)
    if strategy == "similarity":
        if max_tc is not None and len(max_tc) != pool_len:
            raise ValueError(
                f"pool_max_tc has {len(max_tc)} entries but the pool has {pool_len}"
            )
        if max_tc is None:
            max_tc = max_tanimoto_to_reference(
                pool_df["SMILES"].values, labeled_df["SMILES"].values
            )

    records: list[dict] = []
    purchases: list[dict] = []
    purchased: list[int] = []
    bought_set: set[int] = set()
    rng = RandomSelector(random_state=seed) if strategy == "random" else None

    # ---- round 0: baseline, nothing bought yet ----
    backend.fit(labeled_df, labeled_X, labeled_y)
    yhat = backend.predict(test_df, test_X)
    rmse = float(np.sqrt(mean_squared_error(test_y, yhat)))
    r2 = float(r2_score(test_y, yhat))
    records.append(
        {
            "model": model_name,
            "strategy": strategy,
            "fraction": fraction,
            "budget": int(budget),
            "seed": seed,
            "round": 0,
            "purchase_pct": 0.0,
            "n_train": L,
            "n_bought": 0,
            "rmse": rmse,
            "r2": r2,
        }
    )
    if strategy == "error_similarity":
        probe_pred = backend.predict(probe_df, probe_X)
        error_sel.set_probe(probe_fps, np.abs(probe_y - probe_pred))

    # ---- buy rounds ----
    for rnd in range(1, n_rounds + 1):
        remaining = np.array([i for i in range(pool_len) if i not in bought_set])
        k = min(step, len(remaining), budget - len(purchased))
        if k <= 0:
            break

        if strategy == "diversity_tanimoto":
            # Fingerprint-native MaxMin: distances are pure Tanimoto
            # chemistry. Candidates whose SMILES failed to parse have no
            # fingerprint; they are appended only if the parseable ones
            # cannot fill the batch (never happens on the standardised pool).
            cand = [tanimoto_fps[int(i)] for i in remaining]
            ok = [i for i, fp in enumerate(cand) if fp is not None]
            bad = [i for i, fp in enumerate(cand) if fp is None]
            sel_sub = tanimoto_maxmin([cand[i] for i in ok], min(k, len(ok)), seed=seed)
            picked = [ok[i] for i in sel_sub] + bad[: max(0, k - len(ok))]
            sel = np.array(picked[:k], dtype=int)
            scores = np.full(len(sel), np.nan)
        elif strategy == "uncertainty":
            cand_feats = pool_X[remaining] if pool_X is not None else None
            std = backend.predict_std(pool_df.iloc[remaining], cand_feats)
            sel = UncertaintySelector().select(None, k, std)
            scores = np.asarray(std)[sel]
        elif strategy == "error_similarity":
            sims = error_sel.score_pool([pool_fps[i] for i in remaining])
            sel = TopScoreSelector().select(None, k, sims)
            scores = sims[sel]
        elif strategy == "similarity":
            sims = max_tc[remaining]
            sel = TopScoreSelector().select(None, k, sims)
            scores = np.asarray(sims)[sel]
        elif strategy == "random":
            sel = rng.select(remaining, k)
            scores = np.full(k, np.nan)
        else:  # pragma: no cover - guarded above
            raise ValueError(strategy)

        pick = remaining[sel]
        for idx_pick, (pi, sc) in enumerate(zip(pick, scores)):
            purchases.append(
                {
                    "strategy": strategy,
                    "fraction": fraction,
                    "budget": int(budget),
                    "seed": seed,
                    "round": rnd,
                    "purchase_pct": 0.0,  # filled after the batch is finalized
                    "pool_index": int(pi),
                    "smiles": str(pool_df.iloc[int(pi)]["SMILES"]),
                    "oracle_logd": float(pool_y[int(pi)]),
                    "acquisition_score": float(sc) if not np.isnan(sc) else np.nan,
                    "rank_in_round": idx_pick,
                }
            )

        purchased.extend(int(pi) for pi in pick)
        bought_set.update(int(pi) for pi in pick)

        pct_now = min(100.0, len(purchased) / budget * 100.0)
        for row in purchases[-(len(pick)):]:
            row["purchase_pct"] = pct_now

        cur_X = (
            np.vstack([labeled_X, pool_X[purchased]]) if labeled_X is not None else None
        )
        cur_y = np.concatenate([labeled_y, pool_y[purchased]])
        cur_df = pd.concat([labeled_df, pool_df.iloc[purchased]], ignore_index=True)

        backend.fit(cur_df, cur_X, cur_y)
        yhat = backend.predict(test_df, test_X)
        rmse = float(np.sqrt(mean_squared_error(test_y, yhat)))
        r2 = float(r2_score(test_y, yhat))
        records.append(
            {
                "model": model_name,
                "strategy": strategy,
                "fraction": fraction,
                "budget": int(budget),
                "seed": seed,
                "round": rnd,
                "purchase_pct": pct_now,
                "n_train": L + len(purchased),
                "n_bought": len(purchased),
                "rmse": rmse,
                "r2": r2,
            }
        )

        if strategy == "error_similarity":
            probe_pred = backend.predict(probe_df, probe_X)
            error_sel.set_probe(probe_fps, np.abs(probe_y - probe_pred))

    return {
        "records": records,
        "purchases": purchases,
        "n_initial_labeled": L,
        "n_probe": n_probe,
        "n_rounds": n_rounds,
        "step": step,
        # Returned so the sweep scripts can cache it across budgets of the same
        # fraction (the reference set only depends on the fraction, not the
        # budget or the seed).
        "pool_max_tc": max_tc if strategy == "similarity" else None,
    }
