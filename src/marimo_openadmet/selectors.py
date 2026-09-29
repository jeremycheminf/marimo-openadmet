"""Acquisition selectors used by the buy-simulation engine.

Every selector returns integer positions into the candidate list it was
given; ``run_buy_simulation`` maps those back to pool indices.
"""
import numpy as np
from rdkit import DataStructs


class RandomSelector:
    def __init__(self, random_state=42):
        self.rng = np.random.default_rng(random_state)

    def select(self, features, k, scores=None):
        indices = np.arange(len(features))
        if k >= len(indices):
            return indices
        return self.rng.choice(indices, size=k, replace=False)


class TopScoreSelector:
    def __init__(self, ascending=False):
        self.ascending = ascending

    def select(self, features, k, scores):
        scores = np.asarray(scores)
        if self.ascending:
            indices = np.argsort(scores)
        else:
            indices = np.argsort(scores)[::-1]
        return indices[:k]


def tanimoto_maxmin(fps, k, seed=42):
    """MaxMin coverage on Tanimoto distance (1 - Tc), fingerprint-native.

    Distances are pure chemistry: no ECFP+descriptor concatenation, no PCA,
    no feature-scale artefact. Returns indices into the candidate list.
    """
    n = len(fps)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    best = np.zeros(n, dtype=float)
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[selected[0]], fps))
    best[:] = sims
    for _ in range(1, k):
        nxt = int(np.argmin(best))
        selected.append(nxt)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[nxt], fps))
        np.maximum(best, sims, out=best)
    return np.array(selected, dtype=int)


class UncertaintySelector:
    """Top-k selection by model uncertainty (ensemble std) - AL error proxy."""

    def select(self, features, k, scores):
        scores = np.asarray(scores)
        indices = np.argsort(scores)[::-1]
        return indices[:k]


class ModelErrorSimilaritySelector:
    """Buy pool compounds most similar (Tanimoto) to labeled molecules the current
    model predicts worst.

    The probe set (labeled molecules with known residuals |y - yhat|) is refreshed
    each round via ``set_probe``; selection is then top-k by max similarity to the
    worst-fit probe compounds.
    """

    def __init__(self, error_frac=0.25, min_sim=0.0):
        self.error_frac = error_frac
        self.min_sim = min_sim
        self._error_fps = []

    def set_probe(self, probe_fps, probe_residuals):
        probe_fps = list(probe_fps)
        residuals = np.asarray(probe_residuals, dtype=float)
        n_error = max(1, int(round(len(residuals) * self.error_frac)))
        keep = np.argsort(residuals)[::-1][:n_error]
        self._error_fps = [probe_fps[i] for i in keep]

    def score_pool(self, pool_fps):
        scores = np.zeros(len(pool_fps), dtype=float)
        for i, fp in enumerate(pool_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fp, self._error_fps)
            scores[i] = max(sims) if sims else 0.0
        return np.maximum(scores, self.min_sim)

    def select(self, pool_fps, k):
        scores = self.score_pool(pool_fps)
        indices = np.argsort(scores)[::-1][:k]
        return indices, scores
