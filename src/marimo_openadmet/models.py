# src/models.py
from pathlib import Path
import subprocess
import pandas as pd
import pickle
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb


class RefinementStack(BaseEstimator, RegressorMixin):
    """Stacking regressor combining LightGBM, XGBoost, and Random Forest
    under a Ridge meta-learner, providing non-GP uncertainty via base model disagreement[cite: 1].
    """

    def __init__(self, n_folds=5, seed=42):
        self.n_folds = n_folds
        self.seed = seed
        self.final_base_models = []
        self.meta_model = Ridge(alpha=1.0)
        self.is_fitted = False

    def _get_base_estimators(self):
        return [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=100, random_state=self.seed, n_jobs=-1
                ),
            ),
            (
                "lgb",
                lgb.LGBMRegressor(n_estimators=100, random_state=self.seed, verbose=-1),
            ),
            (
                "xgb",
                xgb.XGBRegressor(
                    n_estimators=100, random_state=self.seed, verbosity=0, n_jobs=-1
                ),
            ),
        ]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        oof_preds = np.zeros((X.shape[0], 3))

        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            for i, (_, model_class) in enumerate(self._get_base_estimators()):
                model_class.fit(X_train, y_train)
                oof_preds[val_idx, i] = model_class.predict(X_val)

        # Train meta-model on out-of-fold base predictions
        self.meta_model.fit(oof_preds, y)

        # Fit final base models on the entire dataset for inference
        self.final_base_models = []
        for _, model_class in self._get_base_estimators():
            model_class.fit(X, y)
            self.final_base_models.append(model_class)

        self.is_fitted = True
        return self

    def predict(self, X):
        X = np.asarray(X)
        base_preds = np.column_stack([m.predict(X) for m in self.final_base_models])
        return self.meta_model.predict(base_preds)

    def predict_with_std(self, X):
        X = np.asarray(X)
        base_preds = np.column_stack([m.predict(X) for m in self.final_base_models])
        mean_pred = self.meta_model.predict(base_preds)
        # Standard deviation across base model predictions as uncertainty score
        std_pred = np.std(base_preds, axis=1)
        return mean_pred, std_pred

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


class ChempropModel(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for Chemprop v2 GNN training and prediction."""

    def __init__(self, output_dir="models/chemprop_run", ensemble_size=3, seed=42, epochs=5):
        self.output_dir = Path(output_dir)
        self.ensemble_size = ensemble_size
        self.seed = seed
        self.epochs = epochs
        self.checkpoint_dir = self.output_dir / "checkpoints"

    def fit(self, df_train: pd.DataFrame, smiles_col="SMILES", target_col="LogD"):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_csv = self.output_dir / "train.csv"

        df_train[[smiles_col, target_col]].rename(
            columns={smiles_col: "smiles", target_col: "logd"}
        ).to_csv(train_csv, index=False)

        train_cmd = [
            "chemprop",
            "train",
            "--data-path",
            str(train_csv),
            "--task-type",
            "regression",
            "--target-columns",
            "logd",
            "--ensemble-size",
            str(self.ensemble_size),
            "--output-dir",
            str(self.checkpoint_dir),
            "--epochs",
            str(self.epochs),
        ]
        subprocess.run(train_cmd, check=True)
        return self

    def predict(self, df_test: pd.DataFrame, smiles_col="SMILES"):
        test_csv = self.output_dir / "test.csv"
        preds_csv = self.output_dir / "preds.csv"

        df_test[[smiles_col]].rename(columns={smiles_col: "smiles"}).to_csv(
            test_csv, index=False
        )

        predict_cmd = [
            "chemprop",
            "predict",
            "--test-path",
            str(test_csv),
            "--model-path",
            str(self.checkpoint_dir),
            "--preds-path",
            str(preds_csv),
        ]
        subprocess.run(predict_cmd, check=True)

        preds_df = pd.read_csv(preds_csv)
        # Dynamically grab the prediction column (ignoring SMILES/ID columns)
        pred_col = [
            c for c in preds_df.columns if c not in ["smiles", "SMILES", "ID", "id"]
        ][0]
        return preds_df[pred_col].values

    def predict_with_std(self, df_test: pd.DataFrame, smiles_col="SMILES"):
        test_csv = self.output_dir / "test.csv"
        preds_csv = self.output_dir / "preds.csv"

        df_test[[smiles_col]].rename(columns={smiles_col: "smiles"}).to_csv(
            test_csv, index=False
        )

        predict_cmd = [
            "chemprop",
            "predict",
            "--test-path",
            str(test_csv),
            "--model-path",
            str(self.checkpoint_dir),
            "--preds-path",
            str(preds_csv),
        ]
        subprocess.run(predict_cmd, check=True)

        preds_df = pd.read_csv(preds_csv)
        pred_col = [
            c for c in preds_df.columns if c not in ["smiles", "SMILES", "ID", "id"]
        ][0]
        mean_preds = preds_df[pred_col].values

        # 1) Preferred: an explicit uncertainty column. chemprop writes one only
        #    when --uncertainty-method is passed.
        std_col = next(
            (
                c
                for c in preds_df.columns
                if c not in ["smiles", "SMILES", "ID", "id"]
                and ("std" in c.lower() or "var" in c.lower())
            ),
            None,
        )
        if std_col is not None:
            return mean_preds, preds_df[std_col].values

        # 2) Fallback: chemprop always writes per-model predictions for ensembles
        #    (preds_individual.csv -> columns like logd_model_0..N). The spread
        #    across those members *is* the ensemble uncertainty, and it matches how
        #    RefinementStack.predict_with_std scores its base models.
        ind_path = self.output_dir / "preds_individual.csv"
        if ind_path.exists():
            ind_df = pd.read_csv(ind_path)
            member_cols = [
                c
                for c in ind_df.columns
                if c.lower() not in ("smiles", "id") and "model" in c.lower()
            ]
            if len(member_cols) > 1:
                return mean_preds, ind_df[member_cols].std(axis=1).values

        # 3) Never silently return zeros - that would turn uncertainty-based
        #    selection into an arbitrary "last k" pick and quietly invalidate the
        #    whole uncertainty arm of a sweep.
        raise RuntimeError(
            "Could not derive chemprop uncertainty: no std/var column in "
            f"'{preds_csv.name}' and no multi-model '{ind_path.name}'. Train with "
            "ensemble_size > 1 (or pass an explicit chemprop --uncertainty-method)."
        )
