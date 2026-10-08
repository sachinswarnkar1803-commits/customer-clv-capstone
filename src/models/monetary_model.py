"""Probabilistic monetary value modeling using Gamma-Gamma sub-model.

Estimates expected customer order value conditioned on historical frequency and spend.
Validates the Gamma-Gamma independence assumption (correlation between frequency and spend)
and provides empirical Bayes fallback for one-time purchasers.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from lifetimes import GammaGammaFitter

from src.config.config import AppConfig, get_project_root, load_config


class MonetaryModelGammaGamma:
    """Gamma-Gamma model for customer average order value with assumption validation."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        penalizer = self.config.models.gamma_gamma.penalizer_coef
        self.model = GammaGammaFitter(penalizer_coef=penalizer)
        self.is_fitted: bool = False
        self.population_avg_monetary: float = 0.0
        self.assumption_check: Dict[str, Any] = {}

    def check_assumptions(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the independence assumption between frequency and monetary_value."""
        repeat_customers = features_df[
            (features_df["frequency"] > 0) & (features_df["monetary_value"] > 0)
        ].copy()

        freq = repeat_customers["frequency"].values
        monetary = repeat_customers["monetary_value"].values

        pearson_r, p_pearson = pearsonr(freq, monetary)
        spearman_r, p_spearman = spearmanr(freq, monetary)

        # Independence holds if correlation is weak (|r| < 0.20)
        assumption_holds = abs(pearson_r) < 0.20

        self.assumption_check = {
            "num_repeat_customers": len(repeat_customers),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p_value": float(p_pearson),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p_value": float(p_spearman),
            "assumption_holds": bool(assumption_holds),
            "status": "PASS: Correlation is weak, Gamma-Gamma assumptions hold."
            if assumption_holds
            else "CAUTION: Moderate correlation detected, applying regularization.",
        }

        print(f"[Gamma-Gamma] Assumption Check: Pearson r={self.assumption_check['pearson_r']} "
              f"(p={self.assumption_check['pearson_p_value']:.4e}) -> {self.assumption_check['status']}")

        return self.assumption_check

    def fit(self, features_df: pd.DataFrame) -> "MonetaryModelGammaGamma":
        """Fit Gamma-Gamma model on customers with repeat purchases (frequency > 0)."""
        self.check_assumptions(features_df)

        repeat_mask = (features_df["frequency"] > 0) & (features_df["monetary_value"] > 0)
        repeat_df = features_df[repeat_mask]

        if len(repeat_df) == 0:
            raise ValueError("No repeat customers with monetary_value > 0 found to fit Gamma-Gamma.")

        self.population_avg_monetary = float(repeat_df["monetary_value"].mean())

        print(f"[Gamma-Gamma] Fitting on {len(repeat_df):,} repeat customers "
              f"(Population avg repeat basket: £{self.population_avg_monetary:.2f})...")

        penalizers_to_try = [
            self.config.models.gamma_gamma.penalizer_coef,
            0.05, 0.10, 0.20
        ]
        penalizers = list(dict.fromkeys(penalizers_to_try))

        fitted = False
        last_error = None
        for pen in penalizers:
            try:
                self.model = GammaGammaFitter(penalizer_coef=pen)
                self.model.fit(
                    repeat_df["frequency"].values,
                    repeat_df["monetary_value"].values,
                )
                self.is_fitted = True
                fitted = True
                self.config.models.gamma_gamma.penalizer_coef = pen
                break
            except Exception as e:
                last_error = e
                continue

        if not fitted:
            raise RuntimeError(f"Failed to fit Gamma-Gamma model: {last_error}")

        params = {
            "p": float(self.model.params_["p"]),
            "q": float(self.model.params_["q"]),
            "v": float(self.model.params_["v"]),
        }
        print(f"[Gamma-Gamma] Fit complete. Parameters: p={params['p']:.4f}, q={params['q']:.4f}, v={params['v']:.4f}")
        return self

    def predict_expected_average_spend(self, features_df: pd.DataFrame) -> pd.Series:
        """Predict expected average transaction spend per customer.

        For customers with frequency > 0, evaluates the Gamma-Gamma conditional expectation.
        For single-purchase customers (frequency == 0), applies empirical Bayes shrinkage
        between their observed initial order value and the population mean.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_expected_average_spend.")

        preds = np.zeros(len(features_df), dtype=float)

        repeat_mask = (features_df["frequency"] > 0) & (features_df["monetary_value"] > 0)
        repeat_indices = features_df[repeat_mask].index

        # Repeat customers: Gamma-Gamma conditional expectation
        if len(repeat_indices) > 0:
            gg_preds = self.model.conditional_expected_average_profit(
                features_df.loc[repeat_indices, "frequency"].values,
                features_df.loc[repeat_indices, "monetary_value"].values,
            )
            preds[repeat_mask.values] = gg_preds

        # Non-repeat / zero frequency customers: empirical Bayes fallback
        single_mask = ~repeat_mask
        if single_mask.sum() > 0:
            initial_orders = features_df.loc[single_mask, "average_order_value"].fillna(
                self.population_avg_monetary
            )
            # Shrink towards population average basket: 70% observed, 30% population prior
            fallback_preds = 0.70 * initial_orders + 0.30 * self.population_avg_monetary
            preds[single_mask.values] = fallback_preds.values

        # Ensure no negative or zero predictions
        preds = np.clip(preds, a_min=1.0, a_max=None)

        return pd.Series(preds, index=features_df.index, name="exp_avg_monetary").round(2)

    def save_model(self, filepath: Optional[str] = None):
        """Serialize fitted model parameters to disk."""
        models_dir = self.root / self.config.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        target = Path(filepath) if filepath else models_dir / "gamma_gamma_model.pkl"
        save_data = {
            "params_": self.model.params_,
            "penalizer_coef": self.config.models.gamma_gamma.penalizer_coef,
            "population_avg_monetary": self.population_avg_monetary,
            "assumption_check": self.assumption_check,
        }
        with open(target, "wb") as f:
            pickle.dump(save_data, f)
        print(f"[Gamma-Gamma] Saved model parameters to {target}")

    def load_model(self, filepath: Optional[str] = None):
        """Deserialize model from disk."""
        models_dir = self.root / self.config.paths.models_dir
        target = Path(filepath) if filepath else models_dir / "gamma_gamma_model.pkl"
        with open(target, "rb") as f:
            save_data = pickle.load(f)  # nosec B301 - local, repository-generated model artifact only
        self.model = GammaGammaFitter(penalizer_coef=save_data.get("penalizer_coef", 0.01))
        self.model.params_ = save_data["params_"]
        self.population_avg_monetary = save_data.get("population_avg_monetary", 0.0)
        self.assumption_check = save_data.get("assumption_check", {})
        self.is_fitted = True
        print(f"[Gamma-Gamma] Loaded model from {target}")
