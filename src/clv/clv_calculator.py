"""Probabilistic customer lifetime value with predictive uncertainty intervals."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config
from src.models.purchase_model import PurchaseModelBGNBD
from src.models.monetary_model import MonetaryModelGammaGamma


class ProbabilisticCLVCalculator:
    """Compute discounted CLV and empirical Monte Carlo prediction intervals."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.discount_rate = self.config.clv.discount_rate
        self.horizons = self.config.clv.horizons_days
        self.confidence_level = self.config.clv.confidence_level

    def compute_clv(
        self,
        features_df: pd.DataFrame,
        bgf_wrapper: PurchaseModelBGNBD,
        ggf_wrapper: MonetaryModelGammaGamma,
        n_simulation_samples: int = 500,
    ) -> pd.DataFrame:
        """Compute expected CLV and predictive uncertainty.

        The interval is intentionally named a *Monte Carlo prediction interval*.
        It captures stochastic future purchase counts and basket values, but does
        not claim to include fitted-parameter uncertainty as a bootstrap CI.
        """
        df = features_df.copy()
        p_alive = bgf_wrapper.predict_p_alive(df)
        exp_monetary = ggf_wrapper.predict_expected_average_spend(df)
        df["p_alive"] = np.clip(np.asarray(p_alive, dtype=float), 0.0, 1.0)
        df["exp_avg_monetary"] = np.maximum(np.nan_to_num(np.asarray(exp_monetary, dtype=float), nan=0.0), 0.0)

        for horizon_days in self.horizons:
            months = horizon_days / 30.0
            exp_purch = np.maximum(np.asarray(bgf_wrapper.predict_expected_purchases(df, horizon_days), dtype=float), 0.0)
            df[f"exp_purchases_{horizon_days}d"] = exp_purch
            discount_factor = 1.0 / ((1.0 + self.discount_rate) ** (months / 2.0))
            df[f"clv_expected_{horizon_days}d"] = np.round(exp_purch * df["exp_avg_monetary"] * discount_factor, 2)

        horizon = self.config.clv.default_horizon_days
        months = horizon / 30.0
        discount = 1.0 / ((1.0 + self.discount_rate) ** (months / 2.0))
        mean_purch = np.clip(np.nan_to_num(df[f"exp_purchases_{horizon}d"].to_numpy(float), nan=0.0), 0.0, 300.0)
        mean_spend = np.clip(np.nan_to_num(df["exp_avg_monetary"].to_numpy(float), nan=0.0), 0.0, 20000.0)

        samples = max(100, int(n_simulation_samples))
        rng = np.random.default_rng(self.config.project.random_seed)
        # Gamma shape is estimated from the cross-sectional basket distribution;
        # using a robust fixed shape avoids pretending this is parameter bootstrap.
        positive_spend = mean_spend[mean_spend > 0]
        shape = 4.0
        if len(positive_spend) > 20 and np.std(positive_spend) > 0:
            shape = float(np.clip(np.mean(positive_spend) ** 2 / np.var(positive_spend), 0.5, 20.0))

        sample_clv = np.empty((samples, len(df)), dtype=float)
        for i in range(samples):
            purchases = rng.poisson(mean_purch)
            spend = np.where(
                mean_spend > 0,
                rng.gamma(shape=shape, scale=np.maximum(mean_spend, 1e-9) / shape),
                0.0,
            )
            sample_clv[i] = purchases * spend * discount

        alpha = (1.0 - self.confidence_level) / 2.0
        lower = np.quantile(sample_clv, alpha, axis=0)
        upper = np.quantile(sample_clv, 1.0 - alpha, axis=0)
        level = int(round(self.confidence_level * 100))
        df[f"clv_lower_{level}pct"] = np.round(lower, 2)
        df[f"clv_upper_{level}pct"] = np.round(upper, 2)
        expected = np.maximum(df[f"clv_expected_{horizon}d"].to_numpy(float), 0.0)
        df["clv_uncertainty_spread"] = np.round((upper - lower) / (expected + 1.0), 3)
        df["clv_interval_method"] = "Monte Carlo predictive interval"
        return df

    def explain_uncertainty_interval(self) -> str:
        level = int(self.confidence_level * 100)
        return (
            f"The {level}% interval is an empirical Monte Carlo prediction interval for the next "
            f"{self.config.clv.default_horizon_days} days. It propagates stochastic variation in future "
            "transaction counts and basket values. It is not a bootstrap confidence interval for fitted model parameters."
        )
