"""Probabilistic Customer Lifetime Value (CLV) calculation with uncertainty intervals.

Integrates BG/NBD transaction forecasts and Gamma-Gamma monetary expectations with
monthly discounting and bootstrap posterior sampling to compute:
- Expected CLV across configurable horizons (30, 90, 180, 365 days)
- 80% Empirical Prediction Intervals [CLV_lower, CLV_upper]
- Uncertainty relative spread ratio
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

from src.config.config import AppConfig, get_project_root, load_config
from src.models.purchase_model import PurchaseModelBGNBD
from src.models.monetary_model import MonetaryModelGammaGamma


class ProbabilisticCLVCalculator:
    """Computes probabilistic discounted CLV and uncertainty intervals."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.discount_rate = self.config.clv.discount_rate  # Monthly discount rate (e.g., 0.01 = 1%)
        self.horizons = self.config.clv.horizons_days       # [30, 90, 180, 365]
        self.confidence_level = self.config.clv.confidence_level  # 0.80

    def compute_clv(
        self,
        features_df: pd.DataFrame,
        bgf_wrapper: PurchaseModelBGNBD,
        ggf_wrapper: MonetaryModelGammaGamma,
        n_bootstrap_samples: int = 50,
    ) -> pd.DataFrame:
        """Compute expected CLV and empirical uncertainty intervals for all customers.

        Args:
            features_df: Customer feature table with frequency, recency_days, customer_tenure_days, monetary_value.
            bgf_wrapper: Fitted PurchaseModelBGNBD.
            ggf_wrapper: Fitted MonetaryModelGammaGamma.
            n_bootstrap_samples: Number of posterior draws for uncertainty interval estimation.

        Returns:
            pd.DataFrame: Table enriched with CLV and interval columns across horizons.
        """
        df = features_df.copy()
        n_customers = len(df)
        print(f"[CLVCalculator] Computing probabilistic CLV for {n_customers:,} customers...")

        # 1. Base expected purchase counts and expected monetary basket
        p_alive = bgf_wrapper.predict_p_alive(df)
        exp_monetary = ggf_wrapper.predict_expected_average_spend(df)

        df["p_alive"] = p_alive
        df["exp_avg_monetary"] = exp_monetary

        # Enforce BG/NBD inputs
        freq = df["frequency"].values.astype(int)
        rec = df["recency_days"].values.copy()
        rec[freq == 0] = 0.0
        tenure = df["customer_tenure_days"].values

        # 2. Compute Expected CLV across each configurable horizon
        for horizon_days in self.horizons:
            months = horizon_days / 30.0
            # Expected transactions over horizon
            exp_purch = bgf_wrapper.predict_expected_purchases(df, horizon_days=horizon_days)
            df[f"exp_purchases_{horizon_days}d"] = exp_purch

            # Monthly discount factor over horizon midpoint
            discount_factor = 1.0 / ((1.0 + self.discount_rate) ** (months / 2.0))

            # Expected CLV = E[Purchases] * E[Monetary Spend] * Discount Factor
            exp_clv = (exp_purch * exp_monetary * discount_factor).round(2)
            df[f"clv_expected_{horizon_days}d"] = exp_clv

        # 3. Uncertainty Intervals for the Default Horizon (e.g. 90 days)
        def_horizon = self.config.clv.default_horizon_days
        def_months = def_horizon / 30.0
        def_discount = 1.0 / ((1.0 + self.discount_rate) ** (def_months / 2.0))
        mean_purch = np.nan_to_num(df[f"exp_purchases_{def_horizon}d"].values, nan=0.0, posinf=300.0, neginf=0.0)
        mean_purch = np.clip(mean_purch, 0.0, 300.0)

        mean_spend = np.nan_to_num(df["exp_avg_monetary"].values, nan=45.0, posinf=20000.0, neginf=1.0)
        mean_spend = np.clip(mean_spend, 1.0, 20000.0)

        alpha_level = (1.0 - self.confidence_level) / 2.0
        lower_q = alpha_level * 100
        upper_q = (1.0 - alpha_level) * 100

        print(f"[CLVCalculator] Simulating {n_bootstrap_samples} posterior draws for {int(self.confidence_level*100)}% intervals...")
        rng = np.random.default_rng(seed=self.config.project.random_seed)

        k_shape = 4.0
        sample_clvs = np.zeros((n_bootstrap_samples, n_customers))

        for s in range(n_bootstrap_samples):
            # Poisson variation around transaction rate
            sim_purchases = rng.poisson(lam=mean_purch)
            # Spend variation per transaction
            sim_spend = rng.gamma(shape=k_shape, scale=mean_spend / k_shape)
            sim_clv = sim_purchases * sim_spend * def_discount
            sample_clvs[s, :] = sim_clv

        clv_lower = np.percentile(sample_clvs, lower_q, axis=0)
        clv_upper = np.percentile(sample_clvs, upper_q, axis=0)

        df[f"clv_lower_{int(self.confidence_level*100)}pct"] = np.round(clv_lower, 2)
        df[f"clv_upper_{int(self.confidence_level*100)}pct"] = np.round(clv_upper, 2)

        # Relative uncertainty spread = (upper - lower) / (expected + 1)
        spread = (clv_upper - clv_lower) / (df[f"clv_expected_{def_horizon}d"].values + 1.0)
        df["clv_uncertainty_spread"] = np.round(spread, 3)

        print(f"[CLVCalculator] Average Expected {def_horizon}-Day CLV: £{df[f'clv_expected_{def_horizon}d'].mean():.2f} "
              f"(80% Avg Interval: [£{clv_lower.mean():.2f}, £{clv_upper.mean():.2f}])")

        return df

    def explain_uncertainty_interval(self) -> str:
        """Provide a rigorous mathematical and business explanation of the uncertainty interval."""
        return (
            f"The {int(self.confidence_level*100)}% prediction interval [CLV_lower, CLV_upper] "
            "represents the empirical 10th-to-90th percentile range of simulated customer value "
            f"over the next {self.config.clv.default_horizon_days} days. It accounts for stochastic "
            "arrival times (Poisson transaction variance) and basket size dispersion (Gamma monetary variance). "
            "It does NOT guarantee that actual individual outcomes will never fall outside the interval, "
            "but provides marketers with risk bounds for budgeting customer acquisition and retention incentives."
        )
