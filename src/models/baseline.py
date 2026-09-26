"""Baseline customer value models for benchmark comparison.

Implements:
1. Baseline A: Historical Run-Rate / Average Spend Extrapolation
2. Baseline B: RFM Segment-Average Benchmarking
"""

from typing import Optional
import numpy as np
import pandas as pd
from src.config.config import AppConfig, load_config


class HistoricalAverageBaseline:
    """Baseline A: Predicts future revenue by annualizing/extrapolating historical daily spend rate."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.population_daily_rate: float = 0.0

    def fit(self, features_df: pd.DataFrame) -> "HistoricalAverageBaseline":
        """Compute population daily spend rate for empirical Bayesian shrinkage."""
        valid = features_df[features_df["customer_tenure_days"] > 0]
        total_rev = valid["total_revenue"].sum()
        total_tenure = valid["customer_tenure_days"].sum()
        self.population_daily_rate = float(total_rev / max(1.0, total_tenure))
        return self

    def predict(
        self,
        features_df: pd.DataFrame,
        horizon_days: int = 90,
        shrinkage_weight: float = 30.0,
    ) -> pd.Series:
        """Predict expected future revenue over horizon_days.

        Uses shrinkage towards population mean to stabilize customers with very short tenure.
        """
        tenure = features_df["customer_tenure_days"].clip(lower=1.0)
        observed_rev = features_df["total_revenue"]

        # Shrunk daily spend rate
        shrunk_daily_rate = (
            observed_rev + shrinkage_weight * self.population_daily_rate
        ) / (tenure + shrinkage_weight)

        predicted_val = shrunk_daily_rate * horizon_days
        return predicted_val.round(2)


class RFMBaseline:
    """Baseline B: Predicts future revenue based on historical RFM segment average spend."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.segment_means: dict = {}
        self.global_mean: float = 0.0

    def fit(self, features_df: pd.DataFrame) -> "RFMBaseline":
        """Learn historical average spend per RFM segment."""
        self.global_mean = float(features_df["total_revenue"].mean())
        # Average spend per rfm_segment
        seg_group = features_df.groupby("rfm_segment")["total_revenue"].mean()
        self.segment_means = seg_group.to_dict()
        return self

    def predict(
        self,
        features_df: pd.DataFrame,
        horizon_days: int = 90,
        historical_span_days: float = 365.0,
    ) -> pd.Series:
        """Predict future spend by scaling segment historical spend by horizon ratio."""
        scaling_factor = horizon_days / max(1.0, historical_span_days)
        segment_preds = features_df["rfm_segment"].map(self.segment_means).fillna(self.global_mean)
        return (segment_preds * scaling_factor).round(2)
