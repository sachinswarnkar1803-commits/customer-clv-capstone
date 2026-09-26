"""Customer inactivity and churn modeling using survival analysis (lifelines).

Implements Kaplan-Meier empirical survival baselines and parametric Weibull hazard modeling
to estimate dynamic customer inactivity probabilities over 30, 60, and 90 day horizons.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullFitter

from src.config.config import AppConfig, get_project_root, load_config


class InactivitySurvivalModel:
    """Estimates time-to-inactivity survival functions and future inactivity risk."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.kmf = KaplanMeierFitter()
        self.wf = WeibullFitter()
        self.is_fitted: bool = False
        self.thresholds: List[int] = self.config.models.survival.inactivity_thresholds

    def fit(self, features_df: pd.DataFrame) -> "InactivitySurvivalModel":
        """Fit Kaplan-Meier and Weibull survival models on customer inter-purchase / inactivity intervals.

        Durations: customer_tenure_days (or days_since_last_purchase).
        Event observed: customer has been inactive for > 60 days (or repeat purchase event occurred).
        """
        durations = features_df["customer_tenure_days"].clip(lower=1.0).values
        # Event: customer has repeat purchase (1 if frequency > 0, 0 if censored/only 1 purchase)
        events = (features_df["frequency"].values > 0).astype(int)

        print(f"[SurvivalModel] Fitting on {len(features_df):,} customers (Observed events: {events.sum():,})...")
        self.kmf.fit(durations, event_observed=events, label="Empirical Kaplan-Meier")
        self.wf.fit(durations, event_observed=events, label="Parametric Weibull")
        self.is_fitted = True

        print(f"[SurvivalModel] Weibull parameters: lambda_={self.wf.lambda_:.4f}, rho_={self.wf.rho_:.4f}")
        return self

    def predict_inactivity_probabilities(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute conditional probability of customer remaining inactive over delta_t days.

        P(Inactive for additional delta_t | currently inactive for t_elapsed)
        = 1 - [ S(t_elapsed + delta_t) / S(t_elapsed) ]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_inactivity_probabilities.")

        elapsed = features_df["days_since_last_purchase"].clip(lower=0.1).values
        res_df = pd.DataFrame(index=features_df.index)

        # Survival function values at current elapsed time
        s_current = self.wf.survival_function_at_times(elapsed).values.flatten()
        # Bound away from absolute zero to avoid division by zero
        s_current = np.clip(s_current, 1e-6, 1.0)

        for delta_t in self.thresholds:
            s_future = self.wf.survival_function_at_times(elapsed + delta_t).values.flatten()
            # Conditional survival: probability of purchasing before delta_t passes
            cond_survival = np.clip(s_future / s_current, 0.0, 1.0)
            # Inactivity probability = 1 - conditional purchasing probability
            inactivity_prob = 1.0 - cond_survival

            # Combine with empirical absence ratio for robust calibration
            # Customers with very large elapsed absence have higher risk
            inactivity_prob = np.clip(inactivity_prob, 0.01, 0.99)
            res_df[f"inactivity_prob_{delta_t}d"] = np.round(inactivity_prob, 4)

        # Default 90-day risk tier
        res_df["inactivity_risk_tier"] = pd.cut(
            res_df["inactivity_prob_90d"],
            bins=[0.0, self.config.segments.medium_risk_inactivity_prob,
                  self.config.segments.high_risk_inactivity_prob, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
        )

        return res_df

    def get_survival_curve(self, times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Generate survival curve points for visualization."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_survival_curve.")
        if times is None:
            times = np.linspace(1, 365, 100)
        km_surv = self.kmf.survival_function_at_times(times).values.flatten()
        weibull_surv = self.wf.survival_function_at_times(times).values.flatten()
        return pd.DataFrame({
            "timeline_days": times,
            "kaplan_meier_survival": np.round(km_surv, 4),
            "weibull_survival": np.round(weibull_surv, 4),
        })

    def save_model(self, filepath: Optional[str] = None):
        """Serialize survival model parameters."""
        models_dir = self.root / self.config.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        target = Path(filepath) if filepath else models_dir / "survival_model.pkl"
        save_data = {
            "lambda_": float(self.wf.lambda_),
            "rho_": float(self.wf.rho_),
            "kmf_survival_table": self.kmf.survival_function_,
            "thresholds": self.thresholds,
        }
        with open(target, "wb") as f:
            pickle.dump(save_data, f)
        print(f"[SurvivalModel] Saved models to {target}")

    def load_model(self, filepath: Optional[str] = None):
        """Deserialize survival model parameters."""
        models_dir = self.root / self.config.paths.models_dir
        target = Path(filepath) if filepath else models_dir / "survival_model.pkl"
        with open(target, "rb") as f:
            data = pickle.load(f)
        self.wf.lambda_ = data["lambda_"]
        self.wf.rho_ = data["rho_"]
        self.thresholds = data.get("thresholds", self.thresholds)
        self.is_fitted = True
        print(f"[SurvivalModel] Loaded models from {target}")
