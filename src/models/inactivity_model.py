"""Survival model for time-to-90-day inactivity.

The event is defined as a customer reaching the configured inactivity threshold
without another purchase. Customers who have not reached the threshold by the
observation cutoff are right-censored. This makes the survival target an actual
inactivity outcome rather than a repeat-purchase proxy.
"""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, WeibullFitter

from src.config.config import AppConfig, get_project_root, load_config


class InactivitySurvivalModel:
    """Estimate time-to-inactivity and conditional future inactivity risk."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.kmf = KaplanMeierFitter()
        self.wf = WeibullFitter()
        self.is_fitted = False
        self.thresholds: List[int] = self.config.models.survival.inactivity_thresholds
        self.inactivity_threshold_days = int(max(self.thresholds))

    def _build_survival_target(
        self,
        features_df: pd.DataFrame,
        transactions_df: Optional[pd.DataFrame] = None,
        observation_cutoff: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build duration/event pairs for reaching the inactivity threshold.

        If transactions are available, an event occurs when any inter-purchase
        gap reaches the threshold. Otherwise the method uses a conservative
        feature-only fallback and marks customers as events only when their
        observed inactivity already exceeds the threshold.
        """
        threshold = self.inactivity_threshold_days
        if transactions_df is not None and len(transactions_df) > 0:
            tx = transactions_df.copy()
            tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
            cutoff = pd.to_datetime(observation_cutoff) if observation_cutoff else tx["transaction_date"].max()
            tx = tx[tx["transaction_date"] <= cutoff].sort_values(["customer_id", "transaction_date"])

            durations, events = [], []
            feature_index = features_df.set_index("customer_id")
            for customer_id in features_df["customer_id"]:
                cust = tx.loc[tx["customer_id"] == customer_id, "transaction_date"].drop_duplicates().sort_values()
                if cust.empty:
                    durations.append(1.0)
                    events.append(0)
                    continue

                first = cust.iloc[0]
                gaps = cust.diff().dt.total_seconds().div(86400.0).iloc[1:]
                event_positions = np.flatnonzero(gaps.to_numpy() >= threshold)
                if len(event_positions):
                    prior_date = cust.iloc[event_positions[0]]
                    event_date = prior_date + pd.Timedelta(days=threshold)
                    duration = max(1.0, (event_date - first).total_seconds() / 86400.0)
                    durations.append(duration)
                    events.append(1)
                else:
                    duration = max(1.0, (cutoff - first).total_seconds() / 86400.0)
                    durations.append(duration)
                    events.append(0)
            return np.asarray(durations, dtype=float), np.asarray(events, dtype=int)

        # Feature-only fallback for unit tests or downstream scoring contexts.
        tenure = pd.to_numeric(features_df["customer_tenure_days"], errors="coerce").fillna(1.0)
        inactivity = pd.to_numeric(features_df["days_since_last_purchase"], errors="coerce").fillna(0.0)
        event = (inactivity >= threshold).astype(int).to_numpy()
        duration = np.minimum(tenure.to_numpy(dtype=float), np.maximum(1.0, inactivity.to_numpy(dtype=float) + 1.0))
        return duration, event

    def fit(
        self,
        features_df: pd.DataFrame,
        transactions_df: Optional[pd.DataFrame] = None,
        observation_cutoff: Optional[str] = None,
    ) -> "InactivitySurvivalModel":
        """Fit KM and Weibull models to time-to-inactivity observations."""
        durations, events = self._build_survival_target(features_df, transactions_df, observation_cutoff)
        if events.sum() == 0:
            raise ValueError("No observed inactivity events were found. Increase the observation window or lower the threshold.")

        print(f"[SurvivalModel] Fitting time-to-inactivity model on {len(durations):,} customers; events={events.sum():,}.")
        self.kmf.fit(durations, event_observed=events, label="Kaplan-Meier")
        self.wf.fit(durations, event_observed=events, label="Weibull")
        self.is_fitted = True
        print(f"[SurvivalModel] Weibull parameters: lambda_={self.wf.lambda_:.4f}, rho_={self.wf.rho_:.4f}")
        return self

    def predict_inactivity_probabilities(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate probability of reaching inactivity within each future horizon."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        age = pd.to_numeric(features_df["customer_tenure_days"], errors="coerce").fillna(1.0).clip(lower=0.1).to_numpy()
        current_gap = pd.to_numeric(features_df["days_since_last_purchase"], errors="coerce").fillna(0.0).to_numpy()
        already_inactive = current_gap >= self.inactivity_threshold_days
        s_current = np.clip(self.wf.survival_function_at_times(age).to_numpy().reshape(-1), 1e-8, 1.0)

        result = pd.DataFrame(index=features_df.index)
        for delta_t in self.thresholds:
            s_future = np.clip(self.wf.survival_function_at_times(age + delta_t).to_numpy().reshape(-1), 0.0, 1.0)
            risk = 1.0 - np.clip(s_future / s_current, 0.0, 1.0)
            risk[already_inactive] = 1.0
            result[f"inactivity_prob_{delta_t}d"] = np.round(np.clip(risk, 0.0, 1.0), 4)

        result["inactivity_risk_tier"] = pd.cut(
            result["inactivity_prob_90d"],
            bins=[-0.001, self.config.segments.medium_risk_inactivity_prob,
                  self.config.segments.high_risk_inactivity_prob, 1.001],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True,
        )
        return result

    def get_survival_curve(self, times: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before get_survival_curve.")
        times = np.linspace(1, 365, 100) if times is None else np.asarray(times)
        return pd.DataFrame({
            "timeline_days": times,
            "kaplan_meier_survival": np.round(self.kmf.survival_function_at_times(times).to_numpy().reshape(-1), 4),
            "weibull_survival": np.round(self.wf.survival_function_at_times(times).to_numpy().reshape(-1), 4),
        })

    def save_model(self, filepath: Optional[str] = None):
        models_dir = self.root / self.config.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        target = Path(filepath) if filepath else models_dir / "survival_model.pkl"
        with open(target, "wb") as f:
            pickle.dump({
                "lambda_": float(self.wf.lambda_),
                "rho_": float(self.wf.rho_),
                "kmf_survival_table": self.kmf.survival_function_,
                "thresholds": self.thresholds,
                "inactivity_threshold_days": self.inactivity_threshold_days,
            }, f)

    def load_model(self, filepath: Optional[str] = None):
        models_dir = self.root / self.config.paths.models_dir
        target = Path(filepath) if filepath else models_dir / "survival_model.pkl"
        with open(target, "rb") as f:
            data = pickle.load(f)  # nosec B301 - local, repository-generated model artifact only
        self.wf.lambda_ = data["lambda_"]
        self.wf.rho_ = data["rho_"]
        self.thresholds = data.get("thresholds", self.thresholds)
        self.inactivity_threshold_days = int(data.get("inactivity_threshold_days", max(self.thresholds)))
        self.is_fitted = True
