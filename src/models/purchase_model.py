"""Probabilistic repeat-purchase modeling using Beta-Geometric / Negative Binomial Distribution (BG/NBD).

Models customer transaction rates and dropout probabilities to estimate:
- P(Alive): Probability that customer is currently active
- E[Y(t)]: Expected future transactions over horizon t
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter

from src.config.config import AppConfig, get_project_root, load_config


class PurchaseModelBGNBD:
    """BG/NBD probabilistic transaction model."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        penalizer = self.config.models.bgnbd.penalizer_coef
        self.model = BetaGeoFitter(penalizer_coef=penalizer)
        self.is_fitted: bool = False
        self.diagnostics: Dict[str, Any] = {}

    def fit(self, features_df: pd.DataFrame) -> "PurchaseModelBGNBD":
        """Fit BG/NBD model on customer summary features (frequency, recency_days, customer_tenure_days)."""
        print(f"[BG/NBD] Fitting model on {len(features_df):,} customers...")
        freq = features_df["frequency"].values.astype(int)
        rec = features_df["recency_days"].values.copy()
        tenure = features_df["customer_tenure_days"].values

        # Enforce BG/NBD standard: recency must be 0 if frequency is 0
        rec[freq == 0] = 0.0

        penalizers_to_try = [
            self.config.models.bgnbd.penalizer_coef,
            0.05, 0.10, 0.20, 0.50
        ]
        # Remove duplicates while preserving order
        penalizers = list(dict.fromkeys(penalizers_to_try))

        fitted = False
        last_error = None
        for pen in penalizers:
            try:
                self.model = BetaGeoFitter(penalizer_coef=pen)
                self.model.fit(freq, rec, tenure)
                self.is_fitted = True
                fitted = True
                self.config.models.bgnbd.penalizer_coef = pen
                break
            except Exception as e:
                last_error = e
                continue

        if not fitted:
            raise RuntimeError(f"Failed to fit BG/NBD model after multiple penalizers: {last_error}")

        params = {
            "r": float(self.model.params_["r"]),
            "alpha": float(self.model.params_["alpha"]),
            "a": float(self.model.params_["a"]),
            "b": float(self.model.params_["b"]),
        }
        self.diagnostics = {
            "parameters": params,
            "penalizer_coef": self.config.models.bgnbd.penalizer_coef,
            "fitted_customers": len(features_df),
        }
        print(f"[BG/NBD] Fit complete. Parameters: r={params['r']:.4f}, alpha={params['alpha']:.4f}, "
              f"a={params['a']:.4f}, b={params['b']:.4f}")
        return self

    def predict_p_alive(self, features_df: pd.DataFrame) -> pd.Series:
        """Compute P(Alive | x, t_x, T) for each customer."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_p_alive.")
        freq = features_df["frequency"].values.astype(int)
        rec = features_df["recency_days"].values.copy()
        rec[freq == 0] = 0.0
        tenure = features_df["customer_tenure_days"].values

        p_alive = self.model.conditional_probability_alive(freq, rec, tenure)
        return pd.Series(p_alive, index=features_df.index, name="p_alive").round(4)

    def predict_expected_purchases(
        self,
        features_df: pd.DataFrame,
        horizon_days: int = 90,
    ) -> pd.Series:
        """Predict expected number of repeat purchases in the next horizon_days."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_expected_purchases.")
        freq = features_df["frequency"].values.astype(int)
        rec = features_df["recency_days"].values.copy()
        rec[freq == 0] = 0.0
        tenure = features_df["customer_tenure_days"].values

        exp_purchases = self.model.conditional_expected_number_of_purchases_up_to_time(
            horizon_days,
            freq,
            rec,
            tenure,
        )
        return pd.Series(
            exp_purchases,
            index=features_df.index,
            name=f"exp_purchases_{horizon_days}d",
        ).round(3)

    def save_model(self, filepath: Optional[str] = None):
        """Serialize fitted model parameters to disk."""
        models_dir = self.root / self.config.paths.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)
        target = Path(filepath) if filepath else models_dir / "bg_nbd_model.pkl"
        save_data = {
            "params_": self.model.params_,
            "penalizer_coef": self.config.models.bgnbd.penalizer_coef,
            "diagnostics": self.diagnostics,
        }
        with open(target, "wb") as f:
            pickle.dump(save_data, f)
        print(f"[BG/NBD] Saved model parameters to {target}")

    def load_model(self, filepath: Optional[str] = None):
        """Deserialize model from disk."""
        models_dir = self.root / self.config.paths.models_dir
        target = Path(filepath) if filepath else models_dir / "bg_nbd_model.pkl"
        with open(target, "rb") as f:
            save_data = pickle.load(f)  # nosec B301 - local, repository-generated model artifact only
        self.model = BetaGeoFitter(penalizer_coef=save_data.get("penalizer_coef", 0.05))
        self.model.params_ = save_data["params_"]
        self.diagnostics = save_data.get("diagnostics", {})
        self.is_fitted = True
        print(f"[BG/NBD] Loaded model from {target}")
