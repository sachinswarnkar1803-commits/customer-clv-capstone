"""Unit tests for probabilistic CLV calculation, uncertainty intervals, segmentation, and NBA."""

import numpy as np
import pandas as pd
import pytest

from src.models.purchase_model import PurchaseModelBGNBD
from src.models.monetary_model import MonetaryModelGammaGamma
from src.clv.clv_calculator import ProbabilisticCLVCalculator
from src.segmentation.segmenter import ActionSegmenter
from src.nba.nba_engine import NBAEngine


@pytest.fixture
def fitted_models_and_features():
    np.random.seed(42)
    n = 80
    tenure = np.random.uniform(50, 365, n)
    recency = tenure * np.random.uniform(0.1, 0.9, n)
    frequency = np.random.poisson(lam=3, size=n)
    recency[frequency == 0] = 0.0

    monetary = np.random.gamma(shape=4, scale=25, size=n)
    monetary[frequency == 0] = 0.0
    total_rev = np.where(frequency > 0, monetary * frequency + 40.0, 40.0)

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "frequency": frequency,
        "recency_days": np.round(recency, 1),
        "customer_tenure_days": np.round(tenure, 1),
        "days_since_last_purchase": np.round(tenure - recency, 1),
        "monetary_value": np.round(monetary, 2),
        "total_revenue": np.round(total_rev, 2),
        "average_order_value": np.round(total_rev / (frequency + 1), 2),
        "inactivity_prob_90d": np.random.uniform(0.1, 0.9, n).round(3),
    })

    bgf = PurchaseModelBGNBD().fit(df)
    ggf = MonetaryModelGammaGamma().fit(df)
    return df, bgf, ggf


def test_clv_calculator_outputs(fitted_models_and_features):
    df, bgf, ggf = fitted_models_and_features
    clv_calc = ProbabilisticCLVCalculator()
    clv_df = clv_calc.compute_clv(df, bgf, ggf, n_bootstrap_samples=30)

    assert "clv_expected_90d" in clv_df.columns
    assert "clv_expected_365d" in clv_df.columns
    assert "clv_lower_80pct" in clv_df.columns
    assert "clv_upper_80pct" in clv_df.columns
    assert "clv_uncertainty_spread" in clv_df.columns

    # Verify interval ordering
    assert (clv_df["clv_upper_80pct"] >= clv_df["clv_lower_80pct"]).all()
    # Longer horizons have greater or equal expected value
    assert (clv_df["clv_expected_365d"] >= clv_df["clv_expected_90d"] * 0.8).all()


def test_action_segmentation(fitted_models_and_features):
    df, bgf, ggf = fitted_models_and_features
    clv_calc = ProbabilisticCLVCalculator()
    clv_df = clv_calc.compute_clv(df, bgf, ggf, n_bootstrap_samples=20)

    segmenter = ActionSegmenter()
    seg_df = segmenter.segment_customers(clv_df)

    assert "action_segment" in seg_df.columns
    valid_segments = set(segmenter.get_segment_definitions().keys())
    assert set(seg_df["action_segment"].unique()).issubset(valid_segments)


def test_nba_engine_prescriptions(fitted_models_and_features):
    df, bgf, ggf = fitted_models_and_features
    clv_calc = ProbabilisticCLVCalculator()
    clv_df = clv_calc.compute_clv(df, bgf, ggf, n_bootstrap_samples=20)
    seg_df = ActionSegmenter().segment_customers(clv_df)

    nba = NBAEngine()
    final_df = nba.generate_recommendations(seg_df)

    assert "nba_recommended_action" in final_df.columns
    assert "nba_priority" in final_df.columns
    assert "nba_channel" in final_df.columns
    assert "nba_estimated_cost" in final_df.columns
    assert "nba_expected_incremental_value" in final_df.columns
    assert (final_df["nba_estimated_cost"] > 0).all()
