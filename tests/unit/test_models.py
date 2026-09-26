"""Unit tests for baseline, purchase, monetary, and survival models."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import HistoricalAverageBaseline, RFMBaseline
from src.models.purchase_model import PurchaseModelBGNBD
from src.models.monetary_model import MonetaryModelGammaGamma
from src.models.inactivity_model import InactivitySurvivalModel


@pytest.fixture
def synthetic_customer_features():
    """Generates a small customer feature dataframe suitable for fitting models."""
    np.random.seed(42)
    n = 100
    # Customer tenure between 30 and 400 days
    tenure = np.random.uniform(30, 400, n)
    # Recency <= tenure
    recency = tenure * np.random.uniform(0.1, 0.95, n)
    # Frequency: count of repeat purchases
    frequency = np.random.poisson(lam=3, size=n)
    # Monetary: average repeat basket spend
    monetary = np.random.gamma(shape=5, scale=20, size=n)
    # For customers with frequency == 0, monetary is 0
    monetary[frequency == 0] = 0.0

    days_since_last = tenure - recency
    total_rev = monetary * frequency + 50.0  # include first order spend

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "frequency": frequency,
        "recency_days": np.round(recency, 1),
        "customer_tenure_days": np.round(tenure, 1),
        "days_since_last_purchase": np.round(days_since_last, 1),
        "monetary_value": np.round(monetary, 2),
        "total_revenue": np.round(total_rev, 2),
        "average_order_value": np.round(total_rev / (frequency + 1), 2),
        "rfm_segment": np.random.choice(["Champions", "Loyal Customers", "At Risk", "Lost / Dormant"], size=n),
    })
    return df


def test_baseline_models(synthetic_customer_features):
    hist_base = HistoricalAverageBaseline().fit(synthetic_customer_features)
    p_hist = hist_base.predict(synthetic_customer_features, horizon_days=90)
    assert len(p_hist) == len(synthetic_customer_features)
    assert (p_hist >= 0).all()

    rfm_base = RFMBaseline().fit(synthetic_customer_features)
    p_rfm = rfm_base.predict(synthetic_customer_features, horizon_days=90)
    assert len(p_rfm) == len(synthetic_customer_features)
    assert (p_rfm >= 0).all()


def test_purchase_model_bgnbd(synthetic_customer_features):
    bgf = PurchaseModelBGNBD().fit(synthetic_customer_features)
    p_alive = bgf.predict_p_alive(synthetic_customer_features)
    exp_purch = bgf.predict_expected_purchases(synthetic_customer_features, horizon_days=90)

    assert len(p_alive) == len(synthetic_customer_features)
    assert (p_alive >= 0.0).all() and (p_alive <= 1.0).all()
    assert len(exp_purch) == len(synthetic_customer_features)
    assert (exp_purch >= 0.0).all()


def test_monetary_model_gamma_gamma(synthetic_customer_features):
    ggf = MonetaryModelGammaGamma().fit(synthetic_customer_features)
    exp_monetary = ggf.predict_expected_average_spend(synthetic_customer_features)

    assert len(exp_monetary) == len(synthetic_customer_features)
    assert (exp_monetary > 0).all()
    assert ggf.assumption_check["assumption_holds"] in (True, False)


def test_survival_inactivity_model(synthetic_customer_features):
    surv = InactivitySurvivalModel().fit(synthetic_customer_features)
    probs = surv.predict_inactivity_probabilities(synthetic_customer_features)

    assert "inactivity_prob_30d" in probs.columns
    assert "inactivity_prob_60d" in probs.columns
    assert "inactivity_prob_90d" in probs.columns
    assert "inactivity_risk_tier" in probs.columns
    assert (probs["inactivity_prob_90d"] >= 0.0).all() and (probs["inactivity_prob_90d"] <= 1.0).all()
