"""Unit tests for temporal splitting, leakage prevention, and customer feature engineering."""

import pandas as pd
import pytest
from src.features.splitter import TemporalSplitter
from src.features.feature_builder import CustomerFeatureBuilder


@pytest.fixture
def sample_transaction_history():
    return pd.DataFrame({
        "customer_id": ["C1", "C1", "C1", "C2", "C2", "C3"],
        "invoice_id": ["INV1", "INV2", "INV3", "INV4", "INV5", "INV6"],
        "transaction_date": [
            "2010-01-15", # C1 order 1
            "2010-03-20", # C1 order 2
            "2011-02-10", # C1 order 3 (FUTURE beyond cutoff 2010-12-09)
            "2010-05-10", # C2 order 1
            "2010-05-10", # C2 order 1 (same day order 2)
            "2011-05-15", # C3 order 1 (PURELY FUTURE)
        ],
        "stock_code": ["P1", "P2", "P3", "P4", "P5", "P6"],
        "description": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6"],
        "quantity": [2, 1, 3, 5, 2, 1],
        "unit_price": [10.0, 20.0, 15.0, 4.0, 5.0, 50.0],
        "revenue": [20.0, 20.0, 45.0, 20.0, 10.0, 50.0],
        "invoice_month": ["2010-01", "2010-03", "2011-02", "2010-05", "2010-05", "2011-05"],
        "country": ["United Kingdom"] * 6,
    })


def test_temporal_splitter_strict_leakage_barrier(sample_transaction_history):
    splitter = TemporalSplitter()
    cutoff = "2010-12-09"
    train_df, holdout_df = splitter.split_train_holdout(
        sample_transaction_history,
        cutoff_date=cutoff,
    )

    assert train_df["transaction_date"].max() <= pd.to_datetime(cutoff)
    assert holdout_df["transaction_date"].min() > pd.to_datetime(cutoff)

    # In sample: C1 order 1, 2 and C2 orders are train (4 transactions)
    # C1 order 3 and C3 order are holdout (2 transactions)
    assert len(train_df) == 4
    assert len(holdout_df) == 2
    assert "C3" not in train_df["customer_id"].values
    assert "C3" in holdout_df["customer_id"].values


def test_feature_builder_temporal_cutoff(sample_transaction_history):
    builder = CustomerFeatureBuilder()
    cutoff = "2010-12-09"
    features = builder.build_customer_features(
        sample_transaction_history,
        observation_cutoff=cutoff,
    )

    # Customer C3 should NOT exist in training features
    assert "C3" not in features["customer_id"].values

    # Customer C1 has 2 orders prior to cutoff (INV1, INV2)
    c1 = features[features["customer_id"] == "C1"].iloc[0]
    assert c1["total_orders"] == 2
    assert c1["frequency"] == 1  # 2 distinct order days - 1
    assert c1["total_revenue"] == 40.0  # 20.0 + 20.0 (excludes future 45.0)

    # Customer C2 had 2 invoices on the same order day -> frequency 0
    c2 = features[features["customer_id"] == "C2"].iloc[0]
    assert c2["total_orders"] == 2
    assert c2["frequency"] == 0
    assert c2["total_revenue"] == 30.0

    # Ensure RFM scores exist
    assert "r_score" in features.columns
    assert "f_score" in features.columns
    assert "m_score" in features.columns
    assert "rfm_segment" in features.columns
