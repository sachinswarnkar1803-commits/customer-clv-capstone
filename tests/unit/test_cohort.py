"""Unit tests for cohort analysis, retention matrices, and stability metrics."""

import pandas as pd
import pytest
from src.cohort.cohort_analysis import CohortAnalyzer


@pytest.fixture
def cohort_test_transactions():
    """Create multi-month transaction history across multiple customer cohorts."""
    return pd.DataFrame({
        "customer_id": [
            "C1", "C1", "C1",   # Cohort 2010-01, active period 0, 1, 2
            "C2", "C2",         # Cohort 2010-01, active period 0, 1
            "C3",               # Cohort 2010-01, active period 0
            "C4", "C4",         # Cohort 2010-02, active period 0, 1
            "C5",               # Cohort 2010-02, active period 0
        ],
        "invoice_id": [f"INV_{i}" for i in range(1, 10)],
        "transaction_date": [
            "2010-01-10", "2010-02-12", "2010-03-14",
            "2010-01-20", "2010-02-18",
            "2010-01-25",
            "2010-02-05", "2010-03-22",
            "2010-02-28",
        ],
        "quantity": [1] * 9,
        "unit_price": [100.0] * 9,
        "revenue": [100.0] * 9,
        "stock_code": ["P1"] * 9,
        "description": ["Item"] * 9,
        "country": ["United Kingdom"] * 9,
    })


def test_cohort_analyzer_retention_matrix(cohort_test_transactions):
    analyzer = CohortAnalyzer()
    summary, retention_mat, rev_mat, stability = analyzer.compute_cohorts(cohort_test_transactions)

    # Cohort 2010-01 has 3 customers (C1, C2, C3)
    # Period 0: 3 active -> retention = 1.0
    # Period 1: 2 active (C1, C2) -> retention = 2/3 = 0.6667
    # Period 2: 1 active (C1) -> retention = 1/3 = 0.3333

    assert "2010-01" in retention_mat.index
    assert retention_mat.loc["2010-01", 0] == 1.0
    assert abs(retention_mat.loc["2010-01", 1] - (2 / 3)) < 1e-3
    assert abs(retention_mat.loc["2010-01", 2] - (1 / 3)) < 1e-3

    # Cohort 2010-02 has 2 customers (C4, C5)
    # Period 0: 2 active -> retention = 1.0
    # Period 1: 1 active (C4) -> retention = 0.5
    assert "2010-02" in retention_mat.index
    assert retention_mat.loc["2010-02", 0] == 1.0
    assert abs(retention_mat.loc["2010-02", 1] - 0.5) < 1e-3

    assert stability["total_cohorts"] == 2
    assert stability["min_cohort_size"] == 2
    assert stability["max_cohort_size"] == 3
