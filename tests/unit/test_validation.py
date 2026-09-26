"""Unit tests for data validation, cleaning, and audit trail generation."""

import pandas as pd
import numpy as np
import pytest
from src.config.config import load_config
from src.validation.validator import DataValidator


@pytest.fixture
def sample_raw_data():
    """Create a controlled synthetic raw transaction batch covering various edge cases."""
    return pd.DataFrame({
        "Invoice": [
            "489434",      # Valid purchase 1
            "489434",      # Valid purchase 2 (same invoice)
            "C489435",     # Cancellation
            "489436",      # Missing Customer ID
            "489437",      # Negative quantity (return)
            "489438",      # Zero unit price
            "489439",      # Duplicate row A
            "489439",      # Duplicate row B (exact duplicate)
            "489440",      # Invalid date (future/out of bounds)
            "489441",      # Extreme quantity outlier
        ],
        "StockCode": [
            "85048", "79323P", "22386", "48187", "84997C",
            "21508", "21508", "21508", "21508", "21508",
        ],
        "Description": [
            "15CM CHRISTMAS GLASS BALL",
            "PINK CHERRY LIGHTS",
            "JUMBO BAG RED RETROSPOT",
            "DOOR MAT NEW ENGLAND",
            "RED SPOTTY BOWL",
            "CERAMIC CAKE STAND",
            "CERAMIC CAKE STAND",
            "CERAMIC CAKE STAND",
            "CERAMIC CAKE STAND",
            "CERAMIC CAKE STAND",
        ],
        "Quantity": [
            12, 6, -1, 4, -5, 10, 2, 2, 1, 999999,
        ],
        "InvoiceDate": [
            "2009-12-01 07:45:00",
            "2009-12-01 07:45:00",
            "2009-12-01 09:24:00",
            "2009-12-01 10:15:00",
            "2009-12-01 11:30:00",
            "2009-12-01 12:00:00",
            "2009-12-01 13:00:00",
            "2009-12-01 13:00:00",
            "2029-01-01 00:00:00", # Out of range
            "2009-12-01 14:00:00", # Outlier
        ],
        "Price": [
            6.95, 6.75, 4.25, 5.95, 2.55, 0.00, 3.50, 3.50, 4.00, 1.00,
        ],
        "Customer ID": [
            13085.0, 13085.0, 13085.0, np.nan, 14527.0, 15311.0, 16000.0, 16000.0, 17000.0, 18000.0,
        ],
        "Country": [
            "United Kingdom", "United Kingdom", "United Kingdom",
            "United Kingdom", "United Kingdom", "United Kingdom",
            "United Kingdom", "United Kingdom", "United Kingdom", "United Kingdom",
        ],
    })


def test_validator_standardizes_columns(sample_raw_data):
    validator = DataValidator()
    df_std = validator.standardize_column_names(sample_raw_data)
    assert "invoice_id" in df_std.columns
    assert "stock_code" in df_std.columns
    assert "unit_price" in df_std.columns
    assert "customer_id" in df_std.columns
    assert "transaction_date" in df_std.columns


def test_validator_filters_anomalies_correctly(sample_raw_data):
    validator = DataValidator()
    clean_df, annotated_df, audit_log = validator.validate_and_clean(sample_raw_data)

    # In our sample of 10 rows:
    # Row 0: Valid (Invoice 489434, qty 12, price 6.95, cust 13085)
    # Row 1: Valid (Invoice 489434, qty 6, price 6.75, cust 13085)
    # Row 2: Cancelled (C489435) -> filtered
    # Row 3: Missing Customer ID -> filtered
    # Row 4: Negative qty (-5) -> filtered
    # Row 5: Zero price (0.00) -> filtered
    # Row 6: Valid row (Invoice 489439, qty 2, price 3.50, cust 16000)
    # Row 7: Duplicate of Row 6 -> deduplicated
    # Row 8: Out of bounds date (2029) -> filtered
    # Row 9: Extreme quantity (999999) -> filtered

    assert len(clean_df) == 3
    assert set(clean_df["customer_id"].unique()) == {"13085", "16000"}

    # Check revenue calculation: qty * unit_price
    expected_rev_0 = round(12 * 6.95, 4)
    expected_rev_1 = round(6 * 6.75, 4)
    expected_rev_2 = round(2 * 3.50, 4)
    actual_revs = clean_df["revenue"].tolist()

    assert expected_rev_0 in actual_revs
    assert expected_rev_1 in actual_revs
    assert expected_rev_2 in actual_revs


def test_validator_audit_trail_populated(sample_raw_data):
    validator = DataValidator()
    clean_df, annotated_df, audit_log = validator.validate_and_clean(sample_raw_data)

    assert audit_log["initial_row_count"] == 10
    assert audit_log["final_clean_row_count"] == 3
    assert len(audit_log["steps"]) > 0
    assert audit_log["metrics"]["unique_customers"] == 2
    assert audit_log["metrics"]["missing_customer_rows"] == 1
    assert audit_log["metrics"]["cancelled_rows"] == 1
    assert audit_log["metrics"]["return_rows"] == 2 # Row 2 & Row 4
