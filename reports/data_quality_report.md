# Data Quality & Preprocessing Audit Report

**Project**: Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments (BDS-34)
**Generated**: 2026-09-24T14:31:45.966968+00:00

## 1. Executive Summary

- **Initial Raw Records**: 1,067,371
- **Clean Analytical Records**: 779,421
- **Retention Rate**: 73.02%
- **Total Clean Revenue**: £17,128,599.22
- **Unique Active Customers**: 5,878
- **Date Span**: 2009-12-01 to 2011-12-09

## 2. Non-Silent Filtering Decisions

| Pipeline Step | Rows Removed | Remaining Rows | Methodological Rationale |
| :--- | :--- | :--- | :--- |
| `drop_missing_customer_id` | 243,007 | 824,364 | Guest checkouts without persistent customer ID cannot be modeled longitudinally. |
| `exclude_cancellations` | 19,494 | 805,620 | Invoices prefixed with 'C' indicate canceled transactions. |
| `filter_non_positive_quantity` | 22,950 | 805,620 | Quantity must be >= 1 for completed purchases. |
| `filter_non_positive_price` | 6,207 | 805,549 | Unit price must be >= 0.001 GBP. |
| `filter_extreme_outliers` | 2 | 805,547 | Extreme outliers (data entry errors or manual adjustments). |
| `filter_invalid_dates` | 0 | 805,547 | Dates must be within official dataset scope (2009-12-01 to 2011-12-31). |
| `deduplicate_records` | 34,337 | 779,421 | Exact duplicate line items removed. |

## 3. Data Anomaly Breakdown

- **Missing Customer IDs**: 243,007 records
- **Cancellations (Invoice starts with 'C')**: 19,494 records
- **Non-Positive Quantities**: 22,950 records
- **Zero / Negative Unit Prices**: 6,207 records
- **Duplicate Rows**: 34,337 records
