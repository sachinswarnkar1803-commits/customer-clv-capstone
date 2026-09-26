"""Data validation and cleaning pipeline module for BDS-34 Capstone Project.

Ensures strict schema validation, anomaly detection, non-silent filtering,
and complete audit trail logging for the UCI Online Retail II dataset.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config


class DataValidator:
    """Validates raw retail transactions, extracts quality metrics, and builds clean tables."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.audit_log: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": self.config.project.version,
            "cleaning_config": self.config.cleaning.model_dump(),
            "steps": [],
            "metrics": {},
        }

    def _log_step(self, step_name: str, dropped_rows: int, remaining_rows: int, notes: str = ""):
        self.audit_log["steps"].append({
            "step": step_name,
            "rows_affected": int(dropped_rows),
            "remaining_rows": int(remaining_rows),
            "notes": notes,
        })

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map raw UCI column names to standardized snake_case schema."""
        col_mapping = {
            "Invoice": "invoice_id",
            "invoice": "invoice_id",
            "StockCode": "stock_code",
            "stockcode": "stock_code",
            "Description": "description",
            "description": "description",
            "Quantity": "quantity",
            "quantity": "quantity",
            "InvoiceDate": "transaction_date",
            "invoicedate": "transaction_date",
            "Price": "unit_price",
            "price": "unit_price",
            "UnitPrice": "unit_price",
            "Customer ID": "customer_id",
            "customer_id": "customer_id",
            "CustomerID": "customer_id",
            "Country": "country",
            "country": "country",
        }
        df = df.copy()
        # Rename existing columns
        rename_dict = {col: col_mapping[col] for col in df.columns if col in col_mapping}
        df = df.rename(columns=rename_dict)
        return df

    def validate_and_clean(
        self,
        df_raw: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Execute validation and cleaning pipeline with full audit trail.

        Returns:
            clean_df: Filtered dataset containing valid completed commercial purchases.
            annotated_df: Complete dataset with anomaly and quality indicator flags.
            audit_report: Comprehensive audit dictionary.
        """
        df = self.standardize_column_names(df_raw)
        initial_count = len(df)
        self.audit_log["initial_row_count"] = initial_count
        print(f"[Validation] Starting validation on {initial_count:,} raw records...")

        # 1. Type casting and date parsing
        df["invoice_id"] = df["invoice_id"].astype(str).str.strip()
        df["stock_code"] = df["stock_code"].astype(str).str.strip()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
        df["description"] = df["description"].fillna("").astype(str).str.strip()

        # Format Customer ID: clean floats (e.g. 12345.0 -> '12345')
        def format_cust_id(val):
            if pd.isna(val):
                return np.nan
            try:
                # Handle numeric float representation
                num_val = float(val)
                if np.isnan(num_val):
                    return np.nan
                return str(int(num_val))
            except (ValueError, TypeError):
                s = str(val).strip()
                return s if s and s.lower() != "nan" else np.nan

        df["customer_id"] = df["customer_id"].apply(format_cust_id)

        # 2. Anomaly & Quality Indicator Flags
        df["is_cancelled"] = df["invoice_id"].str.startswith("C", na=False)
        df["is_return"] = (df["quantity"] < 0) | df["is_cancelled"]
        df["has_valid_customer"] = df["customer_id"].notna()
        df["has_positive_price"] = df["unit_price"] > 0
        df["has_positive_quantity"] = df["quantity"] > 0
        df["is_valid_date"] = df["transaction_date"].notna() & (
            df["transaction_date"] >= "2009-12-01"
        ) & (df["transaction_date"] <= "2011-12-31")

        # Duplicate check across raw transaction attributes
        df["is_duplicate"] = df.duplicated(
            subset=["invoice_id", "stock_code", "customer_id", "transaction_date", "quantity", "unit_price"],
            keep="first",
        )

        annotated_df = df.copy()

        # 3. Non-silent sequential filtering for clean analytical transactions
        clean_mask = pd.Series(True, index=df.index)

        # Check: Missing customer ID
        missing_cust_count = (~df["has_valid_customer"]).sum()
        if self.config.cleaning.drop_missing_customer_id:
            clean_mask &= df["has_valid_customer"]
            self._log_step("drop_missing_customer_id", missing_cust_count, clean_mask.sum(),
                           "Guest checkouts without persistent customer ID cannot be modeled longitudinally.")

        # Check: Cancellations
        cancelled_count = df["is_cancelled"].sum()
        if self.config.cleaning.exclude_cancellations:
            clean_mask &= ~df["is_cancelled"]
            self._log_step("exclude_cancellations", cancelled_count, clean_mask.sum(),
                           "Invoices prefixed with 'C' indicate canceled transactions.")

        # Check: Non-positive quantities (returns, write-offs)
        non_pos_qty = (df["quantity"] <= 0).sum()
        clean_mask &= (df["quantity"] >= self.config.cleaning.min_quantity)
        self._log_step("filter_non_positive_quantity", non_pos_qty, clean_mask.sum(),
                       f"Quantity must be >= {self.config.cleaning.min_quantity} for completed purchases.")

        # Check: Non-positive price
        non_pos_price = (df["unit_price"] <= 0).sum()
        clean_mask &= (df["unit_price"] >= self.config.cleaning.min_unit_price)
        self._log_step("filter_non_positive_price", non_pos_price, clean_mask.sum(),
                       f"Unit price must be >= {self.config.cleaning.min_unit_price} GBP.")

        # Check: Outlier quantity and price
        outlier_qty = (df["quantity"] > self.config.cleaning.max_quantity_outlier).sum()
        clean_mask &= (df["quantity"] <= self.config.cleaning.max_quantity_outlier)
        outlier_price = (df["unit_price"] > self.config.cleaning.max_price_outlier).sum()
        clean_mask &= (df["unit_price"] <= self.config.cleaning.max_price_outlier)
        self._log_step("filter_extreme_outliers", outlier_qty + outlier_price, clean_mask.sum(),
                       "Extreme outliers (data entry errors or manual adjustments).")

        # Check: Invalid dates
        invalid_dates = (~df["is_valid_date"]).sum()
        clean_mask &= df["is_valid_date"]
        self._log_step("filter_invalid_dates", invalid_dates, clean_mask.sum(),
                       "Dates must be within official dataset scope (2009-12-01 to 2011-12-31).")

        # Check: Duplicates
        dup_count = df["is_duplicate"].sum()
        if self.config.cleaning.deduplicate:
            clean_mask &= ~df["is_duplicate"]
            self._log_step("deduplicate_records", dup_count, clean_mask.sum(),
                           "Exact duplicate line items removed.")

        clean_df = df[clean_mask].copy()

        # 4. Feature derivation on clean transactions
        clean_df["revenue"] = (clean_df["quantity"] * clean_df["unit_price"]).round(4)
        clean_df["invoice_month"] = clean_df["transaction_date"].dt.to_period("M").astype(str)

        # Anonymize customer IDs if enabled
        if self.config.cleaning.anonymize_customer_id:
            cust_mapping = {cid: f"CUST_{i:06d}" for i, cid in enumerate(clean_df["customer_id"].unique(), 1)}
            clean_df["customer_id"] = clean_df["customer_id"].map(cust_mapping)

        # 5. Summary metrics
        final_count = len(clean_df)
        total_revenue = float(clean_df["revenue"].sum())
        unique_customers = int(clean_df["customer_id"].nunique())
        unique_invoices = int(clean_df["invoice_id"].nunique())
        unique_products = int(clean_df["stock_code"].nunique())
        min_date = clean_df["transaction_date"].min().strftime("%Y-%m-%d")
        max_date = clean_df["transaction_date"].max().strftime("%Y-%m-%d")

        self.audit_log["final_clean_row_count"] = final_count
        self.audit_log["retention_rate_pct"] = round((final_count / initial_count) * 100, 2)
        self.audit_log["metrics"] = {
            "total_clean_revenue_gbp": round(total_revenue, 2),
            "unique_customers": unique_customers,
            "unique_invoices": unique_invoices,
            "unique_products": unique_products,
            "min_transaction_date": min_date,
            "max_transaction_date": max_date,
            "missing_customer_rows": int(missing_cust_count),
            "cancelled_rows": int(cancelled_count),
            "return_rows": int(non_pos_qty),
            "zero_price_rows": int(non_pos_price),
            "duplicate_rows": int(dup_count),
        }

        print(f"[Validation] Cleaned dataset ready: {final_count:,} records "
              f"({self.audit_log['retention_rate_pct']}% retained).")
        print(f"[Validation] Unique Customers: {unique_customers:,} | Total Revenue: £{total_revenue:,.2f}")

        return clean_df, annotated_df, self.audit_log

    def save_reports(
        self,
        clean_df: pd.DataFrame,
        annotated_df: pd.DataFrame,
        audit_log: Dict[str, Any],
    ):
        """Save cleaned datasets and audit trail reports to disk."""
        validation_dir = self.root / self.config.paths.validation_dir
        reports_dir = self.root / self.config.paths.reports_dir
        processed_dir = self.root / self.config.paths.processed_dir

        validation_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save audit log JSON
        audit_path = validation_dir / "audit_trail.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_log, f, indent=2)
        print(f"[Validation] Saved audit trail: {audit_path}")

        # 2. Save Markdown Data Quality Report
        md_path = reports_dir / "data_quality_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Data Quality & Preprocessing Audit Report\n\n")
            f.write(f"**Project**: {self.config.project.name} ({self.config.project.code})\n")
            f.write(f"**Generated**: {audit_log['timestamp']}\n\n")
            f.write("## 1. Executive Summary\n\n")
            f.write(f"- **Initial Raw Records**: {audit_log['initial_row_count']:,}\n")
            f.write(f"- **Clean Analytical Records**: {audit_log['final_clean_row_count']:,}\n")
            f.write(f"- **Retention Rate**: {audit_log['retention_rate_pct']}%\n")
            f.write(f"- **Total Clean Revenue**: £{audit_log['metrics']['total_clean_revenue_gbp']:,.2f}\n")
            f.write(f"- **Unique Active Customers**: {audit_log['metrics']['unique_customers']:,}\n")
            f.write(f"- **Date Span**: {audit_log['metrics']['min_transaction_date']} to {audit_log['metrics']['max_transaction_date']}\n\n")
            f.write("## 2. Non-Silent Filtering Decisions\n\n")
            f.write("| Pipeline Step | Rows Removed | Remaining Rows | Methodological Rationale |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for step in audit_log["steps"]:
                f.write(f"| `{step['step']}` | {step['rows_affected']:,} | {step['remaining_rows']:,} | {step['notes']} |\n")
            f.write("\n## 3. Data Anomaly Breakdown\n\n")
            f.write(f"- **Missing Customer IDs**: {audit_log['metrics']['missing_customer_rows']:,} records\n")
            f.write(f"- **Cancellations (Invoice starts with 'C')**: {audit_log['metrics']['cancelled_rows']:,} records\n")
            f.write(f"- **Non-Positive Quantities**: {audit_log['metrics']['return_rows']:,} records\n")
            f.write(f"- **Zero / Negative Unit Prices**: {audit_log['metrics']['zero_price_rows']:,} records\n")
            f.write(f"- **Duplicate Rows**: {audit_log['metrics']['duplicate_rows']:,} records\n")
        print(f"[Validation] Saved data quality report: {md_path}")

        # 3. Save clean analytical dataset
        clean_parquet = self.root / self.config.paths.clean_transactions
        clean_csv = self.root / self.config.paths.clean_transactions_csv
        try:
            clean_df.to_parquet(clean_parquet, index=False)
            print(f"[Validation] Saved clean parquet: {clean_parquet}")
        except Exception as e:
            print(f"[Validation] Parquet save note: {e}")

        # Always save CSV copy as well for interoperability
        clean_df.to_csv(clean_csv, index=False)
        print(f"[Validation] Saved clean CSV: {clean_csv}")


def run_validation_pipeline(use_cache: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete ingestion, validation, and quality reporting."""
    from src.data.ingest import get_raw_data

    raw_df = get_raw_data(use_cache=use_cache)
    validator = DataValidator()
    clean_df, annotated_df, audit_log = validator.validate_and_clean(raw_df)
    validator.save_reports(clean_df, annotated_df, audit_log)
    return clean_df, audit_log


if __name__ == "__main__":
    run_validation_pipeline()
