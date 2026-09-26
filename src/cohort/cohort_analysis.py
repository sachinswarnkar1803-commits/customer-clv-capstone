"""Cohort and customer retention dynamics analysis module.

Groups customers by acquisition month, computes monthly active customer retention matrices,
repeat purchase rates, cohort revenue decay curves, and cohort stability metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config


class CohortAnalyzer:
    """Computes cohort acquisition matrices, retention rates, and revenue curves."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()

    def compute_cohorts(
        self,
        df_transactions: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Compute full monthly cohort retention and revenue dynamics.

        Args:
            df_transactions: Clean transactions DataFrame.

        Returns:
            cohort_summary: Tabular aggregation by cohort and period.
            retention_matrix: Pivot table of retention rates (0.0 to 1.0).
            revenue_matrix: Pivot table of cohort revenue (£).
            metrics: Cohort stability and comparison metrics.
        """
        df = df_transactions.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["transaction_date"]):
            df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        # Order month and order period
        df["order_month"] = df["transaction_date"].dt.to_period("M")

        # Determine customer first purchase month (Acquisition Cohort)
        first_purchases = df.groupby("customer_id")["order_month"].min().reset_index()
        first_purchases.rename(columns={"order_month": "cohort_month"}, inplace=True)

        df = df.merge(first_purchases, on="customer_id", how="left")

        # Calculate period index (months elapsed since acquisition)
        def get_period_diff(row):
            return (row["order_month"].year - row["cohort_month"].year) * 12 + (
                row["order_month"].month - row["cohort_month"].month
            )

        df["cohort_period"] = (
            (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
            + (df["order_month"].dt.month - df["cohort_month"].dt.month)
        )

        # Aggregate at (cohort_month, cohort_period)
        grouped = df.groupby(["cohort_month", "cohort_period"])
        cohort_summary = grouped.agg(
            active_customers=("customer_id", "nunique"),
            total_orders=("invoice_id", "nunique"),
            total_revenue=("revenue", "sum"),
        ).reset_index()

        # Compute cohort size (active customers at period 0)
        cohort_sizes = (
            cohort_summary[cohort_summary["cohort_period"] == 0]
            .set_index("cohort_month")["active_customers"]
            .to_dict()
        )
        cohort_summary["cohort_size"] = cohort_summary["cohort_month"].map(cohort_sizes)
        cohort_summary["retention_rate"] = (
            cohort_summary["active_customers"] / cohort_summary["cohort_size"]
        ).round(4)
        cohort_summary["revenue_per_customer"] = (
            cohort_summary["total_revenue"] / cohort_summary["cohort_size"]
        ).round(2)

        # Cumulative revenue and cumulative CLV per cohort
        cohort_summary["cumulative_revenue"] = cohort_summary.groupby("cohort_month")[
            "total_revenue"
        ].cumsum()
        cohort_summary["cumulative_clv"] = (
            cohort_summary["cumulative_revenue"] / cohort_summary["cohort_size"]
        ).round(2)

        # Format cohort_month to string YYYY-MM
        cohort_summary["cohort_month_str"] = cohort_summary["cohort_month"].astype(str)

        # Build pivot tables
        retention_matrix = cohort_summary.pivot(
            index="cohort_month_str",
            columns="cohort_period",
            values="retention_rate",
        )

        revenue_matrix = cohort_summary.pivot(
            index="cohort_month_str",
            columns="cohort_period",
            values="total_revenue",
        ).fillna(0.0)

        # Cohort stability analysis
        # Compare Period 1 retention across cohorts
        p1_retention = cohort_summary[cohort_summary["cohort_period"] == 1]["retention_rate"]
        stability_metrics = {
            "total_cohorts": int(cohort_summary["cohort_month"].nunique()),
            "avg_cohort_size": round(float(pd.Series(cohort_sizes).mean()), 1),
            "min_cohort_size": int(pd.Series(cohort_sizes).min()),
            "max_cohort_size": int(pd.Series(cohort_sizes).max()),
            "avg_period_1_retention": round(float(p1_retention.mean()), 4) if len(p1_retention) > 0 else 0.0,
            "std_period_1_retention": round(float(p1_retention.std()), 4) if len(p1_retention) > 1 else 0.0,
            "period_1_retention_min": round(float(p1_retention.min()), 4) if len(p1_retention) > 0 else 0.0,
            "period_1_retention_max": round(float(p1_retention.max()), 4) if len(p1_retention) > 0 else 0.0,
        }

        print(f"[CohortAnalysis] Analyzed {stability_metrics['total_cohorts']} acquisition cohorts.")
        print(f"[CohortAnalysis] Average Month 1 Retention: {stability_metrics['avg_period_1_retention']*100:.1f}% "
              f"(std: {stability_metrics['std_period_1_retention']*100:.1f}%)")

        return cohort_summary, retention_matrix, revenue_matrix, stability_metrics

    def save_cohort_artifacts(
        self,
        cohort_summary: pd.DataFrame,
        retention_matrix: pd.DataFrame,
        revenue_matrix: pd.DataFrame,
        stability_metrics: Dict[str, Any],
    ):
        """Save cohort tables and stability metrics."""
        proc_dir = self.root / self.config.paths.processed_dir
        rep_dir = self.root / self.config.paths.reports_dir
        proc_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)

        retention_matrix.to_csv(proc_dir / "cohort_retention_matrix.csv")
        revenue_matrix.to_csv(proc_dir / "cohort_revenue_matrix.csv")
        cohort_summary.to_csv(proc_dir / "cohort_summary.csv", index=False)

        metrics_file = rep_dir / "cohort_stability_report.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(stability_metrics, f, indent=2)

        print(f"[CohortAnalysis] Saved matrices to {proc_dir} and report to {metrics_file}")
