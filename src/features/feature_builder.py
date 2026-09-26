"""Customer-level feature engineering and RFM baseline computation module.

Extracts customer behavioral attributes, lifetime parameters (x, t_x, T),
and traditional RFM segmentation respecting strict temporal causality.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config


class CustomerFeatureBuilder:
    """Builds customer-level feature matrices and RFM baselines."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()

    def build_customer_features(
        self,
        df_transactions: pd.DataFrame,
        observation_cutoff: Optional[str] = None,
    ) -> pd.DataFrame:
        """Derive customer-level features strictly using transactions prior to cutoff.

        Args:
            df_transactions: Clean transactions DataFrame.
            observation_cutoff: Date string (YYYY-MM-DD). If None, maximum date in df is used.

        Returns:
            pd.DataFrame: Customer-level feature table.
        """
        df = df_transactions.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["transaction_date"]):
            df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        # Determine cutoff date
        if observation_cutoff:
            cutoff = pd.to_datetime(observation_cutoff)
            df = df[df["transaction_date"] <= cutoff].copy()
        else:
            cutoff = df["transaction_date"].max()

        print(f"[FeatureBuilder] Building customer features as of cutoff: {cutoff.strftime('%Y-%m-%d')}")
        print(f"[FeatureBuilder] Total transactions analyzed: {len(df):,}")

        # Group by customer and invoice to get order-level totals first
        orders = (
            df.groupby(["customer_id", "invoice_id"], as_index=False)
            .agg(
                order_date=("transaction_date", "min"),
                order_revenue=("revenue", "sum"),
                order_items=("quantity", "sum"),
                order_unique_skus=("stock_code", "nunique"),
                country=("country", "first"),
            )
        )

        # Truncate order_date to calendar date for day-level cadence
        orders["order_day"] = orders["order_date"].dt.normalize()

        # Customer-level aggregations
        customers = []
        for cust_id, cust_orders in orders.groupby("customer_id"):
            sorted_orders = cust_orders.sort_values("order_date")
            first_date = sorted_orders["order_date"].min()
            last_date = sorted_orders["order_date"].max()

            total_revenue = sorted_orders["order_revenue"].sum()
            total_orders = len(sorted_orders)
            unique_order_days = sorted_orders["order_day"].nunique()

            # Lifetimes model conventions:
            # frequency (x): repeat purchase days (unique order days - 1)
            frequency = max(0, unique_order_days - 1)

            # tenure (T): days between first purchase and observation cutoff
            tenure_days = max(1.0, (cutoff - first_date).total_seconds() / 86400.0)

            # recency (t_x): days between first and last purchase
            recency_days = max(0.0, (last_date - first_date).total_seconds() / 86400.0)

            # days since last purchase (inactivity measure)
            days_since_last = max(0.0, (cutoff - last_date).total_seconds() / 86400.0)

            # monetary_value: average order value across repeat purchases
            if frequency > 0:
                # Exclude the very first order day for repeat basket spend
                first_day = sorted_orders["order_day"].min()
                repeat_orders = sorted_orders[sorted_orders["order_day"] > first_day]
                if len(repeat_orders) > 0:
                    repeat_monetary = repeat_orders["order_revenue"].mean()
                else:
                    repeat_monetary = sorted_orders["order_revenue"].mean()
            else:
                repeat_monetary = 0.0

            customers.append({
                "customer_id": cust_id,
                "first_purchase_date": first_date,
                "last_purchase_date": last_date,
                "customer_tenure_days": round(tenure_days, 2),
                "recency_days": round(recency_days, 2),
                "days_since_last_purchase": round(days_since_last, 2),
                "frequency": int(frequency),
                "total_orders": int(total_orders),
                "monetary_value": round(float(repeat_monetary), 2),
                "total_revenue": round(float(total_revenue), 2),
                "average_order_value": round(float(total_revenue / max(1, total_orders)), 2),
                "cohort_month": first_date.strftime("%Y-%m"),
                "country": sorted_orders["country"].iloc[-1],
            })

        features_df = pd.DataFrame(customers)

        # Merge product variety and active months
        catalog_stats = (
            df.groupby("customer_id")
            .agg(
                unique_products=("stock_code", "nunique"),
                active_months=("invoice_month", "nunique"),
            )
            .reset_index()
        )
        features_df = features_df.merge(catalog_stats, on="customer_id", how="left")

        # Purchase frequency per 90 days tenure
        features_df["purchase_frequency"] = (
            features_df["total_orders"] / (features_df["customer_tenure_days"] / 30.0)
        ).round(3)

        # 2. Traditional RFM Quantile Scoring Baseline
        features_df = self.calculate_rfm_baseline(features_df)

        print(f"[FeatureBuilder] Generated features for {len(features_df):,} customers.")
        return features_df

    def calculate_rfm_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute transparent 1-5 quintile RFM scores and rule-based RFM baseline segments."""
        df = df.copy()

        # Recency score: Lower days since last purchase -> higher score (5)
        # Using qcut with duplicate handling
        try:
            df["r_score"] = pd.qcut(
                df["days_since_last_purchase"],
                q=5,
                labels=[5, 4, 3, 2, 1],
                duplicates="drop",
            ).astype(int)
        except Exception:
            df["r_score"] = pd.cut(
                df["days_since_last_purchase"].rank(method="first"),
                bins=5,
                labels=[5, 4, 3, 2, 1],
            ).astype(int)

        # Frequency score: Higher frequency -> higher score (5)
        # Rank-based binning to handle heavy tie zeros/ones
        df["f_score"] = pd.cut(
            df["frequency"].rank(method="first"),
            bins=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

        # Monetary score: Higher total revenue -> higher score (5)
        df["m_score"] = pd.cut(
            df["total_revenue"].rank(method="first"),
            bins=5,
            labels=[1, 2, 3, 4, 5],
        ).astype(int)

        df["rfm_composite"] = (
            df["r_score"].astype(str) + df["f_score"].astype(str) + df["m_score"].astype(str)
        )

        # Heuristic RFM segmentation
        def segment_rfm(row):
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 4 and f <= 2:
                return "Recent / New Customers"
            elif r <= 2 and f >= 3 and m >= 3:
                return "Can't Lose Them"
            elif r <= 2 and f >= 2:
                return "At Risk"
            elif r >= 3 and f <= 2:
                return "Promising / Potential"
            elif r <= 2 and f <= 2 and m >= 3:
                return "Hibernating"
            else:
                return "Lost / Dormant"

        df["rfm_segment"] = df.apply(segment_rfm, axis=1)
        return df

    def save_features(self, df_features: pd.DataFrame, filename_suffix: str = ""):
        """Save feature table to processed parquet and CSV."""
        proc_dir = self.root / self.config.paths.processed_dir
        proc_dir.mkdir(parents=True, exist_ok=True)

        pq_name = f"customer_features{filename_suffix}.parquet"
        csv_name = f"customer_features{filename_suffix}.csv"

        df_features.to_parquet(proc_dir / pq_name, index=False)
        df_features.to_csv(proc_dir / csv_name, index=False)
        print(f"[FeatureBuilder] Saved features to {proc_dir / pq_name} and {proc_dir / csv_name}")
