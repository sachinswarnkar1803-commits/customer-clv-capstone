"""Next-Best-Action (NBA) decision engine module.

Prescribes tailored marketing interventions, communication channels, incentives,
and contact costs for every individual customer based on probabilistic CLV and risk metrics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config


class NBAEngine:
    """Evaluates customer state and assigns prescriptive Next-Best-Action recommendations."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.channel_costs = {
            channel: info["cost_per_contact"]
            for channel, info in self.config.nba.channels.items()
        }

    def generate_recommendations(self, df_segmented: pd.DataFrame) -> pd.DataFrame:
        """Enrich customer table with prescriptive NBA attributes."""
        df = df_segmented.copy()
        def_horizon = self.config.clv.default_horizon_days
        clv_col = f"clv_expected_{def_horizon}d"

        recommendations = []
        for _, row in df.iterrows():
            cust_id = row["customer_id"]
            segment = row["action_segment"]
            clv = row[clv_col]
            p_alive = row.get("p_alive", 0.5)
            inact_p = row.get("inactivity_prob_90d", 0.5)
            spread = row.get("clv_uncertainty_spread", 1.5)

            # Confidence based on posterior spread
            if spread < 1.0:
                confidence = "HIGH"
            elif spread < 2.5:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # Risk level
            if inact_p >= self.config.segments.high_risk_inactivity_prob or p_alive < 0.40:
                risk_level = "HIGH"
            elif inact_p >= self.config.segments.medium_risk_inactivity_prob:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # Decision Logic per Segment
            if segment == "Champions / High Value Active":
                action = "LOYALTY_REWARD"
                priority = "HIGH"
                channel = "direct_mail" if clv > 500 else "email"
                offer = "Exclusive VIP Catalog Preview & Handwritten Thank You"
                reason = f"Top-tier CLV (£{clv:.2f}) with active engagement (P_Alive={p_alive:.2f}). Nurture brand advocacy."
                exp_uplift_pct = 0.08

            elif segment == "High Value At Risk":
                action = "RETENTION"
                priority = "HIGH"
                channel = "sms" if clv > 300 else "email"
                offer = "15% Welcome-Back Voucher on orders over £75"
                reason = f"High future value (£{clv:.2f}) at severe risk of lapse (Inactivity={inact_p:.1%}). Urgent intervention required."
                exp_uplift_pct = 0.18

            elif segment == "Growing Customer":
                action = "UPSELL"
                priority = "MEDIUM"
                channel = "email"
                offer = "Tiered Discount: Spend £60 Get £10 Off Next Basket"
                reason = f"Rising order cadence (frequency={row['frequency']}) with healthy engagement. Promote larger basket sizes."
                exp_uplift_pct = 0.12

            elif segment == "Loyal Mid Value":
                action = "CROSS_SELL"
                priority = "MEDIUM"
                channel = "email"
                offer = "Complementary Product Recommendation with Free Shipping"
                reason = "Consistent repeat order history. Expand category exploration across the product catalog."
                exp_uplift_pct = 0.10

            elif segment == "New Customer":
                action = "LOW_COST_ENGAGEMENT"
                priority = "MEDIUM"
                channel = "email"
                offer = "Onboarding Welcome Series & 10% First Repeat Discount"
                reason = f"Recent acquisition (Tenure={row['customer_tenure_days']:.0f}d). Establish repeat purchase cadence."
                exp_uplift_pct = 0.15

            elif segment == "Reactivation Candidate":
                action = "REACTIVATION"
                priority = "MEDIUM"
                channel = "email"
                offer = "Comeback Gift: £10 Gift Card on Orders Over £50"
                reason = f"Substantial past revenue (£{row['total_revenue']:.2f}) now dormant. Salvageable customer relationship."
                exp_uplift_pct = 0.10

            elif segment == "Low Value Active":
                action = "LOW_COST_ENGAGEMENT"
                priority = "LOW"
                channel = "push" if "push" in self.channel_costs else "email"
                offer = "Seasonal Trend Highlights & Promotional Clearance Alerts"
                reason = "Active engagement with modest spend. Maintain brand presence with near-zero marginal cost."
                exp_uplift_pct = 0.05

            else:  # Dormant
                action = "NO_ACTION"
                priority = "LOW"
                channel = "email"
                offer = "None (Suppressed)"
                reason = f"Low expected return (£{clv:.2f}) coupled with high dormancy. Paid acquisition or outreach is economically unviable."
                exp_uplift_pct = 0.0

            contact_cost = self.channel_costs.get(channel, 0.05)
            expected_incremental_val = round(clv * exp_uplift_pct, 2)

            recommendations.append({
                "nba_recommended_action": action,
                "nba_priority": priority,
                "nba_channel": channel,
                "nba_offer": offer,
                "nba_reason": reason,
                "nba_risk_level": risk_level,
                "nba_confidence": confidence,
                "nba_estimated_cost": contact_cost,
                "nba_expected_incremental_value": expected_incremental_val,
            })

        nba_df = pd.DataFrame(recommendations, index=df.index)
        combined_df = pd.concat([df, nba_df], axis=1)

        print(f"[NBAEngine] Processed recommendations for {len(combined_df):,} customers.")
        print("[NBAEngine] Action Distribution:")
        for action_name, count in combined_df["nba_recommended_action"].value_counts().items():
            print(f"  - {action_name:22s}: {count:5,d}")

        return combined_df

    def save_results(self, df_nba: pd.DataFrame):
        """Save complete customer CLV and NBA table to processed directory."""
        proc_dir = self.root / self.config.paths.processed_dir
        proc_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = self.root / self.config.paths.clv_segments
        csv_path = self.root / self.config.paths.clv_segments_csv

        # Convert object columns to clean strings for parquet
        df_save = df_nba.copy()
        for col in df_save.select_dtypes(include="object").columns:
            df_save[col] = df_save[col].astype(str)

        try:
            df_save.to_parquet(parquet_path, index=False)
            print(f"[NBAEngine] Saved parquet to {parquet_path}")
        except Exception as e:
            print(f"[NBAEngine] Parquet save note: {e}")

        df_save.to_csv(csv_path, index=False)
        print(f"[NBAEngine] Saved CSV to {csv_path}")
