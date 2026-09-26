"""Action-oriented customer segmentation module.

Translates probabilistic model outputs (CLV, P(Alive), Inactivity Hazard, Tenure)
into distinct, mutually exclusive business segments with transparent criteria.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config


class ActionSegmenter:
    """Assigns reproducible, model-informed action segments to customers."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()

    def get_segment_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Return full metadata, rationale, and business criteria for each segment."""
        return {
            "Champions / High Value Active": {
                "description": "Top-tier CLV customers actively transacting with low churn risk.",
                "objective": "Reward loyalty, provide VIP treatment, and cross-sell premium catalog items.",
                "default_action": "LOYALTY_REWARD",
                "priority": "HIGH",
            },
            "High Value At Risk": {
                "description": "High lifetime value customers displaying elevated inactivity hazard.",
                "objective": "Intervene immediately with dedicated retention incentives before permanent churn.",
                "default_action": "RETENTION",
                "priority": "HIGH",
            },
            "Growing Customer": {
                "description": "Active customers with healthy repeat purchase cadence and upward potential.",
                "objective": "Nurture order frequency and upsell larger basket bundles.",
                "default_action": "UPSELL",
                "priority": "MEDIUM",
            },
            "Loyal Mid Value": {
                "description": "Consistent buyers with moderate basket spend and steady engagement.",
                "objective": "Maintain engagement cadence and introduce complementary categories.",
                "default_action": "CROSS_SELL",
                "priority": "MEDIUM",
            },
            "New Customer": {
                "description": "Recently acquired customers with tenure <= 60 days.",
                "objective": "Welcome, nurture onboarding experience, and incentivize 2nd purchase.",
                "default_action": "LOW_COST_ENGAGEMENT",
                "priority": "MEDIUM",
            },
            "Reactivation Candidate": {
                "description": "Lapsed customers with strong historical revenue who can potentially be salvaged.",
                "objective": "Re-engage via personalized comeback offer or product updates.",
                "default_action": "REACTIVATION",
                "priority": "MEDIUM",
            },
            "Low Value Active": {
                "description": "Currently active buyers with modest ticket sizes and low expected CLV.",
                "objective": "Maintain low-cost automated engagement without high campaign expenditure.",
                "default_action": "LOW_COST_ENGAGEMENT",
                "priority": "LOW",
            },
            "Dormant": {
                "description": "Low-value customers with very high inactivity risk and minimal expected return.",
                "objective": "Suppress high-cost outreach; preserve campaign marketing budget.",
                "default_action": "NO_ACTION",
                "priority": "LOW",
            },
        }

    def segment_customers(self, df_clv: pd.DataFrame) -> pd.DataFrame:
        """Assign segments based on probabilistic parameters and behavioral logic."""
        df = df_clv.copy()
        def_horizon = self.config.clv.default_horizon_days
        clv_col = f"clv_expected_{def_horizon}d"

        # Determine dynamic percentile thresholds
        high_clv_thresh = float(df[clv_col].quantile(self.config.segments.high_value_clv_percentile))
        low_clv_thresh = float(df[clv_col].quantile(self.config.segments.low_value_clv_percentile))
        high_risk_prob = self.config.segments.high_risk_inactivity_prob
        med_risk_prob = self.config.segments.medium_risk_inactivity_prob

        median_rev = float(df["total_revenue"].median())

        print(f"[Segmenter] Dynamic Thresholds: High CLV (P75): £{high_clv_thresh:.2f} | "
              f"Low CLV (P25): £{low_clv_thresh:.2f} | High Risk Prob: {high_risk_prob}")

        def assign_segment(row) -> str:
            clv = row[clv_col]
            inact_p = row.get("inactivity_prob_90d", 0.5)
            p_alive = row.get("p_alive", 0.5)
            tenure = row["customer_tenure_days"]
            freq = row["frequency"]
            tot_rev = row["total_revenue"]
            days_since = row["days_since_last_purchase"]

            # 1. New Customer
            if tenure <= 60.0 and freq <= 1:
                return "New Customer"

            # 2. Champions / High Value Active
            if clv >= high_clv_thresh and inact_p < high_risk_prob and p_alive >= 0.50:
                return "Champions / High Value Active"

            # 3. High Value At Risk
            if clv >= high_clv_thresh and (inact_p >= high_risk_prob or p_alive < 0.50):
                return "High Value At Risk"

            # 4. Growing Customer
            if low_clv_thresh <= clv < high_clv_thresh and freq >= 2 and inact_p < med_risk_prob:
                return "Growing Customer"

            # 5. Loyal Mid Value
            if low_clv_thresh <= clv < high_clv_thresh and freq >= 1 and inact_p < high_risk_prob:
                return "Loyal Mid Value"

            # 6. Reactivation Candidate
            if tot_rev >= median_rev and (inact_p >= high_risk_prob or p_alive < 0.35) and days_since <= 300:
                return "Reactivation Candidate"

            # 7. Low Value Active
            if clv < low_clv_thresh and inact_p < high_risk_prob:
                return "Low Value Active"

            # 8. Dormant
            return "Dormant"

        df["action_segment"] = df.apply(assign_segment, axis=1)

        # Log segment population distribution
        seg_counts = df["action_segment"].value_counts()
        print("[Segmenter] Segment Distribution:")
        for seg_name, cnt in seg_counts.items():
            pct = (cnt / len(df)) * 100
            print(f"  - {seg_name:30s}: {cnt:5,d} ({pct:5.1f}%)")

        return df
