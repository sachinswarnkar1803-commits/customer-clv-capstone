"""Synthetic campaign data generator and interactive scenario simulator module.

NOTE: All campaign response data is explicitly SYNTHETIC. The UCI Online Retail II dataset
contains only transactional records. This simulation layer models 'what-if' marketing campaigns
and ROI projections without conflating assumptions with observed historical reality.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config


class SyntheticCampaignGenerator:
    """Generates synthetic historical campaign interaction records for methodology demonstration."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.synth_dir = self.root / self.config.paths.synthetic_dir
        self.synth_dir.mkdir(parents=True, exist_ok=True)

    def generate_synthetic_history(
        self,
        customer_ids: List[str],
        num_campaigns: int = 5,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Create a synthetic campaign response log for historical experimentation."""
        rng = np.random.default_rng(seed=random_seed)
        records = []
        channels = ["email", "sms", "direct_mail", "push"]
        actions = ["RETENTION", "UPSELL", "WIN_BACK", "LOYALTY_REWARD"]
        offers = ["10% Discount", "Free Shipping", "£15 Off £60", "VIP Gift"]

        campaign_dates = [
            "2010-03-15", "2010-06-20", "2010-09-10", "2010-11-25", "2011-04-12"
        ]

        for camp_idx in range(num_campaigns):
            camp_id = f"CAMP_{camp_idx + 1:03d}"
            camp_date = campaign_dates[camp_idx % len(campaign_dates)]
            channel = rng.choice(channels, p=[0.55, 0.25, 0.10, 0.10])
            action = rng.choice(actions)
            offer = rng.choice(offers)
            discount = rng.choice([0.05, 0.10, 0.15, 0.20])

            # Sample 20% to 50% of customers to be targeted
            sample_size = int(len(customer_ids) * rng.uniform(0.20, 0.50))
            targeted_custs = rng.choice(customer_ids, size=sample_size, replace=False)

            contact_cost = self.config.nba.channels.get(channel, {}).get("cost_per_contact", 0.05)

            for cid in targeted_custs:
                # Synthetic response probability logistic model
                base_logit = -2.2 + 2.5 * discount
                prob_resp = 1.0 / (1.0 + np.exp(-base_logit))
                responded = int(rng.binomial(1, prob_resp))

                records.append({
                    "synthetic_flag": True,  # Explicit marker
                    "campaign_id": camp_id,
                    "customer_id": cid,
                    "campaign_date": camp_date,
                    "channel": channel,
                    "action": action,
                    "offer_type": offer,
                    "discount": discount,
                    "responded": responded,
                    "campaign_cost": contact_cost,
                })

        synth_df = pd.DataFrame(records)
        target_path = self.synth_dir / "synthetic_campaign_history.csv"
        synth_df.to_csv(target_path, index=False)
        print(f"[Simulation] Generated {len(synth_df):,} synthetic campaign records -> {target_path}")
        return synth_df


class ScenarioSimulator:
    """Simulates expected financial return, ROI, and uncertainty for marketing campaigns."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()

    def simulate_campaign(
        self,
        segment_df: pd.DataFrame,
        target_segment: str,
        action: str,
        campaign_size: int,
        channel: str = "email",
        base_response_rate: float = 0.12,
        avg_incremental_aov: float = 45.0,
        discount_pct: float = 0.10,
        fixed_creative_cost: float = 250.0,
        n_mc_simulations: int = 500,
    ) -> Dict[str, Any]:
        """Simulate single campaign scenario with Monte Carlo uncertainty bounds.

        Returns dictionary of scenario financial outcomes.
        """
        # Filter customers in target segment
        seg_mask = segment_df["action_segment"] == target_segment
        available_customers = int(seg_mask.sum())

        if available_customers == 0:
            target_customers = 0
            mean_clv = 0.0
        else:
            target_customers = min(campaign_size, available_customers)
            def_horizon = self.config.clv.default_horizon_days
            clv_col = f"clv_expected_{def_horizon}d"
            target_slice = segment_df[seg_mask].sort_values(clv_col, ascending=False).head(target_customers)
            mean_clv = float(target_slice[clv_col].mean())

        cost_per_contact = self.config.nba.channels.get(channel, {}).get("cost_per_contact", 0.05)
        channel_contact_cost = target_customers * cost_per_contact

        # Expected baseline outcomes
        exp_responders = int(round(target_customers * base_response_rate))
        gross_incremental_rev = exp_responders * avg_incremental_aov
        discount_cost = gross_incremental_rev * discount_pct
        total_campaign_cost = fixed_creative_cost + channel_contact_cost + discount_cost
        net_value = gross_incremental_rev - total_campaign_cost
        roi_pct = (net_value / max(1.0, total_campaign_cost)) * 100.0

        # Monte Carlo Uncertainty Simulation
        rng = np.random.default_rng(seed=self.config.project.random_seed)
        sim_responders = rng.binomial(n=target_customers, p=base_response_rate, size=n_mc_simulations)
        sim_aov = rng.normal(loc=avg_incremental_aov, scale=avg_incremental_aov * 0.20, size=n_mc_simulations)
        sim_aov = np.clip(sim_aov, a_min=5.0, a_max=None)

        sim_gross_rev = sim_responders * sim_aov
        sim_discount = sim_gross_rev * discount_pct
        sim_costs = fixed_creative_cost + channel_contact_cost + sim_discount
        sim_net_vals = sim_gross_rev - sim_costs

        net_val_lower_80 = float(np.percentile(sim_net_vals, 10))
        net_val_upper_80 = float(np.percentile(sim_net_vals, 90))

        return {
            "scenario_name": f"{action} on {target_segment}",
            "target_segment": target_segment,
            "action": action,
            "channel": channel,
            "available_customers_in_segment": available_customers,
            "targeted_customers": target_customers,
            "cost_per_contact_gbp": cost_per_contact,
            "expected_response_rate": base_response_rate,
            "expected_responders": exp_responders,
            "responders_80pct_range": [int(np.percentile(sim_responders, 10)), int(np.percentile(sim_responders, 90))],
            "gross_incremental_revenue_gbp": round(float(gross_incremental_rev), 2),
            "channel_cost_gbp": round(float(channel_contact_cost), 2),
            "discount_cost_gbp": round(float(discount_cost), 2),
            "fixed_cost_gbp": round(float(fixed_creative_cost), 2),
            "total_campaign_cost_gbp": round(float(total_campaign_cost), 2),
            "expected_net_value_gbp": round(float(net_value), 2),
            "net_value_80pct_range": [round(net_val_lower_80, 2), round(net_val_upper_80, 2)],
            "roi_percent": round(float(roi_pct), 2),
            "avg_target_customer_clv": round(float(mean_clv), 2),
        }

    def compare_scenarios(
        self,
        segment_df: pd.DataFrame,
        scenarios: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Compare multiple marketing campaign scenarios side by side."""
        results = []
        for sc in scenarios:
            res = self.simulate_campaign(segment_df=segment_df, **sc)
            results.append(res)
        return pd.DataFrame(results)
