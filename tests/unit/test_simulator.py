"""Unit tests for synthetic campaign generation and scenario simulation."""

import pandas as pd
import pytest
from src.simulation.campaign_simulator import SyntheticCampaignGenerator, ScenarioSimulator


@pytest.fixture
def sample_segmented_customers():
    return pd.DataFrame({
        "customer_id": [f"C_{i}" for i in range(100)],
        "action_segment": ["High Value At Risk"] * 40 + ["Champions / High Value Active"] * 60,
        "clv_expected_90d": [500.0] * 40 + [800.0] * 60,
    })


def test_synthetic_campaign_generator():
    gen = SyntheticCampaignGenerator()
    cust_ids = [f"CUST_{i}" for i in range(50)]
    df = gen.generate_synthetic_history(cust_ids, num_campaigns=3)

    assert "campaign_id" in df.columns
    assert "responded" in df.columns
    assert "channel" in df.columns
    assert "synthetic_flag" in df.columns
    assert (df["synthetic_flag"] == True).all()


def test_scenario_simulator(sample_segmented_customers):
    sim = ScenarioSimulator()
    res = sim.simulate_campaign(
        segment_df=sample_segmented_customers,
        target_segment="High Value At Risk",
        action="RETENTION",
        campaign_size=30,
        channel="sms",
        base_response_rate=0.20,
        avg_incremental_aov=70.0,
        discount_pct=0.10,
        fixed_creative_cost=100.0,
    )

    assert res["target_segment"] == "High Value At Risk"
    assert res["targeted_customers"] == 30
    assert res["expected_responders"] == 6  # 30 * 0.20
    assert res["gross_incremental_revenue_gbp"] == 420.0  # 6 * 70.0
    assert res["total_campaign_cost_gbp"] > 0
    assert len(res["net_value_80pct_range"]) == 2
    assert res["net_value_80pct_range"][1] >= res["net_value_80pct_range"][0]
