"""Script to generate all 12 educational and reproducible capstone Jupyter Notebooks."""

import json
from pathlib import Path


def create_notebook(title: str, description: str, code_cells: list) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title}\n",
                f"**Project**: BDS-34 | T.Y. B.Sc. Data Science - Semester V\n\n",
                f"{description}\n"
            ]
        }
    ]
    for code in code_cells:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def main():
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        (
            "01_data_exploration.ipynb",
            "01. Raw Data Exploration & Provenance",
            "Explore the raw UCI Online Retail II dataset, inspect schemas, and verify dataset provenance.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.config.config import load_config
from src.data.ingest import DataIngestion

config = load_config()
ingest = DataIngestion(config)
df_raw = ingest.load_raw_data(nrows=5000)
print('Loaded raw records sample:', len(df_raw))
df_raw.head()""",
                """# Inspect data types and summary statistics
df_raw.info()
df_raw.describe()"""
            ]
        ),
        (
            "02_data_quality.ipynb",
            "02. Data Quality & Preprocessing Pipeline",
            "Validate schema, flag cancellations, returns, missing customer IDs, and review the non-silent audit trail.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.validation.validator import DataValidator, run_validation_pipeline

clean_df, audit_log = run_validation_pipeline(use_cache=True)
print('Clean records:', len(clean_df))
print('Retention rate:', audit_log['retention_rate_pct'], '%')""",
                """# Inspect the audit trail steps
pd.DataFrame(audit_log['steps'])"""
            ]
        ),
        (
            "03_cohort_analysis.ipynb",
            "03. Cohort & Retention Dynamics Analysis",
            "Group customers by acquisition month and calculate monthly active retention matrices and revenue curves.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.cohort.cohort_analysis import CohortAnalyzer

clean_df = pd.read_parquet('../data/processed/clean_transactions.parquet')
analyzer = CohortAnalyzer()
summary, ret_mat, rev_mat, metrics = analyzer.compute_cohorts(clean_df)
print('Analyzed cohorts:', metrics['total_cohorts'])
ret_mat.head(10)"""
            ]
        ),
        (
            "04_rfm_baseline.ipynb",
            "04. Customer Features & Traditional RFM Baseline",
            "Derive customer-level features respecting temporal cutoff causality and calculate 1-5 quintile RFM scores.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.features.splitter import TemporalSplitter
from src.features.feature_builder import CustomerFeatureBuilder

clean_df = pd.read_parquet('../data/processed/clean_transactions.parquet')
splitter = TemporalSplitter()
train_df, holdout_df = splitter.split_train_holdout(clean_df)

builder = CustomerFeatureBuilder()
features_df = builder.build_customer_features(train_df)
print('Customer features:', len(features_df))
features_df[['customer_id', 'frequency', 'recency_days', 'customer_tenure_days', 'r_score', 'f_score', 'm_score', 'rfm_segment']].head()"""
            ]
        ),
        (
            "05_purchase_model.ipynb",
            "05. BG/NBD Probabilistic Repeat-Purchase Model",
            "Fit the BG/NBD model on frequency, recency, and tenure to estimate P(Alive) and expected future transactions.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.models.purchase_model import PurchaseModelBGNBD

features_df = pd.read_parquet('../data/processed/customer_features.parquet')
bgf = PurchaseModelBGNBD().fit(features_df)
p_alive = bgf.predict_p_alive(features_df)
exp_purch = bgf.predict_expected_purchases(features_df, horizon_days=90)
print('Average P(Alive):', p_alive.mean())
print('Average Expected 90d Purchases:', exp_purch.mean())"""
            ]
        ),
        (
            "06_monetary_model.ipynb",
            "06. Gamma-Gamma Probabilistic Monetary Value Model",
            "Verify the frequency-spend independence assumption and fit the Gamma-Gamma sub-model.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.models.monetary_model import MonetaryModelGammaGamma

features_df = pd.read_parquet('../data/processed/customer_features.parquet')
ggf = MonetaryModelGammaGamma()
check = ggf.check_assumptions(features_df)
print('Assumption Check:', check)
ggf.fit(features_df)
exp_spend = ggf.predict_expected_average_spend(features_df)
print('Average Expected Order Value: £', exp_spend.mean())"""
            ]
        ),
        (
            "07_inactivity_model.ipynb",
            "07. Customer Inactivity Hazard & Survival Modeling",
            "Fit Kaplan-Meier and parametric Weibull survival curves to compute 30, 60, and 90-day inactivity hazards.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.models.inactivity_model import InactivitySurvivalModel

features_df = pd.read_parquet('../data/processed/customer_features.parquet')
surv = InactivitySurvivalModel().fit(features_df)
inact_probs = surv.predict_inactivity_probabilities(features_df)
inact_probs.head()"""
            ]
        ),
        (
            "08_clv_model.ipynb",
            "08. Probabilistic CLV & 80% Uncertainty Intervals",
            "Combine transaction and spend expectations with monthly discounting and Monte Carlo predictive simulation.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.models.purchase_model import PurchaseModelBGNBD
from src.models.monetary_model import MonetaryModelGammaGamma
from src.clv.clv_calculator import ProbabilisticCLVCalculator

features_df = pd.read_parquet('../data/processed/customer_features.parquet')
bgf = PurchaseModelBGNBD()
bgf.load_model()
ggf = MonetaryModelGammaGamma()
ggf.load_model()

clv_calc = ProbabilisticCLVCalculator()
clv_df = clv_calc.compute_clv(features_df, bgf, ggf, n_simulation_samples=30)
clv_df[['customer_id', 'clv_expected_90d', 'clv_lower_80pct', 'clv_upper_80pct', 'clv_uncertainty_spread']].head()"""
            ]
        ),
        (
            "09_segmentation.ipynb",
            "09. Action-Oriented Customer Segmentation",
            "Segment customers into Champions, High Value At Risk, Growing, Loyal, New, and Dormant groups.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.segmentation.segmenter import ActionSegmenter

clv_df = pd.read_parquet('../data/processed/customer_clv_segments.parquet')
segmenter = ActionSegmenter()
print('Segment Definitions:')
for name, meta in segmenter.get_segment_definitions().items():
    print(f"- {name}: {meta['description']}")"""
            ]
        ),
        (
            "10_nba.ipynb",
            "10. Next-Best-Action (NBA) Prescriptive Engine",
            "Prescribe personalized actions, channels, incentives, and contact costs for every customer.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.nba.nba_engine import NBAEngine

clv_df = pd.read_parquet('../data/processed/customer_clv_segments.parquet')
nba = NBAEngine()
clv_df[['customer_id', 'action_segment', 'nba_recommended_action', 'nba_priority', 'nba_channel', 'nba_offer']].head(10)"""
            ]
        ),
        (
            "11_scenario_simulation.ipynb",
            "11. Synthetic Campaign Scenario Simulator",
            "Model campaign ROI, contact costs, and Monte Carlo uncertainty intervals for marketing interventions.",
            [
                """import sys
sys.path.insert(0, '..')
import pandas as pd
from src.simulation.campaign_simulator import ScenarioSimulator

clv_df = pd.read_parquet('../data/processed/customer_clv_segments.parquet')
sim = ScenarioSimulator()
result = sim.simulate_campaign(
    segment_df=clv_df,
    target_segment='Champions / High Value Active',
    action='LOYALTY_REWARD',
    campaign_size=500,
    channel='email',
    base_response_rate=0.20,
    avg_incremental_aov=75.0,
    discount_pct=0.08,
)
print('Simulated Campaign Result:')
for k, v in result.items():
    print(f"  {k}: {v}")"""
            ]
        ),
        (
            "12_evaluation.ipynb",
            "12. Model Evaluation, Benchmarking & Monitoring",
            "Benchmark Probabilistic CLV against baselines, measure interval coverage, calibration, and PSI drift.",
            [
                """import sys
sys.path.insert(0, '..')
import json
import pandas as pd

with open('../reports/model_evaluation_report.json', 'r') as f:
    eval_rep = json.load(f)

print('Holdout Revenue Benchmarking:')
pd.DataFrame(eval_rep['benchmark_results'])""",
                """print('Interval Coverage:')
print(eval_rep['interval_coverage'])
print('\\nSparse-History Sensitivity:')
pd.DataFrame(eval_rep['sparse_history_sensitivity'])"""
            ]
        ),
    ]

    for fname, title, desc, cells in notebooks:
        nb_json = create_notebook(title, desc, cells)
        nb_path = nb_dir / fname
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=2)
        print(f"Generated notebook: {fname}")


if __name__ == "__main__":
    main()
