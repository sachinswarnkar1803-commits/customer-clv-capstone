"""Master End-to-End Capstone Pipeline Orchestrator.

Executes the complete industry-grade workflow:
DATA -> CLEANING -> FEATURES -> COHORTS -> PURCHASE MODEL -> MONETARY MODEL
-> INACTIVITY SURVIVAL -> PROBABILISTIC CLV & INTERVALS -> ACTION SEGMENTS
-> NEXT-BEST-ACTION -> SYNTHETIC CAMPAIGN SIMULATION -> HOLDOUT EVALUATION -> MONITORING
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Suppress harmless numpy RuntimeWarnings from internal computations
warnings.filterwarnings("ignore", category=RuntimeWarning)

from src.config.config import AppConfig, get_project_root, load_config
from src.validation.validator import DataValidator
from src.features.splitter import TemporalSplitter
from src.features.feature_builder import CustomerFeatureBuilder
from src.cohort.cohort_analysis import CohortAnalyzer
from src.models.baseline import HistoricalAverageBaseline, RFMBaseline
from src.models.purchase_model import PurchaseModelBGNBD
from src.models.monetary_model import MonetaryModelGammaGamma
from src.models.inactivity_model import InactivitySurvivalModel
from src.clv.clv_calculator import ProbabilisticCLVCalculator
from src.segmentation.segmenter import ActionSegmenter
from src.nba.nba_engine import NBAEngine
from src.simulation.campaign_simulator import SyntheticCampaignGenerator, ScenarioSimulator
from src.evaluation.evaluator import ModelEvaluator
from src.monitoring.monitor import SystemMonitor


def run_full_pipeline(config_path: Optional[str] = None) -> bool:
    """Execute complete end-to-end customer intelligence and CLV pipeline."""
    start_time = time.time()
    config = load_config(config_path)
    root = get_project_root()

    print("=" * 80)
    print(f"STARTING BDS-34 CAPSTONE END-TO-END PIPELINE: {config.project.name}")
    print(f"Project Code: {config.project.code} | Version: {config.project.version}")
    print("=" * 80)

    # ---------------------------------------------------------
    # STEP 1: LOAD CLEAN TRANSACTIONS (or validate if missing)
    # ---------------------------------------------------------
    clean_pq = root / config.paths.clean_transactions
    if clean_pq.exists():
        print(f"\n[Step 1] Loading pre-validated clean transactions: {clean_pq}")
        clean_df = pd.read_parquet(clean_pq)
    else:
        print(f"\n[Step 1] Clean parquet not found. Running validation pipeline...")
        from src.validation.validator import run_validation_pipeline
        clean_df, _ = run_validation_pipeline(use_cache=True)

    print(f"[Step 1] Loaded {len(clean_df):,} clean transaction records.")

    # ---------------------------------------------------------
    # STEP 2: TEMPORAL SPLIT (LEAKAGE PREVENTION)
    # ---------------------------------------------------------
    print("\n[Step 2] Executing strict temporal train/holdout partition...")
    splitter = TemporalSplitter(config)
    train_df, holdout_df = splitter.split_train_holdout(
        clean_df,
        cutoff_date=config.temporal_splits.train_end_date,
    )

    # ---------------------------------------------------------
    # STEP 3: CUSTOMER FEATURE STORE & RFM BASELINE
    # ---------------------------------------------------------
    print("\n[Step 3] Deriving customer-level behavioral features and RFM baseline...")
    feature_builder = CustomerFeatureBuilder(config)
    train_features = feature_builder.build_customer_features(
        train_df,
        observation_cutoff=config.temporal_splits.train_end_date,
    )
    feature_builder.save_features(train_features)

    # ---------------------------------------------------------
    # STEP 4: COHORT DYNAMICS & RETENTION MATRIX
    # ---------------------------------------------------------
    print("\n[Step 4] Computing monthly acquisition cohort retention matrices...")
    cohort_analyzer = CohortAnalyzer(config)
    cohort_summary, ret_mat, rev_mat, stability_metrics = cohort_analyzer.compute_cohorts(clean_df)
    cohort_analyzer.save_cohort_artifacts(cohort_summary, ret_mat, rev_mat, stability_metrics)

    # ---------------------------------------------------------
    # STEP 5: FIT PROBABILISTIC PURCHASE MODEL (BG/NBD)
    # ---------------------------------------------------------
    print("\n[Step 5] Fitting BG/NBD probabilistic repeat-purchase model...")
    bgf_model = PurchaseModelBGNBD(config).fit(train_features)
    bgf_model.save_model()

    # ---------------------------------------------------------
    # STEP 6: FIT MONETARY VALUE MODEL (GAMMA-GAMMA)
    # ---------------------------------------------------------
    print("\n[Step 6] Fitting Gamma-Gamma monetary value model...")
    ggf_model = MonetaryModelGammaGamma(config).fit(train_features)
    ggf_model.save_model()

    # ---------------------------------------------------------
    # STEP 7: FIT INACTIVITY SURVIVAL MODEL (LIFELINES)
    # ---------------------------------------------------------
    print("\n[Step 7] Fitting survival analysis inactivity hazard models...")
    surv_model = InactivitySurvivalModel(config).fit(
        train_features,
        transactions_df=train_df,
        observation_cutoff=config.temporal_splits.train_end_date,
    )
    inactivity_preds = surv_model.predict_inactivity_probabilities(train_features)
    surv_model.save_model()

    # ---------------------------------------------------------
    # STEP 8: PROBABILISTIC CLV & MONTE CARLO PREDICTIVE INTERVALS
    # ---------------------------------------------------------
    print("\n[Step 8] Estimating probabilistic CLV across horizons with 80% Monte Carlo prediction intervals...")
    clv_calc = ProbabilisticCLVCalculator(config)
    clv_df = clv_calc.compute_clv(train_features, bgf_model, ggf_model, n_simulation_samples=500)

    # Merge inactivity survival probabilities
    clv_df = pd.concat([clv_df, inactivity_preds], axis=1)

    # ---------------------------------------------------------
    # STEP 9: ACTION-ORIENTED SEGMENTATION
    # ---------------------------------------------------------
    print("\n[Step 9] Classifying customers into action-oriented business segments...")
    segmenter = ActionSegmenter(config)
    segmented_df = segmenter.segment_customers(clv_df)

    # ---------------------------------------------------------
    # STEP 10: NEXT-BEST-ACTION (NBA) DECISION ENGINE
    # ---------------------------------------------------------
    print("\n[Step 10] Prescribing Next-Best-Action recommendations, channels, and offers...")
    nba_engine = NBAEngine(config)
    final_customer_df = nba_engine.generate_recommendations(segmented_df)
    nba_engine.save_results(final_customer_df)

    # ---------------------------------------------------------
    # STEP 11: SYNTHETIC CAMPAIGN & SCENARIO SIMULATION LAYER
    # ---------------------------------------------------------
    print("\n[Step 11] Generating synthetic campaign history and running scenario simulations...")
    synth_gen = SyntheticCampaignGenerator(config)
    synth_gen.generate_synthetic_history(
        customer_ids=final_customer_df["customer_id"].tolist()[:500],
        num_campaigns=5,
    )

    simulator = ScenarioSimulator(config)
    scenarios_to_run = [
        {
            "target_segment": "High Value At Risk",
            "action": "RETENTION",
            "campaign_size": 500,
            "channel": "sms",
            "base_response_rate": 0.16,
            "avg_incremental_aov": 65.0,
            "discount_pct": 0.15,
        },
        {
            "target_segment": "Champions / High Value Active",
            "action": "LOYALTY_REWARD",
            "campaign_size": 500,
            "channel": "email",
            "base_response_rate": 0.22,
            "avg_incremental_aov": 80.0,
            "discount_pct": 0.05,
        },
        {
            "target_segment": "Growing Customer",
            "action": "UPSELL",
            "campaign_size": 800,
            "channel": "email",
            "base_response_rate": 0.14,
            "avg_incremental_aov": 50.0,
            "discount_pct": 0.10,
        },
    ]
    scenario_comparison = simulator.compare_scenarios(final_customer_df, scenarios_to_run)
    rep_dir = root / config.paths.reports_dir
    scenario_comparison.to_csv(rep_dir / "campaign_scenario_simulations.csv", index=False)
    print(f"[Step 11] Simulated {len(scenarios_to_run)} campaigns -> {rep_dir / 'campaign_scenario_simulations.csv'}")

    # ---------------------------------------------------------
    # STEP 12: HOLDOUT EVALUATION & BENCHMARKING
    # ---------------------------------------------------------
    print("\n[Step 12] Benchmarking against actual holdout window ground truth...")
    # Calculate actual future revenue for training customers during holdout window
    holdout_cust_rev = (
        holdout_df.groupby("customer_id")["revenue"].sum().rename("actual_holdout_rev")
    )

    eval_df = train_features.set_index("customer_id").join(holdout_cust_rev, how="left")
    eval_df["actual_holdout_rev"] = eval_df["actual_holdout_rev"].fillna(0.0)

    # Baselines
    baseline_a = HistoricalAverageBaseline(config).fit(train_features)
    pred_base_a = baseline_a.predict(train_features, horizon_days=90)
    pred_base_a.index = train_features["customer_id"]

    baseline_b = RFMBaseline(config).fit(train_features)
    pred_base_b = baseline_b.predict(train_features, horizon_days=90)
    pred_base_b.index = train_features["customer_id"]

    prob_clv_pred = final_customer_df.set_index("customer_id")["clv_expected_90d"]
    clv_lower = final_customer_df.set_index("customer_id")["clv_lower_80pct"]
    clv_upper = final_customer_df.set_index("customer_id")["clv_upper_80pct"]

    evaluator = ModelEvaluator(config)
    bench_results = evaluator.evaluate_holdout_revenue(
        y_true=eval_df["actual_holdout_rev"],
        y_pred_probabilistic=prob_clv_pred,
        y_pred_baseline_a=pred_base_a,
        y_pred_baseline_b=pred_base_b,
    )

    cov_report = evaluator.evaluate_interval_coverage(
        y_true=eval_df["actual_holdout_rev"],
        lower_bounds=clv_lower,
        upper_bounds=clv_upper,
        nominal_level=0.80,
    )

    # Inactivity calibration target: no purchase during the first 90 days after the
    # training cutoff. This matches the model's 90-day inactivity horizon.
    train_cutoff = pd.to_datetime(config.temporal_splits.train_end_date)
    calibration_end = train_cutoff + pd.Timedelta(days=90)
    first_90d = holdout_df[
        (pd.to_datetime(holdout_df["transaction_date"]) > train_cutoff)
        & (pd.to_datetime(holdout_df["transaction_date"]) <= calibration_end)
    ]
    purchased_90d = set(first_90d["customer_id"].astype(str).unique())
    actual_inactive = pd.Series(
        (~eval_df.index.astype(str).isin(purchased_90d)).astype(int),
        index=eval_df.index,
        name="actual_inactive_90d",
    )
    inact_pred = final_customer_df.set_index("customer_id")["inactivity_prob_90d"]
    calib_report = evaluator.evaluate_calibration(inact_pred, actual_inactive)

    sparse_report = evaluator.evaluate_sparse_history_sensitivity(
        train_features,
        y_true=eval_df["actual_holdout_rev"],
        y_pred=prob_clv_pred,
    )

    # ---------------------------------------------------------
    # STEP 13: PRODUCTION MONITORING, SEGMENT STABILITY & DRIFT
    # ---------------------------------------------------------
    print("\n[Step 13] Running segment stability and Population Stability Index monitoring...")
    # Initialise the monitoring service before evaluating drift.  This was
    # previously referenced without construction, causing a NameError after
    # segmentation completed successfully.
    monitor = SystemMonitor(config)
    # Target features include customer history accumulated through the monitoring
    # cutoff. This preserves customer history instead of treating the holdout window
    # as a brand-new customer population.
    target_features = feature_builder.build_customer_features(
        clean_df,
        observation_cutoff=config.temporal_splits.holdout_end_date,
    )
    target_clv_df = clv_calc.compute_clv(target_features, bgf_model, ggf_model, n_simulation_samples=250)
    target_segmented = segmenter.segment_customers(target_clv_df)

    baseline_segments = final_customer_df.set_index("customer_id")["action_segment"]
    target_segments = target_segmented.set_index("customer_id")["action_segment"]
    segment_stability = evaluator.evaluate_segment_stability(baseline_segments, target_segments)

    drift_report = monitor.evaluate_drift(
        baseline_features=train_features,
        target_features=target_features,
        baseline_clv=final_customer_df,
        target_clv=target_segmented,
    )
    drift_report["segment_stability"] = segment_stability
    evaluator.save_evaluation_report(bench_results, cov_report, calib_report, sparse_report, segment_stability)
    monitor.save_monitoring_report(drift_report)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"CAPSTONE PIPELINE COMPLETED SUCCESSFULLY in {elapsed} seconds.")
    print("All processed tables, models, reports, and artifacts are fully generated.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    run_full_pipeline()
