"""Integration tests verifying complete pipeline execution and artifact integrity."""

from pathlib import Path
import pandas as pd
import pytest

from src.config.config import load_config, get_project_root


def test_processed_artifacts_integrity():
    config = load_config()
    root = get_project_root()

    # 1. Clean transactions
    clean_path = root / config.paths.clean_transactions
    assert clean_path.exists(), f"Clean transactions missing at {clean_path}"
    df_clean = pd.read_parquet(clean_path)
    assert len(df_clean) > 500000
    assert "revenue" in df_clean.columns
    assert (df_clean["revenue"] > 0).all()

    # 2. Customer features
    features_path = root / config.paths.customer_features
    assert features_path.exists()
    df_feat = pd.read_parquet(features_path)
    assert len(df_feat) > 3000
    assert "frequency" in df_feat.columns
    assert "recency_days" in df_feat.columns
    assert "rfm_segment" in df_feat.columns

    # 3. CLV & NBA Segments
    clv_path = root / config.paths.clv_segments
    assert clv_path.exists()
    df_clv = pd.read_parquet(clv_path)
    assert len(df_clv) == len(df_feat)
    assert "clv_expected_90d" in df_clv.columns
    assert "clv_lower_80pct" in df_clv.columns
    assert "clv_upper_80pct" in df_clv.columns
    assert "action_segment" in df_clv.columns
    assert "nba_recommended_action" in df_clv.columns
    assert "nba_channel" in df_clv.columns

    # 4. Cohort Retention Matrix
    ret_path = root / config.paths.processed_dir / "cohort_retention_matrix.csv"
    assert ret_path.exists()
    df_ret = pd.read_csv(ret_path)
    assert len(df_ret) >= 12

    # 5. Model Evaluation Report
    eval_json = root / config.paths.reports_dir / "model_evaluation_report.json"
    assert eval_json.exists()

    # 6. Monitoring Drift Report
    mon_json = root / config.paths.reports_dir / "monitoring_report.json"
    assert mon_json.exists()
