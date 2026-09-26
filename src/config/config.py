"""Configuration loader and management for BDS-34 Capstone Project."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "Probabilistic Customer Lifetime Value with Cohort Dynamics and Next-Best-Action Segments"
    code: str = "BDS-34"
    version: str = "1.0.0"
    random_seed: int = 42


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"
    validation_dir: str = "data/validation"
    models_dir: str = "models"
    reports_dir: str = "reports"
    raw_zip_name: str = "online_retail_ii.zip"
    raw_excel_name: str = "online_retail_II.xlsx"
    clean_transactions: str = "data/processed/clean_transactions.parquet"
    clean_transactions_csv: str = "data/processed/clean_transactions.csv"
    customer_features: str = "data/processed/customer_features.parquet"
    customer_features_csv: str = "data/processed/customer_features.csv"
    clv_segments: str = "data/processed/customer_clv_segments.parquet"
    clv_segments_csv: str = "data/processed/customer_clv_segments.csv"


class DataSourceConfig(BaseModel):
    url: str = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
    archive_inner_file: str = "online_retail_II.xlsx"
    expected_sheets: List[str] = ["Year 2009-2010", "Year 2010-2011"]
    columns: Dict[str, str] = {
        "invoice": "Invoice",
        "stock_code": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoice_date": "InvoiceDate",
        "price": "Price",
        "customer_id": "Customer ID",
        "country": "Country",
    }


class CleaningConfig(BaseModel):
    drop_missing_customer_id: bool = True
    exclude_cancellations: bool = True
    min_quantity: int = 1
    min_unit_price: float = 0.001
    max_quantity_outlier: int = 50000
    max_price_outlier: float = 50000.0
    deduplicate: bool = True
    anonymize_customer_id: bool = False


class TemporalSplitsConfig(BaseModel):
    train_start_date: str = "2009-12-01"
    train_end_date: str = "2010-12-09"
    validation_end_date: str = "2011-06-30"
    holdout_end_date: str = "2011-12-09"


class CohortConfig(BaseModel):
    frequency: str = "MS"
    retention_periods: int = 12


class RFMConfig(BaseModel):
    recency_bins: int = 5
    frequency_bins: int = 5
    monetary_bins: int = 5


class BGNBDConfig(BaseModel):
    penalizer_coef: float = 0.01


class GammaGammaConfig(BaseModel):
    penalizer_coef: float = 0.01


class SurvivalConfig(BaseModel):
    time_unit: str = "days"
    inactivity_thresholds: List[int] = [30, 60, 90]
    parametric_distribution: str = "weibull"


class ModelsConfig(BaseModel):
    bgnbd: BGNBDConfig = BGNBDConfig()
    gamma_gamma: GammaGammaConfig = GammaGammaConfig()
    survival: SurvivalConfig = SurvivalConfig()


class CLVConfig(BaseModel):
    discount_rate: float = 0.01
    horizons_days: List[int] = [30, 90, 180, 365]
    default_horizon_days: int = 90
    confidence_level: float = 0.80


class SegmentsConfig(BaseModel):
    high_value_clv_percentile: float = 0.75
    low_value_clv_percentile: float = 0.25
    high_risk_inactivity_prob: float = 0.60
    medium_risk_inactivity_prob: float = 0.35


class NBAConfig(BaseModel):
    actions: List[str] = [
        "RETENTION",
        "WIN_BACK",
        "UPSELL",
        "CROSS_SELL",
        "LOYALTY_REWARD",
        "REACTIVATION",
        "LOW_COST_ENGAGEMENT",
        "NO_ACTION",
    ]
    channels: Dict[str, Dict[str, float]] = {
        "email": {"cost_per_contact": 0.05},
        "sms": {"cost_per_contact": 0.20},
        "direct_mail": {"cost_per_contact": 2.50},
        "push": {"cost_per_contact": 0.02},
    }


class SimulationConfig(BaseModel):
    default_campaign_size: int = 1000
    default_base_response_rate: float = 0.12
    default_discount_pct: float = 0.10
    default_avg_incremental_aov: float = 45.0


class AppConfig(BaseModel):
    project: ProjectConfig = ProjectConfig()
    paths: PathsConfig = PathsConfig()
    data_source: DataSourceConfig = DataSourceConfig()
    cleaning: CleaningConfig = CleaningConfig()
    temporal_splits: TemporalSplitsConfig = TemporalSplitsConfig()
    cohort: CohortConfig = CohortConfig()
    rfm: RFMConfig = RFMConfig()
    models: ModelsConfig = ModelsConfig()
    clv: CLVConfig = CLVConfig()
    segments: SegmentsConfig = SegmentsConfig()
    nba: NBAConfig = NBAConfig()
    simulation: SimulationConfig = SimulationConfig()


def get_project_root() -> Path:
    """Return the absolute path to customer-clv-capstone root directory."""
    # This file is located at <project_root>/src/config/config.py
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load application configuration from YAML file or return defaults."""
    root = get_project_root()
    if config_path is None:
        config_path = str(root / "configs" / "default_config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        return AppConfig(**yaml_data)
    return AppConfig()
