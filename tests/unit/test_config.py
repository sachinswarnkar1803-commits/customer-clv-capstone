"""Unit tests for configuration loading and validation."""

import pytest
from src.config.config import load_config, AppConfig, get_project_root


def test_config_loads_defaults():
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project.code == "BDS-34"
    assert config.cleaning.drop_missing_customer_id is True
    assert config.clv.default_horizon_days == 90
    assert "RETENTION" in config.nba.actions
    assert len(config.models.survival.inactivity_thresholds) == 3


def test_project_root_exists():
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "default_config.yaml").exists()
