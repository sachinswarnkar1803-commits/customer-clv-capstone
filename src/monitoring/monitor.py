"""Production-grade data quality and prediction drift monitoring module.

Computes Population Stability Index (PSI), distribution shift statistics,
and segment population transitions to detect data or concept drift across time windows.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config.config import AppConfig, get_project_root, load_config


class SystemMonitor:
    """Monitors distribution drift and data quality health across temporal partitions."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()

    def calculate_psi(
        self,
        baseline: np.ndarray,
        target: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Compute Population Stability Index (PSI) between baseline and target distributions.

        Rules of thumb:
            PSI < 0.10: Stable (No shift)
            0.10 <= PSI < 0.25: Moderate shift (Monitor closely)
            PSI >= 0.25: Significant shift (Action/Retrain needed)
        """
        baseline = baseline[~np.isnan(baseline)]
        target = target[~np.isnan(target)]

        if len(baseline) == 0 or len(target) == 0:
            return 0.0

        # Bin edges based on baseline quantiles
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(baseline, quantiles)
        bin_edges[0] -= 1e-5
        bin_edges[-1] += 1e-5
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        base_counts, _ = np.histogram(baseline, bins=bin_edges)
        target_counts, _ = np.histogram(target, bins=bin_edges)

        base_pct = (base_counts / len(baseline)) + epsilon
        target_pct = (target_counts / len(target)) + epsilon

        # Re-normalize with epsilon
        base_pct /= base_pct.sum()
        target_pct /= target_pct.sum()

        psi_val = np.sum((target_pct - base_pct) * np.log(target_pct / base_pct))
        return float(np.round(psi_val, 4))

    def evaluate_drift(
        self,
        baseline_features: pd.DataFrame,
        target_features: pd.DataFrame,
        baseline_clv: pd.DataFrame,
        target_clv: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute comprehensive drift metrics across feature and prediction distributions."""
        drift_metrics = {}

        # 1. Feature Drift (PSI)
        features_to_monitor = [
            "frequency",
            "customer_tenure_days",
            "total_revenue",
            "average_order_value",
        ]

        for feat in features_to_monitor:
            if feat in baseline_features.columns and feat in target_features.columns:
                psi = self.calculate_psi(
                    baseline_features[feat].values,
                    target_features[feat].values,
                )
                drift_metrics[f"psi_{feat}"] = {
                    "psi_value": psi,
                    "status": "STABLE" if psi < 0.10 else ("MODERATE_DRIFT" if psi < 0.25 else "SIGNIFICANT_DRIFT"),
                }

        # 2. Prediction Drift (CLV & Inactivity)
        def_horizon = self.config.clv.default_horizon_days
        clv_col = f"clv_expected_{def_horizon}d"

        if clv_col in baseline_clv.columns and clv_col in target_clv.columns:
            clv_psi = self.calculate_psi(
                baseline_clv[clv_col].values,
                target_clv[clv_col].values,
            )
            drift_metrics["psi_clv_expected"] = {
                "psi_value": clv_psi,
                "status": "STABLE" if clv_psi < 0.10 else ("MODERATE_DRIFT" if clv_psi < 0.25 else "SIGNIFICANT_DRIFT"),
            }

        if "inactivity_prob_90d" in baseline_clv.columns and "inactivity_prob_90d" in target_clv.columns:
            inact_psi = self.calculate_psi(
                baseline_clv["inactivity_prob_90d"].values,
                target_clv["inactivity_prob_90d"].values,
            )
            drift_metrics["psi_inactivity_prob_90d"] = {
                "psi_value": inact_psi,
                "status": "STABLE" if inact_psi < 0.10 else ("MODERATE_DRIFT" if inact_psi < 0.25 else "SIGNIFICANT_DRIFT"),
            }

        # 3. Segment Population Drift
        if "action_segment" in baseline_clv.columns and "action_segment" in target_clv.columns:
            base_seg_pct = baseline_clv["action_segment"].value_counts(normalize=True).to_dict()
            targ_seg_pct = target_clv["action_segment"].value_counts(normalize=True).to_dict()

            all_segs = sorted(list(set(base_seg_pct.keys()).union(set(targ_seg_pct.keys()))))
            seg_drift = []
            for seg in all_segs:
                b_p = base_seg_pct.get(seg, 0.0)
                t_p = targ_seg_pct.get(seg, 0.0)
                seg_drift.append({
                    "segment": seg,
                    "baseline_pct": round(b_p * 100, 2),
                    "target_pct": round(t_p * 100, 2),
                    "drift_pct_points": round((t_p - b_p) * 100, 2),
                })
            drift_metrics["segment_population_drift"] = seg_drift

        # Overall Status
        all_psis = [v["psi_value"] for k, v in drift_metrics.items() if isinstance(v, dict) and "psi_value" in v]
        max_psi = max(all_psis) if all_psis else 0.0
        drift_metrics["overall_health"] = {
            "max_psi": max_psi,
            "overall_status": "HEALTHY" if max_psi < 0.10 else ("WARNING" if max_psi < 0.25 else "ALERT"),
        }

        print(f"[Monitoring] Drift Analysis Complete. Overall Health: {drift_metrics['overall_health']['overall_status']} (Max PSI: {max_psi:.4f})")
        return drift_metrics

    def save_monitoring_report(self, drift_metrics: Dict[str, Any]):
        """Save monitoring metrics to reports/monitoring_report.json."""
        rep_dir = self.root / self.config.paths.reports_dir
        rep_dir.mkdir(parents=True, exist_ok=True)
        report_path = rep_dir / "monitoring_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(drift_metrics, f, indent=2)
        print(f"[Monitoring] Saved monitoring report to {report_path}")
