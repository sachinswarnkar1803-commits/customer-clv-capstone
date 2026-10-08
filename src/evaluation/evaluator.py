"""Model evaluation, benchmarking, interval coverage, calibration, and sensitivity analysis module.

Provides rigorous empirical validation required for capstone evaluation:
- Holdout Revenue Error benchmarking (Probabilistic CLV vs Baseline A vs Baseline B)
- 80% CLV Prediction Interval Coverage Rate
- Inactivity Probability Calibration (Brier score & reliability)
- Sparse-History Sensitivity (evaluating degradation for 1, 2, 3 purchases)
- Segment Stability & Transition Matrix
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error

from src.config.config import AppConfig, get_project_root, load_config


class ModelEvaluator:
    """Computes comprehensive quantitative evaluation metrics across models and customer segments."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()

    def evaluate_holdout_revenue(
        self,
        y_true: pd.Series,
        y_pred_probabilistic: pd.Series,
        y_pred_baseline_a: pd.Series,
        y_pred_baseline_b: pd.Series,
    ) -> pd.DataFrame:
        """Benchmark Probabilistic CLV against Baseline A (Historical Run Rate) and Baseline B (RFM)."""
        models = {
            "Probabilistic CLV (BG/NBD + GG)": y_pred_probabilistic,
            "Baseline A (Historical Spend Extrapolation)": y_pred_baseline_a,
            "Baseline B (RFM Segment Benchmarks)": y_pred_baseline_b,
        }

        actual_total = float(y_true.sum())
        results = []

        for name, pred in models.items():
            # Align series and fill NaNs
            common_idx = y_true.index.intersection(pred.index)
            y_t = np.nan_to_num(y_true.loc[common_idx].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
            y_p = np.nan_to_num(pred.loc[common_idx].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            pred_total = float(np.sum(y_p))
            agg_error = pred_total - actual_total
            agg_pct_error = (agg_error / max(1.0, actual_total)) * 100.0

            # Revenue-weighted MAE: weights errors proportionally to customer actual spend
            rev_weights = (y_t + 1.0) / (np.sum(y_t) + len(y_t))
            rev_weighted_mae = float(np.sum(np.abs(y_t - y_p) * rev_weights))

            # Spearman rank correlation
            corr, _ = spearmanr(y_t, y_p)

            results.append({
                "Model": name,
                "MAE (£)": round(float(mae), 2),
                "RMSE (£)": round(float(rmse), 2),
                "Rev-Weighted MAE (£)": round(float(rev_weighted_mae), 2),
                "Spearman Rank Corr": round(float(corr), 3),
                "Aggregate Predicted Rev (£)": round(pred_total, 2),
                "Actual Holdout Rev (£)": round(actual_total, 2),
                "Aggregate Bias (%)": round(float(agg_pct_error), 2),
            })

        df_bench = pd.DataFrame(results)
        print("\n[Evaluation] Holdout Revenue Benchmarking Results:")
        print(df_bench.to_string(index=False))
        return df_bench

    def evaluate_interval_coverage(
        self,
        y_true: pd.Series,
        lower_bounds: pd.Series,
        upper_bounds: pd.Series,
        nominal_level: float = 0.80,
    ) -> Dict[str, Any]:
        """Compute empirical coverage rate of the uncertainty intervals against actual holdout revenue."""
        common_idx = y_true.index.intersection(lower_bounds.index).intersection(upper_bounds.index)
        y_t = y_true.loc[common_idx].values
        low = lower_bounds.loc[common_idx].values
        high = upper_bounds.loc[common_idx].values

        # Inside interval condition
        inside = (y_t >= low) & (y_t <= high)
        empirical_coverage = float(np.mean(inside))
        below = float(np.mean(y_t < low))
        above = float(np.mean(y_t > high))

        coverage_report = {
            "nominal_target": nominal_level,
            "empirical_coverage": round(empirical_coverage, 4),
            "empirical_coverage_pct": round(empirical_coverage * 100.0, 2),
            "below_lower_bound_pct": round(below * 100.0, 2),
            "above_upper_bound_pct": round(above * 100.0, 2),
            "evaluated_customers": len(common_idx),
            "status": "Coverage is within the configured tolerance band."
            if abs(empirical_coverage - nominal_level) <= 0.05
            else "Coverage is outside the configured tolerance band; treat the interval as a model limitation and investigate recalibration.",
        }

        print(f"[Evaluation] Interval Coverage: {coverage_report['empirical_coverage_pct']}% "
              f"(Target: {int(nominal_level*100)}%) -> {coverage_report['status']}")
        return coverage_report

    def evaluate_calibration(
        self,
        p_inactivity: pd.Series,
        is_actually_inactive: pd.Series,
        n_bins: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate calibration of inactivity probability predictions via Brier score and bin reliability."""
        common_idx = p_inactivity.index.intersection(is_actually_inactive.index)
        p = np.nan_to_num(p_inactivity.loc[common_idx].values.astype(float), nan=0.5)
        p = np.clip(p, 0.0, 1.0)
        y = np.nan_to_num(is_actually_inactive.loc[common_idx].values.astype(int), nan=1)

        brier = float(brier_score_loss(y, p))

        # Quantile binning for calibration curve
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_records = []
        for i in range(n_bins):
            mask = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
            if mask.sum() > 0:
                mean_pred = float(np.mean(p[mask]))
                obs_rate = float(np.mean(y[mask]))
                bin_records.append({
                    "bin": f"{bins[i]:.2f}-{bins[i+1]:.2f}",
                    "count": int(mask.sum()),
                    "mean_predicted_prob": round(mean_pred, 3),
                    "observed_inactivity_rate": round(obs_rate, 3),
                    "abs_calibration_gap": round(abs(mean_pred - obs_rate), 3),
                })

        calibration_report = {
            "brier_score": round(brier, 4),
            "brier_interpretation": "Lower is better; compare with a prevalence baseline and inspect the reliability table." ,
            "calibration_bins": bin_records,
        }
        print(f"[Evaluation] Inactivity Calibration Brier Score: {brier:.4f}")
        return calibration_report


    def evaluate_segment_stability(
        self,
        baseline_segments: pd.Series,
        target_segments: pd.Series,
    ) -> Dict[str, Any]:
        """Measure segment membership stability and customer-level transitions."""
        common = baseline_segments.index.intersection(target_segments.index)
        if len(common) == 0:
            return {"evaluated_customers": 0, "agreement_rate": None, "transition_matrix": []}

        base = baseline_segments.loc[common].astype(str)
        target = target_segments.loc[common].astype(str)
        agreement = float((base == target).mean())
        transition = pd.crosstab(base, target, normalize="index").round(4)
        counts = pd.crosstab(base, target)
        return {
            "evaluated_customers": int(len(common)),
            "agreement_rate": round(agreement, 4),
            "agreement_rate_pct": round(agreement * 100.0, 2),
            "transition_matrix": transition.reset_index().to_dict(orient="records"),
            "transition_counts": counts.reset_index().to_dict(orient="records"),
            "method": "Row-normalized customer segment transition matrix",
        }

    def evaluate_sparse_history_sensitivity(
        self,
        features_df: pd.DataFrame,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> pd.DataFrame:
        """Examine how prediction error degrades as customer history length becomes sparse (1, 2, 3 purchases)."""
        common = features_df.set_index("customer_id").join(
            pd.DataFrame({"actual": y_true, "predicted": y_pred}),
            how="inner",
        ).fillna({"actual": 0.0, "predicted": 0.0})

        def bucket_history(f):
            if f == 0:
                return "1 Purchase (Zero Repeat, Sparse)"
            elif f == 1:
                return "2 Purchases (1 Repeat)"
            elif f == 2:
                return "3 Purchases (2 Repeats)"
            else:
                return "4+ Purchases (Rich History)"

        common["history_bucket"] = common["frequency"].apply(bucket_history)

        grouped = common.groupby("history_bucket")
        results = []
        order = [
            "1 Purchase (Zero Repeat, Sparse)",
            "2 Purchases (1 Repeat)",
            "3 Purchases (2 Repeats)",
            "4+ Purchases (Rich History)",
        ]

        for bucket in order:
            if bucket in grouped.groups:
                grp = grouped.get_group(bucket)
                act = np.nan_to_num(grp["actual"].values)
                pred = np.nan_to_num(grp["predicted"].values)
                mae = mean_absolute_error(act, pred)
                rmse = np.sqrt(mean_squared_error(act, pred))
                mean_actual = np.mean(act)
                mean_pred = np.mean(pred)
                results.append({
                    "History Depth": bucket,
                    "Customer Count": len(grp),
                    "Actual Mean Rev (£)": round(float(mean_actual), 2),
                    "Predicted Mean Rev (£)": round(float(mean_pred), 2),
                    "MAE (£)": round(float(mae), 2),
                    "RMSE (£)": round(float(rmse), 2),
                })

        df_sparse = pd.DataFrame(results)
        print("\n[Evaluation] Sparse-History Sensitivity Analysis:")
        print(df_sparse.to_string(index=False))
        return df_sparse

    def save_evaluation_report(
        self,
        benchmark_df: pd.DataFrame,
        coverage_report: Dict[str, Any],
        calibration_report: Dict[str, Any],
        sparse_df: pd.DataFrame,
        segment_stability: Optional[Dict[str, Any]] = None,
    ):
        """Save evaluation results to reports/model_evaluation_report.json and .md."""
        rep_dir = self.root / self.config.paths.reports_dir
        rep_dir.mkdir(parents=True, exist_ok=True)

        full_eval = {
            "benchmark_results": benchmark_df.to_dict(orient="records"),
            "interval_coverage": coverage_report,
            "calibration": calibration_report,
            "sparse_history_sensitivity": sparse_df.to_dict(orient="records"),
            "segment_stability": segment_stability or {},
        }

        # 1. JSON report
        with open(rep_dir / "model_evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(full_eval, f, indent=2)

        def df_to_md(df: pd.DataFrame) -> str:
            try:
                return df.to_markdown(index=False)
            except Exception:
                headers = list(df.columns)
                lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for _, r in df.iterrows():
                    lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
                return "\n".join(lines)

        # 2. Markdown report
        with open(rep_dir / "model_evaluation_report.md", "w", encoding="utf-8") as f:
            f.write("# Model Evaluation & Benchmark Report\n\n")
            f.write(f"**Project**: {self.config.project.name} ({self.config.project.code})\n\n")
            f.write("## 1. Holdout Revenue Error Benchmarking\n\n")
            f.write(df_to_md(benchmark_df))
            f.write("\n\n## 2. 80% CLV Prediction Interval Empirical Coverage\n\n")
            f.write(f"- **Nominal Target**: {coverage_report['nominal_target']*100:.0f}%\n")
            f.write(f"- **Empirical Coverage Rate**: {coverage_report['empirical_coverage_pct']}%\n")
            f.write(f"- **Evaluated Customers**: {coverage_report['evaluated_customers']:,}\n")
            f.write(f"- **Notes**: {coverage_report['status']}\n\n")
            f.write("## 3. Inactivity Risk Probability Calibration\n\n")
            f.write(f"- **Brier Score**: {calibration_report['brier_score']} ({calibration_report['brier_interpretation']})\n\n")
            f.write("### Calibration Reliability Table\n\n")
            f.write(df_to_md(pd.DataFrame(calibration_report["calibration_bins"])))
            f.write("\n\n## 4. Sparse-History Sensitivity\n\n")
            f.write(df_to_md(sparse_df))
            f.write("\n\n## 5. Segment Stability and Customer Transitions\n\n")
            if segment_stability:
                f.write(f"- Evaluated customers: {segment_stability.get('evaluated_customers', 0):,}\n")
                f.write(f"- Segment agreement rate: {segment_stability.get('agreement_rate_pct', 0):.2f}%\n")
                f.write("- The transition matrix is row-normalized and is intended for monitoring, not as a target performance score.\n")

        print(f"[Evaluation] Saved evaluation reports to {rep_dir}")
