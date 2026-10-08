"""Regression tests for dashboard/pipeline bugs fixed in the CRM release."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_initializes_monitor_before_drift():
    text = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    marker = "monitor = SystemMonitor(config)"
    drift = "drift_report = monitor.evaluate_drift("
    assert marker in text
    assert text.index(marker) < text.index(drift)


def test_dashboard_normalizes_evaluation_metric_names():
    text = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    assert '"MAE (£)": "MAE"' in text
    assert '"RMSE (£)": "RMSE"' in text
    assert 'px.bar(bench_plot, x="Model", y="MAE"' in text


def test_dashboard_does_not_use_streamlit_ternary_command_expression():
    text = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    bad_coverage = 'st.warning(cov.get("status", "")) if abs(actual-target) > 5 else st.success'
    bad_validation = '(st.success if status == "PASS" else st.warning)'
    assert bad_coverage not in text
    assert bad_validation not in text
