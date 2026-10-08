import pandas as pd

from src.evaluation.evaluator import ModelEvaluator


def test_segment_stability_transition_matrix():
    evaluator = ModelEvaluator()
    baseline = pd.Series(["A", "A", "B", "C"], index=[1, 2, 3, 4])
    target = pd.Series(["A", "B", "B", "C"], index=[1, 2, 3, 4])
    result = evaluator.evaluate_segment_stability(baseline, target)

    assert result["evaluated_customers"] == 4
    assert result["agreement_rate"] == 0.75
    assert result["transition_matrix"]
