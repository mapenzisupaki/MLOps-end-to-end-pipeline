import numpy as np
import pandas as pd

from credit_risk.fairness import fairness_summary


def test_fairness_summary_contains_required_metrics():
    X = pd.DataFrame({"Age": [22, 30, 44, 67], "Sex": ["male", "female", "male", "female"]})
    y_true = pd.Series([1, 0, 1, 0])
    probabilities = np.array([0.8, 0.2, 0.7, 0.3])
    report = fairness_summary(y_true, probabilities, X, threshold=0.5)
    assert "Sex" in report
    assert "Age" in report
    assert "demographic_parity_ratio" in report["Sex"]
    assert "equalized_odds_difference" in report["Age"]
