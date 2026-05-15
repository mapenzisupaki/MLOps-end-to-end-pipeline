from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.explain import explain_single_prediction
from credit_risk.train import build_xgboost_pipeline


def test_explain_single_prediction_returns_top_features(tmp_path: Path):
    X = pd.DataFrame(
        {
            "Age": [25, 40, 55, 32, 47, 61],
            "Sex": ["male", "female", "male", "female", "male", "female"],
            "Job": [1, 2, 2, 3, 1, 2],
            "Housing": ["own", "rent", "own", "free", "rent", "own"],
            "Saving accounts": ["little", "moderate", "No Account", "rich", "little", "moderate"],
            "Checking account": ["moderate", "little", "No Account", "rich", "little", "moderate"],
            "Credit amount": [1000, 4000, 7000, 3000, 5000, 9000],
            "Duration": [12, 24, 36, 18, 30, 48],
            "Purpose": ["car", "radio/TV", "business", "car", "education", "business"],
        }
    )
    y = np.array([0, 1, 1, 0, 1, 1])
    pipeline: Pipeline = build_xgboost_pipeline(
        preprocessor=__import__("credit_risk.features", fromlist=["build_preprocessor"]).build_preprocessor(X),
        scale_pos_weight=1.0,
    )
    pipeline.fit(X, y)

    explanation = explain_single_prediction(pipeline, X.head(1), top_n=3)

    assert explanation["method"] == "xgboost_tree_shap_pred_contribs"
    assert explanation["model_family"] == "XGBClassifier"
    assert len(explanation["top_features"]) == 3
    assert "shap_value" in explanation["top_features"][0]

