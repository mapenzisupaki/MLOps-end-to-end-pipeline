from __future__ import annotations

import json

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_risk.config import (
    ARTIFACTS_DIR,
    FAIRNESS_REPORT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    THRESHOLD_PATH,
    load_json_config,
)
from credit_risk.data import load_raw_data, prepare_modeling_table, split_data
from credit_risk.evaluate import calculate_metrics, optimize_threshold
from credit_risk.fairness import fairness_summary
from credit_risk.features import build_preprocessor

MODEL_CONFIG = load_json_config("model_config.json")


def build_xgboost_pipeline(preprocessor, scale_pos_weight: float) -> Pipeline:
    random_state = MODEL_CONFIG["random_state"]
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    X, y = prepare_modeling_table(raw)
    split = split_data(X, y)
    preprocessor = build_preprocessor(split.X_train)
    positive_count = int(split.y_train.sum())
    negative_count = int(len(split.y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    model = build_xgboost_pipeline(preprocessor, scale_pos_weight)
    model.fit(split.X_train, split.y_train)

    valid_probabilities = model.predict_proba(split.X_valid)[:, 1]
    threshold_info = optimize_threshold(split.y_valid, valid_probabilities)
    validation_metrics = calculate_metrics(
        split.y_valid,
        valid_probabilities,
        threshold_info["threshold"],
    )
    test_probabilities = model.predict_proba(split.X_test)[:, 1]
    test_metrics = calculate_metrics(
        split.y_test,
        test_probabilities,
        threshold_info["threshold"],
    )
    fairness_report = fairness_summary(
        split.y_test,
        test_probabilities,
        split.X_test,
        threshold_info["threshold"],
    )

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "selected_model": "xgboost",
                "model_family": "XGBClassifier",
                "optimization_focus": ["bad_recall", "bad_f2", "roc_auc"],
                "scale_pos_weight": scale_pos_weight,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    THRESHOLD_PATH.write_text(json.dumps(threshold_info, indent=2), encoding="utf-8")
    FAIRNESS_REPORT_PATH.write_text(
        json.dumps(fairness_report, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    train()
