from __future__ import annotations

import json

import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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


def build_candidate_models(preprocessor) -> dict[str, Pipeline]:
    random_state = MODEL_CONFIG["random_state"]
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def train() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    X, y = prepare_modeling_table(raw)
    split = split_data(X, y)
    preprocessor = build_preprocessor(split.X_train)

    best_name = ""
    best_model = None
    best_auc = -1.0
    comparison: dict[str, dict[str, float]] = {}

    for name, model in build_candidate_models(preprocessor).items():
        model.fit(split.X_train, split.y_train)
        probabilities = model.predict_proba(split.X_valid)[:, 1]
        threshold = optimize_threshold(split.y_valid, probabilities)["threshold"]
        metrics = calculate_metrics(split.y_valid, probabilities, threshold)
        comparison[name] = {
            key: value for key, value in metrics.items() if isinstance(value, float)
        }
        if float(metrics["roc_auc"]) > best_auc:
            best_name = name
            best_model = model
            best_auc = float(metrics["roc_auc"])

    if best_model is None:
        raise RuntimeError("No model was trained.")

    valid_probabilities = best_model.predict_proba(split.X_valid)[:, 1]
    threshold_info = optimize_threshold(split.y_valid, valid_probabilities)
    test_probabilities = best_model.predict_proba(split.X_test)[:, 1]
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

    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "validation_model_comparison": comparison,
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
