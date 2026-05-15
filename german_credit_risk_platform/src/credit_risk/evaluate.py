from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score

from credit_risk.config import load_json_config

MODEL_CONFIG = load_json_config("model_config.json")


def calculate_metrics(y_true: pd.Series, y_probability: np.ndarray, threshold: float) -> dict[str, float | list[list[int]]]:
    y_pred = (y_probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "bad_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "bad_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "bad_f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def optimize_threshold(y_true: pd.Series, y_probability: np.ndarray) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    thresholds = np.linspace(
        MODEL_CONFIG["threshold_grid_start"],
        MODEL_CONFIG["threshold_grid_stop"],
        MODEL_CONFIG["threshold_grid_steps"],
    )
    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        business_cost = fn * MODEL_CONFIG["false_negative_cost"] + fp * MODEL_CONFIG["false_positive_cost"]
        rows.append(
            {
                "threshold": float(threshold),
                "business_cost": float(business_cost),
                "bad_recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "bad_f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
            }
        )
    return sorted(rows, key=lambda row: (row["business_cost"], -row["bad_recall"]))[0]
