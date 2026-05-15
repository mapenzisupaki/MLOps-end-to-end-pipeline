from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float | str | int | None
    shap_value: float
    impact: str


def _to_dense_array(transformed) -> np.ndarray:
    if hasattr(transformed, "toarray"):
        return transformed.toarray()
    return np.asarray(transformed)


def _feature_value(value: float) -> float | int:
    if np.isclose(value, round(value)):
        return int(round(value))
    return float(value)


def explain_single_prediction(
    model_pipeline: Pipeline,
    applicant: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, object]:
    """Return local SHAP contributions for one normalized applicant row.

    XGBoost's `pred_contribs=True` returns exact TreeSHAP-style feature
    contributions plus a final bias term for tree boosters.
    """
    if len(applicant) != 1:
        raise ValueError("SHAP explanations require exactly one applicant row.")

    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    transformed = _to_dense_array(preprocessor.transform(applicant))
    feature_names = preprocessor.get_feature_names_out().tolist()

    matrix = xgb.DMatrix(transformed, feature_names=feature_names)
    contributions = model.get_booster().predict(matrix, pred_contribs=True)[0]
    row_values = contributions[:-1]
    base_value = float(contributions[-1])
    transformed_row = transformed[0]

    ordered_indices = np.argsort(np.abs(row_values))[::-1][:top_n]
    top_features = []
    for index in ordered_indices:
        shap_value = float(row_values[index])
        top_features.append(
            FeatureContribution(
                feature=feature_names[index],
                value=_feature_value(float(transformed_row[index])),
                shap_value=shap_value,
                impact="increases_risk" if shap_value >= 0 else "decreases_risk",
            ).__dict__
        )

    return {
        "method": "xgboost_tree_shap_pred_contribs",
        "model_family": type(model).__name__,
        "base_value": base_value,
        "top_features": top_features,
    }
