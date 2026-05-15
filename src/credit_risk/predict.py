from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.config import MODEL_PATH, THRESHOLD_PATH
from credit_risk.data import normalize_account_columns


def model_artifacts_ready() -> bool:
    return MODEL_PATH.exists() and THRESHOLD_PATH.exists()


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def load_threshold(path: Path = THRESHOLD_PATH) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Threshold artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["threshold"])


def score_dataframe(df: pd.DataFrame, model=None, threshold: float | None = None) -> pd.DataFrame:
    active_model = model if model is not None else load_model()
    active_threshold = threshold if threshold is not None else load_threshold()
    normalized = normalize_account_columns(df)
    probability = active_model.predict_proba(normalized)[:, 1]
    prediction = (probability >= active_threshold).astype(int)
    output = normalized.copy()
    output["bad_loan_probability"] = probability
    output["predicted_bad"] = prediction
    output["recommended_action"] = output["predicted_bad"].map({0: "approve_or_fast_track", 1: "review_or_decline"})
    return output
