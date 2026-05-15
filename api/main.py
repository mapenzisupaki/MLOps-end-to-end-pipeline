from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.predict import (
    load_model,
    load_threshold,
    model_artifacts_ready,
    score_single_application,
)

app = FastAPI(title="German Credit Risk API", version="0.2.0")


class LoanApplication(BaseModel):
    Age: int = Field(ge=18, le=100)
    Sex: str = Field(pattern="^(male|female)$")
    Job: int = Field(ge=0, le=3)
    Housing: str
    Saving_accounts: str | None = None
    Checking_account: str | None = None
    Credit_amount: int = Field(gt=0)
    Duration: int = Field(gt=0)
    Purpose: str

    def to_model_record(self) -> dict[str, object]:
        return {
            "Age": self.Age,
            "Sex": self.Sex,
            "Job": self.Job,
            "Housing": self.Housing,
            "Saving accounts": self.Saving_accounts,
            "Checking account": self.Checking_account,
            "Credit amount": self.Credit_amount,
            "Duration": self.Duration,
            "Purpose": self.Purpose,
        }


@lru_cache(maxsize=1)
def load_runtime_assets():
    return load_model(), load_threshold()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_ready": model_artifacts_ready(),
        "explainability": "xgboost_tree_shap_pred_contribs",
    }


@app.post("/predict")
def predict(application: LoanApplication) -> dict[str, object]:
    if not model_artifacts_ready():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run training before serving predictions.",
        )
    try:
        model, threshold = load_runtime_assets()
        frame = pd.DataFrame([application.to_model_record()])
        return score_single_application(
            frame,
            model=model,
            threshold=threshold,
            explanation_top_n=5,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

