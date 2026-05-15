# German Credit Risk Platform

A production-minded machine learning and MLOps scaffold for retail banking credit risk assessment using the German Credit Risk dataset. The system predicts whether a loan applicant is a `Good` or `Bad` credit risk and is designed around explainability, fairness auditing, FastAPI serving, Docker packaging, and a GCP deployment path.

This folder is a clean production rebuild. The earlier `german_credit_risk_modeling` project remains a prototype/reference.

## Business Objective

A bank must decide whether to approve or reject each loan application. The most costly modeling error is approving a bad-risk applicant. For that reason, this project prioritizes:

- recall on the `Bad` class
- F2-score for the `Bad` class
- AUC-ROC
- transparent threshold selection based on false negative and false positive costs
- real-time SHAP explanations for each API prediction

Raw accuracy is reported but is not the primary optimization target.

## Dataset

The German Credit Risk dataset contains 1,000 loan applicants labeled as `Good` or `Bad` risk.

Target:

- `Risk`: `good` or `bad`

Features:

- `Age`: applicant age
- `Sex`: `male` or `female`
- `Job`: 0 unskilled non-resident, 1 unskilled resident, 2 skilled, 3 highly skilled
- `Housing`: `own`, `rent`, or `free`
- `Saving accounts`: savings account status, missing values treated as `No Account`
- `Checking account`: checking account status, missing values treated as `No Account`
- `Credit amount`: requested credit amount
- `Duration`: loan duration in months
- `Purpose`: loan purpose

Protected attributes for fairness audit:

- `Age`
- `Sex`

## Project Structure

```text
api/                    FastAPI application
configs/                JSON configuration files
data/raw/               raw German Credit dataset
deployment/             Docker and GCP deployment guidance
docs/                   model card, fairness audit notes, production checklist
models/                 generated model artifacts
artifacts/              generated metrics and audit outputs
src/credit_risk/        reusable ML package
tests/                  pytest suite
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model and generate artifacts:

```bash
$env:PYTHONPATH = "src"
python -m credit_risk.train
```

Run tests:

```bash
$env:PYTHONPATH = "src"
pytest -q
```

Run the FastAPI app:

```bash
$env:PYTHONPATH = "src"
uvicorn api.main:app --reload
```

Build the serving container:

```bash
docker build -f deployment/Dockerfile -t german-credit-risk-api .
```

## API

- `GET /health`: liveness, model readiness, and explainability status
- `POST /predict`: single applicant inference with SHAP-based local explanations

Example request:

```json
{
  "Age": 35,
  "Sex": "male",
  "Job": 2,
  "Housing": "own",
  "Saving_accounts": "little",
  "Checking_account": "moderate",
  "Credit_amount": 5000,
  "Duration": 24,
  "Purpose": "car"
}
```

## Explainable AI

The serving model uses `XGBClassifier` and returns real-time SHAP values in the `/predict` response. The `explanation` object uses XGBoost native TreeSHAP contributions through `pred_contribs=True`, then returns the method, model family, base value, and top transformed feature drivers ranked by absolute SHAP value. Positive SHAP values increase predicted bad-loan risk; negative values decrease it.

Example response fields include:

```json
{
  "bad_loan_probability": 0.61,
  "predicted_bad": 1,
  "recommended_action": "review_or_decline",
  "explanation": {
    "method": "xgboost_tree_shap_pred_contribs",
    "model_family": "XGBClassifier",
    "top_features": [
      {"feature": "numeric__Duration", "shap_value": 0.22, "impact": "increases_risk"}
    ]
  }
}
```

## Fairness And Compliance

The project includes custom fairness metrics for `Age` and `Sex`:

- demographic parity difference and ratio
- equalized odds difference
- true positive rate by group
- false positive rate by group
- approval rate by group

`Age` is converted into configured age bands for audit reporting. Missing `Saving accounts` and `Checking account` values are explicitly encoded as `No Account` to avoid silently dropping meaningful financial access signals.

## GCP Deployment Path

The repository includes documentation for a GCP deployment workflow:

- stage raw data in BigQuery
- train and version artifacts
- package the FastAPI app with Docker
- push the image to Artifact Registry
- deploy to Cloud Run or Vertex AI custom containers
- monitor drift, model performance, and fairness metrics

This scaffold does not deploy anything automatically.

## Current Status

This is a production-grade scaffold, not a certified production deployment. Before release, run training, review generated metrics, approve the fairness audit, harden environment-specific secrets/configuration, and deploy through a controlled CI/CD workflow.


