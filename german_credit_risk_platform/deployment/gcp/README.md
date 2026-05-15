# GCP Deployment Notes

## BigQuery staging

Example table pattern:

```sql
CREATE OR REPLACE TABLE `PROJECT_ID.credit_risk.german_credit_raw` AS
SELECT * FROM `PROJECT_ID.credit_risk_staging.german_credit_upload`;
```

## Artifact Registry

```bash
gcloud artifacts repositories create credit-risk --repository-format=docker --location=REGION

gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/credit-risk/german-credit-risk-api:latest -f deployment/Dockerfile .
```

## Cloud Run

```bash
gcloud run deploy german-credit-risk-api \
  --image REGION-docker.pkg.dev/PROJECT_ID/credit-risk/german-credit-risk-api:latest \
  --region REGION \
  --platform managed
```

## Vertex AI

Use the same container image as a custom prediction container after adapting model artifact loading to the selected model registry path.
```
