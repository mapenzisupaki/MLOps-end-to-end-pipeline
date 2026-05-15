# End-to-End MLOps Production Pipeline

An enterprise-grade, fully automated Machine Learning Operations (MLOps) platform designed to manage the entire data lifecycle of predictive models. This repository implements strict  CI/CD automation, environment isolation, and continuous evaluation systems to bridge the  gap between data science experimentation and scalable production deployment.

## Core Architecture & Lifecycle Stages:

* **Data Engineering & Versioning:** Automated schema enforcement and data lineage tracking.
* **Orchestrated Pipelines:** Structured step execution decoupling feature engineering from training.
* **Model Registry:** Centralized artifact versioning, audit logging, and stage transitions.
* **Containerized Deployment:** High-performance APIs wrapped in isolated runtimes.
* **Production Observability:** Active telemetry logging to catch live data and concept drift.
## Projects

| Project | Description | Preview and Run Instructions |
| --- | --- | --- |
| [`german_credit_risk_platform`](german_credit_risk_platform/) | German Credit Risk ML/MLOps platform with FastAPI scoring, XGBoost SHAP explanations, fairness auditing, Streamlit dashboard, Docker, tests, and GCP deployment notes. | See [`german_credit_risk_platform/README.md`](german_credit_risk_platform/README.md#hiring-manager-preview-links). |

## Repository Layout

This repository is designed to hold multiple MLOps projects. GitHub Actions workflows live at the repository root in `.github/workflows/`, while each project lives in its own folder. Project-specific workflows should use path filters so future projects do not trigger unrelated CI jobs.

