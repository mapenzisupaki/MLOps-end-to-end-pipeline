from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = MODELS_DIR / "credit_risk_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FAIRNESS_REPORT_PATH = ARTIFACTS_DIR / "fairness_report.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "decision_threshold.json"


def load_json_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
