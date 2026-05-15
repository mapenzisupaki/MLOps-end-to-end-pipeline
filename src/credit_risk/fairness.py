from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import load_json_config

DATA_CONFIG = load_json_config("data_config.json")


def make_age_group(age: pd.Series) -> pd.Series:
    return pd.cut(
        age,
        bins=DATA_CONFIG["age_bins"],
        labels=DATA_CONFIG["age_labels"],
        right=False,
        include_lowest=True,
    ).astype(str)


def group_rates(y_true: pd.Series, y_pred: np.ndarray, group: pd.Series) -> list[dict[str, float | str | int]]:
    frame = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred, "group": group.values})
    rows: list[dict[str, float | str | int]] = []
    for group_value, part in frame.groupby("group", dropna=False):
        positives = part[part["y_true"] == 1]
        negatives = part[part["y_true"] == 0]
        approval_rate = float((part["y_pred"] == 0).mean())
        true_positive_rate = float((positives["y_pred"] == 1).mean()) if len(positives) else 0.0
        false_positive_rate = float((negatives["y_pred"] == 1).mean()) if len(negatives) else 0.0
        rows.append(
            {
                "group": str(group_value),
                "n": int(len(part)),
                "approval_rate": approval_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
            }
        )
    return rows


def fairness_summary(y_true: pd.Series, y_probability: np.ndarray, X: pd.DataFrame, threshold: float) -> dict[str, object]:
    y_pred = (y_probability >= threshold).astype(int)
    protected_groups = {"Sex": X["Sex"].astype(str), "Age": make_age_group(X["Age"])}
    report: dict[str, object] = {}
    for attribute, group in protected_groups.items():
        rates = group_rates(y_true, y_pred, group)
        approval_rates = [float(row["approval_rate"]) for row in rates]
        tpr_values = [float(row["true_positive_rate"]) for row in rates]
        fpr_values = [float(row["false_positive_rate"]) for row in rates]
        min_approval = min(approval_rates) if approval_rates else 0.0
        max_approval = max(approval_rates) if approval_rates else 0.0
        odds_gap = max(max(tpr_values) - min(tpr_values), max(fpr_values) - min(fpr_values)) if tpr_values and fpr_values else 0.0
        report[attribute] = {
            "groups": rates,
            "demographic_parity_difference": float(max_approval - min_approval),
            "demographic_parity_ratio": float(min_approval / max_approval) if max_approval else 0.0,
            "equalized_odds_difference": float(odds_gap),
        }
    return report
