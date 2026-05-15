from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(cuts) < 3:
        cuts = np.linspace(min(expected.min(), actual.min()), max(expected.max(), actual.max()), buckets + 1)
    expected_counts, _ = np.histogram(expected, bins=cuts)
    actual_counts, _ = np.histogram(actual, bins=cuts)
    expected_share = np.where(expected_counts == 0, 1e-6, expected_counts / expected_counts.sum())
    actual_share = np.where(actual_counts == 0, 1e-6, actual_counts / actual_counts.sum())
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def compare_numeric_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    numeric_columns = [column for column in reference.columns if pd.api.types.is_numeric_dtype(reference[column])]
    for column in numeric_columns:
        if column in current.columns:
            psi = population_stability_index(reference[column].to_numpy(), current[column].to_numpy())
            rows.append({"feature": column, "psi": psi, "drift_flag": "high" if psi >= 0.25 else "moderate" if psi >= 0.1 else "low"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)
