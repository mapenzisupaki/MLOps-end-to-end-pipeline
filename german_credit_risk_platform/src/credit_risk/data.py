from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import PROJECT_ROOT, load_json_config

DATA_CONFIG = load_json_config("data_config.json")
MODEL_CONFIG = load_json_config("model_config.json")


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    data_path = PROJECT_ROOT / (path or DATA_CONFIG["data_path"])
    return pd.read_csv(data_path)


def normalize_account_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    missing_label = DATA_CONFIG["missing_account_label"]
    for column in ["Saving accounts", "Checking account"]:
        if column in result.columns:
            result[column] = result[column].fillna(missing_label).astype(str)
    return result


def prepare_modeling_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    result = normalize_account_columns(df)
    id_column = DATA_CONFIG["id_column"]
    target_column = DATA_CONFIG["target_column"]
    if id_column in result.columns:
        result = result.drop(columns=[id_column])
    y = result[target_column].str.lower().map({"good": 0, "bad": 1})
    if y.isna().any():
        raise ValueError("Target column contains values outside {'good', 'bad'}.")
    X = result.drop(columns=[target_column])
    return X, y.astype(int)


def split_data(X: pd.DataFrame, y: pd.Series) -> DataSplit:
    random_state = MODEL_CONFIG["random_state"]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=MODEL_CONFIG["test_size"],
        stratify=y,
        random_state=random_state,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=MODEL_CONFIG["validation_size"],
        stratify=y_train_full,
        random_state=random_state,
    )
    return DataSplit(X_train, X_valid, X_test, y_train, y_valid, y_test)
