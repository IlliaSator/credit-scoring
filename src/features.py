from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.core.config import FEATURE_NAMES, ID_COLUMN, RAW_FEATURES, TARGET_COLUMN


API_TO_DATASET_COLUMNS = {
    "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse",
}


def normalize_feature_names(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert API-safe field names to the original dataset feature names."""
    normalized = dict(values)
    for api_name, dataset_name in API_TO_DATASET_COLUMNS.items():
        if api_name in normalized:
            normalized[dataset_name] = normalized.pop(api_name)
    return normalized


def load_training_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if ID_COLUMN in data.columns:
        data = data.drop(columns=[ID_COLUMN])
    return data


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if TARGET_COLUMN in frame.columns:
        frame = frame.drop(columns=[TARGET_COLUMN])
    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])

    frame = frame.rename(columns=API_TO_DATASET_COLUMNS)
    frame["RevolvingUtilizationOfUnsecuredLines"] = frame[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(0, 1)
    frame["MonthlyIncome"] = frame["MonthlyIncome"].fillna(
        frame["MonthlyIncome"].median()
    )
    frame["NumberOfDependents"] = frame["NumberOfDependents"].fillna(0)

    frame["TotalLatePayments"] = (
        frame["NumberOfTime30-59DaysPastDueNotWorse"]
        + frame["NumberOfTimes90DaysLate"]
        + frame["NumberOfTime60-89DaysPastDueNotWorse"]
    )
    frame["HasAnyLatePayment"] = (frame["TotalLatePayments"] > 0).astype(int)
    frame["DebtPerDependent"] = frame["DebtRatio"] / (
        frame["NumberOfDependents"] + 1
    )
    frame["IsYoung"] = (frame["age"] < 35).astype(int)
    frame["LoansPerIncome"] = frame["NumberOfOpenCreditLinesAndLoans"] / (
        frame["MonthlyIncome"] + 1
    )

    return frame[FEATURE_NAMES]


def build_feature_vector(values: Mapping[str, Any]) -> np.ndarray:
    normalized = normalize_feature_names(values)
    frame = pd.DataFrame([normalized], columns=RAW_FEATURES)
    return prepare_features(frame).to_numpy(dtype=float)


def extract_target(data: pd.DataFrame) -> pd.Series:
    return data[TARGET_COLUMN].astype(int)
