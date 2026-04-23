from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.core.config import DEFAULT_THRESHOLD, RISK_THRESHOLDS
from src.features import build_feature_vector


APPROVE_DECISION = "Одобрить"
DECLINE_DECISION = "Отказать"


def risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "LOW"
    if probability < RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "HIGH"


def decision_from_probability(
    probability: float, threshold: float = DEFAULT_THRESHOLD
) -> str:
    return DECLINE_DECISION if probability >= threshold else APPROVE_DECISION


def predict_default(
    model: Any,
    values: Mapping[str, Any],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    features = build_feature_vector(values)
    probability = float(model.predict_proba(features)[0][1])
    return {
        "default_probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "decision": decision_from_probability(probability, threshold),
        "threshold_used": threshold,
    }


def raw_probability(model: Any, values: Mapping[str, Any]) -> tuple[float, np.ndarray]:
    features = build_feature_vector(values)
    return float(model.predict_proba(features)[0][1]), features
