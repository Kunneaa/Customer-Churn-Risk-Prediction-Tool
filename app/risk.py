from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import CLASSIFICATION_THRESHOLD, HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD


def map_risk_level(
    churn_probability: pd.Series,
    low_threshold: float = LOW_RISK_THRESHOLD,
    high_threshold: float = HIGH_RISK_THRESHOLD,
) -> pd.Series:
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("Thresholds must satisfy 0 <= low < high <= 1")

    conditions = [
        churn_probability < low_threshold,
        (churn_probability >= low_threshold) & (churn_probability < high_threshold),
        churn_probability >= high_threshold,
    ]
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    return pd.Series(np.select(conditions, labels, default="Medium Risk"), index=churn_probability.index)


def to_binary_label(churn_probability: pd.Series, threshold: float = CLASSIFICATION_THRESHOLD) -> pd.Series:
    return (churn_probability >= threshold).astype(int)
