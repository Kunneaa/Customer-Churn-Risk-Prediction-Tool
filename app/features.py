from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_churn_target(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        invalid = series[normalized.isna()].dropna().unique().tolist()
        raise ValueError(f"Unsupported churn target values: {invalid}")
    return normalized.astype(int)


def split_feature_types(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_feature_types(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
