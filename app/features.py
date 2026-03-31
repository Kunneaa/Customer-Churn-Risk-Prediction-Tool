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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip()

    if "tenure" in out.columns:
        out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    if "MonthlyCharges" in out.columns:
        out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if {"MonthlyCharges", "tenure"}.issubset(out.columns):
            out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"] * out["tenure"].fillna(0))

    if "tenure" in out.columns:
        out["tenure_group"] = pd.cut(
            out["tenure"].fillna(-1),
            bins=[-1, 12, 48, 120],
            labels=["new", "mid", "loyal"],
        )

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    existing_service_cols = [c for c in service_cols if c in out.columns]
    if existing_service_cols:
        for col in existing_service_cols:
            out[col] = out[col].fillna("No")
        out["num_services"] = (out[existing_service_cols] == "Yes").sum(axis=1)

    if {"InternetService", "Contract"}.issubset(out.columns):
        out["fiber_month_to_month"] = (
            (out["InternetService"] == "Fiber optic") & (out["Contract"] == "Month-to-month")
        ).astype(int)

    if {"MonthlyCharges", "tenure"}.issubset(out.columns):
        out["monthly_by_tenure"] = out["MonthlyCharges"] / out["tenure"].clip(lower=1)

    return out


def build_preprocessor(features: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    numeric_cols, categorical_cols = split_feature_types(features)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
