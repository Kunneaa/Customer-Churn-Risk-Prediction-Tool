from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import ID_COLUMN, TARGET_COLUMN


SERVICE_COLUMNS = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
NUMERIC_FEATURE_COLUMNS = ("tenure", "MonthlyCharges", "TotalCharges")
TARGET_MAPPING: dict[str, int] = {
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0,
}


def normalize_churn_target(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    if normalized.isna().any():
        raise ValueError("Churn target contains missing values.")

    mapped = normalized.map(TARGET_MAPPING)
    if mapped.isna().any():
        invalid = sorted({str(value) for value in normalized[mapped.isna()].dropna().unique().tolist()})
        raise ValueError(f"Unsupported churn target values: {invalid}")
    return mapped.astype(int)


def split_feature_types(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def _strip_text_columns(df: pd.DataFrame) -> None:
    for col in df.select_dtypes(include=["object", "string"]).columns:
        cleaned = df[col].astype("string").str.strip()
        df[col] = cleaned.mask(cleaned == "", pd.NA)


def _coerce_numeric_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _strip_text_columns(out)
    _coerce_numeric_columns(out, NUMERIC_FEATURE_COLUMNS)

    if "TotalCharges" in out.columns and {"MonthlyCharges", "tenure"}.issubset(out.columns):
        out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"] * out["tenure"].fillna(0))

    if "tenure" in out.columns:
        out["tenure_group"] = pd.cut(
            out["tenure"],
            bins=[-np.inf, 12, 48, np.inf],
            labels=["new", "mid", "loyal"],
            include_lowest=True,
        )

    existing_service_cols = [c for c in SERVICE_COLUMNS if c in out.columns]
    if existing_service_cols:
        for col in existing_service_cols:
            out[col] = out[col].fillna("No")
        out["num_services"] = out[existing_service_cols].eq("Yes").sum(axis=1)

    if {"InternetService", "Contract"}.issubset(out.columns):
        out["fiber_month_to_month"] = (
            (out["InternetService"] == "Fiber optic") & (out["Contract"] == "Month-to-month")
        ).astype(int)

    if {"MonthlyCharges", "tenure"}.issubset(out.columns):
        out["monthly_by_tenure"] = out["MonthlyCharges"] / out["tenure"].clip(lower=1)

    return out


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore").copy()


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    featured = add_features(df)
    if TARGET_COLUMN not in featured.columns:
        raise ValueError(f"Training data must include target column: {TARGET_COLUMN}")
    return prepare_model_features(featured), normalize_churn_target(featured[TARGET_COLUMN])


def align_features_to_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    fill_value: object = np.nan,
) -> pd.DataFrame:
    preprocessor = pipeline.named_steps.get("preprocessor")
    expected = list(getattr(preprocessor, "feature_names_in_", [])) if preprocessor is not None else []
    if not expected:
        return features.copy()

    aligned = features.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = fill_value
    return aligned.reindex(columns=expected)


def prepare_scoring_features(df: pd.DataFrame, pipeline: Pipeline | None = None) -> pd.DataFrame:
    features = prepare_model_features(add_features(df))
    if pipeline is None:
        return features
    return align_features_to_pipeline(pipeline, features)


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
