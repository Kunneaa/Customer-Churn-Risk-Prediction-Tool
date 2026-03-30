from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from app.features import build_preprocessor, normalize_churn_target


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if TARGET_COLUMN in out.columns:
        out = out.drop(columns=[TARGET_COLUMN])
    if ID_COLUMN in out.columns:
        out = out.drop(columns=[ID_COLUMN])
    return out


def evaluate_scores(y_true: pd.Series, y_score: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def train_pipeline(train_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    y = normalize_churn_target(train_df[TARGET_COLUMN])
    X = prepare_features(train_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", build_classifier()),
        ]
    )
    pipeline.fit(X_train, y_train)

    valid_score = pipeline.predict_proba(X_valid)[:, 1]
    metrics = evaluate_scores(y_valid, valid_score)
    return pipeline, metrics
