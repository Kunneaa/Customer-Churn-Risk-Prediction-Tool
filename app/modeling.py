from __future__ import annotations

import json
import time
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.config import MODEL_NAME, PARALLEL_JOBS, RANDOM_STATE, SELECTED_FEATURES_FILE
from app.features import build_preprocessor, prepare_training_data


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=PARALLEL_JOBS,
    )


def load_selected_features() -> list[str]:
    if not SELECTED_FEATURES_FILE.exists():
        return []

    with SELECTED_FEATURES_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    selected = payload.get("selected_features_raw") or payload.get("selected_features") or []
    return [str(c) for c in selected]


def evaluate_scores(y_true: pd.Series, y_score: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def _filter_selected_columns(X: pd.DataFrame, selected_features: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    if not selected_features:
        return X.copy(), X.columns.tolist()

    kept = [c for c in selected_features if c in X.columns]
    if not kept:
        return X.copy(), X.columns.tolist()
    return X[kept].copy(), kept


def select_model_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return _filter_selected_columns(X, load_selected_features())


def fit_random_forest_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 3,
    verbose: bool = True,
) -> Tuple[Pipeline, Dict[str, object]]:
    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=False)),
            ("model", build_model()),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    started = time.perf_counter()
    cv_result = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={"auc": "roc_auc", "ap": "average_precision"},
        n_jobs=PARALLEL_JOBS,
        return_train_score=False,
    )
    elapsed = time.perf_counter() - started

    auc_scores = cv_result["test_auc"]
    ap_scores = cv_result["test_ap"]
    auc_mean = float(np.mean(auc_scores))
    ap_mean = float(np.mean(ap_scores))
    auc_std = float(np.std(auc_scores))
    ap_std = float(np.std(ap_scores))

    if verbose:
        print(
            f"{MODEL_NAME}: AUC={auc_mean:.5f} (+/- {auc_std:.5f}), "
            f"AP={ap_mean:.5f}, time={elapsed:.1f}s, cv_splits={cv_splits}"
        )

    pipeline.fit(X, y)
    train_score = pipeline.predict_proba(X)[:, 1]
    metrics = {
        "selected_model": MODEL_NAME,
        "train_roc_auc": float(roc_auc_score(y, train_score)),
        "train_average_precision": float(average_precision_score(y, train_score)),
        "cv_roc_auc_mean": auc_mean,
        "cv_roc_auc_std": auc_std,
        "cv_ap_mean": ap_mean,
        "cv_ap_std": ap_std,
        "n_features_used": int(X.shape[1]),
        "cv_splits": int(cv_splits),
    }
    return pipeline, metrics


def train_pipeline(
    train_df: pd.DataFrame,
    cv_splits: int = 3,
    verbose: bool = True,
) -> Tuple[Pipeline, Dict[str, object]]:
    X_all, y = prepare_training_data(train_df)
    X, _ = select_model_features(X_all)
    return fit_random_forest_pipeline(X, y, cv_splits=cv_splits, verbose=verbose)
