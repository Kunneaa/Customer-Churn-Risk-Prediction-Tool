from __future__ import annotations

import json
import time
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.config import PARALLEL_JOBS, RANDOM_STATE, SELECTED_FEATURES_FILE
from app.features import build_preprocessor, prepare_training_data


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)


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


def _candidate_models(mode: str = "full") -> list[tuple[str, object, bool]]:
    specs: list[tuple[str, object, bool]] = [
        ("LogisticRegression", build_classifier(), True),
        ("DecisionTree", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6, min_samples_leaf=20), False),
        (
            "RandomForest",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=PARALLEL_JOBS,
            ),
            False,
        ),
        ("GaussianNB", GaussianNB(), True),
        (
            "SVM_Linear",
            SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        (
            "SVM_Poly",
            SVC(kernel="poly", degree=3, C=1.0, probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        ("KNN", KNeighborsClassifier(n_neighbors=15, weights="distance"), True),
        ("AdaBoost", AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=300, learning_rate=0.05), False),
    ]

    if mode == "fast":
        keep = {"LogisticRegression", "RandomForest", "AdaBoost"}
        return [spec for spec in specs if spec[0] in keep]

    try:
        from xgboost import XGBClassifier

        specs.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=PARALLEL_JOBS,
                ),
                False,
            )
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        specs.append(
            (
                "CatBoost",
                CatBoostClassifier(
                    loss_function="Logloss",
                    depth=6,
                    learning_rate=0.05,
                    n_estimators=500,
                    random_seed=RANDOM_STATE,
                    verbose=False,
                ),
                False,
            )
        )
    except ImportError:
        pass

    return specs


def _filter_selected_columns(X: pd.DataFrame, selected_features: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    if not selected_features:
        return X.copy(), X.columns.tolist()

    kept = [c for c in selected_features if c in X.columns]
    if not kept:
        return X.copy(), X.columns.tolist()
    return X[kept].copy(), kept


def select_model_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return _filter_selected_columns(X, load_selected_features())


def fit_best_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    mode: str = "full",
    cv_splits: int = 5,
    verbose: bool = True,
) -> Tuple[Pipeline, Dict[str, object]]:
    if cv_splits < 2:
        raise ValueError("cv_splits must be >= 2")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    model_specs = _candidate_models(mode=mode)
    if verbose:
        print(f"Benchmark mode={mode}, cv_splits={cv_splits}, candidates={len(model_specs)}")

    best_name = ""
    best_pipeline: Pipeline | None = None
    best_cv_auc = -np.inf
    best_cv_ap = -np.inf
    best_auc_std = 0.0
    best_ap_std = 0.0

    for idx, (name, estimator, scale_numeric) in enumerate(model_specs, start=1):
        if verbose:
            print(f"[{idx}/{len(model_specs)}] Evaluating {name}...")

        pipeline_i = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
                ("model", estimator),
            ]
        )

        started = time.perf_counter()
        cv_result = cross_validate(
            pipeline_i,
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

        if verbose:
            print(
                f"[{idx}/{len(model_specs)}] {name}: "
                f"AUC={auc_mean:.5f} (+/- {float(np.std(auc_scores)):.5f}), "
                f"AP={ap_mean:.5f}, time={elapsed:.1f}s"
            )

        if auc_mean > best_cv_auc:
            best_cv_auc = auc_mean
            best_cv_ap = ap_mean
            best_auc_std = float(np.std(auc_scores))
            best_ap_std = float(np.std(ap_scores))
            best_name = name
            best_pipeline = pipeline_i
            if verbose:
                print(f"New best model: {best_name} (AUC={best_cv_auc:.5f})")

    if best_pipeline is None:
        raise RuntimeError("No model candidate was successfully evaluated")

    best_pipeline.fit(X, y)
    train_score = best_pipeline.predict_proba(X)[:, 1]
    metrics = {
        "selected_model": best_name,
        "train_roc_auc": float(roc_auc_score(y, train_score)),
        "train_average_precision": float(average_precision_score(y, train_score)),
        "cv_roc_auc_mean": best_cv_auc,
        "cv_roc_auc_std": best_auc_std,
        "cv_ap_mean": best_cv_ap,
        "cv_ap_std": best_ap_std,
        "n_features_used": int(X.shape[1]),
        "n_candidates": int(len(model_specs)),
        "train_mode": mode,
        "cv_splits": int(cv_splits),
    }
    return best_pipeline, metrics


def train_pipeline(
    train_df: pd.DataFrame,
    mode: str = "full",
    cv_splits: int = 5,
    verbose: bool = True,
) -> Tuple[Pipeline, Dict[str, object]]:
    X_all, y = prepare_training_data(train_df)
    X, _ = select_model_features(X_all)
    return fit_best_pipeline(X, y, mode=mode, cv_splits=cv_splits, verbose=verbose)
