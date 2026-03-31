from __future__ import annotations

import joblib
import pandas as pd

from app.config import (
    CHURN_PROBABILITY_COLUMN,
    ID_COLUMN,
    MODEL_FILE,
    PREDICTIONS_FILE,
    RISK_LEVEL_COLUMN,
    SUBMISSION_FILE,
    TARGET_COLUMN,
    TEST_FILE,
)
from app.features import prepare_scoring_features
from app.io_utils import ensure_parent_dir
from app.risk import map_risk_level, to_binary_label


def predict_with_risk(pipeline, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ID_COLUMN not in test_df.columns:
        raise ValueError(f"Prediction data must include id column: {ID_COLUMN}")

    features = prepare_scoring_features(test_df, pipeline=pipeline)
    probabilities = pd.Series(
        pipeline.predict_proba(features)[:, 1],
        name=CHURN_PROBABILITY_COLUMN,
        index=test_df.index,
    ).clip(0.0, 1.0)
    risk = map_risk_level(probabilities)
    churn_label = to_binary_label(probabilities)

    predictions = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            CHURN_PROBABILITY_COLUMN: probabilities,
            RISK_LEVEL_COLUMN: risk,
            TARGET_COLUMN: churn_label,
        }
    )
    submission = predictions[[ID_COLUMN, TARGET_COLUMN]].copy()
    return predictions, submission


def run_predict() -> tuple[str, str]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    pipeline = joblib.load(MODEL_FILE)
    test_df = pd.read_csv(TEST_FILE)
    out, submission = predict_with_risk(pipeline, test_df)

    ensure_parent_dir(PREDICTIONS_FILE)
    out.to_csv(PREDICTIONS_FILE, index=False)

    ensure_parent_dir(SUBMISSION_FILE)
    submission.to_csv(SUBMISSION_FILE, index=False)

    return str(PREDICTIONS_FILE), str(SUBMISSION_FILE)


if __name__ == "__main__":
    print(run_predict())
