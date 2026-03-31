from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from app.config import ID_COLUMN, MODEL_FILE, PREDICTIONS_FILE, SUBMISSION_FILE, TEST_FILE
from app.features import add_features
from app.io_utils import ensure_parent_dir
from app.modeling import prepare_features
from app.risk import map_risk_level, to_binary_label


def _align_with_trained_features(pipeline, features: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipeline.named_steps.get("preprocessor")
    expected = list(getattr(preprocessor, "feature_names_in_", [])) if preprocessor is not None else []
    if not expected:
        return features

    aligned = features.copy()
    for col in expected:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned.reindex(columns=expected)


def run_predict() -> tuple[str, str]:
    pipeline = joblib.load(MODEL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    test_fe = add_features(test_df)
    features = prepare_features(test_fe)
    features = _align_with_trained_features(pipeline, features)
    prob = pd.Series(pipeline.predict_proba(features)[:, 1], name="churn_probability", index=test_df.index).clip(0.0, 1.0)
    risk = map_risk_level(prob)
    churn_label = to_binary_label(prob)

    out = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            "churn_probability": prob,
            "risk_level": risk,
            "Churn": churn_label,
        }
    )

    ensure_parent_dir(PREDICTIONS_FILE)
    out.to_csv(PREDICTIONS_FILE, index=False)

    submission = out[[ID_COLUMN, "Churn"]].copy()
    ensure_parent_dir(SUBMISSION_FILE)
    submission.to_csv(SUBMISSION_FILE, index=False)

    return str(PREDICTIONS_FILE), str(SUBMISSION_FILE)


if __name__ == "__main__":
    print(run_predict())
