from __future__ import annotations

import joblib
import pandas as pd

from app.config import ID_COLUMN, MODEL_FILE, PREDICTIONS_FILE, SUBMISSION_FILE, TEST_FILE
from app.io_utils import ensure_parent_dir
from app.modeling import prepare_features
from app.risk import map_risk_level, to_binary_label


def run_predict() -> tuple[str, str]:
    pipeline = joblib.load(MODEL_FILE)
    test_df = pd.read_csv(TEST_FILE)

    features = prepare_features(test_df)
    prob = pd.Series(pipeline.predict_proba(features)[:, 1], name="churn_probability", index=test_df.index)
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
