from __future__ import annotations

import joblib
import pandas as pd

from app.config import METRICS_FILE, MODEL_FILE, TRAIN_FILE
from app.io_utils import ensure_parent_dir, save_json
from app.modeling import train_pipeline


def run_train() -> dict[str, float]:
    train_df = pd.read_csv(TRAIN_FILE)
    pipeline, metrics = train_pipeline(train_df)

    ensure_parent_dir(MODEL_FILE)
    joblib.dump(pipeline, MODEL_FILE)
    save_json(METRICS_FILE, metrics)
    return metrics


if __name__ == "__main__":
    print(run_train())
