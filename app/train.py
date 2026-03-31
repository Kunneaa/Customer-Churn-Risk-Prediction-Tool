from __future__ import annotations

import joblib
import pandas as pd

from app.config import METRICS_FILE, MODEL_BENCHMARK_FILE, MODEL_FILE, TRAIN_FILE
from app.io_utils import ensure_parent_dir, save_json
from app.modeling import train_pipeline


def run_train(mode: str = "full", cv_splits: int | None = None, verbose: bool = True) -> dict[str, object]:
    train_df = pd.read_csv(TRAIN_FILE)
    effective_cv = cv_splits if cv_splits is not None else (3 if mode == "fast" else 5)
    if effective_cv < 2:
        raise ValueError("cv_splits must be >= 2")
    pipeline, metrics = train_pipeline(train_df, mode=mode, cv_splits=effective_cv, verbose=verbose)

    ensure_parent_dir(MODEL_FILE)
    joblib.dump(pipeline, MODEL_FILE)
    save_json(METRICS_FILE, metrics)
    benchmark = {
        "selected_model": metrics.get("selected_model"),
        "cv_roc_auc_mean": metrics.get("cv_roc_auc_mean"),
        "cv_roc_auc_std": metrics.get("cv_roc_auc_std"),
        "cv_ap_mean": metrics.get("cv_ap_mean"),
        "cv_ap_std": metrics.get("cv_ap_std"),
    }
    save_json(MODEL_BENCHMARK_FILE, benchmark)
    return metrics


if __name__ == "__main__":
    print(run_train())
