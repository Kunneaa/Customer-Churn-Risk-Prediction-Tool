from __future__ import annotations

import argparse

from app.predict import run_predict
from app.train import run_train

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer churn risk prediction pipeline")
    parser.add_argument(
        "--stage",
        choices=["train", "predict", "all"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--train-mode",
        choices=["full", "fast"],
        default="full",
        help="Training benchmark mode: full (all models) or fast (subset of models)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        help="Override CV folds used in training benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in {"train", "all"}:
        metrics = run_train(mode=args.train_mode, cv_splits=args.cv_splits)
        print("Validation metrics:")
        print(metrics)

    if args.stage in {"predict", "all"}:
        prediction_path, submission_path = run_predict()
        print(f"Predictions: {prediction_path}")
        print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
