from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
TRAIN_FE_FILE = DATA_DIR / "train_fe.csv"
TEST_FE_FILE = DATA_DIR / "test_fe.csv"

MODEL_FILE = MODELS_DIR / "churn_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
MODEL_BENCHMARK_FILE = MODELS_DIR / "model_benchmark.json"
SELECTED_FEATURES_FILE = MODELS_DIR / "selected_features_gsa.json"

PREDICTIONS_FILE = OUTPUT_DIR / "predictions_with_risk.csv"
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

TARGET_COLUMN = "Churn"
ID_COLUMN = "id"

RANDOM_STATE = 42
TEST_SIZE = 0.2

LOW_RISK_THRESHOLD = 0.33
HIGH_RISK_THRESHOLD = 0.66
CLASSIFICATION_THRESHOLD = 0.5
