# Customer Churn Risk Prediction Tool

End-to-end machine learning project for customer churn prediction and business risk segmentation.

## 1. Problem Framing

### Business Problem
Customer churn is a critical issue in subscription-based businesses (especially telecommunications). Losing customers directly impacts revenue, customer lifetime value, and long-term growth.

### Objective
- Predict the probability of churn for each customer.
- Segment customers into risk levels:
  - Low Risk
  - Medium Risk
  - High Risk
- Support retention decisions with an interpretable risk ranking.

## 2. Machine Learning Formulation

- Task: Binary classification.
- Target: `Churn` (`0` = stay, `1` = leave).
- Model output:
  - Churn probability.
  - Risk category (Low / Medium / High).

### Evaluation Metrics
- Primary metric: ROC-AUC.
- Secondary metric: Precision-Recall.

## 3. About The Dataset

The dataset is synthetically generated from a deep learning model trained on a real-world churn dataset.

### Characteristics
- Similar to real-world data but not identical.
- Helps reduce direct leakage from production records.
- Suitable for rapid experimentation and model comparison.

### Considerations
- May contain distribution shifts versus real production data.
- Strong offline scores may not fully generalize to live environments.

## 4. Project Pipeline

```text
Data -> EDA -> Feature Engineering -> Model Training -> Evaluation -> Prediction -> Risk Mapping -> Output
```

## 5. Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- app/
|   |-- __init__.py
|   |-- __main__.py
|   |-- config.py
|   |-- features.py
|   |-- modeling.py
|   |-- risk.py
|   |-- io_utils.py
|   |-- train.py
|   |-- predict.py
|   |-- web_app.py
|   `-- main.py
|-- data/
|   |-- train.csv
|   `-- test.csv
|-- docker/
|   |-- Dockerfile
|   `-- docker-compose.yml
|-- models/                  # auto-created after training
|   |-- churn_model.joblib
|   `-- metrics.json
|-- output/                  # auto-created after prediction
|   |-- predictions_with_risk.csv
|   `-- submission.csv
|-- notebooks/                 
|   |-- problem_framing.ipynb
|   |-- eda.ipynb
|   |-- feature_engineering.ipynb
|   |-- model_selection.ipynb
|   `-- evaluation.ipynb
```

## 5.1 Detailed File-by-File Explanation

### Root Files
- `README.md`
  - Project documentation: business context, ML framing, pipeline, setup, and usage.
  - First file to read when onboarding new contributors.
- `requirements.txt`
  - Single dependency file for local run, development, and Docker build.
- `.gitignore`
  - Prevents committing local artifacts and generated outputs such as virtual environments, model binaries, and prediction files.

### Application Package (`app/`)
- `app/__init__.py`
  - Marks `app` as a Python package.
  - Keeps imports clean when running modules with `python -m`.
- `app/config.py`
  - Central configuration file for paths, column names, random seed, and risk thresholds.
  - Single source of truth for constants used across training and prediction.
- `app/features.py`
  - Feature preprocessing utilities.
  - Normalizes the churn target to numeric labels and builds the sklearn preprocessing pipeline (imputation, scaling, one-hot encoding).
- `app/modeling.py`
  - Core modeling logic.
  - Defines baseline classifier, train/validation split, model fitting, and evaluation metrics (ROC-AUC, Average Precision).
- `app/risk.py`
  - Converts churn probabilities into business-friendly risk levels (Low/Medium/High).
  - Also provides binary conversion using a configurable decision threshold.
- `app/io_utils.py`
  - Utility functions for safe output writing.
  - Creates parent directories and saves JSON artifacts (for example model metrics).
- `app/train.py`
  - Training entry module.
  - Reads training data, fits the pipeline, saves trained model and validation metrics.
- `app/predict.py`
  - Prediction entry module.
  - Loads trained model, predicts churn probability on test data, maps risk levels, and exports final CSV outputs.
- `app/main.py`
  - Unified command-line entrypoint.
  - Supports stage-based execution: `train`, `predict`, or `all`.
- `app/__main__.py`
  - Enables shorter command `python -m app`.
- `app/web_app.py`
  - Streamlit web UI for file upload, analytics, training, ROC curve, and CSV export.

### Data Files (`data/`)
- `data/train.csv`
  - Labeled training dataset containing features and the `Churn` target.
  - Used for model training and validation.
- `data/test.csv`
  - Unlabeled dataset for inference.
  - Used to generate churn probabilities and risk categories.

### Containerization (`docker/`)
- `docker/Dockerfile`
  - Defines container image for reproducible execution.
  - Installs dependencies and runs pipeline from the `app` package.
- `docker/docker-compose.yml`
  - Local orchestration file to build and run the project with one command.
  - Mounts project folder for easier iterative development.

### Generated Artifacts
- `models/churn_model.joblib`
  - Serialized trained sklearn pipeline.
  - Required by prediction stage.
- `models/churn_model_train_test_only.joblib`
  - Optional model artifact generated by notebook workflow.
- `models/metrics.json`
  - Validation metrics saved after training.
  - Useful for quick experiment tracking.
- `output/predictions_with_risk.csv`
  - Business-friendly prediction output with `id`, churn probability, risk level, and binary churn label.
- `output/submission.csv`
  - Minimal submission-style output (`id`, `Churn`).

### Analysis Notebooks (`notebooks/`)
- `notebooks/problem_framing.ipynb`
  - Defines business objective, constraints, and success metrics.
- `notebooks/eda.ipynb`
  - Exploratory data analysis: distributions, missing data, and churn patterns.
- `notebooks/feature_engineering.ipynb`
  - Experiments with handcrafted features and transformations.
- `notebooks/model_selection.ipynb`
  - Compares candidate models and tuning strategies.
- `notebooks/evaluation.ipynb`
  - Final evaluation, error analysis, and interpretation of model performance.

## 6. Suggested Workflow

1. Define scope and assumptions in `problem_framing.ipynb`.
2. Analyze distributions and churn patterns in `eda.ipynb`.
3. Build candidate features in `feature_engineering.ipynb`.
4. Train and compare models in `model_selection.ipynb`.
5. Evaluate final model and generate output in `evaluation.ipynb`.
6. Run production-like pipeline from `app/main.py` for reproducible artifacts.

## 7. Environment Setup

Requirements:
- Python 3.10+.
- pip or conda.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run end-to-end pipeline:

```bash
python -m app.main --stage all
```

Equivalent shorter command:

```bash
python -m app --stage all
```

Run only training:

```bash
python -m app.main --stage train
```

Run only prediction (requires existing model):

```bash
python -m app.main --stage predict
```

Docker execution:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Run only pipeline service:

```bash
docker compose -f docker/docker-compose.yml up --build churn-risk-pipeline
```

Run only web service:

```bash
docker compose -f docker/docker-compose.yml up --build churn-risk-web
```

Deploy note:
- Docker uses `requirements.txt`.

## 7.1 Web Application (Streamlit)

The web tool supports:
- Upload `train.csv` and `test.csv`.
- Deep data analytics: missing values, min/max statistics, distribution tables, and outlier detection.
- Model training with ROC-AUC evaluation.
- Export `submission.csv` and `predictions_with_risk.csv`.

Run locally:

```bash
streamlit run app/web_app.py
```

Run with Docker Compose (web service):

```bash
docker compose -f docker/docker-compose.yml up --build churn-risk-web
```

Then open: `http://localhost:8501`

## 8. Feature Engineering Direction

- Behavioral features:
  - `tenure_group` (new, mid, loyal)
  - `num_services` (total subscribed services)
- Interaction features:
  - `Contract x InternetService`
  - `MonthlyCharges x PaymentMethod`
  - `FiberOptic x MonthToMonth`
- Risk flags:
  - High-risk behavior patterns discovered during EDA.

## 9. Modeling Direction

- Baseline: Logistic Regression.
- Current implementation in `app/`: Logistic Regression with class balancing.
- Current validation in `app/`: Stratified train/validation holdout split (`train_test_split`).
- Notebooks can be used to experiment with advanced models (XGBoost, LightGBM, CatBoost) as optional extensions.
- Model selection focus:
  - Best ROC-AUC.
  - Stable Precision-Recall on high-risk segments.

## 10. Business Output

Expected output for each customer:
- Predicted churn probability.
- Assigned risk level (Low / Medium / High).
- Ranked list for retention campaign prioritization.
