# Customer Churn Risk Prediction Tool

End-to-end machine learning project for telecom churn prediction, risk segmentation, batch scoring, and lightweight Streamlit-based analysis.

## Overview

This repository covers the full churn workflow:

- Train a churn classifier from `data/train.csv`
- Benchmark multiple model families in CLI training mode
- Score `data/test.csv`
- Convert churn probability into `Low Risk`, `Medium Risk`, and `High Risk`
- Export business-ready CSV outputs
- Explore datasets and generate predictions through a Streamlit UI

## What The Project Produces

After a successful run, the project generates:

- `models/churn_model.joblib`: trained sklearn pipeline
- `models/metrics.json`: training and cross-validation summary
- `models/model_benchmark.json`: reduced benchmark snapshot
- `output/predictions_with_risk.csv`: `id`, churn probability, risk level, binary label
- `output/submission.csv`: `id`, `Churn`

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- streamlit_app.py
|-- app/
|   |-- __init__.py
|   |-- __main__.py
|   |-- config.py
|   |-- features.py
|   |-- io_utils.py
|   |-- main.py
|   |-- modeling.py
|   |-- predict.py
|   |-- risk.py
|   |-- streamlit_app.py
|   |-- train.py
|   `-- web_app.py
|-- data/
|   |-- train.csv
|   `-- test.csv
|-- docker/
|   |-- Dockerfile
|   `-- docker-compose.yml
|-- models/
|   |-- churn_model.joblib
|   |-- churn_model_train_test_only.joblib
|   |-- metrics.json
|   `-- model_benchmark.json
`-- output/
    |-- predictions_with_risk.csv
    `-- submission.csv
```

## Dataset Snapshot

The current checked-in CSV files have the following shape:

- `train.csv`: `594,194` rows x `21` columns
- `test.csv`: `254,655` rows x `20` columns
- Target column: `Churn`
- Identifier column: `id`

Core feature groups in the dataset:

- Customer profile: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Account tenure and billing: `tenure`, `MonthlyCharges`, `TotalCharges`
- Contract and billing preferences: `Contract`, `PaperlessBilling`, `PaymentMethod`
- Service subscriptions: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

Observed target distribution in the current train set:

- `No`: `460,377`
- `Yes`: `133,817`

## Pipeline

### 1. Feature Engineering

Implemented in `app/features.py`:

- strips whitespace from object columns
- coerces `tenure`, `MonthlyCharges`, and `TotalCharges` to numeric
- fills missing `TotalCharges` with `MonthlyCharges * tenure` when possible
- adds `tenure_group`
- adds `num_services`
- adds `fiber_month_to_month`
- adds `monthly_by_tenure`

### 2. Preprocessing

Implemented with sklearn transformers:

- numeric columns: median imputation and optional scaling
- categorical columns: most-frequent imputation and one-hot encoding

### 3. Training

CLI training is handled by `app/train.py` and `app/modeling.py`.

Supported modes:

- `full`: benchmark the full candidate set
- `fast`: benchmark a smaller subset for quicker iteration

The training benchmark selects the best model by mean cross-validated ROC-AUC, then fits that model on the full training set and saves the pipeline artifact.

### 4. Prediction

Prediction is handled by `app/predict.py`:

- load saved pipeline
- align test columns with trained features
- generate churn probability
- map probability to risk band
- export scored outputs

### 5. Risk Mapping

Implemented in `app/risk.py`:

- `Low Risk`: probability `< 0.33`
- `Medium Risk`: probability `>= 0.33` and `< 0.66`
- `High Risk`: probability `>= 0.66`
- binary churn label threshold: `0.50`

## Current Model Snapshot

Based on the checked-in `models/metrics.json`, the latest saved run in this repository reports:

- selected model: `RandomForest`
- train mode: `fast`
- CV folds: `3`
- mean CV ROC-AUC: `0.9122`
- mean CV average precision: `0.7418`

Treat these numbers as the latest saved artifact in the repo, not as a guaranteed result for every future run.

## Local Setup

Requirements:

- Python `3.10+`
- `pip`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## CLI Usage

Run the full pipeline:

```bash
python -m app --stage all
```

Run the full pipeline in quicker benchmark mode:

```bash
python -m app --stage all --train-mode fast
```

Run training only:

```bash
python -m app --stage train --train-mode fast
```

Run prediction only using an existing model artifact:

```bash
python -m app --stage predict
```

Optional training flag:

```bash
python -m app --stage train --train-mode fast --cv-splits 2
```

Notes:

- `--stage predict` requires `models/churn_model.joblib`
- `full` mode can take noticeably longer because it benchmarks more candidates

## Streamlit App

The preferred web entrypoint is the repository-root file:

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

The UI supports:

- upload of `train.csv` and `test.csv`
- dataset summary and missing-value review
- numeric statistics and outlier analysis
- train/validation evaluation with ROC curve
- download of `submission.csv` and `predictions_with_risk.csv`

Legacy compatibility entrypoint:

```bash
streamlit run app/web_app.py
```

That wrapper exists for older commands, but new deployments should prefer `streamlit_app.py`.

## Docker

Build and run the batch pipeline:

```bash
docker compose -f docker/docker-compose.yml up --build churn-risk-pipeline
```

Build and run the Streamlit app:

```bash
docker compose -f docker/docker-compose.yml up --build churn-risk-web
```

The web service exposes port `8501`.

## Deploy Notes

If your hosting platform asks for a Streamlit entry file, use:

```text
streamlit_app.py
```

If your platform asks for a start command, use:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Why this matters:

- the previous command `streamlit run app/web_app.py` pointed to a file that was missing in this repo
- the root-level `streamlit_app.py` is now the canonical deploy entrypoint
- `app/web_app.py` is kept only as a backward-compatible wrapper

## Troubleshooting

### Error: `File does not exist: app/web_app.py`

Use:

```bash
streamlit run streamlit_app.py
```

or update your deployment command to:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### Error: model file not found during prediction

Run training first:

```bash
python -m app --stage train --train-mode fast
```

### Prediction output looks stale

The repository already contains generated artifacts in `models/` and `output/`. Re-run training and prediction if you want fresh outputs for the current data.

## Key Files

- `app/main.py`: CLI entrypoint for `train`, `predict`, or `all`
- `app/train.py`: model training and artifact persistence
- `app/predict.py`: batch scoring and CSV export
- `app/modeling.py`: candidate model selection and benchmark logic
- `app/features.py`: feature engineering and preprocessing setup
- `app/streamlit_app.py`: Streamlit UI implementation
- `streamlit_app.py`: deployment-friendly root launcher
- `docker/docker-compose.yml`: local container orchestration

## Suggested Workflow

1. Run `python -m app --stage train --train-mode fast` for a quick benchmark.
2. Run `python -m app --stage predict` to refresh output files.
3. Launch `streamlit run streamlit_app.py` for interactive analysis and downloadable outputs.
