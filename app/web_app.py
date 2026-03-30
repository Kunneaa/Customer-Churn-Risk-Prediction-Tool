from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import RocCurveDisplay, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from app.config import (
        CLASSIFICATION_THRESHOLD,
        HIGH_RISK_THRESHOLD,
        ID_COLUMN,
        LOW_RISK_THRESHOLD,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
    )
    from app.features import build_preprocessor, normalize_churn_target
    from app.modeling import build_classifier, prepare_features
    from app.risk import map_risk_level, to_binary_label
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from app.config import (
        CLASSIFICATION_THRESHOLD,
        HIGH_RISK_THRESHOLD,
        ID_COLUMN,
        LOW_RISK_THRESHOLD,
        RANDOM_STATE,
        TARGET_COLUMN,
        TEST_SIZE,
    )
    from app.features import build_preprocessor, normalize_churn_target
    from app.modeling import build_classifier, prepare_features
    from app.risk import map_risk_level, to_binary_label


sns.set_theme(style="whitegrid")


def summarize_dataset(df: pd.DataFrame, name: str) -> None:
    st.subheader(f"{name} - Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    st.write("Data types")
    st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame())

    st.write("Missing values by column")
    miss = df.isna().sum().sort_values(ascending=False).rename("missing_count").to_frame()
    miss["missing_rate_pct"] = (miss["missing_count"] / max(len(df), 1) * 100).round(2)
    st.dataframe(miss)


def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    stats = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["min"] = df[numeric_cols].min()
    stats["max"] = df[numeric_cols].max()
    stats["variance"] = df[numeric_cols].var(numeric_only=True)
    stats["skewness"] = df[numeric_cols].skew(numeric_only=True)
    stats["kurtosis"] = df[numeric_cols].kurt(numeric_only=True)
    return stats


def iqr_outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()

    records = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        count = int(mask.sum())
        rate = round(count / max(len(s), 1) * 100, 3)
        records.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": count,
                "outlier_rate_pct": rate,
            }
        )
    return pd.DataFrame(records).sort_values("outlier_rate_pct", ascending=False)


def isolation_forest_outlier_rate(df: pd.DataFrame, sample_size: int = 50000) -> Dict[str, float]:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_cols:
        return {"sample_rows": 0, "outlier_rows": 0, "outlier_rate_pct": 0.0}

    sample = df[numeric_cols].copy()
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=RANDOM_STATE)
    sample = sample.fillna(sample.median(numeric_only=True))

    iso = IsolationForest(contamination="auto", random_state=RANDOM_STATE, n_estimators=200)
    pred = iso.fit_predict(sample)
    out_count = int((pred == -1).sum())
    return {
        "sample_rows": int(len(sample)),
        "outlier_rows": out_count,
        "outlier_rate_pct": round(out_count / max(len(sample), 1) * 100, 3),
    }


def train_and_evaluate(train_df: pd.DataFrame):
    y = normalize_churn_target(train_df[TARGET_COLUMN])
    X = prepare_features(train_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", build_classifier()),
        ]
    )
    pipeline.fit(X_train, y_train)

    valid_proba = pipeline.predict_proba(X_valid)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, valid_proba)),
        "average_precision": float(average_precision_score(y_valid, valid_proba)),
    }
    return pipeline, metrics, y_valid, valid_proba


def prediction_outputs(pipeline: Pipeline, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_test = prepare_features(test_df)
    test_proba = pd.Series(pipeline.predict_proba(X_test)[:, 1], name="churn_probability", index=test_df.index)

    risk = map_risk_level(test_proba, low_threshold=LOW_RISK_THRESHOLD, high_threshold=HIGH_RISK_THRESHOLD)
    churn_bin = to_binary_label(test_proba, threshold=CLASSIFICATION_THRESHOLD)

    pred_df = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            "churn_probability": test_proba,
            "risk_level": risk,
            TARGET_COLUMN: churn_bin,
        }
    )
    submission_df = pred_df[[ID_COLUMN, TARGET_COLUMN]].copy()
    return pred_df, submission_df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def run_app() -> None:
    st.set_page_config(page_title="Customer Churn Risk Prediction Tool", layout="wide")
    st.title("Customer Churn Risk Prediction Tool")
    st.caption("Upload train.csv and test.csv, run deep analytics, train model, view ROC-AUC, and export submission.")

    col1, col2 = st.columns(2)
    train_file = col1.file_uploader("Upload train.csv", type=["csv"])
    test_file = col2.file_uploader("Upload test.csv", type=["csv"])

    if not train_file or not test_file:
        st.info("Please upload both train.csv and test.csv to continue.")
        return

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    missing_cols = [c for c in [TARGET_COLUMN, ID_COLUMN] if c not in train_df.columns]
    if missing_cols:
        st.error(f"train.csv is missing required columns: {missing_cols}")
        return
    if ID_COLUMN not in test_df.columns:
        st.error(f"test.csv must contain column: {ID_COLUMN}")
        return

    tab1, tab2, tab3 = st.tabs(["Data Overview", "Deep Analytics", "Model and Outputs"])

    with tab1:
        summarize_dataset(train_df, "Train")
        summarize_dataset(test_df, "Test")
        st.subheader("Train Preview")
        st.dataframe(train_df.head(20))
        st.subheader("Test Preview")
        st.dataframe(test_df.head(20))

    with tab2:
        st.subheader("Numeric Statistics (Train)")
        train_stats = numeric_statistics(train_df)
        if train_stats.empty:
            st.warning("No numeric columns found in train.csv")
        else:
            st.dataframe(train_stats)

        st.subheader("Outlier Detection - IQR (Train)")
        outlier_tbl = iqr_outlier_table(train_df)
        if outlier_tbl.empty:
            st.warning("No numeric columns available for outlier analysis")
        else:
            st.dataframe(outlier_tbl)

        st.subheader("Outlier Probability - IsolationForest (Train)")
        iso_stats = isolation_forest_outlier_rate(train_df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sample Rows", f"{iso_stats['sample_rows']:,}")
        c2.metric("Predicted Outlier Rows", f"{iso_stats['outlier_rows']:,}")
        c3.metric("Outlier Rate %", f"{iso_stats['outlier_rate_pct']}%")

        st.subheader("Distribution Plots")
        numeric_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in train_df.columns]
        if numeric_cols:
            fig, axes = plt.subplots(1, len(numeric_cols), figsize=(6 * len(numeric_cols), 4))
            if len(numeric_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, numeric_cols):
                sns.histplot(train_df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    with tab3:
        if st.button("Train Model and Generate Outputs", type="primary"):
            with st.spinner("Training model and generating predictions..."):
                pipeline, metrics, y_valid, valid_proba = train_and_evaluate(train_df)
                pred_df, submission_df = prediction_outputs(pipeline, test_df)

            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            c2.metric("Average Precision", f"{metrics['average_precision']:.4f}")

            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(7, 5))
            RocCurveDisplay.from_predictions(y_valid, valid_proba, ax=ax)
            ax.set_title("Validation ROC Curve")
            st.pyplot(fig)

            st.subheader("Predictions with Risk")
            st.dataframe(pred_df.head(30))

            st.download_button(
                label="Download submission.csv",
                data=to_csv_bytes(submission_df),
                file_name="submission.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download predictions_with_risk.csv",
                data=to_csv_bytes(pred_df),
                file_name="predictions_with_risk.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    run_app()
