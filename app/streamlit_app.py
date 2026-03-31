from __future__ import annotations

import io
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from app.features import prepare_training_data
from app.modeling import evaluate_scores, fit_best_pipeline, select_model_features
from app.predict import predict_with_risk


sns.set_theme(style="whitegrid")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number", "bool"]).columns.tolist()


@st.cache_data(show_spinner=False)
def _read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


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


@st.cache_data(show_spinner=False)
def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        return pd.DataFrame()

    stats = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["min"] = df[numeric_cols].min()
    stats["max"] = df[numeric_cols].max()
    stats["variance"] = df[numeric_cols].var(numeric_only=True)
    stats["skewness"] = df[numeric_cols].skew(numeric_only=True)
    stats["kurtosis"] = df[numeric_cols].kurt(numeric_only=True)
    return stats


@st.cache_data(show_spinner=False)
def iqr_outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = _numeric_columns(df)
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


@st.cache_data(show_spinner=False)
def isolation_forest_outlier_rate(df: pd.DataFrame, sample_size: int = 50000) -> Dict[str, float]:
    numeric_cols = _numeric_columns(df)
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


def train_and_evaluate(
    train_df: pd.DataFrame,
    mode: str = "full",
    cv_splits: int = 3,
) -> tuple[Pipeline, dict[str, float | int | str], pd.Series, pd.Series]:
    X, y = prepare_training_data(train_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_selected, used_features = select_model_features(X_train)
    X_valid_selected = X_valid.reindex(columns=used_features).copy()

    pipeline, benchmark_metrics = fit_best_pipeline(
        X_train_selected,
        y_train,
        mode=mode,
        cv_splits=cv_splits,
        verbose=False,
    )

    valid_proba = pd.Series(
        pipeline.predict_proba(X_valid_selected)[:, 1],
        index=y_valid.index,
        name="validation_probability",
    )
    metrics = evaluate_scores(y_valid, valid_proba)
    metrics.update(
        {
            "selected_model": str(benchmark_metrics["selected_model"]),
            "train_mode": str(benchmark_metrics["train_mode"]),
            "cv_splits": int(benchmark_metrics["cv_splits"]),
            "n_candidates": int(benchmark_metrics["n_candidates"]),
            "benchmark_cv_roc_auc_mean": float(benchmark_metrics["cv_roc_auc_mean"]),
            "benchmark_cv_ap_mean": float(benchmark_metrics["cv_ap_mean"]),
        }
    )
    return pipeline, metrics, y_valid, valid_proba


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

    train_df = _read_uploaded_csv(train_file.getvalue())
    test_df = _read_uploaded_csv(test_file.getvalue())

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
            plt.close(fig)

    with tab3:
        st.subheader("Model Selection")
        train_mode = st.radio(
            "Search mode",
            options=["fast", "full"],
            index=1,
            horizontal=True,
            help="full benchmarks more candidate models and usually takes longer; fast searches a smaller shortlist.",
        )
        cv_splits = st.select_slider(
            "CV folds for model selection",
            options=[2, 3, 4, 5],
            value=3,
        )

        if st.button("Train Model and Generate Outputs", type="primary"):
            with st.spinner("Training model and generating predictions..."):
                pipeline, metrics, y_valid, valid_proba = train_and_evaluate(
                    train_df,
                    mode=train_mode,
                    cv_splits=int(cv_splits),
                )
                pred_df, submission_df = predict_with_risk(pipeline, test_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            c2.metric("Average Precision", f"{metrics['average_precision']:.4f}")
            c3.metric("Selected Model", str(metrics["selected_model"]))
            st.caption(
                f"Model search used `{metrics['train_mode']}` mode with `{metrics['cv_splits']}` CV folds "
                f"across `{metrics['n_candidates']}` candidates. "
                f"Best benchmark ROC-AUC: `{metrics['benchmark_cv_roc_auc_mean']:.4f}`."
            )

            st.subheader("ROC Curve")
            fig, ax = plt.subplots(figsize=(7, 5))
            RocCurveDisplay.from_predictions(y_valid, valid_proba, ax=ax)
            ax.set_title("Validation ROC Curve")
            st.pyplot(fig)
            plt.close(fig)

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

    with st.expander("Deploy Command", expanded=False):
        st.code("streamlit run streamlit_app.py", language="bash")


if __name__ == "__main__":
    run_app()
