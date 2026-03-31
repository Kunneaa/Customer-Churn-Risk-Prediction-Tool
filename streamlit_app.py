from __future__ import annotations

import io
import math
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import DEFAULT_CV_SPLITS, ID_COLUMN, MODEL_NAME, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from app.features import add_features, normalize_churn_target, prepare_training_data
from app.modeling import evaluate_scores, fit_random_forest_pipeline, select_model_features
from app.predict import predict_with_risk


sns.set_theme(style="whitegrid")

NUMERIC_DISTRIBUTION_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "num_services"]
CATEGORICAL_DISTRIBUTION_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "tenure_group"]
CHURN_SEGMENT_COLUMNS = ["Contract", "InternetService", "PaperlessBilling", "tenure_group"]


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(52, 152, 128, 0.12), transparent 28%),
                    linear-gradient(180deg, #f6fbf9 0%, #eef5f1 52%, #e7efeb 100%);
            }
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            .hero-card {
                padding: 1.4rem 1.5rem;
                border-radius: 26px;
                background: linear-gradient(135deg, rgba(223, 245, 236, 0.96), rgba(244, 250, 247, 0.98));
                border: 1px solid rgba(56, 122, 101, 0.16);
                box-shadow: 0 18px 50px rgba(26, 82, 65, 0.10);
                margin-bottom: 1rem;
            }
            .hero-kicker {
                letter-spacing: 0.18em;
                text-transform: uppercase;
                font-size: 0.72rem;
                color: #2f6d5f;
                margin-bottom: 0.4rem;
                font-weight: 700;
            }
            .hero-title {
                font-size: 2.2rem;
                line-height: 1.05;
                color: #173f35;
                margin: 0;
                font-weight: 800;
            }
            .hero-copy {
                margin-top: 0.65rem;
                color: #365f54;
                font-size: 1rem;
                max-width: 58rem;
            }
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(40, 103, 85, 0.10);
                border-radius: 18px;
                padding: 0.9rem 1rem;
                box-shadow: 0 12px 30px rgba(27, 74, 61, 0.06);
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.55rem;
                background: rgba(255, 255, 255, 0.5);
                padding: 0.35rem;
                border-radius: 999px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 0.6rem 1rem;
                background: transparent;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #1c6b57, #2f8d75);
                color: white;
            }
            .section-copy {
                color: #4c6c63;
                margin-bottom: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number", "bool"]).columns.tolist()


@st.cache_data(show_spinner=False)
def _read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _prepare_analytics_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    featured = add_features(train_df)
    featured["_churn_binary"] = normalize_churn_target(featured[TARGET_COLUMN])
    return featured


@st.cache_data(show_spinner=False)
def _sample_for_visuals(df: pd.DataFrame, max_rows: int = 50000) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(max_rows, random_state=RANDOM_STATE)


def _plot_target_donut(train_df: pd.DataFrame) -> None:
    counts = train_df[TARGET_COLUMN].astype("string").fillna("Missing").value_counts()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.pie(
        counts.values,
        labels=counts.index.tolist(),
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2e8b72", "#d97554", "#97b7ad"],
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"color": "#20453b", "fontsize": 10},
    )
    ax.text(0, 0, "Churn\nMix", ha="center", va="center", fontsize=16, fontweight="bold", color="#20453b")
    ax.set_title("Class Distribution", fontsize=13, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)


def _plot_missing_values_bar(df: pd.DataFrame, title: str, top_n: int = 10) -> None:
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)

    if missing.empty:
        st.success(f"{title}: no missing values detected.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, color="#2f8d75")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Missing Rate (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


def _plot_categorical_distribution_grid(df: pd.DataFrame, columns: list[str]) -> None:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        st.info("No categorical columns available for distribution charts.")
        return

    rows = math.ceil(len(existing) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, existing):
        counts = (
            df[col]
            .astype("string")
            .fillna("Missing")
            .value_counts()
            .head(6)
            .sort_values(ascending=True)
        )
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#4fa58d")
        ax.set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Customers")
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.2)

    for ax in axes[len(existing):]:
        ax.axis("off")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_churn_rate_by_category(df: pd.DataFrame, columns: list[str]) -> None:
    existing = [c for c in columns if c in df.columns]
    if not existing or "_churn_binary" not in df.columns:
        st.info("No target-based classification charts available.")
        return

    rows = math.ceil(len(existing) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, existing):
        rate = (
            df[[col, "_churn_binary"]]
            .assign(**{col: df[col].astype("string").fillna("Missing")})
            .groupby(col)["_churn_binary"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .head(6)
            .sort_values(ascending=True)
        )
        sns.barplot(x=rate.values, y=rate.index, ax=ax, color="#d97554")
        ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Churn Rate (%)")
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.2)

    for ax in axes[len(existing):]:
        ax.axis("off")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_numeric_distributions(df: pd.DataFrame, columns: list[str]) -> None:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        st.info("No numeric columns available for distribution charts.")
        return

    rows = math.ceil(len(existing) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, existing):
        sns.histplot(df[col], kde=True, ax=ax, color="#2a7b67", bins=35)
        ax.set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.grid(axis="y", linestyle="--", alpha=0.15)

    for ax in axes[len(existing):]:
        ax.axis("off")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_outlier_rate_chart(outlier_tbl: pd.DataFrame) -> None:
    if outlier_tbl.empty:
        st.info("No numeric columns available for outlier chart.")
        return

    top = outlier_tbl.head(10).sort_values("outlier_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    sns.barplot(x=top["outlier_rate_pct"], y=top["feature"], ax=ax, color="#8a5fd1")
    ax.set_title("Top IQR Outlier Rates", fontsize=13, fontweight="bold")
    ax.set_xlabel("Outlier Rate (%)")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = [c for c in _numeric_columns(df) if c != "_churn_binary"]
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return

    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.heatmap(corr, cmap="crest", center=0, square=False, ax=ax, linewidths=0.5)
    ax.set_title("Numeric Feature Correlation", fontsize=13, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)


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


def summarize_dataset(df: pd.DataFrame, name: str) -> None:
    st.subheader(f"{name} Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    left, right = st.columns([1.2, 1.0])
    with left:
        st.dataframe(df.head(15), use_container_width=True)
    with right:
        dtypes = df.dtypes.astype(str).rename("dtype").to_frame()
        st.dataframe(dtypes, use_container_width=True)


def train_and_evaluate(
    train_df: pd.DataFrame,
    cv_splits: int = DEFAULT_CV_SPLITS,
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

    pipeline, training_metrics = fit_random_forest_pipeline(
        X_train_selected,
        y_train,
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
            "selected_model": str(training_metrics["selected_model"]),
            "cv_splits": int(training_metrics["cv_splits"]),
            "cv_roc_auc_mean": float(training_metrics["cv_roc_auc_mean"]),
            "cv_ap_mean": float(training_metrics["cv_ap_mean"]),
        }
    )
    return pipeline, metrics, y_valid, valid_proba


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def run_app() -> None:
    st.set_page_config(page_title="Customer Churn Risk Prediction Tool", layout="wide")
    apply_app_theme()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Customer Retention Intelligence</div>
            <h1 class="hero-title">Customer Churn Risk Prediction Tool</h1>
            <div class="hero-copy">
                Upload train and test data, inspect the portfolio through charts, train a RandomForest churn model,
                and export churn predictions with business-friendly risk segmentation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_left, upload_right = st.columns(2)
    with upload_left:
        st.markdown("**Training Dataset**")
        train_file = st.file_uploader("Upload train.csv", type=["csv"], key="train_upload")
    with upload_right:
        st.markdown("**Prediction Dataset**")
        test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_upload")

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

    analytics_df = _prepare_analytics_frame(train_df)
    visual_df = _sample_for_visuals(analytics_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Rows", f"{len(train_df):,}")
    c2.metric("Test Rows", f"{len(test_df):,}")
    c3.metric("Train Columns", f"{train_df.shape[1]:,}")
    c4.metric("Churn Positives", f"{int(analytics_df['_churn_binary'].sum()):,}")

    tab1, tab2, tab3 = st.tabs(["Overview", "Deep Analytics", "Model and Outputs"])

    with tab1:
        st.markdown(
            '<div class="section-copy">Quick schema review and a first look at both uploaded datasets.</div>',
            unsafe_allow_html=True,
        )
        overview_left, overview_right = st.columns(2)
        with overview_left:
            summarize_dataset(train_df, "Train")
        with overview_right:
            summarize_dataset(test_df, "Test")

    with tab2:
        st.markdown(
            '<div class="section-copy">Deep analytics is shown as charts first: class mix, missingness, segmentation, correlations, and outlier behavior.</div>',
            unsafe_allow_html=True,
        )

        top_left, top_right = st.columns(2)
        with top_left:
            _plot_target_donut(train_df)
        with top_right:
            _plot_missing_values_bar(train_df, "Train Missing Values")

        mid_left, mid_right = st.columns(2)
        with mid_left:
            st.markdown("**Categorical Volume**")
            _plot_categorical_distribution_grid(analytics_df, CATEGORICAL_DISTRIBUTION_COLUMNS)
        with mid_right:
            st.markdown("**Churn Classification by Segment**")
            _plot_churn_rate_by_category(analytics_df, CHURN_SEGMENT_COLUMNS)

        lower_left, lower_right = st.columns(2)
        with lower_left:
            st.markdown("**Numeric Distributions**")
            _plot_numeric_distributions(visual_df, NUMERIC_DISTRIBUTION_COLUMNS)
        with lower_right:
            st.markdown("**Correlation Heatmap**")
            _plot_correlation_heatmap(visual_df)

        outlier_tbl = iqr_outlier_table(analytics_df)
        iso_stats = isolation_forest_outlier_rate(analytics_df)

        st.markdown("**Outlier Diagnostics**")
        c1, c2, c3 = st.columns(3)
        c1.metric("IsolationForest Sample", f"{iso_stats['sample_rows']:,}")
        c2.metric("Predicted Outlier Rows", f"{iso_stats['outlier_rows']:,}")
        c3.metric("Outlier Rate", f"{iso_stats['outlier_rate_pct']}%")

        outlier_left, outlier_right = st.columns([1.0, 1.15])
        with outlier_left:
            _plot_outlier_rate_chart(outlier_tbl)
        with outlier_right:
            train_stats = numeric_statistics(analytics_df)
            if train_stats.empty:
                st.info("No numeric statistics available.")
            else:
                st.dataframe(train_stats.round(4), use_container_width=True)

        with st.expander("Detailed Outlier Table", expanded=False):
            if outlier_tbl.empty:
                st.info("No outlier table available.")
            else:
                st.dataframe(outlier_tbl.round(4), use_container_width=True)

    with tab3:
        st.markdown(
            '<div class="section-copy">Train the fixed RandomForest pipeline, review validation metrics, and export prediction files.</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Model Configuration")
        st.info(f"The application now uses a single model only: `{MODEL_NAME}`.")
        cv_splits = st.select_slider(
            "CV folds for RandomForest evaluation",
            options=[2, 3, 4, 5],
            value=DEFAULT_CV_SPLITS,
        )

        if st.button("Train Model and Generate Outputs", type="primary"):
            with st.spinner("Training model and generating predictions..."):
                pipeline, metrics, y_valid, valid_proba = train_and_evaluate(
                    train_df,
                    cv_splits=int(cv_splits),
                )
                pred_df, submission_df = predict_with_risk(pipeline, test_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            c2.metric("Average Precision", f"{metrics['average_precision']:.4f}")
            c3.metric("Selected Model", str(metrics["selected_model"]))
            st.caption(
                f"`{metrics['selected_model']}` was evaluated with `{metrics['cv_splits']}` CV folds. "
                f"Cross-validation ROC-AUC: `{metrics['cv_roc_auc_mean']:.4f}`."
            )

            roc_left, roc_right = st.columns([1.05, 0.95])
            with roc_left:
                st.subheader("ROC Curve")
                fig, ax = plt.subplots(figsize=(7, 5))
                RocCurveDisplay.from_predictions(y_valid, valid_proba, ax=ax)
                ax.set_title("Validation ROC Curve")
                st.pyplot(fig)
                plt.close(fig)
            with roc_right:
                st.subheader("Prediction Preview")
                st.dataframe(pred_df.head(30), use_container_width=True)

            download_left, download_right = st.columns(2)
            with download_left:
                st.download_button(
                    label="Download submission.csv",
                    data=to_csv_bytes(submission_df),
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with download_right:
                st.download_button(
                    label="Download predictions_with_risk.csv",
                    data=to_csv_bytes(pred_df),
                    file_name="predictions_with_risk.csv",
                    mime="text/csv",
                    use_container_width=True,
                )



if __name__ == "__main__":
    run_app()
