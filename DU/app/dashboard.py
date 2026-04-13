"""Streamlit dashboard for DU."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from du.core.loader import load
from du.eda.insights import insights
from du.eda.summary import summary
from du.ml.train import train


def run_app(df: pd.DataFrame | None = None) -> None:
    """Run Streamlit dashboard with upload, charts, and predictions."""
    st.set_page_config(page_title="DU Dashboard", layout="wide")
    st.title("DU - Data Understanding Dashboard")

    uploaded = st.file_uploader("Upload data (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

    if df is None and uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(uploaded)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_json(uploaded)

    if df is None:
        st.info("Upload a dataset to begin.")
        return

    st.subheader("Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Summary")
    stats = summary(df)
    st.write("Shape:", stats["shape"])
    st.write("Missing values:", stats["missing_values"])
    st.write("Data types:", stats["dtypes"])

    if not stats["correlation_matrix"].empty:
        st.subheader("Correlation Heatmap")
        st.dataframe(stats["correlation_matrix"])

    st.subheader("Insights")
    for item in insights(df):
        st.write(f"- {item}")

    st.subheader("Quick Prediction Workflow")
    target = st.selectbox("Select target column", options=df.columns.tolist())
    if st.button("Train model"):
        result = train(df, target)
        st.success(f"Model trained ({result.task_type}).")
        st.session_state["du_model"] = result.model
        st.session_state["du_features"] = [c for c in df.columns if c != target]

    if "du_model" in st.session_state:
        st.markdown("Enter one record to predict:")
        row: dict[str, Any] = {}
        for col in st.session_state["du_features"]:
            row[col] = st.text_input(f"{col}", key=f"input_{col}")
        if st.button("Predict"):
            record = pd.DataFrame([row])
            pred = st.session_state["du_model"].predict(record)[0]
            st.write("Prediction:", pred)
