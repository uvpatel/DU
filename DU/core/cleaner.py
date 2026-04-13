"""Data cleaning and preprocessing helpers."""

from __future__ import annotations

import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned DataFrame with simple defaults.

    - Numeric columns: median imputation
    - Categorical columns: mode imputation
    """
    cleaned = df.copy()

    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    categorical_cols = cleaned.select_dtypes(exclude=["number"]).columns

    for col in numeric_cols:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in categorical_cols:
        mode = cleaned[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        cleaned[col] = cleaned[col].fillna(fill_value)

    return cleaned
