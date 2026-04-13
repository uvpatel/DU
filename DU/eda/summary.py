"""EDA summary utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a high-level data summary.

    Returns keys:
    - shape
    - missing_values
    - dtypes
    - correlation_matrix
    """
    corr = df.select_dtypes(include=["number"]).corr(numeric_only=True)

    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "correlation_matrix": corr,
    }
