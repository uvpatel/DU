"""Human-readable EDA insights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def insights(df: pd.DataFrame, corr_threshold: float = 0.7) -> list[str]:
    """Generate human-readable insights from a DataFrame."""
    findings: list[str] = []

    missing_pct = (df.isna().mean() * 100).round(2)
    missing_warnings = missing_pct[missing_pct > 0]
    for col, pct in missing_warnings.items():
        findings.append(f"Column '{col}' has {pct}% missing values.")

    corr = df.select_dtypes(include=["number"]).corr().abs()
    if not corr.empty:
        upper = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
        strong = [
            (row, col, val)
            for row in upper.index
            for col, val in upper.loc[row].dropna().items()
            if val >= corr_threshold
        ]
        for a, b, c in strong:
            findings.append(
                f"Strong correlation detected between '{a}' and '{b}' (|r|={c:.2f})."
            )

    if not findings:
        findings.append("No major quality issues or strong correlations detected.")

    return findings
