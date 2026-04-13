"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from du.utils.logger import get_logger

logger = get_logger(__name__)


def plot(df: pd.DataFrame, output_dir: str | Path | None = None) -> dict[str, str]:
    """Create histogram, heatmap and pairplot charts.

    Args:
        df: Input DataFrame.
        output_dir: Optional output directory. If provided, charts are saved.

    Returns:
        Mapping of chart names to saved file paths. Empty if not saved.
    """
    sns.set_theme(style="whitegrid")
    saved: dict[str, str] = {}

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        numeric_df.hist(figsize=(10, 6))
        fig = plt.gcf()
        plt.tight_layout()
        if output_path:
            file_path = output_path / "histogram.png"
            fig.savefig(file_path, dpi=150)
            saved["histogram"] = str(file_path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        plt.tight_layout()
        if output_path:
            file_path = output_path / "heatmap.png"
            fig.savefig(file_path, dpi=150)
            saved["heatmap"] = str(file_path)
        plt.close(fig)

        pair = sns.pairplot(numeric_df)
        if output_path:
            file_path = output_path / "pairplot.png"
            pair.savefig(file_path, dpi=150)
            saved["pairplot"] = str(file_path)
        plt.close(pair.fig)

    logger.info("Generated plots: %s", list(saved.keys()))
    return saved
