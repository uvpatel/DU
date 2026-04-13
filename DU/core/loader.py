"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from du.utils.logger import get_logger

logger = get_logger(__name__)


SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json"}


def load(file_path: str | Path) -> pd.DataFrame:
    """Load data from CSV, Excel, or JSON.

    Args:
        file_path: Path to source file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file format is unsupported.

    Returns:
        Loaded DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    logger.info("Loading data from %s", path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    # Handle both JSON array and line-delimited JSON.
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)
