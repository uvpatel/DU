"""Model evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate(model: Any, X_test: Any, y_test: Any) -> dict[str, float]:
    """Evaluate an sklearn-compatible model.

    Automatically selects classification or regression metrics.
    """
    preds = model.predict(X_test)

    # Heuristic: low number of unique labels and non-floating outputs => classification.
    unique_y = len(np.unique(y_test))
    is_classification = unique_y <= 20 and not np.issubdtype(np.asarray(y_test).dtype, np.floating)

    if is_classification:
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
        }

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": rmse,
        "r2": float(r2_score(y_test, preds)),
    }
