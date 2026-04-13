"""Model training module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class TrainResult:
    """Training output container."""

    model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    task_type: str


def _is_classification_task(y: pd.Series) -> bool:
    if y.dtype == "O" or str(y.dtype).startswith("category") or y.dtype == bool:
        return True
    unique_count = y.nunique(dropna=True)
    return unique_count <= 20


def train(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    """Train an sklearn model with automatic task detection.

    Args:
        df: Input DataFrame.
        target: Target column name.
        test_size: Test split size.
        random_state: Random seed.

    Returns:
        TrainResult containing model and holdout set.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    is_classification = _is_classification_task(y)
    estimator: Any
    if is_classification:
        estimator = RandomForestClassifier(n_estimators=200, random_state=random_state)
        task = "classification"
    else:
        estimator = RandomForestRegressor(n_estimators=300, random_state=random_state)
        task = "regression"

    model = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

    stratify = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model.fit(X_train, y_train)

    return TrainResult(model=model, X_test=X_test, y_test=y_test, task_type=task)
