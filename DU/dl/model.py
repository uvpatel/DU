"""Keras model definitions and training helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from tensorflow import keras


def build_dense_model(input_dim: int, output_units: int = 1, task: str = "regression") -> keras.Model:
    """Build a simple dense neural network with TensorFlow/Keras backend."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(output_units, activation="sigmoid" if task == "classification" else "linear"),
        ]
    )

    if task == "classification":
        loss = "binary_crossentropy" if output_units == 1 else "sparse_categorical_crossentropy"
        metrics = ["accuracy"]
    else:
        loss = "mse"
        metrics = ["mae"]

    model.compile(optimizer="adam", loss=loss, metrics=metrics)
    return model


def train_keras_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "regression",
    epochs: int = 20,
    batch_size: int = 32,
) -> keras.Model:
    """Train a simple dense Keras model."""
    output_units = len(np.unique(y_train)) if task == "classification" and len(np.unique(y_train)) > 2 else 1
    model = build_dense_model(input_dim=X_train.shape[1], output_units=output_units, task=task)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model
