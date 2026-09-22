"""
Inference steps shared by Online Serving (POST /predict) and Batch Serving.

Both paths go through prepare_model_input() and predict_frame(), so a flight
gets the same features and the same model regardless of how it arrives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.api.model_loader import (
    ModelState,
    get_state,
    load_pipeline,
)
from src.features.build_feature import build_features


# Predictions are reported in minutes, rounded to 2 decimals.
PREDICTION_PRECISION = 2


class ModelInputError(Exception):
    """The engineered frame cannot be fed to the model."""


class MissingModelFeaturesError(ModelInputError):

    def __init__(self, missing: list[str]) -> None:
        super().__init__(f"Missing model features: {missing}")
        self.missing = missing


class NullModelInputError(ModelInputError):

    def __init__(self, columns: list[str]) -> None:
        super().__init__(f"Model input contains NaN: {columns}")
        self.columns = columns


def ensure_model_loaded() -> ModelState:
    """Return the shared model state, loading the MLflow pipeline if needed."""

    state = get_state()

    if state.model is None:
        load_pipeline()

    return state


def prepare_model_input(
    raw: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Raw flight records -> exactly the columns the fitted preprocessor expects.

    Raises ValueError (from build_features) for invalid raw values,
    MissingModelFeaturesError / NullModelInputError for a bad model frame.
    """

    features = build_features(raw)

    missing = [
        column
        for column in expected_columns
        if column not in features.columns
    ]

    if missing:
        raise MissingModelFeaturesError(missing)

    X = features[expected_columns]

    if X.isna().any().any():
        raise NullModelInputError(
            X.columns[X.isna().any()].tolist()
        )

    return X


def predict_frame(model, X: pd.DataFrame) -> np.ndarray:
    """One vectorised model.predict() call for every row of X."""

    return np.asarray(model.predict(X), dtype=float)
