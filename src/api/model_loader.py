# src/api/model_loader.py

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from config.config import Config


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MODEL STATE
# ============================================================

_MODEL: Any | None = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model() -> Any:
    """
    Load the trained MLflow sklearn pipeline once.

    The loaded object must expose:
        predict()

    Expected pipeline structure:

        Pipeline
        ├── preprocessor
        └── model
    """

    global _MODEL

    # --------------------------------------------------------
    # Return already loaded model
    # --------------------------------------------------------

    if _MODEL is not None:

        logger.info(
            "MLflow model already loaded."
        )

        return _MODEL

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    tracking_uri = Config.MLFLOW.TRACKING_URI
    model_uri = Config.MLFLOW.MODEL_URI

    if not tracking_uri:
        raise RuntimeError(
            "MLflow tracking URI is not configured."
        )

    if not model_uri:
        raise RuntimeError(
            "MLflow model URI is not configured."
        )

    logger.info(
        "MLflow tracking URI: %s",
        tracking_uri,
    )

    logger.info(
        "MLflow model URI: %s",
        model_uri,
    )

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        tracking_uri
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        pipeline = mlflow.sklearn.load_model(
            model_uri
        )

    except Exception as exc:

        logger.exception(
            "Failed to load MLflow model."
        )

        raise RuntimeError(
            f"Could not load MLflow model: {model_uri}"
        ) from exc

    # --------------------------------------------------------
    # Validate predict()
    # --------------------------------------------------------

    if not hasattr(
        pipeline,
        "predict",
    ):

        raise RuntimeError(
            "The loaded MLflow artifact does not "
            "expose predict()."
        )

    # --------------------------------------------------------
    # Validate sklearn Pipeline
    # --------------------------------------------------------

    if not hasattr(
        pipeline,
        "named_steps",
    ):

        raise RuntimeError(
            "The loaded MLflow model is not "
            "a sklearn Pipeline."
        )

    named_steps = pipeline.named_steps

    logger.info(
        "Pipeline steps: %s",
        list(named_steps.keys()),
    )

    # --------------------------------------------------------
    # Validate preprocessor
    # --------------------------------------------------------

    preprocessor = named_steps.get(
        "preprocessor"
    )

    if preprocessor is None:

        raise RuntimeError(
            "The loaded pipeline does not contain "
            "'preprocessor'."
        )

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    model = named_steps.get(
        "model"
    )

    if model is None:

        raise RuntimeError(
            "The loaded pipeline does not contain "
            "'model'."
        )

    logger.info(
        "Model type: %s",
        type(model).__name__,
    )

    # --------------------------------------------------------
    # Store model in memory
    # --------------------------------------------------------

    _MODEL = pipeline

    logger.info(
        "MLflow model loaded successfully."
    )

    return _MODEL


# ============================================================
# GET MODEL
# ============================================================

def get_model() -> Any:
    """
    Return the loaded model.

    Raises an error if the model has not been loaded.
    """

    if _MODEL is None:

        raise RuntimeError(
            "Model is not loaded. "
            "Call load_model() during application startup."
        )

    return _MODEL


# ============================================================
# GET STATE
# ============================================================

def get_state() -> dict[str, Any]:
    """
    Return the current API model state.
    """

    return {
        "model": _MODEL,
        "loaded": _MODEL is not None,
    }


# ============================================================
# EXPECTED INPUT COLUMNS
# ============================================================

def get_expected_columns(
    pipeline: Any | None = None,
) -> list[str]:
    """
    Return the raw feature columns expected by
    the fitted preprocessor.

    If no pipeline is supplied, the loaded model
    is used.
    """

    if pipeline is None:
        pipeline = get_model()

    # --------------------------------------------------------
    # Validate Pipeline
    # --------------------------------------------------------

    if not hasattr(
        pipeline,
        "named_steps",
    ):

        raise RuntimeError(
            "Loaded model is not a sklearn Pipeline."
        )

    # --------------------------------------------------------
    # Get preprocessor
    # --------------------------------------------------------

    preprocessor = pipeline.named_steps.get(
        "preprocessor"
    )

    if preprocessor is None:

        raise RuntimeError(
            "Pipeline does not contain "
            "'preprocessor'."
        )

    # --------------------------------------------------------
    # Get input columns
    # --------------------------------------------------------

    expected_columns = getattr(
        preprocessor,
        "feature_names_in_",
        None,
    )

    if expected_columns is None:

        raise RuntimeError(
            "The fitted preprocessor does not contain "
            "'feature_names_in_'."
        )

    columns = list(
        expected_columns
    )

    logger.info(
        "Expected input columns: %d",
        len(columns),
    )

    logger.debug(
        "Expected columns: %s",
        columns,
    )

    return columns


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info(
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """
    Return information about the loaded model.
    """

    if pipeline is None:
        pipeline = get_model()

    info: dict[str, Any] = {
        "pipeline_type": type(
            pipeline
        ).__name__,
    }

    # --------------------------------------------------------
    # Pipeline information
    # --------------------------------------------------------

    if hasattr(
        pipeline,
        "named_steps",
    ):

        info["pipeline_steps"] = list(
            pipeline.named_steps.keys()
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = pipeline.named_steps.get(
            "model"
        )

        if model is not None:

            info["model_type"] = type(
                model
            ).__name__

        # ----------------------------------------------------
        # Preprocessor
        # ----------------------------------------------------

        preprocessor = pipeline.named_steps.get(
            "preprocessor"
        )

        if preprocessor is not None:

            columns = getattr(
                preprocessor,
                "feature_names_in_",
                None,
            )

            if columns is not None:

                info["input_features"] = len(
                    columns
                )

    return info


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
    )

    try:

        pipeline = load_model()

        expected_columns = (
            get_expected_columns(
                pipeline
            )
        )

        model_info = get_model_info(
            pipeline
        )

        print("\n" + "=" * 70)
        print("MODEL LOADED")
        print("=" * 70)

        print("\nModel information:")

        for key, value in model_info.items():

            print(
                f"{key}: {value}"
            )

        print("\nExpected input columns:")

        for column in expected_columns:

            print(
                f"  - {column}"
            )

        print(
            "\nModel loading test passed."
        )

    except Exception:

        logger.exception(
            "Model loading test failed."
        )

        raise