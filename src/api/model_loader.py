from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from dotenv import load_dotenv

from config.config import Config


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PATHS
# ============================================================

_API_DIR = Path(__file__).resolve().parent
_SRC_DIR = _API_DIR.parent
_PROJECT_DIR = _SRC_DIR.parent

sys.path.insert(0, str(_PROJECT_DIR))
sys.path.insert(0, str(_SRC_DIR))


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================
#
# Config is the single source of truth for tracking/model URIs
# (itself backed by the MLFLOW_TRACKING_URI / MLFLOW_MODEL_URI
# env vars). Do not duplicate the lookup here - a previous drift
# between this module's defaults and config/config.py's defaults
# caused the API to silently keep loading a stale model.

MLFLOW_TRACKING_URI = Config.MLFLOW.TRACKING_URI

MODEL_URI = Config.MLFLOW.MODEL_URI


# ============================================================
# APPLICATION STATE
# ============================================================

class ModelState:
    """
    Holds the loaded MLflow pipeline
    and model information.
    """

    model: Any = None

    version: str = "unknown"

    source: str = "mlflow"

    model_uri: str = MODEL_URI

    expected_columns: list[str] = []

_state = ModelState()


# ============================================================
# LOAD PIPELINE
# ============================================================

def load_pipeline() -> None:
    """
    Load the complete sklearn pipeline from MLflow.

    The model is loaded once during FastAPI startup.

    Expected pipeline structure:

        Pipeline
        ├── preprocessor
        └── model

    The loaded pipeline must expose:
        predict()
    """

    # ========================================================
    # LOG CONFIGURATION
    # ========================================================

    logger.info(
        "MLflow tracking URI: %s",
        MLFLOW_TRACKING_URI,
    )

    logger.info(
        "MLflow model URI: %s",
        MODEL_URI,
    )

    # ========================================================
    # VALIDATE CONFIGURATION
    # ========================================================

    if not MLFLOW_TRACKING_URI:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not configured."
        )

    if not MODEL_URI:
        raise RuntimeError(
            "MLFLOW_MODEL_URI is not configured."
        )

    # ========================================================
    # CONFIGURE MLFLOW
    # ========================================================

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    try:

        pipeline = mlflow.sklearn.load_model(
            MODEL_URI
        )

    except Exception as exc:

        logger.exception(
            "Failed to load MLflow model."
        )

        raise RuntimeError(
            f"Could not load MLflow model: "
            f"{MODEL_URI}"
        ) from exc

    # ========================================================
    # VALIDATE predict()
    # ========================================================

    if not hasattr(
        pipeline,
        "predict",
    ):
        raise RuntimeError(
            "Loaded MLflow model does not "
            "expose predict()."
        )

    # ========================================================
    # VALIDATE SKLEARN PIPELINE
    # ========================================================

    if not hasattr(
        pipeline,
        "named_steps",
    ):
        raise RuntimeError(
            "Loaded MLflow model is not "
            "a sklearn Pipeline."
        )

    named_steps = pipeline.named_steps

    logger.info(
        "Pipeline steps: %s",
        list(named_steps.keys()),
    )

    # ========================================================
    # VALIDATE PREPROCESSOR
    # ========================================================

    preprocessor = named_steps.get(
        "preprocessor"
    )

    if preprocessor is None:
        raise RuntimeError(
            "Pipeline does not contain "
            "'preprocessor' step."
        )

    logger.info(
        "Preprocessor type: %s",
        type(preprocessor).__name__,
    )

    # ========================================================
    # VALIDATE MODEL
    # ========================================================

    model = named_steps.get(
        "model"
    )

    if model is None:
        raise RuntimeError(
            "Pipeline does not contain "
            "'model' step."
        )

    logger.info(
        "Model type: %s",
        type(model).__name__,
    )

    # ========================================================
    # GET EXPECTED INPUT COLUMNS
    # ========================================================

    expected_columns = getattr(
        preprocessor,
        "feature_names_in_",
        None,
    )

    if expected_columns is None:

        raise RuntimeError(
            "Fitted preprocessor does not contain "
            "'feature_names_in_'."
        )

    expected_columns = list(
        expected_columns
    )

    # ========================================================
    # VALIDATE EXPECTED COLUMNS
    # ========================================================

    if not expected_columns:

        raise RuntimeError(
            "The fitted preprocessor has "
            "no expected input columns."
        )

    logger.info(
        "Expected input features: %d",
        len(expected_columns),
    )

    logger.info(
        "Expected columns: %s",
        expected_columns,
    )

    # ========================================================
    # STORE MODEL STATE
    # ========================================================

    _state.model = pipeline

    _state.expected_columns = (
        expected_columns
    )

    _state.source = "mlflow"

    _state.model_uri = MODEL_URI

    # ========================================================
    # SUCCESS
    # ========================================================

    logger.info(
        "MLflow pipeline loaded successfully."
    )

    logger.info(
        "Model source: %s",
        MODEL_URI,
    )

    logger.info(
        "Model ready for prediction."
    )

# ============================================================
# GET STATE
# ============================================================

def get_state() -> ModelState:
    """
    Return current model state.
    """

    return _state


# ============================================================
# GET MODEL
# ============================================================

def get_model() -> Any:
    """
    Return loaded MLflow pipeline.
    """

    if _state.model is None:

        raise RuntimeError(
            "Model has not been loaded."
        )

    return _state.model


# ============================================================
# GET EXPECTED COLUMNS
# ============================================================

def get_expected_columns() -> list[str]:
    """
    Return raw columns expected by
    the fitted preprocessor.
    """

    if not _state.expected_columns:

        raise RuntimeError(
            "Model has not been loaded "
            "or expected columns are unavailable."
        )

    return _state.expected_columns


# ============================================================
# GET MODEL INFO
# ============================================================

def get_model_info() -> dict[str, Any]:
    """
    Return information about the loaded model.
    """

    if _state.model is None:

        return {
            "loaded": False,
            "source": "mlflow",
        }

    pipeline = _state.model

    info: dict[str, Any] = {
        "loaded": True,
        "source": _state.source,
        "model_uri": _state.model_uri,
        "version": _state.version,
        "model_name": Config.MODEL.MODEL_NAME,
        "pipeline_type": type(
            pipeline
        ).__name__,
        "expected_features": len(
            _state.expected_columns
        ),
        "expected_columns": list(
            _state.expected_columns
        ),
        "historical_features_used": False,
    }

    if hasattr(
        pipeline,
        "named_steps",
    ):

        info["pipeline_steps"] = list(
            pipeline.named_steps.keys()
        )

        model = pipeline.named_steps.get(
            "model"
        )

        if model is not None:

            info["model_type"] = type(
                model
            ).__name__

        preprocessor = (
            pipeline.named_steps.get(
                "preprocessor"
            )
        )

        if preprocessor is not None:

            info["preprocessor_type"] = type(
                preprocessor
            ).__name__

    return info


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
    )

    load_pipeline()

    state = get_state()

    print("\n" + "=" * 70)
    print("MODEL LOADED")
    print("=" * 70)

    print(
        "\nModel:",
        Config.MODEL.MODEL_NAME,
    )

    print(
        "Alias:",
        Config.MODEL.MODEL_ALIAS,
    )

    print(
        "Version:",
        state.version,
    )

    print(
        "Expected features:",
        len(state.expected_columns),
    )

    print("\nExpected columns:")

    for column in state.expected_columns:
        print(
            f"  - {column}"
        )

    print(
        "\nModel information:"
    )

    print(
        get_model_info()
    )

    print(
        "\nModel loading test passed."
    )