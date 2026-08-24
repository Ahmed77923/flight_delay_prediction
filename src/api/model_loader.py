import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from config.config import Config

load_dotenv()

logger = logging.getLogger(__name__)


def load_pipeline() -> Any:
    tracking_uri = os.getenv(Config.MLFLOW.TRACKING_URI)
    model_uri = os.getenv(Config.MLFLOW.MODEL_URI, "")

    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not configured.")
    if not model_uri:
        raise RuntimeError("MLFLOW_MODEL_URI is not configured.")

    logger.info("Loading MLflow model from %s", model_uri)
    mlflow.set_tracking_uri(tracking_uri)
    pipeline = mlflow.sklearn.load_model(model_uri)

    if not hasattr(pipeline, "predict"):
        raise RuntimeError("The MLflow artifact does not expose predict().")

    logger.info("MLflow model loaded successfully")
    return pipeline


def get_expected_columns(pipeline: Any) -> list[str]:
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    expected_columns = getattr(preprocessor, "feature_names_in_", None)
    if expected_columns is None:
        raise RuntimeError("The loaded pipeline has no fitted input column metadata.")
    return list(expected_columns)


def get_model_path() -> Path:
    model_uri = os.getenv(Config.MLFLOW.MODEL_URI, "")
    return Path(model_uri)


if __name__ == "__main__":
    try:
        pipeline = load_pipeline()
        expected_columns = get_expected_columns(pipeline)
        model_path = get_model_path()

        print("Expected columns: %s", expected_columns)
        print("Model path: %s", model_path)

    except Exception as e:
        logger.error("Error loading the model: %s", e)