from pathlib import Path


class DataConfig:
    """Configuration related to data."""

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

    TARGET = "ARR_DELAY"


class ModelConfig:
    """Configuration related to machine learning models."""

    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MODEL_DIR = (
        DataConfig.PROJECT_ROOT / "models"
    )


class PreprocessingConfig:
    """Configuration for preprocessing."""

    CATEGORICAL_FEATURES = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
    ]

    NUMERICAL_FEATURES = [
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "year",
        "month",
        "quarter",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend",
        "departure_hour",
        "departure_minute",
        "arrival_hour",
        "arrival_minute",
    ]


class MLflowConfig:
    """Configuration related to MLflow."""

    EXPERIMENT_NAME = "flight-delay-regression"

    TRACKING_URI = "mlruns"


class APIConfig:
    """Configuration related to FastAPI."""

    HOST = "0.0.0.0"
    PORT = 8000


class Config:
    """
    Main configuration class.

    Combines all project configurations.
    """

    DATA = DataConfig
    MODEL = ModelConfig
    PREPROCESSING = PreprocessingConfig
    MLFLOW = MLflowConfig
    API = APIConfig