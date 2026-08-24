from pathlib import Path
from typing import List, Type


class DataConfig:
    """Configuration related to data."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    DATA_PATH: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_PATH / "raw"
    PROCESSED_DATA_DIR: Path = DATA_PATH / "processed"

    TARGET: str = "ARR_DELAY"


class ModelConfig:
    """Configuration related to machine learning models."""

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    MODEL_NAME: str = "flight-arr-delay"
    MODEL_DIR: Path = DataConfig.PROJECT_ROOT / "models"


class FeatureConfig:
    """Configuration for feature engineering and online feature serving."""

    FEATURE_STATE_DIR: Path = (
        DataConfig.PROJECT_ROOT / "models" / "feature_state"
    )

    FEATURE_STATE_FILE: Path = (
        FEATURE_STATE_DIR / "feature_state.joblib"
    )

    RECENT_WINDOWS: tuple[int, ...] = (7, 30)


class PreprocessingConfig:
    """Configuration for model preprocessing."""

    CATEGORICAL_FEATURES: List[str] = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "route",
        "departure_period",
        "carrier_origin",
    ]

    NUMERICAL_FEATURES: List[str] = [
        "CRS_ELAPSED_TIME",
        "DISTANCE",

        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend",

        "departure_hour",
        "departure_minute",
        "arrival_hour",
        "arrival_minute",

        "departure_hour_sin",
        "departure_hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",

        "distance_log",
        "is_peak_departure",

        "carrier_historical_delay",
        "carrier_recent_delay_7",
        "carrier_recent_delay_30",

        "route_historical_delay",
        "route_recent_delay_7",
        "route_recent_delay_30",

        "origin_historical_delay",
        "origin_recent_delay_7",
        "origin_recent_delay_30",

        "carrier_origin_historical_delay",

        "aircraft_previous_delay",
    ]


class MLflowConfig:
    """Configuration related to MLflow."""

    TRACKING_URI: str = "MLFLOW_TRACKING_URI"
    MODEL_URI: str = "MLFLOW_MODEL_URI"

    EXPERIMENT_NAME: str = (
        "flight_arr_delay_prediction_categorical_features_V3"
    )


class APIConfig:
    """Configuration related to FastAPI."""

    HOST: str = "0.0.0.0"
    PORT: int = 8000


class Config:
    """Main project configuration."""

    DATA: Type[DataConfig] = DataConfig
    MODEL: Type[ModelConfig] = ModelConfig
    FEATURES: Type[FeatureConfig] = FeatureConfig
    PREPROCESSING: Type[PreprocessingConfig] = PreprocessingConfig
    MLFLOW: Type[MLflowConfig] = MLflowConfig
    API: Type[APIConfig] = APIConfig