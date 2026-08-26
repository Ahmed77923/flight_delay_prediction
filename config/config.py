from pathlib import Path
from typing import List, Type

from dotenv import load_dotenv


load_dotenv()
import os

class DataConfig:
    """Configuration related to data."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    DATA_PATH: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_PATH / "raw"
    PROCESSED_DATA_DIR: Path = DATA_PATH / "processed"

    TARGET: str = "ARR_DELAY"

    TRAINING_YEAR: int = int(
        os.getenv("TRAINING_YEAR", "2025")
    )


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
        "quarter",
        "day",
        "day_of_week",
        "week_of_year",
        "is_weekend",

        "departure_hour",
        "departure_minute",
        "departure_time_minutes",
        "arrival_hour",
        "arrival_minute",
        "arrival_time_minutes",

        "departure_hour_sin",
        "departure_hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",

        "distance_log",
        "is_peak_departure",

        # Historical features  
        # i have problem with these features, so i will not use them for now
        # "carrier_historical_delay",
        # "route_historical_delay",
        # "origin_historical_delay",
        # "carrier_origin_historical_delay",
        # "aircraft_previous_delay",
        # "carrier_recent_delay_7",
        # "carrier_recent_delay_30",
        # "route_recent_delay_7",
        # "route_recent_delay_30",
        # "origin_recent_delay_7",
        # "origin_recent_delay_30",
    ]



class MLflowConfig:
    """Configuration related to MLflow."""

    TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
         "http://127.0.0.1:5000",
    )

    MODEL_URI: str = os.getenv(
        "MLFLOW_MODEL_URI",
        "runs:/94c6dd2a7b5340849733f1089e979ae7/model",
    )

    EXPERIMENT_NAME: str = (
        "flight_arr_delay_prediction_V4"
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