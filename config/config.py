from pathlib import Path
from typing import List, Type


class DataConfig:
    """Configuration related to data."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

    TARGET: str = "ARR_DELAY"


class ModelConfig:
    """Configuration related to machine learning models."""

    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    MODEL_DIR: Path = (
        DataConfig.PROJECT_ROOT / "models"
    )


class PreprocessingConfig:
    """
    Configuration for preprocessing.
    """
    CATEGORICAL_FEATURES: List[str] = [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "route",
        "departure_period",
        "carrier_origin",
        # "carrier_departure_period"
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
        
        "departure_hour_sin",
        "departure_hour_cos",
        
        "day_of_week_sin",
        "day_of_week_cos",
        
        "month_sin",
        "month_cos",

        "arrival_hour",
        "arrival_minute",

        "distance_log",
        "is_peak_departure",
        # "scheduled_arrival",
        # "scheduled_departure",
        
        "carrier_historical_delay",
        "route_historical_delay",
        
        "origin_historical_delay",
        
        "origin_recent_delay_7",
        "origin_recent_delay_30",
           
        "carrier_recent_delay_7",
        "carrier_recent_delay_30",
        
        "route_recent_delay_7",
        "route_recent_delay_30",
        
        "carrier_origin_historical_delay",
        "aircraft_previous_delay"    
    
    ]
        
# numerical__departure_time_minutes,0
# numerical__departure_hour,0

# numerical__departure_minute,0
# numerical__arrival_hour,0
# numerical__arrival_minute,0
# numerical__is_peak_departure,0
# numerical__arrival_time_minutes,0
# numerical__quarter
# numerical__distance_log
    
class MLflowConfig:
    """Configuration related to MLflow."""

    EXPERIMENT_NAME: str = "flight-delay-regression"

    TRACKING_URI: str = "mlruns"


class APIConfig:
    """Configuration related to FastAPI."""

    HOST: str = "0.0.0.0"
    PORT: int = 8000


class Config:
    """
    Main configuration class.

    Combines all project configurations.
    """

    DATA: Type[DataConfig] = DataConfig
    MODEL: Type[ModelConfig] = ModelConfig
    PREPROCESSING: Type[PreprocessingConfig] = PreprocessingConfig
    MLFLOW: Type[MLflowConfig] = MLflowConfig
    API: Type[APIConfig] = APIConfig