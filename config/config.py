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
    MODEL_ALIAS: str = "flight-arr-delay"
    MODEL_DIR: Path = DataConfig.PROJECT_ROOT / "models"


class PreprocessingConfig:
    """Configuration for model preprocessing."""

    CATEGORICAL_FEATURES: list[str] = [  # noqa: RUF012
        "OP_UNIQUE_CARRIER",                        
        "ORIGIN",                                  
        "DEST",                                    
        "route",
        "departure_period",
        "carrier_origin",

    ]

    NUMERICAL_FEATURES: List[str] = [  # noqa: RUF012
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
   
    ]

class MLflowConfig:
    """Configuration related to MLflow."""

    TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:1040",
    )

    MODEL_URI: str = os.getenv(
        "MLFLOW_MODEL_URI",
        "/app/model_artifact",
    )

    EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "flight_arr_delay_champion_model1",
    )


class APIConfig:
    """Configuration related to FastAPI."""

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Base URL HTTP clients (e.g. the Streamlit UI) use to reach the
    # FastAPI service. Kept separate from HOST/PORT, which describe the
    # bind address the server listens on, not necessarily how a client
    # reaches it.
    BASE_URL: str = os.getenv(
        "API_URL",
         "http://127.0.0.1:1041",
    )

class BatchConfig:
    """Configuration for Batch Inference / Offline Serving."""

    # Batch files live under data/batch/{input,output,metadata}. The root
    # can be relocated (e.g. onto a Docker volume) with BATCH_DATA_DIR.
    BATCH_DIR: Path = Path(
        os.getenv(
            "BATCH_DATA_DIR",
            str(DataConfig.DATA_PATH / "batch"),
        )
    )
    INPUT_DIR: Path = BATCH_DIR / "input"
    OUTPUT_DIR: Path = BATCH_DIR / "output"
    METADATA_DIR: Path = BATCH_DIR / "metadata"

    PREDICTION_COLUMN: str = "predicted_arr_delay"

    # Rows read, featurised and predicted per step. Bounds memory use on
    # very large files while keeping model.predict() fully vectorised.
    CHUNK_SIZE: int = int(os.getenv("BATCH_CHUNK_SIZE", "100000"))

    MAX_UPLOAD_MB: int = int(os.getenv("BATCH_MAX_UPLOAD_MB", "200"))

    # Batch jobs started through the API run in a background pool. One
    # worker keeps memory bounded and leaves CPU for online /predict.
    MAX_WORKERS: int = int(os.getenv("BATCH_MAX_WORKERS", "1"))


class Config:
    """Main project configuration."""

    DATA: Type[DataConfig] = DataConfig
    MODEL: Type[ModelConfig] = ModelConfig
    PREPROCESSING: Type[PreprocessingConfig] = PreprocessingConfig
    MLFLOW: Type[MLflowConfig] = MLflowConfig
    API: Type[APIConfig] = APIConfig
    BATCH: Type[BatchConfig] = BatchConfig