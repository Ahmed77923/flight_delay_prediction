from contextlib import asynccontextmanager
import logging
import os
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from config.config import Config
from src.features.build_feature import build_features
from src.api.model_loader import get_expected_columns, load_pipeline
from src.api.schemas import FlightRequest, HealthResponse, PredictionResponse, request_to_dataframe

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        pipeline = load_pipeline()
        app.state.pipeline = pipeline
        app.state.expected_columns = get_expected_columns(pipeline)
    except Exception as exc:
        logger.exception("Unable to load MLflow model")
        raise RuntimeError("Unable to load the configured MLflow model.") from exc
    yield
    app.state.pipeline = None


app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_pipeline(request: Request) -> Any:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return pipeline


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {"status": "ok", "service": "Flight Delay Prediction API"}


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    loaded = getattr(request.app.state, "pipeline", None) is not None
    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(flight: FlightRequest, request: Request) -> PredictionResponse:
    """Predict flight delay for a single flight record.

    Flow:
    1. Validate request with Pydantic (FlightRequest)
    2. Convert to DataFrame with ARR_DELAY as float64 NaN (request_to_dataframe)
    3. Apply existing feature engineering (build_features)
    4. Predict using the loaded MLflow pipeline
    """
    pipeline = _get_pipeline(request)

    try:
        # Step 1-2: Request boundary ensures ARR_DELAY is float64 NaN
        raw_frame = request_to_dataframe(flight)

        # Step 3: Existing feature engineering (unchanged)
        features = build_features(raw_frame)

        # Step 4: Predict with saved pipeline
        expected_columns = request.app.state.expected_columns
        missing_columns = [
            column for column in expected_columns if column not in features.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Feature engineering did not produce required columns: {missing_columns}"
            )
        prediction = float(pipeline.predict(features[expected_columns])[0])
        return PredictionResponse(
            prediction=prediction,
            target=Config.DATA.TARGET,
            model="lightgbm",
            source="mlflow",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=422,
            detail="Feature engineering failed for the supplied flight data.",
        ) from exc
