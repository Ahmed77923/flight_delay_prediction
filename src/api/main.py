from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.model_loader import (
    get_state,
    load_pipeline,
    get_expected_columns,
    get_model_info,
)

from src.api.schemas import (
    FlightRequest,
    PredictionResponse,
)

from src.features.build_feature import build_features


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
)


# ============================================================
# APPLICATION STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting Flight Delay Prediction API..."
    )

    try:

        # ----------------------------------------------------
        # Load MLflow pipeline
        # ----------------------------------------------------

        load_pipeline()

        state = get_state()

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        if state.model is None:
            raise RuntimeError(
                "MLflow model was not loaded."
            )

        # ----------------------------------------------------
        # Get expected columns
        # ----------------------------------------------------

        expected_columns = get_expected_columns()

        logger.info(
            "Expected model input columns: %d",
            len(expected_columns),
        )

        logger.info(
            "Model loaded successfully."
        )

        logger.info(
            "Model type: %s",
            type(state.model).__name__,
        )

    except Exception:

        logger.exception(
            "Failed to initialize API."
        )

        raise

    yield

    logger.info(
        "Flight Delay Prediction API stopped."
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Flight Arrival Delay Prediction API",
    description=(
        "Predict flight arrival delay using "
        "the trained MLflow LightGBM pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Flight Arrival Delay Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/model",
            "/predict",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    state = get_state()

    if state.model is None:

        return {
            "status": "unhealthy",
            "model_loaded": False,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(
            state.model
        ).__name__,
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model")
def model_info():

    state = get_state()

    if state.model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    return get_model_info()


# ============================================================
# PREDICT
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    flight: FlightRequest,
):

    # --------------------------------------------------------
    # Get shared model state
    # --------------------------------------------------------

    state = get_state()

    if state.model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    try:

        # ====================================================
        # 1. REQUEST → DATAFRAME
        # ====================================================

        df = pd.DataFrame(
            [flight.model_dump()]
        )

        logger.info(
            "Received flight request: %s",
            df.to_dict(
                orient="records"
            )[0],
        )

        # ====================================================
        # 2. FEATURE ENGINEERING
        # ====================================================

        features = build_features(
            df
        )

        logger.info(
            "Feature engineering completed. "
            "Shape: %s",
            features.shape,
        )

        logger.info(
            "Generated features: %s",
            list(features.columns),
        )

        # ====================================================
        # 3. EXPECTED MODEL FEATURES
        # ====================================================

        expected_columns = (
            get_expected_columns()
        )

        missing = [
            column
            for column in expected_columns
            if column not in features.columns
        ]

        if missing:

            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Missing model features",
                    "missing_columns": missing,
                },
            )

        # ----------------------------------------------------
        # Select exactly the features used during training
        # ----------------------------------------------------

        X = features[
            expected_columns
        ]

        # ====================================================
        # 4. CHECK NaN
        # ====================================================

        if X.isna().any().any():

            null_columns = (
                X.columns[
                    X.isna().any()
                ].tolist()
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Model input contains NaN",
                    "columns": null_columns,
                },
            )

        logger.info(
            "Prediction input shape: %s",
            X.shape,
        )

        # ====================================================
        # 5. PREDICTION
        # ====================================================

        prediction = float(
            state.model.predict(X)[0]
        )

        logger.info(
            "Prediction: %.4f",
            prediction,
        )

        # ====================================================
        # 6. RESPONSE
        # ====================================================

        return PredictionResponse(
            prediction=round(
                prediction,
                2,
            ),
            target="ARR_DELAY",
            model="lightgbm",
            source="mlflow",
        )

    except HTTPException:

        raise

    except ValueError as exc:

        logger.exception(
            "Feature validation failed."
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc