from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from config.config import Config
from src.api.schemas import FlightRequest, PredictionResponse
from src.api.model_loader import (
    get_expected_columns,
    get_model,
    get_state,
    load_model,
)
from src.features.build_feature import MODEL_FEATURES, build_features


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the MLflow model once when the API starts.
    """

    load_model()

    yield


app = FastAPI(
    title="Flight Arrival Delay Prediction API",
    description="Predict ARR_DELAY using the trained LightGBM model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Flight Arrival Delay Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    """
    Check whether the API and model are ready.
    """

    state = get_state()

    return {
        "status": "healthy",
        "model_loaded": state.get("model") is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: FlightRequest):
    """
    Predict ARR_DELAY for one flight.

    No historical data is used.
    """

    try:
        # --------------------------------------------------
        # 1. Convert user request to DataFrame
        # --------------------------------------------------

        payload = request.model_dump()
        payload[Config.DATA.TARGET] = 0.0

        df = pd.DataFrame([payload])

        # --------------------------------------------------
        # 2. Build normal features only
        # --------------------------------------------------

        features = build_features(df)

        # --------------------------------------------------
        # 3. Build exact fitted-pipeline input columns
        # --------------------------------------------------

        model = get_model()
        expected_columns = get_expected_columns(model)

        missing = [
            column
            for column in expected_columns
            if column not in features.columns
        ]

        categorical = set(
            Config.PREPROCESSING.CATEGORICAL_FEATURES
        )

        for column in missing:
            if column in categorical:
                features[column] = "UNKNOWN"
            else:
                features[column] = 0.0

        X = features[expected_columns]

        # --------------------------------------------------
        # 4. Prediction
        # --------------------------------------------------

        prediction = model.predict(X)
        print("\n========== API REQUEST ==========")
        print(df.to_dict(orient="records"))

        print("\n========== FEATURES ==========")
        print(
            features[MODEL_FEATURES]
            .to_string(index=False)
        )

        print("\n========== PREDICTION ==========")
        print(prediction)
        return PredictionResponse(
            prediction=float(prediction[0]),
            target="ARR_DELAY",
            model=type(model.named_steps["model"]).__name__,
            source="mlflow_pipeline",
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )