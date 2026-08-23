from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.schemas import FlightRequest, request_to_dataframe


class FakePreprocessor:
    feature_names_in_ = [
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
        "aircraft_previous_delay",
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "route",
        "departure_period",
        "carrier_origin",
    ]


class FakePipeline:
    named_steps = {"preprocessor": FakePreprocessor()}

    def predict(self, frame):
        assert list(frame.columns) == self.named_steps["preprocessor"].feature_names_in_
        return [12.45]


def client(monkeypatch):
    monkeypatch.setattr(main, "load_pipeline", lambda: FakePipeline())
    return TestClient(main.app)


def valid_payload():
    return {
        "FL_DATE": "2026-08-22",
        "CRS_DEP_TIME": 800,
        "CRS_ARR_TIME": 1100,
        "OP_UNIQUE_CARRIER": "AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "DISTANCE": 2475,
        "ARR_TIME": 1100,
        "TAIL_NUM": "N12345",
        "CRS_ELAPSED_TIME": 180,  # Required by build_features()
    }


def test_root(monkeypatch):
    with client(monkeypatch) as api_client:
        response = api_client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_health_reports_loaded_model(monkeypatch):
    with client(monkeypatch) as api_client:
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_uses_loaded_pipeline(monkeypatch):
    with client(monkeypatch) as api_client:
        response = api_client.post("/predict", json=valid_payload())
        assert response.status_code == 200
        assert response.json() == {
            "prediction": 12.45,
            "target": "ARR_DELAY",
            "model": "lightgbm",
            "source": "mlflow",
        }


def test_predict_rejects_missing_fields(monkeypatch):
    payload = valid_payload()
    del payload["ORIGIN"]
    with client(monkeypatch) as api_client:
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422


def test_model_loading_failure_is_clear(monkeypatch):
    def fail_load():
        raise RuntimeError("MLflow unavailable")

    monkeypatch.setattr(main, "load_pipeline", fail_load)
    try:
        TestClient(main.app).__enter__()
    except RuntimeError as exc:
        assert str(exc) == "Unable to load the configured MLflow model."
    else:
        raise AssertionError("Expected startup to fail when model loading fails")


def test_request_to_dataframe_creates_arr_delay_as_float64_nan():
    """Verify request_to_dataframe() ensures ARR_DELAY is float64 NaN."""
    flight = FlightRequest(
        FL_DATE=date(2026, 8, 22),
        CRS_ELAPSED_TIME=180,
        CRS_DEP_TIME=800,
        CRS_ARR_TIME=1100,
        OP_UNIQUE_CARRIER="AA",
        ORIGIN="JFK",
        DEST="LAX",
        DISTANCE=2475,
        ARR_TIME=1100,
        TAIL_NUM="N12345",
    )

    df = request_to_dataframe(flight)

    # Verify ARR_DELAY exists
    assert "ARR_DELAY" in df.columns
    # Verify dtype is float64
    assert df["ARR_DELAY"].dtype == "float64"
    # Verify value is NaN
    assert pd.isna(df["ARR_DELAY"].iloc[0])


def test_request_schema_does_not_accept_arr_delay():
    """Verify schema rejects extra fields (including if ARR_DELAY were provided)."""
    payload = {
        "FL_DATE": "2026-08-22",
        "CRS_ELAPSED_TIME": 180,
        "CRS_DEP_TIME": 800,
        "CRS_ARR_TIME": 1100,
        "OP_UNIQUE_CARRIER": "AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "DISTANCE": 2475,
        "ARR_TIME": 1100,
        "TAIL_NUM": "N12345",
        "ARR_DELAY": 5.0,  # This should be rejected
    }

    with pytest.raises(ValueError):
        FlightRequest(**payload)


def test_predict_does_not_accept_arr_delay_from_client(monkeypatch):
    """Verify /predict endpoint rejects ARR_DELAY if sent by client."""
    payload = {
        "FL_DATE": "2026-08-22",
        "CRS_ELAPSED_TIME": 180,
        "CRS_DEP_TIME": 800,
        "CRS_ARR_TIME": 1100,
        "OP_UNIQUE_CARRIER": "AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "DISTANCE": 2475,
        "ARR_TIME": 1100,
        "TAIL_NUM": "N12345",
        "ARR_DELAY": 5.0,  # Extra field
    }

    with client(monkeypatch) as api_client:
        response = api_client.post("/predict", json=payload)
        # Should reject because extra="forbid" in Pydantic config
        assert response.status_code == 422
