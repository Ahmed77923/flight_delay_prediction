import pytest
from fastapi.testclient import TestClient

from config.config import Config
from src.api import main
from src.api.model_loader import get_state


# ============================================================
# FAKE MODEL
# ============================================================

class FakePipeline:

    def predict(self, frame):
        return [12.45]


# ============================================================
# TEST CLIENT
# ============================================================

@pytest.fixture
def client(monkeypatch):

    def fake_load_pipeline():
        # Mirrors what a real load_pipeline() does: populate the
        # shared model state used by both /health and /predict.
        state = get_state()
        state.model = FakePipeline()
        state.source = "test"
        state.expected_columns = (
            Config.PREPROCESSING.CATEGORICAL_FEATURES
            + Config.PREPROCESSING.NUMERICAL_FEATURES
        )

    monkeypatch.setattr(
        main,
        "load_pipeline",
        fake_load_pipeline,
    )

    with TestClient(main.app) as api_client:
        yield api_client


# ============================================================
# VALID REQUEST
# ============================================================

def valid_payload():

    return {
        "FL_DATE": "2026-08-22",
        "CRS_DEP_TIME": 800,
        "CRS_ARR_TIME": 1100,
        "CRS_ELAPSED_TIME": 180,
        "DISTANCE": 2475,
        "OP_UNIQUE_CARRIER": "AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
    }


# ============================================================
# HEALTH
# ============================================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["model_loaded"] is True


# ============================================================
# PREDICT
# ============================================================

def test_predict(client):

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 200

    assert response.json() == {
        "prediction": 12.45,
        "target": "ARR_DELAY",
        "model": "lightgbm",
        "source": "mlflow",
    }


# ============================================================
# MISSING FIELD
# ============================================================

def test_predict_missing_origin(client):

    payload = valid_payload()

    del payload["ORIGIN"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# ARR_DELAY MUST NOT BE ACCEPTED
# ============================================================

def test_predict_rejects_arr_delay(client):

    payload = valid_payload()

    payload["ARR_DELAY"] = 5.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# ============================================================
# MODEL INFO
# ============================================================

def test_model_info(client):

    response = client.get("/model")

    assert response.status_code == 200

    body = response.json()

    assert body["loaded"] is True
    assert body["historical_features_used"] is False


# ============================================================
# NO HISTORICAL FEATURES REACH THE MODEL
# ============================================================

HISTORICAL_FEATURES = {
    "carrier_historical_delay",
    "carrier_recent_delay_7",
    "carrier_recent_delay_30",
    "route_recent_delay_7",
    "route_recent_delay_30",
    "route_historical_delay",
    "origin_historical_delay",
    "origin_recent_delay_7",
    "origin_recent_delay_30",
    "carrier_origin_historical_delay",
    "aircraft_previous_delay",
}


def test_expected_columns_have_no_historical_features(client):

    response = client.get("/model")

    expected_columns = set(
        response.json()["expected_columns"]
    )

    assert expected_columns.isdisjoint(HISTORICAL_FEATURES)


def test_config_features_have_no_historical_features():

    model_features = set(
        Config.PREPROCESSING.CATEGORICAL_FEATURES
        + Config.PREPROCESSING.NUMERICAL_FEATURES
    )

    assert model_features.isdisjoint(HISTORICAL_FEATURES)