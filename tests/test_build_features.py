import inspect

import pandas as pd

from config.config import Config
from src.features.build_feature import build_features, MODEL_FEATURES


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


def sample_request_frame() -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "FL_DATE": "2026-08-26",
                "CRS_DEP_TIME": 800,
                "CRS_ARR_TIME": 1100,
                "CRS_ELAPSED_TIME": 180,
                "DISTANCE": 2475,
                "OP_UNIQUE_CARRIER": "AA",
                "ORIGIN": "JFK",
                "DEST": "LAX",
            }
        ]
    )


# ============================================================
# build_features() WORKS WITHOUT history
# ============================================================

def test_build_features_does_not_require_history():

    features = build_features(sample_request_frame())

    assert len(features) == 1


def test_build_features_history_parameter_is_optional():

    signature = inspect.signature(build_features)

    assert signature.parameters["history"].default is None


# ============================================================
# NO HISTORICAL FEATURES ARE PRODUCED
# ============================================================

def test_model_features_contain_no_historical_features():

    assert HISTORICAL_FEATURES.isdisjoint(set(MODEL_FEATURES))


def test_built_features_contain_no_historical_columns():

    features = build_features(sample_request_frame())

    assert HISTORICAL_FEATURES.isdisjoint(set(features.columns))


# ============================================================
# ALL CONFIGURED MODEL FEATURES ARE PRESENT
# ============================================================

def test_build_features_produces_all_configured_features():

    features = build_features(sample_request_frame())

    model_features = (
        Config.PREPROCESSING.CATEGORICAL_FEATURES
        + Config.PREPROCESSING.NUMERICAL_FEATURES
    )

    missing = [
        column
        for column in model_features
        if column not in features.columns
    ]

    assert missing == []
