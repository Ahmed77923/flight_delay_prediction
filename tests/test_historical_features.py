from datetime import date

import pandas as pd

from src.features.historical_features import (
    HISTORICAL_COLUMNS,
    HistoricalFeatureIndex,
    normalize_to_training_year,
)


def history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FL_DATE": ["2025-08-20", "2025-08-24", "2025-08-25"],
            "CRS_DEP_TIME": [800, 900, 1000],
            "OP_UNIQUE_CARRIER": ["AA", "AA", "DL"],
            "ORIGIN": ["DFW", "DFW", "JFK"],
            "DEST": ["ORD", "ORD", "LAX"],
            "TAIL_NUM": ["N101NN", "N101NN", "N999XX"],
            "ARR_DELAY": [10.0, 30.0, 99.0],
        }
    )


def test_normalize_to_training_year() -> None:
    assert normalize_to_training_year(date(2027, 1, 3)).year == 2025
    assert normalize_to_training_year(date(2026, 8, 25)).date() == date(
        2025, 8, 25
    )


def test_index_uses_only_prior_records_and_latest_aircraft_delay() -> None:
    index = HistoricalFeatureIndex.from_frame(history_frame())
    flight = pd.Series(
        {
            "FL_DATE": date(2026, 8, 25),
            "CRS_DEP_TIME": 1100,
            "OP_UNIQUE_CARRIER": "AA",
            "ORIGIN": "DFW",
            "DEST": "ORD",
            "TAIL_NUM": "N101NN",
        }
    )

    features = index.features_for(flight)

    assert features["carrier_historical_delay"] == 20.0
    assert features["aircraft_previous_delay"] == 30.0
    assert features["route_recent_delay_7"] == 20.0
    assert list(features) == HISTORICAL_COLUMNS


def test_index_returns_zero_for_unknown_entities() -> None:
    index = HistoricalFeatureIndex.from_frame(history_frame())
    flight = pd.Series(
        {
            "FL_DATE": date(2026, 8, 25),
            "CRS_DEP_TIME": 1100,
            "OP_UNIQUE_CARRIER": "UA",
            "ORIGIN": "SEA",
            "DEST": "MIA",
            "TAIL_NUM": "UNKNOWN",
        }
    )

    features = index.features_for(flight)

    assert all(value == 0.0 for value in features.values())
