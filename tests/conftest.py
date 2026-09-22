import numpy as np
import pandas as pd
import pytest

from config.config import Config
from src.api.model_loader import get_state
from src.batch.store import BatchStore


REQUIRED = [
    "FL_DATE",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
]


class FakeBatchModel:
    """Deterministic per-row prediction (DISTANCE / 100) that counts calls."""

    def __init__(self):
        self.calls = 0
        self.rows_seen = 0

    def predict(self, frame):
        self.calls += 1
        self.rows_seen += len(frame)
        return frame["DISTANCE"].to_numpy(dtype=float) / 100


def make_flights(n: int = 10) -> pd.DataFrame:
    """Small flight table, as text, with extra non-model columns."""

    origins = ["JFK", "ATL", "ORD", "LAX", "SFO"]
    dests = ["LAX", "JFK", "DFW", "SEA", "MIA"]

    return pd.DataFrame(
        {
            "FL_DATE": [f"2026-09-{(i % 28) + 1:02d}" for i in range(n)],
            "OP_UNIQUE_CARRIER": (["AA", "DL", "UA", "WN", "B6"] * (n // 5 + 1))[:n],
            "ORIGIN": (origins * (n // 5 + 1))[:n],
            "DEST": (dests * (n // 5 + 1))[:n],
            # Leading zero must survive the round trip.
            "CRS_DEP_TIME": [f"{6 + i % 12:02d}30" for i in range(n)],
            "CRS_ARR_TIME": [f"{9 + i % 12:02d}45" for i in range(n)],
            "CRS_ELAPSED_TIME": [str(120 + 10 * i) for i in range(n)],
            "DISTANCE": [str(500 + 100 * i) for i in range(n)],
            "ARR_DELAY": [str(i - 5) for i in range(n)],
            "CANCELLED": ["0.00"] * n,
        }
    )[
        [
            "FL_DATE",
            "OP_UNIQUE_CARRIER",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
            "ARR_DELAY",
            "CANCELLED",
            "CRS_ELAPSED_TIME",
            "DISTANCE",
        ]
    ]


@pytest.fixture
def flights():
    return make_flights


@pytest.fixture(autouse=True)
def batch_dirs(tmp_path, monkeypatch):
    """Point Config.BATCH at tmp dirs so tests never touch data/batch."""

    root = tmp_path / "batch"

    monkeypatch.setattr(Config.BATCH, "BATCH_DIR", root)
    monkeypatch.setattr(Config.BATCH, "INPUT_DIR", root / "input")
    monkeypatch.setattr(Config.BATCH, "OUTPUT_DIR", root / "output")
    monkeypatch.setattr(Config.BATCH, "METADATA_DIR", root / "metadata")

    return root


@pytest.fixture
def store(batch_dirs):
    return BatchStore.from_config()


@pytest.fixture
def fake_model(monkeypatch):
    """Install a fake model in the shared state (restored after the test)."""

    model = FakeBatchModel()
    state = get_state()

    monkeypatch.setattr(state, "model", model)
    monkeypatch.setattr(
        state,
        "expected_columns",
        Config.PREPROCESSING.CATEGORICAL_FEATURES
        + Config.PREPROCESSING.NUMERICAL_FEATURES,
    )
    monkeypatch.setattr(state, "model_uri", "fake://model")
    monkeypatch.setattr(state, "version", "test-1")

    return model


@pytest.fixture
def write_csv(tmp_path):
    def _write(df: pd.DataFrame, name: str = "flights.csv"):
        path = tmp_path / name
        df.to_csv(path, index=False)
        return path

    return _write
