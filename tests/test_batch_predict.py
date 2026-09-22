import numpy as np
import pandas as pd
import pytest

from config.config import Config
from src.api import model_loader
from src.api.model_loader import get_state
from src.batch import predict as batch_predict
from src.batch.errors import (
    BatchInputError,
    BatchOutputError,
    BatchPredictionError,
)
from src.batch.predict import (
    PredictionWriter,
    build_batch_features,
    count_rows,
    inspect_input_file,
    predict_batch,
    predict_file,
    save_predictions,
    validate_batch_input,
)
from src.inference.predictor import prepare_model_input


PREDICTION_COLUMN = Config.BATCH.PREDICTION_COLUMN


# ============================================================
# INPUT VALIDATION
# ============================================================

def test_validate_returns_typed_frame(flights):
    typed = validate_batch_input(flights(5))

    assert typed["CRS_DEP_TIME"].dtype == "int64"
    assert typed["CRS_ARR_TIME"].dtype == "int64"
    assert typed["DISTANCE"].dtype == "float64"
    assert typed["CRS_ELAPSED_TIME"].dtype == "float64"
    # "0630" -> 630, the same value the online API receives as an int.
    assert typed["CRS_DEP_TIME"].iloc[0] == 630


def test_validate_reports_all_missing_columns_sorted(flights):
    df = flights(3).drop(columns=["ORIGIN", "DEST"])

    with pytest.raises(BatchInputError) as exc:
        validate_batch_input(df)

    assert exc.value.message == "Missing required columns: DEST, ORIGIN"


def test_validate_rejects_empty_frame(flights):
    with pytest.raises(BatchInputError, match="no data rows"):
        validate_batch_input(flights(3).iloc[0:0])


def test_validate_rejects_non_numeric_with_row_numbers(flights):
    df = flights(5)
    df.loc[2, "DISTANCE"] = "abc"

    with pytest.raises(BatchInputError) as exc:
        validate_batch_input(df)

    # Data row index 2 is row 4 of the file (header is row 1).
    assert "DISTANCE" in exc.value.message
    assert "row 4" in exc.value.message


def test_validate_row_numbers_account_for_chunk_offset(flights):
    df = flights(3)
    df.loc[0, "DISTANCE"] = ""

    with pytest.raises(BatchInputError, match="row 102"):
        validate_batch_input(df, start_row=100)


def test_validate_rejects_fractional_time(flights):
    df = flights(3)
    df.loc[1, "CRS_DEP_TIME"] = "800.5"

    with pytest.raises(BatchInputError, match="CRS_DEP_TIME"):
        validate_batch_input(df)


def test_validate_rejects_blank_text(flights):
    df = flights(3)
    df.loc[0, "ORIGIN"] = "  "

    with pytest.raises(BatchInputError, match="ORIGIN"):
        validate_batch_input(df)


def test_validate_rejects_invalid_date(flights):
    df = flights(3)
    df.loc[1, "FL_DATE"] = "not-a-date"

    with pytest.raises(BatchInputError, match="FL_DATE"):
        validate_batch_input(df)


def test_inspect_missing_file(tmp_path):
    with pytest.raises(BatchInputError, match="not found"):
        inspect_input_file(tmp_path / "nope.csv")


def test_inspect_rejects_non_csv(tmp_path):
    path = tmp_path / "flights.txt"
    path.write_text("a,b\n1,2\n")

    with pytest.raises(BatchInputError, match=r"\.csv"):
        inspect_input_file(path)


def test_inspect_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")

    with pytest.raises(BatchInputError, match="empty"):
        inspect_input_file(path)


def test_inspect_rejects_invalid_encoding(tmp_path):
    path = tmp_path / "binary.csv"
    path.write_bytes(b"\xff\xfe\x00\x81\x82,\xff\n\xff\xff\n")

    with pytest.raises(BatchInputError):
        inspect_input_file(path)


def test_inspect_rejects_existing_prediction_column(flights, write_csv):
    df = flights(3)
    df[PREDICTION_COLUMN] = "1"

    with pytest.raises(BatchInputError, match=PREDICTION_COLUMN):
        inspect_input_file(write_csv(df))


def test_inspect_returns_columns(flights, write_csv):
    columns = inspect_input_file(write_csv(flights(3)))

    assert "ORIGIN" in columns and "ARR_DELAY" in columns


def test_count_rows(flights, write_csv):
    assert count_rows(write_csv(flights(7))) == 7


# ============================================================
# BATCH FEATURE ENGINEERING = ONLINE FEATURE ENGINEERING
# ============================================================

def _expected_columns():
    return (
        Config.PREPROCESSING.CATEGORICAL_FEATURES
        + Config.PREPROCESSING.NUMERICAL_FEATURES
    )


def test_batch_features_have_expected_columns(flights):
    X = build_batch_features(flights(6), _expected_columns())

    assert list(X.columns) == _expected_columns()
    assert len(X) == 6
    assert not X.isna().any().any()


def test_batch_features_match_online_features(flights):
    """
    The online API builds a frame from FlightRequest (ints / floats /
    datetimes) and calls prepare_model_input(). A batch row must yield
    exactly the same model input.
    """

    raw = flights(6)

    online_frame = pd.DataFrame(
        {
            "FL_DATE": pd.to_datetime(raw["FL_DATE"]),
            "CRS_DEP_TIME": raw["CRS_DEP_TIME"].astype(int),
            "CRS_ARR_TIME": raw["CRS_ARR_TIME"].astype(int),
            "CRS_ELAPSED_TIME": raw["CRS_ELAPSED_TIME"].astype(float),
            "DISTANCE": raw["DISTANCE"].astype(float),
            "OP_UNIQUE_CARRIER": raw["OP_UNIQUE_CARRIER"],
            "ORIGIN": raw["ORIGIN"],
            "DEST": raw["DEST"],
        }
    )

    online = prepare_model_input(online_frame, _expected_columns())
    batch = build_batch_features(raw, _expected_columns())

    pd.testing.assert_frame_equal(batch, online)


def test_batch_features_reject_invalid_hhmm(flights):
    df = flights(3)
    df.loc[1, "CRS_DEP_TIME"] = "2575"

    with pytest.raises(BatchInputError, match="HHMM"):
        build_batch_features(df, _expected_columns())


def test_batch_features_reject_negative_distance(flights):
    df = flights(3)
    df.loc[0, "DISTANCE"] = "-5"

    with pytest.raises(BatchInputError, match="DISTANCE"):
        build_batch_features(df, _expected_columns())


# ============================================================
# PREDICTION
# ============================================================

def test_predict_batch_calls_model_once_for_whole_frame(flights, fake_model):
    X = build_batch_features(flights(10), _expected_columns())

    predictions = predict_batch(fake_model, X)

    assert fake_model.calls == 1
    assert fake_model.rows_seen == 10
    assert len(predictions) == 10


def test_predict_batch_rounds_like_online_response():
    class Model:
        def predict(self, frame):
            return np.array([1.23456, -2.5678, 7.0])

    predictions = predict_batch(Model(), pd.DataFrame({"x": [1, 2, 3]}))

    assert predictions.tolist() == [1.23, -2.57, 7.0]


def test_predict_batch_wraps_model_errors():
    class Broken:
        def predict(self, frame):
            raise RuntimeError("boom: secret internal detail")

    with pytest.raises(BatchPredictionError) as exc:
        predict_batch(Broken(), pd.DataFrame({"x": [1]}))

    assert "secret" not in exc.value.message


def test_predict_batch_rejects_wrong_length():
    class Short:
        def predict(self, frame):
            return np.array([1.0])

    with pytest.raises(BatchPredictionError):
        predict_batch(Short(), pd.DataFrame({"x": [1, 2]}))


# ============================================================
# FILE -> FILE + OUTPUT
# ============================================================

def test_predict_file_preserves_columns_and_adds_prediction(
    flights, write_csv, fake_model, tmp_path
):
    source = flights(10)
    input_path = write_csv(source)
    output_path = tmp_path / "out" / "predictions.csv"

    rows = predict_file(input_path, output_path, get_state())

    assert rows == 10

    result = pd.read_csv(output_path, dtype=str, keep_default_na=False)

    assert list(result.columns) == list(source.columns) + [PREDICTION_COLUMN]

    # Original values are untouched, including "0630"-style times.
    pd.testing.assert_frame_equal(
        result[list(source.columns)], source, check_dtype=False
    )
    assert result["CRS_DEP_TIME"].iloc[0] == "0630"

    # Predictions line up with their rows (fake model: DISTANCE / 100).
    expected = (source["DISTANCE"].astype(float) / 100).round(2)
    np.testing.assert_allclose(
        result[PREDICTION_COLUMN].astype(float), expected
    )


def test_predict_file_chunks_match_single_pass(
    flights, write_csv, fake_model, tmp_path
):
    input_path = write_csv(flights(10))

    predict_file(input_path, tmp_path / "one.csv", get_state(), chunk_size=100)
    calls_single = fake_model.calls

    predict_file(input_path, tmp_path / "chunked.csv", get_state(), chunk_size=3)

    assert fake_model.calls - calls_single == 4  # ceil(10 / 3): never per row
    assert (tmp_path / "one.csv").read_text() == (
        tmp_path / "chunked.csv"
    ).read_text()


def test_predict_file_reports_progress(flights, write_csv, fake_model, tmp_path):
    seen = []

    predict_file(
        write_csv(flights(10)),
        tmp_path / "o.csv",
        get_state(),
        chunk_size=4,
        on_chunk=lambda n, done: seen.append((n, done)),
    )

    assert seen == [(4, 4), (4, 8), (2, 10)]


def test_predict_file_failure_leaves_no_output(
    flights, write_csv, fake_model, tmp_path
):
    df = flights(10)
    df.loc[8, "DISTANCE"] = "oops"  # bad row lands in the 3rd chunk
    output_path = tmp_path / "out" / "predictions.csv"

    with pytest.raises(BatchInputError) as exc:
        predict_file(write_csv(df), output_path, get_state(), chunk_size=4)

    assert "row 10" in exc.value.message
    assert not output_path.exists()
    assert list(output_path.parent.iterdir()) == []  # temp file cleaned up


def test_save_predictions_writes_csv(flights, tmp_path):
    df = flights(3)
    output_path = tmp_path / "saved.csv"

    save_predictions(df, np.array([1.5, 2.5, 3.5]), output_path)

    result = pd.read_csv(output_path)

    assert result[PREDICTION_COLUMN].tolist() == [1.5, 2.5, 3.5]
    assert len(result) == 3


def test_writer_reports_unwritable_output(tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")

    with pytest.raises(BatchOutputError):
        with PredictionWriter(blocker / "out.csv", PREDICTION_COLUMN):
            pass


def test_model_load_failure_is_reported(monkeypatch):
    def broken():
        raise RuntimeError("cannot open /secret/path")

    monkeypatch.setattr(batch_predict, "ensure_model_loaded", broken)

    with pytest.raises(batch_predict.BatchModelError) as exc:
        batch_predict.load_model()

    assert "/secret/path" not in exc.value.message


# ============================================================
# CLI
# ============================================================

def test_cli_success(flights, write_csv, fake_model, tmp_path, capsys):
    output_path = tmp_path / "cli_out.csv"

    code = batch_predict.main(
        [
            "--input", str(write_csv(flights(6))),
            "--output", str(output_path),
            "--batch-id", "cli-test",
        ]
    )

    assert code == 0
    assert len(pd.read_csv(output_path)) == 6


def test_cli_invalid_input_returns_nonzero(
    flights, write_csv, fake_model, tmp_path, capsys
):
    df = flights(3).drop(columns=["ORIGIN"])

    code = batch_predict.main(
        [
            "--input", str(write_csv(df)),
            "--output", str(tmp_path / "o.csv"),
        ]
    )

    assert code == 1
    assert "Missing required columns: ORIGIN" in capsys.readouterr().err


def test_cli_rejects_bad_batch_id(flights, write_csv, tmp_path, capsys):
    code = batch_predict.main(
        [
            "--input", str(write_csv(flights(3))),
            "--output", str(tmp_path / "o.csv"),
            "--batch-id", "../evil",
        ]
    )

    assert code == 2


def test_cli_requires_input():
    with pytest.raises(SystemExit) as exc:
        batch_predict.main([])

    assert exc.value.code == 2


# ============================================================
# REAL MODEL: BATCH == ONLINE
# ============================================================

REAL_MODEL_DIR = (
    Config.DATA.PROJECT_ROOT / "models" / "model_artifact" / "model"
)


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "MLmodel").exists(),
    reason="bundled model artifact not present",
)
def test_batch_predictions_equal_online_predictions(
    flights, write_csv, tmp_path, monkeypatch
):
    """Same artifact + same features: batch output == POST /predict."""

    from fastapi.testclient import TestClient

    from src.api import main

    state = get_state()

    for attribute in (
        "model", "expected_columns", "model_uri", "version", "source",
    ):
        monkeypatch.setattr(state, attribute, getattr(state, attribute))

    monkeypatch.setattr(model_loader, "MODEL_URI", str(REAL_MODEL_DIR))

    source = flights(8)
    output_path = tmp_path / "real.csv"

    with TestClient(main.app) as client:
        predict_file(write_csv(source), output_path, get_state())

        batch = pd.read_csv(output_path)[PREDICTION_COLUMN].tolist()

        online = []

        for _, row in source.iterrows():
            response = client.post(
                "/predict",
                json={
                    "FL_DATE": row["FL_DATE"],
                    "CRS_DEP_TIME": int(row["CRS_DEP_TIME"]),
                    "CRS_ARR_TIME": int(row["CRS_ARR_TIME"]),
                    "CRS_ELAPSED_TIME": float(row["CRS_ELAPSED_TIME"]),
                    "DISTANCE": float(row["DISTANCE"]),
                    "OP_UNIQUE_CARRIER": row["OP_UNIQUE_CARRIER"],
                    "ORIGIN": row["ORIGIN"],
                    "DEST": row["DEST"],
                },
            )
            assert response.status_code == 200
            online.append(response.json()["prediction"])

    assert batch == online
