"""
Batch prediction: CSV in -> CSV with a prediction column out.

Reuses the online serving code path (src/inference/predictor.py):
same build_features(), same MLflow pipeline, same expected columns. Files
are processed in chunks; every chunk is a single vectorised model.predict().

CLI:

    python -m src.batch.predict \\
        --input data/batch/input/flights.csv \\
        --output data/batch/output/predictions.csv \\
        --batch-id september-2026
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from config.config import Config
from src.api.model_loader import ModelState
from src.batch.errors import (
    BatchError,
    BatchInputError,
    BatchModelError,
    BatchOutputError,
    BatchPredictionError,
)
from src.batch.schemas import (
    INTEGER_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TEXT_COLUMNS,
)
from src.inference.predictor import (
    PREDICTION_PRECISION,
    MissingModelFeaturesError,
    NullModelInputError,
    ensure_model_loaded,
    predict_frame,
    prepare_model_input,
)


logger = logging.getLogger(__name__)


# Every cell is read as text so the output preserves the input exactly
# (e.g. CRS_DEP_TIME "0600" keeps its leading zero). Values are converted
# to numbers only for the model frame, in validate_batch_input().
_CSV_READ_OPTIONS = {
    "dtype": str,
    "keep_default_na": False,
    "encoding": "utf-8-sig",
}

_MAX_ROWS_IN_MESSAGE = 5


# ============================================================
# INPUT VALIDATION
# ============================================================

def _describe_rows(mask: pd.Series, start_row: int) -> str:
    """'rows 3, 7, 12, 15, 20 and 4 more' - numbered as in a spreadsheet (header = 1)."""

    positions = np.flatnonzero(mask.to_numpy())
    numbers = [str(int(p) + start_row + 2) for p in positions]
    shown = ", ".join(numbers[:_MAX_ROWS_IN_MESSAGE])
    extra = len(numbers) - _MAX_ROWS_IN_MESSAGE

    label = "row" if len(numbers) == 1 else "rows"

    return f"{label} {shown}" + (f" and {extra} more" if extra > 0 else "")


def _missing_columns_error(columns) -> BatchInputError:
    missing = [c for c in REQUIRED_COLUMNS if c not in set(columns)]

    return BatchInputError(
        "Missing required columns: " + ", ".join(missing)
    )


def inspect_input_file(path: Path) -> list[str]:
    """
    Cheap checks on the file itself (existence, type, size, header) that
    run before the model is loaded. Returns the CSV column names.
    """

    path = Path(path)

    if not path.is_file():
        raise BatchInputError("Input file not found.")

    if path.suffix.lower() != ".csv":
        raise BatchInputError("Input must be a .csv file.")

    if path.stat().st_size == 0:
        raise BatchInputError("Input file is empty.")

    try:
        header = pd.read_csv(path, nrows=0, **_CSV_READ_OPTIONS)
    except pd.errors.EmptyDataError:
        raise BatchInputError("Input file is empty.") from None
    except UnicodeDecodeError:
        raise BatchInputError(
            "Input is not a valid UTF-8 CSV file."
        ) from None
    except (pd.errors.ParserError, ValueError) as exc:
        raise BatchInputError(f"Invalid CSV file: {exc}") from None

    columns = list(header.columns)

    if any(c not in set(columns) for c in REQUIRED_COLUMNS):
        raise _missing_columns_error(columns)

    prediction_column = Config.BATCH.PREDICTION_COLUMN

    if prediction_column in columns:
        raise BatchInputError(
            f"Input already contains a '{prediction_column}' column. "
            "Remove it and try again."
        )

    return columns


def count_rows(path: Path) -> int:
    """Number of data rows (header excluded), reading a single column."""

    total = 0

    try:
        for chunk in pd.read_csv(
            path,
            usecols=[0],
            chunksize=Config.BATCH.CHUNK_SIZE,
            **_CSV_READ_OPTIONS,
        ):
            total += len(chunk)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise BatchInputError(f"Invalid CSV file: {exc}") from None

    return total


def validate_batch_input(
    chunk: pd.DataFrame,
    start_row: int = 0,
) -> pd.DataFrame:
    """
    Validate raw batch rows and return the typed frame used for features.

    This is the vectorised counterpart of FlightRequest's type checks in
    the online API: same 8 fields, same types (HHMM times are integers,
    elapsed time and distance are floats). Feature engineering itself is
    NOT done here; that is prepare_model_input().

    `start_row` is the number of data rows preceding this chunk, so error
    messages can point at the right rows of the file.
    """

    if any(c not in set(chunk.columns) for c in REQUIRED_COLUMNS):
        raise _missing_columns_error(chunk.columns)

    if chunk.empty:
        raise BatchInputError("Input file contains no data rows.")

    typed = chunk[list(REQUIRED_COLUMNS)].copy()

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(typed[column], errors="coerce")
        invalid = values.isna() | np.isinf(values)

        if column in INTEGER_COLUMNS:
            invalid |= ((values % 1) != 0) | (values.abs() > 1e9)

        if invalid.any():
            raise BatchInputError(
                f"Column {column} has {int(invalid.sum())} invalid "
                f"numeric value(s) ({_describe_rows(invalid, start_row)})."
            )

        typed[column] = (
            values.astype("int64")
            if column in INTEGER_COLUMNS
            else values.astype("float64")
        )

    for column in TEXT_COLUMNS:
        blank = (
            typed[column].isna()
            | typed[column].astype(str).str.strip().eq("")
        )

        if blank.any():
            raise BatchInputError(
                f"Column {column} has {int(blank.sum())} empty "
                f"value(s) ({_describe_rows(blank, start_row)})."
            )

    bad_dates = pd.to_datetime(
        typed["FL_DATE"], errors="coerce"
    ).isna()

    if bad_dates.any():
        raise BatchInputError(
            f"Column FL_DATE has {int(bad_dates.sum())} invalid "
            f"date(s) ({_describe_rows(bad_dates, start_row)})."
        )

    return typed


# ============================================================
# MODEL
# ============================================================

def load_model() -> ModelState:
    """The same MLflow pipeline the API serves (Config.MLFLOW.MODEL_URI)."""

    try:
        return ensure_model_loaded()
    except Exception as exc:
        logger.exception("Failed to load the model.")
        raise BatchModelError(
            "The prediction model could not be loaded."
        ) from exc


def get_model_version(state: ModelState) -> str:
    """Model version, falling back to the id recorded in the MLmodel file."""

    if state.version and state.version != "unknown":
        return state.version

    mlmodel = Path(state.model_uri) / "MLmodel"

    try:
        import yaml

        info = yaml.safe_load(mlmodel.read_text(encoding="utf-8"))
        return str(info.get("model_id") or info.get("run_id") or "unknown")
    except Exception:
        return "unknown"


# ============================================================
# FEATURES + PREDICTION
# ============================================================

def build_batch_features(
    chunk: pd.DataFrame,
    expected_columns: list[str],
    start_row: int = 0,
) -> pd.DataFrame:
    """Validated raw rows -> model input, via the shared online code path."""

    typed = validate_batch_input(chunk, start_row)

    try:
        return prepare_model_input(typed, expected_columns)
    except NullModelInputError as exc:
        raise BatchInputError(
            "Model input contains missing values in: "
            + ", ".join(exc.columns)
        ) from exc
    except MissingModelFeaturesError as exc:
        logger.error("Missing model features: %s", exc.missing)
        raise BatchPredictionError(
            "The model input could not be built."
        ) from exc
    except ValueError as exc:
        # Raised by build_features() for values it rejects (bad HHMM
        # time, negative distance): the same rule the online API applies.
        end_row = start_row + len(chunk) + 1
        raise BatchInputError(
            f"{exc} (file rows {start_row + 2}-{end_row})"
        ) from exc


def predict_batch(model, X: pd.DataFrame) -> np.ndarray:
    """Predict a whole frame at once; rounded like the online response."""

    try:
        predictions = predict_frame(model, X)
    except Exception as exc:
        logger.exception("Model prediction failed.")
        raise BatchPredictionError("Prediction failed.") from exc

    if len(predictions) != len(X):
        raise BatchPredictionError("Prediction failed.")

    return np.round(predictions, PREDICTION_PRECISION)


# ============================================================
# OUTPUT
# ============================================================

class PredictionWriter:
    """
    Appends predicted chunks to a temp file and moves it over the output
    path only on commit(), so a failed batch never leaves a partial CSV.
    """

    def __init__(self, output_path: Path, prediction_column: str) -> None:
        self.output_path = Path(output_path)
        self.prediction_column = prediction_column
        self._tmp: Optional[Path] = None
        self._has_header = False
        self._committed = False

    def __enter__(self) -> "PredictionWriter":
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.exception("Cannot create the output directory.")
            raise BatchOutputError(
                "The output location is not writable."
            ) from exc

        self._tmp = self.output_path.parent / (
            f".{self.output_path.name}.{uuid.uuid4().hex}.tmp"
        )
        return self

    def write(self, chunk: pd.DataFrame, predictions: np.ndarray) -> None:
        out = chunk.copy()
        out[self.prediction_column] = predictions

        try:
            out.to_csv(
                self._tmp,
                mode="a",
                header=not self._has_header,
                index=False,
                lineterminator="\n",
            )
        except OSError as exc:
            logger.exception("Failed to write predictions.")
            raise BatchOutputError(
                "Predictions could not be saved."
            ) from exc

        self._has_header = True

    def commit(self) -> None:
        try:
            os.replace(self._tmp, self.output_path)
        except OSError as exc:
            logger.exception("Failed to finalise the output file.")
            raise BatchOutputError(
                "Predictions could not be saved."
            ) from exc

        self._committed = True

    def __exit__(self, *exc_info) -> None:
        if not self._committed and self._tmp is not None:
            self._tmp.unlink(missing_ok=True)


def save_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    output_path: Path,
    prediction_column: Optional[str] = None,
) -> None:
    """Write `df` plus a prediction column to `output_path` in one go."""

    with PredictionWriter(
        output_path,
        prediction_column or Config.BATCH.PREDICTION_COLUMN,
    ) as writer:
        writer.write(df, predictions)
        writer.commit()


# ============================================================
# FILE -> FILE
# ============================================================

def iter_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    try:
        yield from pd.read_csv(
            path, chunksize=chunk_size, **_CSV_READ_OPTIONS
        )
    except UnicodeDecodeError:
        raise BatchInputError(
            "Input is not a valid UTF-8 CSV file."
        ) from None
    except (pd.errors.ParserError, ValueError) as exc:
        raise BatchInputError(f"Invalid CSV file: {exc}") from None


def predict_file(
    input_path: Path,
    output_path: Path,
    state: ModelState,
    chunk_size: Optional[int] = None,
    on_chunk: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Predict every row of `input_path` and write `output_path`.

    `on_chunk(rows_in_chunk, rows_done)` is called after each chunk.
    Returns the number of rows predicted.
    """

    chunk_size = chunk_size or Config.BATCH.CHUNK_SIZE
    done = 0

    with PredictionWriter(
        output_path, Config.BATCH.PREDICTION_COLUMN
    ) as writer:

        for chunk in iter_chunks(input_path, chunk_size):
            X = build_batch_features(
                chunk, state.expected_columns, start_row=done
            )
            predictions = predict_batch(state.model, X)
            writer.write(chunk, predictions)

            done += len(chunk)

            logger.info(
                "Processed %s rows (features built, predicted)",
                f"{done:,}",
            )

            if on_chunk is not None:
                on_chunk(len(chunk), done)

        if done == 0:
            raise BatchInputError("Input file contains no data rows.")

        writer.commit()

    return done


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.batch.predict",
        description=(
            "Run batch inference on a CSV of flights using the "
            "production model (MLFLOW_MODEL_URI)."
        ),
    )
    parser.add_argument(
        "--input", required=True, help="Input CSV with flight records."
    )
    parser.add_argument(
        "--output",
        help=(
            "Output CSV path (default: "
            "data/batch/output/<batch_id>_predictions.csv)."
        ),
    )
    parser.add_argument(
        "--batch-id",
        help="Batch identifier (default: generated, e.g. batch-20260922-001).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Exit codes: 0 completed, 1 batch failed, 2 invalid arguments."""

    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # The model loader is chatty at INFO (it lists every model column).
    logging.getLogger("src.api.model_loader").setLevel(logging.WARNING)

    # Imported here so `python -m src.batch.predict` does not import this
    # module twice through the runner.
    from src.batch.runner import create_batch, run_batch
    from src.batch.schemas import BatchStatus
    from src.batch.store import BatchStore

    store = BatchStore.from_config()

    try:
        meta = create_batch(
            store,
            input_path=Path(args.input),
            output_path=Path(args.output) if args.output else None,
            batch_id=args.batch_id,
            source="cli",
        )
    except BatchError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 2

    meta = run_batch(store, meta.batch_id)

    if meta.status == BatchStatus.COMPLETED:
        return 0

    print(f"Batch {meta.batch_id} failed: {meta.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
