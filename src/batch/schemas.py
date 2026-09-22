from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.features.build_feature import REQUIRED_RAW_COLUMNS


# Required input columns, in a stable order for messages and UI.
REQUIRED_COLUMNS: tuple[str, ...] = tuple(sorted(REQUIRED_RAW_COLUMNS))

NUMERIC_COLUMNS: tuple[str, ...] = (
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
)

# HHMM times must be whole numbers (FlightRequest declares them as int).
INTEGER_COLUMNS: tuple[str, ...] = (
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
)

TEXT_COLUMNS: tuple[str, ...] = (
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
)


class BatchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset(
    {BatchStatus.COMPLETED, BatchStatus.FAILED}
)


class BatchMetadata(BaseModel):
    """Persisted record of one batch job (data/batch/metadata/<id>.json)."""

    batch_id: str
    status: BatchStatus = BatchStatus.PENDING
    source: str = "cli"  # "cli" | "api"

    input_file: str
    input_filename: Optional[str] = None
    output_file: str

    # Total rows in the input (known once the file has been inspected) and
    # rows already predicted. They are equal once the batch has completed.
    rows: Optional[int] = None
    rows_processed: int = 0
    prediction_column: str

    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    model_uri: Optional[str] = None
    model_version: Optional[str] = None

    error: Optional[str] = None


# ------------------------------------------------------------
# API responses
# ------------------------------------------------------------

class BatchSubmitResponse(BaseModel):
    batch_id: str
    status: BatchStatus


class BatchStatusResponse(BaseModel):
    """API view of a batch: never exposes server filesystem paths."""

    batch_id: str
    status: BatchStatus
    input_filename: Optional[str] = None
    output_file: Optional[str] = None
    rows: Optional[int] = None
    rows_processed: int = 0
    prediction_column: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    model_version: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_metadata(cls, meta: BatchMetadata) -> "BatchStatusResponse":
        completed = meta.status == BatchStatus.COMPLETED

        return cls(
            batch_id=meta.batch_id,
            status=meta.status,
            input_filename=meta.input_filename,
            output_file=(
                Path(meta.output_file).name if completed else None
            ),
            rows=meta.rows,
            rows_processed=meta.rows_processed,
            prediction_column=meta.prediction_column,
            started_at=meta.started_at,
            completed_at=meta.completed_at,
            duration_seconds=meta.duration_seconds,
            model_version=meta.model_version,
            error=meta.error,
        )
