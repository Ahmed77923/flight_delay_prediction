"""
Batch Serving endpoints.

    POST /batch/predict               upload a CSV, start a background batch
    GET  /batch/status/{batch_id}     poll a batch
    GET  /batch/results/{batch_id}    download the predictions CSV

Clients only ever see server-generated batch ids and file names, never
filesystem paths.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from config.config import Config
from src.batch.errors import BatchError
from src.batch.predict import inspect_input_file
from src.batch.runner import create_batch, submit_batch
from src.batch.schemas import (
    BatchStatus,
    BatchStatusResponse,
    BatchSubmitResponse,
)
from src.batch.store import BatchStore, is_valid_batch_id


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])


# Browsers and OSes label CSV files inconsistently.
_CSV_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "text/x-csv",
    "application/x-csv",
    "application/vnd.ms-excel",
    "text/plain",
    "application/octet-stream",
}

_COPY_BLOCK_BYTES = 1024 * 1024


def get_store() -> BatchStore:
    return BatchStore.from_config()


def _failed(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "failed", "error": message},
    )


def _display_name(filename: str) -> str:
    """Client file name reduced to a safe label; never used as a path."""

    name = Path(filename.replace("\\", "/")).name
    return re.sub(r"[^\w.\- ()]", "_", name)[:255]


class _UploadTooLarge(Exception):
    pass


def _save_upload(upload: UploadFile, store: BatchStore) -> Path:
    """Stream the upload to a temp file inside the input dir."""

    max_bytes = Config.BATCH.MAX_UPLOAD_MB * 1024 * 1024
    tmp = store.input_dir / f".upload-{uuid.uuid4().hex}.csv"
    size = 0

    try:
        with open(tmp, "wb") as out:
            while block := upload.file.read(_COPY_BLOCK_BYTES):
                size += len(block)

                if size > max_bytes:
                    raise _UploadTooLarge()

                out.write(block)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    return tmp


# ============================================================
# POST /batch/predict
# ============================================================

@router.post(
    "/predict",
    response_model=BatchSubmitResponse,
    status_code=202,
)
def submit_batch_prediction(
    file: UploadFile = File(...),
    store: BatchStore = Depends(get_store),
):
    filename = file.filename or ""

    if not filename.lower().endswith(".csv"):
        return _failed(400, "Only .csv files are accepted.")

    if (file.content_type or "").split(";")[0].strip().lower() not in (
        _CSV_CONTENT_TYPES | {""}
    ):
        return _failed(400, "Only .csv files are accepted.")

    try:
        store.ensure_dirs()
        tmp = _save_upload(file, store)
    except _UploadTooLarge:
        return _failed(
            413,
            f"File is too large (limit {Config.BATCH.MAX_UPLOAD_MB} MB).",
        )
    except OSError:
        logger.exception("Could not store the uploaded file.")
        return _failed(500, "The uploaded file could not be stored.")

    # Header-level validation now, so the client gets an immediate,
    # specific error instead of a job that fails a moment later.
    try:
        inspect_input_file(tmp)
    except BatchError as exc:
        tmp.unlink(missing_ok=True)
        return _failed(400, exc.message)

    try:
        meta = create_batch(
            store,
            source="api",
            input_filename=_display_name(filename),
        )
        tmp.replace(meta.input_file)
    except BatchError as exc:
        tmp.unlink(missing_ok=True)
        return _failed(400, exc.message)
    except OSError:
        tmp.unlink(missing_ok=True)
        logger.exception("Could not register the batch.")
        return _failed(500, "The batch could not be started.")

    submit_batch(store, meta.batch_id)

    logger.info(
        "Batch %s accepted (%s)", meta.batch_id, meta.input_filename
    )

    # The pool may already have picked the job up.
    current = store.get(meta.batch_id) or meta

    return BatchSubmitResponse(
        batch_id=meta.batch_id,
        status=current.status,
    )


# ============================================================
# GET /batch/status/{batch_id}
# ============================================================

@router.get(
    "/status/{batch_id}",
    response_model=BatchStatusResponse,
)
def batch_status(
    batch_id: str,
    store: BatchStore = Depends(get_store),
):
    meta = store.get(batch_id)

    if meta is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Batch not found."},
        )

    return BatchStatusResponse.from_metadata(meta)


# ============================================================
# GET /batch/results/{batch_id}
# ============================================================

@router.get("/results/{batch_id}")
def batch_results(
    batch_id: str,
    store: BatchStore = Depends(get_store),
):
    meta = store.get(batch_id) if is_valid_batch_id(batch_id) else None

    if meta is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Batch not found."},
        )

    if meta.status != BatchStatus.COMPLETED:
        return JSONResponse(
            status_code=409,
            content={
                "status": meta.status.value,
                "error": (
                    "Results are not available: batch is "
                    f"{meta.status.value}."
                ),
            },
        )

    # Only ever serve files inside the output directory, whatever the
    # metadata says (CLI batches may have written elsewhere).
    output = Path(meta.output_file).resolve()

    if (
        not output.is_relative_to(store.output_dir.resolve())
        or not output.is_file()
    ):
        return JSONResponse(
            status_code=404,
            content={"error": "Results file not found."},
        )

    return FileResponse(
        output,
        media_type="text/csv",
        filename=f"{batch_id}_predictions.csv",
    )
