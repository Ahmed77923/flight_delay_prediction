"""
Runs batch jobs: status tracking, metrics and background execution.

Used by both the CLI (run_batch, in the foreground) and the API
(submit_batch, on a small thread pool so requests return immediately).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from config.config import Config
from src.batch import metrics
from src.batch.errors import BatchError, BatchInputError
from src.batch.predict import (
    count_rows,
    get_model_version,
    inspect_input_file,
    load_model,
    predict_file,
)
from src.batch.schemas import BatchMetadata, BatchStatus
from src.batch.store import BatchStore, utc_now


logger = logging.getLogger(__name__)


# ============================================================
# CREATE
# ============================================================

def create_batch(
    store: BatchStore,
    *,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    batch_id: Optional[str] = None,
    source: str = "cli",
    input_filename: Optional[str] = None,
) -> BatchMetadata:
    """
    Register a pending batch. Without explicit paths, files live at the
    store's default locations for the batch id.
    """

    if input_path is not None:
        input_path = Path(input_path).expanduser().resolve()

    if output_path is not None:
        output_path = Path(output_path).expanduser().resolve()

    if input_path is not None and input_path == output_path:
        raise BatchInputError(
            "The output path must be different from the input path."
        )

    return store.create(
        batch_id=batch_id,
        input_file=input_path,
        output_file=output_path,
        source=source,
        input_filename=(
            input_filename
            or (input_path.name if input_path is not None else None)
        ),
    )


# ============================================================
# RUN
# ============================================================

def run_batch(store: BatchStore, batch_id: str) -> BatchMetadata:
    """
    Execute one batch and return its final metadata. Never raises for a
    failed batch: the failure is recorded in the metadata (status "failed"
    with a user-safe `error`) and the technical detail is logged.
    """

    meta = store.get(batch_id)

    if meta is None:
        raise FileNotFoundError(f"Unknown batch: {batch_id}")

    started = time.perf_counter()
    metrics.BATCH_JOBS_TOTAL.inc()

    store.update(
        batch_id,
        status=BatchStatus.RUNNING,
        started_at=utc_now(),
    )

    logger.info("Batch started: %s", batch_id)
    logger.info("Input: %s", meta.input_file)

    try:
        input_path = Path(meta.input_file)
        output_path = Path(meta.output_file)

        inspect_input_file(input_path)

        rows = count_rows(input_path)

        if rows == 0:
            raise BatchInputError("Input file contains no data rows.")

        store.update(batch_id, rows=rows)
        logger.info("Rows: %d", rows)

        state = load_model()
        store.update(
            batch_id,
            model_uri=state.model_uri,
            model_version=get_model_version(state),
        )
        logger.info("Model loaded")

        def on_chunk(chunk_rows: int, rows_done: int) -> None:
            metrics.BATCH_ROWS_PROCESSED_TOTAL.inc(chunk_rows)
            store.update(batch_id, rows_processed=rows_done)

        processed = predict_file(
            input_path,
            output_path,
            state,
            on_chunk=on_chunk,
        )

        logger.info("Prediction completed: %d rows", processed)
        logger.info("Predictions saved: %s", output_path)

        duration = round(time.perf_counter() - started, 2)

        meta = store.update(
            batch_id,
            status=BatchStatus.COMPLETED,
            rows=processed,
            rows_processed=processed,
            completed_at=utc_now(),
            duration_seconds=duration,
            error=None,
        )

        metrics.BATCH_JOBS_SUCCESS_TOTAL.inc()
        metrics.BATCH_DURATION_SECONDS.observe(duration)

        logger.info("Duration: %.1fs", duration)

        return meta

    except BatchError as exc:
        logger.error("Batch %s failed: %s", batch_id, exc.message)
        return _mark_failed(store, batch_id, exc.message, started)

    except Exception:
        logger.exception("Batch %s failed unexpectedly.", batch_id)
        return _mark_failed(
            store,
            batch_id,
            "Unexpected error while processing the batch.",
            started,
        )


def _mark_failed(
    store: BatchStore,
    batch_id: str,
    message: str,
    started: float,
) -> BatchMetadata:

    duration = round(time.perf_counter() - started, 2)

    metrics.BATCH_JOBS_FAILED_TOTAL.inc()
    metrics.BATCH_DURATION_SECONDS.observe(duration)

    return store.update(
        batch_id,
        status=BatchStatus.FAILED,
        error=message,
        completed_at=utc_now(),
        duration_seconds=duration,
    )


# ============================================================
# BACKGROUND EXECUTION (API)
# ============================================================

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor

    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max(1, Config.BATCH.MAX_WORKERS),
                thread_name_prefix="batch",
            )

        return _executor


def submit_batch(store: BatchStore, batch_id: str) -> Future:
    """Queue a batch on the background pool and return immediately."""

    return _get_executor().submit(run_batch, store, batch_id)


def shutdown_executor() -> None:
    """Stop accepting jobs; queued jobs are dropped (recovered on restart)."""

    global _executor

    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None


def recover_interrupted_jobs(store: Optional[BatchStore] = None) -> int:
    """Fail API batches left pending/running by a previous process."""

    store = store or BatchStore.from_config()

    if not store.metadata_dir.is_dir():
        return 0

    count = store.fail_interrupted(source="api")

    if count:
        logger.warning(
            "Marked %d interrupted batch job(s) as failed.", count
        )

    return count
