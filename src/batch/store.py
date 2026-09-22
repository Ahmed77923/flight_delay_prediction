"""
Lightweight JSON metadata store for batch jobs.

    data/batch/
    ├── input/      uploaded / source CSVs        <batch_id>.csv
    ├── output/     generated predictions         <batch_id>_predictions.csv
    └── metadata/   one JSON record per batch     <batch_id>.json

Batch ids are validated against a strict pattern so they can never be used
for path traversal, and every record is written atomically so a reader
never sees a half-written file.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.config import Config
from src.batch.errors import BatchInputError
from src.batch.schemas import BatchMetadata, BatchStatus


logger = logging.getLogger(__name__)


# Serialises read-modify-write updates of metadata files within a process.
_UPDATE_LOCK = threading.Lock()

BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_valid_batch_id(batch_id: str) -> bool:
    return bool(BATCH_ID_PATTERN.fullmatch(batch_id))


class BatchStore:

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        metadata_dir: Path,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)

    @classmethod
    def from_config(cls) -> "BatchStore":
        return cls(
            Config.BATCH.INPUT_DIR,
            Config.BATCH.OUTPUT_DIR,
            Config.BATCH.METADATA_DIR,
        )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    def ensure_dirs(self) -> None:
        for directory in (
            self.input_dir,
            self.output_dir,
            self.metadata_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def metadata_path(self, batch_id: str) -> Path:
        return self.metadata_dir / f"{batch_id}.json"

    def input_path(self, batch_id: str) -> Path:
        return self.input_dir / f"{batch_id}.csv"

    def output_path(self, batch_id: str) -> Path:
        return self.output_dir / f"{batch_id}_predictions.csv"

    # --------------------------------------------------------
    # Create
    # --------------------------------------------------------

    def create(
        self,
        *,
        batch_id: Optional[str] = None,
        input_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
        source: str = "cli",
        input_filename: Optional[str] = None,
    ) -> BatchMetadata:
        """
        Register a new pending batch. Ids are unique: the metadata file is
        created exclusively, so an existing batch is never overwritten.
        """

        self.ensure_dirs()

        if batch_id is not None:
            if not is_valid_batch_id(batch_id):
                raise BatchInputError(
                    "Invalid batch id. Use 1-64 letters, digits, "
                    "'.', '_' or '-', starting with a letter or digit."
                )

            return self._create_record(
                batch_id, input_file, output_file, source, input_filename
            )

        prefix = f"batch-{datetime.now(timezone.utc):%Y%m%d}-"
        sequence = self._next_sequence(prefix)

        while True:
            try:
                return self._create_record(
                    f"{prefix}{sequence:03d}",
                    input_file,
                    output_file,
                    source,
                    input_filename,
                )
            except BatchInputError:
                # Lost a race with another process; try the next number.
                sequence += 1

    def _next_sequence(self, prefix: str) -> int:
        highest = 0

        for path in self.metadata_dir.glob(f"{prefix}*.json"):
            suffix = path.stem[len(prefix):]
            if suffix.isdigit():
                highest = max(highest, int(suffix))

        return highest + 1

    def _create_record(
        self,
        batch_id: str,
        input_file: Optional[Path],
        output_file: Optional[Path],
        source: str,
        input_filename: Optional[str],
    ) -> BatchMetadata:

        meta = BatchMetadata(
            batch_id=batch_id,
            source=source,
            input_file=str(input_file or self.input_path(batch_id)),
            input_filename=input_filename,
            output_file=str(output_file or self.output_path(batch_id)),
            prediction_column=Config.BATCH.PREDICTION_COLUMN,
            created_at=utc_now(),
        )

        # Write to a temp file, then hard-link it into place: link() fails
        # if the target exists, which gives an atomic "create if absent".
        fd, tmp_name = tempfile.mkstemp(
            dir=self.metadata_dir, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(meta.model_dump_json(indent=2))

            try:
                os.link(tmp_name, self.metadata_path(batch_id))
            except FileExistsError:
                raise BatchInputError(
                    f"Batch id '{batch_id}' already exists. "
                    "Choose a different --batch-id."
                ) from None
        finally:
            os.unlink(tmp_name)

        return meta

    # --------------------------------------------------------
    # Read / update
    # --------------------------------------------------------

    def get(self, batch_id: str) -> Optional[BatchMetadata]:
        if not is_valid_batch_id(batch_id):
            return None

        try:
            raw = self.metadata_path(batch_id).read_text(encoding="utf-8")
            return BatchMetadata.model_validate_json(raw)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logger.exception(
                "Unreadable batch metadata for %s", batch_id
            )
            return None

    def update(self, batch_id: str, **fields: Any) -> BatchMetadata:
        with _UPDATE_LOCK:
            meta = self.get(batch_id)

            if meta is None:
                raise FileNotFoundError(f"Unknown batch: {batch_id}")

            updated = meta.model_copy(update=fields)
            self._write(updated)

            return updated

    def _write(self, meta: BatchMetadata) -> None:
        path = self.metadata_path(meta.batch_id)
        tmp = path.with_suffix(".json.tmp")

        tmp.write_text(
            meta.model_dump_json(indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)

    def list_all(self) -> list[BatchMetadata]:
        if not self.metadata_dir.is_dir():
            return []

        records = []
        for path in sorted(self.metadata_dir.glob("*.json")):
            meta = self.get(path.stem)
            if meta is not None:
                records.append(meta)

        return records

    def fail_interrupted(self, source: str = "api") -> int:
        """
        Mark pending/running batches of `source` as failed. Called at API
        startup: such jobs died with the previous process.
        """

        count = 0

        for meta in self.list_all():
            if meta.source != source:
                continue

            if meta.status in (BatchStatus.PENDING, BatchStatus.RUNNING):
                self.update(
                    meta.batch_id,
                    status=BatchStatus.FAILED,
                    error=(
                        "The service restarted before this batch "
                        "finished. Please submit it again."
                    ),
                    completed_at=utc_now(),
                )
                count += 1

        return count
