"""
Batch errors.

Every BatchError carries a `message` that is safe to show to end users
(no stack traces, no server paths). Technical detail is logged server-side.
"""

from __future__ import annotations


class BatchError(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BatchInputError(BatchError):
    """The input file or its contents are invalid."""


class BatchModelError(BatchError):
    """The model could not be loaded."""


class BatchPredictionError(BatchError):
    """Feature building or prediction failed for a reason not caused by input."""


class BatchOutputError(BatchError):
    """The predictions could not be written."""
