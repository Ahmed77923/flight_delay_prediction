"""
Prometheus metrics for batch jobs.

They use prometheus_client's default registry, which the API's
Instrumentator already exposes at /metrics, so no extra scrape config
is needed. Jobs run by the standalone CLI live in a short-lived process
and are therefore not scraped.
"""

from prometheus_client import Counter, Histogram


BATCH_JOBS_TOTAL = Counter(
    "batch_jobs_total",
    "Batch jobs started.",
)

BATCH_JOBS_SUCCESS_TOTAL = Counter(
    "batch_jobs_success_total",
    "Batch jobs that completed successfully.",
)

BATCH_JOBS_FAILED_TOTAL = Counter(
    "batch_jobs_failed_total",
    "Batch jobs that failed.",
)

BATCH_ROWS_PROCESSED_TOTAL = Counter(
    "batch_rows_processed_total",
    "Rows predicted by batch jobs.",
)

BATCH_DURATION_SECONDS = Histogram(
    "batch_duration_seconds",
    "Batch job duration in seconds.",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)
