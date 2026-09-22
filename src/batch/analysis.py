"""
Analysis of a completed batch's prediction CSV.

Pure pandas / NumPy, no Streamlit: the dashboard calls these functions and
tests exercise them on small frames. Nothing here runs the model; it only
reads the predictions file the batch already produced.

Built for millions of rows: the CSV is read once with only the columns the
analysis needs (text columns as categoricals), every chart input is
aggregated first (histogram bins, groups, days), and route labels are built
after grouping rather than per row. The charts never receive raw rows.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Union

import numpy as np
import pandas as pd

from config.config import Config


PREDICTION_COLUMN = Config.BATCH.PREDICTION_COLUMN

CARRIER_COLUMN = "OP_UNIQUE_CARRIER"
ORIGIN_COLUMN = "ORIGIN"
DEST_COLUMN = "DEST"
DATE_COLUMN = "FL_DATE"

# Delay above this many minutes counts as "delayed" in the KPI card and is
# drawn as the reference line on the distribution chart.
DELAY_THRESHOLD_MINUTES = 15.0

# Delay categories: (label, inclusive upper bound in minutes). Each category
# covers (previous bound, its own bound], so:
#   On Time <= 0 | Minor (0, 15] | Moderate (15, 30] | Severe > 30
# This is the single place to change the thresholds.
DELAY_CATEGORIES: tuple[tuple[str, float], ...] = (
    ("On Time", 0.0),
    ("Minor Delay", DELAY_THRESHOLD_MINUTES),
    ("Moderate Delay", 30.0),
    ("Severe Delay", math.inf),
)

TOP_N_CHOICES = (5, 10, 15, 20)
DEFAULT_TOP_N = 10

HISTOGRAM_BINS = 50
PREVIEW_ROWS = 1_000

# Optional original columns kept for the data table.
_TABLE_COLUMNS = (
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
)
# Text columns loaded as categoricals: a few hundred distinct values over
# millions of rows, so this is both faster and far smaller than strings.
_CATEGORICAL_COLUMNS = (
    CARRIER_COLUMN,
    ORIGIN_COLUMN,
    DEST_COLUMN,
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
)
DEFAULT_TABLE_COLUMNS = (
    DATE_COLUMN,
    CARRIER_COLUMN,
    ORIGIN_COLUMN,
    DEST_COLUMN,
    PREDICTION_COLUMN,
)


# ============================================================
# ERRORS
# ============================================================

class AnalysisError(Exception):
    """The predictions cannot be analysed at all; the message is user-safe."""


class AnalysisUnavailableError(AnalysisError):
    """One analysis section cannot be produced (e.g. a column is absent)."""


def _require(df: pd.DataFrame, section: str, *columns: str) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        verb = "is" if len(missing) == 1 else "are"

        raise AnalysisUnavailableError(
            f"{section} analysis is unavailable because "
            f"{' and '.join(missing)} {verb} not present."
        )


# ============================================================
# LOADING
# ============================================================

@dataclass(frozen=True)
class FilterOptions:
    """Values the filter widgets offer."""

    carriers: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    destinations: tuple[str, ...] = ()
    date_min: Optional[date] = None
    date_max: Optional[date] = None


@dataclass
class LoadedPredictions:
    frame: pd.DataFrame
    options: FilterOptions
    # Non-fatal remarks to show the user (ignored rows, invalid dates).
    notes: list[str] = field(default_factory=list)
    file_rows: int = 0


PredictionSource = Union[bytes, str, "os.PathLike[str]"]


def _open(source: PredictionSource):
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(bytes(source))

    return source


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Parse a date column by parsing only its *distinct* values (a year of
    flights has ~365) and mapping them back with the category codes. That
    is orders of magnitude faster than parsing millions of strings.
    Unparseable / missing values become NaT.
    """

    if not isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("category")

    distinct = pd.to_datetime(
        pd.Series(series.cat.categories.astype(str)),
        errors="coerce",
        format="mixed",
    )
    lookup = distinct.to_numpy(dtype="datetime64[ns]")

    codes = series.cat.codes.to_numpy()
    parsed = np.full(len(codes), np.datetime64("NaT"), dtype="datetime64[ns]")
    known = codes >= 0
    parsed[known] = lookup[codes[known]]

    return pd.Series(parsed, index=series.index).dt.normalize()


def _distinct(series: pd.Series) -> tuple[str, ...]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        values = series.cat.categories
    else:
        values = series.dropna().unique()

    return tuple(sorted(str(v) for v in values))


def filter_options(df: pd.DataFrame) -> FilterOptions:
    dates = df[DATE_COLUMN].dropna() if DATE_COLUMN in df.columns else None
    has_dates = dates is not None and not dates.empty

    return FilterOptions(
        carriers=_distinct(df[CARRIER_COLUMN]) if CARRIER_COLUMN in df else (),
        origins=_distinct(df[ORIGIN_COLUMN]) if ORIGIN_COLUMN in df else (),
        destinations=_distinct(df[DEST_COLUMN]) if DEST_COLUMN in df else (),
        date_min=dates.min().date() if has_dates else None,
        date_max=dates.max().date() if has_dates else None,
    )


def read_prediction_csv(source: PredictionSource) -> LoadedPredictions:
    """
    Read the batch output once, keeping only the columns the analysis and
    the data table use. Raises AnalysisError (with a user-safe message) if
    the file is missing, empty, unreadable, lacks the prediction column, or
    holds no valid predictions.
    """

    try:
        header = pd.read_csv(
            _open(source), nrows=0, encoding="utf-8-sig"
        ).columns
    except FileNotFoundError:
        raise AnalysisError("The predictions file was not found.") from None
    except pd.errors.EmptyDataError:
        raise AnalysisError("The predictions file is empty.") from None
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError):
        raise AnalysisError(
            "The predictions file could not be read as CSV."
        ) from None

    if PREDICTION_COLUMN not in header:
        raise AnalysisError(
            f"The predictions file has no '{PREDICTION_COLUMN}' column."
        )

    wanted = [
        column
        for column in (
            DATE_COLUMN,
            CARRIER_COLUMN,
            ORIGIN_COLUMN,
            DEST_COLUMN,
            *_TABLE_COLUMNS,
            PREDICTION_COLUMN,
        )
        if column in header
    ]

    try:
        frame = pd.read_csv(
            _open(source),
            usecols=wanted,
            dtype={
                column: "category"
                for column in (*_CATEGORICAL_COLUMNS, DATE_COLUMN)
                if column in wanted
            },
            encoding="utf-8-sig",
        )
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError):
        raise AnalysisError(
            "The predictions file could not be read as CSV."
        ) from None

    file_rows = len(frame)

    if file_rows == 0:
        raise AnalysisError("The predictions file contains no rows.")

    notes: list[str] = []

    delays = pd.to_numeric(frame[PREDICTION_COLUMN], errors="coerce")
    valid = np.isfinite(delays.to_numpy(dtype=float))

    if not valid.all():
        invalid_rows = int((~valid).sum())
        notes.append(
            f"{invalid_rows:,} row(s) with an invalid "
            f"{PREDICTION_COLUMN} value were ignored."
        )
        frame = frame[valid]
        delays = delays[valid]

    if frame.empty:
        raise AnalysisError(
            "The predictions file contains no valid predictions."
        )

    frame = frame.reset_index(drop=True)
    frame[PREDICTION_COLUMN] = delays.to_numpy(dtype=float)

    if DATE_COLUMN in frame.columns:
        frame[DATE_COLUMN] = _parse_dates(frame[DATE_COLUMN])
        bad_dates = int(frame[DATE_COLUMN].isna().sum())

        if bad_dates:
            notes.append(
                f"{bad_dates:,} row(s) have a missing or invalid "
                f"{DATE_COLUMN}; they are left out of the time analysis "
                "and the date filter."
            )

    return LoadedPredictions(
        frame=frame,
        options=filter_options(frame),
        notes=notes,
        file_rows=file_rows,
    )


# ============================================================
# FILTERING
# ============================================================

@dataclass(frozen=True)
class AnalysisFilters:
    """Empty selection = "All"."""

    carriers: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    destinations: tuple[str, ...] = ()
    date_range: Optional[tuple[date, date]] = None

    @property
    def is_active(self) -> bool:
        return bool(
            self.carriers
            or self.origins
            or self.destinations
            or self.date_range
        )


def apply_filters(df: pd.DataFrame, filters: AnalysisFilters) -> pd.DataFrame:
    """
    Rows matching every active filter. Returns `df` itself (no copy) when
    nothing is selected. A filter on a column the data lacks is ignored.
    """

    mask: Optional[np.ndarray] = None

    def add(condition: np.ndarray) -> None:
        nonlocal mask
        mask = condition if mask is None else mask & condition

    for column, selected in (
        (CARRIER_COLUMN, filters.carriers),
        (ORIGIN_COLUMN, filters.origins),
        (DEST_COLUMN, filters.destinations),
    ):
        if selected and column in df.columns:
            add(df[column].isin(selected).to_numpy())

    if filters.date_range and DATE_COLUMN in df.columns:
        start, end = filters.date_range
        dates = df[DATE_COLUMN]
        add(
            ((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
            .to_numpy()
        )

    return df if mask is None else df[mask]


# ============================================================
# KPIs, DISTRIBUTION, CATEGORIES
# ============================================================

@dataclass(frozen=True)
class BatchKPIs:
    total_flights: int
    average: Optional[float]
    median: Optional[float]
    maximum: Optional[float]
    minimum: Optional[float]
    # Share of flights (0-100) predicted to be delayed by more than
    # DELAY_THRESHOLD_MINUTES.
    percent_over_threshold: Optional[float]


def calculate_batch_kpis(
    df: pd.DataFrame,
    threshold: float = DELAY_THRESHOLD_MINUTES,
) -> BatchKPIs:
    values = df[PREDICTION_COLUMN].to_numpy(dtype=float)

    if values.size == 0:
        return BatchKPIs(0, None, None, None, None, None)

    return BatchKPIs(
        total_flights=int(values.size),
        average=float(values.mean()),
        median=float(np.median(values)),
        maximum=float(values.max()),
        minimum=float(values.min()),
        percent_over_threshold=float((values > threshold).mean() * 100),
    )


def delay_histogram(
    df: pd.DataFrame,
    bins: int = HISTOGRAM_BINS,
) -> pd.DataFrame:
    """Bin edges and flight counts: `bins` rows however large the batch is."""

    values = df[PREDICTION_COLUMN].to_numpy(dtype=float)

    if values.size == 0:
        return pd.DataFrame(columns=["bin_start", "bin_end", "flights"])

    counts, edges = np.histogram(values, bins=bins)

    return pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "flights": counts,
        }
    )


_UPPER_BOUNDS = np.array([bound for _, bound in DELAY_CATEGORIES])
_CATEGORY_LABELS = [label for label, _ in DELAY_CATEGORIES]


def categorize_delays(delays) -> pd.Categorical:
    """
    Label each predicted delay: On Time (<= 0), Minor Delay (> 0 and <= 15),
    Moderate Delay (> 15 and <= 30) or Severe Delay (> 30). NaN stays NaN.
    """

    values = np.asarray(delays, dtype=float)
    codes = np.searchsorted(_UPPER_BOUNDS, values, side="left")
    codes = np.where(np.isnan(values), -1, codes)

    return pd.Categorical.from_codes(codes, categories=_CATEGORY_LABELS)


def summarize_delay_categories(df: pd.DataFrame) -> pd.DataFrame:
    """One row per category (all four, in order): flights and percentage."""

    codes = categorize_delays(df[PREDICTION_COLUMN]).codes
    counts = np.bincount(
        codes[codes >= 0], minlength=len(_CATEGORY_LABELS)
    )
    total = int(counts.sum())

    return pd.DataFrame(
        {
            "category": _CATEGORY_LABELS,
            "flights": counts,
            "percentage": (
                counts / total * 100 if total else np.zeros(len(counts))
            ),
        }
    )


# ============================================================
# GROUP AGGREGATIONS
# ============================================================

def _group_delay(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean predicted delay and flight count per group, highest first."""

    grouped = (
        df.groupby(keys, observed=True)[PREDICTION_COLUMN]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_predicted_delay", "count": "flights"})
    )

    for key in keys:
        grouped[key] = grouped[key].astype(str)

    return grouped.sort_values(
        ["avg_predicted_delay", "flights"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def aggregate_by_carrier(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, "Carrier", CARRIER_COLUMN)

    return _group_delay(df, [CARRIER_COLUMN]).rename(
        columns={CARRIER_COLUMN: "carrier"}
    )


def aggregate_by_origin(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, "Origin", ORIGIN_COLUMN)

    return _group_delay(df, [ORIGIN_COLUMN]).rename(
        columns={ORIGIN_COLUMN: "origin"}
    )


def aggregate_by_destination(df: pd.DataFrame) -> pd.DataFrame:
    _require(df, "Destination", DEST_COLUMN)

    return _group_delay(df, [DEST_COLUMN]).rename(
        columns={DEST_COLUMN: "destination"}
    )


def aggregate_by_route(df: pd.DataFrame) -> pd.DataFrame:
    """Per ORIGIN -> DEST route. Labels are built after grouping."""

    _require(df, "Route", ORIGIN_COLUMN, DEST_COLUMN)

    routes = _group_delay(df, [ORIGIN_COLUMN, DEST_COLUMN]).rename(
        columns={ORIGIN_COLUMN: "origin", DEST_COLUMN: "destination"}
    )
    routes.insert(0, "route", routes["origin"] + " → " + routes["destination"])

    return routes


def select_top(
    table: pd.DataFrame,
    top_n: int,
    min_flights: int = 1,
) -> pd.DataFrame:
    """The `top_n` highest-delay groups having at least `min_flights` flights."""

    return table[table["flights"] >= min_flights].head(top_n)


def aggregate_daily_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Average predicted delay and flight count per day, in date order."""

    _require(df, "Time", DATE_COLUMN)

    dated = df.loc[df[DATE_COLUMN].notna(), [DATE_COLUMN, PREDICTION_COLUMN]]

    if dated.empty:
        raise AnalysisUnavailableError(
            "Time analysis is unavailable because "
            f"{DATE_COLUMN} contains no valid dates."
        )

    return (
        dated.groupby(DATE_COLUMN)[PREDICTION_COLUMN]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                DATE_COLUMN: "date",
                "mean": "avg_predicted_delay",
                "count": "flights",
            }
        )
    )


# ============================================================
# ONE-SHOT ANALYSIS (what the dashboard caches)
# ============================================================

@dataclass
class BatchAnalysis:
    """Everything the dashboard shows for one filter selection."""

    kpis: BatchKPIs
    histogram: pd.DataFrame
    categories: pd.DataFrame
    carriers: Optional[pd.DataFrame] = None
    origins: Optional[pd.DataFrame] = None
    destinations: Optional[pd.DataFrame] = None
    routes: Optional[pd.DataFrame] = None
    daily: Optional[pd.DataFrame] = None
    # A small slice of the matching rows for the data table.
    preview: pd.DataFrame = field(default_factory=pd.DataFrame)
    # section name -> why it cannot be shown
    unavailable: dict[str, str] = field(default_factory=dict)


def run_analysis(
    df: pd.DataFrame,
    filters: Optional[AnalysisFilters] = None,
    preview_rows: int = PREVIEW_ROWS,
) -> BatchAnalysis:
    """
    Filter once, then compute every aggregation from the filtered frame.
    A section that cannot be built (missing column, no valid dates) is
    recorded in `unavailable` instead of failing the whole analysis.
    """

    matching = apply_filters(df, filters or AnalysisFilters())

    analysis = BatchAnalysis(
        kpis=calculate_batch_kpis(matching),
        histogram=delay_histogram(matching),
        categories=summarize_delay_categories(matching),
        preview=matching.head(preview_rows).copy(),
    )

    for section, build in (
        ("carriers", aggregate_by_carrier),
        ("origins", aggregate_by_origin),
        ("destinations", aggregate_by_destination),
        ("routes", aggregate_by_route),
        ("daily", aggregate_daily_predictions),
    ):
        try:
            setattr(analysis, section, build(matching))
        except AnalysisUnavailableError as exc:
            analysis.unavailable[section] = str(exc)

    return analysis
