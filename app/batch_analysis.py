"""
"Batch Analysis" section of the Batch Prediction page.

Shown after a batch completes, from the predictions CSV the dashboard has
already downloaded from the API, so no model inference is repeated. All
computation lives in src/batch/analysis.py and the chart specs in
src/batch/charts.py; this module is layout and caching only.

Performance: the CSV is parsed once per batch (cache_resource: the same
DataFrame object is reused, never copied). Each filter selection is
aggregated once (cache_data holds only small result tables), so changing a
Top N selector or a chart does not recompute any groupby. The browser only
ever receives aggregated data.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import streamlit as st

from src.batch import charts
from src.batch.analysis import (
    DEFAULT_TABLE_COLUMNS,
    DEFAULT_TOP_N,
    DELAY_THRESHOLD_MINUTES,
    PREVIEW_ROWS,
    TOP_N_CHOICES,
    AnalysisError,
    AnalysisFilters,
    BatchAnalysis,
    BatchKPIs,
    FilterOptions,
    LoadedPredictions,
    read_prediction_csv,
    run_analysis,
    select_top,
)


logger = logging.getLogger(__name__)


# ============================================================
# CACHING
# ============================================================

@st.cache_resource(max_entries=2, show_spinner="Loading predictions...")
def load_predictions(batch_id: str, _content: bytes) -> LoadedPredictions:
    """Parse the predictions CSV once per batch (`_content` is not hashed)."""

    return read_prediction_csv(_content)


@st.cache_data(max_entries=64, show_spinner="Analysing predictions...")
def cached_analysis(
    batch_id: str,
    filter_key: tuple,
    _frame: pd.DataFrame,
) -> BatchAnalysis:
    """One filter + aggregation pass per (batch, filter selection)."""

    carriers, origins, destinations, date_range = filter_key

    return run_analysis(
        _frame,
        AnalysisFilters(carriers, origins, destinations, date_range),
    )


# ============================================================
# FORMATTING
# ============================================================

def _minutes(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:,.1f} min"


def _percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}%"


# ============================================================
# SECTIONS
# ============================================================

def _render_kpis(kpis: BatchKPIs) -> None:
    first = st.columns(3)
    second = st.columns(3)

    cards = (
        (first[0], "Total Flights", f"{kpis.total_flights:,}"),
        (first[1], "Average Predicted Delay", _minutes(kpis.average)),
        (first[2], "Median Predicted Delay", _minutes(kpis.median)),
        (second[0], "Maximum Predicted Delay", _minutes(kpis.maximum)),
        (second[1], "Minimum Predicted Delay", _minutes(kpis.minimum)),
        (
            second[2],
            f"Flights Predicted > {DELAY_THRESHOLD_MINUTES:g} min",
            _percent(kpis.percent_over_threshold),
        ),
    )

    for column, label, value in cards:
        column.metric(label, value, border=True)


def _render_filters(
    batch_id: str,
    options: FilterOptions,
) -> tuple[AnalysisFilters, int]:
    """Filter widgets. Returns the selection and the minimum group size."""

    def key(name: str) -> str:
        # Per-batch keys: a new batch never inherits stale selections.
        return f"analysis_{name}_{batch_id}"

    with st.expander("Analysis Filters", expanded=True):
        carrier_col, origin_col, dest_col, date_col = st.columns(4)

        carriers: list[str] = []
        origins: list[str] = []
        destinations: list[str] = []
        date_range = None

        if options.carriers:
            carriers = carrier_col.multiselect(
                "Carrier", options.carriers, placeholder="All",
                key=key("carrier"),
            )
        if options.origins:
            origins = origin_col.multiselect(
                "Origin", options.origins, placeholder="All",
                key=key("origin"),
            )
        if options.destinations:
            destinations = dest_col.multiselect(
                "Destination", options.destinations, placeholder="All",
                key=key("dest"),
            )
        if options.date_min and options.date_max:
            selected = date_col.date_input(
                "Date range",
                value=(options.date_min, options.date_max),
                min_value=options.date_min,
                max_value=options.date_max,
                key=key("date"),
            )
            # Mid-selection (only a start date) or the full range = no filter.
            if (
                isinstance(selected, tuple)
                and len(selected) == 2
                and selected != (options.date_min, options.date_max)
            ):
                date_range = selected

        min_flights = st.number_input(
            "Minimum flights per carrier, airport or route",
            min_value=1,
            value=1,
            step=1,
            key=key("min_flights"),
            help=(
                "Hides groups with fewer flights, so a single delayed "
                "flight cannot top the rankings."
            ),
        )

    return (
        AnalysisFilters(
            tuple(carriers), tuple(origins), tuple(destinations), date_range
        ),
        int(min_flights),
    )


def _top_n_selector(label: str, key: str) -> int:
    return st.selectbox(
        label,
        TOP_N_CHOICES,
        index=TOP_N_CHOICES.index(DEFAULT_TOP_N),
        key=key,
    )


def _render_chart(chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        logger.exception("Chart could not be rendered.")
        st.error("This chart could not be displayed.")


def _unavailable(analysis: BatchAnalysis, section: str) -> bool:
    """Show the reason and return True when a section cannot be built."""

    message = analysis.unavailable.get(section)

    if message:
        st.info(message)

    return message is not None


def _render_bar_section(
    analysis: BatchAnalysis,
    section: str,
    batch_id: str,
    title: str,
    category: str,
    category_title: str,
    min_flights: int,
) -> None:
    st.markdown(f"**{title}**")

    if _unavailable(analysis, section):
        return

    top_n = _top_n_selector("Top N", f"analysis_topn_{section}_{batch_id}")
    top = select_top(getattr(analysis, section), top_n, min_flights)

    if top.empty:
        st.info("No groups have that many flights. Lower the minimum.")
        return

    _render_chart(charts.delay_bar_chart(top, category, category_title))


def _render_carriers(analysis: BatchAnalysis, min_flights: int) -> None:
    st.markdown("**Average Predicted Delay by Carrier**")

    if _unavailable(analysis, "carriers"):
        return

    carriers = select_top(analysis.carriers, len(analysis.carriers), min_flights)

    if carriers.empty:
        st.info("No carriers have that many flights. Lower the minimum.")
        return

    _render_chart(
        charts.delay_bar_chart(
            carriers, "carrier", "Carrier", label_with_flights=True
        )
    )


def _render_categories(analysis: BatchAnalysis) -> None:
    summary = analysis.categories
    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        _render_chart(charts.category_chart(summary))

    with table_col:
        st.dataframe(
            summary.rename(
                columns={
                    "category": "Category",
                    "flights": "Flights",
                    "percentage": "Share (%)",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Flights": st.column_config.NumberColumn(format="localized"),
                "Share (%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )


def _render_table(
    analysis: BatchAnalysis, matching: int, batch_id: str
) -> None:
    preview = analysis.preview

    with st.expander("View Prediction Data"):
        available = list(preview.columns)
        defaults = [c for c in DEFAULT_TABLE_COLUMNS if c in available]

        columns = st.multiselect(
            "Columns", available, default=defaults,
            key=f"analysis_columns_{batch_id}",
        )

        if not columns:
            st.info("Select at least one column to display.")
            return

        shown = preview[columns].copy()

        if "FL_DATE" in shown.columns:
            shown["FL_DATE"] = shown["FL_DATE"].dt.strftime("%Y-%m-%d")

        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption(
            f"Showing the first {min(PREVIEW_ROWS, matching):,} of "
            f"{matching:,} matching flights. Download the predictions "
            "for the complete dataset."
        )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def render_batch_analysis(batch_id: str, content: bytes) -> None:
    """
    Render the analysis for a completed batch.

    Runs as a fragment, so touching a filter or selector reruns only this
    section, not the upload widget or the status panel.
    """

    @st.fragment
    def section() -> None:
        st.divider()
        st.subheader("Batch Analysis")

        try:
            loaded = load_predictions(batch_id, content)
        except AnalysisError as exc:
            st.error(f"Analysis is unavailable. {exc}")
            return
        except Exception:
            logger.exception("Could not load the predictions for analysis.")
            st.error("Analysis is unavailable. The predictions could not be read.")
            return

        for note in loaded.notes:
            st.warning(note)

        kpi_area = st.container()

        filters, min_flights = _render_filters(batch_id, loaded.options)

        try:
            analysis = cached_analysis(
                batch_id,
                (
                    filters.carriers,
                    filters.origins,
                    filters.destinations,
                    filters.date_range,
                ),
                loaded.frame,
            )
        except Exception:
            logger.exception("Batch analysis failed.")
            st.error("The analysis could not be computed.")
            return

        matching = analysis.kpis.total_flights

        with kpi_area:
            _render_kpis(analysis.kpis)

            if filters.is_active:
                st.caption(
                    f"Filters applied: {matching:,} of "
                    f"{len(loaded.frame):,} flights."
                )

        if matching == 0:
            st.warning("No flights match the selected filters.")
            return

        st.subheader("Delay Distribution")
        _render_chart(charts.histogram_chart(analysis.histogram))

        st.subheader("Carrier Analysis")
        _render_carriers(analysis, min_flights)

        st.subheader("Airport Analysis")
        origin_col, dest_col = st.columns(2)

        with origin_col:
            _render_bar_section(
                analysis, "origins", batch_id,
                "Average Predicted Delay by Origin Airport",
                "origin", "Origin airport", min_flights,
            )
        with dest_col:
            _render_bar_section(
                analysis, "destinations", batch_id,
                "Average Predicted Delay by Destination Airport",
                "destination", "Destination airport", min_flights,
            )

        st.subheader("Route Analysis")
        _render_bar_section(
            analysis, "routes", batch_id,
            "Average Predicted Delay by Route",
            "route", "Route", min_flights,
        )

        # Time analysis is optional: without usable dates the section
        # collapses to a one-line explanation instead of an empty chart.
        if "daily" in analysis.unavailable:
            st.caption(analysis.unavailable["daily"])
        else:
            st.subheader("Time Analysis")
            st.markdown("**Predicted Delay Over Time**")
            _render_chart(charts.daily_chart(analysis.daily))

        st.subheader("Delay Categories")
        _render_categories(analysis)

        st.subheader("Prediction Data")
        _render_table(analysis, matching, batch_id)

    section()
