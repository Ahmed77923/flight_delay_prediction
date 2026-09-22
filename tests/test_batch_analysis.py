import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.batch import charts
from src.batch.analysis import (
    DELAY_CATEGORIES,
    DELAY_THRESHOLD_MINUTES,
    PREDICTION_COLUMN,
    AnalysisError,
    AnalysisFilters,
    aggregate_by_carrier,
    aggregate_by_destination,
    aggregate_by_origin,
    aggregate_by_route,
    aggregate_daily_predictions,
    apply_filters,
    calculate_batch_kpis,
    categorize_delays,
    delay_histogram,
    read_prediction_csv,
    run_analysis,
    select_top,
    summarize_delay_categories,
)


# ============================================================
# FIXTURE: 8 flights whose statistics are easy to verify by hand
# ============================================================
#
#  row  date        carrier origin dest  predicted
#   0   2026-09-01  AA      JFK    LAX     -5
#   1   2026-09-01  AA      JFK    LAX     10
#   2   2026-09-01  DL      ATL    JFK     20
#   3   2026-09-01  DL      ATL    LAX     40
#   4   2026-09-02  AA      JFK    LAX      0
#   5   2026-09-02  UA      ORD    JFK     15
#   6   2026-09-02  UA      ORD    LAX     35
#   7   2026-09-03  DL      ATL    JFK     12

@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "FL_DATE": pd.to_datetime(
                ["2026-09-01"] * 4 + ["2026-09-02"] * 3 + ["2026-09-03"]
            ),
            "OP_UNIQUE_CARRIER": ["AA", "AA", "DL", "DL", "AA", "UA", "UA", "DL"],
            "ORIGIN": ["JFK", "JFK", "ATL", "ATL", "JFK", "ORD", "ORD", "ATL"],
            "DEST": ["LAX", "LAX", "JFK", "LAX", "LAX", "JFK", "LAX", "JFK"],
            PREDICTION_COLUMN: [-5.0, 10, 20, 40, 0, 15, 35, 12],
        }
    )


def as_categorical(df):
    """The dtypes read_prediction_csv produces."""

    out = df.copy()
    for column in ("OP_UNIQUE_CARRIER", "ORIGIN", "DEST"):
        out[column] = out[column].astype("category")
    return out


def to_csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode()


# ============================================================
# KPIs
# ============================================================

def test_kpis(predictions):
    kpis = calculate_batch_kpis(predictions)

    assert kpis.total_flights == 8
    assert kpis.average == pytest.approx(127 / 8)
    assert kpis.median == pytest.approx(13.5)
    assert kpis.maximum == 40
    assert kpis.minimum == -5
    # 20, 40 and 35 are above 15; exactly 15 is not "> 15".
    assert kpis.percent_over_threshold == pytest.approx(37.5)


def test_kpis_of_empty_frame(predictions):
    kpis = calculate_batch_kpis(predictions.iloc[0:0])

    assert kpis.total_flights == 0
    assert kpis.average is None and kpis.percent_over_threshold is None


def test_kpi_threshold_is_15_minutes():
    assert DELAY_THRESHOLD_MINUTES == 15


# ============================================================
# DELAY CATEGORIES
# ============================================================

def test_categorize_delays_boundaries():
    labels = categorize_delays(
        [-10, 0, 0.01, 15, 15.01, 30, 30.01, 500, np.nan]
    )

    assert [None if pd.isna(x) else x for x in labels] == [
        "On Time",       # <= 0
        "On Time",       # exactly 0
        "Minor Delay",   # > 0
        "Minor Delay",   # exactly 15
        "Moderate Delay",
        "Moderate Delay",  # exactly 30
        "Severe Delay",
        "Severe Delay",
        None,
    ]


def test_category_summary_counts_and_percentages(predictions):
    summary = summarize_delay_categories(predictions)

    assert summary["category"].tolist() == [
        "On Time", "Minor Delay", "Moderate Delay", "Severe Delay"
    ]
    assert summary["flights"].tolist() == [2, 3, 1, 2]
    assert summary["percentage"].tolist() == pytest.approx(
        [25.0, 37.5, 12.5, 25.0]
    )
    assert summary["percentage"].sum() == pytest.approx(100)


def test_category_summary_keeps_empty_categories(predictions):
    only_on_time = predictions.assign(**{PREDICTION_COLUMN: -3.0})

    summary = summarize_delay_categories(only_on_time)

    assert summary["flights"].tolist() == [8, 0, 0, 0]


def test_category_thresholds_live_in_one_place():
    assert [bound for _, bound in DELAY_CATEGORIES] == [
        0.0, DELAY_THRESHOLD_MINUTES, 30.0, float("inf")
    ]


# ============================================================
# AGGREGATIONS
# ============================================================

def test_carrier_aggregation_sorted_by_average(predictions):
    table = aggregate_by_carrier(as_categorical(predictions))

    assert table["carrier"].tolist() == ["UA", "DL", "AA"]
    assert table["avg_predicted_delay"].tolist() == pytest.approx(
        [25.0, 24.0, 5 / 3]
    )
    assert table["flights"].tolist() == [2, 3, 3]


def test_origin_aggregation(predictions):
    table = aggregate_by_origin(predictions)

    assert table["origin"].tolist() == ["ORD", "ATL", "JFK"]
    assert table["avg_predicted_delay"].tolist() == pytest.approx(
        [25.0, 24.0, 5 / 3]
    )
    assert table["flights"].tolist() == [2, 3, 3]


def test_destination_aggregation(predictions):
    table = aggregate_by_destination(as_categorical(predictions))

    assert table["destination"].tolist() == ["LAX", "JFK"]
    assert table["avg_predicted_delay"].tolist() == pytest.approx(
        [16.0, 47 / 3]
    )
    assert table["flights"].tolist() == [5, 3]


def test_route_aggregation_uses_arrow_labels(predictions):
    table = aggregate_by_route(as_categorical(predictions))

    assert table["route"].tolist() == [
        "ATL → LAX", "ORD → LAX", "ATL → JFK", "ORD → JFK", "JFK → LAX",
    ]
    assert table["avg_predicted_delay"].tolist() == pytest.approx(
        [40.0, 35.0, 16.0, 15.0, 5 / 3]
    )
    assert table["flights"].tolist() == [1, 1, 2, 1, 3]
    assert {"origin", "destination"} <= set(table.columns)


def test_group_labels_are_plain_strings(predictions):
    table = aggregate_by_carrier(as_categorical(predictions))

    assert table["carrier"].dtype == object


def test_select_top_and_minimum_flights(predictions):
    routes = aggregate_by_route(predictions)

    assert select_top(routes, 2)["route"].tolist() == [
        "ATL → LAX", "ORD → LAX"
    ]
    # Routes flown at least twice, still highest average first.
    assert select_top(routes, 10, min_flights=2)["route"].tolist() == [
        "ATL → JFK", "JFK → LAX"
    ]


def test_daily_aggregation(predictions):
    daily = aggregate_daily_predictions(predictions)

    assert daily["date"].tolist() == list(
        pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"])
    )
    assert daily["avg_predicted_delay"].tolist() == pytest.approx(
        [16.25, 50 / 3, 12.0]
    )
    assert daily["flights"].tolist() == [4, 3, 1]


def test_daily_aggregation_ignores_invalid_dates(predictions):
    predictions.loc[0, "FL_DATE"] = pd.NaT

    daily = aggregate_daily_predictions(predictions)

    assert daily["flights"].sum() == 7


def test_daily_aggregation_without_any_valid_date(predictions):
    predictions["FL_DATE"] = pd.NaT

    with pytest.raises(AnalysisError, match="no valid dates"):
        aggregate_daily_predictions(predictions)


def test_histogram_counts_every_flight_in_fixed_bins(predictions):
    hist = delay_histogram(predictions, bins=5)

    assert len(hist) == 5
    assert hist["flights"].sum() == 8
    assert hist["bin_start"].iloc[0] == -5 and hist["bin_end"].iloc[-1] == 40


def test_histogram_of_constant_values():
    hist = delay_histogram(pd.DataFrame({PREDICTION_COLUMN: [7.0] * 4}))

    assert hist["flights"].sum() == 4


# ============================================================
# FILTERING
# ============================================================

def test_no_filter_returns_the_same_frame(predictions):
    assert apply_filters(predictions, AnalysisFilters()) is predictions


def test_filter_by_carrier(predictions):
    result = apply_filters(predictions, AnalysisFilters(carriers=("AA",)))

    assert len(result) == 3
    assert set(result["OP_UNIQUE_CARRIER"]) == {"AA"}


def test_filter_multiple_values_and_columns(predictions):
    filtered = apply_filters(
        as_categorical(predictions),
        AnalysisFilters(carriers=("AA", "DL"), origins=("ATL",)),
    )

    assert filtered[PREDICTION_COLUMN].tolist() == [20, 40, 12]


def test_filter_by_destination(predictions):
    result = apply_filters(predictions, AnalysisFilters(destinations=("JFK",)))

    assert result[PREDICTION_COLUMN].tolist() == [20, 15, 12]


def test_filter_by_date_range_is_inclusive(predictions):
    result = apply_filters(
        predictions,
        AnalysisFilters(date_range=(date(2026, 9, 2), date(2026, 9, 3))),
    )

    assert result[PREDICTION_COLUMN].tolist() == [0, 15, 35, 12]


def test_filters_combine_with_and(predictions):
    result = apply_filters(
        predictions,
        AnalysisFilters(
            carriers=("AA",),
            date_range=(date(2026, 9, 1), date(2026, 9, 1)),
        ),
    )

    assert result[PREDICTION_COLUMN].tolist() == [-5, 10]


def test_filter_with_no_match_is_empty(predictions):
    assert apply_filters(predictions, AnalysisFilters(carriers=("ZZ",))).empty


def test_filter_on_absent_column_is_ignored(predictions):
    without_carrier = predictions.drop(columns=["OP_UNIQUE_CARRIER"])

    result = apply_filters(without_carrier, AnalysisFilters(carriers=("AA",)))

    assert len(result) == 8


def test_filters_update_the_analysis(predictions):
    everything = run_analysis(predictions)
    only_aa = run_analysis(predictions, AnalysisFilters(carriers=("AA",)))

    assert everything.kpis.total_flights == 8
    assert only_aa.kpis.total_flights == 3
    assert only_aa.carriers["carrier"].tolist() == ["AA"]
    assert only_aa.daily["flights"].sum() == 3


# ============================================================
# run_analysis: MISSING OPTIONAL COLUMNS
# ============================================================

def test_missing_carrier_column_only_disables_carrier_analysis(predictions):
    analysis = run_analysis(predictions.drop(columns=["OP_UNIQUE_CARRIER"]))

    assert analysis.unavailable == {
        "carriers": (
            "Carrier analysis is unavailable because "
            "OP_UNIQUE_CARRIER is not present."
        )
    }
    assert analysis.origins is not None and analysis.daily is not None


def test_missing_date_column_only_disables_time_analysis(predictions):
    analysis = run_analysis(predictions.drop(columns=["FL_DATE"]))

    assert analysis.unavailable == {
        "daily": (
            "Time analysis is unavailable because FL_DATE is not present."
        )
    }
    assert analysis.carriers is not None


def test_missing_route_column_names_the_missing_one(predictions):
    analysis = run_analysis(predictions.drop(columns=["DEST"]))

    assert analysis.unavailable["routes"] == (
        "Route analysis is unavailable because DEST is not present."
    )
    assert "destinations" in analysis.unavailable
    assert analysis.origins is not None


def test_analysis_preview_is_limited(predictions):
    analysis = run_analysis(predictions, preview_rows=3)

    assert len(analysis.preview) == 3
    assert analysis.kpis.total_flights == 8


# ============================================================
# LOADING THE PREDICTIONS CSV
# ============================================================

def test_read_prediction_csv_loads_needed_columns_only(predictions):
    extra = predictions.assign(NOISE="x", ARR_DELAY=1.0, CRS_DEP_TIME="0800")

    loaded = read_prediction_csv(to_csv_bytes(extra))
    frame = loaded.frame

    assert "NOISE" not in frame.columns and "ARR_DELAY" not in frame.columns
    assert isinstance(frame["ORIGIN"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(frame["FL_DATE"])
    # Text such as "0800" keeps its leading zero.
    assert frame["CRS_DEP_TIME"].iloc[0] == "0800"
    assert loaded.notes == []
    assert loaded.file_rows == 8
    assert loaded.options.carriers == ("AA", "DL", "UA")
    assert loaded.options.date_min == date(2026, 9, 1)
    assert loaded.options.date_max == date(2026, 9, 3)


def test_read_prediction_csv_matches_in_memory_analysis(predictions):
    loaded = read_prediction_csv(to_csv_bytes(predictions))

    assert calculate_batch_kpis(loaded.frame) == calculate_batch_kpis(
        predictions
    )
    assert (
        aggregate_by_route(loaded.frame)["route"].tolist()
        == aggregate_by_route(predictions)["route"].tolist()
    )


def test_read_prediction_csv_reads_a_file_path(predictions, tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_bytes(to_csv_bytes(predictions))

    assert read_prediction_csv(path).file_rows == 8


def test_read_prediction_csv_parses_batch_output_dates(predictions):
    # The raw monthly files use e.g. "1/1/2026 12:00:00 AM".
    df = predictions.assign(
        FL_DATE=["9/1/2026 12:00:00 AM"] * 4
        + ["9/2/2026 12:00:00 AM"] * 3
        + ["9/3/2026 12:00:00 AM"]
    )

    frame = read_prediction_csv(to_csv_bytes(df)).frame

    assert frame["FL_DATE"].dt.day.tolist() == [1, 1, 1, 1, 2, 2, 2, 3]


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(AnalysisError, match="not found"):
        read_prediction_csv(tmp_path / "gone.csv")


def test_empty_file_is_reported():
    with pytest.raises(AnalysisError, match="empty"):
        read_prediction_csv(b"")


def test_header_only_file_is_reported(predictions):
    header_only = to_csv_bytes(predictions.iloc[0:0])

    with pytest.raises(AnalysisError, match="no rows"):
        read_prediction_csv(header_only)


def test_missing_prediction_column_is_reported(predictions):
    without = predictions.drop(columns=[PREDICTION_COLUMN])

    with pytest.raises(AnalysisError, match=PREDICTION_COLUMN):
        read_prediction_csv(to_csv_bytes(without))


def test_garbage_file_is_reported():
    with pytest.raises(AnalysisError):
        read_prediction_csv(b"\xff\xfe\x00\x81\x82\x83")


def test_invalid_prediction_values_are_dropped_with_a_note(predictions):
    text = to_csv_bytes(predictions).decode().splitlines()
    text[1] = text[1].rsplit(",", 1)[0] + ",not-a-number"
    text[2] = text[2].rsplit(",", 1)[0] + ","

    loaded = read_prediction_csv("\n".join(text).encode())

    assert len(loaded.frame) == 6
    assert loaded.frame[PREDICTION_COLUMN].notna().all()
    assert "2 row(s)" in loaded.notes[0]


def test_all_predictions_invalid_is_reported(predictions):
    broken = predictions.astype({PREDICTION_COLUMN: object})
    broken[PREDICTION_COLUMN] = "n/a"

    with pytest.raises(AnalysisError, match="no valid predictions"):
        read_prediction_csv(to_csv_bytes(broken))


def test_invalid_dates_become_nat_with_a_note(predictions):
    df = predictions.assign(FL_DATE=predictions["FL_DATE"].astype(str))
    df.loc[0, "FL_DATE"] = "not a date"

    loaded = read_prediction_csv(to_csv_bytes(df))

    assert loaded.frame["FL_DATE"].isna().sum() == 1
    assert len(loaded.frame) == 8
    assert "1 row(s)" in loaded.notes[0]
    # Everything except the time analysis still works.
    assert run_analysis(loaded.frame).daily["flights"].sum() == 7


def test_file_without_optional_columns_still_loads():
    loaded = read_prediction_csv(b"predicted_arr_delay\n1.5\n2.5\n")

    assert list(loaded.frame.columns) == [PREDICTION_COLUMN]
    assert loaded.options.carriers == ()
    assert loaded.options.date_min is None
    analysis = run_analysis(loaded.frame)
    assert analysis.kpis.total_flights == 2
    assert set(analysis.unavailable) == {
        "carriers", "origins", "destinations", "routes", "daily"
    }


# ============================================================
# LARGE DATA: charts only ever receive aggregates
# ============================================================

@pytest.fixture(scope="module")
def large_frame():
    rng = np.random.default_rng(0)
    n = 200_000
    airports = np.array([f"A{i:02d}" for i in range(60)])

    return pd.DataFrame(
        {
            "FL_DATE": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(rng.integers(0, 90, n), unit="D"),
            "OP_UNIQUE_CARRIER": pd.Categorical(
                rng.choice(["AA", "DL", "UA", "WN", "B6"], n)
            ),
            "ORIGIN": pd.Categorical(rng.choice(airports, n)),
            "DEST": pd.Categorical(rng.choice(airports, n)),
            PREDICTION_COLUMN: rng.normal(8, 20, n),
        }
    )


def embedded_rows(chart) -> int:
    datasets = chart.to_dict().get("datasets", {})

    return sum(len(rows) for rows in datasets.values())


def test_charts_embed_only_aggregated_rows(large_frame):
    analysis = run_analysis(large_frame)

    specs = {
        "histogram": charts.histogram_chart(analysis.histogram),
        "carriers": charts.delay_bar_chart(
            analysis.carriers, "carrier", "Carrier", True
        ),
        "origins": charts.delay_bar_chart(
            select_top(analysis.origins, 20), "origin", "Origin airport"
        ),
        "routes": charts.delay_bar_chart(
            select_top(analysis.routes, 20), "route", "Route"
        ),
        "daily": charts.daily_chart(analysis.daily),
        "categories": charts.category_chart(analysis.categories),
    }

    for name, chart in specs.items():
        assert embedded_rows(chart) <= 100, name

    # ...and the whole payload the browser gets is tiny next to 200k rows.
    total_json = sum(len(json.dumps(c.to_dict())) for c in specs.values())
    assert total_json < 150_000

    # The full route table is large, but only the top-N slice is charted.
    assert len(analysis.routes) > 1_000
    assert analysis.kpis.total_flights == 200_000
    assert analysis.histogram["flights"].sum() == 200_000
    assert len(analysis.preview) <= 1_000


def test_charts_render_a_single_row_of_data():
    single = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "avg_predicted_delay": [3.0],
            "flights": [1],
        }
    )

    assert charts.daily_chart(single).to_dict()


# ============================================================
# INTERACTIVITY: hover, zoom / pan, legend selection
# ============================================================

def find_values(node, key):
    """All values stored under `key` anywhere in a Vega-Lite spec."""

    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                found.append(v)
            found.extend(find_values(v, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(find_values(item, key))
    return found


def test_charts_are_interactive(predictions):
    analysis = run_analysis(predictions)

    zoomable = {
        "histogram": charts.histogram_chart(analysis.histogram),
        "bars": charts.delay_bar_chart(analysis.carriers, "carrier", "Carrier"),
        "daily": charts.daily_chart(analysis.daily),
    }

    for name, chart in zoomable.items():
        spec = chart.to_dict()
        assert "scales" in find_values(spec, "bind"), f"{name}: no zoom/pan"
        assert find_values(spec, "tooltip"), f"{name}: no hover tooltip"

    legend = charts.category_chart(analysis.categories).to_dict()
    assert "legend" in find_values(legend, "bind")
    assert find_values(legend, "tooltip")


def test_histogram_marks_the_15_minute_threshold(predictions):
    spec = charts.histogram_chart(delay_histogram(predictions)).to_dict()

    rule_data = [
        rows
        for rows in spec["datasets"].values()
        if rows and "x" in rows[0]
    ]

    assert rule_data and rule_data[0][0]["x"] == DELAY_THRESHOLD_MINUTES
