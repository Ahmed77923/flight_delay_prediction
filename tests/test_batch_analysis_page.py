"""Renders the Batch Analysis section with Streamlit's AppTest."""

import importlib.util
import sys
from pathlib import Path

from datetime import date

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

# Streamlit imports app/batch_analysis.py as a sibling of the script. Load
# it the same way here without putting app/ on sys.path, where app/app.py
# would shadow the `app` package other tests import.
_path = Path(__file__).resolve().parents[1] / "app" / "batch_analysis.py"
_spec = importlib.util.spec_from_file_location("batch_analysis", _path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["batch_analysis"] = _module
_spec.loader.exec_module(_module)


def analysis_script(batch_id, content):
    from batch_analysis import render_batch_analysis

    render_batch_analysis(batch_id, content)


def build_csv(rows=8, with_dates=True, with_carrier=True) -> bytes:
    df = pd.DataFrame(
        {
            "FL_DATE": ["2026-09-01"] * (rows // 2) + ["2026-09-02"] * (rows - rows // 2),
            "OP_UNIQUE_CARRIER": ["AA", "DL"] * (rows // 2),
            "ORIGIN": ["JFK", "ATL"] * (rows // 2),
            "DEST": ["LAX", "JFK"] * (rows // 2),
            "CRS_DEP_TIME": ["0800"] * rows,
            "DISTANCE": [500.0] * rows,
            "predicted_arr_delay": [float(i * 5 - 10) for i in range(rows)],
        }
    )

    if not with_dates:
        df = df.drop(columns=["FL_DATE"])
    if not with_carrier:
        df = df.drop(columns=["OP_UNIQUE_CARRIER"])

    return df.to_csv(index=False).encode()


def run_app(batch_id, content):
    at = AppTest.from_function(
        analysis_script, args=(batch_id, content), default_timeout=30
    )
    return at.run()


def metric_values(at):
    return {m.label: m.value for m in at.metric}


def test_section_renders_kpis_and_all_charts():
    at = run_app("ui-full", build_csv())

    assert not at.exception

    kpis = metric_values(at)
    assert kpis["Total Flights"] == "8"
    assert kpis["Minimum Predicted Delay"] == "-10.0 min"
    assert kpis["Maximum Predicted Delay"] == "25.0 min"
    # 20 and 25 are above 15 (15 itself is not): 2 of 8 flights.
    assert kpis["Flights Predicted > 15 min"] == "25.0%"

    headers = [s.value for s in at.subheader]
    for title in (
        "Batch Analysis", "Delay Distribution", "Carrier Analysis",
        "Airport Analysis", "Route Analysis", "Time Analysis",
        "Delay Categories", "Prediction Data",
    ):
        assert title in headers

    # histogram, carriers, origins, destinations, routes, daily, categories
    assert len(at.get("vega_lite_chart")) == 7


def test_carrier_filter_updates_the_analysis():
    at = run_app("ui-filter", build_csv())

    at.multiselect(key="analysis_carrier_ui-filter").set_value(["AA"]).run()

    assert not at.exception
    assert metric_values(at)["Total Flights"] == "4"
    assert any("Filters applied" in c.value for c in at.caption)


def test_filter_matching_nothing_shows_a_warning_not_charts():
    at = run_app("ui-empty", build_csv())

    at.multiselect(key="analysis_carrier_ui-empty").set_value(["AA"]).run()
    at.multiselect(key="analysis_origin_ui-empty").set_value(["ATL"]).run()

    assert not at.exception
    assert metric_values(at)["Total Flights"] == "0"
    assert any("No flights match" in w.value for w in at.warning)
    assert len(at.get("vega_lite_chart")) == 0


def test_top_n_selector_changes_without_error():
    at = run_app("ui-topn", build_csv())

    at.selectbox(key="analysis_topn_origins_ui-topn").set_value(5).run()

    assert not at.exception


def test_time_analysis_hidden_without_fl_date():
    at = run_app("ui-nodate", build_csv(with_dates=False))

    assert not at.exception
    assert "Time Analysis" not in [s.value for s in at.subheader]
    assert any(
        "Time analysis is unavailable because FL_DATE is not present."
        in c.value
        for c in at.caption
    )


def test_carrier_message_without_carrier_column():
    at = run_app("ui-nocarrier", build_csv(with_carrier=False))

    assert not at.exception
    assert any(
        "Carrier analysis is unavailable because OP_UNIQUE_CARRIER is not present."
        in i.value
        for i in at.info
    )


def test_missing_prediction_column_shows_an_error_not_a_crash():
    content = b"FL_DATE,ORIGIN\n2026-09-01,JFK\n"

    at = run_app("ui-nopred", content)

    assert not at.exception
    assert any("predicted_arr_delay" in e.value for e in at.error)


def test_empty_predictions_show_an_error_not_a_crash():
    at = run_app("ui-emptyfile", b"")

    assert not at.exception
    assert any("empty" in e.value for e in at.error)


# ============================================================
# WHAT THE BROWSER RECEIVES
# ============================================================

def sent_rows(at):
    """Rows in every dataset Streamlit ships with the charts."""

    import pyarrow as pa

    rows = []
    for chart in at.get("vega_lite_chart"):
        rows.append(
            sum(
                pa.ipc.open_stream(ds.data.data).read_all().num_rows
                for ds in chart.proto.datasets
            )
        )
    return rows


def many_flights_csv(rows, airports=12) -> bytes:
    rng = np.random.default_rng(1)
    codes = np.array([f"A{i:02d}" for i in range(airports)])

    return pd.DataFrame(
        {
            "FL_DATE": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(rng.integers(0, 60, rows), unit="D"),
            "OP_UNIQUE_CARRIER": rng.choice(["AA", "DL", "UA"], rows),
            "ORIGIN": rng.choice(codes, rows),
            "DEST": rng.choice(codes, rows),
            "predicted_arr_delay": rng.normal(5, 15, rows).round(2),
        }
    ).to_csv(index=False).encode()


def test_browser_only_receives_aggregated_rows():
    at = run_app("ui-payload", many_flights_csv(30_000))

    assert not at.exception

    rows = sent_rows(at)

    assert len(rows) == 7
    assert max(rows) <= 100  # bins / top-N groups / days, never flights
    assert sum(rows) < 300   # vs 30,000 raw rows


def test_top_n_limits_the_rows_charted():
    at = run_app("ui-topn-rows", many_flights_csv(5_000))

    origins = lambda: sent_rows(at)[2]  # histogram, carriers, then origins

    assert origins() == 10  # default Top N

    at.selectbox(key="analysis_topn_origins_ui-topn-rows").set_value(5).run()
    assert origins() == 5

    at.selectbox(key="analysis_topn_origins_ui-topn-rows").set_value(20).run()
    assert origins() == 12  # only 12 airports exist


def test_minimum_flights_hides_small_groups():
    at = run_app("ui-minflights", many_flights_csv(5_000))

    at.number_input(key="analysis_min_flights_ui-minflights").set_value(
        10_000
    ).run()

    assert not at.exception
    assert any("Lower the minimum" in i.value for i in at.info)


def test_date_range_filter_updates_the_analysis():
    at = run_app("ui-date", build_csv())

    picker = at.date_input(key="analysis_date_ui-date")
    picker.set_value((date(2026, 9, 2), date(2026, 9, 2))).run()

    assert not at.exception
    assert metric_values(at)["Total Flights"] == "4"
    assert any("4 of 8" in c.value for c in at.caption)


def test_half_selected_date_range_does_not_filter():
    at = run_app("ui-halfdate", build_csv())

    at.date_input(key="analysis_date_ui-halfdate").set_value(
        (date(2026, 9, 2),)
    ).run()

    assert not at.exception
    assert metric_values(at)["Total Flights"] == "8"


def test_filter_selections_are_per_batch():
    """Filters chosen for one batch never leak into the next one."""

    first = run_app("ui-b1", build_csv())
    first.multiselect(key="analysis_carrier_ui-b1").set_value(["AA"]).run()

    second = run_app("ui-b2", build_csv())

    assert metric_values(second)["Total Flights"] == "8"
