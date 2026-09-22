"""
Altair charts for the Batch Analysis dashboard.

Altair ships with Streamlit, so it adds no dependency. Every chart takes an
already-aggregated frame (histogram bins, top-N groups, days, four
categories) and embeds only that: raw prediction rows never reach the
browser. Charts support hover tooltips, zoom / pan, and (categories)
legend selection.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from src.batch.analysis import DELAY_CATEGORIES, DELAY_THRESHOLD_MINUTES


# Palette of the existing dashboard (teal accents) plus semantic colours
# for delay severity. No text colours are set, so axes and labels follow
# the Streamlit theme (light or dark).
ACCENT = "#16836f"
REFERENCE = "#e5534b"
CATEGORY_COLORS = ["#16836f", "#d9b44a", "#e07b39", "#c0392b"]

DELAY_AXIS_TITLE = "Average predicted delay (minutes)"


def histogram_chart(
    histogram: pd.DataFrame,
    threshold: float = DELAY_THRESHOLD_MINUTES,
) -> alt.LayerChart:
    """Predicted arrival delay distribution with a reference line."""

    bars = (
        alt.Chart()
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X(
                "bin_start:Q",
                bin="binned",
                title="Predicted Arrival Delay (minutes)",
            ),
            x2="bin_end:Q",
            y=alt.Y("flights:Q", title="Number of Flights"),
            tooltip=[
                alt.Tooltip("bin_start:Q", title="From (min)", format=".1f"),
                alt.Tooltip("bin_end:Q", title="To (min)", format=".1f"),
                alt.Tooltip("flights:Q", title="Flights", format=","),
            ],
        )
    )

    reference = pd.DataFrame(
        {"x": [threshold], "label": [f"{threshold:g} min"]}
    )
    rule = (
        alt.Chart(reference)
        .mark_rule(color=REFERENCE, strokeDash=[6, 4], size=2)
        .encode(x="x:Q")
    )
    label = (
        alt.Chart(reference)
        .mark_text(
            align="left", baseline="top", dx=6, dy=2, color=REFERENCE
        )
        .encode(x="x:Q", y=alt.value(0), text="label:N")
    )

    return (
        alt.layer(bars, rule, label, data=histogram)
        .properties(height=320)
        .interactive()
    )


def delay_bar_chart(
    table: pd.DataFrame,
    category: str,
    category_title: str,
    label_with_flights: bool = False,
) -> alt.Chart:
    """
    Horizontal bars of average predicted delay, highest at the top, in the
    order of `table` (callers pass the top-N slice, never every group).
    """

    data = table.copy()

    if label_with_flights:
        data["label"] = (
            data[category] + " (" + data["flights"].map("{:,}".format) + ")"
        )
        axis_field = "label"
        y_title = f"{category_title} (flights)"
    else:
        axis_field = category
        y_title = category_title

    return (
        alt.Chart(data)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("avg_predicted_delay:Q", title=DELAY_AXIS_TITLE),
            y=alt.Y(
                f"{axis_field}:N",
                sort=data[axis_field].tolist(),
                title=y_title,
            ),
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category_title),
                alt.Tooltip(
                    "avg_predicted_delay:Q",
                    title="Avg predicted delay (min)",
                    format=".2f",
                ),
                alt.Tooltip("flights:Q", title="Flights", format=","),
            ],
        )
        .properties(height=max(120, 30 * len(data) + 50))
        .interactive()
    )


def daily_chart(daily: pd.DataFrame) -> alt.LayerChart:
    """Average predicted delay per day (line) over flights per day (bars)."""

    x = alt.X("date:T", title="Date")
    tooltip = [
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip(
            "avg_predicted_delay:Q",
            title="Avg predicted delay (min)",
            format=".2f",
        ),
        alt.Tooltip("flights:Q", title="Flights", format=","),
    ]

    # Zoom / pan along the date axis; both layers share the x scale.
    zoom = alt.selection_interval(bind="scales", encodings=["x"])

    flights = (
        alt.Chart()
        .mark_bar(color=ACCENT, opacity=0.25)
        .encode(
            x=x,
            y=alt.Y(
                "flights:Q",
                title="Flights per day",
                axis=alt.Axis(orient="right"),
            ),
            tooltip=tooltip,
        )
    )
    delay = (
        alt.Chart()
        .mark_line(color=ACCENT, point=alt.OverlayMarkDef(color=ACCENT))
        .encode(
            x=x,
            y=alt.Y("avg_predicted_delay:Q", title=DELAY_AXIS_TITLE),
            tooltip=tooltip,
        )
        .add_params(zoom)
    )

    return (
        alt.layer(flights, delay, data=daily)
        .resolve_scale(y="independent")
        .properties(height=340)
    )


def category_chart(summary: pd.DataFrame) -> alt.Chart:
    """Flights per delay category; click a legend entry to highlight it."""

    order = [label for label, _ in DELAY_CATEGORIES]
    selected = alt.selection_point(fields=["category"], bind="legend")

    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "category:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("flights:Q", title="Number of Flights"),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=order, range=CATEGORY_COLORS),
                legend=alt.Legend(title="Delay category"),
            ),
            opacity=alt.condition(selected, alt.value(1), alt.value(0.25)),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("flights:Q", title="Flights", format=","),
                alt.Tooltip("percentage:Q", title="Share (%)", format=".1f"),
            ],
        )
        .add_params(selected)
        .properties(height=300)
    )
