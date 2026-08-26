from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from config.config import Config


HISTORICAL_COLUMNS = [
    "carrier_historical_delay",
    "carrier_recent_delay_7",
    "carrier_recent_delay_30",
    "route_historical_delay",
    "route_recent_delay_7",
    "route_recent_delay_30",
    "origin_historical_delay",
    "origin_recent_delay_7",
    "origin_recent_delay_30",
    "carrier_origin_historical_delay",
    "aircraft_previous_delay",
]


@dataclass(frozen=True)
class _EntityHistory:
    timestamps: np.ndarray
    delays: np.ndarray


@dataclass(frozen=True)
class HistoricalFeatureIndex:
    """
    Pre-computed historical lookup index.

    The historical dataset is processed ONCE when the API starts.

    For every prediction request, we only perform:
        - dictionary lookup
        - numpy.searchsorted
        - mean calculation

    We do NOT concatenate the complete historical dataset
    with the user's request.
    """

    carrier: dict[str, _EntityHistory]
    route: dict[str, _EntityHistory]
    origin: dict[str, _EntityHistory]
    carrier_origin: dict[str, _EntityHistory]
    tail_num: dict[str, _EntityHistory]

    # =========================================================
    # BUILD INDEX
    # =========================================================

    @classmethod
    def from_frame(
        cls,
        history: pd.DataFrame,
    ) -> "HistoricalFeatureIndex":

        required_columns = {
            "FL_DATE",
            "CRS_DEP_TIME",
            "OP_UNIQUE_CARRIER",
            "ORIGIN",
            "DEST",
            "TAIL_NUM",
            Config.DATA.TARGET,
        }

        missing = sorted(
            required_columns - set(history.columns)
        )

        if missing:
            raise ValueError(
                "Historical data is missing required columns: "
                f"{missing}"
            )

        # -----------------------------------------------------
        # Create compact dataframe
        # -----------------------------------------------------

        scheduled_departure = _scheduled_departure(history)

        compact = pd.DataFrame(
            {
                "scheduled_departure": scheduled_departure,

                "carrier": (
                    history["OP_UNIQUE_CARRIER"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                ),

                "route": (
                    history["ORIGIN"].astype(str).str.strip().str.upper()
                    + "_"
                    + history["DEST"].astype(str).str.strip().str.upper()
                ),

                "origin": (
                    history["ORIGIN"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                ),

                "carrier_origin": (
                    history["OP_UNIQUE_CARRIER"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    + "_"
                    + history["ORIGIN"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                ),

                "tail_num": (
                    history["TAIL_NUM"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                ),

                "delay": pd.to_numeric(
                    history[Config.DATA.TARGET],
                    errors="coerce",
                ),
            }
        )

        # Remove rows that cannot be used
        compact = compact.dropna(
            subset=[
                "scheduled_departure",
                "delay",
            ]
        )

        # -----------------------------------------------------
        # Convert datetime to int64 nanoseconds
        # -----------------------------------------------------

        compact["timestamp"] = (
            compact["scheduled_departure"]
            .astype("int64")
        )

        # -----------------------------------------------------
        # Build lookup dictionaries
        # -----------------------------------------------------

        return cls(
            carrier=_build_lookup(
                compact,
                "carrier",
            ),

            route=_build_lookup(
                compact,
                "route",
            ),

            origin=_build_lookup(
                compact,
                "origin",
            ),

            carrier_origin=_build_lookup(
                compact,
                "carrier_origin",
            ),

            tail_num=_build_lookup(
                compact,
                "tail_num",
            ),
        )

    # =========================================================
    # FEATURES FOR ONE FLIGHT
    # =========================================================

    def features_for(
        self,
        flight: pd.Series,
    ) -> dict[str, float]:

        # -----------------------------------------------------
        # Convert user date to training year
        # -----------------------------------------------------

        training_date = normalize_to_training_year(
            flight["FL_DATE"]
        )

        # Create temporary request only.
        # We are NOT concatenating with historical data.
        request = pd.Series(
            {
                "FL_DATE": training_date,
                "CRS_DEP_TIME": flight["CRS_DEP_TIME"],
            }
        )

        reference_time = _scheduled_departure(
            pd.DataFrame([request])
        ).iloc[0]

        timestamp = reference_time.value

        # -----------------------------------------------------
        # Keys
        # -----------------------------------------------------

        carrier = str(
            flight["OP_UNIQUE_CARRIER"]
        ).strip().upper()

        origin = str(
            flight["ORIGIN"]
        ).strip().upper()

        destination = str(
            flight["DEST"]
        ).strip().upper()

        tail_num = str(
            flight["TAIL_NUM"]
        ).strip().upper()

        route = (
            f"{origin}_{destination}"
        )

        carrier_origin = (
            f"{carrier}_{origin}"
        )

        # -----------------------------------------------------
        # Historical features
        # -----------------------------------------------------

        return {

            # Carrier
            "carrier_historical_delay":
                self._mean_before(
                    self.carrier,
                    carrier,
                    timestamp,
                ),

            "carrier_recent_delay_7":
                self._recent_mean(
                    self.carrier,
                    carrier,
                    timestamp,
                    7,
                ),

            "carrier_recent_delay_30":
                self._recent_mean(
                    self.carrier,
                    carrier,
                    timestamp,
                    30,
                ),

            # Route
            "route_historical_delay":
                self._mean_before(
                    self.route,
                    route,
                    timestamp,
                ),

            "route_recent_delay_7":
                self._recent_mean(
                    self.route,
                    route,
                    timestamp,
                    7,
                ),

            "route_recent_delay_30":
                self._recent_mean(
                    self.route,
                    route,
                    timestamp,
                    30,
                ),

            # Origin
            "origin_historical_delay":
                self._mean_before(
                    self.origin,
                    origin,
                    timestamp,
                ),

            "origin_recent_delay_7":
                self._recent_mean(
                    self.origin,
                    origin,
                    timestamp,
                    7,
                ),

            "origin_recent_delay_30":
                self._recent_mean(
                    self.origin,
                    origin,
                    timestamp,
                    30,
                ),

            # Carrier + Origin
            "carrier_origin_historical_delay":
                self._mean_before(
                    self.carrier_origin,
                    carrier_origin,
                    timestamp,
                ),

            # Aircraft
            "aircraft_previous_delay":
                self._previous(
                    self.tail_num,
                    tail_num,
                    timestamp,
                ),
        }

    # =========================================================
    # HISTORICAL MEAN
    # =========================================================

    @staticmethod
    def _mean_before(
        lookup: dict[str, _EntityHistory],
        key: str,
        timestamp: int,
    ) -> float:

        entity = lookup.get(key)

        if entity is None:
            return 0.0

        # Only observations BEFORE the requested flight
        end = np.searchsorted(
            entity.timestamps,
            timestamp,
            side="left",
        )

        if end == 0:
            return 0.0

        return float(
            entity.delays[:end].mean()
        )

    # =========================================================
    # RECENT MEAN
    # =========================================================

    @staticmethod
    def _recent_mean(
        lookup: dict[str, _EntityHistory],
        key: str,
        timestamp: int,
        days: int,
    ) -> float:

        entity = lookup.get(key)

        if entity is None:
            return 0.0

        window = (
            pd.Timedelta(days=days).value
        )

        start_timestamp = (
            timestamp - window
        )

        start = np.searchsorted(
            entity.timestamps,
            start_timestamp,
            side="left",
        )

        end = np.searchsorted(
            entity.timestamps,
            timestamp,
            side="left",
        )

        if start >= end:
            return 0.0

        return float(
            entity.delays[start:end].mean()
        )

    # =========================================================
    # PREVIOUS AIRCRAFT DELAY
    # =========================================================

    @staticmethod
    def _previous(
        lookup: dict[str, _EntityHistory],
        key: str,
        timestamp: int,
    ) -> float:

        entity = lookup.get(key)

        if entity is None:
            return 0.0

        end = np.searchsorted(
            entity.timestamps,
            timestamp,
            side="left",
        )

        if end == 0:
            return 0.0

        return float(
            entity.delays[end - 1]
        )


# =============================================================
# NORMALIZE USER DATE TO TRAINING YEAR
# =============================================================

def normalize_to_training_year(
    value: date | pd.Timestamp,
) -> pd.Timestamp:

    parsed = pd.Timestamp(value)

    training_year = Config.DATA.TRAINING_YEAR

    try:

        return parsed.replace(
            year=training_year
        )

    except ValueError:

        # Handle February 29
        if (
            parsed.month == 2
            and parsed.day == 29
        ):

            return parsed.replace(
                year=training_year,
                day=28,
            )

        raise ValueError(
            f"Date {parsed.date()} cannot be represented "
            f"in training year {training_year}."
        )


# =============================================================
# CREATE SCHEDULED DEPARTURE
# =============================================================

def _scheduled_departure(
    frame: pd.DataFrame,
) -> pd.Series:

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------

    dates = pd.to_datetime(
        frame["FL_DATE"],
        errors="coerce",
        format="mixed",
    )

    if dates.isna().any():

        raise ValueError(
            "Historical data contains invalid FL_DATE values."
        )

    # ---------------------------------------------------------
    # HHMM
    # ---------------------------------------------------------

    departure = pd.to_numeric(
        frame["CRS_DEP_TIME"],
        errors="coerce",
    )

    if departure.isna().any():

        raise ValueError(
            "Historical data contains invalid "
            "CRS_DEP_TIME values."
        )

    departure = departure.astype("int64")

    hours = departure // 100
    minutes = departure % 100

    invalid = (
        (hours < 0)
        | (hours > 23)
        | (minutes < 0)
        | (minutes > 59)
    )

    if invalid.any():

        raise ValueError(
            "Historical data contains invalid HHMM values."
        )

    # ---------------------------------------------------------
    # datetime
    # ---------------------------------------------------------

    return (
        dates
        + pd.to_timedelta(
            hours,
            unit="h",
        )
        + pd.to_timedelta(
            minutes,
            unit="m",
        )
    )


# =============================================================
# BUILD ONE LOOKUP DICTIONARY
# =============================================================

def _build_lookup(
    frame: pd.DataFrame,
    key_column: str,
) -> dict[str, _EntityHistory]:

    lookup: dict[str, _EntityHistory] = {}

    for key, group in frame.groupby(
        key_column,
        sort=False,
    ):

        ordered = group.sort_values(
            "timestamp"
        )

        lookup[str(key)] = _EntityHistory(

            timestamps=ordered[
                "timestamp"
            ].to_numpy(
                dtype="int64"
            ),

            delays=ordered[
                "delay"
            ].to_numpy(
                dtype="float64"
            ),
        )

    return lookup