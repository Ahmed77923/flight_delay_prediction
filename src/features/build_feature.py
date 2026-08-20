import numpy as np
import pandas as pd

from src.data.clean_data import clean_data
from src.data.load import load_data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the flight delay prediction model.

    Features:
    - Date features
    - Departure / arrival time features
    - Cyclical time features
    - Distance features
    - Route features
    - Peak departure feature
    - Carrier historical delay
    """

    df = df.copy()

    # ========================================================
    # DATE FEATURES
    # ========================================================

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])

    df["year"] = df["FL_DATE"].dt.year
    df["month"] = df["FL_DATE"].dt.month
    df["quarter"] = df["FL_DATE"].dt.quarter
    df["day"] = df["FL_DATE"].dt.day
    df["day_of_week"] = df["FL_DATE"].dt.dayofweek

    df["week_of_year"] = (
        df["FL_DATE"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # ========================================================
    # DEPARTURE TIME
    # ========================================================

    dep_time = pd.to_numeric(
        df["CRS_DEP_TIME"],
        errors="coerce"
    )

    df["departure_hour"] = dep_time // 100
    df["departure_minute"] = dep_time % 100

    # ========================================================
    # ARRIVAL TIME
    # ========================================================

    arr_time = pd.to_numeric(
        df["CRS_ARR_TIME"],
        errors="coerce"
    )

    df["arrival_hour"] = arr_time // 100
    df["arrival_minute"] = arr_time % 100

    # ========================================================
    # CYCLICAL FEATURES
    # ========================================================

    df["departure_hour_sin"] = np.sin(
        2 * np.pi * df["departure_hour"] / 24
    )

    df["departure_hour_cos"] = np.cos(
        2 * np.pi * df["departure_hour"] / 24
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # ========================================================
    # DISTANCE
    # ========================================================

    df["distance_log"] = np.log1p(
        df["DISTANCE"]
    )

    # ========================================================
    # TIME IN MINUTES
    # ========================================================

    df["departure_time_minutes"] = (
        df["departure_hour"] * 60
        + df["departure_minute"]
    )

    df["arrival_time_minutes"] = (
        df["arrival_hour"] * 60
        + df["arrival_minute"]
    )

    # ========================================================
    # ROUTE
    # ========================================================

    df["route"] = (
        df["ORIGIN"].astype(str)
        + "_"
        + df["DEST"].astype(str)
    )

    # ========================================================
    # PEAK DEPARTURE
    # ========================================================

    df["is_peak_departure"] = (
        df["departure_hour"].between(7, 9)
        |
        df["departure_hour"].between(16, 19)
    ).astype(int)

    # ========================================================
    # SCHEDULED DEPARTURE
    # ========================================================

    df["scheduled_departure"] = (
        df["FL_DATE"]
        + pd.to_timedelta(
            df["departure_hour"],
            unit="h"
        )
        + pd.to_timedelta(
            df["departure_minute"],
            unit="m"
        )
    )

    # ========================================================
    # CARRIER HISTORICAL DELAY
    # ========================================================

    # Sort chronologically
    df = df.sort_values(
        "scheduled_departure"
    ).reset_index(drop=True)

    # Previous cumulative ARR_DELAY
    cumulative_delay = (
        df.groupby("OP_UNIQUE_CARRIER")["ARR_DELAY"]
        .transform("cumsum")
        - df["ARR_DELAY"]
    )

    # Number of previous flights
    previous_flights = (
        df.groupby("OP_UNIQUE_CARRIER")
        .cumcount()
    )

    # Historical average delay
    df["carrier_historical_delay"] = (
        cumulative_delay / previous_flights
    )

    # Replace invalid values
    df["carrier_historical_delay"] = (
        df["carrier_historical_delay"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )
    # ========================================================
    # ROLLING HISTORICAL DELAY
    # ========================================================

    # --------------------------------------------------------
    # Carrier recent delay
    # --------------------------------------------------------

    df["carrier_recent_delay_7"] = (
        df.groupby("OP_UNIQUE_CARRIER")["ARR_DELAY"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=7,
                    min_periods=1
                )
                .mean()
            )
        )
    )

    df["carrier_recent_delay_30"] = (
        df.groupby("OP_UNIQUE_CARRIER")["ARR_DELAY"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=30,
                    min_periods=1
                )
                .mean()
            )
        )
    )
    
    # ========================================================
    # ROUTE RECENT DELAY
    # ========================================================

    df["route_recent_delay_7"] = (
        df.groupby("route")["ARR_DELAY"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=7,
                    min_periods=1
                )
                .mean()
            )
        )
    )

    df["route_recent_delay_30"] = (
        df.groupby("route")["ARR_DELAY"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=30,
                    min_periods=1
                )
                .mean()
            )
        )
    )
    # Previous cumulative delay for each route
    route_cumulative_delay = (
        df.groupby("route")["ARR_DELAY"]
        .transform("cumsum")
        - df["ARR_DELAY"]
    )

    # Number of previous flights for each route
    route_previous_flights = (
        df.groupby("route")
        .cumcount()
    )

    # Historical average delay for the route
    df["route_historical_delay"] = (
        route_cumulative_delay
        / route_previous_flights
    )

    # First flight of a route has no history
    df["route_historical_delay"] = (
        df["route_historical_delay"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )
    # ========================================================
    # ORIGIN HISTORICAL DELAY
    # ========================================================

    # Previous cumulative delay for each origin airport
    origin_cumulative_delay = (
        df.groupby("ORIGIN")["ARR_DELAY"]
        .transform("cumsum")
        - df["ARR_DELAY"]
    )

    # Number of previous flights from each origin
    origin_previous_flights = (
        df.groupby("ORIGIN")
        .cumcount())

    # Historical average delay at the origin airport
    df["origin_historical_delay"] = (
        origin_cumulative_delay
        / origin_previous_flights
    )

    # First flight from an origin has no historical information
    df["origin_historical_delay"] = (
        df["origin_historical_delay"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )
    df["departure_period"] = pd.cut(
    df["departure_hour"],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=[
        "night",
        "morning",
        "afternoon",
        "evening",
        "late_evening"
    ])
    # ========================================================
    # HISTORICAL / RECENT ORIGIN DELAY
    # ========================================================

    # Make sure data is chronologically ordered
    df = (
        df
        .sort_values("scheduled_departure")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Previous delay for the same origin
    # --------------------------------------------------------

    origin_delay = (
        df.groupby("ORIGIN")["ARR_DELAY"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Recent 7-flight average delay
    # --------------------------------------------------------

    df["origin_recent_delay_7"] = (
        origin_delay
        .groupby(df["ORIGIN"])
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Recent 30-flight average delay
    # --------------------------------------------------------

    df["origin_recent_delay_30"] = (
        origin_delay
        .groupby(df["ORIGIN"])
        .transform(
            lambda x: x.rolling(
                window=30,
                min_periods=1
            ).mean()
        )
    )
    
        
    # ========================================================
    # CARRIER + ORIGIN HISTORICAL DELAY
    # ========================================================

    # Create carrier-origin group
    df["carrier_origin"] = (
        df["OP_UNIQUE_CARRIER"].astype(str)
        + "_"
        + df["ORIGIN"].astype(str)
    )

    # Previous cumulative delay
    carrier_origin_cumulative_delay = (
        df.groupby("carrier_origin")["ARR_DELAY"]
        .transform("cumsum")
        - df["ARR_DELAY"]
    )

    # Number of previous flights
    carrier_origin_previous_flights = (
        df.groupby("carrier_origin")
        .cumcount()
    )

    # Historical average delay
    df["carrier_origin_historical_delay"] = (
        carrier_origin_cumulative_delay
        / carrier_origin_previous_flights
    )

    # Replace invalid values
    df["carrier_origin_historical_delay"] = (
        df["carrier_origin_historical_delay"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ========================================================
    # AIRCRAFT PREVIOUS FLIGHT DELAY
    # ========================================================

    # Convert actual arrival time to numeric
    actual_arrival_time = pd.to_numeric(
        df["ARR_TIME"],
        errors="coerce"
    )

    arrival_hour = actual_arrival_time // 100
    arrival_minute = actual_arrival_time % 100

    # Build actual arrival datetime
    df["actual_arrival_datetime"] = (
        df["FL_DATE"]
        + pd.to_timedelta(
            arrival_hour.fillna(0),
            unit="h"
        )
        + pd.to_timedelta(
            arrival_minute.fillna(0),
            unit="m"
        )
    )

    # Sort by aircraft and scheduled departure
    df = (
        df.sort_values(
            ["TAIL_NUM", "scheduled_departure"]
        )
        .reset_index(drop=True)
    )

    # Previous flight's actual arrival
    df["previous_aircraft_arrival"] = (
        df.groupby("TAIL_NUM")["actual_arrival_datetime"]
        .shift(1)
    )

    # Previous flight's arrival delay
    df["aircraft_previous_delay"] = (
        df.groupby("TAIL_NUM")["ARR_DELAY"]
        .shift(1)
    )

    # Prevent temporal leakage
    invalid_previous = (
        df["previous_aircraft_arrival"].isna()
        |
        (
            df["previous_aircraft_arrival"]
            >= df["scheduled_departure"]
        )
    )

    df.loc[
        invalid_previous,
        "aircraft_previous_delay"
    ] = np.nan

    # Cleanup
    df.drop(
        columns=[
            "actual_arrival_datetime",
            "previous_aircraft_arrival",
        ],
        inplace=True
    )

    # Restore chronological order
    df = (
        df.sort_values("scheduled_departure")
        .reset_index(drop=True)
    ) 
    # TAIL_NUM is only used to build aircraft historical features
    df.drop(
    columns=[
        "TAIL_NUM",
        "ARR_TIME",
        "DEP_TIME",
    ],
    inplace=True,
    errors="ignore",)

    return df




if __name__ == "__main__":

    data_directory = "data/"

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data(data_directory)

    print("\nBefore cleaning:")
    print(df.shape)

    # ========================================================
    # CLEAN
    # ========================================================

    df = clean_data(df)

    print("\nAfter cleaning:")
    print(df.shape)

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df = build_features(df)

    print("\nAfter feature engineering:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # ========================================================
    # HISTORICAL FEATURE CHECK
    # ========================================================

    print("\nCarrier historical delay:")
    print(df["carrier_historical_delay"].describe())

    print("\nMissing carrier historical delay:")

    print(
        df["carrier_historical_delay"]
        .isna()
        .mean()
    )

    print("\nFirst rows:")
    print(df.head())