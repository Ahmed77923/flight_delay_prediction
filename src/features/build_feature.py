# build_feature.py
# │
# ├── validate_raw_columns()
# ├── create_scheduled_departure()
# ├── create_date_features()
# ├── create_time_features()
# ├── create_cyclical_features()
# ├── create_categorical_features()
# ├── create_numerical_features()
# ├── create_historical_features()
# └── build_features()

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import Config


TARGET = Config.DATA.TARGET

CATEGORICAL_FEATURES = Config.PREPROCESSING.CATEGORICAL_FEATURES
NUMERICAL_FEATURES = Config.PREPROCESSING.NUMERICAL_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def validate_raw_columns(df: pd.DataFrame) -> None:
    required = {
        "FL_DATE",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "TAIL_NUM",
        TARGET,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def create_scheduled_departure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["FL_DATE"] = pd.to_datetime(
        df["FL_DATE"],
        errors="coerce",
    )
    if df["FL_DATE"].isna().any():
        raise ValueError("FL_DATE contains invalid dates.")

    dep_time = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce")

    if dep_time.isna().any():
        raise ValueError("CRS_DEP_TIME contains invalid values.")

    dep_time = dep_time.astype("int32")

    hours = dep_time // 100
    minutes = dep_time % 100

    invalid = (
        (hours > 23)
        | (minutes > 59)
        | (hours < 0)
        | (minutes < 0)
    )

    if invalid.any():
        raise ValueError("CRS_DEP_TIME contains invalid HHMM values.")

    df["scheduled_departure"] = (
        df["FL_DATE"]
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )

    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["FL_DATE"].dt.year
    df["month"] = df["FL_DATE"].dt.month
    df["quarter"] = df["FL_DATE"].dt.quarter
    df["day"] = df["FL_DATE"].dt.day
    df["day_of_week"] = df["FL_DATE"].dt.dayofweek
    df["week_of_year"] = df["FL_DATE"].dt.isocalendar().week.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dep_time = pd.to_numeric(df["CRS_DEP_TIME"]).astype("int32")
    arr_time = pd.to_numeric(df["CRS_ARR_TIME"]).astype("int32")

    df["departure_hour"] = dep_time // 100
    df["departure_minute"] = dep_time % 100
    df["departure_time_minutes"] = (
        df["departure_hour"] * 60 + df["departure_minute"]
    )

    df["arrival_hour"] = arr_time // 100
    df["arrival_minute"] = arr_time % 100
    df["arrival_time_minutes"] = (
        df["arrival_hour"] * 60 + df["arrival_minute"]
    )

    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["departure_hour_sin"] = np.sin(2 * np.pi * df["departure_hour"] / 24)
    df["departure_hour_cos"] = np.cos(2 * np.pi * df["departure_hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["route"] = df["ORIGIN"].astype(str) + "_" + df["DEST"].astype(str)

    df["carrier_origin"] = (
        df["OP_UNIQUE_CARRIER"].astype(str)
        + "_"
        + df["ORIGIN"].astype(str)
    )

    df["departure_period"] = pd.cut(
        df["departure_hour"],
        bins=[-1, 6, 12, 18, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)

    return df


def create_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if (df["DISTANCE"] < 0).any():
        raise ValueError("DISTANCE contains negative values.")

    df["distance_log"] = np.log1p(df["DISTANCE"])

    df["is_peak_departure"] = (
        df["departure_hour"].between(7, 9)
        | df["departure_hour"].between(16, 19)
    ).astype("int8")

    return df

def create_historical_features(
    df: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Create historical features using only observations
    before the current flight.

    The current flight target is NEVER used.
    """

    df = df.copy()

    # ========================================================
    # NO HISTORY
    # ========================================================

    if history is None or history.empty:

        historical_columns = [
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

        for column in historical_columns:
            df[column] = 0.0

        return df

    # ========================================================
    # PREPARE HISTORY
    # ========================================================

    history = history

    # History must contain the target
    if TARGET not in history.columns:
        raise ValueError(
            f"Historical data must contain target: {TARGET}"
        )

    # ========================================================
    # SORT HISTORY
    # ========================================================

    history = (
        history
        .sort_values("scheduled_departure")
        .reset_index(drop=True)
    )

    def historical_mean(group_column: str) -> pd.Series:
        values = []

        for _, row in df.iterrows():
            eligible = history[
                history["scheduled_departure"]
                < row["scheduled_departure"]
            ]
            values.append(
                eligible.loc[
                    eligible[group_column] == row[group_column],
                    TARGET,
                ].mean()
            )

        return pd.Series(values, index=df.index, dtype="float64")

    def previous_aircraft_delay() -> pd.Series:
        values = []

        for _, row in df.iterrows():
            eligible = history[
                (history["scheduled_departure"]
                 < row["scheduled_departure"])
                & (history["TAIL_NUM"] == row["TAIL_NUM"])
            ]
            values.append(
                eligible.sort_values("scheduled_departure")[TARGET]
                .iloc[-1]
                if not eligible.empty
                else np.nan
            )

        return pd.Series(values, index=df.index, dtype="float64")

    # ========================================================
    # HISTORICAL AGGREGATES
    # ========================================================

    # --------------------------------------------------------
    # Carrier
    # --------------------------------------------------------

    df["carrier_historical_delay"] = historical_mean(
        "OP_UNIQUE_CARRIER"
    )

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    df["route_historical_delay"] = historical_mean("route")

    # --------------------------------------------------------
    # Origin
    # --------------------------------------------------------

    df["origin_historical_delay"] = historical_mean("ORIGIN")

    # --------------------------------------------------------
    # Carrier + Origin
    # --------------------------------------------------------

    df["carrier_origin_historical_delay"] = historical_mean(
        "carrier_origin"
    )

    # ========================================================
    # AIRCRAFT PREVIOUS DELAY
    # ========================================================

    # Get the latest historical flight for each aircraft.

    df["aircraft_previous_delay"] = previous_aircraft_delay()

    # ========================================================
    # RECENT DELAYS
    # ========================================================

    def recent_delay(
        current_df: pd.DataFrame,
        history_df: pd.DataFrame,
        group_column: str,
        days: int,
    ) -> pd.Series:

        values = []

        for _, row in current_df.iterrows():

            cutoff = (
                row["scheduled_departure"]
                - pd.Timedelta(days=days)
            )

            mask = (
                (history_df["scheduled_departure"] >= cutoff)
                &
                (
                    history_df["scheduled_departure"]
                    < row["scheduled_departure"]
                )
                &
                (
                    history_df[group_column]
                    == row[group_column]
                )
            )

            values.append(
                history_df.loc[
                    mask,
                    TARGET
                ].mean()
            )

        return pd.Series(
            values,
            index=current_df.index,
            dtype="float64",
        )

    # ========================================================
    # CARRIER RECENT
    # ========================================================

    df["carrier_recent_delay_7"] = recent_delay(
        df,
        history,
        "OP_UNIQUE_CARRIER",
        7,
    )

    df["carrier_recent_delay_30"] = recent_delay(
        df,
        history,
        "OP_UNIQUE_CARRIER",
        30,
    )

    # ========================================================
    # ROUTE RECENT
    # ========================================================

    df["route_recent_delay_7"] = recent_delay(
        df,
        history,
        "route",
        7,
    )

    df["route_recent_delay_30"] = recent_delay(
        df,
        history,
        "route",
        30,
    )

    # ========================================================
    # ORIGIN RECENT
    # ========================================================

    df["origin_recent_delay_7"] = recent_delay(
        df,
        history,
        "ORIGIN",
        7,
    )

    df["origin_recent_delay_30"] = recent_delay(
        df,
        history,
        "ORIGIN",
        30,
    )

    # ========================================================
    # FILL UNKNOWN HISTORY
    # ========================================================

    historical_columns = [
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

    df[historical_columns] = (
        df[historical_columns]
        .fillna(0.0)
    )

    return df

def build_features(
    df: pd.DataFrame,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all features required by the model."""

    validate_raw_columns(df)

    df = df.copy()
    df = create_scheduled_departure(df)
    df = create_date_features(df)
    df = create_time_features(df)
    df = create_cyclical_features(df)
    df = create_categorical_features(df)
    df = create_numerical_features(df)
    

    if history is not None:
        history = history.copy()
        history = create_scheduled_departure(history)
        history = create_date_features(history)
        history = create_time_features(history)
        history = create_cyclical_features(history)
        history = create_categorical_features(history)
        history = create_numerical_features(history)

    df = create_historical_features(df, history)

    missing = [
        feature for feature in MODEL_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    if df[MODEL_FEATURES].isna().any().any():
        null_columns = df[MODEL_FEATURES].columns[
            df[MODEL_FEATURES].isna().any()
        ].tolist()

        raise ValueError(
            f"Model features contain NaN: {null_columns}"
        )

    return df











if __name__ == "__main__":
    import argparse

    from src.data.clean_data import clean_data
    from src.data.split_data import split_data
    from src.data.load import load_data

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of rows to use for testing",
    )

    args = parser.parse_args()

    df = load_data("data/")
    df = clean_data(df)

    if args.sample_size is not None:
        df = df.head(args.sample_size)

    print(f"Using rows: {len(df):,}")

    train_df, test_df = split_data(df)

    print(f"Train size: {len(train_df):,}")
    print(f"Test size : {len(test_df):,}")

    train_features = build_features(train_df)

    print("Train features:", train_features.shape)

    test_features = build_features(
        test_df,
        history=train_df,
    )

    print("Test features:", test_features.shape)
    print("X_test:", test_features[MODEL_FEATURES].shape)
    
    
    
    
    