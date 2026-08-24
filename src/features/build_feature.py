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

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"],format="%m/%d/%Y %I:%M:%S %p",errors="coerce",)
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

    df["arrival_hour"] = arr_time // 100
    df["arrival_minute"] = arr_time % 100

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
    Create historical features without using the current flight target.

    history contains observations before the current dataframe.
    """

    df = df.copy()

    if history is not None and not history.empty:
        combined = pd.concat([history, df], ignore_index=True)
    else:
        combined = df.copy()

    combined = combined.sort_values("scheduled_departure").reset_index(drop=True)

    combined["carrier_historical_delay"] = (
        combined.groupby("OP_UNIQUE_CARRIER")[TARGET]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    combined["route_historical_delay"] = (
        combined.groupby("route")[TARGET]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    combined["origin_historical_delay"] = (
        combined.groupby("ORIGIN")[TARGET]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    combined["carrier_origin_historical_delay"] = (
        combined.groupby("carrier_origin")[TARGET]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    combined["aircraft_previous_delay"] = (
        combined.groupby("TAIL_NUM")[TARGET]
        .shift(1)
    )

    for group, prefix in [
        ("OP_UNIQUE_CARRIER", "carrier"),
        ("route", "route"),
        ("ORIGIN", "origin"),
    ]:
        grouped = combined.set_index("scheduled_departure").groupby(group)[TARGET]

        combined[f"{prefix}_recent_delay_7"] = (
            grouped.rolling("7D", closed="left").mean()
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

        combined[f"{prefix}_recent_delay_30"] = (
            grouped.rolling("30D", closed="left").mean()
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

    if history is not None and not history.empty:
        history_size = len(history)
        df = combined.iloc[history_size:].copy()
    else:
        df = combined.copy()

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

    df[historical_columns] = df[historical_columns].fillna(0.0)

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


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


if __name__ == "__main__":
    from src.data.clean_data import clean_data
    from src.data.split_data import split_data
    from src.data.load import load_data
    
        
    df = load_data("data/")
    df = clean_data(df)

    train_df, test_df = split_data(df)

    train_features = build_features(train_df)

    print("Train features:", train_features.shape)
    
    test_features = build_features(
    test_df,
    history=train_df,
)

    print("Test features:", test_features.shape)
    print("X_test:", test_features[MODEL_FEATURES].shape)