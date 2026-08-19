import pandas as pd
# from src.data import clean_data
from src.data.clean_data import clean_data
from src.data.load import load_data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the flight delay prediction model.

    Features created:
    - Date features
    - Departure time features
    - Arrival time features


    Parameters
    ----------
    df : pd.DataFrame
        Cleaned flight data.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features.
    """

    df = df.copy()

    
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    
    df["year"] = df["FL_DATE"].dt.year

    df["month"] = df["FL_DATE"].dt.month

    df["quarter"] = df["FL_DATE"].dt.quarter

    df["day"] = df["FL_DATE"].dt.day

    df["day_of_week"] = df["FL_DATE"].dt.dayofweek

    df["week_of_year"] = (df["FL_DATE"].dt.isocalendar().week.astype(int))

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)


    # Departure Time Features

    df["CRS_DEP_TIME"] = pd.to_datetime(
        df["CRS_DEP_TIME"]
    )

    df["departure_hour"] = (
        df["CRS_DEP_TIME"].dt.hour
    )

    df["departure_minute"] = (
        df["CRS_DEP_TIME"].dt.minute
    )

    # Arrival Time Features
    

    df["CRS_ARR_TIME"] = pd.to_datetime(
        df["CRS_ARR_TIME"]
    )

    df["arrival_hour"] = (
        df["CRS_ARR_TIME"].dt.hour
    )

    df["arrival_minute"] = (
        df["CRS_ARR_TIME"].dt.minute
    )

    

    return df





if __name__ == "__main__":


    data_directory = "data/"

    # Load
    df = load_data(data_directory)

    print("\nBefore cleaning:")
    print(df.shape)

    # Clean
    df = clean_data(df)

    print("\nAfter cleaning:")
    print(df.shape)

    # Feature engineering
    df = build_features(df)

    print("\nAfter feature engineering:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())