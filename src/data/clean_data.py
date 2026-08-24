import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean flight data before feature engineering.

    Steps:
    - Validate required columns
    - Remove cancelled flights
    - Remove diverted flights
    - Remove invalid scheduled duration
    - Remove rows with missing target
    - Remove extreme ARR_DELAY outliers using IQR
    """

    required_columns = {
        "CANCELLED",
        "DIVERTED",
        "CRS_ELAPSED_TIME",
        "ARR_DELAY",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.copy()

    # Remove cancelled flights
    df = df[df["CANCELLED"] == 0]

    # Remove diverted flights
    df = df[df["DIVERTED"] == 0]

    # Remove invalid scheduled duration
    df = df[df["CRS_ELAPSED_TIME"] > 0]

    # Target is required for supervised learning
    df = df.dropna(subset=["ARR_DELAY"])

    # Remove extreme target outliers
    q1 = df["ARR_DELAY"].quantile(0.25)
    q3 = df["ARR_DELAY"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df = df[
        df["ARR_DELAY"].between(
            lower_bound,
            upper_bound,
        )
    ]

    return df.reset_index(drop=True)



if __name__ == "__main__":
    from src.data.load import load_data

    df = load_data("data/")

    print("Before cleaning:", df.shape)

    df = clean_data(df)

    print("After cleaning:", df.shape)
    print("Missing ARR_DELAY:", df["ARR_DELAY"].isna().sum())
    print("Negative duration:",
          (df["CRS_ELAPSED_TIME"] < 0).sum())