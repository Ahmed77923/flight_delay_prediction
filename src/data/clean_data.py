import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean flight data before feature engineering.

    Steps:
    - Remove cancelled flights
    - Remove diverted flights
    - Remove invalid CRS elapsed time
    - Remove rows with missing target
    """

    df = df.copy()

    # Remove cancelled flights
    df = df[df["CANCELLED"] == 0]

    # Remove diverted flights
    df = df[df["DIVERTED"] == 0]

    # Remove invalid scheduled duration
    df = df[df["CRS_ELAPSED_TIME"] >= 0]

    # Target cannot be missing
    df = df.dropna(subset=["ARR_DELAY"])

    # Reset index
    df = df.reset_index(drop=True)
    # IQR
    Q1 = df["ARR_DELAY"].quantile(0.25)
    Q3 = df["ARR_DELAY"].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # upper_bound = 120
    # lower_bound = -30 
    df = df[
        (df["ARR_DELAY"] >= lower_bound) &
        (df["ARR_DELAY"] <= upper_bound)
    ].copy()
    print(f"Q1: {Q1}")
    print(f"Q3: {Q3}")
    print(f"IQR: {IQR}")
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")
    return df



if __name__ == "__main__":
    from load import load_data

    df = load_data("data/")

    print("Before cleaning:", df.shape)

    df = clean_data(df)

    print("After cleaning:", df.shape)
    print("Missing ARR_DELAY:", df["ARR_DELAY"].isna().sum())
    print("Negative duration:",
          (df["CRS_ELAPSED_TIME"] < 0).sum())