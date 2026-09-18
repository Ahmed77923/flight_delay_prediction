import pandas as pd


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    df = df.sort_values("FL_DATE").reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df
