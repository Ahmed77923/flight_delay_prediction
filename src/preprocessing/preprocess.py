from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from config.config import Config


def build_preprocessor() -> ColumnTransformer:

    numerical_features = (
        Config.PREPROCESSING.NUMERICAL_FEATURES
    )

    categorical_features = (
        Config.PREPROCESSING.CATEGORICAL_FEATURES
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def preprocess_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit: bool = False,
):
    if fit:
        return preprocessor.fit_transform(df)

    return preprocessor.transform(df)


    
    
    
    
    
    
if __name__ == "__main__":
    import argparse
    import numpy as np  

    from src.data.load import load_data
    from src.data.clean_data import clean_data
    from src.data.split_data import split_data
    from src.features.build_feature import build_features

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of rows to use for testing",
    )

    args = parser.parse_args()

    # ======================================================
    # LOAD
    # ======================================================

    df = load_data("data/")

    # ======================================================
    # SAMPLE
    # ======================================================

    if args.sample_size is not None:
        df = df.head(args.sample_size)

    print(f"Using rows: {len(df):,}")

    # ======================================================
    # CLEAN
    # ======================================================

    df = clean_data(df)

    # ======================================================
    # SPLIT
    # ======================================================

    train_df, test_df = split_data(df)

    print(f"Train size: {len(train_df):,}")
    print(f"Test size : {len(test_df):,}")

    # ======================================================
    # FEATURES
    # ======================================================

    train_features = build_features(
        train_df
    )

    test_features = build_features(
        test_df,
        history=train_df,
    )

    # ======================================================
    # X / Y
    # ======================================================

    model_features = (
        Config.PREPROCESSING.CATEGORICAL_FEATURES
        + Config.PREPROCESSING.NUMERICAL_FEATURES
    )

    X_train = train_features[
        model_features
    ]

    y_train = train_features[
        Config.DATA.TARGET
    ]

    X_test = test_features[
        model_features
    ]

    y_test = test_features[
        Config.DATA.TARGET
    ]

    print("\nX_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # ======================================================
    # PREPROCESSOR
    # ======================================================

    preprocessor = build_preprocessor()

    # Train → fit + transform
    X_train_processed = preprocess_data(
        X_train,
        preprocessor,
        fit=True,
    )

    # Test → transform only
    X_test_processed = preprocess_data(
        X_test,
        preprocessor,
        fit=False,
    )

    # ======================================================
    # RESULTS
    # ======================================================

    print(
        "\nProcessed X_train:",
        X_train_processed.shape,
    )

    print(
        "Processed X_test :",
        X_test_processed.shape,
    )

    print(
        "Processed type   :",
        type(X_train_processed),
    )

    print(
        "Train NaN:",
        np.isnan(X_train_processed.data).sum(),
    )

    print(
        "Test NaN :",
        np.isnan(X_test_processed.data).sum(),
    )