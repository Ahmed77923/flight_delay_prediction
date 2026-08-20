from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from config.config import Config
from src.preprocessing.target_encoder import TargetEncoder

def build_preprocessor():
    """
    Build preprocessing for LightGBM native categorical features.

    Numerical:
        Median imputation

    Categorical:
        Passed through unchanged.
        They will be converted to Pandas 'category'
        before training LightGBM.

    IMPORTANT:
        No OneHotEncoder is used.
        LightGBM handles categorical features natively.
    """

    # ========================================================
    # FEATURES
    # ========================================================

    numerical_features = Config.PREPROCESSING.NUMERICAL_FEATURES
    categorical_features = Config.PREPROCESSING.CATEGORICAL_FEATURES

    # ========================================================
    # NUMERICAL PIPELINE
    # ========================================================

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    # ========================================================
    # COLUMN TRANSFORMER
    # ========================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "target_encoding",
                TargetEncoder(
                    columns=categorical_features,
                    random_state=Config.MODEL.RANDOM_STATE,
                ),
                categorical_features,
            ),
        ],
        remainder="drop"
    )

    return preprocessor



def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Preprocessing for LightGBM native categorical features.

    Numerical:
        Median imputation

    Categorical:
        Convert to pandas category dtype.

    IMPORTANT:
        No OneHotEncoder.
        No ColumnTransformer.

        LightGBM receives categorical columns directly.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    numerical_features = (
        Config.PREPROCESSING.NUMERICAL_FEATURES
    )

    categorical_features = (
        Config.PREPROCESSING.CATEGORICAL_FEATURES
    )

    # ========================================================
    # NUMERICAL FEATURES
    # ========================================================

    numerical_imputer = SimpleImputer(
        strategy="median"
    )

    X_train[numerical_features] = (
        numerical_imputer.fit_transform(
            X_train[numerical_features]
        )
    )

    X_test[numerical_features] = (
        numerical_imputer.transform(
            X_test[numerical_features]
        )
    )

    # ========================================================
    # CATEGORICAL FEATURES
    # ========================================================

    for col in categorical_features:

        # Combine train + test categories so both datasets
        # use the same category definition.
        categories = pd.concat(
            [
                X_train[col],
                X_test[col]
            ]
        ).astype("category").cat.categories

        X_train[col] = pd.Categorical(
            X_train[col],
            categories=categories
        )

        X_test[col] = pd.Categorical(
            X_test[col],
            categories=categories
        )

    return (
        X_train,
        X_test,
        numerical_imputer,
    )


def split_data(df):

    target = Config.DATA.TARGET

    df = df.dropna(
        subset=[target]
    ).copy()

    if "FL_DATE" not in df.columns:
        raise ValueError("FL_DATE is required for chronological splitting.")

    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    if df["FL_DATE"].isna().any():
        raise ValueError("FL_DATE contains values that cannot be parsed as dates.")

    df = df.sort_values("FL_DATE").reset_index(drop=True)

    X = df.drop(
        columns=[
            target,
            "FL_DATE"
        ]
    )

    y = df[target]

    split_index = int(len(df) * (1 - Config.MODEL.TEST_SIZE))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("The dataset is too small for the configured test split.")

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    print("\nTrain shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )



if __name__ == "__main__":

    from src.data.load import load_data
    from src.data.clean_data import clean_data
    from src.features.build_feature import build_features

    data_dir = "data/"

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data(
        data_dir
    )

    # ========================================================
    # CLEAN
    # ========================================================

    df = clean_data(
        df
    )

    # ========================================================
    # FEATURES
    # ========================================================

    df = build_features(
        df
    )

    # ========================================================
    # SPLIT
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        df.drop(columns=[Config.DATA.TARGET]),
        df[Config.DATA.TARGET],
        test_size=0.2,
        random_state=42
    )
    print("\nX_train shape:",X_train.shape)


    print("X_test shape:",X_test.shape)

    print(
        "\ny_train shape:",
        y_train.shape
    )

    print(
        "y_test shape:",
        y_test.shape
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    (
        X_train_processed,
        X_test_processed,
        preprocessor
    ) = preprocess_data(
        X_train,
        X_test
    )

    print(
        "\nProcessed X_train shape:",
        X_train_processed.shape
    )

    print(
        "Processed X_test shape:",
        X_test_processed.shape
    )