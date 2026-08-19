import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from config.config import Config


def build_preprocessor():
    """
    Build the baseline preprocessing pipeline.

    Numerical:
        Median imputation
        ↓
        StandardScaler

    Categorical:
        Most-frequent imputation
        ↓
        OneHotEncoder

    IMPORTANT:
        TargetEncoder is NOT used here.

        Target Encoding is tested separately because
        it requires the target variable and OOF encoding.
    """

    # ========================================================
    # FEATURES
    # ========================================================

    numerical_features = (
        Config.PREPROCESSING.NUMERICAL_FEATURES
    )

    categorical_features = (
        Config.PREPROCESSING.CATEGORICAL_FEATURES
    )
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
    # CATEGORICAL PIPELINE
    # ========================================================

    categorical_pipeline = Pipeline(
        steps=[
  
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    # ========================================================
    # COLUMN TRANSFORMER
    # ========================================================

    transformers = [
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train=None
):
    """
    Fit preprocessing only on training data
    and transform both train and test data.

    This function is for the baseline One-Hot
    preprocessing.

    Target Encoding is NOT performed here.
    """

    preprocessor = build_preprocessor()

    # ========================================================
    # FIT ON TRAINING DATA ONLY
    # ========================================================

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # ========================================================
    # TRANSFORM TEST DATA
    # ========================================================

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    return (
        X_train_processed,
        X_test_processed,
        preprocessor
    )

from sklearn.model_selection import train_test_split

def split_data(df):

    target = Config.DATA.TARGET

    df = df.dropna(
        subset=[target]
    ).copy()

    X = df.drop(
        columns=[
            target,
            "FL_DATE"
        ]
    )

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

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