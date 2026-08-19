import time
import warnings

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.data.load import load_data
from src.data.clean_data import clean_data
from src.features.build_feature import build_features

from src.preprocessing.preprocess import (
    split_data,
    preprocess_data,
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/"

EXPERIMENT_NAME = (
    "flight_arr_delay_prediction_v2"
)

TRACKING_URI = "file:./mlruns"

TARGET = "ARR_DELAY"


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


# ============================================================
# LOAD + PREPROCESS DATA
# ============================================================

def prepare_data():

    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    df = load_data(
        DATA_PATH
    )

    print(
        f"\nRaw shape: {df.shape}"
    )

    # ========================================================
    # CLEAN
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 2: CLEAN DATA")
    print("=" * 70)

    df = clean_data(
        df
    )

    print(
        f"Cleaned shape: {df.shape}"
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)

    df = build_features(
        df
    )

    print(
        f"Feature shape: {df.shape}"
    )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 4: CHRONOLOGICAL SPLIT")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        df
    )

    # ========================================================
    # REMOVE MISSING TARGET
    # ========================================================

    train_mask = (
        ~y_train.isna()
    )

    test_mask = (
        ~y_test.isna()
    )

    X_train = X_train.loc[
        train_mask
    ]

    y_train = y_train.loc[
        train_mask
    ]

    X_test = X_test.loc[
        test_mask
    ]

    y_test = y_test.loc[
        test_mask
    ]

    print(
        f"\nTrain rows: {len(X_train):,}"
    )

    print(
        f"Test rows : {len(X_test):,}"
    )

    # ========================================================
    # YOUR PREPROCESSING
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 5: PREPROCESSING")
    print("=" * 70)

    (
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = preprocess_data(
        X_train,
        X_test,
    )

    print(
        "\nProcessed train shape:",
        X_train_processed.shape,
    )

    print(
        "Processed test shape :",
        X_test_processed.shape,
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


# ============================================================
# DATA VALIDATION
# ============================================================
def validate_data(
    X_train,
    X_test,
    y_train,
    y_test,
):

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    if X_train.shape[0] != len(y_train):

        raise ValueError(
            "X_train and y_train "
            "have different number of rows."
        )

    if X_test.shape[0] != len(y_test):

        raise ValueError(
            "X_test and y_test "
            "have different number of rows."
        )

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Test shape : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Target NaN
    # --------------------------------------------------------

    train_target_nan = (
        y_train.isna().sum()
    )

    test_target_nan = (
        y_test.isna().sum()
    )

    print(
        f"\nTrain target NaN: "
        f"{train_target_nan}"
    )

    print(
        f"Test target NaN : "
        f"{test_target_nan}"
    )

    if train_target_nan > 0:

        raise ValueError(
            "NaN values found in y_train."
        )

    if test_target_nan > 0:

        raise ValueError(
            "NaN values found in y_test."
        )

    # --------------------------------------------------------
    # Processed features
    # --------------------------------------------------------

    from scipy.sparse import issparse

    if issparse(X_train):

        train_nan = np.isnan(
            X_train.data
        ).sum()

    else:

        train_nan = np.isnan(
            X_train
        ).sum()

    if issparse(X_test):

        test_nan = np.isnan(
            X_test.data
        ).sum()

    else:

        test_nan = np.isnan(
            X_test
        ).sum()

    print(
        f"\nTrain feature NaN: "
        f"{train_nan}"
    )

    print(
        f"Test feature NaN : "
        f"{test_nan}"
    )

    if train_nan > 0:

        raise ValueError(
            "NaN values found in X_train."
        )

    if test_nan > 0:

        raise ValueError(
            "NaN values found in X_test."
        )

    # --------------------------------------------------------
    # Sparse information
    # --------------------------------------------------------

    if issparse(X_train):

        print(
            f"\nX_train format: "
            f"{X_train.getformat()}"
        )

        print(
            f"X_train non-zero values: "
            f"{X_train.nnz:,}"
        )

    if issparse(X_test):

        print(
            f"X_test format: "
            f"{X_test.getformat()}"
        )

        print(
            f"X_test non-zero values: "
            f"{X_test.nnz:,}"
        )

    print(
        "\nData validation passed."
    )
# ============================================================
# MODELS
# ============================================================
def get_models():

    configurations = [
        {
            "num_leaves": 7,
            "learning_rate": 0.05,
            "n_estimators": 1000,
        },
        {
            "num_leaves": 15,
            "learning_rate": 0.05,
            "n_estimators": 1000,
        },
        {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 1000,
        },
        {
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 1000,
        },
    ]

    models = {}

    for i, params in enumerate(configurations):

        model_name = (
            f"lightgbm_exp_{i + 1}"
        )

        models[model_name] = {

            "model": LGBMRegressor(
                **params,
                max_depth=-1,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbosity=-1,
            ),

            "flavor": mlflow.lightgbm,
        }

    return models
# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# ============================================================
# LOG PARAMETERS
# ============================================================

def log_parameters(
    model,
    X_train,
    X_test,
):

    mlflow.log_params(
        {
            "random_state": RANDOM_STATE,
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "num_features": X_train.shape[1],
            "target": TARGET,
        }
    )

    model_params = (
        model.get_params()
    )

    clean_params = {}

    for key, value in model_params.items():

        clean_params[
            str(key)
        ] = str(value)

    mlflow.log_params(
        clean_params
    )


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_name,
    model_config,
    X_train,
    y_train,
    X_test,
    y_test,
):

    model = model_config["model"]

    flavor = model_config["flavor"]

    print("\n" + "=" * 70)
    print(
        f"TRAINING: {model_name}"
    )
    print("=" * 70)

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # ----------------------------------------------------
        # TAGS
        # ----------------------------------------------------

        mlflow.set_tags(
            {
                "model": model_name,
                "target": TARGET,
                "train_period": "2025",
                "test_period": "2026-01_to_2026-06",
            }
        )

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        log_parameters(
            model,
            X_train,
            X_test,
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print(
            "\nTraining model..."
        )

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.time()
            - start_time
        )

        print(
            f"Training completed "
            f"in {training_time:.2f}s"
        )

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        # ----------------------------------------------------
        # TRAIN PREDICTIONS
        # ----------------------------------------------------

        train_predictions = (
            model.predict(
                X_train
            )
        )

        train_metrics = (
            calculate_metrics(
                y_train,
                train_predictions,
            )
        )

        # ----------------------------------------------------
        # TEST PREDICTIONS
        # ----------------------------------------------------

        print(
            "\nPredicting test data..."
        )

        test_predictions = (
            model.predict(
                X_test
            )
        )

        test_metrics = (
            calculate_metrics(
                y_test,
                test_predictions,
            )
        )

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        mlflow.log_metrics(
            {
                "train_rmse":
                    train_metrics["rmse"],

                "train_mae":
                    train_metrics["mae"],

                "train_r2":
                    train_metrics["r2"],

                "test_rmse":
                    test_metrics["rmse"],

                "test_mae":
                    test_metrics["mae"],

                "test_r2":
                    test_metrics["r2"],
            }
        )

        # ----------------------------------------------------
        # LOG MODEL
        # ----------------------------------------------------

        flavor.log_model(
            model,
            name="model",
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print(
            "\nModel Evaluation"
        )

        print(
            "-" * 30
        )

        print(
            f"MAE  : "
            f"{test_metrics['mae']:.4f}"
        )

        print(
            f"RMSE : "
            f"{test_metrics['rmse']:.4f}"
        )

        print(
            f"R²   : "
            f"{test_metrics['r2']:.4f}"
        )

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        }


# ============================================================
# MAIN
# ============================================================

def run_training():

    print("\n" + "=" * 70)
    print("FLIGHT DELAY MODEL TRAINING")
    print("=" * 70)

    # ========================================================
    # PREPROCESSING
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = prepare_data()

    # ========================================================
    # VALIDATION
    # ========================================================

    validate_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = get_models()

    # ========================================================
    # TRAIN
    # ========================================================

    results = []

    for (
        model_name,
        model_config,
    ) in models.items():

        result = train_model(
            model_name=model_name,
            model_config=model_config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(
            result
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "test_rmse"
        )
        .reset_index(
            drop=True
        )
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model_name",
                "test_rmse",
                "test_mae",
                "test_r2",
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best = (
        results_df.iloc[0]
    )

    print("\n" + "=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Model : {best['model_name']}"
    )

    print(
        f"RMSE  : {best['test_rmse']:.4f}"
    )

    print(
        f"MAE   : {best['test_mae']:.4f}"
    )

    print(
        f"R²    : {best['test_r2']:.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df.to_csv(
        "model_comparison_results.csv",
        index=False,
    )

    print(
        "\nComparison saved to:"
        "\nmodel_comparison_results.csv"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_training()