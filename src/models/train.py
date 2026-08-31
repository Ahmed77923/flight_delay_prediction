# Load
#   ↓
# Clean
#   ↓
# Chronological Split
#   ↓
# Build Features
#   ↓
# X / y
#   ↓
# Preprocessing
#   ↓
# X_train_processed
# X_test_processed
#   ↓
# LightGBM
#   ↓
# Evaluation
#   ↓
# MLflow

import time
import warnings
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn import pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from config.config import Config
from src.data.load import load_data
from src.data.clean_data import clean_data
from src.features.build_feature import build_features
from src.preprocessing.preprocess import (
    build_preprocessor,
    preprocess_data

)  
from src.data.split_data import split_data
from mlflow.tracking import MlflowClient



warnings.filterwarnings("ignore")

# ============================================================
# MLFLOW SETUP
# ============================================================
# mlflow.set_tracking_uri("http://127.0.0.1:5000")1
USE_MLFLOW = True 
if USE_MLFLOW:
    mlflow.set_tracking_uri(
        Config.MLFLOW.TRACKING_URI
    ) 
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment(
        Config.MLFLOW.EXPERIMENT_NAME
    )




# ============================================================
# LOAD + PREPROCESS DATA
# ============================================================
def prepare_data(
    sample_size: Optional[int] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Any,
    Any,
    pd.Series,
    pd.Series,
    Any,
]:

    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    df = load_data(Config.DATA.DATA_PATH)

    print(f"\nRaw shape: {df.shape}")

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
    # SAMPLE
    # ========================================================

    if sample_size is not None:

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero."
            )

        if sample_size < len(df):

            df = df.sort_values(
                "FL_DATE"
            ).head(
                sample_size
            ).reset_index(
                drop=True
            )

        print(
            f"Sampled shape: {df.shape}"
        )

    # ========================================================
    # CHRONOLOGICAL SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 3: CHRONOLOGICAL SPLIT")
    print("=" * 70)

    train_df, test_df = split_data(
        df
    )

    print(
        f"\nTrain rows: {len(train_df):,}"
    )

    print(
        f"Test rows : {len(test_df):,}"
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 70)

    # Train features
    train_features = build_features(
        train_df
    )

    print(
        "\nTrain features:",
        train_features.shape
    )

    # Test features
    # IMPORTANT:
    # Train is used as  context.
    test_features = build_features(
        test_df,
        history=None,  # train_features,   # iam not using  features for now
    )

    print(
        "Test features :",
        test_features.shape
    )

    # ========================================================
    # X / Y
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 5: SPLIT FEATURES / TARGET")
    print("=" * 70)

    model_features = (
        Config.PREPROCESSING.CATEGORICAL_FEATURES
        + Config.PREPROCESSING.NUMERICAL_FEATURES
    )

    target = Config.DATA.TARGET

    X_train = train_features[
        model_features
    ].copy()

    y_train = train_features[
        target
    ].copy()

    X_test = test_features[
        model_features
    ].copy()

    y_test = test_features[
        target
    ].copy()

    print(
        "\nX_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "y_test :",
        y_test.shape
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 6: PREPROCESSING")
    print("=" * 70)

    preprocessor = build_preprocessor()

    # TRAIN:
    # fit + transform
    X_train_processed = preprocess_data(
        X_train,
        preprocessor,
        fit=True,
    )

    # TEST:
    # transform only
    X_test_processed = preprocess_data(
        X_test,
        preprocessor,
        fit=False,
    )

    print(
        "\nProcessed train shape:",
        X_train_processed.shape,
    )

    print(
        "Processed test shape :",
        X_test_processed.shape,
    )

    # ========================================================
    # RETURN
    # ========================================================

    return (
        X_train,
        X_test,
        y_train,
        y_test,
     
    )






# ============================================================
# DATA VALIDATION
# ============================================================
def validate_data(
    X_train: Any,
    X_test: Any,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:

    print("\n" + "=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    # ========================================================
    # SHAPE
    # ========================================================

    if X_train.shape[0] != len(y_train):
        raise ValueError(
            "X_train and y_train have different number of rows."
        )

    if X_test.shape[0] != len(y_test):
        raise ValueError(
            "X_test and y_test have different number of rows."
        )

    # Train/Test must have the same number of features
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "X_train and X_test have different number of features."
        )

    print(
        f"Train shape: {X_train.shape}"
    )

    print(
        f"Test shape : {X_test.shape}"
    )

    # ========================================================
    # TARGET NaN
    # ========================================================

    train_target_nan = y_train.isna().sum()
    test_target_nan = y_test.isna().sum()

    print(
        f"\nTrain target NaN: {train_target_nan}"
    )

    print(
        f"Test target NaN : {test_target_nan}"
    )

    if train_target_nan > 0:
        raise ValueError(
            "NaN values found in y_train."
        )

    if test_target_nan > 0:
        raise ValueError(
            "NaN values found in y_test."
        )

    # ========================================================
    # FEATURE NaN
    # ========================================================

    from scipy.sparse import issparse

    if issparse(X_train):

        train_nan = np.isnan(
            X_train.data
        ).sum()

        test_nan = np.isnan(
            X_test.data
        ).sum()

    elif isinstance(X_train, pd.DataFrame):

        train_nan = (
            X_train.isna()
            .sum()
            .sum()
        )

        test_nan = (
            X_test.isna()
            .sum()
            .sum()
        )

    else:

        train_nan = np.isnan(
            X_train
        ).sum()

        test_nan = np.isnan(
            X_test
        ).sum()

    print(
        f"\nTrain feature NaN: {train_nan}"
    )

    print(
        f"Test feature NaN : {test_nan}"
    )

    if train_nan > 0:
        raise ValueError(
            "NaN values found in X_train."
        )

    if test_nan > 0:
        raise ValueError(
            "NaN values found in X_test."
        )

    # ========================================================
    # SPARSE MATRIX INFORMATION
    # ========================================================

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

    # ========================================================
    # VALIDATION PASSED
    # ========================================================

    print(
        "\nData validation passed."
    )



# ============================================================
# MODELS
# ============================================================
def get_models() -> Dict[str, Dict[str, Any]]:

    models = {

        "lightgbm": {

            "model": LGBMRegressor(

                num_leaves=63,
                max_depth=-1,

                learning_rate=0.10,
                n_estimators=1000,

                min_child_samples=200,

                reg_alpha=0.0,
                reg_lambda=1.0,

                colsample_bytree=1.0,

                n_jobs=-1,

                random_state=(
                    Config.MODEL.RANDOM_STATE
                ),

                verbosity=-1,
            ),
        }
    }

    return models
# ============================================================
# METRICS
# ============================================================
def calculate_metrics(y_true: Any,predictions: Any,) -> Dict[str, float]:

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
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    pipeline: Any,
) -> pd.DataFrame:

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = (
        model.feature_importances_
    )

    if len(feature_names) != len(importance):

        raise ValueError(
            "Number of feature names does not match "
            "number of model importances."
        )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    })

    total_importance = (
        importance_df["importance"].sum()
    )

    if total_importance > 0:

        importance_df[
            "importance_percent"
        ] = (
            importance_df["importance"]
            / total_importance
            * 100
        )

    else:

        importance_df[
            "importance_percent"
        ] = 0.0

    importance_df = (
        importance_df
        .sort_values(
            "importance_percent",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\nTop 60 Features by Percentage:"
    )

    print(
        importance_df[
            [
                "feature",
                "importance_percent",
            ]
        ]
        .head(60)
        .to_string(index=False)
    )

    return importance_df
# ============================================================
# LOG PARAMETERS
# ============================================================

def log_parameters(model: Any,X_train: Any,X_test: Any,) -> None:

    params = {
        "random_state": Config.MODEL.RANDOM_STATE,
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "num_features": X_train.shape[1],
        "target": Config.DATA.TARGET,
    }

    # --------------------------------------------------------
    # MODEL PARAMETERS
    # --------------------------------------------------------

    model_params = model.get_params()

    for key, value in model_params.items():

        params[str(key)] = str(value)

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.log_params(params)

# ============================================================
# TRAIN ONE MODEL
# ============================================================



def train_model(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
) -> Pipeline:

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model = model_config["model"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("\nTraining pipeline...")

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start_time

    print(
        f"Training completed in "
        f"{training_time:.2f}s"
    )

    print(
        f"Training time: "
        f"{training_time:.2f}s"
    )

    return pipeline
# ============================================================
# TRAINING
# ============================================================

def run_training(
    sample_size: Optional[int] = None,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("FLIGHT DELAY MODEL TRAINING")
    print("=" * 70)

    # ========================================================
    # 1. PREPARE DATA
    # ========================================================

    print("\nPreparing data...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_data(
        sample_size=sample_size
    )

    # ========================================================
    # 2. VALIDATE DATA
    # ========================================================

    validate_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ========================================================
    # 3. GET MODELS
    # ========================================================

    models = get_models()

    # ========================================================
    # 4. TRAIN MODELS
    # ========================================================

    results = []

    for model_name, model_config in models.items():

        print("\n" + "=" * 70)
        print(f"STARTING MODEL: {model_name}")
        print("=" * 70)

        # ----------------------------------------------------
        # Create a NEW preprocessor for each model
        # ----------------------------------------------------

        preprocessor = build_preprocessor()

        # ====================================================
        # MLFLOW
        # ====================================================

        if USE_MLFLOW:

            with mlflow.start_run(
                run_name=model_name
            ):

                # --------------------------------------------
                # LOG PARAMETERS
                # --------------------------------------------

                log_parameters(
                    model=model_config["model"],
                    X_train=X_train,
                    X_test=X_test,
                )

                # --------------------------------------------
                # TRAIN
                # --------------------------------------------

                pipeline = train_model(
                    model_name=model_name,
                    model_config=model_config,
                    X_train=X_train,
                    y_train=y_train,
                    preprocessor=preprocessor,
                )

                # --------------------------------------------
                # PREDICTIONS
                # --------------------------------------------

                train_predictions = (
                    pipeline.predict(X_train)
                )

                test_predictions = (
                    pipeline.predict(X_test)
                )

                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                train_metrics = calculate_metrics(
                    y_train,
                    train_predictions,
                )

                test_metrics = calculate_metrics(
                    y_test,
                    test_predictions,
                )

                # --------------------------------------------
                # LOG METRICS
                # --------------------------------------------

                mlflow.log_metrics({
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "train_r2": train_metrics["r2"],

                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                })

                # --------------------------------------------
                # FEATURE IMPORTANCE
                # --------------------------------------------

                importance_df = get_feature_importance(
                    pipeline
                )

                # --------------------------------------------
                # LOG FEATURE IMPORTANCE
                # --------------------------------------------

                importance_df.to_csv(
                    "feature_importance.csv",
                    index=False,
                )

                mlflow.log_artifact(
                    "feature_importance.csv"
                )

                # --------------------------------------------
                # LOG MODEL
                # --------------------------------------------

                mlflow.sklearn.log_model(
                    pipeline,
                    name="model",
                    skops_trusted_types=[
                        "collections.OrderedDict",
                        "lightgbm.basic.Booster",
                        "lightgbm.sklearn.LGBMRegressor",
                        "numpy.dtype",
                    ],
                )

                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                result = {
                    "model_name": model_name,

                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "train_r2": train_metrics["r2"],

                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                }

        # ====================================================
        # WITHOUT MLFLOW
        # ====================================================

        else:

            trained_model = train_model(
                model_name=model_name,
                model_config=model_config,
                X_train=X_train,
                y_train=y_train,
                preprocessor=preprocessor,
            )

            # --------------------------------------------
            # PREDICTIONS
            # --------------------------------------------

            train_predictions = (
                trained_model.predict(X_train)
            )

            test_predictions = (
                trained_model.predict(X_test)
            )

            # --------------------------------------------
            # METRICS
            # --------------------------------------------

            train_metrics = calculate_metrics(
                y_train,
                train_predictions,
            )

            test_metrics = calculate_metrics(
                y_test,
                test_predictions,
            )

            # --------------------------------------------
            # FEATURE IMPORTANCE
            # --------------------------------------------

            importance_df = get_feature_importance(
                trained_model
            )

            # --------------------------------------------
            # RESULT
            # --------------------------------------------

            result = {
                "model_name": model_name,

                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_r2": train_metrics["r2"],

                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
            }

        results.append(result)

    # ========================================================
    # 5. MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df = (
        results_df
        .sort_values(
            by="test_rmse",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model_name",
                "train_rmse",
                "train_mae",
                "train_r2",
                "test_rmse",
                "test_mae",
                "test_r2",
            ]
        ].to_string(index=False)
    )


    # ========================================================
    # 7. COMPLETION
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    return results_df
# ============================================================
# Test run
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the LightGBM flight-delay model."
    )
    parser.add_argument(
        "--sample-size",
        "--sample_size",
        type=int,
        default=None,
        help="Optional number of cleaned rows to use, for example 1000000.",
    )
    args = parser.parse_args()

    run_training(
        sample_size=args.sample_size
    )