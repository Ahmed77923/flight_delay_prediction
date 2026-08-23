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
    split_data,
)  
from mlflow.tracking import MlflowClient


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================






# ============================================================
# MLFLOW SETUP
# ============================================================
# mlflow.set_tracking_uri("http://127.0.0.1:5000")1
mlflow.set_tracking_uri(
    Config.MLFLOW.TRACKING_URI
)

mlflow.set_experiment(
    Config.MLFLOW.EXPERIMENT_NAME
)


def get_feature_importance(
    model: Any,
    preprocessor: Any,
) -> pd.DataFrame:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    ).reset_index(drop=True)
    importance_df["importance_percent"] = (
        importance_df["importance"]
        / importance_df["importance"].sum()
        * 100
    )

    importance_df = importance_df.sort_values(
        "importance_percent",
        ascending=False
    )

    print("\nTop 60 Features by Percentage:")
    print(
        importance_df[
            ["feature", "importance_percent"]
        ]
        .head(60)
        .to_string(index=False)
    )

    return importance_df

# ============================================================
# LOAD + PREPROCESS DATA
# ============================================================
def prepare_data(
    sample_size: Optional[int] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Any,
]:

    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    df = load_data(
        Config.DATA.DATA_PATH
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
# ----------------------------------------------------------
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than zero.")

        if sample_size < len(df):
            df = df.sample(
                n=sample_size,
                random_state=Config.MODEL.RANDOM_STATE,
            ).reset_index(drop=True)

            print(
                f"Sampled shape: {df.shape}"
            )

    # ========================================================
# ----------------------------------------------------------
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

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(
        "\nProcessed train shape:",
        X_train_processed.shape,
    )

    print(
        "Processed test shape :",
        X_test_processed.shape,
    )

    return (
        X_train,
        X_test,
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
    X_train: Any,
    X_test: Any,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:

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
    # Feature NaN
    # --------------------------------------------------------

    from scipy.sparse import issparse

    # ========================================================
    # CASE 1: Sparse Matrix
    # ========================================================

    if issparse(X_train):

        train_nan = np.isnan(
            X_train.data
        ).sum()

        test_nan = np.isnan(
            X_test.data
        ).sum()

    # ========================================================
    # CASE 2: Pandas DataFrame
    # ========================================================

    elif isinstance(X_train, pd.DataFrame):

        # Pandas handles:
        # float
        # int
        # category
        # object

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

    # ========================================================
    # CASE 3: NumPy Array
    # ========================================================

    else:

        train_nan = np.isnan(
            X_train
        ).sum()

        test_nan = np.isnan(
            X_test
        ).sum()

    # --------------------------------------------------------
    # Print NaN information
    # --------------------------------------------------------

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
    # Categorical information
    # --------------------------------------------------------

    if isinstance(X_train, pd.DataFrame):

        categorical_columns = (
            X_train
            .select_dtypes(
                include=["category"]
            )
            .columns
            .tolist()
        )

        print(
            f"\nCategorical columns: "
            f"{len(categorical_columns)}"
        )

        if categorical_columns:

            print(
                "Categorical features:"
            )

            for col in categorical_columns:

                print(
                    f"  - {col}"
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

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    if isinstance(X_train, pd.DataFrame):

        print(
            "\nFeature dtypes:"
        )

        print(
            X_train.dtypes
            .value_counts()
        )

    # --------------------------------------------------------
    # Validation passed
    # --------------------------------------------------------

    print(
        "\nData validation passed."
    )

#  ============================================================
# MODELS
# ============================================================

def get_models() -> Dict[str, Dict[str, Any]]:

    models = {

        "lightgbm": {

            "model": LGBMRegressor(

                # Tree complexity
                num_leaves=63,
                max_depth=-1,

                # Learning
                learning_rate=0.10,
                n_estimators=1000,

                # Regularization
                min_child_samples=200,
                reg_alpha=0.0,
                reg_lambda=1.0,

                # Feature sampling
                colsample_bytree=1.0,

                # Performance
                n_jobs=-1,

                # Reproducibility
                random_state=Config.MODEL.RANDOM_STATE,

                # Disable LightGBM logs
                verbosity=-1,
            ),

            "flavor": mlflow.lightgbm,
        }
    }

    return models

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: Any,
    predictions: Any,
) -> Dict[str, float]:

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
    model: Any,
    X_train: Any,
    X_test: Any,
) -> None:

    mlflow.log_params(
        {
            "random_state": Config.MODEL.RANDOM_STATE,
            "train_rows": X_train.shape[0],
            "test_rows": X_test.shape[0],
            "num_features": X_train.shape[1],
            "target": Config.DATA.TARGET,
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
    model_name: str,
    model_config: Dict[str, Any],
    X_train: Any,
    y_train: pd.Series,
    X_test: Any,
    y_test: pd.Series,
    feature_names: Any,
    preprocessor: Any,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> Dict[str, Any]:

    model = model_config["model"]
    flavor = model_config["flavor"]

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # ====================================================
        # TAGS
        # ====================================================

        mlflow.set_tags(
            {
                "model": model_name,
                "target": Config.DATA.TARGET,
                "train_period": "2025",
                "test_period": "2026-01_to_2026-06",
                "categorical_strategy": "onehot",
            }
        )

        # ====================================================
        # PARAMETERS
        # ====================================================

        log_parameters(
            model,
            X_train,
            X_test,
        )

        # ====================================================
        # TRAIN
        # ====================================================

        print("\nTraining model...")

        start_time = time.time()

        model.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.time() - start_time
        )

        print(
            f"Training completed "
            f"in {training_time:.2f}s"
        )

        mlflow.log_metric(
            "training_time_seconds",
            training_time,
        )

        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        print(
            "\nCalculating feature importance..."
        )

        feature_importance = (
            model.feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": feature_importance,
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        # ====================================================
        # IMPORTANCE PERCENTAGE
        # ====================================================

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

        # ====================================================
        # TOP 50
        # ====================================================

        print("\nTop 50 Features:")

        print(
            importance_df
            .head(50)
            .to_string(index=False)
        )

        # ====================================================
        # SAVE FEATURE IMPORTANCE
        # ====================================================

        importance_path = (
            f"feature_importance_"
            f"{model_name}.csv"
        )

        importance_df.to_csv(
            importance_path,
            index=False,
        )

        mlflow.log_artifact(
            importance_path
        )

        # ====================================================
        # TRAIN PREDICTIONS
        # ====================================================

        print(
            "\nPredicting training data..."
        )

        train_predictions = (
            model.predict(X_train)
        )

        train_metrics = (
            calculate_metrics(
                y_train,
                train_predictions,
            )
        )

        # ====================================================
        # TEST PREDICTIONS
        # ====================================================

        print(
            "\nPredicting test data..."
        )

        test_predictions = (
            model.predict(X_test)
        )

        test_metrics = (
            calculate_metrics(
                y_test,
                test_predictions,
            )
        )

        # Package the ALREADY-FITTED components for reproducible inference.
        # Do not call final_pipeline.fit(): both steps were fitted above.
        final_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline_predictions = final_pipeline.predict(X_test_raw)
        if not np.allclose(
            test_predictions,
            pipeline_predictions,
        ):
            raise ValueError(
                "Pipeline predictions differ from the existing workflow."
            )

        print(
            "Prediction equivalence check: PASSED"
        )

        model_dir = Path("models")
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "flight_delay_model.joblib"
        metadata_path = model_dir / "model_metadata.json"
        feature_names_path = model_dir / "feature_names.json"

        joblib.dump(final_pipeline, model_path)

        feature_names_path.write_text(
            json.dumps(list(feature_names), indent=4),
            encoding="utf-8",
        )

        metadata = {
            "model_name": model_name,
            "model_version": "1.0",
            "training_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "number_of_features": int(len(feature_names)),
            "categorical_features": list(
                Config.PREPROCESSING.CATEGORICAL_FEATURES
            ),
            "numerical_features": list(
                Config.PREPROCESSING.NUMERICAL_FEATURES
            ),
            "metrics": {
                "mae": float(test_metrics["mae"]),
                "rmse": float(test_metrics["rmse"]),
                "r2": float(test_metrics["r2"]),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )

        loaded_pipeline = joblib.load(model_path)
        original_predictions = final_pipeline.predict(
            X_test_raw.iloc[:10]
        )
        loaded_predictions = loaded_pipeline.predict(
            X_test_raw.iloc[:10]
        )

        if not np.allclose(
            original_predictions,
            loaded_predictions,
        ):
            raise ValueError(
                "Reloaded pipeline predictions do not match the original."
            )

        print(
            "\n========================================\n"
            "FINAL MODEL TRAINING\n"
            "========================================\n"
            f"\nModel: {model_name}"
            f"\nTraining samples: {len(y_train):,}"
            f"\nTest samples: {len(y_test):,}"
            f"\nFinal feature count: {len(feature_names):,}"
            f"\n\nMAE: {test_metrics['mae']:.4f}"
            f"\nRMSE: {test_metrics['rmse']:.4f}"
            f"\nR²: {test_metrics['r2']:.4f}"
            "\n\n========================================\n"
            "MODEL SAVED\n"
            "========================================\n"
            f"\nModel: {model_path}"
            f"\nMetadata: {metadata_path}"
            f"\nFeature names: {feature_names_path}"
            "\n\nModel reload verification: PASSED"
        )

        # ====================================================
        # LOG METRICS
        # ====================================================

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

        # ====================================================
        # LOG MODEL
        # ====================================================

        try:
            mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                name="flight_delay_pipeline",
                serialization_format="pickle",
            )
            print(f"\nMLflow pipeline logged successfully in {run.info.run_id}")
            active_run = mlflow.active_run()
          
            run_id = mlflow.active_run().info.run_id

            client = MlflowClient()

            artifacts = client.list_artifacts(run_id)

            print("\n=== MLFLOW ARTIFACTS ===")

            for artifact in artifacts:
                print(
                    artifact.path,
                    artifact.is_dir,
                    artifact.file_size,
                )

            print("\n=== RUN INFO ===")
            run = client.get_run(run_id)

            print("Run ID:", run.info.run_id)
            print("Artifact URI:", run.info.artifact_uri)
            print("Tracking URI:", mlflow.get_tracking_uri())
            if active_run:
                print("Run ID:", active_run.info.run_id)
                print("Artifact URI:", active_run.info.artifact_uri)
            print("MLflow Tracking URI:", mlflow.get_tracking_uri())
        except (FileNotFoundError, OSError, mlflow.exceptions.MlflowException) as exc:
            print(
                f"\nMLflow pipeline logging skipped: {exc}"
            )

        # ====================================================
        # RESULTS
        # ====================================================

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
            "\nTraining Metrics"
        )

        print(
            "-" * 30
        )

        print(
            f"MAE  : "
            f"{train_metrics['mae']:.4f}"
        )

        print(
            f"RMSE : "
            f"{train_metrics['rmse']:.4f}"
        )

        print(
            f"R²   : "
            f"{train_metrics['r2']:.4f}"
        )

        print(
            f"\nRun ID: "
            f"{run.info.run_id}"
        )

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "model_name": model_name,
            "run_id": run.info.run_id,

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
        X_train_raw,
        X_test_raw,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = prepare_data(
        sample_size=sample_size
    )

    # ========================================================
    # 2. ONE-HOT PREPROCESSING
    # ========================================================

    print("\n" + "=" * 70)
    print("ONE-HOT PREPROCESSING")
    print("=" * 70)

    feature_names = preprocessor.get_feature_names_out()

    # ========================================================
    # 3. VALIDATE DATA
    # ========================================================

    validate_data(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ========================================================
    # 4. MODELS
    # ========================================================

    models = get_models()

    # ========================================================
    # 5. TRAIN MODELS
    # ========================================================

    results = []

    for (
        model_name,
        model_config,
    ) in models.items():

        print("\n" + "=" * 70)
        print(
            f"STARTING MODEL: {model_name}"
        )
        print("=" * 70)

        result = train_model(
            model_name=model_name,
            model_config=model_config,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            preprocessor=preprocessor,
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
        )

        results.append(result)

    # ========================================================
    # 6. MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by="test_rmse",
            ascending=True,
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
    # 8. SAVE RESULTS
    # ========================================================

    results_path = (
        "model_comparison_results_one_Hot.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        "\nComparison saved to:"
    )

    print(
        results_path
    )

    # ========================================================
    # 9. COMPLETION
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    return results_df

# ============================================================
# ENTRY POINT
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