from typing import Any, List, Optional

import mlflow
from mlflow.tracking import MlflowClient


# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME: str = "flight_arr_delay_prediction_categorical_features"

REGISTERED_MODEL_NAME: str = (
    "flight_arr_delay_best_model"
)

METRIC: str = "test_rmse"


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    "file:./mlruns"
)

client: MlflowClient = MlflowClient()


# ============================================================
# FIND EXPERIMENT
# ============================================================

experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise ValueError(
        f"Experiment not found: "
        f"{EXPERIMENT_NAME}"
    )


# ============================================================
# GET ALL MODEL RUNS
# ============================================================

runs: List[Any] = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=(
        "attributes.status = 'FINISHED'"
    ),
    order_by=[
        f"metrics.{METRIC} ASC"
    ],
)

if not runs:
    raise ValueError(
        "No finished MLflow runs found."
    )


# ============================================================
# FIND BEST RUN
# ============================================================

best_run: Any = None

for run in runs:

    rmse = run.data.metrics.get(
        METRIC
    )

    if rmse is None:
        continue

    best_run = run
    break


if best_run is None:
    raise ValueError(
        f"No run contains metric: {METRIC}"
    )


# ============================================================
# BEST RUN INFORMATION
# ============================================================

run_id: str = best_run.info.run_id

model_name: str = (
    best_run.data.tags.get(
        "model_type",
        "unknown"
    )
)

rmse: float = (
    best_run.data.metrics[
        "test_rmse"
    ]
)

mae: Optional[float] = (
    best_run.data.metrics.get(
        "test_mae"
    )
)

r2: Optional[float] = (
    best_run.data.metrics.get(
        "test_r2"
    )
)


print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"Model : {model_name}"
)

print(
    f"Run ID: {run_id}"
)

print(
    f"RMSE  : {rmse:.4f}"
)

if mae is not None:
    print(
        f"MAE   : {mae:.4f}"
    )

if r2 is not None:
    print(
        f"R²    : {r2:.4f}"
    )


# ============================================================
# CHECK MODEL ARTIFACT
# ============================================================

model_uri: str = (
    f"runs:/{run_id}/model"
)

print(
    f"\nModel URI:\n{model_uri}"
)


# ============================================================
# REGISTER MODEL
# ============================================================

print("\n" + "=" * 60)
print("REGISTERING BEST MODEL")
print("=" * 60)

registered_model: Any = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)


# ============================================================
# ADD VERSION TAGS
# ============================================================

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="model_type",
    value=model_name,
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="selection_metric",
    value=METRIC,
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="test_rmse",
    value=str(rmse),
)

if mae is not None:

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="test_mae",
        value=str(mae),
    )

if r2 is not None:

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
        key="test_r2",
        value=str(r2),
    )


# ============================================================
# RESULT
# ============================================================

print(
    "\nRegistered model:"
)

print(
    f"Name   : "
    f"{REGISTERED_MODEL_NAME}"
)

print(
    f"Version: "
    f"{registered_model.version}"
)

print(
    f"Model  : "
    f"{model_name}"
)

print(
    f"RMSE   : "
    f"{rmse:.4f}"
)

print(
    "\nRegistration completed."
)