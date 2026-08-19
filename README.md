Flight_Delay_Prediction/
│
├── config/
│ └── config.py
│
├── data/
│ ├── raw/
│ │ ├── month_1.csv
│ │ ├── month_2.csv

# Flight Delay Prediction

This project trains regression models to predict a flight's arrival delay in minutes. The target is `ARR_DELAY`, using flight schedule, route, carrier, distance, and calendar/time features from monthly flight records.

The current training workflow compares four LightGBM regressors, records each run in a local MLflow tracking store, and saves the comparison table to `model_comparison_results.csv`.

## Project Status

The repository currently contains the data preparation, feature engineering, model training, MLflow tracking, model registration, and unit-test pieces of the project. A FastAPI service and Docker configuration are not present in the current checkout.

## Pipeline

`src/models/train.py` runs the following workflow:

1. Load and concatenate every CSV file matching `data/*.csv`.
2. Remove cancelled and diverted flights.
3. Remove rows with invalid `CRS_ELAPSED_TIME` values or a missing `ARR_DELAY`.
4. Remove `ARR_DELAY` outliers using the 1.5 IQR rule.
5. Create date, day-of-week, week, weekend, departure-time, and arrival-time features.
6. Split the data with `train_test_split(test_size=0.2, random_state=42)`.
7. Fit preprocessing on the training data only.
8. Train four LightGBM configurations.
9. Evaluate each model with RMSE, MAE, and R2.
10. Log parameters, metrics, and model artifacts to MLflow.
11. Select the model with the lowest test RMSE and write the comparison table.

The split is currently random, despite the historical comments referring to a chronological split. Use a chronological split before treating the reported test metrics as a future-period estimate.

## Repository Layout

```text
.
├── config/
│   └── config.py                 Shared feature, path, model, and MLflow settings
├── data/
│   ├── month_*.csv               Monthly input data used by the trainer
│   └── *_26.csv                  Additional monthly input files
├── models/
│   └── flight_delay_ridge.joblib A previously saved model artifact
├── mlruns/                       Local MLflow file store and run artifacts
├── notebook/
│   ├── note1.ipynb               Exploratory analysis or experiments
│   └── note2.ipynb
├── src/
│   ├── data/
│   │   ├── load.py               CSV loading and concatenation
│   │   └── clean_data.py         Data filtering and target cleanup
│   ├── features/
│   │   └── build_feature.py      Date and time feature creation
│   ├── preprocessing/
│   │   ├── preprocess.py         ColumnTransformer and data splitting
│   │   └── target_encoder.py     Target-encoding implementation
│   └── models/
│       ├── train.py              Main training and evaluation script
│       └── register.py           Registers the best MLflow run
├── tests/
│   └── test_target_encoding.py   Target-encoding tests
├── model_comparison_results.csv  Stored model comparison output
└── README.md
```

## Requirements

Use Python 3.9 or newer. The training script requires at least:

- pandas
- NumPy
- scikit-learn
- LightGBM
- XGBoost
- MLflow
- SciPy
- pytest for the test suite

There is currently no `requirements.txt` in the repository, so install the dependencies in a virtual environment with your preferred package manager. For example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn lightgbm xgboost mlflow scipy pytest
```

If PowerShell prevents activation, run the commands from an activated Python environment or use its Python executable directly.

## Input Data

Place monthly CSV files directly in `data/`. The loader reads all files matching `data/*.csv` and concatenates them in sorted filename order. Each file should contain the columns used by cleaning and feature engineering, including:

```text
ARR_DELAY
CANCELLED
DIVERTED
CRS_ELAPSED_TIME
FL_DATE
CRS_DEP_TIME
CRS_ARR_TIME
OP_UNIQUE_CARRIER
ORIGIN
DEST
DISTANCE
```

The input data is expected to follow the US flight-delay dataset conventions. `FL_DATE` must be parseable as a date. Scheduled departure and arrival time values must be parseable by pandas as datetime values.

## Train Models

Run the trainer from the repository root so its relative paths and local MLflow URI resolve correctly:

```powershell
python -m src.models.train
```

The script prints dataset shapes, validation results, training progress, and metrics for each model. It creates or updates:

- `model_comparison_results.csv`: models sorted by ascending test RMSE.
- `mlruns/`: local MLflow experiments, metrics, parameters, and model artifacts.

The four model configurations vary `num_leaves` across 7, 15, 31, and 63. They use a learning rate of `0.05` and `1,000` estimators, with the remaining parameters defined in `get_models()` in `src/models/train.py`.

## MLflow

Training uses the local file-backed tracking URI `file:./mlruns` and the experiment name `flight_arr_delay_prediction_v2`.

To inspect recorded runs with the MLflow UI:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Then open the URL printed by MLflow, normally `http://127.0.0.1:5000`.

## Register the Best Model

After training, `src/models/register.py` searches MLflow runs by the lowest `test_rmse` and registers the selected artifact as `flight_arr_delay_best_model`:

```powershell
python -m src.models.register
```

Run this from the repository root. Before using the script, verify that its `EXPERIMENT_NAME` matches the experiment created by the trainer. The current trainer uses `flight_arr_delay_prediction_v2`, while the registration script is configured for `flight_arr_delay_prediction`; update that constant if the older experiment is not intended.

## Preprocessing and Features

The baseline preprocessor in `src/preprocessing/preprocess.py` uses:

- Median imputation for numerical features.
- One-hot encoding for `OP_UNIQUE_CARRIER`, `ORIGIN`, and `DEST`.
- `handle_unknown="ignore"` so unseen categorical values do not fail transformation.
- Sparse output for the encoded feature matrix.

The configured numerical features are:

```text
CRS_ELAPSED_TIME, DISTANCE, year, month, quarter, day,
day_of_week, week_of_year, is_weekend, departure_hour,
departure_minute, arrival_hour, arrival_minute
```

`target_encoder.py` contains a separate target-encoding implementation and tests. It is not used by the current baseline training path.

## Testing

Run the tests from the repository root:

```powershell
python -m pytest
```

The tests exercise target-encoding behavior and the expected preprocessing configuration. Tests may require the project root to be on `PYTHONPATH`; running them from the repository root normally provides that import context.

## Configuration

Shared settings are defined in `config/config.py`, including:

- Target column: `ARR_DELAY`
- Model output directory: `models/`
- Random state: `42`
- Test size: `0.2`
- Categorical and numerical feature lists
- MLflow and API-related defaults

The executable training constants in `src/models/train.py` currently take precedence for data path, experiment name, tracking URI, and target name.

## Reproducibility Notes

- Keep the input files and their filenames fixed when comparing runs.
- Run commands from the repository root.
- The trainer fixes the model random state at `42`, but library versions can still affect results.
- The saved comparison CSV is overwritten on each training run.
- MLflow artifacts and generated files can become large; avoid committing new run artifacts unless they are intentionally part of the project history.

## Limitations and Next Steps

- Replace the random split with a date-based split for realistic future-flight evaluation.
- Align the MLflow experiment name in the training and registration scripts.
- Add a pinned dependency file such as `requirements.txt`.
- Persist the fitted preprocessor with the model if predictions will be served outside the training process.
- Add a prediction interface and end-to-end tests before deploying a model.
