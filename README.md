# Flight Delay Prediction

An end-to-end machine learning project for predicting a flight's arrival delay in minutes. The prediction target is `ARR_DELAY`. The project combines flight schedule information, route and carrier attributes, calendar features, cyclical time features, and historical delay statistics.

The current training pipeline uses a LightGBM regression model with median imputation for numerical columns and sparse one-hot encoding for categorical columns. Training runs and metrics are tracked with MLflow.

## Project Goals

- Predict arrival delay as a continuous value in minutes.
- Build useful schedule, route, time, and historical-delay features.
- Avoid using future flight information when creating previous-flight features.
- Evaluate performance with RMSE, MAE, and R2.
- Track experiments and model artifacts locally with MLflow.

## Current Status

Implemented:

- Monthly CSV loading and concatenation
- Data cleaning and target filtering
- Feature engineering
- Chronological train/test splitting
- Numerical median imputation
- Sparse one-hot encoding for categorical variables
- LightGBM regression training
- Feature importance export
- MLflow experiment tracking
- Best-run registration script
- Unit tests for preprocessing and target encoding

Not currently implemented in this checkout:

- FastAPI prediction service
- Docker configuration
- A pinned `requirements.txt`
- A complete persisted preprocessing-plus-model inference pipeline

## Machine Learning Workflow

The main entry point is `src/models/train.py`. The workflow is:

1. Load every CSV file in `data/`.
2. Concatenate the monthly files into one DataFrame.
3. Remove cancelled and diverted flights.
4. Remove rows with invalid `CRS_ELAPSED_TIME` values.
5. Remove rows with missing `ARR_DELAY`.
6. Remove `ARR_DELAY` outliers using the 1.5 IQR rule.
7. Optionally sample a fixed number of cleaned rows.
8. Create date, time, route, cyclical, and historical-delay features.
9. Sort the data chronologically and split the earliest 80% for training and the latest 20% for testing.
10. Fit the preprocessing transformer on training data only.
11. Transform training and test data into the same sparse feature space.
12. Train the LightGBM regressor.
13. Calculate training and test RMSE, MAE, and R2.
14. Save feature importance and model-comparison CSV files.
15. Log parameters, metrics, tags, and the model artifact to MLflow.

The split is chronological, not random. This approximates training on earlier flights and evaluating on later flights.

## Repository Structure

```text
.
├── config/
│   └── config.py                 Shared project configuration
├── data/
│   ├── month_1.csv               Monthly input data
│   ├── month_2.csv
│   └── ...
├── models/                       Saved or exported model files
├── mlruns/                       Local MLflow tracking store
├── notebook/
│   ├── note1.ipynb               Exploratory work
│   └── note2.ipynb
├── src/
│   ├── data/
│   │   ├── load.py               CSV loading
│   │   └── clean_data.py         Data cleaning
│   ├── features/
│   │   └── build_feature.py      Feature engineering
│   ├── models/
│   │   ├── train.py              Training and evaluation
│   │   └── register.py           MLflow model registration
│   └── preprocessing/
│       ├── preprocess.py         Splitting and one-hot preprocessing
│       └── target_encoder.py     Standalone target-encoder implementation
├── tests/
│   └── test_target_encoding.py   Preprocessing and target-encoder tests
├── feature_importance_lightgbm.csv
├── model_comparison_results.csv
└── README.md
```

The `mlruns/` directory can contain many generated run artifacts and may be large.

## Requirements

Use Python 3.9 or newer. The project uses:

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- MLflow
- SciPy
- pytest

XGBoost is present in the historical dependency list, but the current `get_models()` implementation trains LightGBM only.

## Installation

Run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn lightgbm mlflow scipy pytest
```

If PowerShell prevents virtual-environment activation, use the Python executable inside `.venv\Scripts\python.exe` directly or activate the environment using the shell appropriate for your system.

## Input Data

Place monthly CSV files directly inside `data/`. The loader searches for files matching `data/*.csv`, sorts them by filename, reads them with pandas, and concatenates them.

The raw files should contain at least these columns:

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
ARR_TIME
DEP_TIME
TAIL_NUM
```

`ARR_DELAY` is the regression target. `FL_DATE` must be parseable as a date. Scheduled departure and arrival times are expected in values such as `1430`, representing 2:30 PM. Aircraft and actual-flight columns are used to create previous-flight features and are removed afterward when no longer needed.

## Data Cleaning

`src/data/clean_data.py` applies these filters:

- Keep only `CANCELLED == 0`.
- Keep only `DIVERTED == 0`.
- Keep rows where `CRS_ELAPSED_TIME >= 0`.
- Drop rows where `ARR_DELAY` is missing.
- Remove target outliers outside the 1.5 IQR range.

The cleaning function prints the calculated first quartile, third quartile, IQR, and outlier bounds.

## Feature Engineering

Feature construction is implemented in `src/features/build_feature.py`.

### Calendar Features

- `year`
- `month`
- `quarter`
- `day`
- `day_of_week`
- `week_of_year`
- `is_weekend`

### Schedule and Time Features

- `departure_hour`
- `departure_minute`
- `arrival_hour`
- `arrival_minute`
- `departure_time_minutes`
- `arrival_time_minutes`
- `scheduled_departure`
- `departure_period`
- `is_peak_departure`

### Cyclical Time Features

Repeating time values are represented with sine and cosine transformations:

- `departure_hour_sin`
- `departure_hour_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `month_sin`
- `month_cos`

### Route and Distance Features

- `DISTANCE`
- `distance_log`
- `route`, created from `ORIGIN` and `DEST`
- `carrier_origin`, created from carrier and origin

### Historical Delay Features

Historical and recent delay features are calculated using earlier records in chronological order:

- `carrier_historical_delay`
- `carrier_recent_delay_7`
- `carrier_recent_delay_30`
- `route_historical_delay`
- `route_recent_delay_7`
- `route_recent_delay_30`
- `origin_historical_delay`
- `origin_recent_delay_7`
- `origin_recent_delay_30`
- `carrier_origin_historical_delay`
- `aircraft_previous_delay`

The feature builder uses shifted values and previous-flight logic for rolling and aircraft features. The first record in a group naturally receives a missing historical value; numerical median imputation handles these missing values before training.

## Preprocessing

The active trainer uses `build_preprocessor()` from `src/preprocessing/preprocess.py`.

### Numerical Features

The configured numerical features are:

```text
CRS_ELAPSED_TIME
DISTANCE
year
month
day
day_of_week
week_of_year
is_weekend
departure_hour
departure_minute
departure_hour_sin
departure_hour_cos
day_of_week_sin
day_of_week_cos
month_sin
month_cos
arrival_hour
arrival_minute
distance_log
is_peak_departure
carrier_historical_delay
route_historical_delay
origin_historical_delay
origin_recent_delay_7
origin_recent_delay_30
carrier_recent_delay_7
carrier_recent_delay_30
route_recent_delay_7
route_recent_delay_30
carrier_origin_historical_delay
aircraft_previous_delay
```

Numerical values are processed by `SimpleImputer(strategy="median")`.

### Categorical Features

The configured categorical features are:

```text
OP_UNIQUE_CARRIER
ORIGIN
DEST
route
departure_period
carrier_origin
```

They are processed by `OneHotEncoder(handle_unknown="ignore")`. Categories found in the test set that were not present during fitting do not cause a transformation error. The `ColumnTransformer` preserves sparse output, which is suitable for high-cardinality route and airport features.

The custom `TargetEncoder` remains in the repository as a separately tested transformer, but it is not used by the current training pipeline.

## Model

The current model is `lightgbm.LGBMRegressor` with these main settings:

```text
num_leaves=63
max_depth=-1
learning_rate=0.10
n_estimators=1000
min_child_samples=200
reg_alpha=0.0
reg_lambda=1.0
colsample_bytree=1.0
n_jobs=-1
random_state=42
verbosity=-1
```

The model predicts a continuous arrival-delay value in minutes.

## Training

Run the complete training workflow from the repository root:

```powershell
python -m src.models.train
```

To use a smaller sample for a faster experiment:

```powershell
python -m src.models.train --sample-size 1000000
```

The sample size must be greater than zero. If it is smaller than the cleaned dataset, sampling uses random state `42`. The full dataset is used when `--sample-size` is omitted.

The script prints loading, cleaning, feature-engineering, split, preprocessing, validation, training, evaluation, and MLflow information to the terminal.

## Evaluation Metrics

The trainer calculates:

- **RMSE**: Penalizes larger prediction errors more strongly.
- **MAE**: Average absolute prediction error in minutes.
- **R2**: Proportion of target variance explained by the model.

Metrics are calculated separately for training and test data. Test metrics are the primary values for comparing model quality.

## Generated Outputs

Training can create or update:

- `model_comparison_results_one_Hot.csv`: Model metrics sorted by ascending test RMSE.
- `feature_importance_<model_name>.csv`: Feature importance values and percentages.
- `mlruns/`: MLflow experiments, metrics, parameters, tags, and model artifacts.

Feature names include the `ColumnTransformer` prefixes and one-hot category names, for example `onehot__ORIGIN_JFK`.

## MLflow

The trainer uses a local file-backed MLflow store:

```text
Tracking URI: sqlite:///mlflow.db
Experiment: flight_arr_delay_prediction_categorical_features_V3
```

Each training run logs:

- Model name and target name
- Training and test row counts
- Number of transformed features
- Model parameters
- Training time
- Training and test RMSE, MAE, and R2
- Feature-importance artifact
- LightGBM model artifact

Start the MLflow UI with:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Open the address printed by MLflow, normally `http://127.0.0.1:5000`.

## Model Registration

The registration script is `src/models/register.py`:

```powershell
python -m src.models.register
```

It searches a configured MLflow experiment for the finished run with the lowest `test_rmse` and registers the artifact under `flight_arr_delay_best_model`.

Before using this script, verify that its `EXPERIMENT_NAME` matches the experiment created by `src/models/train.py`. The current trainer uses `flight_arr_delay_prediction_categorical_features_V2`, while the registration script contains a different experiment-name constant. This mismatch must be aligned before registration can reliably find the newest training runs.

## Testing

Run all tests from the repository root:

```powershell
python -m pytest
```

The current test file checks:

- The preprocessor contains a one-hot transformer.
- Unknown-category handling is set to `ignore`.
- The custom target encoder produces out-of-fold values that differ from a full-data fit.

For a quick import and syntax check:

```powershell
python -m py_compile src/models/train.py
python -c "from src.models.train import run_training; print('training module import passed')"
```

## Configuration

Shared configuration is defined in `config/config.py`:

- Target: `ARR_DELAY`
- Test size: `0.2`
- Random state: `42`
- Model directory: `models/`
- Numerical feature list
- Categorical feature list
- MLflow defaults
- API host and port defaults

Some executable values are currently defined directly in `src/models/train.py`, including the data directory, experiment name, tracking URI, and target name. When changing these values, update both configuration locations or consolidate them into one configuration source.

## Reproducibility

- Run commands from the repository root.
- Keep input filenames and data contents fixed when comparing runs.
- Keep Python and library versions consistent.
- Use the same random state when sampling data or comparing experiments.
- Fit the preprocessor only on training data.
- Do not commit large generated MLflow artifacts unless they are intentionally part of the project history.

## Known Limitations and Recommended Improvements

1. Persist the fitted preprocessor together with the model so future prediction code uses the exact same one-hot vocabulary and numerical imputation values.
2. Create a single inference function that accepts raw flight data and applies cleaning, feature engineering, preprocessing, and prediction in the correct order.
3. Align the MLflow experiment name in `register.py` with the trainer.
4. Add a pinned dependency file such as `requirements.txt` or `pyproject.toml`.
5. Add end-to-end tests covering raw input through prediction.
6. Add explicit checks for missing input columns and invalid time values.
7. Add a deployment interface, such as FastAPI, only after the inference pipeline is persisted and tested.
8. Review historical feature generation carefully whenever the time split or production prediction horizon changes.

## License

No license file is currently included in the repository. Add a project license before distributing the code publicly.
