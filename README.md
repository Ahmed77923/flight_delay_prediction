# Flight Delay Prediction

## Project Overview

Flight Delay Prediction is a machine-learning system that estimates a flight's arrival delay in minutes using information available before departure. The project covers data loading and cleaning, feature engineering, model training and evaluation, MLflow artifact tracking, model loading, a FastAPI prediction service, and a Streamlit user interface.

This implementation is a supervised **regression** problem, not a delayed/not-delayed classification problem. The target is the continuous `ARR_DELAY` value. The API returns a numeric prediction in minutes and does not currently return a probability or confidence score.

```text
Historical flight data
  -> Data cleaning
  -> Feature engineering
  -> Chronological train/test split
  -> Preprocessing
  -> LightGBM regression model
  -> Evaluation and MLflow logging
  -> Loaded model pipeline
  -> FastAPI prediction API
  -> Streamlit interface
```

## Main Objective

The project is intended to provide an end-to-end flight-delay prediction workflow that can:

- process historical monthly flight CSV files;
- prepare date, time, route, carrier, and distance features;
- train and evaluate a LightGBM regression model;
- track training runs and model artifacts with MLflow;
- load a selected local model artifact for inference;
- expose predictions through FastAPI; and
- provide a browser-based Streamlit client.

Monitoring services, authentication, model registry promotion workflows, and future-date mapping are not implemented in the current checkout.

The model uses information available for one flight request: scheduled date and times, carrier, airports, scheduled duration, and distance. It does not query historical flights or use previous-flight delay statistics at inference time.

The current model uses information available for one flight request: scheduled date and times, carrier, airports, scheduled duration, and distance. It does not query historical flights or use previous-flight delay statistics at inference time.

## Quick Start

From the repository root on Windows PowerShell:

```powershell

python -m pip install -r requirements.txt
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, activate the same environment and start the UI:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/app.py
```

Open `http://127.0.0.1:8501` for Streamlit or `http://127.0.0.1:8000/docs` for FastAPI. The API loads the committed model artifact at startup.

For the containerized application:

```powershell
docker compose up --build
```

Then open `http://127.0.0.1:1043` for Streamlit and `http://127.0.0.1:1041/docs` for FastAPI.

## Contents

- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Requirements and Installation](#requirements-and-installation)
- [Configuration](#configuration)
- [Running Locally](#running-locally)
- [Machine-Learning Pipeline](#machine-learning-pipeline)
- [API](#api)
- [Streamlit Application](#streamlit-application)
- [Docker](#docker)
- [MLflow](#mlflow)
- [Testing](#testing)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)
- [Limitations and Security](#limitations-and-security)
- [Repository State](#repository-state)

## Architecture

### Training flow

```text
data/month_*.csv
    -> load_data()
    -> clean_data()
    -> chronological 80/20 split
    -> build_features()
    -> zero-imputation + sparse one-hot encoding
    -> LightGBM regression pipeline
    -> RMSE, MAE, and R2 evaluation
    -> MLflow run, feature-importance artifact, and model artifact
```

### Inference flow

```text
FlightRequest JSON
    -> FastAPI POST /predict
    -> pandas DataFrame
    -> build_features()
    -> select fitted preprocessor input columns
    -> loaded MLflow sklearn Pipeline
    -> PredictionResponse
```

Streamlit is a thin HTTP client. It sends requests to FastAPI and does not load the model or perform feature engineering itself.

There is no Prometheus, Grafana, Kubernetes, Helm, Terraform, CI/CD, database application layer, or authentication implementation in this repository. MLflow is used for tracking and model artifacts; the API serves the local artifact directly.

## Repository Structure

```text
.
├── app/
│   └── app.py                         Streamlit client application
├── config/
│   └── config.py                      Shared configuration classes
├── data/
│   ├── month_1.csv ... month_12.csv   Input monthly CSV data
├── mlruns/
│   └── 4/                             Local MLflow experiment and model artifact
├── notebook/
│   ├── note1.ipynb                    Notebook analysis
│   └── note2.ipynb                    Notebook analysis
├── src/
│   ├── api/
│   │   ├── main.py                    FastAPI application and endpoints
│   │   ├── model_loader.py             MLflow pipeline loading and state
│   │   └── schemas.py                  Pydantic request/response schemas
│   ├── data/
│   │   ├── load.py                     CSV loading and concatenation
│   │   ├── clean_data.py               Training-data filtering
│   │   └── split_data.py               Chronological train/test split
│   ├── features/
│   │   └── build_feature.py            Feature engineering
│   ├── models/
│   │   └── train.py                    Model training and MLflow logging
│   └── preprocessing/
│       ├── preprocess.py               Imputation and one-hot preprocessing
│       └── target_encoder.py            Separately tested custom encoder
├── tests/
│   ├── test_api.py                     FastAPI endpoint tests
│   ├── test_build_features.py           Feature engineering tests
│   └── test_target_encoding.py          Target-encoder tests
├── .dockerignore                       Docker build exclusions
├── .env                                Local environment values; do not commit secrets
├── .gitignore                          Git exclusions
├── docker-compose.yml                  API and Streamlit services
├── Dockerfile                          Python image and API startup
├── mlflow.db                           Local MLflow backend database
├── requirements.txt                    Pinned Python dependencies
└── README.md                           Project documentation
```

`models/` is referenced by configuration but is not present in this checkout. Generated CSV files and ordinary local model files are ignored by Git. The committed `mlruns/` content includes the model used by the API.

## Requirements and Installation

The Dockerfile uses `python:3.11-slim`; the committed model metadata was created with Python `3.11.9`. Use Python 3.11 for the closest match.

The project has no Node.js, GPU/CUDA, database-server, or operating-system dependency for local Python execution. Docker installs Linux `libgomp1`, required by LightGBM in the container.

Install the exact pinned dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins Streamlit, pandas, NumPy, scikit-learn, LightGBM, SciPy, MLflow, pytest, FastAPI, Uvicorn, python-dotenv, httpx, requests, and skops.

## Configuration

Configuration is loaded by `config/config.py` with `python-dotenv`.

| Variable              | Required | Default in code                                                | Purpose                                                       |
| --------------------- | -------: | -------------------------------------------------------------- | ------------------------------------------------------------- |
| `MLFLOW_TRACKING_URI` |       No | `http://127.0.0.1:5000`                                        | MLflow tracking URI used by training and model loading        |
| `MLFLOW_MODEL_URI`    |       No | `mlruns/4/models/m-6d479b8fd10a4744862b3b6ec29260d8/artifacts` | Model artifact loaded by the API                              |
| `API_HOST`            |       No | `0.0.0.0` in Docker configuration                              | API bind host                                                 |
| `API_PORT`            |       No | `8000`                                                         | API listening port                                            |
| `API_URL`             |       No | `http://127.0.0.1:8000`                                        | URL used by Streamlit to reach FastAPI                        |
| `TRAINING_YEAR`       |       No | `2025`                                                         | Defined in configuration; not used by the active trainer flow |
| `LOG_LEVEL`           |       No | Not consumed by application code                               | Present in local `.env`, but no code reads it                 |

The local `.env` contains machine-specific values and is not reproduced here. There is no `.env.example` file. A safe template is:

```text
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_MODEL_URI=mlruns/4/models/m-6d479b8fd10a4744862b3b6ec29260d8/artifacts
API_HOST=127.0.0.1
API_PORT=8000
API_URL=http://127.0.0.1:8000
```

## Running Locally

### Start FastAPI

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

The application loads and validates the MLflow pipeline during startup. A missing or incompatible model prevents startup.

### Start Streamlit

With FastAPI running in another terminal:

```powershell
streamlit run app/app.py
```

To point the UI at another API:

```powershell
$env:API_URL = "http://127.0.0.1:9000"
streamlit run app/app.py
```

### Train

Training reads all `*.csv` files under `data/`, cleans them, performs a chronological split, builds features, trains LightGBM, evaluates it, and logs the run and model to MLflow:

```powershell
python -m src.models.train
```

For a smaller experiment:

```powershell
python -m src.models.train --sample-size 100000
```

The alias `--sample_size` is also accepted. A positive sample size is required. The active trainer defines one model, `lightgbm`, and always enables MLflow tracking.

### Inspect the model loader

```powershell
python -m src.api.model_loader
```

## Machine-Learning Pipeline

### Dataset and cleaning

`src.data.load.load_data()` reads and concatenates sorted CSV files matching `data/*.csv`. The checked-in dataset contains `month_1.csv` through `month_12.csv`.

Training cleaning requires `CANCELLED`, `DIVERTED`, `CRS_ELAPSED_TIME`, and `ARR_DELAY`. It keeps non-cancelled, non-diverted flights with positive scheduled duration, drops missing targets, and removes `ARR_DELAY` values outside the 1.5 IQR bounds.

Feature construction additionally requires:

```text
FL_DATE, CRS_DEP_TIME, CRS_ARR_TIME, CRS_ELAPSED_TIME,
DISTANCE, OP_UNIQUE_CARRIER, ORIGIN, DEST
```

`FL_DATE` must be parseable as a date. Scheduled times are integer `HHMM` values, such as `800` or `1430`.

### Split and features

`split_data()` sorts by `FL_DATE` and assigns the earliest 80% to training and the latest 20% to testing. The preprocessor is fitted on training features and only transformed on test features.

`build_features()` creates calendar features (`year`, `month`, `quarter`, `day`, `day_of_week`, `week_of_year`, `is_weekend`), schedule/time features, cyclical sine/cosine features, route and carrier-origin categories, `distance_log`, and `is_peak_departure`.

The optional `history` parameter is accepted for compatibility but no historical statistics are calculated. The API calls `build_features()` with a single request.

### Preprocessing and model

Numerical columns use `SimpleImputer(strategy="constant", fill_value=0)`. Categorical columns use `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`. The resulting `ColumnTransformer` preserves sparse output.

The active model is `lightgbm.LGBMRegressor` with:

```text
num_leaves=63, max_depth=-1, learning_rate=0.10, n_estimators=1000,
min_child_samples=200, reg_alpha=0.0, reg_lambda=1.0,
colsample_bytree=1.0, n_jobs=-1, random_state=42, verbosity=-1
```

The fitted preprocessor and model are stored together in an sklearn `Pipeline`, then logged to MLflow using skops serialization. The checked-in artifact is model ID `m-6d479b8fd10a4744862b3b6ec29260d8`, from run `5407198a78d545f4a3d1ded962e0ed07`.

Training calculates RMSE, MAE, and R2 for train and test predictions and logs feature importance as `feature_importance.csv` inside the MLflow run.

## API

The entry point is `src.api.main:app`. FastAPI loads the model once during application startup.

| Method | Endpoint   | Description                                 |
| ------ | ---------- | ------------------------------------------- |
| `GET`  | `/`        | Service name, version, and endpoint list    |
| `GET`  | `/health`  | Model-loaded health status                  |
| `GET`  | `/model`   | Loaded pipeline and expected input metadata |
| `POST` | `/predict` | Predict arrival delay                       |
| `GET`  | `/docs`    | Interactive FastAPI documentation           |

### Request

`POST /predict` accepts exactly these JSON fields. Unknown fields, including `ARR_DELAY`, are rejected with HTTP 422.

```json
{
  "FL_DATE": "2026-08-22",
  "CRS_DEP_TIME": 800,
  "CRS_ARR_TIME": 1100,
  "CRS_ELAPSED_TIME": 180,
  "DISTANCE": 2475,
  "OP_UNIQUE_CARRIER": "AA",
  "ORIGIN": "JFK",
  "DEST": "LAX"
}
```

`FL_DATE` is parsed by Pydantic as a datetime. Times must be valid `HHMM` values and distance must be non-negative.

Example request:

```powershell
$body = @'
{
  "FL_DATE": "2026-08-22",
  "CRS_DEP_TIME": 800,
  "CRS_ARR_TIME": 1100,
  "CRS_ELAPSED_TIME": 180,
  "DISTANCE": 2475,
  "OP_UNIQUE_CARRIER": "AA",
  "ORIGIN": "JFK",
  "DEST": "LAX"
}
'@
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $body
```

Successful response:

```json
{
  "prediction": 12.45,
  "target": "ARR_DELAY",
  "model": "lightgbm",
  "source": "mlflow"
}
```

Typical status codes are 200 for success, 422 for schema validation, 400 for feature validation errors, 500 for inference or feature-contract failures, and 503 when the model is not loaded.

## Streamlit Application

`app/app.py` collects carrier, origin, destination, date, scheduled departure and arrival times, distance, and scheduled duration. It sends those values to `POST /predict` and displays the predicted delay. The sidebar calls `/health`, and the model-information expander calls `/model`; both checks are cached for 15 seconds.

## Docker

The `Dockerfile` uses `python:3.11-slim`, installs `libgomp1` and `requirements.txt`, copies `config`, `src`, `app`, and the pinned model artifact to `/app/model_artifact`, and starts FastAPI by default. It exposes container ports 8000 and 8501.

Build and run the API directly:

```powershell
docker build -t flight-delay-prediction .
docker run --rm -p 8000:8000 flight-delay-prediction
```

Compose starts:

| Service     | Container port | Host port | Behavior                            |
| ----------- | -------------: | --------: | ----------------------------------- |
| `api`       |           8000 |      1041 | FastAPI; healthchecked at `/health` |
| `streamlit` |           1040 |      1043 | Streamlit; waits for a healthy API  |

```powershell
docker compose up --build
docker compose down
```

Compose mounts `./mlruns` read-only and points the API at `/app/mlruns/.../artifacts`; the direct image default points at `/app/model_artifact`. Keep the artifact path, Dockerfile model ID, and `.dockerignore` exceptions synchronized when changing the served model.

## MLflow

The local `.env` sets `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`; the code default is `http://127.0.0.1:5000`. The configured experiment name is `flight_arr_delay_champion_model`.

The checked-in model is under:

```text
mlruns/4/models/m-6d479b8fd10a4744862b3b6ec29260d8/artifacts/
├── MLmodel
├── conda.yaml
├── model.skops
├── python_env.yaml
└── requirements.txt
```

Its metadata identifies MLflow `3.15.1`, scikit-learn `1.9.0`, Python `3.11.9`, and skops serialization. `src.api.model_loader` uses `mlflow.sklearn.load_model()`, validates the `predict()` method and `preprocessor`/`model` pipeline steps, and records the fitted preprocessor's expected columns.

No MLflow server startup script is included. The API does not require a running MLflow server when `MLFLOW_MODEL_URI` points to the local artifact.

## Testing

Run the suite with the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests cover API health, prediction, validation, model metadata, rejection of `ARR_DELAY`, absence of historical feature names, feature engineering, preprocessing behavior, and the custom target encoder. No lint, formatter, type checker, or CI command is configured.

## Development Guide

- Change loading or cleaning in `src/data/load.py` and `src/data/clean_data.py`.
- Change the chronological split in `src/data/split_data.py`.
- Add derived features in `src/features/build_feature.py` and synchronize feature lists in `config/config.py`.
- Change imputation or category handling in `src/preprocessing/preprocess.py`.
- Change model parameters, evaluation, or MLflow logging in `src/models/train.py`.
- Change request/response contracts in `src/api/schemas.py` and endpoints in `src/api/main.py`.
- Change the Streamlit client in `app/app.py`.
- Add tests under `tests/`.
- Change ports, healthchecks, or service dependencies in `docker-compose.yml`; change image contents in `Dockerfile`.

When changing the model, retrain and log a new artifact, update `MLFLOW_MODEL_URI`, and update the Dockerfile model ID plus matching `.dockerignore` exceptions. Test both local and Compose startup because their artifact paths differ.

## Troubleshooting

| Problem                                          | Likely cause                                     | Solution                                                                         |
| ------------------------------------------------ | ------------------------------------------------ | -------------------------------------------------------------------------------- |
| `No module named pytest` or another dependency   | System Python is being used                      | Activate `.venv` or use `.\.venv\Scripts\python.exe -m ...`                      |
| API fails during startup with a model-load error | Model URI is missing, incorrect, or incompatible | Confirm the artifact exists and install `requirements.txt`                       |
| Streamlit says the API is unavailable            | FastAPI is stopped or `API_URL` is wrong         | Start FastAPI first and set `API_URL` to its reachable base URL                  |
| Docker healthcheck fails                         | Model artifact path or service startup problem   | Run `docker compose logs api`; verify model ID and `.dockerignore` exceptions    |
| Prediction returns 422                           | Missing/wrong/unknown JSON field                 | Match the `FlightRequest` example; do not send `ARR_DELAY`                       |
| Prediction returns 400                           | Invalid date, time, distance, or duration        | Use a parseable date, valid `HHMM`, non-negative distance, and positive duration |
| Training cannot load data                        | No CSV files under `data/`                       | Place monthly input files directly in `data/`                                    |
| Port conflict                                    | Another process uses 8000, 8501, 1041, or 1043   | Stop it or choose another port and update `API_URL`                              |

## Limitations and Security

- Model quality depends on the available monthly data and is not guaranteed for future schedules or unseen operating conditions.
- The API has no authentication, authorization, rate limiting, or TLS. Do not expose it to an untrusted network without adding those controls.
- The Streamlit selector lists are static common carrier and airport codes; the API accepts arbitrary strings and ignores unknown one-hot categories.
- Local MLflow state, `.env`, and data should be treated as development assets. Never commit credentials or private data.
- No license file is present.

## Repository State

The documented checkout is on branch `main`. The current `HEAD` is:

```text
daf919c (HEAD -> main, origin/main) Update the mlflow
```

Recent commits cover MLflow and EDA updates, Docker deployment, API fixes, feature updates, and API/Dockerfile additions. This README describes the current checkout, not older revisions.

Basic Git commands:

```powershell
git status
git add README.md
git commit -m "Document project setup and architecture"
git push
```

No branching strategy is specified by the repository.
