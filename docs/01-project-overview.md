# Project Overview

## Purpose

Flight Delay Prediction estimates a flight's arrival delay in minutes. It is a supervised regression system whose target is `ARR_DELAY`; it does not predict a delay class or probability.

## Architecture / Concept

```mermaid
flowchart LR
    Data[Historical CSV files] --> Train[Training code]
    Train --> MLflow[MLflow run and model artifact]
    MLflow --> Image[Docker image]
    Image --> API[FastAPI]
    User[User] --> UI[Streamlit]
    UI -->|HTTP| API
    API --> Features[Feature engineering]
    Features --> Model[LightGBM pipeline]
    Model --> Prediction[Arrival delay in minutes]
    API --> Metrics[/metrics]
    Metrics --> Prom[Prometheus]
    Prom --> Grafana[Grafana]
```

## Implementation

The deployed inference path accepts date, scheduled times, scheduled duration, distance, carrier, origin, and destination. FastAPI builds the features and calls the loaded sklearn pipeline. The response contains a numeric prediction plus `target`, `model`, and `source` metadata.

Major components:

| Component | Role |
|---|---|
| `src/models/train.py` | Loads, cleans, splits, trains, evaluates, and logs a LightGBM pipeline. |
| `mlruns/.../model.skops` | Committed serialized model artifact used for inference. |
| `src/api/` | FastAPI app, request/response schemas, and model loader. |
| `app/app.py` | Streamlit client that calls the API over HTTP. |
| Prometheus | Scrapes FastAPI instrumentation at `/metrics`. |
| Grafana | Compose service with persistent storage only; dashboards are not provisioned. |

## Current status

Inference and Compose configuration are present. The repository does not currently include training data or the imported `src.data.split_data` module, so the training workflow needs those missing inputs before it can be reproduced. No CI/CD, authentication, TLS, model registry promotion, or automated Grafana provisioning is implemented.

## Technologies

Python 3.11 is the container base. The application uses FastAPI, Uvicorn, pandas, scikit-learn, LightGBM, MLflow, skops, Streamlit, Prometheus FastAPI Instrumentator, Prometheus, Grafana, Docker, and Docker Compose. MLflow tracks training metadata and loads the packaged pipeline; it is not a running Compose service.
