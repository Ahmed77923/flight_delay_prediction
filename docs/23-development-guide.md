# Development Guide

## Purpose

This guide is for a developer changing the current project without losing the runtime contract.

## Setup and tests

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

Python 3.11 is the closest match to the container and recorded artifact metadata. The checked-in inference path can be tested without training data because the model artifact is committed.

## Run locally

Start the API:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another terminal:

```bash
API_URL=http://127.0.0.1:8000 streamlit run app/app.py
```

Then use `http://127.0.0.1:8501` for the default Streamlit port and `http://127.0.0.1:8000/docs` for FastAPI. The application startup loads the configured local artifact.

## Run Compose

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f api
```

Use host ports 1041, 1043, 1042, and 1044 for API, Streamlit, Prometheus, and Grafana.

## Change workflow

- Data loading and cleaning: `src/data/load.py`, `src/data/clean_data.py`.
- Split logic: restore or modify the missing `src/data/split_data.py`.
- Features: `src/features/build_feature.py`; keep `config/config.py` feature lists synchronized.
- Preprocessing: `src/preprocessing/preprocess.py`.
- Model and metrics: `src/models/train.py`.
- API contract: `src/api/schemas.py`; endpoints and startup: `src/api/main.py`.
- UI and HTTP error handling: `app/app.py`.
- Deployment and model packaging: `Dockerfile`, `docker-compose.yml`, `.dockerignore`.
- Tests: add focused tests under `tests/` for each changed contract.

## Retrain and replace the artifact

The intended command is:

```bash
python -m src.models.train --sample-size 100000
```

It currently requires the missing split module and `data/*.csv`. After a successful MLflow run, update the model URI for local use, update `MLFLOW_MODEL_ID` in the Dockerfile, update the matching `.dockerignore` exceptions, rebuild the image, and test both `/health` and `/predict`.

## Verify monitoring

```bash
curl http://127.0.0.1:1041/metrics
curl http://127.0.0.1:1042/api/v1/targets
```

Grafana dashboards are manual because provisioning is not implemented. No CI pipeline runs these checks automatically.
