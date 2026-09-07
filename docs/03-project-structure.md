# Project Structure

## Purpose

The table records the important files that exist in the current checkout.

## Implementation

| Path | Purpose | Important details |
|---|---|---|
| `app/app.py` | Streamlit UI and API client | Uses `API_URL`; calls `/health`, `/model`, and `/predict`. |
| `config/config.py` | Shared constants and environment lookup | Defines target, feature lists, MLflow URIs, and API client URL. |
| `src/api/main.py` | FastAPI application | Lifespan model load, endpoints, logging, Prometheus instrumentation. |
| `src/api/model_loader.py` | MLflow pipeline loader | Validates sklearn pipeline, preprocessor, model, and expected columns. |
| `src/api/schemas.py` | Pydantic contracts | `FlightRequest` forbids unknown fields. |
| `src/data/load.py` | CSV loader | Reads sorted `*.csv` files from a directory. |
| `src/data/clean_data.py` | Training cleaning | Removes cancelled/diverted/invalid rows and target outliers. |
| `src/data/` | Data package | `split_data.py` is referenced by code but is absent in this checkout. |
| `src/features/build_feature.py` | Feature construction | Builds calendar, time, cyclical, route, and distance features. |
| `src/preprocessing/preprocess.py` | ColumnTransformer | Constant numerical imputation and sparse one-hot encoding. |
| `src/preprocessing/target_encoder.py` | Custom encoder | Tested separately; not used by the active pipeline. |
| `src/models/train.py` | Training and MLflow logging | Defines LightGBM, metrics, artifact logging, and CLI. |
| `tests/` | Pytest suite | 14 tests cover API, features, preprocessing, and target encoder. |
| `notebook/note1.ipynb` | EDA notebook | 33 code cells; uses a local `../data/month_1.csv` path. |
| `notebook/note2.ipynb` | Notebook placeholder | Empty file. |
| `mlruns/4/models/.../artifacts/` | Served model artifact | Contains `MLmodel`, `model.skops`, and environment metadata. |
| `monitoring/prometheus/prometheus.yml` | Prometheus scrape config | Scrapes `api:8000/metrics` every 15 seconds. |
| `Dockerfile` | Image build | Copies source and one pinned model artifact; defaults to FastAPI. |
| `docker-compose.yml` | Local orchestration | API, Streamlit, Prometheus, Grafana, and `grafana_data` volume. |
| `requirements.txt` | Pinned Python dependencies | Includes runtime, test, MLflow, and metrics packages. |
| `.dockerignore` | Build exclusions | Excludes local data, env, notebooks, and most MLflow files. |
| `.env` | Local environment values | Tracked in this checkout; handle as local configuration. |
| `README.md` | Existing project README | Some sections are stale relative to current Compose and source files. |
| `docs/` | This documentation system | Current-state technical documentation. |

## Configuration

No CI files, shell deployment scripts, Grafana provisioning directory, or `.env.example` file are present.
