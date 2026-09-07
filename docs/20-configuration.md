# Configuration

## Purpose

This is the reference for environment values and code-defined settings that affect runtime behavior.

## Environment variables

| Variable | Purpose | Current default or value | Required |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking URI used by training/loading setup | Code: `http://127.0.0.1:5000`; local `.env`: `sqlite:///mlflow.db` | No |
| `MLFLOW_MODEL_URI` | Model artifact loaded by API | Local relative artifact path; Compose/Docker: `/app/model_artifact` | No, but a valid model is required at startup |
| `API_HOST` | Uvicorn bind host in Docker command | `0.0.0.0` in Docker | No |
| `API_PORT` | Uvicorn port in Docker command | `8000` | No |
| `API_URL` | Streamlit base URL for FastAPI | `http://127.0.0.1:8000`; Compose: `http://api:8000` | No |
| `TRAINING_YEAR` | Defined data setting | `2025`; not used by active trainer flow | No |
| `LOG_LEVEL` | Present in `.env` | `INFO`; not read by application code | No |

The repository has no `.env.example`. Do not copy private values into documentation. The current `.env` contains no API key or password, but it is tracked and should be reviewed as configuration hygiene.

## Code-defined settings

- Target: `ARR_DELAY`.
- Model name: `flight-arr-delay`.
- Experiment: `flight_arr_delay_champion_model`.
- Test size constant: `0.2`; actual split behavior cannot be verified because `split_data.py` is missing.
- Random state: `42`.
- Streamlit API timeout: 10 seconds.
- Streamlit health/model cache TTL: 15 seconds.

## Configuration mismatch

`API_HOST` and `API_PORT` are used by the Dockerfile's shell command, while `Config.API.HOST` and `Config.API.PORT` are hard-coded constants. This works for the current Compose values but means changing those environment variables does not update every code path.
