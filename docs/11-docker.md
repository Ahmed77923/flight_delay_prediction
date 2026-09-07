# Docker

## Purpose

The Dockerfile builds one Python image containing the API, Streamlit code, dependencies, and the selected model artifact.

## Implementation

| Instruction | Current behavior |
|---|---|
| `FROM` | `python:3.11-slim`. |
| `RUN` | Installs `libgomp1`, the OpenMP runtime required by LightGBM's compiled extension. |
| `WORKDIR` | `/app`. |
| `ENV` | Unbuffered Python, no bytecode, no pip cache; API/model defaults. |
| `COPY` | Dependencies, `config`, `src`, `app`, and one model artifact directory. |
| `ARG` | `MLFLOW_MODEL_ID`, default `m-6d479b8fd10a4744862b3b6ec29260d8`. |
| `EXPOSE` | `8000 8501` (documentation metadata only; Compose uses Streamlit 1040). |
| `CMD` | Starts Uvicorn using `API_HOST` and `API_PORT`. |

Build and run the API:

```bash
docker build -t flight-delay-prediction .
docker run --rm -p 8000:8000 flight-delay-prediction
```

The Dockerfile does not train a model. It copies `mlruns/4/models/<MLFLOW_MODEL_ID>/artifacts` to `/app/model_artifact`, and sets `MLFLOW_MODEL_URI=/app/model_artifact`.

## Configuration

When replacing the model, update the build argument, ensure the matching artifact is available in the build context, and update the `.dockerignore` exceptions. The model artifact is part of the image; Compose does not mount `mlruns`.

## Troubleshooting

A startup model-load failure usually means the artifact ID/path is wrong, the artifact was not included by `.dockerignore`, or dependency versions are incompatible. Check container logs and verify `/app/model_artifact` exists in the built image.
