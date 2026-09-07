# API

## Purpose

FastAPI exposes the loaded model as an HTTP service. Entrypoint: `src.api.main:app`.

## Architecture / Concept

The lifespan loads and validates the MLflow pipeline before serving requests. A load failure raises during startup. `/predict` performs request validation, feature engineering, model-column selection, and inference.

## Implementation

| Method | Endpoint | Purpose | Request | Response |
|---|---|---|---|---|
| `GET` | `/` | Service metadata and endpoint list | None | JSON |
| `GET` | `/health` | Model-loaded status | None | JSON |
| `GET` | `/model` | Loaded model metadata | None | JSON; `503` if unavailable |
| `POST` | `/predict` | Predict `ARR_DELAY` | `FlightRequest` JSON | `PredictionResponse` |
| `GET` | `/metrics` | Prometheus instrumentation | None | Prometheus text |
| `GET` | `/docs` | FastAPI Swagger UI | None | HTML |

`FlightRequest` requires exactly these fields: `FL_DATE`, `CRS_DEP_TIME`, `CRS_ARR_TIME`, `CRS_ELAPSED_TIME`, `DISTANCE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, and `DEST`. Extra fields, including `ARR_DELAY`, are rejected.

Example:

```bash
curl http://127.0.0.1:1041/health
curl http://127.0.0.1:1041/model
curl -X POST http://127.0.0.1:1041/predict \
  -H 'Content-Type: application/json' \
  -d '{"FL_DATE":"2026-08-22","CRS_DEP_TIME":800,"CRS_ARR_TIME":1100,"CRS_ELAPSED_TIME":180,"DISTANCE":2475,"OP_UNIQUE_CARRIER":"AA","ORIGIN":"JFK","DEST":"LAX"}'
```

Successful prediction shape:

```json
{"prediction": 12.45, "target": "ARR_DELAY", "model": "lightgbm", "source": "mlflow"}
```

## Status codes and errors

- `200`: successful endpoint response.
- `400`: feature validation failure caught as `ValueError`.
- `422`: Pydantic validation failure, missing field, invalid type, or extra field.
- `500`: feature contract or inference failure.
- `503`: model is not loaded for `/model` or `/predict`.

## Configuration

The container binds to `0.0.0.0:8000`; Compose publishes it as host port `1041`. Local development commonly uses `127.0.0.1:8000`.

## Limitations

There is no authentication, rate limit, request ID, custom exception schema, or business-level prediction monitoring. The metrics endpoint comes from `prometheus-fastapi-instrumentator` and is not listed in the root endpoint response.
