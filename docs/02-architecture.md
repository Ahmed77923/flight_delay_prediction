# Architecture

## Purpose

This page describes the runtime, training, and container-network boundaries that exist in the checkout.

## Architecture / Concept

```mermaid
flowchart TB
    subgraph Host
      Browser -->|1043| Streamlit
      Browser -->|1041| API
      Browser -->|1042| Prometheus
      Browser -->|1044| Grafana
    end
    subgraph Compose network
      Streamlit -->|http://api:8000| API[api: FastAPI]
      Prometheus -->|http://api:8000/metrics| API
      Grafana -->|manual datasource: http://prometheus:9090| Prometheus
      API --> Artifact[/app/model_artifact]
    end
```

The host ports are published ports. The service names and container ports are used for traffic between Compose services. For example, `127.0.0.1:1041` is a host/browser address, while `api:8000` is the address Streamlit and Prometheus use inside the Compose network.

## Implementation

The API loads the model once in the FastAPI lifespan. `/predict` converts the Pydantic request to a DataFrame, calls `build_features`, selects the fitted preprocessor columns, and invokes `predict`. The Streamlit process never loads the model.

The Compose services share the default project network. `streamlit` waits for an API healthcheck; Prometheus also waits for a healthy API. Grafana only waits for Prometheus to be started, not ready.

| Boundary | Address |
|---|---|
| Host to API | `http://127.0.0.1:1041` |
| Host to Streamlit | `http://127.0.0.1:1043` |
| Host to Prometheus | `http://127.0.0.1:1042` |
| Host to Grafana | `http://127.0.0.1:1044` |
| Container to API | `http://api:8000` |
| Container to Prometheus | `http://prometheus:9090` |
| Container to Streamlit | `http://streamlit:1040` |
| Container to Grafana | `http://grafana:3000` |

## Configuration

The API container receives `API_HOST=0.0.0.0`, `API_PORT=8000`, and `MLFLOW_MODEL_URI=/app/model_artifact`. Streamlit receives `API_URL=http://api:8000`. Prometheus targets `api:8000` and scrapes `/metrics` every 15 seconds.

## Limitations

There is no application database, Kubernetes layer, external MLflow server service, reverse proxy, TLS, authentication, or automated Grafana configuration. The Dockerfile declares port `8501`, while Compose starts Streamlit on container port `1040`; the Compose mapping is the active containerized UI configuration.
