# Flight Delay Prediction Documentation

## Purpose

This directory documents the current repository implementation, including its limitations. It is the source of truth for developers, reviewers, MLOps work, and Docker-based deployment.

## Contents

1. [Project overview](01-project-overview.md)
2. [Architecture](02-architecture.md)
3. [Project structure](03-project-structure.md)
4. [Data](04-data.md)
5. [Feature engineering](05-feature-engineering.md)
6. [Modeling](06-modeling.md)
7. [Training](07-training.md)
8. [MLflow](08-mlflow.md)
9. [API](09-api.md)
10. [Streamlit](10-streamlit.md)
11. [Docker](11-docker.md)
12. [Docker Compose](12-docker-compose.md)
13. [Monitoring](13-monitoring.md)
14. [Prometheus](14-prometheus.md)
15. [Grafana](15-grafana.md)
16. [Testing](16-testing.md)
17. [CI/CD](17-ci-cd.md)
18. [Deployment](18-deployment.md)
19. [VPS deployment](19-vps-deployment.md)
20. [Configuration](20-configuration.md)
21. [Troubleshooting](21-troubleshooting.md)
22. [Security](22-security.md)
23. [Development guide](23-development-guide.md)

## Current status at a glance

- **Inference:** implemented with FastAPI and a committed MLflow/skops artifact.
- **UI:** implemented with Streamlit as an HTTP client of FastAPI.
- **Container orchestration:** implemented with Docker Compose for API, Streamlit, Prometheus, and Grafana.
- **Monitoring:** Prometheus scraping is configured; Grafana datasource and dashboard provisioning are not configured.
- **Training:** trainer code exists, but the current checkout has no `data/` directory and no `src/data/split_data.py`, so retraining from this checkout is not currently runnable.
- **CI/CD:** not implemented.
- **Authentication and HTTPS:** not implemented.

Commands and values in these pages are based on the files currently present in the repository, not on hypothetical infrastructure.
