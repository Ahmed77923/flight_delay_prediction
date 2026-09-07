# Troubleshooting

## Purpose

This page provides checks for the runtime, Compose networking, model loading, and deployment issues supported by the current repository.

## API is unhealthy

```bash
docker compose ps
docker compose logs api --tail=100
curl http://127.0.0.1:1041/health
```

The API loads the model during startup. Check for a missing artifact, incompatible dependency, or incorrect `MLFLOW_MODEL_URI`. The healthcheck uses container port 8000, not host port 1041.

## Docker permission denied

If Docker reports permission denied for `/var/run/docker.sock`, the current user cannot access the Docker daemon. Confirm the group membership and start a new login session after adding the user to the Docker group. Do not expose the Docker socket to containers as a workaround.

## Port already allocated

```bash
sudo ss -ltnp | grep ':1041'
docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Ports}}'
```

Find the process or container using the published port. Change the host side of the Compose mapping only after updating the URLs used by clients and operators.

## Prometheus permission denied or restarting

```bash
docker compose logs prometheus --tail=100
curl http://127.0.0.1:1042/-/healthy
curl http://127.0.0.1:1042/-/ready
curl http://127.0.0.1:1042/api/v1/targets
```

The config is bind-mounted read-only with `:Z`, which supports SELinux relabeling on Fedora-like systems. Verify the file exists and that Docker can read it. A Docker daemon permission error is separate from a Prometheus bind-mount error.

## Grafana cannot connect to Prometheus

Configure the datasource URL as `http://prometheus:9090`. Do not use `localhost:9090` from inside Grafana; that resolves to the Grafana container. The repository does not provision this datasource automatically.

## MLflow model loading error

Check that `MLFLOW_MODEL_URI` points to the complete artifact directory containing `MLmodel` and `model.skops`. In Compose, the expected path is `/app/model_artifact`; the Dockerfile copies the selected model there. The `MLmodel` metadata contains a historical Windows artifact path, but the API should load the packaged current directory rather than that machine-specific path.

## Streamlit cannot connect to API

Inside Compose, use `API_URL=http://api:8000`. From the host, use `http://127.0.0.1:1041`. Check Streamlit logs and the API health endpoint.

## Training cannot start

The current checkout lacks both `data/` and `src/data/split_data.py`. Add the expected CSV inputs and restore the missing split module before running `python -m src.models.train`.
