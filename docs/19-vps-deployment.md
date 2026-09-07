# VPS Deployment

## Purpose

This is a practical manual deployment procedure for the existing Compose configuration.

## 1. Prepare the VPS

Install Docker Engine and the Docker Compose plugin using the distribution's official instructions. Add the deployment user to the Docker group if appropriate, then start a new login session before retrying Docker commands.

```bash
docker --version
docker compose version
```

The current development session has already encountered `permission denied` for `/var/run/docker.sock`; adding a group membership does not affect the current shell until the session is refreshed.

## 2. Clone and build

```bash
git clone <repository-url>
cd flight_delay_prediction
docker compose build
docker compose up -d
docker compose ps
```

The placeholder repository URL must be replaced with the actual repository location; no remote URL is documented in the checkout.

## 3. Firewall and access

Allow only the services intended for users. The current host mappings are:

| Service | Host address |
|---|---|
| API | `http://<public-ip>:1041` |
| Prometheus | `http://<public-ip>:1042` |
| Streamlit | `http://<public-ip>:1043` |
| Grafana | `http://<public-ip>:1044` |

Prometheus should normally remain private. Grafana and the API also need authentication/TLS before public exposure, but those controls are not implemented here.

## 4. Verify

```bash
curl http://127.0.0.1:1041/health
curl http://127.0.0.1:1042/-/healthy
curl http://127.0.0.1:1042/-/ready
curl http://127.0.0.1:1042/api/v1/targets
```

Open Streamlit at `http://<public-ip>:1043` and API docs at `http://<public-ip>:1041/docs`. Grafana is at `http://<public-ip>:1044`; its Prometheus datasource must be configured manually as `http://prometheus:9090` from inside Compose.

## Address rules

- `127.0.0.1` or `localhost`: the machine making the request; on the VPS, this means the VPS itself.
- Public IP: a client outside the VPS.
- `api`, `prometheus`, `streamlit`, `grafana`: Compose DNS names usable between containers only.
- Host published ports `1041`-`1044`: usable from the host or external clients when firewall rules permit.

## Updates

```bash
git pull
docker compose build
docker compose up -d
docker compose ps
```

## Not implemented

There is no deployment script, zero-downtime strategy, backup procedure beyond the Grafana volume, automated rollback, DNS configuration, HTTPS certificate management, or CI/CD integration.
