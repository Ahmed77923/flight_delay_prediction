# Prometheus

## Purpose

Prometheus scrapes the FastAPI metrics endpoint and stores time-series data for querying by Grafana.

## Implementation

The complete current configuration is:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "flight-delay-api"
    metrics_path: /metrics
    static_configs:
      - targets:
          - "api:8000"
```

Inside the Compose network, `api:8000` resolves to the API container. `127.0.0.1:1041` would refer to the Prometheus container itself, not the host-published API port, so it is the wrong scrape address. Host users access Prometheus at `http://127.0.0.1:1042`.

Useful checks:

```bash
curl http://127.0.0.1:1042/-/healthy
curl http://127.0.0.1:1042/-/ready
curl http://127.0.0.1:1042/api/v1/targets
curl http://127.0.0.1:1041/metrics
```

## Configuration

Compose mounts the config read-only at `/etc/prometheus/prometheus.yml` and starts Prometheus with `--config.file=/etc/prometheus/prometheus.yml`.

## Troubleshooting

If the target is down, inspect `docker compose logs api` and `docker compose logs prometheus`, then check the API health and metrics endpoints from the host. A bind-mount permission problem on Fedora/SELinux is mitigated in Compose by the `:Z` suffix. Socket permissions are unrelated to Prometheus and affect whether the user can run Docker commands.

## Limitations

Only one static API target is configured. There is no Docker service discovery, alerting rule, remote write, authentication, or explicit retention setting.
