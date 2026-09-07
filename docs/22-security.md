# Security

## Purpose

This page separates controls present in the repository from risks that remain open.

## Implemented security-related behavior

- Request schemas forbid unknown fields, including the target `ARR_DELAY`.
- Model artifacts are loaded from a configured local path rather than accepting a user-supplied model URI.
- The Docker build excludes `.env`, local data, notebooks, and most MLflow content, then explicitly includes the selected artifact.
- FastAPI validates dates, HHMM times, required columns, NaN values, and negative distance.
- The Streamlit client applies request timeouts and avoids displaying raw exception traces.

## Missing security / recommendations

- No API authentication, authorization, rate limiting, audit identity, or request signing.
- No HTTPS/TLS or reverse proxy.
- API, Prometheus, Streamlit, and Grafana ports are published by Compose; firewall them on a VPS.
- Grafana credentials and datasource provisioning are not configured by the repository.
- Prometheus has no access control.
- The Docker socket is not used by the application; keep it protected and do not grant broader privileges than necessary.
- The tracked `.env` should be reviewed and replaced with an ignored local file or secret manager if it ever contains credentials.
- Model and input data access policies are not defined.
- Dependencies and container images are pinned in Python requirements but Compose uses `latest` for Prometheus and Grafana, so image provenance is not reproducible.

Do not describe the current deployment as production-secure until network access, credentials, TLS, image pinning, and operational controls are added.
