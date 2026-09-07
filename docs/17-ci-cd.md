# CI/CD

## Purpose

This page records the repository's automation status.

## Implementation

No GitHub Actions, GitLab CI, Jenkins file, CI YAML, Docker publishing workflow, or deployment pipeline is present. There are no configured triggers, automated tests, image builds, registry pushes, or deployments.

## Current workflow

Validation and deployment are manual:

```bash
python -m pytest
docker compose build
docker compose up -d
docker compose ps
```

## Planned / optional

A future pipeline could run tests, build the image, scan dependencies, publish a tagged image, and deploy Compose. Those activities are not implemented and should not be assumed by maintainers or reviewers.
