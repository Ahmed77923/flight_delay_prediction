# Testing

## Purpose

The repository uses pytest for focused unit and API tests.

## Implementation

The suite contains 14 passing tests covering:

- FastAPI health and prediction responses.
- Missing request fields and rejection of `ARR_DELAY`.
- Model metadata and absence of historical features.
- Feature construction and optional history behavior.
- One-hot preprocessing configuration.
- Custom target encoder out-of-fold behavior.

The API tests use FastAPI `TestClient` and replace model loading with a fake pipeline, so they do not require a live server or the real model artifact.

## Commands

```bash
python -m pytest
```

The verified environment command was:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ahmed_alsafi/data-science-env/bin/python -m pytest -p no:cacheprovider -q
```

Result: `14 passed, 2 warnings`. The warnings are dependency deprecations involving Starlette/httpx.

## Limitations

There is no configured lint, formatter, static type check, load test, end-to-end Compose test, model-quality regression test, or CI test job. The real artifact load was separately verified in the development environment, but it is not a pytest integration test.
