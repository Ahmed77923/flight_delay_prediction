FROM python:3.11-slim

# libgomp1 provides the OpenMP runtime LightGBM's compiled extension links
# against. Without it, `import lightgbm` fails with:
#   OSError: libgomp.so.1: cannot open shared object file
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
COPY app ./app

# Pin the exact trained model artifact this image serves. The model was
# already trained/logged to MLflow outside this image (see project README);
# this build does not train or modify it. If the served model changes,
# update this build arg AND the matching exception rules in .dockerignore.
ARG MLFLOW_MODEL_ID=m-6d479b8fd10a4744862b3b6ec29260d8
COPY mlruns/4/models/${MLFLOW_MODEL_ID}/artifacts ./model_artifact

ENV MLFLOW_MODEL_URI=/app/model_artifact \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000 8501

# Default command runs the FastAPI service; docker-compose overrides this
# for the Streamlit service.
CMD ["sh", "-c", "uvicorn src.api.main:app --host ${API_HOST} --port ${API_PORT}"]
