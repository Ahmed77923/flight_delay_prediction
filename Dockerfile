FROM python:3.11-slim

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
COPY models/model_artifact/model ./model_artifact

ENV MLFLOW_MODEL_URI=/app/model_artifact \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api.main:app --host ${API_HOST} --port ${API_PORT}"]