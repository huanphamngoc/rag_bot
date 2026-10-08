# One image for the CLI (ingest, chat, ...), the web chat (gunicorn) and the tests.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    MLFLOW_DISABLE_TELEMETRY=true

WORKDIR /app
RUN useradd --create-home --uid 10001 rag

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

COPY pyproject.toml .
COPY crawlerrag ./crawlerrag
# The business rules; compose mounts the same folder over it so they can be edited without a rebuild.
COPY rules ./rules
COPY tests ./tests

USER rag
ENTRYPOINT ["python", "-m", "crawlerrag"]
CMD ["--help"]
