# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ARG POETRY_VERSION=1.8.3
ENV VIRTUAL_ENV=/opt/venv \
    PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VIRTUAL_ENV}

RUN pip install --upgrade pip setuptools wheel && \
    pip install "poetry==${POETRY_VERSION}"

WORKDIR /src
COPY pyproject.toml poetry.lock* ./

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip install -r requirements.txt && \
    rm requirements.txt

FROM python:3.11-slim AS final

ENV VIRTUAL_ENV=/opt/venv \
    PATH="${VIRTUAL_ENV}/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY . .

RUN chown -R appuser:appuser /app

# Security: Ensure we run as non-root
USER appuser

# Verify user
RUN whoami

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]