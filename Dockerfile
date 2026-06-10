# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ARG POETRY_VERSION=1.8.3
ARG TORCH_CPU_VERSION=2.9.0+cpu
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYPI_INDEX_URL=https://pypi.org/simple
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 update && \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VIRTUAL_ENV}

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install "poetry==${POETRY_VERSION}"

WORKDIR /src
COPY pyproject.toml poetry.lock* ./

RUN --mount=type=cache,target=/root/.cache/pip \
    poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    grep -Ev '^(torch|triton|nvidia-[^= ]+)(==|[<>=!~ ])' requirements.txt > requirements.docker.txt && \
    pip install \
        --index-url "${PYTORCH_CPU_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        "torch==${TORCH_CPU_VERSION}" && \
    pip install -r requirements.docker.txt && \
    python -c "import importlib.metadata as md; names={dist.metadata['Name'].lower() for dist in md.distributions()}; blocked=sorted(name for name in names if name.startswith('nvidia-') or name == 'triton'); assert not blocked, blocked; import torch; assert torch.version.cuda is None, torch.version.cuda; import sentence_transformers" && \
    rm requirements.txt requirements.docker.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

FROM python:3.11-slim AS final

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 update && \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY . .

# Make entrypoint executable and set ownership
RUN chmod +x entrypoint.sh && chown -R appuser:appuser /app

# Security: Ensure we run as non-root
USER appuser

# Verify user
RUN whoami

EXPOSE 8000 5555

ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
