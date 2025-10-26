# ============================================================================
# Multi-Stage Production Dockerfile
# Optimized for: security, size, build speed, layer caching
# ============================================================================

# ----------------------------------------------------------------------------
# Stage 1: Builder - Compile dependencies and install packages
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Download spaCy language model
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Note: sentence-transformers model will be downloaded on first use

# ----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# ----------------------------------------------------------------------------
FROM python:3.11-slim

# Metadata labels
LABEL maintainer="content-automation-team@example.com"
LABEL version="1.0.0"
LABEL description="Advanced NLP-Driven SEO Content Automation Engine"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    # Application configuration
    LOG_LEVEL=INFO \
    WORKERS=4 \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -s /bin/bash appuser && \
    mkdir -p ${APP_HOME} && \
    chown -R appuser:appuser ${APP_HOME}

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code
WORKDIR ${APP_HOME}
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p logs data cache && \
    chown -R appuser:appuser logs data cache

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command: Run API server
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]