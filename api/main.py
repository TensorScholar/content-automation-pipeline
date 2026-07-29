"""
Main FastAPI application definitions, middleware, and request handlers.
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

# Rate limiting and tracing
from fastapi_limiter import FastAPILimiter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Import dependency injection functions from new dependencies module
from api.dependencies import (
    get_content_agent,
    get_content_service,
    get_database,
    get_metrics,
    get_project_service,
    get_redis,
    get_user_service,
)

# Import exception handlers from separate module
from api.exceptions import add_exception_handlers

# Import route modules
from api.routes import auth, content, projects, system

# Import schemas from separate module
from api.schemas import (
    ArticleResponse,
    CreateProjectRequest,
    ErrorResponse,
    GenerateContentRequest,
    HealthCheckResponse,
    ProjectResponse,
    TaskStatusResponse,
    WorkflowStatusResponse,
)
from config.settings import settings
from core.models import ContentPlan, GeneratedArticle, Project
from infrastructure.error_tracking import initialize_sentry

# Import structured logging and configure
from infrastructure.monitoring import MetricsCollector, configure_structlog, get_logger

# Import rate limiting middleware
from infrastructure.rate_limiter import RateLimitConfig, RateLimitMiddleware

# Import request logging
from infrastructure.request_logger import RequestLoggingMiddleware
from orchestration.content_agent import ContentAgent
from security import SECURITY_HEADERS, get_security_headers
from services.content_service import ContentService
from services.project_service import ProjectService

# Configure structlog for the application
configure_structlog()
initialize_sentry("api")
logger = get_logger(__name__)


def _dependency_is_healthy(raw_status: str) -> bool:
    """Avoid treating strings like 'unhealthy' as healthy via substring checks."""
    normalized = raw_status.lower()
    return "healthy" in normalized and "unhealthy" not in normalized


def _secret_is_configured(value) -> bool:
    """Return True for populated SecretStr/string settings without exposing values."""
    if value is None:
        return False
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return bool(str(value).strip())


# ============================================================================
# MIDDLEWARE STACK (Cross-Cutting Concerns)
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach environment-aware security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Get environment-specific headers (includes configurable CSP)
        headers = get_security_headers()
        for k, v in headers.items():
            if k not in response.headers:
                response.headers[k] = v
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID for distributed tracing and correlation."""

    async def dispatch(self, request: Request, call_next):
        # Get request ID from header or generate new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Add to request state for access in route handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Enforce request timeout to prevent hanging connections.

    Protects against slow clients, network issues, and runaway queries
    that could exhaust server resources.
    """

    def __init__(self, app, timeout_seconds: float = 60.0):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        try:
            # Use asyncio.wait_for to enforce timeout
            response = await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
            return response
        except asyncio.TimeoutError:
            logger.warning(
                f"Request timeout after {self.timeout_seconds}s",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "request_id": getattr(request.state, "request_id", "unknown"),
                },
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "detail": f"Request timeout after {self.timeout_seconds} seconds",
                    "request_id": getattr(request.state, "request_id", None),
                },
            )


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

# Keep-alive task handle (prevents Neon Cloud compute from pausing)
_keepalive_task: asyncio.Task | None = None


async def _db_keepalive(database_manager) -> None:
    """Ping the DB every 4 minutes to keep Neon Cloud compute warm."""
    from sqlalchemy import text

    while True:
        await asyncio.sleep(240)  # 4 minutes
        try:
            async with database_manager.session() as session:
                await session.execute(text("SELECT 1"))
            logger.debug("db_keepalive_ping_ok")
        except Exception as e:
            logger.warning("db_keepalive_ping_failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager for startup/shutdown.

    Replaces deprecated @app.on_event decorators with context manager pattern.
    """
    # Startup
    try:
        if os.getenv("PYTEST_CURRENT_TEST"):
            logger.info("Test mode detected; skipping startup initialization")
            yield
            return

        # Validate critical environment variables
        logger.info("Validating environment configuration...")
        validation_errors = []
        validation_warnings = []

        # Check Python version
        import sys

        py_version = sys.version_info
        if py_version.major != 3 or py_version.minor < 11 or py_version.minor > 12:
            validation_warnings.append(
                f"Python 3.11-3.12 recommended, found {py_version.major}.{py_version.minor}.{py_version.micro}"
            )

        # Check database URL
        if not settings.database.url:
            validation_errors.append("DATABASE_URL is required")

        # Check Redis URL
        if not settings.redis.url:
            validation_errors.append("REDIS_URL is required")

        # Check LLM API keys (at least one required)
        has_anthropic = _secret_is_configured(settings.llm.anthropic_api_key)
        has_openai = _secret_is_configured(settings.llm.openai_api_key)
        has_gemini = _secret_is_configured(settings.llm.gemini_api_key)
        has_openai_compatible = bool(
            settings.llm.openai_compatible_base_url and settings.llm.openai_compatible_api_key
        )
        has_local = bool(settings.llm.local_llm_url and settings.llm.local_llm_url.strip())
        if (
            not has_anthropic
            and not has_openai
            and not has_gemini
            and not has_openai_compatible
            and not has_local
        ):
            validation_errors.append(
                "At least one LLM provider is required (managed API credentials, OPENAI_COMPATIBLE_BASE_URL/OPENAI_COMPATIBLE_API_KEY, or LOCAL_LLM_URL)"
            )
        selected_provider_configured = {
            "anthropic": has_anthropic,
            "openai": has_openai,
            "gemini": has_gemini,
            "openai_compatible": has_openai_compatible,
            "local": has_local,
        }.get(settings.llm.provider, False)
        if not selected_provider_configured:
            validation_errors.append(
                f"LLM_PROVIDER={settings.llm.provider} is selected but its required credentials are not configured"
            )

        # Check secret key strength
        if len(settings.secret_key.get_secret_value()) < 32:
            validation_errors.append("SECRET_KEY must be at least 32 characters long")

        # Check spacy model
        try:
            import spacy

            spacy.load("en_core_web_sm")
        except Exception:
            validation_warnings.append(
                "spacy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
            )

        # Log warnings
        for warning in validation_warnings:
            logger.warning(warning)

        if validation_errors:
            error_msg = "Environment configuration validation failed:\n" + "\n".join(
                f"  - {err}" for err in validation_errors
            )
            logger.error(error_msg)
            if settings.is_production:
                raise RuntimeError(error_msg)
            else:
                logger.warning("Continuing with invalid configuration (development mode)")
        else:
            logger.info("✓ Environment configuration validated successfully")

        database_manager = get_database()
        try:
            await asyncio.wait_for(database_manager.initialize(), timeout=60.0)
            logger.info("database_manager_initialized")
        except asyncio.TimeoutError:
            logger.error(
                "Database initialization timed out after 60s - connection pool may be exhausted"
            )
            if settings.is_production:
                raise RuntimeError("Failed to initialize database - connection pool timeout")
            else:
                logger.warning(
                    "Continuing without database initialization (development mode - will retry on first request)"
                )

        # Initialize FastAPI Limiter with raw Redis connection
        # FastAPILimiter requires native redis-py async client (with script_load method)
        try:
            redis_client = get_redis()
            await asyncio.wait_for(
                redis_client.initialize(), timeout=10.0
            )  # Ensure pool is initialized
            raw_redis = await redis_client.get_raw_connection()
            await asyncio.wait_for(FastAPILimiter.init(raw_redis), timeout=5.0)
            logger.info("rate_limiter_initialized")

            # Initialize token blacklist for secure logout
            from infrastructure.token_blacklist import init_token_blacklist

            init_token_blacklist(redis_client)
            logger.info("token_blacklist_initialized")
        except asyncio.TimeoutError:
            logger.error("Redis initialization timed out")
            if settings.is_production:
                raise RuntimeError("Redis connection timeout in production")
            logger.warning("Continuing without rate limiting (development mode)")
        except Exception as e:
            logger.error("rate_limiter_initialization_failed", error=str(e))
            # Fail startup in production if Redis is unavailable
            if settings.is_production:
                raise RuntimeError(f"Redis is required in production but unavailable: {e}")
            logger.warning("Continuing without rate limiting (development mode)")

        logger.info("application_startup_complete")

        # Launch DB keep-alive background task
        global _keepalive_task
        _keepalive_task = asyncio.create_task(_db_keepalive(get_database()))
        logger.info("db_keepalive_started", interval_seconds=240)

    except Exception as e:
        logger.warning("container_initialization_failed", error=str(e))
        if settings.is_production:
            raise
        logger.info("application_startup_complete", container_initialized=False)

    yield  # Application runs here --- keep-alive runs in background

    # Shutdown
    try:
        # Stop keep-alive task
        if _keepalive_task and not _keepalive_task.done():
            _keepalive_task.cancel()
            try:
                await _keepalive_task
            except asyncio.CancelledError:
                pass
        # Close rate limiter
        try:
            await FastAPILimiter.close()
            logger.info("rate_limiter_closed")
        except Exception as e:
            logger.warning("rate_limiter_cleanup_failed", error=str(e))

        database_manager = get_database()
        await database_manager.close()
        logger.info("database_manager_closed")
    except Exception as e:
        logger.warning("database_cleanup_failed", error=str(e))

    logger.info("application_shutdown_complete")


# Initialize rate limit configuration
rate_limit_config = RateLimitConfig(
    default_limit=100,  # 100 requests per minute
    default_window=60,
    auth_limit=20,  # 20 login attempts per minute (10 was too aggressive for testing)
    auth_window=60,
    concurrent_limit=10,  # 10 concurrent requests per user
)


app = FastAPI(
    title="Content Automation Engine API",
    description="Advanced NLP-driven SEO content automation platform with adaptive intelligence",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
)

# Add exception handlers
add_exception_handlers(app)

# Instrument FastAPI with OpenTelemetry for distributed tracing (if enabled)
if settings.monitoring.enable_tracing:
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry tracing enabled")
else:
    logger.info("OpenTelemetry tracing disabled")


# Simple dependency functions for FastAPI
def get_project_service_dependency():
    """Get ProjectService instance for FastAPI dependency injection."""
    return get_project_service()


def get_content_agent_dependency():
    """Get ContentAgent instance for FastAPI dependency injection."""
    return get_content_agent()


# Include route modules
logger.info("Registering routes...")
app.include_router(content.router)
logger.info(f"Registered content router with prefix: {content.router.prefix}")
app.include_router(projects.router)
app.include_router(system.router)
app.include_router(auth.router)


@app.get("/")
async def root():
    """Return a small service descriptor; development links to API docs."""
    if not settings.is_production:
        return RedirectResponse(url="/docs")
    return {
        "service": "Content Automation Engine API",
        "version": "1.0.0",
        "status": "running",
        "docs": "disabled in production",
    }


@app.get("/metrics", include_in_schema=False)
async def internal_metrics(metrics: MetricsCollector = Depends(get_metrics)) -> Response:
    """
    Internal Prometheus scrape endpoint.

    This route is intentionally mounted at the API container root so Prometheus
    can scrape `api:8000/metrics` over the private Docker network. nginx does
    not route `/metrics` publicly in the production config.
    """
    try:
        metrics_content = metrics.export_metrics()
        content_type = metrics.get_content_type()
        if not metrics_content or metrics_content.strip() == "":
            metrics_content = """# HELP system_info System information
# TYPE system_info gauge
system_info{version="1.0.0",status="running"} 1
"""
        return Response(content=metrics_content, media_type=content_type)
    except Exception as exc:
        fallback_metrics = f"""# HELP system_error System error occurred
# TYPE system_error gauge
system_error{{component="metrics",error="{str(exc)}"}} 1
"""
        return Response(
            content=fallback_metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


# Health alias at root to match Docker healthcheck and docs
@app.get("/health", response_model=HealthCheckResponse)
async def root_health(request: Request):
    try:
        db = get_database()
        redis = get_redis()
        dependencies = {}

        # Check database - skip pgvector check for performance
        # IMPROVED: Use asyncio.shield to prevent pool corruption on timeout
        try:
            if not db._is_initialized:
                dependencies["database"] = "degraded: not initialized"
            else:
                # Shield health check to prevent connection pool corruption
                health_task = asyncio.create_task(db.health_check(skip_vector_check=True))
                try:
                    await asyncio.wait_for(asyncio.shield(health_task), timeout=3.0)
                    dependencies["database"] = "healthy"
                except asyncio.TimeoutError:
                    dependencies["database"] = "degraded: timeout"
                    logger.warning("Database health check timed out after 3s")
        except asyncio.TimeoutError:
            dependencies["database"] = "degraded: timeout"
        except Exception as e:
            dependencies["database"] = f"unhealthy: {e}"
            logger.warning(f"Database health check failed: {e}")

        # Check Redis with timeout
        try:
            # IMPROVED: Shorter timeout and shielded task
            redis_task = asyncio.create_task(redis.ping())
            ok = await asyncio.wait_for(asyncio.shield(redis_task), timeout=1.0)
            dependencies["redis"] = "healthy" if ok else "unhealthy"
        except asyncio.TimeoutError:
            dependencies["redis"] = "unhealthy: timeout"
            logger.warning("Redis health check timeout")
        except Exception as e:
            dependencies["redis"] = f"unhealthy: {e}"
            logger.warning(f"Redis health check failed: {e}")

        overall_status = (
            "healthy"
            if all(_dependency_is_healthy(v) for v in dependencies.values())
            else "degraded"
        )
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            dependencies=dependencies,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthCheckResponse(
            status="unhealthy",  # More accurate status
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            dependencies={"error": str(e)},  # Include error details
        )


# Middleware stack (order matters: last added = first executed)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=60.0)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Accept",
        "Accept-Language",
        "Content-Type",
        "Authorization",
        "X-Request-ID",
        "X-Correlation-ID",
    ],
    expose_headers=[
        "X-Request-ID",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    ],
)


class HostValidationMiddleware(BaseHTTPMiddleware):
    """Validate Host header against settings.allowed_hosts."""

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0].lower()
        if (
            settings.allowed_hosts
            and host
            and host not in [h.lower() for h in settings.allowed_hosts]
        ):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid Host header"}
            )
        return await call_next(request)


# Add rate limiting middleware
try:
    if not os.getenv("PYTEST_CURRENT_TEST"):
        import asyncio as _asyncio

        _redis_wrapper = get_redis()
        # Must pass raw AsyncRedis — the wrapper has no .pipeline() method
        # Use run_until_complete only if no event loop is running yet (module-level)
        # In practice this block runs at import time before the event loop starts,
        # so we store a coroutine and resolve it lazily via lifespan.
        # The cleanest fix: create a bare AsyncRedis directly from REDIS_URL.
        import os as _os

        from redis.asyncio import Redis as _AsyncRedis

        _raw_redis_for_middleware = _AsyncRedis.from_url(
            _os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            decode_responses=False,
        )
        app.add_middleware(
            RateLimitMiddleware, redis_client=_raw_redis_for_middleware, config=rate_limit_config
        )
        logger.info(
            "Rate limiting middleware initialized",
            limit=rate_limit_config.default_limit,
            window=rate_limit_config.default_window,
        )
    else:
        logger.info("Rate limiting middleware skipped in test mode")
except Exception as e:
    logger.warning(f"Failed to initialize rate limiting: {e}")


# Add host validation as the first executed middleware
app.add_middleware(HostValidationMiddleware)


# ============================================================================
# API ENDPOINTS (Command/Query Handlers)
# ============================================================================
# Note: Project and content routes are handled by routers in api/routes/
# These routers are included via app.include_router() calls above.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info", access_log=True
    )
