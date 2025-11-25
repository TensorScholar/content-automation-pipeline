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
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Rate limiting and tracing
from fastapi_limiter import FastAPILimiter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentation

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
from container import (
    DatabaseManager as _DB,
)
from container import (
    RedisClient as _RC,
)

# Import dependency injection functions from container
from container import (
    container,
    get_content_agent,
    get_content_service,
    get_database,
    get_project_service,
    get_redis,
    get_user_service,
)
from core.models import ContentPlan, GeneratedArticle, Project

# Import structured logging and configure
from infrastructure.monitoring import configure_structlog, get_logger
from orchestration.content_agent import ContentAgent
from security import SECURITY_HEADERS, get_security_headers
from services.content_service import ContentService
from services.project_service import ProjectService

# Configure structlog for the application
configure_structlog()
logger = get_logger(__name__)

# ============================================================================
# MIDDLEWARE STACK (Cross-Cutting Concerns)
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach environment-aware security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            # Get environment-specific headers (includes configurable CSP)
            headers = get_security_headers()
            for k, v in headers.items():
                if k not in response.headers:
                    response.headers[k] = v
        except Exception:
            pass
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


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager for startup/shutdown.

    Replaces deprecated @app.on_event decorators with context manager pattern.
    """
    # Startup
    try:
        # Validate critical environment variables
        logger.info("Validating environment configuration...")
        validation_errors = []
        
        # Check database URL
        if not settings.database.url:
            validation_errors.append("DATABASE_URL is required")
        
        # Check Redis URL
        if not settings.redis.url:
            validation_errors.append("REDIS_URL is required")
        
        # Check LLM API keys (at least one required)
        has_anthropic = settings.llm.anthropic_api_key is not None
        has_openai = settings.llm.openai_api_key is not None
        if not has_anthropic and not has_openai:
            validation_errors.append("At least one LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY) is required")
        
        # Check secret key strength
        if len(settings.secret_key.get_secret_value()) < 32:
            validation_errors.append("SECRET_KEY must be at least 32 characters long")
        
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
        
        database_manager = container.database()
        await database_manager.initialize()
        logger.info("database_manager_initialized")

        # Initialize FastAPI Limiter with Redis
        try:
            redis = get_redis()
            await FastAPILimiter.init(redis)
            logger.info("rate_limiter_initialized")
        except Exception as e:
            logger.error("rate_limiter_initialization_failed", error=str(e))
            # Fail startup in production if Redis is unavailable
            if settings.is_production:
                raise RuntimeError(f"Redis is required in production but unavailable: {e}")
            logger.warning("Continuing without rate limiting (development mode)")

        logger.info("application_startup_complete")
    except Exception as e:
        logger.warning("container_initialization_failed", error=str(e))
        logger.info("application_startup_complete", container_initialized=False)

    yield  # Application runs here

    # Shutdown
    try:
        # Close rate limiter
        try:
            await FastAPILimiter.close()
            logger.info("rate_limiter_closed")
        except Exception as e:
            logger.warning("rate_limiter_cleanup_failed", error=str(e))

        database_manager = container.database()
        await database_manager.close()
        logger.info("database_manager_closed")
    except Exception as e:
        logger.warning("database_cleanup_failed", error=str(e))

    logger.info("application_shutdown_complete")


app = FastAPI(
    title="Content Automation Engine API",
    description="Advanced NLP-driven SEO content automation platform with adaptive intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add exception handlers
add_exception_handlers(app)

# Instrument FastAPI with OpenTelemetry for distributed tracing (if enabled)
if settings.monitoring.enable_tracing:
    FastAPIInstrumentation.instrument_app(app)
    logger.info("OpenTelemetry tracing enabled")
else:
    logger.info("OpenTelemetry tracing disabled")

# Wire container to enable dependency injection BEFORE including routes
container.wire(
    modules=[
        "api.routes.content",
        "api.routes.projects",
        "api.routes.system",
        "api.routes.auth",
        "security",  # ensure security dependencies are wired
    ]
)


# Simple dependency functions for FastAPI
def get_project_service_dependency():
    """Get ProjectService instance for FastAPI dependency injection."""
    return get_project_service()


def get_content_agent_dependency():
    """Get ContentAgent instance for FastAPI dependency injection."""
    return container.content_agent()


# Include route modules
app.include_router(content.router)
app.include_router(projects.router)
app.include_router(system.router)
app.include_router(auth.router)

# API Root - redirect to docs
@app.get("/")
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


# Health alias at root to match Docker healthcheck and docs
@app.get("/health", response_model=HealthCheckResponse)
async def root_health(request: Request):
    try:
        db = get_database()
        redis = get_redis()
        dependencies = {}
        try:
            await db.health_check()
            dependencies["database"] = "healthy"
        except Exception as e:
            dependencies["database"] = f"unhealthy: {e}"
        try:
            ok = await redis.ping()
            dependencies["redis"] = "healthy" if ok else "unhealthy"
        except Exception as e:
            dependencies["redis"] = f"unhealthy: {e}"
        overall_status = (
            "healthy" if all("healthy" in v for v in dependencies.values()) else "degraded"
        )
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            dependencies=dependencies,
        )
    except Exception:
        return HealthCheckResponse(
            status="degraded",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            dependencies={"error": "unexpected"},
        )


# Middleware stack (order matters: last added = first executed)
app.add_middleware(RequestIDMiddleware)
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
    expose_headers=["X-Request-ID"],
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
