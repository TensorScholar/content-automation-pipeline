"""
API Exception Handlers: Domain Error → HTTP Error Mapping

Centralized exception handling for clean error responses.
Maps domain-specific exceptions to appropriate HTTP status codes.
"""

import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.exceptions import (
    DistributionError,
    ProjectNotFoundError,
    TokenBudgetExceededError,
    WorkflowError,
)

logger = logging.getLogger(__name__)


def _sanitize_validation_errors(errors: list) -> list:
    """
    Ensure all validation error details are JSON-serializable.

    Pydantic v2 may embed raw ValueError instances in ctx.error which
    cannot be serialised by json.dumps / JSONResponse, causing a 500.
    """
    sanitized = []
    for err in errors:
        clean = dict(err)
        ctx = clean.get("ctx")
        if isinstance(ctx, dict):
            clean["ctx"] = {
                k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for k, v in ctx.items()
            }
        sanitized.append(clean)
    return sanitized


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": _sanitize_validation_errors(exc.errors()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def project_not_found_handler(request: Request, exc: ProjectNotFoundError):
    """Handle project not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Project Not Found",
            "detail": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def workflow_error_handler(request: Request, exc: WorkflowError):
    """Handle workflow execution errors."""
    logger.error(f"Workflow error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Workflow Error",
            "detail": "Content generation workflow failed. Please try again.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def budget_exceeded_handler(request: Request, exc: TokenBudgetExceededError):
    """Handle token budget exceeded errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Token Budget Exceeded",
            "detail": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def distribution_error_handler(request: Request, exc: DistributionError):
    """Handle content distribution errors."""
    logger.error(f"Distribution error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "Distribution Error",
            "detail": "Content distribution failed. Please check WordPress configuration.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unhandled exceptions. Prevents leaking internals."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def add_exception_handlers(app: FastAPI):
    """Add all exception handlers to the FastAPI app."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ProjectNotFoundError, project_not_found_handler)
    app.add_exception_handler(WorkflowError, workflow_error_handler)
    app.add_exception_handler(TokenBudgetExceededError, budget_exceeded_handler)
    app.add_exception_handler(DistributionError, distribution_error_handler)
    # Global catch-all: prevents raw tracebacks from reaching clients
    app.add_exception_handler(Exception, global_exception_handler)
