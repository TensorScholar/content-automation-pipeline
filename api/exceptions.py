"""
API Exception Handlers: Domain Error → HTTP Error Mapping

Centralized exception handling for clean error responses.
Maps domain-specific exceptions to appropriate HTTP status codes.
"""

from datetime import datetime

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.exceptions import (
    DistributionError,
    ProjectNotFoundError,
    TokenBudgetExceededError,
    WorkflowError,
)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": datetime.utcnow().isoformat(),
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
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def workflow_error_handler(request: Request, exc: WorkflowError):
    """Handle workflow execution errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Workflow Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
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
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def distribution_error_handler(request: Request, exc: DistributionError):
    """Handle content distribution errors."""
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={
            "error": "Distribution Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
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
