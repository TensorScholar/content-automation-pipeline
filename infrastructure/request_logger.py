"""
Request Logging Module for Comprehensive API Audit Trail

Provides:
- Detailed API request/response logging with user ID, timestamp, action
- LLM API call tracking for cost analysis
- Request body truncation for sensitive data
- Structured JSON logging for log aggregation
- Performance metrics (response time, payload sizes)
- Error tracking and debugging information
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from structlog import get_logger
from infrastructure.redaction import redact_secrets

logger = get_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

class LogLevel(str, Enum):
    """Logging levels for request logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ActionType(str, Enum):
    """Types of actions for audit trail."""
    GENERATE_CONTENT = "generate_content"
    CREATE_PROJECT = "create_project"
    UPDATE_PROJECT = "update_project"
    DELETE_PROJECT = "delete_project"
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"
    API_CALL = "api_call"
    LLM_CALL = "llm_call"
    DATABASE_OPERATION = "database_operation"


# Max sizes for logging (to prevent log explosion)
MAX_BODY_LOG_SIZE = 1000        # Truncate request body to 1KB
MAX_RESPONSE_LOG_SIZE = 5000    # Truncate response to 5KB
MAX_ERROR_LOG_SIZE = 10000      # Truncate error messages to 10KB

# Sensitive fields to mask in logs
SENSITIVE_FIELDS = {
    'password', 'token', 'api_key', 'secret', 'authorization',
    'x-api-key', 'x-auth-token', 'credit_card', 'ssn',
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mask_sensitive_data(data: dict, fields: set = SENSITIVE_FIELDS) -> dict:
    """Mask sensitive fields in dictionary."""
    del fields
    return redact_secrets(data)


def truncate_string(text: str, max_length: int) -> str:
    """Truncate string to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [truncated, total: {len(text)} chars]"


def safe_json_encode(obj: Any, max_size: int = MAX_BODY_LOG_SIZE) -> str:
    """Safely encode object to JSON with size limit."""
    try:
        json_str = json.dumps(obj, default=str)
        return truncate_string(json_str, max_size)
    except (TypeError, ValueError) as e:
        return f"[Could not encode: {str(e)[:100]}]"


# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive request/response logging middleware.

    Logs:
    - Request: method, path, query params, headers, user
    - Response: status code, response time, size
    - Errors: exception details, traceback
    """

    def __init__(self, app, log_request_body: bool = True, log_response_body: bool = False):
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive logging."""

        # Start timing
        start_time = time.perf_counter()

        # Extract request information
        request_id = request.headers.get("X-Request-ID", "unknown")
        user_id = getattr(request.state, "user_id", None)
        method = request.method
        path = request.url.path
        query_string = str(request.url.query)

        # Extract headers (mask sensitive ones)
        headers = dict(request.headers)
        headers = mask_sensitive_data(headers)

        # Determine if we should log this request
        should_log = not self._is_excluded_path(path)

        if should_log:
            # Log incoming request
            self._log_request(
                request_id=request_id,
                user_id=user_id,
                method=method,
                path=path,
                query_string=query_string,
                headers=headers,
            )

        # Process request and capture response
        try:
            response = await call_next(request)
            elapsed = time.perf_counter() - start_time

            if should_log:
                self._log_response(
                    request_id=request_id,
                    user_id=user_id,
                    method=method,
                    path=path,
                    status_code=response.status_code,
                    elapsed=elapsed,
                )

            return response

        except Exception as e:
            elapsed = time.perf_counter() - start_time

            if should_log:
                self._log_error(
                    request_id=request_id,
                    user_id=user_id,
                    method=method,
                    path=path,
                    error=e,
                    elapsed=elapsed,
                )

            raise

    @staticmethod
    def _is_excluded_path(path: str) -> bool:
        """Check if path should be excluded from logging."""
        excluded = {"/metrics", "/health", "/system/health", "/docs", "/openapi.json"}
        return path in excluded

    @staticmethod
    def _log_request(request_id: str, user_id: Optional[str], method: str,
                     path: str, query_string: str, headers: dict):
        """Log incoming request."""
        logger.info(
            "http_request_received",
            request_id=request_id,
            user_id=user_id,
            method=method,
            path=path,
            query_string=query_string,
            headers=headers,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _log_response(request_id: str, user_id: Optional[str], method: str,
                     path: str, status_code: int, elapsed: float):
        """Log response."""
        # structlog's filtering logger expects numeric levels
        level = logging.WARNING if status_code >= 400 else logging.INFO
        logger.log(
            level,
            "http_response_sent",
            request_id=request_id,
            user_id=user_id,
            method=method,
            path=path,
            status_code=status_code,
            response_time_ms=round(elapsed * 1000, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _log_error(request_id: str, user_id: Optional[str], method: str,
                  path: str, error: Exception, elapsed: float):
        """Log error."""
        logger.error(
            "http_request_error",
            request_id=request_id,
            user_id=user_id,
            method=method,
            path=path,
            error_type=type(error).__name__,
            error_message=str(error)[:MAX_ERROR_LOG_SIZE],
            response_time_ms=round(elapsed * 1000, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            exc_info=True,
        )


# ============================================================================
# LLM CALL LOGGING
# ============================================================================

class LLMCallLogger:
    """Logger for LLM API calls with cost tracking."""

    @staticmethod
    def log_call(
        provider: str,
        model: str,
        action: str,
        tokens_used: int,
        cost: float,
        user_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Log LLM API call for audit and cost tracking.

        Args:
            provider: LLM provider (openai, anthropic)
            model: Model name
            action: Action type (generate, analyze, summarize)
            tokens_used: Total tokens used
            cost: Cost in USD
            user_id: User making the call
            metadata: Additional context
        """
        logger.info(
            "llm_call",
            provider=provider,
            model=model,
            action=action,
            tokens_used=tokens_used,
            cost=cost,
            user_id=user_id,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def log_error(
        provider: str,
        model: str,
        action: str,
        error: Exception,
        user_id: Optional[str] = None,
    ):
        """Log LLM API error."""
        logger.error(
            "llm_call_error",
            provider=provider,
            model=model,
            action=action,
            error_type=type(error).__name__,
            error_message=str(error)[:MAX_ERROR_LOG_SIZE],
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ============================================================================
# AUDIT TRAIL LOGGING
# ============================================================================

class AuditLogger:
    """Logger for audit trail (user actions, changes, etc.)."""

    @staticmethod
    def log_action(
        action: ActionType,
        user_id: str,
        resource: str,
        resource_id: str,
        changes: Optional[dict] = None,
        status: str = "success",
        metadata: Optional[dict] = None,
    ):
        """
        Log user action for audit trail.

        Args:
            action: Type of action
            user_id: User performing action
            resource: Resource type (project, article, etc.)
            resource_id: ID of resource
            changes: What was changed (before/after)
            status: success/failure
            metadata: Additional context
        """
        logger.info(
            "audit_action",
            action=action.value,
            user_id=user_id,
            resource=resource,
            resource_id=resource_id,
            changes=changes,
            status=status,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def log_access(
        user_id: str,
        resource: str,
        resource_id: str,
        access_type: str,  # read, write, delete
        granted: bool,
        reason: Optional[str] = None,
    ):
        """Log access attempt (read/write/delete)."""
        level = "info" if granted else "warning"
        logger.log(
            level,
            "audit_access",
            user_id=user_id,
            resource=resource,
            resource_id=resource_id,
            access_type=access_type,
            granted=granted,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ============================================================================
# DATABASE OPERATION LOGGING
# ============================================================================

class DatabaseLogger:
    """Logger for database operations."""

    @staticmethod
    def log_query(
        operation: str,  # SELECT, INSERT, UPDATE, DELETE
        table: str,
        rows_affected: int,
        duration_ms: float,
        user_id: Optional[str] = None,
    ):
        """Log database query."""
        logger.info(
            "database_query",
            operation=operation,
            table=table,
            rows_affected=rows_affected,
            duration_ms=round(duration_ms, 2),
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def log_error(
        operation: str,
        table: str,
        error: Exception,
        user_id: Optional[str] = None,
    ):
        """Log database error."""
        logger.error(
            "database_error",
            operation=operation,
            table=table,
            error_type=type(error).__name__,
            error_message=str(error)[:MAX_ERROR_LOG_SIZE],
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ============================================================================
# DECORATORS FOR FUNCTION-LEVEL LOGGING
# ============================================================================

def log_llm_call(provider: str, model: str, action: str):
    """Decorator for logging LLM function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            user_id = kwargs.get("user_id", None)
            try:
                result = await func(*args, **kwargs)

                # Extract tokens and cost from result if available
                tokens = getattr(result, "tokens_used", 0)
                cost = getattr(result, "cost", 0.0)

                LLMCallLogger.log_call(
                    provider=provider,
                    model=model,
                    action=action,
                    tokens_used=tokens,
                    cost=cost,
                    user_id=user_id,
                )

                return result
            except Exception as e:
                LLMCallLogger.log_error(
                    provider=provider,
                    model=model,
                    action=action,
                    error=e,
                    user_id=user_id,
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            user_id = kwargs.get("user_id", None)
            try:
                result = func(*args, **kwargs)

                tokens = getattr(result, "tokens_used", 0)
                cost = getattr(result, "cost", 0.0)

                LLMCallLogger.log_call(
                    provider=provider,
                    model=model,
                    action=action,
                    tokens_used=tokens,
                    cost=cost,
                    user_id=user_id,
                )

                return result
            except Exception as e:
                LLMCallLogger.log_error(
                    provider=provider,
                    model=model,
                    action=action,
                    error=e,
                    user_id=user_id,
                )
                raise

        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_database_operation(operation: str, table: str):
    """Decorator for logging database operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            user_id = kwargs.get("user_id", None)
            start = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000

                # Try to extract rows affected
                rows = len(result) if isinstance(result, list) else (1 if result else 0)

                DatabaseLogger.log_query(
                    operation=operation,
                    table=table,
                    rows_affected=rows,
                    duration_ms=elapsed,
                    user_id=user_id,
                )

                return result
            except Exception as e:
                DatabaseLogger.log_error(
                    operation=operation,
                    table=table,
                    error=e,
                    user_id=user_id,
                )
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else func

    return decorator
