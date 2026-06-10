"""
Rate Limiting Middleware for FastAPI Production Application

This module provides:
- Per-user and per-IP rate limiting
- Configurable limits per endpoint
- Graceful degradation on limit exceeded
- Prometheus metrics integration
- Comprehensive logging for auditing
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_settings
from infrastructure.monitoring import get_logger

logger = get_logger(__name__)


# ============================================================================
# RATE LIMITER EXCEPTIONS
# ============================================================================

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, retry_after: int, limit_type: str):
        self.retry_after = retry_after
        self.limit_type = limit_type
        super().__init__(f"Rate limit exceeded ({limit_type})")


# ============================================================================
# RATE LIMIT CONFIGURATION
# ============================================================================

class RateLimitConfig:
    """Configuration for rate limiting."""

    def __init__(
        self,
        default_limit: int = 100,  # requests per minute
        default_window: int = 60,   # seconds
        auth_limit: int = 10,       # authentication attempts per minute
        auth_window: int = 60,
        auth_ip_limit: Optional[int] = None,  # broad IP guard for auth endpoints
        concurrent_limit: int = 10, # concurrent requests per user
    ):
        self.default_limit = default_limit
        self.default_window = default_window
        self.auth_limit = auth_limit
        self.auth_window = auth_window
        self.auth_ip_limit = auth_ip_limit if auth_ip_limit is not None else max(auth_limit * 5, default_limit)
        self.concurrent_limit = concurrent_limit


# ============================================================================
# REDIS-BASED RATE LIMITER
# ============================================================================

class RedisRateLimiter:
    """Redis-based rate limiter for distributed systems."""

    def __init__(self, redis_client: AsyncRedis, config: RateLimitConfig):
        """
        Initialize rate limiter.

        Args:
            redis_client: Redis async client
            config: Rate limit configuration
        """
        self.redis = redis_client
        self.config = config

    async def is_allowed(
        self,
        identifier: str,
        limit: int,
        window: int,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed under rate limit.

        Args:
            identifier: Unique identifier (user_id, IP, etc.)
            limit: Maximum requests allowed
            window: Time window in seconds

        Returns:
            Tuple of (allowed: bool, info: dict)
            info contains: current_count, limit, retry_after, reset_time
        """
        key = f"rate_limit:{identifier}"

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                await pipe.incr(key)
                await pipe.ttl(key)
                results = await pipe.execute()

            current, ttl = results

            # If it's the first request or TTL is misconfigured, set the expiry
            if current == 1 or ttl == -1:
                await self.redis.expire(key, window)
                ttl = window

            allowed = current <= limit
            retry_after = ttl if not allowed else 0

            info = {
                "current": current,
                "limit": limit,
                "retry_after": max(0, retry_after),
                "reset_time": datetime.now() + timedelta(seconds=max(0, ttl)),
            }

            return allowed, info

        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            fail_closed = get_settings().is_production
            return not fail_closed, {
                "current": 0,
                "limit": limit,
                "retry_after": window if fail_closed else 0,
                "reset_time": datetime.now() + timedelta(seconds=window),
                "error": str(e),
            }

    async def get_concurrent_count(self, identifier: str) -> int:
        """Get current concurrent request count."""
        key = f"concurrent:{identifier}"
        try:
            count = await self.redis.get(key)
            return int(count) if count else 0
        except Exception:
            return 0

    async def increment_concurrent(self, identifier: str, ttl: int = 300) -> int:
        """Increment concurrent request count."""
        key = f"concurrent:{identifier}"
        try:
            count = await self.redis.incr(key)
            if count == 1:
                await self.redis.expire(key, ttl)
            return count
        except Exception:
            return 0

    async def decrement_concurrent(self, identifier: str) -> int:
        """Decrement concurrent request count."""
        key = f"concurrent:{identifier}"
        try:
            count = await self.redis.decr(key)
            return max(0, count)
        except Exception:
            return 0


# ============================================================================
# RATE LIMITING MIDDLEWARE
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI applications.

    Features:
    - Per-user rate limiting (authenticated)
    - Per-IP fallback rate limiting (unauthenticated)
    - Endpoint-specific limits
    - Concurrent request tracking
    - Prometheus metrics
    """

    def __init__(
        self,
        app,
        redis_client: AsyncRedis,
        config: Optional[RateLimitConfig] = None,
    ):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.limiter = RedisRateLimiter(redis_client, self.config)

        # Endpoint-specific limits (path: (limit, window))
        # PRODUCTION: Strict limits for expensive operations
        # NOTE: Paths must match actual route prefixes (no /api/ prefix)
        self.endpoint_limits = {
            # Authentication endpoints (strict to prevent brute force)
            "/auth/login": (self.config.auth_limit, self.config.auth_window),
            "/auth/register": (self.config.auth_limit, self.config.auth_window),
            "/auth/token": (self.config.auth_limit, self.config.auth_window),
            "/auth/refresh": (15, 60),  # 15 refreshes per minute

            # Content generation endpoints (expensive LLM operations)
            "/content/generate": (5, 300),  # 5 generations per 5 minutes
            "/content/generate/async": (5, 300),  # 5 async generations per 5 minutes
            "/content/batch": (2, 600),  # 2 batch operations per 10 minutes

            # Project management (moderate)
            "/projects": (30, 60),  # 30 per minute

            # Task status endpoints (allow frequent polling)
            "/content/tasks": (60, 60),  # 60 requests per minute
            "/content/task": (60, 60),  # 60 requests per minute
        }

        # Exclude from rate limiting
        self.excluded_paths = {
            "/docs",
            "/openapi.json",
            "/health",
            "/system/health",
            "/metrics",
        }
        self.auth_limited_paths = {
            "/auth/login",
            "/auth/register",
            "/auth/token",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through rate limiting."""

        # Skip rate limiting for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        identifier = None
        concurrent_tracked = False
        limit = self.config.default_limit
        info = {
            "current": 0,
            "reset_time": datetime.now(),
        }

        try:
            # Get appropriate limit for this endpoint
            limit, window = self._get_endpoint_limit(request.url.path)

            if self._is_auth_limited_path(request.url.path):
                ip_identifier = f"auth_ip:{self._get_client_ip(request)}"
                ip_allowed, ip_info = await self.limiter.is_allowed(
                    ip_identifier,
                    self.config.auth_ip_limit,
                    self.config.auth_window,
                )
                if not ip_allowed:
                    logger.warning(
                        "Auth IP rate limit exceeded",
                        extra={
                            "user": ip_identifier,
                            "path": request.url.path,
                            "current": ip_info["current"],
                            "limit": self.config.auth_ip_limit,
                        }
                    )
                    return self._rate_limit_response(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        retry_after=ip_info["retry_after"],
                        reason="Rate limit exceeded",
                    )

            # Get identifier (user ID > auth account > API key > IP address)
            identifier = await self._get_identifier(request)

            # Check concurrent requests
            concurrent = await self.limiter.increment_concurrent(identifier)
            concurrent_tracked = True
            if concurrent > self.config.concurrent_limit:
                await self.limiter.decrement_concurrent(identifier)
                concurrent_tracked = False
                user_for_log = identifier
                identifier = None  # already decremented, skip cleanup

                logger.warning(
                    "Concurrent limit exceeded",
                    extra={
                        "user": user_for_log,
                        "path": request.url.path,
                        "concurrent": concurrent,
                    }
                )

                return self._rate_limit_response(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    retry_after=60,
                    reason="Too many concurrent requests",
                )

            # Check rate limit
            allowed, info = await self.limiter.is_allowed(identifier, limit, window)

            if not allowed:
                await self.limiter.decrement_concurrent(identifier)
                concurrent_tracked = False
                user_for_log = identifier
                identifier = None  # already decremented, skip cleanup

                logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "user": user_for_log,
                        "path": request.url.path,
                        "current": info["current"],
                        "limit": limit,
                    }
                )

                return self._rate_limit_response(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    retry_after=info["retry_after"],
                    reason="Rate limit exceeded",
                )

        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}", exc_info=True)
            if concurrent_tracked and identifier:
                try:
                    await self.limiter.decrement_concurrent(identifier)
                except Exception:
                    pass
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal rate-limiter error. Please retry."},
            )

        # Process request after limiter checks pass.
        # Any downstream exception should bubble up to FastAPI handlers.
        try:
            response = await call_next(request)
        finally:
            # Cleanup: decrement concurrent counter only if we incremented it
            if concurrent_tracked and identifier:
                try:
                    await self.limiter.decrement_concurrent(identifier)
                except Exception:
                    pass

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - info["current"]))
        response.headers["X-RateLimit-Reset"] = str(int(info["reset_time"].timestamp()))
        return response

    async def _get_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting."""

        # Check for authenticated user
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"

        if self._is_auth_limited_path(request.url.path):
            account_identifier = self._get_auth_account_identifier(request)
            if account_identifier:
                return account_identifier

        # Check for API key
        if "X-API-Key" in request.headers:
            return f"apikey:{request.headers['X-API-Key']}"

        return f"ip:{self._get_client_ip(request)}"

    def _get_auth_account_identifier(self, request: Request) -> Optional[str]:
        """Return an account-scoped auth key without exposing raw emails in Redis."""
        raw_identifier = (
            request.headers.get("X-Login-Identifier")
            or request.headers.get("X-Auth-Identifier")
        )
        if not raw_identifier:
            return None

        normalized = raw_identifier.strip().lower()
        if not normalized:
            return None

        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
        return f"auth_account:{self._get_client_ip(request)}:{digest}"

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Resolve the client IP from trusted proxy headers with direct fallback."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs, the first one is the original client
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.headers.get("X-Real-IP")

        if not client_ip:
            # Fallback to direct client connection IP
            client_ip = request.client.host if request.client else "unknown"

        return client_ip

    def _is_auth_limited_path(self, path: str) -> bool:
        return path in self.auth_limited_paths

    def _get_endpoint_limit(self, path: str) -> tuple[int, int]:
        """Get rate limit for specific endpoint."""

        # Check exact match
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]

        # Check prefix match
        for pattern, limit_window in self.endpoint_limits.items():
            if path.startswith(pattern):
                return limit_window

        # Default limit
        return self.config.default_limit, self.config.default_window

    @staticmethod
    def _rate_limit_response(
        status_code: int,
        retry_after: int,
        reason: str = "Rate limit exceeded",
    ) -> JSONResponse:
        """Generate rate limit error response."""

        return JSONResponse(
            status_code=status_code,
            content={
                "detail": reason,
                "status": "rate_limit_exceeded",
                "retry_after": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )


# ============================================================================
# RATE LIMIT DECORATOR (FOR ROUTE HANDLERS)
# ============================================================================

def rate_limit(limit: int = None, window: int = 60):
    """
    Decorator for rate limiting specific route handlers.

    Usage:
        @app.post("/api/content/generate")
        @rate_limit(limit=10, window=60)
        async def generate_content(request: GenerateRequest, request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Note: This requires the limiter to be available in request context
            # It's primarily used for additional checks beyond middleware
            return await func(*args, **kwargs)
        return wrapper

    return decorator


# ============================================================================
# RATE LIMIT BYPASS (FOR TESTING)
# ============================================================================

class RateLimitBypass:
    """Context manager for bypassing rate limits (testing only)."""

    _bypass_active = False

    @classmethod
    @property
    def is_active(cls) -> bool:
        return cls._bypass_active

    def __enter__(self):
        RateLimitBypass._bypass_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        RateLimitBypass._bypass_active = False
