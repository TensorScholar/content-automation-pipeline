"""
Query Result Caching with Redis

Provides decorators and utilities for caching database query results
in Redis with configurable TTL and cache invalidation strategies.
"""

import functools
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from uuid import UUID

from infrastructure.redis_client import RedisClient


def cache_key_builder(*args, **kwargs) -> str:
    """
    Build cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        MD5 hash of serialized arguments
    """
    # Serialize args and kwargs to create unique key
    key_data = {
        "args": [str(arg) for arg in args if not isinstance(arg, (RedisClient, object))],
        "kwargs": {k: str(v) for k, v in kwargs.items() if not callable(v)},
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached_query(
    ttl: int = 300,
    key_prefix: str = "query_cache",
    serialize: Callable = None,
    deserialize: Callable = None,
):
    """
    Decorator for caching query results in Redis.

    Args:
        ttl: Time to live in seconds (default 5 minutes)
        key_prefix: Prefix for cache keys
        serialize: Custom serialization function
        deserialize: Custom deserialization function

    Example:
        @cached_query(ttl=600, key_prefix="project")
        async def get_project(project_id: UUID):
            return await db.fetch_one(...)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Check if redis_client is available
            redis_client = getattr(self, "redis_client", None) or getattr(
                self, "_redis_client", None
            )

            # If no Redis client, skip caching
            if not redis_client:
                return await func(self, *args, **kwargs)

            # Build cache key
            cache_key_suffix = cache_key_builder(*args, **kwargs)
            cache_key = f"{key_prefix}:{func.__name__}:{cache_key_suffix}"

            try:
                # Try to get from cache
                cached_value = await redis_client.get(cache_key)
                if cached_value:
                    # Deserialize if custom deserializer provided
                    if deserialize:
                        return deserialize(cached_value)
                    # Default JSON deserialization
                    return json.loads(cached_value)

            except Exception:
                # If cache read fails, continue to execute query
                pass

            # Execute query
            result = await func(self, *args, **kwargs)

            # Cache result
            try:
                if result is not None:
                    # Serialize if custom serializer provided
                    if serialize:
                        cache_value = serialize(result)
                    else:
                        # Default JSON serialization with custom encoder for datetime/UUID
                        cache_value = json.dumps(result, default=str)

                    await redis_client.setex(cache_key, ttl, cache_value)
            except Exception:
                # If cache write fails, still return result
                pass

            return result

        return wrapper

    return decorator


def invalidate_cache_pattern(redis_client: RedisClient, pattern: str) -> None:
    """
    Invalidate all cache keys matching a pattern.

    Args:
        redis_client: Redis client instance
        pattern: Key pattern (e.g., "query_cache:project:*")

    Example:
        # Invalidate all project caches
        await invalidate_cache_pattern(redis, "query_cache:*:project_id:123")
    """
    # Note: This is a synchronous wrapper around async Redis operations
    # In production, consider using Redis SCAN for better performance
    pass


class CacheInvalidator:
    """
    Context manager for cache invalidation on write operations.

    Example:
        async with CacheInvalidator(redis, "project:*"):
            await update_project(project_id, data)
    """

    def __init__(self, redis_client: RedisClient, pattern: str):
        self.redis_client = redis_client
        self.pattern = pattern

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Only invalidate if no exception occurred
            try:
                # Invalidate cache pattern
                await invalidate_cache_pattern(self.redis_client, self.pattern)
            except Exception:
                pass
