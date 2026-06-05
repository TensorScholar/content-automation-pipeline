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
    # Sort kwargs to ensure deterministic key regardless of order
    key_data = {
        "args": [str(arg) for arg in args if not isinstance(arg, (RedisClient, object))],
        "kwargs": {k: str(v) for k, v in sorted(kwargs.items()) if not callable(v)},
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


async def invalidate_query_cache(
    redis_client: RedisClient,
    entity_type: str,
    entity_id: Optional[str] = None
) -> int:
    """
    Invalidate query cache for specific entity.

    Args:
        redis_client: Redis client
        entity_type: Type of entity (e.g., "project", "article")
        entity_id: Optional specific ID to invalidate

    Returns:
        Number of keys invalidated
    """
    if entity_id:
        pattern = f"query_cache:*:{entity_type}:{entity_id}*"
    else:
        pattern = f"query_cache:*:{entity_type}:*"

    return await invalidate_cache_pattern(redis_client, pattern)


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
                        # Smart serialization for Pydantic models and lists
                        # Check for Pydantic v2
                        if hasattr(result, "model_dump"):
                            cache_value = json.dumps(result.model_dump(), default=str)
                        # Check for Pydantic v1
                        elif hasattr(result, "dict"):
                            cache_value = json.dumps(result.dict(), default=str)
                        # Check for list of Pydantic models
                        elif isinstance(result, list) and result:
                            if hasattr(result[0], "model_dump"):
                                cache_value = json.dumps([r.model_dump() for r in result], default=str)
                            elif hasattr(result[0], "dict"):
                                cache_value = json.dumps([r.dict() for r in result], default=str)
                            else:
                                cache_value = json.dumps(result, default=str)
                        else:
                            # Default JSON serialization with custom encoder for datetime/UUID
                            cache_value = json.dumps(result, default=str)

                    await redis_client.setex(cache_key, ttl, cache_value)
            except Exception as e:
                # If cache write fails, still return result
                # Log error for debugging if logger available
                pass

            return result

        return wrapper

    return decorator


async def invalidate_cache_pattern(redis_client: RedisClient, pattern: str) -> int:
    """
    Invalidate all cache keys matching a pattern.

    Args:
        redis_client: Redis client instance
        pattern: Key pattern (e.g., "query_cache:project:*")

    Returns:
        Number of keys invalidated

    Example:
        # Invalidate all project caches
        count = await invalidate_cache_pattern(redis, "query_cache:*:project_id:123")
    """
    try:
        # Use Redis SCAN for efficient pattern matching
        keys_deleted = 0
        cursor = 0

        while True:
            # SCAN through keys matching pattern
            cursor, keys = await redis_client.scan(cursor=cursor, match=pattern, count=100)

            if keys:
                # Delete matched keys
                await redis_client.delete(*keys)
                keys_deleted += len(keys)

            # cursor 0 means we've scanned everything
            if cursor == 0:
                break

        return keys_deleted

    except Exception as e:
        # If Redis doesn't support SCAN or other error, try fallback
        try:
            keys = await redis_client.keys(pattern)
            if keys:
                await redis_client.delete(*keys)
                return len(keys)
            return 0
        except Exception:
            # If all fails, return 0
            return 0


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


async def set_query_result(
    redis_client: RedisClient,
    key: str,
    value: Any,
    ttl: int = 300
) -> bool:
    """
    Set a query result in the cache.

    Args:
        redis_client: Redis client instance
        key: Cache key
        value: Data to cache (must be serializable)
        ttl: Time to live in seconds

    Returns:
        True if successful
    """
    try:
        serialized = json.dumps(value, default=str)
        await redis_client.setex(key, ttl, serialized)
        return True
    except Exception:
        # Log error but don't fail
        return False


async def get_query_result(redis_client: RedisClient, key: str) -> Optional[Any]:
    """
    Get a query result from the cache.

    Args:
        redis_client: Redis client instance
        key: Cache key

    Returns:
        Cached data or None
    """
    try:
        cached = await redis_client.get(key)
        if cached:
            return json.loads(cached)
        return None
    except Exception:
        return None
