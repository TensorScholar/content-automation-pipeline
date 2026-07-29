"""
Advanced Redis Client for Multi-Tier Caching Infrastructure
=============================================================

Provides type-safe, async Redis operations optimized for:
- Embedding vector storage (binary serialization)
- LLM response caching (content-addressed)
- Semantic deduplication via LSH
- Connection health monitoring
- Circuit breaker pattern for fault tolerance

Architecture: Connection pool with automatic failover and
transparent serialization layer.
"""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncGenerator, Awaitable, Dict, List, Optional, TypeVar, Union, cast

import numpy as np
import redis.asyncio as aioredis
from loguru import logger
from pydantic import Field
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError

from config.settings import settings
from core.exceptions import CacheError, InfrastructureError

T = TypeVar("T")


async def _await_redis(value: Awaitable[T] | T) -> T:
    """Narrow redis-py's sync/async stub union at the async client boundary."""
    return await cast(Awaitable[T], value)


class RedisConnectionPool:
    """
    Singleton connection pool manager with health monitoring.

    Implements exponential backoff reconnection strategy and
    circuit breaker pattern to prevent cascade failures.

    Note: Not a singleton to support 'solo' worker mode with multiple event loops.
    """

    def __init__(self):
        self._pool: Optional[ConnectionPool] = None
        self._circuit_breaker_open = False
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._backoff_multiplier = 1.0
        self._max_backoff = 300  # 5 minutes max backoff

    async def initialize(self) -> None:
        """Initialize Redis connection pool with optimized parameters."""
        try:
            self._pool = ConnectionPool.from_url(
                str(settings.redis.url),
                encoding="utf-8",
                decode_responses=False,  # Handle binary data manually
                max_connections=settings.redis.max_connections,
                socket_timeout=settings.redis.socket_timeout,
                socket_connect_timeout=settings.redis.socket_connect_timeout,
                socket_keepalive=True,
                health_check_interval=30,
            )

            # Verify connectivity
            async with self.get_connection() as conn:
                await conn.ping()

            logger.info("Redis connection pool initialized successfully")
            self._circuit_breaker_open = False
            self._failure_count = 0

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection pool: {e}")
            raise InfrastructureError(f"Redis initialization failed: {e}")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Redis, None]:
        """
        Context manager for acquiring Redis connections with circuit breaker.

        Yields:
            Redis connection from pool

        Raises:
            CacheError: When circuit breaker is open or connection fails
        """
        if self._circuit_breaker_open:
            current_time = asyncio.get_event_loop().time()
            time_since_failure = current_time - (self._last_failure_time or 0)
            # IMPROVED: More gradual backoff - start with 5s instead of 60s
            backoff_time = min(5 * self._backoff_multiplier, self._max_backoff)

            if time_since_failure < backoff_time:
                raise CacheError(
                    f"Circuit breaker open: Redis unavailable. "
                    f"Retry in {backoff_time - time_since_failure:.1f}s"
                )
            else:
                # Attempt to close circuit breaker with exponential backoff
                logger.info(f"Attempting to close Redis circuit breaker (backoff: {backoff_time}s)")
                self._circuit_breaker_open = False
                # Don't reset failure count immediately - give it a chance to prove stability
                # self._failure_count = 0  # Removed - reset only on successful operation

        if self._pool is None:
            await self.initialize()

        connection = None
        try:
            connection = Redis(connection_pool=self._pool)
            yield connection

            # SUCCESS: Reset circuit breaker state
            if self._failure_count > 0:
                logger.info(f"Redis connection successful, resetting circuit breaker")
            self._failure_count = 0
            self._backoff_multiplier = 1.0
            self._circuit_breaker_open = False

        except (ConnectionError, TimeoutError) as e:
            self._failure_count += 1
            self._last_failure_time = asyncio.get_event_loop().time()

            # IMPROVED: Require 5 consecutive failures before opening circuit (was 3)
            # This prevents temporary network blips from taking down Redis
            if self._failure_count >= 5:
                self._circuit_breaker_open = True
                # IMPROVED: More gradual backoff increase - 1.5x instead of 2x
                self._backoff_multiplier = min(self._backoff_multiplier * 1.5, 16)
                logger.error(
                    f"Circuit breaker opened after {self._failure_count} failures. "
                    f"Backoff multiplier: {self._backoff_multiplier}x"
                )
            else:
                logger.warning(
                    f"Redis connection error ({self._failure_count}/5): {e}. "
                    f"Will open circuit breaker after {5 - self._failure_count} more failures."
                )

            raise CacheError(f"Redis connection error: {e}")

        finally:
            if connection:
                await connection.aclose()

    async def close(self) -> None:
        """Gracefully close connection pool."""
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
            logger.info("Redis connection pool closed")


class RedisClient:
    """
    High-level Redis client with semantic caching capabilities.

    Features:
    - Type-safe operations with automatic serialization
    - Embedding vector storage (NumPy binary format)
    - Content-addressed LLM response caching
    - Hash-based metadata storage
    - TTL management with intelligent defaults
    """

    def __init__(self):
        self._pool = RedisConnectionPool()
        self._default_ttl = 86400 * 30  # 30 days

    async def initialize(self) -> None:
        """Initialize Redis client and verify connectivity."""
        await self._pool.initialize()

    # =========================================================================
    # EMBEDDING OPERATIONS
    # =========================================================================

    async def store_embedding(
        self,
        key: str,
        embedding: np.ndarray,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store embedding vector in binary format for space efficiency.

        Args:
            key: Storage key (e.g., "emb:rule:uuid")
            embedding: NumPy array or list of floats
            ttl: Time-to-live in seconds (default: 30 days)

        Returns:
            True if stored successfully
        """
        try:
            # Convert list to numpy array if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)

            # Serialize to bytes (more compact than JSON)
            serialized = embedding.astype(np.float32).tobytes()

            async with self._pool.get_connection() as conn:
                await conn.set(key, serialized, ex=ttl or self._default_ttl)

            logger.debug(f"Stored embedding: {key} ({embedding.shape})")
            return True

        except Exception as e:
            logger.error(f"Failed to store embedding {key}: {e}")
            raise CacheError(f"Embedding storage failed: {e}")

    async def get_embedding(self, key: str, shape: tuple = (384,)) -> Optional[np.ndarray]:
        """
        Retrieve embedding vector from cache.

        Args:
            key: Storage key
            shape: Expected embedding shape for reconstruction

        Returns:
            NumPy array or None if not found
        """
        try:
            async with self._pool.get_connection() as conn:
                data = await conn.get(key)

            if data is None:
                return None

            # Deserialize from bytes
            embedding = np.frombuffer(data, dtype=np.float32).reshape(shape)
            logger.debug(f"Retrieved embedding: {key}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to retrieve embedding {key}: {e}")
            return None

    async def store_embeddings_batch(
        self,
        embeddings: Dict[str, np.ndarray],
        ttl: Optional[int] = None,
    ) -> int:
        """
        Store multiple embeddings atomically using pipeline.

        Args:
            embeddings: Dict mapping keys to embedding arrays
            ttl: Time-to-live in seconds

        Returns:
            Number of embeddings stored successfully
        """
        try:
            async with self._pool.get_connection() as conn:
                pipe = conn.pipeline()

                for key, embedding in embeddings.items():
                    serialized = embedding.astype(np.float32).tobytes()
                    pipe.set(key, serialized, ex=ttl or self._default_ttl)

                await pipe.execute()

            logger.info(f"Stored {len(embeddings)} embeddings in batch")
            return len(embeddings)

        except Exception as e:
            logger.error(f"Batch embedding storage failed: {e}")
            raise CacheError(f"Batch storage failed: {e}")

    # =========================================================================
    # LLM RESPONSE CACHING
    # =========================================================================

    @staticmethod
    def _compute_prompt_hash(
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Compute content-addressed hash for prompt caching.

        Uses SHA-256 for collision resistance. Hash includes all
        parameters that affect generation output.
        """
        content = f"{prompt}|{model}|{temperature:.3f}|{max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get_cached_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached LLM response if exists.

        Returns:
            Dict with 'response', 'tokens_used', 'created_at' or None
        """
        prompt_hash = self._compute_prompt_hash(prompt, model, temperature, max_tokens)
        key = f"llm_cache:{prompt_hash}"

        try:
            async with self._pool.get_connection() as conn:
                data = await conn.get(key)

            if data is None:
                logger.debug(f"Cache miss: {prompt_hash[:16]}...")
                return None

            # Deserialize cached response
            cached = json.loads(data)

            # Update access metadata
            await self._update_cache_access(key)

            logger.info(f"Cache hit: {prompt_hash[:16]}... (saved {cached['tokens_used']} tokens)")
            return cached

        except Exception as e:
            logger.error(f"Cache retrieval failed for {prompt_hash[:16]}: {e}")
            return None

    async def cache_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response: str,
        tokens_used: int,
    ) -> bool:
        """
        Cache LLM response with metadata.

        Args:
            prompt: Original prompt text
            model: Model identifier
            temperature: Generation temperature
            max_tokens: Max tokens parameter
            response: Generated response text
            tokens_used: Actual tokens consumed

        Returns:
            True if cached successfully
        """
        prompt_hash = self._compute_prompt_hash(prompt, model, temperature, max_tokens)
        key = f"llm_cache:{prompt_hash}"

        cache_object = {
            "response": response,
            "tokens_used": tokens_used,
            "model": model,
            "temperature": temperature,
            "created_at": asyncio.get_event_loop().time(),
            "access_count": 1,
        }

        try:
            serialized = json.dumps(cache_object)

            async with self._pool.get_connection() as conn:
                await conn.set(key, serialized, ex=self._default_ttl)

            logger.debug(f"Cached LLM response: {prompt_hash[:16]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
            return False

    async def _update_cache_access(self, key: str) -> None:
        """Update access count and last accessed time for cache entry."""
        try:
            async with self._pool.get_connection() as conn:
                data = await conn.get(key)
                if data:
                    cached = json.loads(data)
                    cached["access_count"] = cached.get("access_count", 0) + 1
                    cached["last_accessed"] = asyncio.get_event_loop().time()

                    serialized = json.dumps(cached)
                    ttl = await conn.ttl(key)
                    await conn.set(key, serialized, ex=max(ttl, 3600))
        except Exception as e:
            logger.warning(f"Failed to update cache access metadata: {e}")

    # =========================================================================
    # LIST OPERATIONS (for queues and audit logs)
    # =========================================================================

    async def lpush(self, key: str, *values: Any) -> int:
        """Prepend one or multiple values to a list."""
        try:
            async with self._pool.get_connection() as conn:
                return await _await_redis(conn.lpush(key, *values))
        except Exception as e:
            logger.error(f"Failed lpush on {key}: {e}")
            raise CacheError(f"lpush failed: {e}")

    async def ltrim(self, key: str, start: int, stop: int) -> bool:
        """Trim an existing list so that it will contain only the specified range of elements."""
        try:
            async with self._pool.get_connection() as conn:
                result = await _await_redis(conn.ltrim(key, start, stop))
            return bool(result)
        except Exception as e:
            logger.error(f"Failed ltrim on {key}: {e}")
            raise CacheError(f"ltrim failed: {e}")

    async def lrange(self, key: str, start: int, stop: int) -> List[Any]:
        """Get a range of elements from a list."""
        try:
            async with self._pool.get_connection() as conn:
                return await _await_redis(conn.lrange(key, start, stop))
        except Exception as e:
            logger.error(f"Failed lrange on {key}: {e}")
            raise CacheError(f"lrange failed: {e}")

    # =========================================================================
    # GENERIC KEY-VALUE OPERATIONS
    # =========================================================================

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        ex: Optional[int] = None,  # Standard Redis parameter for expiry
        nx: bool = False,  # Standard Redis parameter for set-if-not-exists
    ) -> bool:
        """
        Store arbitrary Python object with automatic serialization.

        Args:
            key: Storage key
            value: Any picklable Python object
            ttl: Time-to-live in seconds (legacy parameter)
            ex: Expiry time in seconds (standard Redis parameter)
            nx: Only set if key doesn't exist (standard Redis parameter)

        Returns:
            True if stored successfully, False if nx=True and key exists
        """
        try:
            serialized = json.dumps(value)

            # Use ex if provided, otherwise fall back to ttl
            expiry = ex if ex is not None else (ttl or self._default_ttl)

            async with self._pool.get_connection() as conn:
                if nx:
                    # SET with NX option - only set if key doesn't exist
                    result = await conn.set(key, serialized, ex=expiry, nx=True)
                    return result is not None  # Returns None if key exists
                else:
                    await conn.set(key, serialized, ex=expiry)
                    return True

        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            raise CacheError(f"Set operation failed: {e}")

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve and deserialize value from cache.

        Args:
            key: Storage key

        Returns:
            Deserialized Python object or None if not found
        """
        try:
            async with self._pool.get_connection() as conn:
                data = await conn.get(key)

            if data is None:
                return None

            return json.loads(data)

        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            async with self._pool.get_connection() as conn:
                result = await conn.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete matching cache keys without blocking Redis with KEYS."""
        deleted = 0
        try:
            async with self._pool.get_connection() as conn:
                async for key in conn.scan_iter(match=pattern, count=100):
                    deleted += int(await conn.delete(key))
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete keys matching {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            async with self._pool.get_connection() as conn:
                result = await conn.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Atomically increment counter."""
        try:
            async with self._pool.get_connection() as conn:
                result = await conn.incrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Failed to increment key {key}: {e}")
            raise CacheError(f"Increment failed: {e}")

    async def incr(self, key: str, amount: int = 1) -> int:
        """Alias for increment to mirror redis-py interface used by middleware."""
        return await self.increment(key, amount)

    async def decr(self, key: str, amount: int = 1) -> int:
        """Atomically decrement counter."""
        try:
            async with self._pool.get_connection() as conn:
                result = await conn.decrby(key, amount)
            return result
        except Exception as e:
            logger.error(f"Failed to decrement key {key}: {e}")
            raise CacheError(f"Decrement failed: {e}")

    async def ttl(self, key: str) -> int:
        """Get time-to-live for a key."""
        try:
            async with self._pool.get_connection() as conn:
                return await conn.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            raise CacheError(f"TTL retrieval failed: {e}")

    # =========================================================================
    # SORTED SET OPERATIONS (for rate limiting)
    # =========================================================================

    async def zremrangebyscore(self, key: str, min_score: int, max_score: int) -> int:
        """Remove all members in a sorted set within the given scores."""
        try:
            async with self._pool.get_connection() as conn:
                return await conn.zremrangebyscore(key, min_score, max_score)
        except Exception as e:
            logger.error(f"Failed zremrangebyscore on {key}: {e}")
            raise CacheError(f"zremrangebyscore failed: {e}")

    async def zcard(self, key: str) -> int:
        """Get the number of members in a sorted set."""
        try:
            async with self._pool.get_connection() as conn:
                return await conn.zcard(key)
        except Exception as e:
            logger.error(f"Failed zcard on {key}: {e}")
            raise CacheError(f"zcard failed: {e}")

    async def zrange_withscores(self, key: str, start: int, stop: int) -> list:
        """Return a range of members with their scores in a sorted set."""
        try:
            async with self._pool.get_connection() as conn:
                return await conn.zrange(key, start, stop, withscores=True)
        except Exception as e:
            logger.error(f"Failed zrange on {key}: {e}")
            raise CacheError(f"zrange failed: {e}")

    async def zadd(self, key: str, mapping: Dict[str, int]) -> int:
        """Add one or more members to a sorted set, or update score."""
        try:
            async with self._pool.get_connection() as conn:
                return await conn.zadd(key, mapping)
        except Exception as e:
            logger.error(f"Failed zadd on {key}: {e}")
            raise CacheError(f"zadd failed: {e}")

    async def expire(self, key: str, seconds: int) -> bool:
        """Set a timeout on key. After the timeout has expired, the key will be deleted."""
        try:
            async with self._pool.get_connection() as conn:
                return await conn.expire(key, seconds)
        except Exception as e:
            logger.error(f"Failed expire on {key}: {e}")
            raise CacheError(f"expire failed: {e}")

    async def setex(self, key: str, seconds: int, value: Any) -> bool:
        """Set key to hold string value with expiration in seconds."""
        try:
            serialized = json.dumps(value)
            async with self._pool.get_connection() as conn:
                await conn.setex(key, seconds, serialized)
            return True
        except Exception as e:
            logger.error(f"Failed setex on {key}: {e}")
            raise CacheError(f"setex failed: {e}")

    # =========================================================================
    # HASH OPERATIONS (for structured metadata)
    # =========================================================================

    async def hset(
        self,
        key: str,
        field: str,
        value: Any,
    ) -> bool:
        """Set field in hash."""
        try:
            serialized = json.dumps(value)
            async with self._pool.get_connection() as conn:
                await _await_redis(conn.hset(key, field, serialized))
            return True
        except Exception as e:
            logger.error(f"Failed to set hash field {key}.{field}: {e}")
            return False

    async def hget(self, key: str, field: str) -> Optional[Any]:
        """Get field from hash."""
        try:
            async with self._pool.get_connection() as conn:
                data = await _await_redis(conn.hget(key, field))

            if data is None:
                return None

            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get hash field {key}.{field}: {e}")
            return None

    async def hgetall(self, key: str) -> Dict[str, Any]:
        """Get all fields from hash."""
        try:
            async with self._pool.get_connection() as conn:
                data = await _await_redis(conn.hgetall(key))

            # Deserialize all values
            return {field.decode(): json.loads(value) for field, value in data.items()}
        except Exception as e:
            logger.error(f"Failed to get all hash fields for {key}: {e}")
            return {}

    # =========================================================================
    # UTILITY & MONITORING
    # =========================================================================

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Retrieve cache statistics for monitoring."""
        try:
            async with self._pool.get_connection() as conn:
                info = await conn.info("stats")
                memory = await conn.info("memory")

            return {
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._compute_hit_rate(
                    info.get("keyspace_hits", 0), info.get("keyspace_misses", 0)
                ),
                "used_memory_mb": memory.get("used_memory", 0) / (1024 * 1024),
                "used_memory_peak_mb": memory.get("used_memory_peak", 0) / (1024 * 1024),
                "max_memory_mb": memory.get("maxmemory", 0) / (1024 * 1024),
            }
        except Exception as e:
            logger.error(f"Failed to retrieve cache stats: {e}")
            return {}

    @staticmethod
    def _compute_hit_rate(hits: int, misses: int) -> float:
        """Compute cache hit rate percentage."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    async def flush_cache(self, pattern: Optional[str] = None) -> int:
        """
        Flush cache entries matching pattern.

        Args:
            pattern: Redis pattern (e.g., "llm_cache:*") or None for all

        Returns:
            Number of keys deleted
        """
        try:
            async with self._pool.get_connection() as conn:
                if pattern:
                    keys = await conn.keys(pattern)
                    if keys:
                        deleted = await conn.delete(*keys)
                        logger.warning(f"Flushed {deleted} keys matching '{pattern}'")
                        return deleted
                    return 0
                else:
                    await conn.flushdb()
                    logger.warning("Flushed entire Redis database")
                    return -1  # Unknown count
        except Exception as e:
            logger.error(f"Failed to flush cache: {e}")
            raise CacheError(f"Flush operation failed: {e}")

    async def close(self) -> None:
        """Close Redis connection pool."""
        await self._pool.close()

    async def ping(self) -> bool:
        """Ping Redis to verify connectivity."""
        try:
            async with self._pool.get_connection() as conn:
                await conn.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    async def health_check(self) -> bool:
        """Compatibility health check used by startup and monitoring code."""
        return await self.ping()

    async def info(self) -> dict:
        """Return Redis server INFO as a dict (mirrors aioredis/redis-py interface)."""
        try:
            async with self._pool.get_connection() as conn:
                raw = await conn.info()
            return raw if isinstance(raw, dict) else {}
        except Exception as e:
            logger.error(f"Redis info failed: {e}")
            return {}

    async def get_raw_connection(self) -> Redis:
        """Get raw Redis connection for third-party libraries.

        Returns a native aioredis.Redis instance for libraries like
        FastAPILimiter that expect raw Redis client methods (e.g., script_load).

        Note: Caller is responsible for connection lifecycle.
        """
        if self._pool._pool is None:
            await self._pool.initialize()
        return Redis(connection_pool=self._pool._pool)


# Singleton instance for application-wide use
redis_client = RedisClient()
