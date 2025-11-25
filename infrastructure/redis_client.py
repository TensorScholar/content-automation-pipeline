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
from typing import Any, AsyncGenerator, Dict, List, Optional, TypeVar, Union

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


class RedisConnectionPool:
    """
    Singleton connection pool manager with health monitoring.

    Implements exponential backoff reconnection strategy and
    circuit breaker pattern to prevent cascade failures.
    """

    _instance: Optional["RedisConnectionPool"] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "RedisConnectionPool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._pool: Optional[ConnectionPool] = None
            self._circuit_breaker_open = False
            self._failure_count = 0
            self._last_failure_time: Optional[float] = None
            self._backoff_multiplier = 1
            self._max_backoff = 300  # 5 minutes max backoff
            self._initialized = True

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
            backoff_time = min(60 * self._backoff_multiplier, self._max_backoff)
            
            if time_since_failure < backoff_time:
                raise CacheError(
                    f"Circuit breaker open: Redis unavailable. "
                    f"Retry in {backoff_time - time_since_failure:.1f}s"
                )
            else:
                # Attempt to close circuit breaker with exponential backoff
                logger.info(f"Attempting to close Redis circuit breaker (backoff: {backoff_time}s)")
                self._circuit_breaker_open = False
                self._failure_count = 0

        if self._pool is None:
            await self.initialize()

        connection = None
        try:
            connection = Redis(connection_pool=self._pool)
            yield connection
            self._failure_count = 0  # Reset on success
            self._backoff_multiplier = 1  # Reset backoff on success

        except (ConnectionError, TimeoutError) as e:
            self._failure_count += 1
            self._last_failure_time = asyncio.get_event_loop().time()

            if self._failure_count >= 3:
                self._circuit_breaker_open = True
                self._backoff_multiplier = min(self._backoff_multiplier * 2, 16)  # Cap at 16x
                logger.error(
                    f"Circuit breaker opened after {self._failure_count} failures. "
                    f"Backoff multiplier: {self._backoff_multiplier}x"
                )

            raise CacheError(f"Redis connection error: {e}")

        finally:
            if connection:
                await connection.close()

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
            embedding: NumPy array (any shape)
            ttl: Time-to-live in seconds (default: 30 days)

        Returns:
            True if stored successfully
        """
        try:
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
    # GENERIC KEY-VALUE OPERATIONS
    # =========================================================================

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store arbitrary Python object with automatic serialization.

        Args:
            key: Storage key
            value: Any picklable Python object
            ttl: Time-to-live in seconds

        Returns:
            True if stored successfully
        """
        try:
            serialized = json.dumps(value)

            async with self._pool.get_connection() as conn:
                await conn.set(key, serialized, ex=ttl or self._default_ttl)

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
                await conn.hset(key, field, serialized)
            return True
        except Exception as e:
            logger.error(f"Failed to set hash field {key}.{field}: {e}")
            return False

    async def hget(self, key: str, field: str) -> Optional[Any]:
        """Get field from hash."""
        try:
            async with self._pool.get_connection() as conn:
                data = await conn.hget(key, field)

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
                data = await conn.hgetall(key)

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


# Singleton instance for application-wide use
redis_client = RedisClient()
