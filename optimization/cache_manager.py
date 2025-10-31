"""
Cache Manager - Unified Caching Orchestration
==============================================

Orchestrates multi-tier caching strategy:

L1: In-memory (Python dict) - Hot data, microsecond access
L2: Redis - Warm data, millisecond access
L3: Database - Cold data, 10-100ms access

Features:
- Automatic cache warming on startup
- Intelligent invalidation policies
- Hit-rate tracking and optimization
- Circuit breaker for cache failures
- Graceful degradation

Design Philosophy: Cache everything safely, invalidate intelligently.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from loguru import logger

from core.exceptions import CacheError
from infrastructure.redis_client import RedisClient

T = TypeVar("T")


class CacheLevel(str, Enum):
    """Cache tier levels."""

    MEMORY = "memory"  # L1: In-memory
    REDIS = "redis"  # L2: Redis
    DATABASE = "database"  # L3: Database (read-through)


@dataclass
class CacheStats:
    """Cache statistics for monitoring."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    invalidations: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def total_operations(self) -> int:
        """Total cache operations."""
        return self.hits + self.misses + self.sets


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False

        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class CachePolicy:
    """Cache eviction and invalidation policies."""

    @staticmethod
    def should_cache(key: str, value: Any) -> bool:
        """Determine if value should be cached."""
        # Don't cache None or very large objects
        if value is None:
            return False

        # Estimate size (rough)
        try:
            import sys

            size_bytes = sys.getsizeof(value)
            return size_bytes < 1024 * 1024  # 1MB limit
        except:
            return True

    @staticmethod
    def get_ttl(key: str, cache_level: CacheLevel) -> Optional[int]:
        """Determine TTL for key based on level and type."""
        # Embeddings: long TTL
        if key.startswith("emb_") or key.startswith("bp_emb:"):
            return 86400 * 90  # 90 days

        # LLM responses: medium TTL
        if key.startswith("llm_cache:"):
            return 86400 * 30  # 30 days

        # Project data: short TTL (can change frequently)
        if key.startswith("project:"):
            return 3600  # 1 hour

        # Default based on level
        if cache_level == CacheLevel.MEMORY:
            return 300  # 5 minutes
        elif cache_level == CacheLevel.REDIS:
            return 3600  # 1 hour
        else:
            return None  # No expiry for database


class CacheManager:
    """
    Unified cache management system.

    Coordinates all caching operations across memory, Redis, and database.
    Provides single interface with automatic fallback and optimization.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        max_memory_entries: int = 1000,
        metrics_collector: Optional[Any] = None,
    ):
        """
        Initialize cache manager.

        Args:
            redis_client: Redis client instance for L2 cache
            max_memory_entries: Maximum entries in L1 memory cache
            metrics_collector: Optional metrics collector for Prometheus metrics
        """
        # L1: In-memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self.redis_client = redis_client
        self.max_memory_entries = max_memory_entries

        # Statistics tracking
        self.stats_by_level = {
            CacheLevel.MEMORY: CacheStats(),
            CacheLevel.REDIS: CacheStats(),
            CacheLevel.DATABASE: CacheStats(),
        }

        # Global stats
        self.global_stats = CacheStats()

        # Cache warming state
        self._warming_in_progress = False
        self._warm_keys: Set[str] = set()

        # Policy
        self.policy = CachePolicy()

        # Metrics collector
        self.metrics_collector = metrics_collector

        logger.info(f"Cache manager initialized (max memory entries: {max_memory_entries})")

    # =========================================================================
    # UNIFIED GET/SET INTERFACE
    # =========================================================================

    async def get(
        self,
        key: str,
        fallback: Optional[Callable[[], Any]] = None,
    ) -> Optional[Any]:
        """
        Get value from cache with automatic fallback through tiers.

        Process:
        1. Check L1 (memory)
        2. If miss, check L2 (Redis) and promote to L1
        3. If miss, check L3 (database via fallback) and promote
        4. Return None if all miss

        Args:
            key: Cache key
            fallback: Optional async function to compute value on miss

        Returns:
            Cached value or None
        """
        # L1: Memory cache
        value = await self._get_from_memory(key)
        if value is not None:
            self.stats_by_level[CacheLevel.MEMORY].hits += 1
            self.global_stats.hits += 1

            # Record cache hit metrics
            if self.metrics_collector:
                self.metrics_collector.record_cache_hit("memory", "general")

            return value

        self.stats_by_level[CacheLevel.MEMORY].misses += 1

        # Record cache miss metrics
        if self.metrics_collector:
            self.metrics_collector.record_cache_miss("memory", "general")

        # L2: Redis cache
        value = await self._get_from_redis(key)
        if value is not None:
            self.stats_by_level[CacheLevel.REDIS].hits += 1
            self.global_stats.hits += 1

            # Record cache hit metrics
            if self.metrics_collector:
                self.metrics_collector.record_cache_hit("redis", "general")

            # Promote to L1
            await self._set_in_memory(key, value)
            return value

        self.stats_by_level[CacheLevel.REDIS].misses += 1

        # Record cache miss metrics
        if self.metrics_collector:
            self.metrics_collector.record_cache_miss("redis", "general")

        # L3: Fallback (database or computation)
        if fallback:
            try:
                value = await fallback() if asyncio.iscoroutinefunction(fallback) else fallback()

                if value is not None:
                    self.stats_by_level[CacheLevel.DATABASE].hits += 1
                    self.global_stats.hits += 1

                    # Populate caches
                    await self.set(key, value)
                    return value

            except Exception as e:
                logger.error(f"Fallback failed for key {key}: {e}")
                self.stats_by_level[CacheLevel.DATABASE].errors += 1

        # Complete miss
        self.global_stats.misses += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        levels: Optional[List[CacheLevel]] = None,
    ) -> bool:
        """
        Set value in cache across specified levels.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use policy default)
            levels: Which cache levels to write to (None = all)

        Returns:
            True if successfully cached in at least one level
        """
        if not self.policy.should_cache(key, value):
            return False

        # Default to all levels
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS]

        success = False

        # Set in specified levels
        if CacheLevel.MEMORY in levels:
            if await self._set_in_memory(key, value, ttl):
                self.stats_by_level[CacheLevel.MEMORY].sets += 1
                success = True

        if CacheLevel.REDIS in levels:
            if await self._set_in_redis(key, value, ttl):
                self.stats_by_level[CacheLevel.REDIS].sets += 1
                success = True

        if success:
            self.global_stats.sets += 1

        return success

    async def delete(
        self,
        key: str,
        levels: Optional[List[CacheLevel]] = None,
    ) -> bool:
        """
        Delete key from cache across specified levels.

        Args:
            key: Cache key
            levels: Which levels to delete from (None = all)

        Returns:
            True if deleted from at least one level
        """
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS]

        success = False

        if CacheLevel.MEMORY in levels:
            if key in self._memory_cache:
                del self._memory_cache[key]
                self.stats_by_level[CacheLevel.MEMORY].invalidations += 1
                success = True

        if CacheLevel.REDIS in levels:
            if await self.redis_client.delete(key):
                self.stats_by_level[CacheLevel.REDIS].invalidations += 1
                success = True

        if success:
            self.global_stats.invalidations += 1

        return success

    async def invalidate_pattern(
        self,
        pattern: str,
        levels: Optional[List[CacheLevel]] = None,
    ) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "project:*", "llm_cache:*")
            levels: Which levels to invalidate (None = all)

        Returns:
            Number of keys invalidated
        """
        if levels is None:
            levels = [CacheLevel.MEMORY, CacheLevel.REDIS]

        count = 0

        # Memory cache
        if CacheLevel.MEMORY in levels:
            import fnmatch

            matching_keys = [k for k in self._memory_cache.keys() if fnmatch.fnmatch(k, pattern)]

            for key in matching_keys:
                del self._memory_cache[key]
                count += 1

            self.stats_by_level[CacheLevel.MEMORY].invalidations += count

        # Redis cache
        if CacheLevel.REDIS in levels:
            redis_count = await self.redis_client.flush_cache(pattern)
            if redis_count > 0:
                count += redis_count
                self.stats_by_level[CacheLevel.REDIS].invalidations += redis_count

        self.global_stats.invalidations += count

        logger.info(f"Invalidated {count} keys matching pattern: {pattern}")
        return count

    # =========================================================================
    # L1: MEMORY CACHE OPERATIONS
    # =========================================================================

    async def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from L1 memory cache."""
        if key not in self._memory_cache:
            return None

        entry = self._memory_cache[key]

        # Check expiry
        if entry.is_expired():
            del self._memory_cache[key]
            return None

        # Update access metadata
        entry.touch()

        return entry.value

    async def _set_in_memory(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in L1 memory cache."""
        try:
            # Evict if at capacity
            if len(self._memory_cache) >= self.max_memory_entries:
                await self._evict_lru()

            # Get TTL from policy if not specified
            if ttl is None:
                ttl = self.policy.get_ttl(key, CacheLevel.MEMORY)

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl_seconds=ttl,
            )

            self._memory_cache[key] = entry
            return True

        except Exception as e:
            logger.error(f"Failed to set memory cache for key {key}: {e}")
            self.stats_by_level[CacheLevel.MEMORY].errors += 1
            return False

    async def _evict_lru(self) -> None:
        """Evict least recently used entry from memory cache."""
        if not self._memory_cache:
            return

        # Find LRU entry
        lru_key = min(self._memory_cache.keys(), key=lambda k: self._memory_cache[k].accessed_at)

        del self._memory_cache[lru_key]
        logger.debug(f"Evicted LRU entry: {lru_key}")

    # =========================================================================
    # L2: REDIS CACHE OPERATIONS
    # =========================================================================

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from L2 Redis cache."""
        try:
            return await self.redis_client.get(key)
        except Exception as e:
            logger.warning(f"Redis get failed for key {key}: {e}")
            self.stats_by_level[CacheLevel.REDIS].errors += 1
            return None

    async def _set_in_redis(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in L2 Redis cache."""
        try:
            # Get TTL from policy if not specified
            if ttl is None:
                ttl = self.policy.get_ttl(key, CacheLevel.REDIS)

            return await self.redis_client.set(key, value, ttl=ttl)

        except Exception as e:
            logger.warning(f"Redis set failed for key {key}: {e}")
            self.stats_by_level[CacheLevel.REDIS].errors += 1
            return False

    # =========================================================================
    # CACHE WARMING
    # =========================================================================

    async def warm_cache(
        self,
        warm_keys: List[Tuple[str, Callable]],
    ) -> int:
        """
        Warm cache with frequently accessed data.

        Args:
            warm_keys: List of (key, loader_function) tuples

        Returns:
            Number of keys successfully warmed
        """
        if self._warming_in_progress:
            logger.warning("Cache warming already in progress")
            return 0

        self._warming_in_progress = True
        logger.info(f"Starting cache warming for {len(warm_keys)} keys...")

        warmed = 0

        try:
            for key, loader in warm_keys:
                try:
                    # Load value
                    value = await loader() if asyncio.iscoroutinefunction(loader) else loader()

                    # Cache it
                    if await self.set(key, value):
                        self._warm_keys.add(key)
                        warmed += 1

                except Exception as e:
                    logger.warning(f"Failed to warm key {key}: {e}")
                    continue

            logger.info(f"Cache warming complete: {warmed}/{len(warm_keys)} keys warmed")
            return warmed

        finally:
            self._warming_in_progress = False

    # =========================================================================
    # STATISTICS & MONITORING
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_size = len(self._memory_cache)
        memory_utilization = (memory_size / self.max_memory_entries) * 100

        return {
            "global": {
                "hit_rate": self.global_stats.hit_rate,
                "total_operations": self.global_stats.total_operations,
                "hits": self.global_stats.hits,
                "misses": self.global_stats.misses,
                "sets": self.global_stats.sets,
                "invalidations": self.global_stats.invalidations,
            },
            "by_level": {
                level.value: {
                    "hit_rate": stats.hit_rate,
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "sets": stats.sets,
                    "errors": stats.errors,
                }
                for level, stats in self.stats_by_level.items()
            },
            "memory_cache": {
                "size": memory_size,
                "max_size": self.max_memory_entries,
                "utilization_percent": memory_utilization,
                "warmed_keys": len(self._warm_keys),
            },
        }

    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.global_stats = CacheStats()
        for level in self.stats_by_level:
            self.stats_by_level[level] = CacheStats()

        logger.info("Cache statistics reset")

    async def get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis-specific statistics."""
        return await self.redis_client.get_cache_stats()

    # =========================================================================
    # MAINTENANCE
    # =========================================================================

    async def cleanup_expired(self) -> int:
        """
        Clean up expired entries from memory cache.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self.__memory_cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._memory_cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)


async def optimize(self) -> Dict[str, int]:
    """
    Optimize cache performance.

    Actions:
    - Remove expired entries
    - Evict cold data (low access count)
    - Compact memory

    Returns:
        Dict with optimization metrics
    """
    logger.info("Starting cache optimization...")

    # Remove expired
    expired_removed = await self.cleanup_expired()

    # Identify cold data (not accessed in last hour)
    cutoff_time = datetime.utcnow() - timedelta(hours=1)
    cold_keys = [
        key
        for key, entry in self._memory_cache.items()
        if entry.accessed_at < cutoff_time and entry.access_count < 5
    ]

    # Remove bottom 20% of cold data if cache is >80% full
    memory_utilization = len(self._memory_cache) / self.max_memory_entries
    cold_removed = 0

    if memory_utilization > 0.8 and cold_keys:
        # Sort by access count (ascending)
        cold_keys.sort(key=lambda k: self._memory_cache[k].access_count)

        # Remove bottom 20%
        remove_count = max(1, len(cold_keys) // 5)
        for key in cold_keys[:remove_count]:
            del self._memory_cache[key]
            cold_removed += 1

    logger.info(f"Optimization complete: expired={expired_removed}, cold={cold_removed}")

    return {
        "expired_removed": expired_removed,
        "cold_removed": cold_removed,
        "final_size": len(self._memory_cache),
    }


async def clear_all(self) -> None:
    """Clear all caches (DANGEROUS - use with caution)."""
    logger.warning("Clearing ALL caches...")

    # Clear memory
    self._memory_cache.clear()
    self._warm_keys.clear()

    # Clear Redis
    await self.redis_client.flush_cache()

    # Reset stats
    self.reset_statistics()

    logger.warning("All caches cleared")
