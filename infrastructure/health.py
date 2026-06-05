"""
Infrastructure Health Check System
===================================
Comprehensive health monitoring for all infrastructure components.
Provides unified health status for Kubernetes probes and monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels for components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class SystemHealth:
    """Aggregated health status of the entire system."""
    status: HealthStatus
    components: List[ComponentHealth]
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "checked_at": self.checked_at.isoformat(),
            "components": [c.to_dict() for c in self.components],
        }


class HealthChecker:
    """
    Centralized health check orchestrator.

    Manages health probes for all infrastructure components and
    provides aggregated health status.
    """

    def __init__(self):
        self._start_time = datetime.utcnow()
        self._checks: Dict[str, Callable] = {}
        self._timeout_seconds: float = 5.0

    def register_check(self, name: str, check_fn: Callable) -> None:
        """
        Register a health check function.

        Args:
            name: Component name
            check_fn: Async function returning ComponentHealth
        """
        self._checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)

    async def check_component(self, name: str) -> ComponentHealth:
        """
        Run health check for a single component.

        Args:
            name: Component name

        Returns:
            ComponentHealth result
        """
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for '{name}'",
            )

        check_fn = self._checks[name]
        start = asyncio.get_event_loop().time()

        try:
            result = await asyncio.wait_for(
                check_fn(),
                timeout=self._timeout_seconds,
            )
            result.latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            return result

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
                message=f"Health check timed out after {self._timeout_seconds}s",
            )
        except Exception as e:
            logger.exception(f"Health check failed for {name}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
                message=str(e),
            )

    async def check_all(self) -> SystemHealth:
        """
        Run all registered health checks concurrently.

        Returns:
            Aggregated SystemHealth
        """
        if not self._checks:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                components=[],
                uptime_seconds=self._get_uptime(),
            )

        # Run all checks concurrently
        tasks = [
            self.check_component(name)
            for name in self._checks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        components: List[ComponentHealth] = []
        for result in results:
            if isinstance(result, ComponentHealth):
                components.append(result)
            elif isinstance(result, Exception):
                components.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                ))

        # Determine overall status
        overall_status = self._aggregate_status(components)

        return SystemHealth(
            status=overall_status,
            components=components,
            uptime_seconds=self._get_uptime(),
        )

    async def liveness_check(self) -> bool:
        """
        Simple liveness check for Kubernetes.

        Returns:
            True if application is alive
        """
        return True

    async def readiness_check(self) -> bool:
        """
        Readiness check for Kubernetes.

        Returns:
            True if application can handle requests
        """
        health = await self.check_all()
        return health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def _get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return (datetime.utcnow() - self._start_time).total_seconds()

    @staticmethod
    def _aggregate_status(components: List[ComponentHealth]) -> HealthStatus:
        """
        Aggregate component statuses into overall status.

        Rules:
        - All healthy -> healthy
        - Any unhealthy -> unhealthy
        - Any degraded (none unhealthy) -> degraded
        - No components -> unknown
        """
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN


# =============================================================================
# COMPONENT-SPECIFIC HEALTH CHECKS
# =============================================================================


async def check_database_health() -> ComponentHealth:
    """Check PostgreSQL database health."""
    from infrastructure.database import get_db_manager

    try:
        manager = get_db_manager()

        if not manager._is_initialized:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database not initialized",
            )

        # Run health check
        await manager.health_check()

        # Get pool status
        pool_status = manager.get_pool_status()

        # Determine health based on pool utilization
        total_connections = pool_status.get("total_connections", 0)
        checked_out = pool_status.get("checked_out", 0)

        if total_connections > 0:
            utilization = checked_out / total_connections
            if utilization > 0.9:
                status = HealthStatus.DEGRADED
                message = f"High pool utilization: {utilization:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = "Database operational"
        else:
            status = HealthStatus.HEALTHY
            message = "Database operational"

        return ComponentHealth(
            name="database",
            status=status,
            message=message,
            details=pool_status,
        )

    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


async def check_redis_health() -> ComponentHealth:
    """Check Redis cache health."""
    from infrastructure.redis_client import redis_client

    try:
        # Ping Redis
        is_healthy = await redis_client.ping()

        if not is_healthy:
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Redis ping failed",
            )

        # Get cache stats
        stats = await redis_client.get_cache_stats()

        # Check memory utilization if available
        used_memory = stats.get("used_memory_mb", 0)
        peak_memory = stats.get("used_memory_peak_mb", 0)

        if peak_memory > 0 and used_memory / peak_memory > 0.9:
            status = HealthStatus.DEGRADED
            message = f"High memory utilization: {used_memory:.1f}MB"
        else:
            status = HealthStatus.HEALTHY
            message = "Redis operational"

        return ComponentHealth(
            name="redis",
            status=status,
            message=message,
            details={
                "hit_rate": stats.get("hit_rate", 0),
                "used_memory_mb": used_memory,
                "total_commands": stats.get("total_commands", 0),
            },
        )

    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


async def check_celery_health() -> ComponentHealth:
    """Check Celery worker health."""
    try:
        from orchestration.celery_app import celery_app

        # Inspect active workers
        inspector = celery_app.control.inspect()

        # This is a blocking call, wrap in executor
        import asyncio
        loop = asyncio.get_event_loop()

        # Get active workers with timeout
        try:
            active = await asyncio.wait_for(
                loop.run_in_executor(None, inspector.active),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            return ComponentHealth(
                name="celery",
                status=HealthStatus.DEGRADED,
                message="Worker inspection timed out",
            )

        if active is None:
            return ComponentHealth(
                name="celery",
                status=HealthStatus.UNHEALTHY,
                message="No workers available",
            )

        worker_count = len(active)
        task_count = sum(len(tasks) for tasks in active.values())

        return ComponentHealth(
            name="celery",
            status=HealthStatus.HEALTHY,
            message=f"{worker_count} workers, {task_count} active tasks",
            details={
                "worker_count": worker_count,
                "active_tasks": task_count,
            },
        )

    except Exception as e:
        return ComponentHealth(
            name="celery",
            status=HealthStatus.DEGRADED,
            message=f"Celery check failed: {e}",
        )


async def check_llm_health() -> ComponentHealth:
    """
    Check LLM provider health by verifying connectivity.

    Performs a minimal generation request ('ping') to ensure
    the configured provider is actually reachable and responding.
    """
    try:
        from config.settings import get_settings
        from infrastructure.llm_client import UnifiedLLMClient

        settings = get_settings()
        client = UnifiedLLMClient()

        # Determine which model to test
        model = settings.llm.primary_model or "gpt-4o-mini"

        start_time = asyncio.get_event_loop().time()

        # Attempt minimal generation
        async with asyncio.timeout(5.0):
            response = await client.generate(
                model=model,
                prompt="ping",
                max_tokens=1,
                temperature=0.0
            )

        latency = (asyncio.get_event_loop().time() - start_time) * 1000

        return ComponentHealth(
            name="llm",
            status=HealthStatus.HEALTHY,
            latency_ms=latency,
            message=f"Provider {response.provider.value} operational ({model})",
            details={
                "provider": response.provider.value,
                "model": response.model,
                "latency_ms": round(latency, 2),
            },
        )

    except asyncio.TimeoutError:
        return ComponentHealth(
            name="llm",
            status=HealthStatus.DEGRADED,
            message="LLM ping timed out after 5s",
        )
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
        message = str(e)
        normalized = message.lower()
        if "credit" in normalized or "billing" in normalized or "quota" in normalized:
            message = "LLM provider account has insufficient credits, quota, or billing access."
        elif (
            "403" in normalized
            or "forbidden" in normalized
            or "permission denied" in normalized
            or "provider denied access" in normalized
        ):
            message = "LLM provider denied access. Verify API key permissions, enabled API access, and network or regional restrictions."
        elif "model" in normalized and ("unavailable" in normalized or "rejected" in normalized):
            message = "Configured LLM model is unavailable for this provider account."
        return ComponentHealth(
            name="llm",
            status=HealthStatus.UNHEALTHY,
            message=message,
        )


# =============================================================================
# GLOBAL HEALTH CHECKER INSTANCE
# =============================================================================

_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance."""
    global _health_checker

    if _health_checker is None:
        _health_checker = HealthChecker()

        # Register default checks
        _health_checker.register_check("database", check_database_health)
        _health_checker.register_check("redis", check_redis_health)
        _health_checker.register_check("llm", check_llm_health)
        # PRODUCTION: Celery health check enabled for task queue monitoring
        _health_checker.register_check("celery", check_celery_health)

    return _health_checker


__all__ = [
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "HealthChecker",
    "get_health_checker",
    "check_database_health",
    "check_redis_health",
    "check_celery_health",
    "check_llm_health",
]
