"""
System Routes: Health Check and Monitoring Endpoints

Provides system-level endpoints for health monitoring, metrics collection,
and observability features. These endpoints are essential for production
monitoring and operational visibility.

Architectural Pattern: System API + Health Check Pattern
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import Response
from pydantic import BaseModel

from api.dependencies import get_database, get_metrics, get_redis
from api.schemas import HealthCheckResponse
from infrastructure.database import DatabaseManager
from infrastructure.health import (
    HealthStatus,
    check_llm_health,
    get_health_checker,
)
from infrastructure.llm_options import (
    build_llm_warnings,
    get_llm_provider_options,
    get_selectable_models,
    infer_provider,
)
from infrastructure.monitoring import MetricsCollector
from infrastructure.redis_client import RedisClient
from orchestration.celery_app import app as celery_app
from security import User, get_current_active_user, get_current_superuser, is_manager_user

router = APIRouter(prefix="/system", tags=["System"])


class IncidentResponse(BaseModel):
    """Manager-facing operational incident synthesized from runtime health."""

    id: str
    severity: str
    source: str
    status: str
    audience: str
    user_message: str
    manager_detail: str
    created_at: datetime
    project_id: Optional[str] = None
    task_id: Optional[str] = None


class IncidentListResponse(BaseModel):
    """Incident inbox response."""

    incidents: List[IncidentResponse]
    open_count: int
    critical_count: int
    warning_count: int
    generated_at: datetime


class LLMModelOptionResponse(BaseModel):
    """Safe model option returned to the frontend."""

    provider: str
    model: str
    label: str
    enabled: bool
    recommended: bool = False
    reason: Optional[str] = None


class LLMProviderOptionResponse(BaseModel):
    """Safe provider status returned to the frontend."""

    provider: str
    label: str
    configured: bool
    active: bool
    models: List[LLMModelOptionResponse]


class LLMOptionsResponse(BaseModel):
    """Redacted LLM model/provider options."""

    active_model: str
    active_provider: str
    fallback_model: Optional[str] = None
    selectable_models: List[LLMModelOptionResponse]
    providers: List[LLMProviderOptionResponse]
    warnings: List[str]
    user_message: str
    manager_detail: Optional[str] = None
    generated_at: datetime


def _redact_dependency_status(raw_status: str) -> str:
    """Return user-safe dependency status without internal exception details."""
    normalized = raw_status.lower()
    if "healthy" in normalized and "unhealthy" not in normalized:
        return "healthy"
    if "degraded" in normalized:
        return "degraded"
    if "unhealthy" in normalized or "timeout" in normalized or "error" in normalized:
        return "unhealthy"
    return "unknown"


def _dependency_is_healthy(raw_status: str) -> bool:
    """Avoid substring bugs such as treating 'unhealthy' as healthy."""
    return _redact_dependency_status(raw_status) == "healthy"


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check endpoint",
    description="System health check with dependency status",
)
async def health_check(
    request: Request,
    db: DatabaseManager = Depends(get_database),
    redis: RedisClient = Depends(get_redis),
    user: User = Depends(get_current_active_user),
) -> HealthCheckResponse:
    """
    System health check with dependency status.

    Returns health status of all critical dependencies including:
    - Database connectivity
    - Redis connectivity
    - Application metrics
    """
    dependencies: Dict[str, str] = {}

    # Check database
    try:
        await asyncio.wait_for(db.health_check(), timeout=5.0)
        dependencies["database"] = "healthy"
    except asyncio.TimeoutError:
        dependencies["database"] = "degraded: timeout"
    except Exception as e:
        dependencies["database"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        ok = await redis.ping()
        dependencies["redis"] = "healthy" if ok else "unhealthy"
    except Exception as e:
        dependencies["redis"] = f"unhealthy: {str(e)}"

    # Check Celery workers
    try:
        loop = asyncio.get_event_loop()
        def _inspect():
            i = celery_app.control.inspect(timeout=2)
            return i.active()
        active_workers = await asyncio.wait_for(
            loop.run_in_executor(None, _inspect),
            timeout=3.0,
        )
        if active_workers:
            worker_count = len(active_workers)
            dependencies["celery_workers"] = f"healthy ({worker_count} workers active)"
        else:
            dependencies["celery_workers"] = "degraded: no active workers"
    except Exception as e:
        dependencies["celery_workers"] = f"unhealthy: {str(e)}"

    # Determine overall status
    overall_status = "healthy" if all(_dependency_is_healthy(v) for v in dependencies.values()) else "degraded"

    if not is_manager_user(user):
        dependencies = {
            name: _redact_dependency_status(value)
            for name, value in dependencies.items()
        }

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        dependencies=dependencies,
    )


@router.get(
    "/llm/options",
    response_model=LLMOptionsResponse,
    summary="Available LLM model options",
    description="Return redacted provider/model options that are safe for users to select.",
)
async def get_llm_options(
    user: User = Depends(get_current_active_user),
) -> LLMOptionsResponse:
    """
    Return model choices without exposing API keys.

    The backend remains authoritative: a model is selectable only when its
    provider key or local endpoint is configured in this process.
    """
    from config.settings import get_settings

    settings = get_settings()
    providers = get_llm_provider_options()
    selectable_models = get_selectable_models()
    warnings = build_llm_warnings()
    active_provider = infer_provider(settings.llm.primary_model) or "unknown"

    if selectable_models:
        user_message = "AI generation is available."
    else:
        user_message = "AI generation is unavailable until a manager configures an API key."

    try:
        llm_health = await asyncio.wait_for(check_llm_health(), timeout=8.0)
    except Exception as e:
        llm_health = None
        warning = f"LLM health probe failed: {e}"
        if warning not in warnings:
            warnings.append(warning)
    else:
        if llm_health.status != HealthStatus.HEALTHY:
            if llm_health.message not in warnings:
                warnings.append(llm_health.message)
            user_message = (
                "AI generation is currently unavailable because the active provider has a quota, billing, model-access, or API-access problem."
            )

    manager_detail = None
    if is_manager_user(user):
        if warnings:
            manager_detail = " ".join(warnings)
        else:
            configured = [provider.label for provider in providers if provider.configured]
            manager_detail = f"Configured providers: {', '.join(configured) if configured else 'none'}."

    return LLMOptionsResponse(
        active_model=settings.llm.primary_model,
        active_provider=active_provider,
        fallback_model=settings.llm.fallback_model,
        selectable_models=[LLMModelOptionResponse(**model.__dict__) for model in selectable_models],
        providers=[
            LLMProviderOptionResponse(
                provider=provider.provider,
                label=provider.label,
                configured=provider.configured,
                active=provider.active,
                models=[LLMModelOptionResponse(**model.__dict__) for model in provider.models],
            )
            for provider in providers
        ],
        warnings=warnings if is_manager_user(user) else [],
        user_message=user_message,
        manager_detail=manager_detail,
        generated_at=datetime.now(timezone.utc),
    )


@router.get(
    "/metrics",
    summary="System metrics (Prometheus format)",
    description="Export metrics in Prometheus format for monitoring systems",
)
async def get_system_metrics(
    metrics: MetricsCollector = Depends(get_metrics),
    user: User = Depends(get_current_superuser),
) -> Response:
    """
    Export metrics in Prometheus format.

    Compatible with Prometheus scraper for monitoring and alerting.
    Returns system metrics including:
    - Request counts and latencies
    - Task execution metrics
    - Resource utilization
    - Error rates and patterns
    """
    try:
        # Generate Prometheus-formatted metrics
        metrics_content = metrics.export_metrics()
        content_type = metrics.get_content_type()

        # If metrics_content is empty, provide default metrics
        if not metrics_content or metrics_content.strip() == "":
            metrics_content = """# HELP system_info System information
# TYPE system_info gauge
system_info{version="1.0.0",status="running"} 1

# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",status="200"} 0

# HELP system_uptime_seconds System uptime in seconds
# TYPE system_uptime_seconds gauge
system_uptime_seconds 0
"""

        return Response(content=metrics_content, media_type=content_type)

    except Exception as e:
        # Return basic metrics if there's an error
        fallback_metrics = f"""# HELP system_error System error occurred
# TYPE system_error gauge
system_error{{error="{str(e)}"}} 1

# HELP system_info System information
# TYPE system_info gauge
system_info{{version="1.0.0",status="error"}} 1
"""
        return Response(
            content=fallback_metrics, media_type="text/plain; version=0.0.4; charset=utf-8"
        )


@router.get(
    "/status",
    summary="Detailed system status",
    description="Get comprehensive system status including all components",
)
async def get_system_status(
    db: DatabaseManager = Depends(get_database),
    redis: RedisClient = Depends(get_redis),
    metrics: MetricsCollector = Depends(get_metrics),
    user: User = Depends(get_current_superuser),
) -> Dict[str, Any]:
    """
    Get detailed system status including all components.

    Returns comprehensive status information for monitoring dashboards.
    """
    status_info: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "components": {},
    }

    # Database status
    try:
        await db.health_check()
        status_info["components"]["database"] = {"status": "healthy", "connection_pool": "active"}
    except Exception as e:
        status_info["components"]["database"] = {"status": "unhealthy", "error": str(e)}

    # Redis status
    try:
        ok = await redis.ping()
        if ok:
            status_info["components"]["redis"] = {"status": "healthy", "connection": "active"}
        else:
            status_info["components"]["redis"] = {"status": "unhealthy", "error": "ping failed"}
    except Exception as e:
        status_info["components"]["redis"] = {"status": "unhealthy", "error": str(e)}

    # Celery worker status with detailed metrics
    try:
        loop = asyncio.get_event_loop()
        def _inspect_all():
            i = celery_app.control.inspect(timeout=2)
            return i.active(), i.registered(), i.stats()
        active_workers, registered_tasks, stats = await asyncio.wait_for(
            loop.run_in_executor(None, _inspect_all),
            timeout=5.0,
        )

        if active_workers:
            worker_details = []
            for worker_name, tasks in active_workers.items():
                worker_stats = stats.get(worker_name, {}) if stats else {}
                worker_details.append({
                    "name": worker_name,
                    "active_tasks": len(tasks),
                    "pool_size": worker_stats.get("pool", {}).get("max-concurrency", "unknown")
                })

            status_info["components"]["celery"] = {
                "status": "healthy",
                "workers": len(active_workers),
                "worker_details": worker_details,
                "registered_tasks": len(registered_tasks.get(list(registered_tasks.keys())[0], [])) if registered_tasks else 0
            }
        else:
            status_info["components"]["celery"] = {
                "status": "unhealthy",
                "error": "no active workers detected"
            }
    except Exception as e:
        status_info["components"]["celery"] = {"status": "unhealthy", "error": str(e)}

    # Metrics status
    try:
        # Get basic metrics info
        status_info["components"]["metrics"] = {"status": "healthy", "collector": "active"}
    except Exception as e:
        status_info["components"]["metrics"] = {"status": "unhealthy", "error": str(e)}

    # Overall status
    component_statuses = [comp.get("status") for comp in status_info["components"].values()]

    if all(status == "healthy" for status in component_statuses):
        status_info["overall_status"] = "healthy"
    elif any(status == "unhealthy" for status in component_statuses):
        status_info["overall_status"] = "degraded"
    else:
        status_info["overall_status"] = "unknown"

    return status_info


@router.get(
    "/incidents",
    response_model=IncidentListResponse,
    summary="Manager incident inbox",
    description="Synthesizes operational incidents from dependency, cost, queue, and task-failure signals.",
)
async def get_incident_inbox(
    db: DatabaseManager = Depends(get_database),
    redis: RedisClient = Depends(get_redis),
    user: User = Depends(get_current_superuser),
) -> IncidentListResponse:
    """
    Return current manager-facing incidents.

    This first version is intentionally read-only and computed from existing
    health signals, avoiding a persistence migration before the runtime is
    stable. User-facing messages are redacted; manager_detail carries the
    technical signal.
    """
    incidents: list[IncidentResponse] = []
    now = datetime.now(timezone.utc)

    def add_incident(
        incident_id: str,
        severity: str,
        source: str,
        user_message: str,
        manager_detail: str,
        project_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        incidents.append(
            IncidentResponse(
                id=incident_id,
                severity=severity,
                source=source,
                status="open",
                audience="manager",
                user_message=user_message,
                manager_detail=manager_detail,
                created_at=now,
                project_id=project_id,
                task_id=task_id,
            )
        )

    try:
        await asyncio.wait_for(db.health_check(skip_vector_check=True), timeout=5.0)
    except Exception as e:
        add_incident(
            "database-unavailable",
            "critical",
            "database",
            "The content service is degraded.",
            f"Database health check failed: {e}",
        )

    try:
        redis_ok = await asyncio.wait_for(redis.ping(), timeout=2.0)
    except Exception as e:
        redis_ok = False
        redis_error = str(e)
    else:
        redis_error = "Redis ping returned false"

    if not redis_ok:
        add_incident(
            "redis-unavailable",
            "critical",
            "redis",
            "Background processing is temporarily unavailable.",
            redis_error,
        )

    try:
        loop = asyncio.get_event_loop()

        def inspect_workers():
            return celery_app.control.inspect(timeout=1.5).active()

        active_workers = await asyncio.wait_for(
            loop.run_in_executor(None, inspect_workers),
            timeout=2.5,
        )
    except Exception as e:
        active_workers = None
        worker_detail = f"Celery inspect failed: {e}"
    else:
        worker_detail = "No active Celery workers detected"

    if not active_workers:
        add_incident(
            "worker-unavailable",
            "critical",
            "worker",
            "Generation jobs cannot start right now.",
            worker_detail,
        )

    try:
        from infrastructure.performance_monitor import PerformanceMonitor

        monitor = PerformanceMonitor(db, redis)
        daily_costs = await monitor.check_daily_costs()
        connection_pool = await monitor.check_connection_pool()
    except Exception as e:
        add_incident(
            "performance-check-failed",
            "warning",
            "backend",
            "Operational metrics are partially unavailable.",
            f"Performance monitor failed: {e}",
        )
        daily_costs = {}
        connection_pool = {}

    if daily_costs.get("status") == "warning":
        add_incident(
            "daily-budget-warning",
            "warning",
            "provider",
            "AI usage is approaching the configured daily budget.",
            (
                f"Daily cost ${daily_costs.get('total_cost_usd', 0)} "
                f"exceeded warning threshold ${daily_costs.get('threshold_usd', 0)}"
            ),
        )
    elif daily_costs.get("status") == "error":
        add_incident(
            "daily-budget-check-failed",
            "warning",
            "provider",
            "AI cost tracking is temporarily unavailable.",
            str(daily_costs.get("error", "daily cost check failed")),
        )

    if connection_pool.get("status") == "warning":
        add_incident(
            "database-pool-high",
            "warning",
            "database",
            "Database load is elevated.",
            f"Connection pool utilization {connection_pool.get('utilization_percent')}%",
        )
    elif connection_pool.get("status") == "error":
        add_incident(
            "database-pool-check-failed",
            "warning",
            "database",
            "Database pool metrics are unavailable.",
            str(connection_pool.get("error", "connection pool check failed")),
        )

    try:
        from infrastructure.health import check_llm_health

        llm_health = await asyncio.wait_for(check_llm_health(), timeout=10.0)
    except Exception as e:
        add_incident(
            "llm-health-check-failed",
            "warning",
            "provider",
            "AI provider health could not be verified.",
            f"LLM health check failed: {e}",
        )
    else:
        if llm_health.status != HealthStatus.HEALTHY:
            severity = "critical" if llm_health.status == HealthStatus.UNHEALTHY else "warning"
            add_incident(
                "llm-provider-unavailable",
                severity,
                "provider",
                "AI generation is temporarily unavailable. A manager can switch models or verify provider quota, API access, and network reachability.",
                llm_health.message,
            )

    try:
        from sqlalchemy import func, select

        from orchestration.task_persistence import TaskStatus, task_results_table

        # task_results.created_at is TIMESTAMP WITHOUT TIME ZONE in the
        # current migration, so asyncpg rejects timezone-aware datetimes here.
        since = (now - timedelta(hours=24)).replace(tzinfo=None)
        query = select(func.count().label("failed_count")).select_from(task_results_table).where(
            task_results_table.c.status == TaskStatus.FAILURE.value,
            task_results_table.c.created_at >= since,
        )
        result = await db.fetch_one(query)
        failed_count = int(result.get("failed_count", 0)) if result else 0
    except Exception as e:
        failed_count = 0
        add_incident(
            "task-failure-check-failed",
            "warning",
            "worker",
            "Task history metrics are temporarily unavailable.",
            f"Failed task query failed: {e}",
        )

    if failed_count >= 3:
        severity = "critical" if failed_count >= 10 else "warning"
        add_incident(
            "task-failure-spike",
            severity,
            "worker",
            "Recent content jobs are failing more than expected.",
            f"{failed_count} task failure(s) recorded in the last 24 hours.",
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    incidents.sort(key=lambda item: (severity_order.get(item.severity, 3), item.created_at))

    return IncidentListResponse(
        incidents=incidents,
        open_count=len(incidents),
        critical_count=sum(1 for incident in incidents if incident.severity == "critical"),
        warning_count=sum(1 for incident in incidents if incident.severity == "warning"),
        generated_at=now,
    )


# =============================================================================
# KUBERNETES PROBE ENDPOINTS
# =============================================================================


@router.get(
    "/live",
    summary="Kubernetes liveness probe",
    description="Simple liveness check for Kubernetes",
)
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint.

    Returns 200 if the application is alive.
    Used by Kubernetes to determine if the pod should be restarted.
    """
    checker = get_health_checker()
    is_alive = await checker.liveness_check()

    return {
        "status": "alive" if is_alive else "dead",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/ready",
    summary="Kubernetes readiness probe",
    description="Readiness check for Kubernetes to determine if pod can accept traffic",
)
async def readiness_probe(response: Response) -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.

    Returns 200 if the application can accept traffic.
    Checks all critical dependencies before declaring ready.
    """
    checker = get_health_checker()
    health = await checker.check_all()

    is_ready = health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if is_ready else "not_ready",
        "overall_status": health.status.value,
        "components": {
            c.name: c.status.value
            for c in health.components
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/detailed",
    summary="Detailed health check with all components",
    description="Comprehensive health check with component details and latencies",
)
async def detailed_health_check(
    user: User = Depends(get_current_superuser),
) -> Dict[str, Any]:
    """
    Detailed health check with all component statuses.

    Returns comprehensive health information including:
    - Individual component health
    - Component latencies
    - Error messages
    - System uptime
    """
    checker = get_health_checker()
    health = await checker.check_all()

    return health.to_dict()


@router.get(
    "/performance",
    summary="Performance monitoring metrics",
    description="Get performance metrics including slow queries, costs, and connection pool status",
)
async def get_performance_metrics(
    db: DatabaseManager = Depends(get_database),
    redis: RedisClient = Depends(get_redis),
    user: User = Depends(get_current_superuser),
) -> Dict[str, Any]:
    """
    Get performance monitoring metrics.

    Returns:
    - Slow query detection
    - Daily LLM cost tracking
    - Connection pool utilization
    - Overall system performance status
    """
    from infrastructure.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor(db, redis)
    health = await monitor.get_system_health()

    return health
