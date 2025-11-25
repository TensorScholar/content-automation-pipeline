"""
System Routes: Health Check and Monitoring Endpoints

Provides system-level endpoints for health monitoring, metrics collection,
and observability features. These endpoints are essential for production
monitoring and operational visibility.

Architectural Pattern: System API + Health Check Pattern
"""

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from api.schemas import HealthCheckResponse
from container import (
    DatabaseManager,
    MetricsCollector,
    RedisClient,
    container,
    get_database,
    get_metrics,
    get_redis,
)

router = APIRouter(prefix="/system", tags=["System"])


# Simple dependency functions for FastAPI
def get_database_dependency() -> DatabaseManager:
    """Get DatabaseManager instance for FastAPI dependency injection."""
    return container.database()


def get_redis_dependency() -> RedisClient:
    """Get RedisClient instance for FastAPI dependency injection."""
    return container.redis()


def get_metrics_dependency() -> MetricsCollector:
    """Get MetricsCollector instance for FastAPI dependency injection."""
    return container.metrics()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check endpoint",
    description="System health check with dependency status",
)
async def health_check(
    request: Request,
    db: DatabaseManager = Depends(get_database_dependency),
    redis: RedisClient = Depends(get_redis_dependency),
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
        await db.health_check()
        dependencies["database"] = "healthy"
    except Exception as e:
        dependencies["database"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        ok = await redis.ping()
        dependencies["redis"] = "healthy" if ok else "unhealthy"
    except Exception as e:
        dependencies["redis"] = f"unhealthy: {str(e)}"

    # Task queue check removed (TaskManager was deleted)

    # Determine overall status
    overall_status = "healthy" if all("healthy" in v for v in dependencies.values()) else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        dependencies=dependencies,
    )


@router.get(
    "/metrics",
    summary="System metrics (Prometheus format)",
    description="Export metrics in Prometheus format for monitoring systems",
)
async def get_system_metrics(
    metrics: MetricsCollector = Depends(get_metrics_dependency),
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
    db: DatabaseManager = Depends(get_database_dependency),
    redis: RedisClient = Depends(get_redis_dependency),
    metrics: MetricsCollector = Depends(get_metrics_dependency),
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

    # Task manager status removed (TaskManager was deleted)

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
