"""
Monitoring Infrastructure: Prometheus Metrics Collection

Provides comprehensive observability for the content automation pipeline
using Prometheus metrics for monitoring and alerting.
"""

from typing import Any, Dict, Optional

from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class MetricsCollector:
    """
    Prometheus metrics collector for system observability.

    Tracks key performance indicators (KPIs) including:
    - Workflow duration and success rates
    - LLM API usage and costs
    - Cache performance
    - System health metrics
    """

    def __init__(self):
        """Initialize metrics collector with Prometheus metrics."""
        # Workflow metrics
        self.workflow_duration_seconds = Histogram(
            "workflow_duration_seconds",
            "End-to-end workflow execution time",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],  # 1s to 30min
            labelnames=["project_id", "workflow_type"],
        )

        self.workflow_total_cost = Gauge(
            "workflow_total_cost",
            "Total cost of generated articles",
            labelnames=["project_id", "workflow_type"],
        )

        self.workflow_success_total = Counter(
            "workflow_success_total",
            "Total successful workflows",
            labelnames=["project_id", "workflow_type"],
        )

        self.workflow_failure_total = Counter(
            "workflow_failure_total",
            "Total failed workflows",
            labelnames=["project_id", "workflow_type", "error_type"],
        )

        # LLM API metrics
        self.llm_api_requests_total = Counter(
            "llm_api_requests_total",
            "Total LLM API requests",
            labelnames=["model", "provider", "status"],
        )

        self.llm_api_tokens_total = Counter(
            "llm_api_tokens_total",
            "Total tokens consumed",
            labelnames=["model", "provider", "token_type"],
        )

        self.llm_api_cost_total = Counter(
            "llm_api_cost_total", "Total LLM API costs", labelnames=["model", "provider"]
        )

        self.llm_api_latency_seconds = Histogram(
            "llm_api_latency_seconds",
            "LLM API request latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            labelnames=["model", "provider"],
        )

        # Cache metrics
        self.cache_hits_total = Counter(
            "cache_hits_total", "Total cache hits", labelnames=["cache_level", "cache_type"]
        )

        self.cache_misses_total = Counter(
            "cache_misses_total", "Total cache misses", labelnames=["cache_level", "cache_type"]
        )

        self.cache_size_bytes = Gauge(
            "cache_size_bytes", "Current cache size in bytes", labelnames=["cache_level"]
        )

        # System metrics
        self.active_workflows = Gauge(
            "active_workflows", "Number of currently active workflows", labelnames=["project_id"]
        )

        self.queue_size = Gauge("queue_size", "Current task queue size", labelnames=["queue_name"])

        self.error_rate = Gauge(
            "error_rate", "Current error rate (errors per minute)", labelnames=["error_type"]
        )

        logger.info("Metrics collector initialized with Prometheus metrics")

    def record_workflow_completion(
        self,
        project_id: str,
        workflow_type: str,
        duration_seconds: float,
        cost: float,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record workflow completion metrics.

        Args:
            project_id: Project identifier
            workflow_type: Type of workflow (e.g., "content_generation")
            duration_seconds: Workflow execution time
            cost: Total cost of the workflow
            success: Whether workflow succeeded
            error_type: Error type if failed
        """
        self.workflow_duration_seconds.labels(
            project_id=project_id, workflow_type=workflow_type
        ).observe(duration_seconds)

        self.workflow_total_cost.labels(project_id=project_id, workflow_type=workflow_type).set(
            cost
        )

        if success:
            self.workflow_success_total.labels(
                project_id=project_id, workflow_type=workflow_type
            ).inc()
        else:
            self.workflow_failure_total.labels(
                project_id=project_id,
                workflow_type=workflow_type,
                error_type=error_type or "unknown",
            ).inc()

    def record_llm_api_call(
        self,
        model: str,
        provider: str,
        status: str,
        tokens_used: int,
        cost: float,
        latency_seconds: float,
        token_type: str = "total",
    ) -> None:
        """
        Record LLM API call metrics.

        Args:
            model: Model identifier (e.g., "gpt-4")
            provider: Provider name (e.g., "openai")
            status: Request status ("success", "failure", "timeout")
            tokens_used: Number of tokens consumed
            cost: Cost of the request
            latency_seconds: Request latency
            token_type: Type of tokens ("prompt", "completion", "total")
        """
        self.llm_api_requests_total.labels(model=model, provider=provider, status=status).inc()

        self.llm_api_tokens_total.labels(model=model, provider=provider, token_type=token_type).inc(
            tokens_used
        )

        self.llm_api_cost_total.labels(model=model, provider=provider).inc(cost)

        self.llm_api_latency_seconds.labels(model=model, provider=provider).observe(latency_seconds)

    def record_cache_hit(self, cache_level: str, cache_type: str) -> None:
        """Record cache hit."""
        self.cache_hits_total.labels(cache_level=cache_level, cache_type=cache_type).inc()

    def record_cache_miss(self, cache_level: str, cache_type: str) -> None:
        """Record cache miss."""
        self.cache_misses_total.labels(cache_level=cache_level, cache_type=cache_type).inc()

    def update_cache_size(self, cache_level: str, size_bytes: int) -> None:
        """Update cache size metric."""
        self.cache_size_bytes.labels(cache_level=cache_level).set(size_bytes)

    def update_active_workflows(self, project_id: str, count: int) -> None:
        """Update active workflows count."""
        self.active_workflows.labels(project_id=project_id).set(count)

    def update_queue_size(self, queue_name: str, size: int) -> None:
        """Update task queue size."""
        self.queue_size.labels(queue_name=queue_name).set(size)

    def update_error_rate(self, error_type: str, rate: float) -> None:
        """Update error rate."""
        self.error_rate.labels(error_type=error_type).set(rate)

    def export_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        return generate_latest(REGISTRY)

    def get_content_type(self) -> str:
        """Get content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of current metrics for health checks.

        Returns:
            Dictionary with key metrics
        """
        # This is a simplified summary - in production you might want
        # to collect actual metric values from the registry
        return {
            "metrics_initialized": True,
            "total_metrics": len(REGISTRY._names_to_collectors),
            "workflow_metrics": True,
            "llm_metrics": True,
            "cache_metrics": True,
            "system_metrics": True,
        }
