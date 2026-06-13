"""
Monitoring Infrastructure: Structured Logging with Structlog

Provides JSON-based structured logging for production observability.
Configures structlog with processors for consistent, parseable log output.
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from infrastructure.redaction import (
    configure_loguru_redaction,
    structlog_redaction_processor,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


def configure_structlog() -> None:
    """
    Configure structlog for JSON-based production logging.

    Sets up processors for:
    - Timestamping (ISO 8601)
    - Log level formatting
    - Exception formatting with stack traces
    - JSON rendering
    """
    configure_loguru_redaction()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog_redaction_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def configure_opentelemetry() -> None:
    """
    Configure OpenTelemetry OTLP exporter for distributed tracing.

    Reads configuration from settings and sets up trace export to
    an OTLP-compatible backend (Jaeger, Tempo, etc.).
    """
    try:
        from config.settings import get_settings
        settings = get_settings()

        if not settings.monitoring.enable_tracing:
            return

        if not settings.monitoring.otlp_endpoint:
            logging.warning("OTLP tracing enabled but no endpoint configured")
            return

        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        # Create resource with service name
        resource = Resource.create({
            SERVICE_NAME: settings.monitoring.otlp_service_name
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Create OTLP exporter
        exporter = OTLPSpanExporter(
            endpoint=settings.monitoring.otlp_endpoint,
            insecure=settings.monitoring.otlp_insecure,
        )

        # Add batch processor for performance
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        logging.info(
            f"OpenTelemetry OTLP exporter configured | "
            f"endpoint={settings.monitoring.otlp_endpoint} | "
            f"service={settings.monitoring.otlp_service_name}"
        )
    except ImportError as e:
        logging.warning(f"OpenTelemetry OTLP export not available: {e}")
    except Exception as e:
        logging.error(f"Failed to configure OpenTelemetry: {e}")


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structlog logger instance bound with a name context.

    Args:
        name: Logger name (typically module __name__)

    Returns:
        Configured structlog BoundLogger
    """
    return structlog.get_logger(name)


class MetricsCollector:
    """
    Prometheus metrics collector for system observability.

    Tracks key performance indicators (KPIs) including:
    - Workflow duration and success rates
    - LLM API usage and costs
    - Cache performance
    - System health metrics

    Note:
        Uses singleton pattern for metrics to prevent duplicate registration
        when multiple instances are created.
    """

    # Class-level storage for singleton metrics
    _metrics_initialized = False
    _workflow_duration_seconds = None
    _workflow_total_cost = None
    _workflow_success_total = None
    _workflow_failure_total = None
    _llm_api_requests_total = None
    _llm_api_tokens_total = None
    _llm_api_cost_total = None
    _llm_api_latency_seconds = None
    _cache_hits_total = None
    _cache_misses_total = None
    _cache_size_bytes = None
    _active_workflows = None
    _queue_size = None
    _error_rate = None

    def __init__(self):
        """Initialize metrics collector with Prometheus metrics (singleton pattern)."""
        if not MetricsCollector._metrics_initialized:
            self._initialize_metrics()
            MetricsCollector._metrics_initialized = True

        # Use class-level metrics
        self.workflow_duration_seconds = MetricsCollector._workflow_duration_seconds
        self.workflow_total_cost = MetricsCollector._workflow_total_cost
        self.workflow_success_total = MetricsCollector._workflow_success_total
        self.workflow_failure_total = MetricsCollector._workflow_failure_total
        self.llm_api_requests_total = MetricsCollector._llm_api_requests_total
        self.llm_api_tokens_total = MetricsCollector._llm_api_tokens_total
        self.llm_api_cost_total = MetricsCollector._llm_api_cost_total
        self.llm_api_latency_seconds = MetricsCollector._llm_api_latency_seconds
        self.cache_hits_total = MetricsCollector._cache_hits_total
        self.cache_misses_total = MetricsCollector._cache_misses_total
        self.cache_size_bytes = MetricsCollector._cache_size_bytes
        self.active_workflows = MetricsCollector._active_workflows
        self.queue_size = MetricsCollector._queue_size
        self.error_rate = MetricsCollector._error_rate

        log = get_logger(__name__)
        log.info("metrics_collector_initialized", metrics_type="prometheus")

    @classmethod
    def _initialize_metrics(cls):
        """Initialize Prometheus metrics once at class level."""
        # Workflow metrics
        cls._workflow_duration_seconds = Histogram(
            "workflow_duration_seconds",
            "End-to-end workflow execution time",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],  # 1s to 30min
            labelnames=["project_id", "workflow_type"],
        )

        cls._workflow_total_cost = Gauge(
            "workflow_total_cost",
            "Total cost of generated articles",
            labelnames=["project_id", "workflow_type"],
        )

        cls._workflow_success_total = Counter(
            "workflow_success_total",
            "Total successful workflows",
            labelnames=["project_id", "workflow_type"],
        )

        cls._workflow_failure_total = Counter(
            "workflow_failure_total",
            "Total failed workflows",
            labelnames=["project_id", "workflow_type", "error_type"],
        )

        # LLM API metrics
        cls._llm_api_requests_total = Counter(
            "llm_api_requests_total",
            "Total LLM API requests",
            labelnames=["model", "provider", "status"],
        )

        cls._llm_api_tokens_total = Counter(
            "llm_api_tokens_total",
            "Total tokens consumed",
            labelnames=["model", "provider", "token_type"],
        )

        cls._llm_api_cost_total = Counter(
            "llm_api_cost_total", "Total LLM API costs", labelnames=["model", "provider"]
        )

        cls._llm_api_latency_seconds = Histogram(
            "llm_api_latency_seconds",
            "LLM API request latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            labelnames=["model", "provider"],
        )

        # Cache metrics
        cls._cache_hits_total = Counter(
            "cache_hits_total", "Total cache hits", labelnames=["cache_level", "cache_type"]
        )

        cls._cache_misses_total = Counter(
            "cache_misses_total", "Total cache misses", labelnames=["cache_level", "cache_type"]
        )

        cls._cache_size_bytes = Gauge(
            "cache_size_bytes", "Current cache size in bytes", labelnames=["cache_level"]
        )

        # System metrics
        cls._active_workflows = Gauge(
            "active_workflows", "Number of currently active workflows", labelnames=["project_id"]
        )

        cls._queue_size = Gauge("queue_size", "Current task queue size", labelnames=["queue_name"])

        cls._error_rate = Gauge(
            "error_rate", "Current error rate (errors per minute)", labelnames=["error_type"]
        )

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

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton for testing purposes."""
        # Unregister all collectors from the registry
        for collector in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(collector)

        # Clear cached metrics so a fresh collector can be created
        cls._metrics_initialized = False
        cls._workflow_duration_seconds = None
        cls._workflow_total_cost = None
        cls._workflow_success_total = None
        cls._workflow_failure_total = None
        cls._llm_api_requests_total = None
        cls._llm_api_tokens_total = None
        cls._llm_api_cost_total = None
        cls._llm_api_latency_seconds = None
        cls._cache_hits_total = None
        cls._cache_misses_total = None
        cls._cache_size_bytes = None
        cls._active_workflows = None
        cls._queue_size = None
        cls._error_rate = None

    async def reset(self) -> None:
        """Async-compatible reset wrapper for test suites."""
        self.__class__.reset_singleton()
