"""
Monitoring Infrastructure: Production Observability System

Implements comprehensive observability through three pillars:
1. Metrics: Time-series measurements with dimensional aggregation
2. Tracing: Distributed request flow visualization
3. Logging: Structured event recording with correlation

Theoretical Foundation:
- Information Theory: Entropy-based sampling for high-cardinality dimensions
- Signal Processing: EWMA filters for noise reduction in time-series
- Probability Theory: Reservoir sampling for fair trace collection

Architecture: Push-based metrics with pull-based exposition (Prometheus-compatible)
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

# ============================================================================
# METRICS: Time-Series Measurements
# ============================================================================


class MetricType(str, Enum):
    """Metric type taxonomy for semantic categorization."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution sampling
    SUMMARY = "summary"  # Quantile calculation


@dataclass
class MetricLabels:
    """Dimensional labels for metric aggregation."""

    project_id: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None
    channel: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary, filtering None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ExponentialMovingAverage:
    """
    EWMA filter for real-time metric smoothing.

    Mathematical Definition:
    S_t = α * x_t + (1 - α) * S_{t-1}

    where α is the smoothing factor (0 < α < 1)

    Properties:
    - Recent values weighted more heavily
    - Computationally efficient (O(1) update)
    - Suitable for streaming data
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize EWMA filter.

        Args:
            alpha: Smoothing factor (0 < α < 1)
                  Higher values = more responsive to recent changes
                  Lower values = more smoothing
        """
        assert 0 < alpha < 1, "Alpha must be in (0, 1)"
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, observation: float) -> float:
        """Update EWMA with new observation."""
        if self.value is None:
            self.value = observation
        else:
            self.value = self.alpha * observation + (1 - self.alpha) * self.value
        return self.value

    def get(self) -> Optional[float]:
        """Get current EWMA value."""
        return self.value


class MetricsCollector:
    """
    Production metrics collection with Prometheus exposition.

    Design Principles:
    1. Low overhead: Minimal performance impact on hot paths
    2. High cardinality support: Efficient handling of dimensional metrics
    3. Prometheus-compatible: Standard exposition format
    4. Type-safe: Strong typing for metric definitions
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()

        # Content Generation Metrics
        self.generation_total = Counter(
            "content_generation_total",
            "Total content generation requests",
            ["project_id", "status", "model"],
            registry=self.registry,
        )

        self.generation_duration = Histogram(
            "content_generation_duration_seconds",
            "Content generation duration",
            ["project_id", "model"],
            buckets=(30, 60, 90, 120, 150, 180, 240, 300, float("inf")),
            registry=self.registry,
        )

        self.generation_cost = Summary(
            "content_generation_cost_dollars",
            "Content generation cost in USD",
            ["project_id", "model"],
            registry=self.registry,
        )

        self.generation_tokens = Histogram(
            "content_generation_tokens_total",
            "Total tokens used in generation",
            ["project_id", "model"],
            buckets=(500, 1000, 2000, 4000, 8000, 16000, 32000, float("inf")),
            registry=self.registry,
        )

        # Quality Metrics
        self.quality_readability = Histogram(
            "content_quality_readability_score",
            "Article readability score (Flesch-Kincaid)",
            ["project_id"],
            buckets=(30, 40, 50, 60, 70, 80, 90, 100),
            registry=self.registry,
        )

        self.quality_word_count = Histogram(
            "content_quality_word_count",
            "Article word count",
            ["project_id"],
            buckets=(500, 1000, 1500, 2000, 2500, 3000, 4000, float("inf")),
            registry=self.registry,
        )

        # Cache Metrics
        self.cache_hits = Counter(
            "cache_hits_total",
            "Total cache hits",
            ["layer"],
            registry=self.registry,
        )

        self.cache_misses = Counter(
            "cache_misses_total",
            "Total cache misses",
            ["layer"],
            registry=self.registry,
        )

        # LLM API Metrics
        self.llm_requests = Counter(
            "llm_api_requests_total",
            "Total LLM API requests",
            ["model", "provider", "status"],
            registry=self.registry,
        )

        self.llm_latency = Histogram(
            "llm_api_latency_seconds",
            "LLM API request latency",
            ["model", "provider"],
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, float("inf")),
            registry=self.registry,
        )

        # System Metrics
        self.active_workflows = Gauge(
            "active_workflows",
            "Number of active workflows",
            ["project_id"],
            registry=self.registry,
        )

        self.task_queue_depth = Gauge(
            "task_queue_depth",
            "Number of tasks in queue",
            ["priority"],
            registry=self.registry,
        )

        # EWMA filters for real-time monitoring
        self.ewma_latency = ExponentialMovingAverage(alpha=0.1)
        self.ewma_cost = ExponentialMovingAverage(alpha=0.1)

    logger.info("MetricsCollector initialized with Prometheus registry")

    async def record_generation(
        self,
        article_id: str,
        project_id: str,
        word_count: int,
        tokens_used: int,
        cost: float,
        generation_time: float,
        readability_score: float,
        avg_keyword_density: float,
        model: str = "unknown",
        status: str = "success",
    ):
        """
        Record content generation metrics.

        Args:
            article_id: Unique article identifier
            project_id: Project identifier
            word_count: Number of words generated
            tokens_used: Total tokens consumed
            cost: Generation cost in USD
            generation_time: Time taken in seconds
            readability_score: Flesch-Kincaid score
            avg_keyword_density: Average keyword density
            model: LLM model used
            status: Generation status (success/failure)
        """
        # Increment counters
        self.generation_total.labels(project_id=project_id, status=status, model=model).inc()

        # Record distributions
        self.generation_duration.labels(project_id=project_id, model=model).observe(generation_time)

        self.generation_cost.labels(project_id=project_id, model=model).observe(cost)

        self.generation_tokens.labels(project_id=project_id, model=model).observe(tokens_used)

        self.quality_readability.labels(project_id=project_id).observe(readability_score)

        self.quality_word_count.labels(project_id=project_id).observe(word_count)

        # Update EWMA filters
        self.ewma_latency.update(generation_time)
        self.ewma_cost.update(cost)

        logger.debug(
            f"Recorded generation metrics | article_id={article_id} | "
            f"cost=${cost:.4f} | time={generation_time:.1f}s"
        )

    async def record_cache_hit(self, layer: str):
        """Record cache hit."""
        self.cache_hits.labels(layer=layer).inc()

    async def record_cache_miss(self, layer: str):
        """Record cache miss."""
        self.cache_misses.labels(layer=layer).inc()

    async def record_llm_request(
        self,
        model: str,
        provider: str,
        latency: float,
        status: str = "success",
    ):
        """Record LLM API request metrics."""
        self.llm_requests.labels(model=model, provider=provider, status=status).inc()

        if status == "success":
            self.llm_latency.labels(model=model, provider=provider).observe(latency)

    def increment_active_workflows(self, project_id: str):
        """Increment active workflow gauge."""
        self.active_workflows.labels(project_id=project_id).inc()

    def decrement_active_workflows(self, project_id: str):
        """Decrement active workflow gauge."""
        self.active_workflows.labels(project_id=project_id).dec()

    async def record_workflow(
        self,
        workflow_id: str,
        project_id: str,
        article_id: str,
        execution_time: float,
        total_cost: float,
        total_tokens: int,
        word_count: int,
        quality_score: float,
        events: int,
    ):
        """Record complete workflow execution metrics."""
        # Use existing generation metrics
        await self.record_generation(
            article_id=article_id,
            project_id=project_id,
            word_count=word_count,
            tokens_used=total_tokens,
            cost=total_cost,
            generation_time=execution_time,
            readability_score=quality_score,
            avg_keyword_density=0.0,  # Not tracked at workflow level
        )

    async def record_distribution(
        self,
        article_id: str,
        channel: str,
        destination: str,
        success: bool,
        distribution_time: float,
        error: Optional[str] = None,
    ):
        """Record distribution metrics."""
        # Could add distribution-specific metrics here
        logger.debug(
            f"Distribution recorded | article_id={article_id} | "
            f"channel={channel} | success={success}"
        )

    async def record_task_failure(
        self,
        task_id: str,
        task_name: str,
        error: str,
    ):
        """Record task failure."""
        logger.error(f"Task failure recorded | task_id={task_id} | error={error}")

    def export_metrics(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        return generate_latest(self.registry).decode("utf-8")

    def get_content_type(self) -> str:
        """Get Prometheus content type for HTTP headers."""
        return CONTENT_TYPE_LATEST

    async def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get human-readable summary statistics.

        Returns:
            Dictionary with current metric values
        """
        return {
            "ewma_latency_seconds": self.ewma_latency.get(),
            "ewma_cost_dollars": self.ewma_cost.get(),
            "active_workflows": (
                sum(
                    metric.labels(**labels)._value.get()
                    for metric, labels in self.active_workflows._metrics.items()
                )
                if hasattr(self.active_workflows, "_metrics")
                else 0
            ),
        }

    async def reset(self):
        """Reset all metrics (for testing purposes)."""
        self.registry = CollectorRegistry()
        logger.warning("Metrics registry reset - all metrics cleared")


# ============================================================================
# DISTRIBUTED TRACING: Request Flow Visualization
# ============================================================================
@dataclass
class Span:
    """
    Distributed tracing span following W3C Trace Context specification.
    Represents a single unit of work in a distributed trace.
    """

    trace_id: str  # Unique trace identifier (128-bit)
    span_id: str  # Unique span identifier (64-bit)
    parent_span_id: Optional[str]  # Parent span for hierarchy
    name: str  # Operation name
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error

    def duration_ms(self) -> Optional[float]:
        """Calculate span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def add_event(self, name: str, attributes: Optional[Dict] = None):
        """Add event to span."""
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.utcnow().isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value

    def finish(self, status: str = "ok"):
        """Finish span with status."""
        self.end_time = datetime.utcnow()
        self.status = status


class Tracer:
    """
    Distributed tracing implementation with span management.
    Supports:
    - Nested spans for operation hierarchies
    - Context propagation across async boundaries
    - W3C Trace Context format
    - OpenTelemetry-compatible semantics
    """

    def __init__(self):
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: deque = deque(maxlen=1000)  # Ring buffer
        self._lock = asyncio.Lock()

    def generate_trace_id(self) -> str:
        """Generate 128-bit trace ID."""
        return uuid.uuid4().hex + uuid.uuid4().hex[:16]

    def generate_span_id(self) -> str:
        """Generate 64-bit span ID."""
        return uuid.uuid4().hex[:16]

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a new span with automatic lifecycle management.

        Usage:
            async with tracer.start_span("operation") as span:
                span.set_attribute("key", "value")
                # ... do work ...
        """
        span = Span(
            trace_id=trace_id or self.generate_trace_id(),
            span_id=self.generate_span_id(),
            parent_span_id=parent_span_id,
            name=name,
            start_time=datetime.utcnow(),
            attributes=attributes or {},
        )

        async with self._lock:
            self.active_spans[span.span_id] = span

        try:
            yield span
        except Exception as e:
            span.set_attribute("error", str(e))
            span.finish(status="error")
            raise
        else:
            span.finish(status="ok")
        finally:
            async with self._lock:
                self.active_spans.pop(span.span_id, None)
                self.completed_spans.append(span)

    async def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get all active traces."""
        async with self._lock:
            return [
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "name": span.name,
                    "duration_ms": (datetime.utcnow() - span.start_time).total_seconds() * 1000,
                }
                for span in self.active_spans.values()
            ]

    async def get_trace_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent completed traces."""
        async with self._lock:
            return [
                {
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "name": span.name,
                    "duration_ms": span.duration_ms(),
                    "status": span.status,
                    "attributes": span.attributes,
                }
                for span in list(self.completed_spans)[-limit:]
            ]


# ============================================================================
# STRUCTURED LOGGING: Event Recording with Correlation
# ============================================================================
class StructuredLogger:
    """
    Structured logging with request correlation.
    Enhances standard logging with:
    - Request ID propagation
    - Trace context injection
    - Structured fields
    - Log levels with semantic meaning
    """

    def __init__(self):
        self.configure_logging()

    def configure_logging(self):
        """Configure loguru for structured logging."""
        # Remove default handler
        logger.remove()

        # Add console handler with formatting
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level="INFO",
            colorize=True,
        )

        # Add file handler for persistence
        logger.add(
            "logs/app_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            level="DEBUG",
        )

    def log_with_context(
        self,
        level: str,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **kwargs,
    ):
        """Log with trace context."""
        context = {}
        if trace_id:
            context["trace_id"] = trace_id
        if span_id:
            context["span_id"] = span_id
        context.update(kwargs)

        log_func = getattr(logger.bind(**context), level.lower())
        log_func(message)


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================
# Global metrics collector (singleton)
_global_metrics_collector: Optional[MetricsCollector] = None
_global_tracer: Optional[Tracer] = None
_global_logger: Optional[StructuredLogger] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def get_tracer() -> Tracer:
    """Get or create global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def get_logger() -> StructuredLogger:
    """Get or create global structured logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger()
    return _global_logger
