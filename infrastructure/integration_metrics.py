"""Low-overhead Prometheus rendering for durable integration operations snapshots.

The Celery Beat refresh task writes a bounded aggregate to Redis. API workers only
read that snapshot while serving ``/metrics``; Prometheus scrapes never trigger
WordPress/Search Console database queries and worker-local counters are avoided.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

INTEGRATION_NAMES = ("wordpress", "search_console")
ACTIVE_STATES = ("queued", "running", "retrying")
HEALTH_STATES = ("healthy", "idle", "warning", "degraded", "critical", "unknown")


def _finite_number(value: Any, *, minimum: float = 0.0) -> float:
    """Convert untrusted snapshot values into finite, non-negative metrics."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return minimum
    if not math.isfinite(number):
        return minimum
    return max(minimum, number)


def _parse_generated_at(value: Any, *, now: datetime) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        generated_at = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    return max(0.0, (now - generated_at.astimezone(timezone.utc)).total_seconds())


def _line(name: str, value: Any, labels: str = "") -> str:
    number = _finite_number(value)
    rendered = str(int(number)) if number.is_integer() else format(number, ".12g")
    return f"{name}{labels} {rendered}"


def render_integration_snapshot_metrics(
    snapshot: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> str:
    """Render a Redis-backed operations snapshot in Prometheus text format.

    Labels are fixed allow-lists only, preventing cardinality growth from project,
    URL, error, task, or user identifiers. Missing/malformed snapshots are exposed
    explicitly through ``integration_snapshot_available`` rather than raising.
    """
    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    available = isinstance(snapshot, dict)
    age_seconds = _parse_generated_at(snapshot.get("generated_at"), now=current_time) if available else None
    raw_integrations = snapshot.get("integrations") if available else None
    complete_integrations = isinstance(raw_integrations, dict) and all(
        isinstance(raw_integrations.get(name), dict) for name in INTEGRATION_NAMES
    )
    if age_seconds is None or not complete_integrations:
        available = False
        age_seconds = 0.0

    lines = [
        "# HELP integration_snapshot_available Whether the shared integration operations snapshot is valid.",
        "# TYPE integration_snapshot_available gauge",
        _line("integration_snapshot_available", 1 if available else 0),
        "# HELP integration_snapshot_age_seconds Age of the shared integration operations snapshot.",
        "# TYPE integration_snapshot_age_seconds gauge",
        _line("integration_snapshot_age_seconds", age_seconds),
        "# HELP integration_durable_active_items Active durable operations by integration and state.",
        "# TYPE integration_durable_active_items gauge",
        "# HELP integration_durable_stale_items Stale durable operations requiring reconciliation.",
        "# TYPE integration_durable_stale_items gauge",
        "# HELP integration_durable_health One-hot integration health state from durable operational data.",
        "# TYPE integration_durable_health gauge",
        "# HELP integration_durable_recent_total Terminal durable operations in the configured lookback.",
        "# TYPE integration_durable_recent_total gauge",
        "# HELP integration_durable_recent_succeeded Successful durable operations in the configured lookback.",
        "# TYPE integration_durable_recent_succeeded gauge",
        "# HELP integration_durable_recent_failed Failed durable operations in the configured lookback.",
        "# TYPE integration_durable_recent_failed gauge",
        "# HELP integration_durable_failure_rate Fraction of terminal durable operations that failed.",
        "# TYPE integration_durable_failure_rate gauge",
        "# HELP integration_durable_p95_duration_seconds P95 terminal operation duration in the configured lookback.",
        "# TYPE integration_durable_p95_duration_seconds gauge",
        "# HELP integration_durable_recent_truncated Search Console syncs truncated by upstream result limits.",
        "# TYPE integration_durable_recent_truncated gauge",
    ]

    integrations = raw_integrations if isinstance(raw_integrations, dict) else {}

    for integration in INTEGRATION_NAMES:
        summary = integrations.get(integration)
        if not isinstance(summary, dict):
            summary = {}
        status_counts = summary.get("status_counts")
        if not isinstance(status_counts, dict):
            status_counts = {}

        integration_label = f'{{integration="{integration}"}}'
        for state in ACTIVE_STATES:
            labels = f'{{integration="{integration}",state="{state}"}}'
            lines.append(
                _line("integration_durable_active_items", status_counts.get(state, 0), labels)
            )

        lines.extend(
            [
                _line("integration_durable_stale_items", summary.get("stale_count", 0), integration_label),
                _line("integration_durable_recent_total", summary.get("recent_total", 0), integration_label),
                _line(
                    "integration_durable_recent_succeeded",
                    summary.get("recent_succeeded", 0),
                    integration_label,
                ),
                _line("integration_durable_recent_failed", summary.get("recent_failed", 0), integration_label),
                _line(
                    "integration_durable_failure_rate",
                    min(1.0, _finite_number(summary.get("failure_rate", 0))),
                    integration_label,
                ),
                _line(
                    "integration_durable_p95_duration_seconds",
                    summary.get("p95_duration_seconds", 0),
                    integration_label,
                ),
                _line(
                    "integration_durable_recent_truncated",
                    summary.get("recent_truncated", 0),
                    integration_label,
                ),
            ]
        )

        status = str(summary.get("status") or "unknown").lower()
        normalized_status = status if status in HEALTH_STATES else "unknown"
        for health_state in HEALTH_STATES:
            labels = f'{{integration="{integration}",status="{health_state}"}}'
            lines.append(
                _line(
                    "integration_durable_health",
                    1 if health_state == normalized_status else 0,
                    labels,
                )
            )

    return "\n".join(lines) + "\n"
