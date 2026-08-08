"""Manager-facing operational intelligence for external integrations."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from infrastructure.redis_client import RedisClient
    from knowledge.publishing_repository import PublishingRepository
    from knowledge.search_console_repository import SearchConsoleRepository

STATUS_RANK = {"healthy": 0, "idle": 0, "warning": 1, "degraded": 2, "critical": 3}
INTEGRATION_OPERATIONS_SNAPSHOT_KEY = "operations:integration-summary:v1"
INTEGRATION_OPERATIONS_SNAPSHOT_TTL_SECONDS = 900


class IntegrationOperationsService:
    """Aggregate bounded WordPress/Search Console signals for operations and UI."""

    def __init__(
        self,
        *,
        publishing_repository: "PublishingRepository",
        search_console_repository: "SearchConsoleRepository",
        cache: "RedisClient | None" = None,
    ) -> None:
        self.publishing = publishing_repository
        self.search_console = search_console_repository
        self.cache = cache

    async def get_summary(
        self,
        *,
        project_id: UUID | None = None,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        bounded_lookback = max(1, min(lookback_hours, 168))
        wordpress_raw, search_console_raw = await asyncio.gather(
            self.publishing.get_operational_summary(
                project_id=project_id,
                lookback_hours=bounded_lookback,
                recent_limit=10,
            ),
            self.search_console.get_operational_summary(
                project_id=project_id,
                lookback_hours=bounded_lookback,
                recent_limit=10,
            ),
        )
        wordpress = self._wordpress_summary(wordpress_raw)
        search_console = self._search_console_summary(search_console_raw)
        overall = self._worst_status(wordpress["status"], search_console["status"])
        recommendations = self._recommendations(wordpress, search_console)

        result = {
            "generated_at": datetime.now(timezone.utc),
            "project_id": str(project_id) if project_id else None,
            "lookback_hours": bounded_lookback,
            "overall_status": overall,
            "integrations": {
                "wordpress": wordpress,
                "search_console": search_console,
            },
            "recommendations": recommendations,
            "slo": {
                "stale_active_items": 0,
                "warning_failure_rate": 0.10,
                "critical_failure_rate": 0.50,
                "maximum_sync_age_hours": 36,
            },
        }
        if project_id is None:
            await self._store_snapshot_safely(result)
        return result

    async def get_cached_snapshot(self) -> dict[str, Any] | None:
        """Return the shared snapshot used by Prometheus without database work."""
        if self.cache is None:
            return None
        try:
            value = await self.cache.get(INTEGRATION_OPERATIONS_SNAPSHOT_KEY)
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    async def _store_snapshot_safely(self, summary: dict[str, Any]) -> None:
        """Cache a JSON-safe snapshot; cache failures never alter API results."""
        if self.cache is None:
            return
        try:
            await self.cache.set(
                INTEGRATION_OPERATIONS_SNAPSHOT_KEY,
                self._json_safe(summary),
                ttl=INTEGRATION_OPERATIONS_SNAPSHOT_TTL_SECONDS,
            )
        except Exception:
            return

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        return value

    @staticmethod
    def _wordpress_summary(raw: dict[str, Any]) -> dict[str, Any]:
        total = int(raw.get("recent_total") or 0)
        failed = int(raw.get("recent_failed") or 0)
        stale = int(raw.get("stale_count") or 0)
        active = int(raw.get("active_count") or 0)
        failure_rate = failed / total if total else 0.0
        status = "idle" if total == 0 and active == 0 else "healthy"
        reasons: list[str] = []
        if stale > 0:
            status = "critical"
            reasons.append("stale_publish_attempts")
        elif failure_rate >= 0.50 and total >= 2:
            status = "critical"
            reasons.append("high_failure_rate")
        elif failure_rate >= 0.10 or failed > 0:
            status = "warning"
            reasons.append("recent_failures")
        return {
            "status": status,
            "reasons": reasons,
            "active_count": active,
            "stale_count": stale,
            "recent_total": total,
            "recent_succeeded": int(raw.get("recent_succeeded") or 0),
            "recent_failed": failed,
            "failure_rate": round(failure_rate, 4),
            "p95_duration_seconds": round(float(raw.get("p95_duration_seconds") or 0), 3),
            "latest_success_at": raw.get("latest_success_at"),
            "status_counts": dict(raw.get("status_counts") or {}),
            "recent_failures": IntegrationOperationsService._public_failures(
                raw.get("recent_failures") or []
            ),
        }

    @staticmethod
    def _search_console_summary(raw: dict[str, Any]) -> dict[str, Any]:
        total = int(raw.get("recent_total") or 0)
        failed = int(raw.get("recent_failed") or 0)
        stale = int(raw.get("stale_count") or 0)
        active = int(raw.get("active_count") or 0)
        truncated = int(raw.get("recent_truncated") or 0)
        connection_counts = dict(raw.get("connection_counts") or {})
        connected = int(connection_counts.get("connected") or 0)
        disconnected = sum(
            int(count or 0)
            for key, count in connection_counts.items()
            if key != "connected"
        )
        failure_rate = failed / total if total else 0.0
        status = "idle" if connected == 0 and total == 0 and active == 0 else "healthy"
        reasons: list[str] = []
        if stale > 0:
            status = "critical"
            reasons.append("stale_sync_runs")
        elif failure_rate >= 0.50 and total >= 2:
            status = "critical"
            reasons.append("high_failure_rate")
        elif (
            disconnected > 0
            or (connected > 0 and raw.get("latest_success_at") is None)
            or failure_rate >= 0.10
            or failed > 0
            or truncated > 0
        ):
            status = "warning"
            if disconnected > 0:
                reasons.append("connection_attention_required")
            if connected > 0 and raw.get("latest_success_at") is None:
                reasons.append("no_successful_sync")
            if failed > 0:
                reasons.append("recent_failures")
            if truncated > 0:
                reasons.append("truncated_results")
        return {
            "status": status,
            "reasons": reasons,
            "connected_count": connected,
            "attention_connection_count": disconnected,
            "active_count": active,
            "stale_count": stale,
            "recent_total": total,
            "recent_succeeded": int(raw.get("recent_succeeded") or 0),
            "recent_failed": failed,
            "recent_truncated": truncated,
            "failure_rate": round(failure_rate, 4),
            "p95_duration_seconds": round(float(raw.get("p95_duration_seconds") or 0), 3),
            "latest_success_at": raw.get("latest_success_at"),
            "connection_counts": connection_counts,
            "status_counts": dict(raw.get("status_counts") or {}),
            "recent_failures": IntegrationOperationsService._public_failures(
                raw.get("recent_failures") or []
            ),
        }

    @staticmethod
    def _public_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for failure in failures[:10]:
            result.append(
                {
                    "id": str(failure.get("id")),
                    "project_id": str(failure.get("project_id")) if failure.get("project_id") else None,
                    "article_id": str(failure.get("article_id")) if failure.get("article_id") else None,
                    "site_url": failure.get("site_url"),
                    "requested_publish_mode": failure.get("requested_publish_mode"),
                    "error_category": str(failure.get("error_category") or "unknown"),
                    "error_message": str(failure.get("error_message") or "Operation failed"),
                    "retry_count": int(failure.get("retry_count") or 0),
                    "updated_at": failure.get("updated_at"),
                }
            )
        return result

    @staticmethod
    def _worst_status(*statuses: str) -> str:
        return max(statuses, key=lambda value: STATUS_RANK.get(value, 2), default="idle")


    @staticmethod
    def _recommendations(
        wordpress: dict[str, Any],
        search_console: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Return deterministic, bounded operator actions in severity order."""
        recommendations: list[dict[str, str]] = []

        if int(wordpress.get("stale_count") or 0) > 0:
            recommendations.append(
                {
                    "priority": "critical",
                    "integration": "wordpress",
                    "code": "run_wordpress_reconciliation",
                    "message": "Reconcile stale WordPress publication attempts before new live publishes.",
                }
            )
        if wordpress.get("status") in {"critical", "warning"}:
            recommendations.append(
                {
                    "priority": "high" if wordpress.get("status") == "critical" else "medium",
                    "integration": "wordpress",
                    "code": "review_wordpress_failures",
                    "message": "Inspect recent WordPress failures and verify the staging connection before retrying.",
                }
            )

        if int(search_console.get("stale_count") or 0) > 0:
            recommendations.append(
                {
                    "priority": "critical",
                    "integration": "search_console",
                    "code": "run_search_console_reconciliation",
                    "message": "Reconcile stale Search Console synchronization runs.",
                }
            )
        if int(search_console.get("attention_connection_count") or 0) > 0:
            recommendations.append(
                {
                    "priority": "high",
                    "integration": "search_console",
                    "code": "reconnect_search_console",
                    "message": "Reconnect Search Console properties that require authorization attention.",
                }
            )
        if "no_successful_sync" in set(search_console.get("reasons") or []):
            recommendations.append(
                {
                    "priority": "medium",
                    "integration": "search_console",
                    "code": "run_initial_search_console_sync",
                    "message": "Run and verify the first successful Search Console synchronization.",
                }
            )
        if int(search_console.get("recent_truncated") or 0) > 0:
            recommendations.append(
                {
                    "priority": "medium",
                    "integration": "search_console",
                    "code": "review_truncated_syncs",
                    "message": "Review Search Console result truncation before acting on incomplete data.",
                }
            )
        if search_console.get("status") in {"critical", "warning"} and not any(
            item["integration"] == "search_console" for item in recommendations
        ):
            recommendations.append(
                {
                    "priority": "high" if search_console.get("status") == "critical" else "medium",
                    "integration": "search_console",
                    "code": "reconnect_search_console",
                    "message": "Inspect recent Search Console failures and quota or authorization status.",
                }
            )

        priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(
            key=lambda item: (priority_rank.get(item["priority"], 4), item["integration"], item["code"])
        )
        return recommendations[:8]
