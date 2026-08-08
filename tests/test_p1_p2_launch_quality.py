from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from infrastructure.integration_metrics import render_integration_snapshot_metrics
from services.integration_operations_service import (
    INTEGRATION_OPERATIONS_SNAPSHOT_KEY,
    IntegrationOperationsService,
)
from services.seo_intelligence_service import SeoIntelligenceService


class FakeSeoPerformanceRepository:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.project_id = uuid4()
        self.article_id = uuid4()
        today = date.today()
        self.articles = [
            {
                "id": self.article_id,
                "project_id": self.project_id,
                "title": "High-value guide",
                "wordpress_post_url": "https://example.com/high-value-guide",
            },
            {
                "id": uuid4(),
                "project_id": self.project_id,
                "title": "Unmeasured article",
                "wordpress_post_url": "https://example.com/unmeasured",
            },
        ]
        self.snapshots = [
            {
                "id": uuid4(),
                "project_id": self.project_id,
                "article_id": self.article_id,
                "url": "https://example.com/high-value-guide",
                "date_from": today - timedelta(days=7),
                "date_to": today - timedelta(days=1),
                "clicks": 45,
                "impressions": 5000,
                "ctr": 0.009,
                "average_position": 5.4,
                "source": "search_console_api",
                "imported_at": datetime.now(timezone.utc),
            },
            {
                "id": uuid4(),
                "project_id": self.project_id,
                "article_id": self.article_id,
                "url": "https://example.com/high-value-guide",
                "date_from": today - timedelta(days=14),
                "date_to": today - timedelta(days=8),
                "clicks": 60,
                "impressions": 4700,
                "ctr": 0.0128,
                "average_position": 5.1,
                "source": "search_console_api",
                "imported_at": datetime.now(timezone.utc),
            },
        ]
        self.opportunities = [
            {
                "id": uuid4(),
                "project_id": self.project_id,
                "article_id": self.article_id,
                "article_title": "High-value guide",
                "url": "https://example.com/high-value-guide",
                "type": "low_ctr_high_impressions",
                "severity": "high",
                "status": "open",
                "reason": "High impressions with low click-through rate.",
                "suggested_action": "Improve title and meta description.",
                "supporting_metrics": {"impressions": 5000, "ctr": 0.009},
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
            {
                "id": uuid4(),
                "project_id": self.project_id,
                "article_id": None,
                "article_title": None,
                "url": "https://example.com/unmapped",
                "type": "unmapped_url",
                "severity": "low",
                "status": "open",
                "reason": "The URL is not mapped.",
                "suggested_action": "Map the URL.",
                "supporting_metrics": {},
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
        ]

    async def project_exists(self, project_id: UUID) -> bool:
        return self.exists and project_id == self.project_id

    async def list_project_articles(self, project_id: UUID):
        assert project_id == self.project_id
        return self.articles

    async def list_snapshots(self, project_id: UUID, *, limit: int = 50):
        assert project_id == self.project_id
        return self.snapshots[:limit]

    async def list_opportunities(
        self, project_id: UUID, *, status: str = "open", limit: int = 50
    ):
        assert project_id == self.project_id
        return [item for item in self.opportunities if item["status"] == status][:limit]


class FakeSeoSearchConsoleRepository:
    def __init__(self, project_id: UUID) -> None:
        self.project_id = project_id
        self.connection = {
            "project_id": project_id,
            "status": "connected",
            "selected_site_url": "sc-domain:example.com",
        }
        self.runs = [
            {
                "id": uuid4(),
                "project_id": project_id,
                "status": "succeeded",
                "truncated": False,
            }
        ]

    async def get_connection(self, project_id: UUID):
        return self.connection if project_id == self.project_id else None

    async def list_sync_runs(self, project_id: UUID, limit: int = 10):
        assert project_id == self.project_id
        return self.runs[:limit]


@pytest.mark.asyncio
async def test_seo_intelligence_is_deterministic_explainable_and_read_only():
    performance = FakeSeoPerformanceRepository()
    search_console = FakeSeoSearchConsoleRepository(performance.project_id)
    service = SeoIntelligenceService(
        performance_repository=performance,
        search_console_repository=search_console,
    )

    first = await service.get_project_intelligence(performance.project_id)
    second = await service.get_project_intelligence(performance.project_id)

    assert first["engine_version"] == "seo-intelligence-v2.0"
    assert first["guardrails"] == {
        "uses_llm": False,
        "performs_network_requests": False,
        "rewrites_content": False,
        "publishes_content": False,
        "explanation_available": True,
    }
    assert first["portfolio"]["measured_article_count"] == 1
    assert first["portfolio"]["coverage_ratio"] == 0.5
    assert first["portfolio"]["trend"]["clicks_change_percent"] == -25.0
    assert first["recommended_queue"][0]["type"] == "low_ctr_high_impressions"
    assert first["recommended_queue"][0]["priority_score"] > first["recommended_queue"][1]["priority_score"]
    assert first["opportunities"][0]["action_plan"]
    assert first["opportunities"][0]["score_factors"]
    assert [item["id"] for item in first["opportunities"]] == [
        item["id"] for item in second["opportunities"]
    ]
    assert [item["priority_score"] for item in first["opportunities"]] == [
        item["priority_score"] for item in second["opportunities"]
    ]


@pytest.mark.asyncio
async def test_seo_intelligence_degrades_confidence_for_stale_data():
    performance = FakeSeoPerformanceRepository()
    stale_date = date.today() - timedelta(days=120)
    for snapshot in performance.snapshots:
        snapshot["date_from"] = stale_date - timedelta(days=7)
        snapshot["date_to"] = stale_date
    search_console = FakeSeoSearchConsoleRepository(performance.project_id)
    service = SeoIntelligenceService(
        performance_repository=performance,
        search_console_repository=search_console,
    )

    result = await service.get_project_intelligence(performance.project_id)

    assert result["data_quality"]["status"] == "insufficient"
    assert result["opportunities"][0]["freshness_factor"] == 0.45
    assert result["opportunities"][0]["confidence"] < 0.6
    assert any(
        warning["code"] == "performance_data_very_stale"
        for warning in result["data_quality"]["warnings"]
    )


@pytest.mark.asyncio
async def test_seo_intelligence_rejects_unknown_project():
    performance = FakeSeoPerformanceRepository(exists=False)
    search_console = FakeSeoSearchConsoleRepository(performance.project_id)
    service = SeoIntelligenceService(
        performance_repository=performance,
        search_console_repository=search_console,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.get_project_intelligence(performance.project_id)

    assert exc_info.value.status_code == 404


class FakeOperationsRepository:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def get_operational_summary(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class FakeCache:
    def __init__(self, *, fail_set: bool = False, fail_get: bool = False) -> None:
        self.fail_set = fail_set
        self.fail_get = fail_get
        self.values: dict[str, object] = {}
        self.ttls: dict[str, int] = {}

    async def set(self, key: str, value: object, ttl: int | None = None) -> bool:
        if self.fail_set:
            raise RuntimeError("cache unavailable")
        self.values[key] = value
        if ttl is not None:
            self.ttls[key] = ttl
        return True

    async def get(self, key: str):
        if self.fail_get:
            raise RuntimeError("cache unavailable")
        return self.values.get(key)


def _wordpress_payload(**overrides):
    payload = {
        "status_counts": {"queued": 2, "failed": 1, "succeeded": 8},
        "active_count": 2,
        "stale_count": 1,
        "recent_total": 9,
        "recent_succeeded": 8,
        "recent_failed": 1,
        "p95_duration_seconds": 4.25,
        "latest_success_at": datetime.now(timezone.utc),
        "recent_failures": [
            {
                "id": uuid4(),
                "project_id": uuid4(),
                "article_id": uuid4(),
                "error_category": "timeout",
                "error_message": "Request timed out [REDACTED]",
                "retry_count": 2,
            }
        ],
    }
    payload.update(overrides)
    return payload


def _search_console_payload(**overrides):
    payload = {
        "connection_counts": {"connected": 1},
        "status_counts": {"succeeded": 2},
        "active_count": 0,
        "stale_count": 0,
        "recent_total": 2,
        "recent_succeeded": 2,
        "recent_failed": 0,
        "recent_truncated": 0,
        "p95_duration_seconds": 9.75,
        "latest_success_at": datetime.now(timezone.utc),
        "recent_failures": [],
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_integration_operations_prioritizes_stale_work_and_caches_json_safe_snapshot():
    wordpress = FakeOperationsRepository(_wordpress_payload())
    search_console = FakeOperationsRepository(_search_console_payload())
    cache = FakeCache()
    service = IntegrationOperationsService(
        publishing_repository=wordpress,
        search_console_repository=search_console,
        cache=cache,
    )

    result = await service.get_summary(lookback_hours=999)
    cached = await service.get_cached_snapshot()

    assert result["lookback_hours"] == 168
    assert result["overall_status"] == "critical"
    assert result["integrations"]["wordpress"]["status"] == "critical"
    assert result["recommendations"][0]["code"] == "run_wordpress_reconciliation"
    assert cache.ttls[INTEGRATION_OPERATIONS_SNAPSHOT_KEY] == 900
    assert isinstance(cached, dict)
    assert isinstance(cached["generated_at"], str)
    assert cached["integrations"]["wordpress"]["p95_duration_seconds"] == 4.25
    assert wordpress.calls[0]["lookback_hours"] == 168
    assert search_console.calls[0]["recent_limit"] == 10


@pytest.mark.asyncio
async def test_integration_operations_survives_cache_failure_and_flags_unproven_sync():
    wordpress = FakeOperationsRepository(
        _wordpress_payload(
            status_counts={},
            active_count=0,
            stale_count=0,
            recent_total=0,
            recent_succeeded=0,
            recent_failed=0,
            p95_duration_seconds=0,
            latest_success_at=None,
            recent_failures=[],
        )
    )
    search_console = FakeOperationsRepository(
        _search_console_payload(
            status_counts={},
            active_count=0,
            stale_count=0,
            recent_total=0,
            recent_succeeded=0,
            recent_failed=0,
            p95_duration_seconds=0,
            latest_success_at=None,
            recent_failures=[],
        )
    )
    service = IntegrationOperationsService(
        publishing_repository=wordpress,
        search_console_repository=search_console,
        cache=FakeCache(fail_set=True, fail_get=True),
    )

    result = await service.get_summary()

    assert result["overall_status"] == "warning"
    assert result["integrations"]["search_console"]["status"] == "warning"
    assert "no_successful_sync" in result["integrations"]["search_console"]["reasons"]
    assert await service.get_cached_snapshot() is None


@pytest.mark.asyncio
async def test_project_scoped_summary_does_not_replace_global_metrics_snapshot():
    cache = FakeCache()
    service = IntegrationOperationsService(
        publishing_repository=FakeOperationsRepository(_wordpress_payload(stale_count=0)),
        search_console_repository=FakeOperationsRepository(_search_console_payload()),
        cache=cache,
    )

    await service.get_summary(project_id=uuid4())

    assert INTEGRATION_OPERATIONS_SNAPSHOT_KEY not in cache.values


def test_integration_snapshot_metrics_are_fixed_cardinality_and_fail_closed():
    generated_at = datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc)
    snapshot = {
        "generated_at": generated_at.isoformat(),
        "integrations": {
            "wordpress": {
                "status": "critical",
                "status_counts": {"queued": 2, "running": 1, "user-id-123": 999},
                "stale_count": 1,
                "recent_total": 9,
                "recent_succeeded": 8,
                "recent_failed": 1,
                "failure_rate": 1 / 9,
                "p95_duration_seconds": 4.25,
            },
            "search_console": {
                "status": "unexpected-status-from-upstream",
                "status_counts": {},
                "stale_count": 0,
                "recent_total": 2,
                "recent_succeeded": 2,
                "recent_failed": 0,
                "failure_rate": 0,
                "p95_duration_seconds": 9.75,
                "recent_truncated": 1,
            },
            "attacker-controlled": {"status": "critical"},
        },
    }

    output = render_integration_snapshot_metrics(
        snapshot,
        now=generated_at + timedelta(seconds=300),
    )

    assert "integration_snapshot_available 1" in output
    assert "integration_snapshot_age_seconds 300" in output
    assert 'integration_durable_stale_items{integration="wordpress"} 1' in output
    assert 'integration_durable_health{integration="wordpress",status="critical"} 1' in output
    assert 'integration_durable_health{integration="search_console",status="unknown"} 1' in output
    assert 'integration_durable_recent_truncated{integration="search_console"} 1' in output
    assert "attacker-controlled" not in output
    assert "user-id-123" not in output

    unavailable = render_integration_snapshot_metrics({"generated_at": "invalid"})
    assert "integration_snapshot_available 0" in unavailable
