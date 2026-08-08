from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from services.performance_feedback_service import PerformanceFeedbackService


class FakePerformanceRepository:
    def __init__(self, project_id=None, articles=None, previous_snapshots=None):
        self.project_id = project_id or uuid4()
        self.articles = articles or []
        self.snapshots_by_key = {}
        self.previous_snapshots = previous_snapshots or {}
        self.opportunities_by_key = {}
        self.dismissed = set()

    async def project_exists(self, project_id):
        return project_id == self.project_id

    async def list_project_articles(self, project_id):
        assert project_id == self.project_id
        return self.articles

    async def upsert_snapshot(self, snapshot):
        key = (
            snapshot["project_id"],
            snapshot["url"],
            snapshot["date_from"],
            snapshot["date_to"],
            snapshot["source"],
        )
        existing = self.snapshots_by_key.get(key, {})
        stored = {
            **existing,
            **snapshot,
            "id": existing.get("id") or uuid4(),
            "imported_at": datetime.now(timezone.utc),
        }
        self.snapshots_by_key[key] = stored
        return stored

    async def bulk_upsert_snapshots(self, snapshots, *, batch_size=1000):
        return [await self.upsert_snapshot(snapshot) for snapshot in snapshots]

    async def resolve_missing_performance_opportunities(self, *, project_id, article_ids):
        assert project_id == self.project_id
        resolved = 0
        for opportunity in self.opportunities_by_key.values():
            if (
                opportunity.get("type") == "missing_performance_data"
                and opportunity.get("article_id") in article_ids
                and opportunity.get("status") == "open"
            ):
                opportunity["status"] = "resolved"
                resolved += 1
        return resolved

    async def list_snapshots(self, project_id, *, limit=50):
        assert project_id == self.project_id
        return list(self.snapshots_by_key.values())[:limit]

    async def list_article_ids_with_snapshots(self, project_id):
        assert project_id == self.project_id
        return {
            snapshot["article_id"]
            for snapshot in self.snapshots_by_key.values()
            if snapshot.get("article_id")
        }

    async def latest_previous_snapshot(self, *, project_id, url, date_from):
        assert project_id == self.project_id
        return self.previous_snapshots.get(url)

    async def latest_previous_snapshots(self, *, project_id, urls, before_date):
        assert project_id == self.project_id
        return {
            url: snapshot
            for url in urls
            if (snapshot := self.previous_snapshots.get(url)) is not None
            and snapshot.get("date_to", date.min) < before_date
        }

    async def upsert_opportunity(self, opportunity):
        key = (opportunity["project_id"], opportunity["url"], opportunity["type"])
        existing = self.opportunities_by_key.get(key, {})
        stored = {
            **existing,
            **opportunity,
            "id": existing.get("id") or uuid4(),
            "status": "open",
            "created_at": existing.get("created_at") or datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        self.opportunities_by_key[key] = stored
        return stored

    async def bulk_upsert_opportunities(self, opportunities, *, batch_size=1000):
        return [await self.upsert_opportunity(opportunity) for opportunity in opportunities]

    async def resolve_opportunity(self, *, project_id, article_id, opportunity_type):
        assert project_id == self.project_id
        for key, opportunity in self.opportunities_by_key.items():
            if (
                key[2] == opportunity_type
                and opportunity.get("article_id") == article_id
                and opportunity.get("status") == "open"
            ):
                opportunity["status"] = "resolved"

    async def dismiss_opportunity(self, *, project_id, opportunity_id):
        assert project_id == self.project_id
        self.dismissed.add(opportunity_id)
        return True

    async def list_opportunities(self, project_id, *, status="open", limit=50):
        assert project_id == self.project_id
        return [
            opportunity
            for opportunity in self.opportunities_by_key.values()
            if not status or opportunity.get("status") == status
        ][:limit]


def csv_payload(*rows: str) -> str:
    return "\n".join([
        "url,clicks,impressions,ctr,average_position,date_from,date_to",
        *rows,
    ])


@pytest.mark.asyncio
async def test_import_persists_snapshot_and_maps_url_to_article():
    project_id = uuid4()
    article_id = uuid4()
    repository = FakePerformanceRepository(
        project_id=project_id,
        articles=[{
            "id": article_id,
            "project_id": project_id,
            "title": "Mapped article",
            "wordpress_post_url": "https://example.com/posts/mapped/",
        }],
    )
    service = PerformanceFeedbackService(repository)

    result = await service.import_csv(
        project_id=project_id,
        csv_text=csv_payload("https://example.com/posts/mapped,40,2000,0.02,9.5,2026-06-01,2026-06-07"),
    )

    assert result["imported_count"] == 1
    assert len(repository.snapshots_by_key) == 1
    snapshot = next(iter(repository.snapshots_by_key.values()))
    assert snapshot["article_id"] == article_id
    assert snapshot["url"] == "https://example.com/posts/mapped"


@pytest.mark.asyncio
async def test_duplicate_import_updates_single_snapshot():
    project_id = uuid4()
    repository = FakePerformanceRepository(project_id=project_id)
    service = PerformanceFeedbackService(repository)
    first = csv_payload("https://example.com/a,10,1000,0.01,15,2026-06-01,2026-06-07")
    second = csv_payload("https://example.com/a,20,1000,0.02,15,2026-06-01,2026-06-07")

    await service.import_csv(project_id=project_id, csv_text=first)
    await service.import_csv(project_id=project_id, csv_text=second)

    assert len(repository.snapshots_by_key) == 1
    snapshot = next(iter(repository.snapshots_by_key.values()))
    assert snapshot["clicks"] == 20
    assert snapshot["ctr"] == 0.02




@pytest.mark.asyncio
async def test_duplicate_rows_in_one_import_are_collapsed_before_bulk_upsert():
    project_id = uuid4()
    repository = FakePerformanceRepository(project_id=project_id)
    service = PerformanceFeedbackService(repository)

    result = await service.import_csv(
        project_id=project_id,
        csv_text=csv_payload(
            "https://example.com/a,10,1000,0.01,15,2026-06-01,2026-06-07",
            "https://example.com/a,20,1000,0.02,14,2026-06-01,2026-06-07",
        ),
    )

    assert result["imported_count"] == 1
    assert result["deduplicated_input_count"] == 1
    assert len(repository.snapshots_by_key) == 1
    snapshot = next(iter(repository.snapshots_by_key.values()))
    assert snapshot["clicks"] == 20
    assert snapshot["average_position"] == 14.0


@pytest.mark.asyncio
async def test_low_ctr_high_impressions_opportunity_is_rule_based():
    project_id = uuid4()
    repository = FakePerformanceRepository(project_id=project_id)
    service = PerformanceFeedbackService(repository)

    await service.import_csv(
        project_id=project_id,
        csv_text=csv_payload("https://example.com/low-ctr,12,6000,0.4%,5,2026-06-01,2026-06-07"),
    )

    opportunities = await repository.list_opportunities(project_id)
    low_ctr = [item for item in opportunities if item["type"] == "low_ctr_high_impressions"]
    assert low_ctr
    assert low_ctr[0]["severity"] == "high"
    assert "title and meta" in low_ctr[0]["suggested_action"]


@pytest.mark.asyncio
async def test_striking_distance_position_opportunity():
    project_id = uuid4()
    repository = FakePerformanceRepository(project_id=project_id)
    service = PerformanceFeedbackService(repository)

    await service.import_csv(
        project_id=project_id,
        csv_text=csv_payload("https://example.com/rank-12,25,900,0.03,12,2026-06-01,2026-06-07"),
    )

    opportunities = await repository.list_opportunities(project_id)
    assert any(item["type"] == "striking_distance_position" for item in opportunities)


@pytest.mark.asyncio
async def test_declining_clicks_opportunity_uses_previous_snapshot():
    project_id = uuid4()
    url = "https://example.com/decline"
    repository = FakePerformanceRepository(
        project_id=project_id,
        previous_snapshots={url: {"clicks": 100, "date_to": date(2026, 5, 31)}},
    )
    service = PerformanceFeedbackService(repository)

    await service.import_csv(
        project_id=project_id,
        csv_text=csv_payload(f"{url},60,1200,0.05,7,2026-06-01,2026-06-07"),
    )

    opportunities = await repository.list_opportunities(project_id)
    decline = [item for item in opportunities if item["type"] == "declining_clicks"]
    assert decline
    assert decline[0]["supporting_metrics"]["previous_clicks"] == 100
    assert decline[0]["supporting_metrics"]["current_clicks"] == 60


@pytest.mark.asyncio
async def test_unmapped_url_and_missing_performance_data_are_reported():
    project_id = uuid4()
    article_id = uuid4()
    repository = FakePerformanceRepository(
        project_id=project_id,
        articles=[{
            "id": article_id,
            "project_id": project_id,
            "title": "No data yet",
            "wordpress_post_url": "https://example.com/no-data",
        }],
    )
    service = PerformanceFeedbackService(repository)

    await service.import_csv(
        project_id=project_id,
        csv_text=csv_payload("https://example.com/unmapped,5,100,0.05,22,2026-06-01,2026-06-07"),
    )

    opportunities = await repository.list_opportunities(project_id)
    types = {item["type"] for item in opportunities}
    assert "unmapped_url" in types
    assert "missing_performance_data" in types


@pytest.mark.asyncio
async def test_invalid_csv_is_rejected_without_persistence():
    project_id = uuid4()
    repository = FakePerformanceRepository(project_id=project_id)
    service = PerformanceFeedbackService(repository)

    with pytest.raises(HTTPException) as exc:
        await service.import_csv(
            project_id=project_id,
            csv_text=csv_payload("not-a-url,NaN,100,0.01,9,2026-06-01,2026-06-07"),
        )

    assert exc.value.status_code == 400
    assert repository.snapshots_by_key == {}


@pytest.mark.asyncio
async def test_csv_article_id_must_belong_to_project():
    project_id = uuid4()
    repository = FakePerformanceRepository(project_id=project_id, articles=[])
    service = PerformanceFeedbackService(repository)
    foreign_article_id = uuid4()
    payload = "\n".join([
        "article_id,url,clicks,impressions,ctr,average_position,date_from,date_to",
        f"{foreign_article_id},https://example.com/a,10,100,0.1,4,2026-06-01,2026-06-07",
    ])

    with pytest.raises(HTTPException) as exc:
        await service.import_csv(project_id=project_id, csv_text=payload)

    assert exc.value.status_code == 400
    assert "article_id" in exc.value.detail


@pytest.mark.asyncio
async def test_read_model_has_stable_api_shape_without_llm_dependency():
    project_id = uuid4()
    repository = FakePerformanceRepository(project_id=project_id)
    service = PerformanceFeedbackService(repository)
    await service.import_csv(
        project_id=project_id,
        csv_text=csv_payload("https://example.com/a,10,1000,0.01,15,2026-06-01,2026-06-07"),
    )

    payload = await service.get_project_performance(project_id)

    assert payload["project_id"] == str(project_id)
    assert set(payload.keys()) == {"project_id", "summary", "snapshots", "opportunities"}
    assert payload["summary"]["snapshot_count"] == 1
    assert payload["snapshots"][0]["url"] == "https://example.com/a"
    assert payload["opportunities"]
