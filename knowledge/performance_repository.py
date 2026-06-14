"""Repository for read-only content performance feedback data."""

from datetime import date, datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import and_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from infrastructure.database import DatabaseManager
from infrastructure.schema import (
    content_improvement_opportunities_table,
    content_performance_snapshots_table,
    generated_articles_table,
    projects_table,
)


class PerformanceRepository:
    """Data access for manual performance snapshots and rule-based opportunities."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def project_exists(self, project_id: UUID) -> bool:
        row = await self.db.fetch_one(
            select(projects_table.c.id).where(projects_table.c.id == project_id)
        )
        return row is not None

    async def list_project_articles(self, project_id: UUID) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            select(
                generated_articles_table.c.id,
                generated_articles_table.c.project_id,
                generated_articles_table.c.title,
                generated_articles_table.c.wordpress_post_url,
            )
            .where(generated_articles_table.c.project_id == project_id)
            .order_by(generated_articles_table.c.created_at.desc())
        )
        return [dict(row) for row in rows]

    async def upsert_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        values = {
            **snapshot,
            "id": snapshot.get("id") or uuid4(),
            "imported_at": now,
        }
        insert_stmt = pg_insert(content_performance_snapshots_table).values(values)
        update_values = {
            "article_id": insert_stmt.excluded.article_id,
            "clicks": insert_stmt.excluded.clicks,
            "impressions": insert_stmt.excluded.impressions,
            "ctr": insert_stmt.excluded.ctr,
            "average_position": insert_stmt.excluded.average_position,
            "imported_at": now,
        }
        query = (
            insert_stmt.on_conflict_do_update(
                index_elements=[
                    content_performance_snapshots_table.c.project_id,
                    content_performance_snapshots_table.c.url,
                    content_performance_snapshots_table.c.date_from,
                    content_performance_snapshots_table.c.date_to,
                    content_performance_snapshots_table.c.source,
                ],
                set_=update_values,
            )
            .returning(content_performance_snapshots_table)
        )
        async with self.db.session() as session:
            result = await session.execute(query)
            row = result.mappings().one()
        return dict(row)

    async def list_snapshots(
        self,
        project_id: UUID,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            select(content_performance_snapshots_table)
            .where(content_performance_snapshots_table.c.project_id == project_id)
            .order_by(
                content_performance_snapshots_table.c.date_to.desc(),
                content_performance_snapshots_table.c.imported_at.desc(),
            )
            .limit(limit)
        )
        return [dict(row) for row in rows]

    async def list_article_ids_with_snapshots(self, project_id: UUID) -> set[UUID]:
        rows = await self.db.fetch_all(
            select(content_performance_snapshots_table.c.article_id)
            .where(
                content_performance_snapshots_table.c.project_id == project_id,
                content_performance_snapshots_table.c.article_id.is_not(None),
            )
            .distinct()
        )
        return {row["article_id"] for row in rows if row.get("article_id")}

    async def latest_previous_snapshot(
        self,
        *,
        project_id: UUID,
        url: str,
        date_from: date,
    ) -> Optional[dict[str, Any]]:
        row = await self.db.fetch_one(
            select(content_performance_snapshots_table)
            .where(
                content_performance_snapshots_table.c.project_id == project_id,
                content_performance_snapshots_table.c.url == url,
                content_performance_snapshots_table.c.date_to < date_from,
            )
            .order_by(content_performance_snapshots_table.c.date_to.desc())
            .limit(1)
        )
        return dict(row) if row else None

    async def upsert_opportunity(self, opportunity: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        values = {
            **opportunity,
            "id": opportunity.get("id") or uuid4(),
            "status": opportunity.get("status") or "open",
            "created_at": now,
            "updated_at": now,
        }
        insert_stmt = pg_insert(content_improvement_opportunities_table).values(values)
        query = (
            insert_stmt.on_conflict_do_update(
                index_elements=[
                    content_improvement_opportunities_table.c.project_id,
                    content_improvement_opportunities_table.c.url,
                    content_improvement_opportunities_table.c.type,
                ],
                set_={
                    "article_id": insert_stmt.excluded.article_id,
                    "snapshot_id": insert_stmt.excluded.snapshot_id,
                    "severity": insert_stmt.excluded.severity,
                    "reason": insert_stmt.excluded.reason,
                    "suggested_action": insert_stmt.excluded.suggested_action,
                    "supporting_metrics": insert_stmt.excluded.supporting_metrics,
                    "status": "open",
                    "updated_at": now,
                },
            )
            .returning(content_improvement_opportunities_table)
        )
        async with self.db.session() as session:
            result = await session.execute(query)
            row = result.mappings().one()
        return dict(row)

    async def resolve_opportunity(
        self,
        *,
        project_id: UUID,
        article_id: UUID,
        opportunity_type: str,
    ) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.execute(
            update(content_improvement_opportunities_table)
            .where(
                content_improvement_opportunities_table.c.project_id == project_id,
                content_improvement_opportunities_table.c.article_id == article_id,
                content_improvement_opportunities_table.c.type == opportunity_type,
                content_improvement_opportunities_table.c.status == "open",
            )
            .values(status="resolved", updated_at=now)
        )

    async def dismiss_opportunity(self, *, project_id: UUID, opportunity_id: UUID) -> bool:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        result = await self.db.execute(
            update(content_improvement_opportunities_table)
            .where(
                content_improvement_opportunities_table.c.project_id == project_id,
                content_improvement_opportunities_table.c.id == opportunity_id,
            )
            .values(status="dismissed", updated_at=now)
        )
        return result.rowcount > 0 if hasattr(result, "rowcount") else True

    async def list_opportunities(
        self,
        project_id: UUID,
        *,
        status: str = "open",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        filters = [content_improvement_opportunities_table.c.project_id == project_id]
        if status:
            filters.append(content_improvement_opportunities_table.c.status == status)
        rows = await self.db.fetch_all(
            select(
                content_improvement_opportunities_table,
                generated_articles_table.c.title.label("article_title"),
            )
            .select_from(
                content_improvement_opportunities_table.outerjoin(
                    generated_articles_table,
                    content_improvement_opportunities_table.c.article_id
                    == generated_articles_table.c.id,
                )
            )
            .where(and_(*filters))
            .order_by(
                content_improvement_opportunities_table.c.updated_at.desc(),
                content_improvement_opportunities_table.c.created_at.desc(),
            )
            .limit(limit)
        )
        return [dict(row) for row in rows]
