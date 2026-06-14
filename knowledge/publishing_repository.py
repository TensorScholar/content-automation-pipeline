"""Durable publishing state and audit trail persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import desc, insert, select, update

from infrastructure.database import DatabaseManager
from infrastructure.redaction import redact_text
from infrastructure.schema import generated_articles_table, publishing_attempts_table


@dataclass(frozen=True)
class PublishClaim:
    claimed: bool
    article: dict[str, Any] | None
    attempt_id: UUID
    reason: str | None = None


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PublishingRepository:
    """Repository for idempotent WordPress publish state transitions."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def claim_publish(
        self,
        *,
        article_id: UUID,
        project_id: UUID,
        user_id: UUID | None,
        requested_publish_mode: str,
        idempotency_key: str,
        target_site_url: str | None,
        task_id: str | None = None,
        correlation_id: str | None = None,
    ) -> PublishClaim:
        """Atomically claim an article before the external WordPress side effect."""
        attempt_id = uuid4()
        now = _utc_now_naive()

        async with self.db.transaction() as session:
            article_result = await session.execute(
                select(generated_articles_table)
                .where(
                    generated_articles_table.c.id == article_id,
                    generated_articles_table.c.project_id == project_id,
                )
                .with_for_update()
            )
            row = article_result.fetchone()
            article = dict(row._mapping) if row else None

            if not article:
                return PublishClaim(
                    claimed=False,
                    article=None,
                    attempt_id=attempt_id,
                    reason="article_not_found",
                )

            await session.execute(
                insert(publishing_attempts_table).values(
                    id=attempt_id,
                    article_id=article_id,
                    project_id=project_id,
                    user_id=user_id,
                    target_site_url=target_site_url,
                    requested_publish_mode=requested_publish_mode,
                    idempotency_key=idempotency_key,
                    started_at=now,
                    task_id=task_id,
                    correlation_id=correlation_id,
                )
            )

            if article.get("publish_status") == "publishing":
                await session.execute(
                    update(publishing_attempts_table)
                    .where(publishing_attempts_table.c.id == attempt_id)
                    .values(
                        finished_at=now,
                        success=False,
                        error_category="already_publishing",
                        error_message="Another publish attempt is already in progress",
                    )
                )
                return PublishClaim(
                    claimed=False,
                    article=article,
                    attempt_id=attempt_id,
                    reason="already_publishing",
                )

            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    publish_status="publishing",
                    publish_idempotency_key=idempotency_key,
                    publish_error_category=None,
                    publish_error_message=None,
                    publish_attempt_count=generated_articles_table.c.publish_attempt_count + 1,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )

            return PublishClaim(
                claimed=True,
                article=article,
                attempt_id=attempt_id,
            )

    async def record_preflight_failure(
        self,
        *,
        article_id: UUID,
        project_id: UUID,
        user_id: UUID | None,
        requested_publish_mode: str,
        idempotency_key: str,
        target_site_url: str | None,
        error_category: str,
        error_message: str,
    ) -> UUID:
        attempt_id = uuid4()
        now = _utc_now_naive()
        safe_message = redact_text(error_message)

        async with self.db.transaction() as session:
            await session.execute(
                insert(publishing_attempts_table).values(
                    id=attempt_id,
                    article_id=article_id,
                    project_id=project_id,
                    user_id=user_id,
                    target_site_url=target_site_url,
                    requested_publish_mode=requested_publish_mode,
                    idempotency_key=idempotency_key,
                    started_at=now,
                    finished_at=now,
                    success=False,
                    error_category=error_category,
                    error_message=safe_message,
                )
            )
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    publish_status="publish_validation_failed",
                    publish_error_category=error_category,
                    publish_error_message=safe_message,
                    publish_idempotency_key=idempotency_key,
                    publish_attempt_count=generated_articles_table.c.publish_attempt_count + 1,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )
        return attempt_id

    async def record_success(
        self,
        *,
        article_id: UUID,
        attempt_id: UUID,
        wordpress_status: str,
        wordpress_post_id: str | int | None,
        wordpress_post_url: str | None,
        retry_count: int,
    ) -> None:
        now = _utc_now_naive()
        final_status = {
            "draft": "published_as_draft",
            "future": "published_scheduled",
            "publish": "published_public",
        }.get(wordpress_status, "published_as_draft")

        async with self.db.transaction() as session:
            await session.execute(
                update(publishing_attempts_table)
                .where(publishing_attempts_table.c.id == attempt_id)
                .values(
                    finished_at=now,
                    success=True,
                    final_wordpress_status=wordpress_status,
                    wordpress_post_id=str(wordpress_post_id) if wordpress_post_id else None,
                    wordpress_post_url=wordpress_post_url,
                    retry_count=retry_count,
                )
            )
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    publish_status=final_status,
                    wordpress_post_id=str(wordpress_post_id) if wordpress_post_id else None,
                    wordpress_post_url=wordpress_post_url,
                    wordpress_post_status=wordpress_status,
                    wordpress_published_at=now,
                    publish_error_category=None,
                    publish_error_message=None,
                    publish_updated_at=now,
                    distributed_at=now,
                    distribution_channels=["wordpress"],
                    updated_at=now,
                )
            )

    async def record_failure(
        self,
        *,
        article_id: UUID,
        attempt_id: UUID,
        error_category: str,
        error_message: str,
        retry_count: int,
    ) -> None:
        now = _utc_now_naive()
        safe_message = redact_text(error_message)

        async with self.db.transaction() as session:
            await session.execute(
                update(publishing_attempts_table)
                .where(publishing_attempts_table.c.id == attempt_id)
                .values(
                    finished_at=now,
                    success=False,
                    error_category=error_category,
                    error_message=safe_message,
                    retry_count=retry_count,
                )
            )
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    publish_status="publish_failed",
                    publish_error_category=error_category,
                    publish_error_message=safe_message,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )

    async def get_publish_status(self, article_id: UUID) -> dict[str, Any] | None:
        article = await self.db.fetch_one(
            select(
                generated_articles_table.c.id,
                generated_articles_table.c.project_id,
                generated_articles_table.c.publish_status,
                generated_articles_table.c.wordpress_post_id,
                generated_articles_table.c.wordpress_post_url,
                generated_articles_table.c.wordpress_post_status,
                generated_articles_table.c.wordpress_published_at,
                generated_articles_table.c.publish_error_category,
                generated_articles_table.c.publish_error_message,
                generated_articles_table.c.publish_attempt_count,
                generated_articles_table.c.publish_updated_at,
            ).where(generated_articles_table.c.id == article_id)
        )
        if not article:
            return None

        attempts = await self.db.fetch_all(
            select(
                publishing_attempts_table.c.id,
                publishing_attempts_table.c.requested_publish_mode,
                publishing_attempts_table.c.final_wordpress_status,
                publishing_attempts_table.c.wordpress_post_id,
                publishing_attempts_table.c.wordpress_post_url,
                publishing_attempts_table.c.started_at,
                publishing_attempts_table.c.finished_at,
                publishing_attempts_table.c.success,
                publishing_attempts_table.c.error_category,
                publishing_attempts_table.c.error_message,
                publishing_attempts_table.c.retry_count,
            )
            .where(publishing_attempts_table.c.article_id == article_id)
            .order_by(desc(publishing_attempts_table.c.started_at))
            .limit(5)
        )
        result = dict(article)
        result["recent_attempts"] = [dict(row) for row in attempts]
        return result
