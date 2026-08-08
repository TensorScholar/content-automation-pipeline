"""Durable, idempotent WordPress publishing state persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, desc, func, insert, or_, select, update

from infrastructure.database import DatabaseManager
from infrastructure.redaction import redact_text
from infrastructure.schema import generated_articles_table, publishing_attempts_table

ACTIVE_ATTEMPT_STATES = {"queued", "running", "retrying"}


@dataclass(frozen=True)
class PublishClaim:
    claimed: bool
    article: dict[str, Any] | None
    attempt_id: UUID
    reason: str | None = None
    attempt: dict[str, Any] | None = None


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PublishingRepository:
    """Repository for durable WordPress side-effect state transitions."""

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
        scheduled_at: datetime | None = None,
        lease_seconds: int = 900,
    ) -> PublishClaim:
        """Atomically claim one idempotency key before any WordPress side effect."""
        now = _utc_now_naive()
        lease_expires_at = now + timedelta(seconds=lease_seconds)

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
                return PublishClaim(False, None, uuid4(), "article_not_found")

            success_result = await session.execute(
                select(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.article_id == article_id,
                    publishing_attempts_table.c.idempotency_key == idempotency_key,
                    publishing_attempts_table.c.success.is_(True),
                )
                .order_by(desc(publishing_attempts_table.c.finished_at))
                .limit(1)
            )
            success_row = success_result.fetchone()
            if success_row:
                existing = dict(success_row._mapping)
                return PublishClaim(
                    False,
                    article,
                    UUID(str(existing["id"])),
                    "already_succeeded",
                    existing,
                )

            # Serialize all active side effects for an article, not only identical
            # keys. This prevents two status/content revisions from racing remotely.
            active_result = await session.execute(
                select(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.article_id == article_id,
                    publishing_attempts_table.c.status.in_(ACTIVE_ATTEMPT_STATES),
                    or_(
                        publishing_attempts_table.c.lease_expires_at.is_(None),
                        publishing_attempts_table.c.lease_expires_at > now,
                    ),
                )
                .order_by(desc(publishing_attempts_table.c.created_at))
                .limit(1)
            )
            active_row = active_result.fetchone()
            if active_row:
                existing = dict(active_row._mapping)
                reason = (
                    "already_in_progress"
                    if existing.get("idempotency_key") == idempotency_key
                    else "article_publish_busy"
                )
                return PublishClaim(
                    False,
                    article,
                    UUID(str(existing["id"])),
                    reason,
                    existing,
                )

            # No live attempt owns the article. Retire expired active rows before
            # creating a new claim so the periodic reconciler cannot later revive
            # an obsolete task beside the new remote side effect.
            await session.execute(
                update(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.article_id == article_id,
                    publishing_attempts_table.c.status.in_(ACTIVE_ATTEMPT_STATES),
                    publishing_attempts_table.c.lease_expires_at.is_not(None),
                    publishing_attempts_table.c.lease_expires_at <= now,
                )
                .values(
                    finished_at=now,
                    status="superseded",
                    success=False,
                    error_category="stale_replaced_by_new_claim",
                    error_message="Expired publishing attempt was replaced by a new claim",
                    lease_expires_at=None,
                    updated_at=now,
                )
            )

            attempt_id = uuid4()
            attempt_status = "queued" if task_id else "running"
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
                    status=attempt_status,
                    lease_expires_at=lease_expires_at,
                    updated_at=now,
                    created_at=now,
                )
            )
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    publish_status="publish_queued" if task_id else "publishing",
                    publish_idempotency_key=idempotency_key,
                    publish_task_id=task_id,
                    publish_requested_status=requested_publish_mode,
                    publish_scheduled_at=scheduled_at,
                    publish_lease_expires_at=lease_expires_at,
                    publish_error_category=None,
                    publish_error_message=None,
                    publish_attempt_count=generated_articles_table.c.publish_attempt_count + 1,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )
            return PublishClaim(
                True,
                article,
                attempt_id,
                attempt={
                    "id": attempt_id,
                    "status": attempt_status,
                    "task_id": task_id,
                    "requested_publish_mode": requested_publish_mode,
                    "idempotency_key": idempotency_key,
                },
            )

    async def mark_attempt_running(
        self,
        *,
        attempt_id: UUID,
        task_id: str | None,
        lease_seconds: int = 900,
    ) -> bool:
        """Atomically acquire execution ownership for a queued/retrying attempt."""
        now = _utc_now_naive()
        lease = now + timedelta(seconds=lease_seconds)
        async with self.db.transaction() as session:
            result = await session.execute(
                update(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.id == attempt_id,
                    publishing_attempts_table.c.task_id == task_id,
                    publishing_attempts_table.c.success.is_(False),
                    publishing_attempts_table.c.status.in_({"queued", "retrying"}),
                )
                .values(status="running", lease_expires_at=lease, updated_at=now)
                .returning(publishing_attempts_table.c.article_id)
            )
            row = result.fetchone()
            if not row:
                return False
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == row[0])
                .values(
                    publish_status="publishing",
                    publish_lease_expires_at=lease,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )
            return True

    async def record_retry(
        self,
        *,
        article_id: UUID,
        attempt_id: UUID,
        error_category: str,
        error_message: str,
        retry_count: int,
        task_id: str | None,
        lease_seconds: int = 900,
    ) -> bool:
        now = _utc_now_naive()
        lease = now + timedelta(seconds=lease_seconds)
        safe_message = redact_text(error_message)
        async with self.db.transaction() as session:
            await session.execute(
                select(generated_articles_table.c.id)
                .where(generated_articles_table.c.id == article_id)
                .with_for_update()
            )
            attempt_result = await session.execute(
                select(
                    publishing_attempts_table.c.success,
                    publishing_attempts_table.c.status,
                    publishing_attempts_table.c.task_id,
                )
                .where(publishing_attempts_table.c.id == attempt_id)
                .with_for_update()
            )
            attempt_row = attempt_result.fetchone()
            if not attempt_row:
                return False
            current_attempt = dict(attempt_row._mapping)
            if current_attempt.get("task_id") != task_id:
                return False
            if current_attempt.get("success") or current_attempt.get("status") in {
                "succeeded",
                "failed",
                "superseded",
            }:
                return False
            successful_result = await session.execute(
                select(publishing_attempts_table.c.id)
                .where(
                    publishing_attempts_table.c.article_id == article_id,
                    publishing_attempts_table.c.id != attempt_id,
                    publishing_attempts_table.c.success.is_(True),
                )
                .limit(1)
            )
            if successful_result.fetchone():
                # A delayed worker must never regress an article after another
                # attempt has committed a verified remote success.
                await session.execute(
                    update(publishing_attempts_table)
                    .where(publishing_attempts_table.c.id == attempt_id)
                    .values(
                        finished_at=now,
                        success=False,
                        status="superseded",
                        retry_count=retry_count,
                        error_category="late_retry_after_success",
                        error_message=(
                            "Retry suppressed because another publishing attempt already succeeded"
                        ),
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                return False

            await session.execute(
                update(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.id == attempt_id,
                    publishing_attempts_table.c.status.in_(ACTIVE_ATTEMPT_STATES),
                    publishing_attempts_table.c.success.is_(False),
                )
                .values(
                    status="retrying",
                    retry_count=retry_count,
                    error_category=error_category,
                    error_message=safe_message,
                    lease_expires_at=lease,
                    updated_at=now,
                )
            )
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    publish_status="publish_retrying",
                    publish_error_category=error_category,
                    publish_error_message=safe_message,
                    publish_lease_expires_at=lease,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )
            return True

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
                    status="failed",
                    error_category=error_category,
                    error_message=safe_message,
                    updated_at=now,
                    created_at=now,
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
        task_id: str | None,
        warnings: list[dict[str, str]] | None = None,
        remote_verified_at: datetime | None = None,
    ) -> bool:
        now = _utc_now_naive()
        final_status = {
            "draft": "published_as_draft",
            "future": "published_scheduled",
            "publish": "published_public",
        }.get(wordpress_status, "published_as_draft")
        async with self.db.transaction() as session:
            # The article lock serializes competing worker completions. This makes
            # the partial unique success index deterministic rather than a race.
            await session.execute(
                select(generated_articles_table.c.id)
                .where(generated_articles_table.c.id == article_id)
                .with_for_update()
            )
            attempt_result = await session.execute(
                select(publishing_attempts_table)
                .where(publishing_attempts_table.c.id == attempt_id)
                .with_for_update()
            )
            attempt_row = attempt_result.fetchone()
            if not attempt_row:
                return False
            attempt = dict(attempt_row._mapping)
            if attempt.get("task_id") != task_id:
                return False
            if attempt.get("success") or attempt.get("status") == "succeeded":
                return True
            if attempt.get("status") not in ACTIVE_ATTEMPT_STATES:
                return False
            previous_result = await session.execute(
                select(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.article_id == article_id,
                    publishing_attempts_table.c.idempotency_key == attempt["idempotency_key"],
                    publishing_attempts_table.c.success.is_(True),
                    publishing_attempts_table.c.id != attempt_id,
                )
                .order_by(desc(publishing_attempts_table.c.finished_at))
                .limit(1)
            )
            previous_row = previous_result.fetchone()
            if previous_row:
                previous = dict(previous_row._mapping)
                await session.execute(
                    update(publishing_attempts_table)
                    .where(publishing_attempts_table.c.id == attempt_id)
                    .values(
                        finished_at=now,
                        success=False,
                        status="superseded",
                        final_wordpress_status=previous.get("final_wordpress_status"),
                        wordpress_post_id=previous.get("wordpress_post_id"),
                        wordpress_post_url=previous.get("wordpress_post_url"),
                        retry_count=retry_count,
                        warnings=warnings or [],
                        error_category="idempotent_completion",
                        error_message="An equivalent publishing attempt already succeeded",
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                wordpress_status = str(previous.get("final_wordpress_status") or wordpress_status)
                wordpress_post_id = previous.get("wordpress_post_id")
                wordpress_post_url = previous.get("wordpress_post_url")
                final_status = {
                    "draft": "published_as_draft",
                    "future": "published_scheduled",
                    "publish": "published_public",
                }.get(wordpress_status, "published_as_draft")
            else:
                await session.execute(
                    update(publishing_attempts_table)
                    .where(publishing_attempts_table.c.id == attempt_id)
                    .values(
                        finished_at=now,
                        success=True,
                        status="succeeded",
                        final_wordpress_status=wordpress_status,
                        wordpress_post_id=str(wordpress_post_id) if wordpress_post_id else None,
                        wordpress_post_url=wordpress_post_url,
                        retry_count=retry_count,
                        warnings=warnings or [],
                        remote_verified_at=(remote_verified_at or datetime.now(timezone.utc)).replace(tzinfo=None),
                        lease_expires_at=None,
                        error_category=None,
                        error_message=None,
                        updated_at=now,
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
                    publish_task_id=None,
                    publish_lease_expires_at=None,
                    publish_error_category=None,
                    publish_error_message=None,
                    publish_updated_at=now,
                    distributed_at=now,
                    distribution_channels=["wordpress"],
                    updated_at=now,
                )
            )
            return True

    async def record_failure(
        self,
        *,
        article_id: UUID,
        attempt_id: UUID,
        error_category: str,
        error_message: str,
        retry_count: int,
        task_id: str | None,
    ) -> bool:
        now = _utc_now_naive()
        safe_message = redact_text(error_message)
        async with self.db.transaction() as session:
            await session.execute(
                select(generated_articles_table.c.id)
                .where(generated_articles_table.c.id == article_id)
                .with_for_update()
            )
            attempt_result = await session.execute(
                select(
                    publishing_attempts_table.c.success,
                    publishing_attempts_table.c.status,
                    publishing_attempts_table.c.task_id,
                )
                .where(publishing_attempts_table.c.id == attempt_id)
                .with_for_update()
            )
            attempt_row = attempt_result.fetchone()
            if not attempt_row:
                return False
            current_attempt = dict(attempt_row._mapping)
            if current_attempt.get("task_id") != task_id:
                return False
            if current_attempt.get("success") or current_attempt.get("status") in {
                "succeeded",
                "failed",
                "superseded",
            }:
                return False
            successful_result = await session.execute(
                select(publishing_attempts_table.c.id)
                .where(
                    publishing_attempts_table.c.article_id == article_id,
                    publishing_attempts_table.c.id != attempt_id,
                    publishing_attempts_table.c.success.is_(True),
                )
                .limit(1)
            )
            if successful_result.fetchone():
                await session.execute(
                    update(publishing_attempts_table)
                    .where(publishing_attempts_table.c.id == attempt_id)
                    .values(
                        finished_at=now,
                        success=False,
                        status="superseded",
                        error_category="late_failure_after_success",
                        error_message=(
                            "Failure suppressed because another publishing attempt already succeeded"
                        ),
                        retry_count=retry_count,
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                return False

            await session.execute(
                update(publishing_attempts_table)
                .where(publishing_attempts_table.c.id == attempt_id)
                .values(
                    finished_at=now,
                    success=False,
                    status="failed",
                    error_category=error_category,
                    error_message=safe_message,
                    retry_count=retry_count,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    publish_status="publish_failed",
                    publish_task_id=None,
                    publish_lease_expires_at=None,
                    publish_error_category=error_category,
                    publish_error_message=safe_message,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )
            return True

    async def get_attempt(self, attempt_id: UUID) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            select(publishing_attempts_table).where(publishing_attempts_table.c.id == attempt_id)
        )
        return dict(row) if row else None

    async def list_stale_attempts(self, *, limit: int = 100) -> list[dict[str, Any]]:
        now = _utc_now_naive()
        rows = await self.db.fetch_all(
            select(publishing_attempts_table)
            .where(
                publishing_attempts_table.c.status.in_(ACTIVE_ATTEMPT_STATES),
                publishing_attempts_table.c.lease_expires_at.is_not(None),
                publishing_attempts_table.c.lease_expires_at <= now,
            )
            .order_by(publishing_attempts_table.c.lease_expires_at.asc())
            .limit(limit)
        )
        return [dict(row) for row in rows]

    async def requeue_stale_attempt(
        self,
        *,
        attempt_id: UUID,
        task_id: str,
        lease_seconds: int = 900,
    ) -> bool:
        """Recover one stale attempt using the global article-then-attempt lock order."""
        now = _utc_now_naive()
        lease = now + timedelta(seconds=lease_seconds)
        # Read the article identity without a lock, then acquire locks in the same
        # order as completion/retry/failure paths. The attempt is revalidated after
        # both locks are held, avoiding article/attempt deadlocks.
        preliminary = await self.db.fetch_one(
            select(publishing_attempts_table.c.article_id).where(
                publishing_attempts_table.c.id == attempt_id
            )
        )
        if not preliminary:
            return False
        article_id = UUID(str(preliminary["article_id"]))

        async with self.db.transaction() as session:
            article_result = await session.execute(
                select(generated_articles_table.c.id)
                .where(generated_articles_table.c.id == article_id)
                .with_for_update()
            )
            if not article_result.fetchone():
                return False

            attempt_result = await session.execute(
                select(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.id == attempt_id,
                    publishing_attempts_table.c.article_id == article_id,
                )
                .with_for_update()
            )
            attempt_row = attempt_result.fetchone()
            if not attempt_row:
                return False
            attempt = dict(attempt_row._mapping)
            if attempt.get("success") or attempt.get("status") not in ACTIVE_ATTEMPT_STATES:
                return False
            expires_at = attempt.get("lease_expires_at")
            if expires_at is None or expires_at > now:
                return False

            successful_result = await session.execute(
                select(publishing_attempts_table.c.id)
                .where(
                    publishing_attempts_table.c.article_id == article_id,
                    publishing_attempts_table.c.id != attempt_id,
                    publishing_attempts_table.c.success.is_(True),
                )
                .limit(1)
            )
            if successful_result.fetchone():
                await session.execute(
                    update(publishing_attempts_table)
                    .where(publishing_attempts_table.c.id == attempt_id)
                    .values(
                        finished_at=now,
                        status="superseded",
                        success=False,
                        error_category="stale_after_success",
                        error_message=(
                            "Stale attempt was not requeued because another attempt succeeded"
                        ),
                        lease_expires_at=None,
                        updated_at=now,
                    )
                )
                return False

            result = await session.execute(
                update(publishing_attempts_table)
                .where(
                    publishing_attempts_table.c.id == attempt_id,
                    publishing_attempts_table.c.status.in_(ACTIVE_ATTEMPT_STATES),
                    publishing_attempts_table.c.success.is_(False),
                    publishing_attempts_table.c.lease_expires_at <= now,
                )
                .values(
                    task_id=task_id,
                    status="queued",
                    lease_expires_at=lease,
                    updated_at=now,
                )
                .returning(publishing_attempts_table.c.article_id)
            )
            row = result.fetchone()
            if not row:
                return False
            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == row[0])
                .values(
                    publish_task_id=task_id,
                    publish_status="publish_queued",
                    publish_lease_expires_at=lease,
                    publish_updated_at=now,
                    updated_at=now,
                )
            )
            return True

    async def get_operational_summary(
        self,
        *,
        project_id: UUID | None = None,
        lookback_hours: int = 24,
        recent_limit: int = 10,
    ) -> dict[str, Any]:
        """Return bounded WordPress operational signals without high-cardinality labels."""
        now = _utc_now_naive()
        cutoff = now - timedelta(hours=max(1, min(lookback_hours, 168)))
        filters = []
        if project_id is not None:
            filters.append(publishing_attempts_table.c.project_id == project_id)

        status_query = (
            select(
                publishing_attempts_table.c.status,
                func.count().label("count"),
            )
            .where(
                or_(
                    publishing_attempts_table.c.status.in_(ACTIVE_ATTEMPT_STATES),
                    publishing_attempts_table.c.created_at >= cutoff,
                ),
                *filters,
            )
            .group_by(publishing_attempts_table.c.status)
        )
        status_rows = await self.db.fetch_all(status_query)
        status_counts = {str(row["status"]): int(row["count"] or 0) for row in status_rows}

        recent_filters = [
            publishing_attempts_table.c.created_at >= cutoff,
            publishing_attempts_table.c.status.in_({"succeeded", "failed"}),
            *filters,
        ]
        duration_seconds = func.extract(
            "epoch",
            publishing_attempts_table.c.finished_at
            - publishing_attempts_table.c.started_at,
        )
        recent_row = await self.db.fetch_one(
            select(
                func.count().label("total"),
                func.count().filter(publishing_attempts_table.c.success.is_(True)).label("succeeded"),
                func.count().filter(publishing_attempts_table.c.status == "failed").label("failed"),
                func.max(publishing_attempts_table.c.finished_at)
                .filter(publishing_attempts_table.c.success.is_(True))
                .label("latest_success_at"),
                func.percentile_cont(0.95)
                .within_group(duration_seconds)
                .label("p95_duration_seconds"),
            ).where(*recent_filters)
        )
        recent = dict(recent_row) if recent_row else {}

        stale_filters = [
            publishing_attempts_table.c.status.in_(ACTIVE_ATTEMPT_STATES),
            publishing_attempts_table.c.lease_expires_at.is_not(None),
            publishing_attempts_table.c.lease_expires_at <= now,
            *filters,
        ]
        stale_row = await self.db.fetch_one(
            select(func.count().label("count")).where(*stale_filters)
        )
        stale_count = int(stale_row["count"] or 0) if stale_row else 0

        failure_filters = [
            publishing_attempts_table.c.status == "failed",
            publishing_attempts_table.c.updated_at >= cutoff,
            *filters,
        ]
        failure_rows = await self.db.fetch_all(
            select(
                publishing_attempts_table.c.id,
                publishing_attempts_table.c.project_id,
                publishing_attempts_table.c.article_id,
                publishing_attempts_table.c.requested_publish_mode,
                publishing_attempts_table.c.error_category,
                publishing_attempts_table.c.error_message,
                publishing_attempts_table.c.retry_count,
                publishing_attempts_table.c.updated_at,
            )
            .where(*failure_filters)
            .order_by(publishing_attempts_table.c.updated_at.desc())
            .limit(max(1, min(recent_limit, 50)))
        )
        return {
            "status_counts": status_counts,
            "active_count": sum(status_counts.get(state, 0) for state in ACTIVE_ATTEMPT_STATES),
            "stale_count": stale_count,
            "recent_total": int(recent.get("total") or 0),
            "recent_succeeded": int(recent.get("succeeded") or 0),
            "recent_failed": int(recent.get("failed") or 0),
            "latest_success_at": recent.get("latest_success_at"),
            "p95_duration_seconds": float(recent.get("p95_duration_seconds") or 0),
            "recent_failures": [
                {**dict(row), "error_message": redact_text(str(row.get("error_message") or ""))}
                for row in failure_rows
            ],
        }

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
                generated_articles_table.c.publish_task_id,
                generated_articles_table.c.publish_requested_status,
                generated_articles_table.c.publish_scheduled_at,
                generated_articles_table.c.publish_lease_expires_at,
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
                publishing_attempts_table.c.status,
                publishing_attempts_table.c.task_id,
                publishing_attempts_table.c.error_category,
                publishing_attempts_table.c.error_message,
                publishing_attempts_table.c.retry_count,
                publishing_attempts_table.c.warnings,
                publishing_attempts_table.c.remote_verified_at,
                publishing_attempts_table.c.lease_expires_at,
            )
            .where(publishing_attempts_table.c.article_id == article_id)
            .order_by(desc(publishing_attempts_table.c.created_at))
            .limit(5)
        )
        result = dict(article)
        result["recent_attempts"] = [dict(row) for row in attempts]
        return result
