"""Durable finalization for accepted content-generation tasks.

LLM calls are external side effects and cannot be part of a database
transaction. This module makes the part we control atomic: one persisted
article, one project-counter/cost update, one durable task result, and one
post-commit export request for each Celery task id.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult

from core.exceptions import WorkflowError
from core.models import GeneratedArticle
from infrastructure.database import DatabaseManager
from infrastructure.schema import (
    generated_articles_table,
    generation_outbox_events_table,
    projects_table,
)
from orchestration.task_persistence import TaskStatus, task_results_table

EXPORT_REQUEST_EVENT = "article.export.requested"
SOCIAL_DRAFT_REQUEST_EVENT = "article.social_drafts.requested"
EXPORT_LEASE_DURATION = timedelta(minutes=10)
SOCIAL_DISPATCH_LEASE_DURATION = timedelta(minutes=10)
SOCIAL_TASK_NAME = "orchestration.tasks.generate_social_posts_task"


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _task_duration(start_time: datetime | None, end_time: datetime) -> float | None:
    if start_time is None:
        return None
    return max(0.0, (_naive_utc(end_time) - _naive_utc(start_time)).total_seconds())


@dataclass(frozen=True)
class FinalizationOutcome:
    """The durable result for a newly finalized or replayed task."""

    article_id: UUID
    result: dict[str, Any]
    newly_finalized: bool


@dataclass(frozen=True)
class PendingExport:
    """A post-commit filesystem export request and its immutable article data."""

    event_id: UUID
    attempt_number: int
    article: Any
    language: str


@dataclass(frozen=True)
class PendingSocialDispatch:
    """A durable social-draft dispatch request with one logical child ID."""

    event_id: UUID
    attempt_number: int
    parent_task_id: str
    social_task_id: str
    task_kwargs: dict[str, Any]


def social_task_id_for(parent_task_id: str) -> str:
    """Return the stable Celery ID used for every delivery attempt."""
    return str(uuid5(NAMESPACE_URL, f"smarlux:social-drafts:{parent_task_id}"))


class GenerationFinalizationRepository:
    """Atomically persist accepted generation output and its task audit record."""

    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager

    async def get_finalized_result(self, task_id: str) -> FinalizationOutcome | None:
        """Return a completed result only when this task owns a persisted article."""
        query = (
            select(generated_articles_table.c.id, task_results_table.c.result)
            .select_from(
                generated_articles_table.join(
                    task_results_table,
                    task_results_table.c.task_id == generated_articles_table.c.generation_task_id,
                )
            )
            .where(generated_articles_table.c.generation_task_id == task_id)
            .where(task_results_table.c.status == TaskStatus.SUCCESS.value)
        )
        async with self.db.read_session() as session:
            row = (await session.execute(query)).mappings().one_or_none()

        if row is None or not isinstance(row["result"], Mapping):
            return None
        return FinalizationOutcome(
            article_id=row["id"],
            result=dict(row["result"]),
            newly_finalized=False,
        )

    async def finalize(
        self,
        *,
        task_id: str,
        article: GeneratedArticle,
        task_result: Mapping[str, Any],
        language: str,
        social_request: Mapping[str, Any] | None = None,
    ) -> FinalizationOutcome:
        """Commit article, accounting, task success, and export request together."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        async with self.db.transaction() as session:
            existing_article_id = (
                await session.execute(
                    select(generated_articles_table.c.id)
                    .where(generated_articles_table.c.generation_task_id == task_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if existing_article_id is not None:
                return await self._existing_outcome(session, task_id, existing_article_id)

            inserted_article_id = (
                await session.execute(
                    pg_insert(generated_articles_table)
                    .values(
                        id=article.id,
                        generation_task_id=task_id,
                        project_id=article.project_id,
                        content_plan_id=article.content_plan_id,
                        title=article.title,
                        content=article.content,
                        meta_description=article.meta_description,
                        keywords=article.keywords or [],
                        word_count=article.quality_metrics.word_count,
                        readability_score=article.quality_metrics.readability_score,
                        keyword_density=article.quality_metrics.keyword_density,
                        total_tokens_used=article.total_tokens_used,
                        total_cost=article.total_cost_usd,
                        generation_time=article.generation_time_seconds,
                        created_at=_naive_utc(article.created_at),
                        updated_at=_naive_utc(article.updated_at),
                    )
                    .on_conflict_do_nothing(
                        index_elements=["generation_task_id"],
                        index_where=generated_articles_table.c.generation_task_id.is_not(None),
                    )
                    .returning(generated_articles_table.c.id)
                )
            ).scalar_one_or_none()

            if inserted_article_id is None:
                existing_article_id = (
                    await session.execute(
                        select(generated_articles_table.c.id)
                        .where(generated_articles_table.c.generation_task_id == task_id)
                        .with_for_update()
                    )
                ).scalar_one()
                return await self._existing_outcome(session, task_id, existing_article_id)

            counter_update = cast(
                CursorResult[Any],
                await session.execute(
                    update(projects_table)
                    .where(projects_table.c.id == article.project_id)
                    .values(
                        total_articles_generated=func.coalesce(
                            projects_table.c.total_articles_generated, 0
                        )
                        + 1,
                        total_tokens_consumed=func.coalesce(
                            projects_table.c.total_tokens_consumed, 0
                        )
                        + article.total_tokens_used,
                        total_cost_usd=func.coalesce(projects_table.c.total_cost_usd, 0)
                        + article.total_cost_usd,
                        last_active=_naive_utc(article.created_at),
                        updated_at=now,
                    )
                ),
            )
            if counter_update.rowcount != 1:
                raise WorkflowError(
                    f"Project {article.project_id} was not found while finalizing content generation."
                )

            start_time = (
                await session.execute(
                    select(task_results_table.c.start_time)
                    .where(task_results_table.c.task_id == task_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            result = dict(task_result)
            result["article_id"] = str(inserted_article_id)
            result["content_id"] = str(inserted_article_id)
            result["status"] = "success"
            if social_request is not None:
                result["social_task_id"] = social_task_id_for(task_id)
                result["social_dispatch_status"] = "pending"
            task_update = cast(
                CursorResult[Any],
                await session.execute(
                    update(task_results_table)
                    .where(task_results_table.c.task_id == task_id)
                    .values(
                        status=TaskStatus.SUCCESS.value,
                        result=result,
                        error=None,
                        traceback=None,
                        end_time=now,
                        duration_seconds=_task_duration(start_time, now),
                        updated_at=now,
                    )
                ),
            )
            if task_update.rowcount != 1:
                raise WorkflowError(
                    f"Task record {task_id} is missing; accepted content was not finalized."
                )

            await session.execute(
                pg_insert(generation_outbox_events_table)
                .values(
                    id=uuid4(),
                    task_id=task_id,
                    article_id=inserted_article_id,
                    event_type=EXPORT_REQUEST_EVENT,
                    payload={"article_id": str(inserted_article_id), "language": language},
                    status="pending",
                    attempt_count=0,
                    available_at=now,
                    created_at=now,
                    updated_at=now,
                )
                .on_conflict_do_nothing(index_elements=["task_id", "event_type"])
            )
            if social_request is not None:
                await self._prepare_social_dispatch_in_session(
                    session,
                    task_id=task_id,
                    article_id=inserted_article_id,
                    social_request=social_request,
                    now=now,
                )

        return FinalizationOutcome(
            article_id=inserted_article_id,
            result=result,
            newly_finalized=True,
        )

    async def ensure_social_dispatch(
        self,
        *,
        task_id: str,
        article_id: UUID,
        social_request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Repair or confirm the durable social hand-off on task replay."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            parent_result = (
                await session.execute(
                    select(task_results_table.c.result)
                    .where(task_results_table.c.task_id == task_id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if not isinstance(parent_result, Mapping):
                raise WorkflowError(
                    f"Task record {task_id} has no durable result for social dispatch."
                )

            await self._prepare_social_dispatch_in_session(
                session,
                task_id=task_id,
                article_id=article_id,
                social_request=social_request,
                now=now,
            )
            event = (
                await session.execute(
                    select(
                        generation_outbox_events_table.c.status,
                        generation_outbox_events_table.c.last_error,
                    )
                    .where(generation_outbox_events_table.c.task_id == task_id)
                    .where(
                        generation_outbox_events_table.c.event_type
                        == SOCIAL_DRAFT_REQUEST_EVENT
                    )
                )
            ).mappings().one_or_none()

            result = dict(parent_result)
            result["social_task_id"] = social_task_id_for(task_id)
            if event and event["status"] == "completed":
                result["social_dispatch_status"] = "dispatched"
                result.pop("social_dispatch_error", None)
            elif event and event["last_error"]:
                result["social_dispatch_status"] = "retry_pending"
                result["social_dispatch_error"] = str(event["last_error"])[:500]
            else:
                result["social_dispatch_status"] = "pending"
            await session.execute(
                update(task_results_table)
                .where(task_results_table.c.task_id == task_id)
                .where(task_results_table.c.status == TaskStatus.SUCCESS.value)
                .values(result=result, updated_at=now)
            )
        return result

    async def claim_pending_export(self, task_id: str) -> PendingExport | None:
        """Claim a durable export request after article finalization has committed.

        A worker owns an event only while its lease is active. This prevents
        concurrent Celery redeliveries from writing the same export paths, yet
        makes a claim recoverable when its worker crashes before completion.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        lease_expires_at = now + EXPORT_LEASE_DURATION
        eligible_for_claim = or_(
            generation_outbox_events_table.c.status == "pending",
            and_(
                generation_outbox_events_table.c.status == "processing",
                generation_outbox_events_table.c.available_at <= now,
            ),
        )
        query = (
            select(
                generation_outbox_events_table.c.id.label("event_id"),
                generation_outbox_events_table.c.payload,
                generation_outbox_events_table.c.attempt_count,
                generated_articles_table.c.id,
                generated_articles_table.c.project_id,
                generated_articles_table.c.title,
                generated_articles_table.c.content,
                generated_articles_table.c.meta_description,
                generated_articles_table.c.keywords,
                generated_articles_table.c.word_count,
                generated_articles_table.c.total_cost,
                generated_articles_table.c.generation_time,
                generated_articles_table.c.created_at,
            )
            .select_from(
                generation_outbox_events_table.join(
                    generated_articles_table,
                    generated_articles_table.c.id == generation_outbox_events_table.c.article_id,
                )
            )
            .where(generation_outbox_events_table.c.task_id == task_id)
            .where(generation_outbox_events_table.c.event_type == EXPORT_REQUEST_EVENT)
            .where(eligible_for_claim)
            .with_for_update(skip_locked=True)
        )
        async with self.db.transaction() as session:
            row = (await session.execute(query)).mappings().one_or_none()
            if row is None:
                return None
            next_attempt_number = int(row["attempt_count"] or 0) + 1
            await session.execute(
                update(generation_outbox_events_table)
                .where(generation_outbox_events_table.c.id == row["event_id"])
                .where(generation_outbox_events_table.c.attempt_count == row["attempt_count"])
                .values(
                    status="processing",
                    attempt_count=next_attempt_number,
                    available_at=lease_expires_at,
                    updated_at=now,
                )
            )

        payload = row["payload"] if isinstance(row["payload"], Mapping) else {}
        return PendingExport(
            event_id=row["event_id"],
            attempt_number=next_attempt_number,
            article=SimpleNamespace(
                id=row["id"],
                project_id=row["project_id"],
                title=row["title"],
                content=row["content"],
                meta_description=row["meta_description"],
                keywords=row["keywords"] or [],
                quality_metrics=SimpleNamespace(word_count=row["word_count"] or 0),
                total_cost_usd=float(row["total_cost"] or 0.0),
                generation_time_seconds=float(row["generation_time"] or 0.0),
                created_at=row["created_at"],
            ),
            language=str(payload.get("language") or "fa"),
        )

    async def complete_export(self, event_id: UUID, attempt_number: int) -> None:
        """Mark a durable export event complete only after the write succeeds."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            await session.execute(
                update(generation_outbox_events_table)
                .where(generation_outbox_events_table.c.id == event_id)
                .where(generation_outbox_events_table.c.status == "processing")
                .where(generation_outbox_events_table.c.attempt_count == attempt_number)
                .values(
                    status="completed",
                    completed_at=now,
                    available_at=None,
                    last_error=None,
                    updated_at=now,
                )
            )

    async def record_export_failure(self, event_id: UUID, attempt_number: int, error: str) -> None:
        """Keep a failed export pending for recovery without failing content."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            await session.execute(
                update(generation_outbox_events_table)
                .where(generation_outbox_events_table.c.id == event_id)
                .where(generation_outbox_events_table.c.status == "processing")
                .where(generation_outbox_events_table.c.attempt_count == attempt_number)
                .values(
                    status="pending",
                    available_at=now,
                    last_error=error[:2000],
                    updated_at=now,
                )
            )

    async def claim_pending_social_dispatch(
        self,
        task_id: str,
    ) -> PendingSocialDispatch | None:
        """Claim pending/recoverable dispatch with a stable logical child ID.

        A processing event remains reclaimable because the worker may have died
        before broker publication. Duplicate broker publication is possible,
        but every attempt uses the same child ID and the child reuses its
        persisted successful result.
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        lease_expires_at = now + SOCIAL_DISPATCH_LEASE_DURATION
        eligible_for_claim = or_(
            generation_outbox_events_table.c.status == "pending",
            and_(
                generation_outbox_events_table.c.status == "processing",
                generation_outbox_events_table.c.available_at <= now,
            ),
        )
        query = (
            select(
                generation_outbox_events_table.c.id.label("event_id"),
                generation_outbox_events_table.c.payload,
                generation_outbox_events_table.c.attempt_count,
            )
            .where(generation_outbox_events_table.c.task_id == task_id)
            .where(
                generation_outbox_events_table.c.event_type
                == SOCIAL_DRAFT_REQUEST_EVENT
            )
            .where(eligible_for_claim)
            .with_for_update(skip_locked=True)
        )
        async with self.db.transaction() as session:
            row = (await session.execute(query)).mappings().one_or_none()
            if row is None:
                return None
            next_attempt_number = int(row["attempt_count"] or 0) + 1
            claim = cast(
                CursorResult[Any],
                await session.execute(
                    update(generation_outbox_events_table)
                    .where(generation_outbox_events_table.c.id == row["event_id"])
                    .where(generation_outbox_events_table.c.attempt_count == row["attempt_count"])
                    .values(
                        status="processing",
                        attempt_count=next_attempt_number,
                        available_at=lease_expires_at,
                        updated_at=now,
                    )
                ),
            )
            if claim.rowcount != 1:
                return None

        payload = row["payload"] if isinstance(row["payload"], Mapping) else {}
        task_kwargs = payload.get("task_kwargs")
        if not isinstance(task_kwargs, Mapping):
            raise WorkflowError(
                f"Social dispatch event for task {task_id} has invalid task metadata."
            )
        return PendingSocialDispatch(
            event_id=row["event_id"],
            attempt_number=next_attempt_number,
            parent_task_id=task_id,
            social_task_id=str(payload.get("social_task_id") or social_task_id_for(task_id)),
            task_kwargs=dict(task_kwargs),
        )

    async def complete_social_dispatch(
        self,
        event_id: UUID,
        attempt_number: int,
        parent_task_id: str,
    ) -> None:
        """Record broker acceptance without changing the accepted article status."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            completed = cast(
                CursorResult[Any],
                await session.execute(
                    update(generation_outbox_events_table)
                    .where(generation_outbox_events_table.c.id == event_id)
                    .where(generation_outbox_events_table.c.status == "processing")
                    .where(generation_outbox_events_table.c.attempt_count == attempt_number)
                    .values(
                        status="completed",
                        completed_at=now,
                        available_at=None,
                        last_error=None,
                        updated_at=now,
                    )
                ),
            )
            if completed.rowcount == 1:
                await self._update_parent_social_state(
                    session,
                    parent_task_id=parent_task_id,
                    dispatch_status="dispatched",
                    error=None,
                    now=now,
                )

    async def record_social_dispatch_failure(
        self,
        event_id: UUID,
        attempt_number: int,
        parent_task_id: str,
        error: str,
    ) -> None:
        """Retain a failed dispatch for replay while preserving article success."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        bounded_error = error[:2000]
        async with self.db.transaction() as session:
            failed = cast(
                CursorResult[Any],
                await session.execute(
                    update(generation_outbox_events_table)
                    .where(generation_outbox_events_table.c.id == event_id)
                    .where(generation_outbox_events_table.c.status == "processing")
                    .where(generation_outbox_events_table.c.attempt_count == attempt_number)
                    .values(
                        status="pending",
                        available_at=now,
                        last_error=bounded_error,
                        updated_at=now,
                    )
                ),
            )
            if failed.rowcount == 1:
                await self._update_parent_social_state(
                    session,
                    parent_task_id=parent_task_id,
                    dispatch_status="retry_pending",
                    error=bounded_error,
                    now=now,
                )

    @staticmethod
    async def _prepare_social_dispatch_in_session(
        session: Any,
        *,
        task_id: str,
        article_id: UUID,
        social_request: Mapping[str, Any],
        now: datetime,
    ) -> None:
        social_task_id = social_task_id_for(task_id)
        task_kwargs = {
            "article_id": str(article_id),
            "title": str(social_request.get("title") or ""),
            "topic": str(social_request.get("topic") or ""),
            "language": str(social_request.get("language") or "fa"),
            "submitted_by_user_id": social_request.get("submitted_by_user_id"),
            "parent_task_id": task_id,
        }
        await session.execute(
            pg_insert(task_results_table)
            .values(
                id=uuid4(),
                task_id=social_task_id,
                task_name=SOCIAL_TASK_NAME,
                idempotency_key=f"social-drafts:{task_id}",
                status=TaskStatus.PENDING.value,
                args=[],
                kwargs=task_kwargs,
                retry_count=0,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=["task_id"])
        )
        await session.execute(
            pg_insert(generation_outbox_events_table)
            .values(
                id=uuid4(),
                task_id=task_id,
                article_id=article_id,
                event_type=SOCIAL_DRAFT_REQUEST_EVENT,
                payload={
                    "social_task_id": social_task_id,
                    "task_kwargs": task_kwargs,
                },
                status="pending",
                attempt_count=0,
                available_at=now,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=["task_id", "event_type"])
        )

    @staticmethod
    async def _update_parent_social_state(
        session: Any,
        *,
        parent_task_id: str,
        dispatch_status: str,
        error: str | None,
        now: datetime,
    ) -> None:
        parent_result = (
            await session.execute(
                select(task_results_table.c.result)
                .where(task_results_table.c.task_id == parent_task_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if not isinstance(parent_result, Mapping):
            return
        result = dict(parent_result)
        result["social_task_id"] = social_task_id_for(parent_task_id)
        result["social_dispatch_status"] = dispatch_status
        if error:
            result["social_dispatch_error"] = error[:500]
        else:
            result.pop("social_dispatch_error", None)
        await session.execute(
            update(task_results_table)
            .where(task_results_table.c.task_id == parent_task_id)
            .where(task_results_table.c.status == TaskStatus.SUCCESS.value)
            .values(result=result, updated_at=now)
        )


    @staticmethod
    async def _existing_outcome(
        session: Any,
        task_id: str,
        article_id: UUID,
    ) -> FinalizationOutcome:
        result = (
            await session.execute(
                select(task_results_table.c.result)
                .where(task_results_table.c.task_id == task_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if not isinstance(result, Mapping):
            raise WorkflowError(
                "A persisted generation article has no durable completed task result; "
                "manual reconciliation is required before retrying."
            )
        return FinalizationOutcome(
            article_id=article_id,
            result=dict(result),
            newly_finalized=False,
        )
