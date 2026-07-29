"""Production-path tests for durable Social Draft dispatch."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from orchestration.generation_finalization import (
    SOCIAL_DRAFT_REQUEST_EVENT,
    SOCIAL_TASK_NAME,
    GenerationFinalizationRepository,
    PendingSocialDispatch,
    social_task_id_for,
)
from orchestration.task_persistence import SyncTaskResultRepository, TaskStatus
from orchestration.tasks import (
    _deliver_pending_social_dispatch,
    generate_social_posts_task,
)

PARENT_TASK_ID = "generation-task-001"
ARTICLE_ID = uuid4()
SOCIAL_TASK_ID = social_task_id_for(PARENT_TASK_ID)


class _DatabaseResult:
    def __init__(self, *, value=None, mapping=None, rowcount=1):
        self.value = value
        self.mapping = mapping
        self.rowcount = rowcount

    def scalar_one_or_none(self):
        return self.value

    def mappings(self):
        return self

    def one_or_none(self):
        return self.mapping


def _dispatch() -> PendingSocialDispatch:
    return PendingSocialDispatch(
        event_id=uuid4(),
        attempt_number=1,
        parent_task_id=PARENT_TASK_ID,
        social_task_id=SOCIAL_TASK_ID,
        task_kwargs={
            "article_id": str(ARTICLE_ID),
            "title": "Durable article",
            "topic": "durability",
            "language": "en",
            "submitted_by_user_id": "user-123",
            "parent_task_id": PARENT_TASK_ID,
        },
    )


@pytest.mark.asyncio
async def test_production_dispatch_uses_stable_child_id_and_completes_outbox():
    repository = MagicMock()
    repository.claim_pending_social_dispatch = AsyncMock(return_value=_dispatch())
    repository.complete_social_dispatch = AsyncMock()
    repository.record_social_dispatch_failure = AsyncMock()
    repository.get_finalized_result = AsyncMock(
        return_value=SimpleNamespace(
            result={
                "status": "success",
                "social_task_id": SOCIAL_TASK_ID,
                "social_dispatch_status": "dispatched",
            }
        )
    )

    with patch.object(generate_social_posts_task, "apply_async") as apply_async:
        result = await _deliver_pending_social_dispatch(
            repository,
            task_id=PARENT_TASK_ID,
            current_result={"status": "success"},
        )

    apply_async.assert_called_once_with(
        kwargs=_dispatch().task_kwargs,
        task_id=SOCIAL_TASK_ID,
        queue="default",
        routing_key="default",
    )
    repository.complete_social_dispatch.assert_awaited_once()
    repository.record_social_dispatch_failure.assert_not_awaited()
    assert result["status"] == "success"
    assert result["social_task_id"] == SOCIAL_TASK_ID
    assert result["social_dispatch_status"] == "dispatched"


@pytest.mark.asyncio
async def test_production_dispatch_failure_remains_retryable_and_preserves_article_success():
    dispatch = _dispatch()
    repository = MagicMock()
    repository.claim_pending_social_dispatch = AsyncMock(return_value=dispatch)
    repository.complete_social_dispatch = AsyncMock()
    repository.record_social_dispatch_failure = AsyncMock()
    repository.get_finalized_result = AsyncMock(
        return_value=SimpleNamespace(
            result={
                "status": "success",
                "social_task_id": SOCIAL_TASK_ID,
                "social_dispatch_status": "retry_pending",
                "social_dispatch_error": "broker unavailable",
            }
        )
    )

    with patch.object(
        generate_social_posts_task,
        "apply_async",
        side_effect=ConnectionError("broker unavailable"),
    ):
        result = await _deliver_pending_social_dispatch(
            repository,
            task_id=PARENT_TASK_ID,
            current_result={"status": "success"},
        )

    repository.complete_social_dispatch.assert_not_awaited()
    repository.record_social_dispatch_failure.assert_awaited_once_with(
        dispatch.event_id,
        dispatch.attempt_number,
        dispatch.parent_task_id,
        "broker unavailable",
    )
    assert result["status"] == "success"
    assert result["social_dispatch_status"] == "retry_pending"


@pytest.mark.asyncio
async def test_redelivery_without_claim_does_not_duplicate_broker_dispatch():
    repository = MagicMock()
    repository.claim_pending_social_dispatch = AsyncMock(return_value=None)
    repository.get_finalized_result = AsyncMock(
        return_value=SimpleNamespace(
            result={
                "status": "success",
                "social_task_id": SOCIAL_TASK_ID,
                "social_dispatch_status": "dispatched",
            }
        )
    )

    with patch.object(generate_social_posts_task, "apply_async") as apply_async:
        result = await _deliver_pending_social_dispatch(
            repository,
            task_id=PARENT_TASK_ID,
            current_result={"status": "success"},
        )

    apply_async.assert_not_called()
    assert result["social_task_id"] == SOCIAL_TASK_ID


@pytest.mark.asyncio
async def test_prepare_social_dispatch_persists_link_ownership_and_outbox_together():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=1))
    now = SimpleNamespace()

    await GenerationFinalizationRepository._prepare_social_dispatch_in_session(
        session,
        task_id=PARENT_TASK_ID,
        article_id=ARTICLE_ID,
        social_request={
            "title": "Durable article",
            "topic": "durability",
            "language": "fa",
            "submitted_by_user_id": "user-123",
        },
        now=now,
    )

    assert session.execute.await_count == 2
    child_insert = session.execute.await_args_list[0].args[0]
    outbox_insert = session.execute.await_args_list[1].args[0]
    child_params = child_insert.compile().params
    outbox_params = outbox_insert.compile().params

    assert child_params["task_id"] == SOCIAL_TASK_ID
    assert child_params["task_name"] == SOCIAL_TASK_NAME
    assert child_params["status"] == TaskStatus.PENDING.value
    assert child_params["kwargs"]["submitted_by_user_id"] == "user-123"
    assert child_params["kwargs"]["parent_task_id"] == PARENT_TASK_ID
    assert outbox_params["event_type"] == SOCIAL_DRAFT_REQUEST_EVENT
    assert outbox_params["payload"]["social_task_id"] == SOCIAL_TASK_ID
    assert outbox_params["payload"]["task_kwargs"] == child_params["kwargs"]


@pytest.mark.asyncio
async def test_replay_repairs_missing_social_handoff_without_regenerating_article():
    parent_result = {"status": "success", "article_id": str(ARTICLE_ID)}
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            _DatabaseResult(value=parent_result),
            _DatabaseResult(),
            _DatabaseResult(),
            _DatabaseResult(mapping={"status": "pending", "last_error": None}),
            _DatabaseResult(),
        ]
    )

    @asynccontextmanager
    async def transaction():
        yield session

    repository = GenerationFinalizationRepository(
        SimpleNamespace(transaction=transaction)
    )
    repaired = await repository.ensure_social_dispatch(
        task_id=PARENT_TASK_ID,
        article_id=ARTICLE_ID,
        social_request={
            "title": "Durable article",
            "topic": "durability",
            "language": "en",
            "submitted_by_user_id": "user-123",
        },
    )

    assert repaired["status"] == "success"
    assert repaired["article_id"] == str(ARTICLE_ID)
    assert repaired["social_task_id"] == SOCIAL_TASK_ID
    assert repaired["social_dispatch_status"] == "pending"
    parent_update = session.execute.await_args_list[-1].args[0]
    assert parent_update.compile().params["result"] == repaired


@pytest.mark.asyncio
async def test_expired_processing_dispatch_is_reclaimable_with_same_logical_child_id():
    dispatch = _dispatch()
    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[
            _DatabaseResult(
                mapping={
                    "event_id": dispatch.event_id,
                    "payload": {
                        "social_task_id": dispatch.social_task_id,
                        "task_kwargs": dispatch.task_kwargs,
                    },
                    "attempt_count": 1,
                }
            ),
            _DatabaseResult(rowcount=1),
        ]
    )

    @asynccontextmanager
    async def transaction():
        yield session

    repository = GenerationFinalizationRepository(
        SimpleNamespace(transaction=transaction)
    )
    reclaimed = await repository.claim_pending_social_dispatch(PARENT_TASK_ID)

    assert reclaimed is not None
    assert reclaimed.social_task_id == SOCIAL_TASK_ID
    assert reclaimed.attempt_number == 2
    claim_query = session.execute.await_args_list[0].args[0]
    assert claim_query._for_update_arg.skip_locked is True
    assert "generation_outbox_events.available_at <=" in str(claim_query)
    claim_update = session.execute.await_args_list[1].args[0]
    assert claim_update.compile().params["available_at"] > claim_update.compile().params[
        "updated_at"
    ]


def test_social_task_persists_terminal_result_and_reuses_completed_result():
    first_repository = MagicMock()
    first_repository.get_task_by_id.return_value = {
        "task_id": SOCIAL_TASK_ID,
        "status": TaskStatus.PENDING.value,
    }

    with patch.dict(
        generate_social_posts_task._orig_run.__globals__,
        {"SyncTaskResultRepository": MagicMock(return_value=first_repository)},
    ):
        generate_social_posts_task.push_request(id=SOCIAL_TASK_ID, hostname="worker-1")
        try:
            first_result = generate_social_posts_task.run(
                article_id=str(ARTICLE_ID),
                title="Durable article",
                topic="durability",
                language="en",
                submitted_by_user_id="user-123",
                parent_task_id=PARENT_TASK_ID,
            )
        finally:
            generate_social_posts_task.pop_request()

    created_kwargs = first_repository.create_task_record.call_args.kwargs["kwargs"]
    assert created_kwargs["submitted_by_user_id"] == "user-123"
    assert created_kwargs["parent_task_id"] == PARENT_TASK_ID
    first_repository.update_task_success.assert_called_once_with(
        task_id=SOCIAL_TASK_ID,
        result=first_result,
    )
    first_repository.mark_social_dispatch_completed.assert_called_once_with(
        parent_task_id=PARENT_TASK_ID,
        social_task_id=SOCIAL_TASK_ID,
    )

    completed_repository = MagicMock()
    completed_repository.get_task_by_id.return_value = {
        "task_id": SOCIAL_TASK_ID,
        "status": TaskStatus.SUCCESS.value,
        "result": json.dumps(first_result),
    }
    with patch.dict(
        generate_social_posts_task._orig_run.__globals__,
        {"SyncTaskResultRepository": MagicMock(return_value=completed_repository)},
    ):
        generate_social_posts_task.push_request(id=SOCIAL_TASK_ID, hostname="worker-2")
        try:
            replay_result = generate_social_posts_task.run(
                article_id=str(ARTICLE_ID),
                title="Durable article",
                topic="durability",
                language="en",
                submitted_by_user_id="user-123",
                parent_task_id=PARENT_TASK_ID,
            )
        finally:
            generate_social_posts_task.pop_request()

    assert replay_result == first_result
    completed_repository.create_task_record.assert_not_called()
    completed_repository.update_task_success.assert_not_called()
    completed_repository.mark_social_dispatch_completed.assert_called_once_with(
        parent_task_id=PARENT_TASK_ID,
        social_task_id=SOCIAL_TASK_ID,
    )


def test_social_task_id_is_deterministic_per_parent():
    assert social_task_id_for(PARENT_TASK_ID) == SOCIAL_TASK_ID
    assert social_task_id_for(PARENT_TASK_ID) != social_task_id_for("another-parent")


def test_redelivery_start_record_does_not_downgrade_durable_success():
    database = MagicMock()
    database.execute.return_value = {"id": uuid4()}
    repository = SyncTaskResultRepository(database)

    repository.create_task_record(
        task_id=PARENT_TASK_ID,
        task_name="orchestration.tasks.generate_content_task",
        args=("project-1", "topic"),
        kwargs={"submitted_by_user_id": "user-123"},
    )

    query, params = database.execute.call_args.args[:2]
    assert "WHEN task_results.status = %(success_status)s" in query
    assert params["success_status"] == TaskStatus.SUCCESS.value


def test_child_completion_atomically_closes_parent_outbox_and_result_state():
    database = MagicMock()
    repository = SyncTaskResultRepository(database)

    repository.mark_social_dispatch_completed(
        parent_task_id=PARENT_TASK_ID,
        social_task_id=SOCIAL_TASK_ID,
    )

    query, params = database.execute.call_args.args[:2]
    assert "WITH completed_event AS" in query
    assert "UPDATE generation_outbox_events" in query
    assert "UPDATE task_results" in query
    assert "social_dispatch_status" in query
    assert params["parent_task_id"] == PARENT_TASK_ID
    assert params["social_task_id"] == SOCIAL_TASK_ID
    assert params["success_status"] == TaskStatus.SUCCESS.value
