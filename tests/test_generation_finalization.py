from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.exceptions import WorkflowError
from orchestration.generation_finalization import GenerationFinalizationRepository


class _ScalarResult:
    def __init__(self, value=None, mapping=None, rowcount=1):
        self.value = value
        self.mapping = mapping
        self.rowcount = rowcount

    def scalar_one_or_none(self):
        return self.value

    def scalar_one(self):
        if self.value is None:
            raise AssertionError("expected a scalar value")
        return self.value

    def mappings(self):
        return self

    def one_or_none(self):
        return self.mapping


class _FinalizationSession:
    def __init__(self, state):
        self.state = state

    async def execute(self, statement):
        sql = str(statement)
        table = getattr(getattr(statement, "table", None), "name", "")

        if "FROM generated_articles JOIN task_results" in sql:
            mapping = None
            if self.state.article_id is not None and self.state.task_result is not None:
                mapping = {"id": self.state.article_id, "result": self.state.task_result}
            return _ScalarResult(mapping=mapping)
        if "FROM generation_outbox_events JOIN generated_articles" in sql:
            self.state.export_claim_sql = sql
            self.state.export_claim_statement = statement
            return _ScalarResult(mapping=self.state.export_row)
        if statement.is_select and "generated_articles.generation_task_id" in sql:
            return _ScalarResult(self.state.article_id)
        if statement.is_select and "task_results.result" in sql:
            return _ScalarResult(self.state.task_result)
        if statement.is_select and "task_results.start_time" in sql:
            return _ScalarResult(self.state.start_time)
        if statement.is_insert and table == "generated_articles":
            self.state.article_inserts += 1
            self.state.article_insert_sql = sql
            self.state.article_id = self.state.expected_article_id
            return _ScalarResult(self.state.article_id)
        if statement.is_insert and table == "task_results":
            self.state.child_task_inserts += 1
            self.state.child_task_insert_statement = statement
            return _ScalarResult()
        if statement.is_insert and table == "generation_outbox_events":
            self.state.outbox_inserts += 1
            self.state.outbox_event_types.append(statement.compile().params["event_type"])
            return _ScalarResult()
        if statement.is_update and table == "generation_outbox_events":
            self.state.outbox_updates += 1
            self.state.outbox_update_sqls.append(sql)
            self.state.outbox_update_statements.append(statement)
            return _ScalarResult(rowcount=1)
        if statement.is_update and table == "projects":
            self.state.project_updates += 1
            return _ScalarResult(rowcount=1)
        if statement.is_update and table == "task_results":
            self.state.task_updates += 1
            return _ScalarResult(rowcount=self.state.task_update_rowcount)
        raise AssertionError(f"Unexpected SQL statement: {sql}")


class _FinalizationDatabase:
    def __init__(self, state):
        self.state = state
        self.session_instance = _FinalizationSession(state)
        self.commits = 0
        self.rollbacks = 0

    @asynccontextmanager
    async def transaction(self):
        try:
            yield self.session_instance
        except Exception:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1

    @asynccontextmanager
    async def read_session(self):
        yield self.session_instance


def _article(article_id):
    return SimpleNamespace(
        id=article_id,
        project_id=uuid4(),
        content_plan_id=uuid4(),
        title="Durable generation",
        content="<h2>One</h2><p>content</p><h2>Two</h2><p>content</p>",
        meta_description="A durable content-generation finalization test article.",
        keywords=["durability"],
        quality_metrics=SimpleNamespace(
            word_count=900,
            readability_score=9.0,
            keyword_density={"durability": 0.02},
        ),
        total_tokens_used=1200,
        total_cost_usd=0.04,
        generation_time_seconds=4.2,
        created_at=datetime(2026, 7, 21, 10, 0, 0),
        updated_at=datetime(2026, 7, 21, 10, 0, 0),
    )


def _state(article_id, *, task_update_rowcount=1):
    return SimpleNamespace(
        expected_article_id=article_id,
        article_id=None,
        task_result=None,
        start_time=datetime(2026, 7, 21, 9, 59, 0),
        article_inserts=0,
        article_insert_sql="",
        child_task_inserts=0,
        child_task_insert_statement=None,
        project_updates=0,
        task_updates=0,
        outbox_inserts=0,
        outbox_event_types=[],
        outbox_updates=0,
        outbox_update_sqls=[],
        outbox_update_statements=[],
        export_claim_sql="",
        export_claim_statement=None,
        export_row=None,
        task_update_rowcount=task_update_rowcount,
    )


@pytest.mark.asyncio
async def test_finalization_persists_article_task_cost_and_export_event_once():
    article_id = uuid4()
    state = _state(article_id)
    db = _FinalizationDatabase(state)
    repository = GenerationFinalizationRepository(db)
    task_result = {"status": "success", "article_id": str(article_id), "cost": 0.04}

    first = await repository.finalize(
        task_id="task-finalize-once",
        article=_article(article_id),
        task_result=task_result,
        language="fa",
    )
    state.task_result = first.result
    replay = await repository.finalize(
        task_id="task-finalize-once",
        article=_article(uuid4()),
        task_result={"status": "success", "cost": 99.0},
        language="fa",
    )

    assert first.newly_finalized is True
    assert replay.newly_finalized is False
    assert replay.result == first.result
    assert state.article_inserts == 1
    assert state.project_updates == 1
    assert state.task_updates == 1
    assert state.outbox_inserts == 1
    assert db.commits == 2
    assert "ON CONFLICT (generation_task_id) WHERE generation_task_id IS NOT NULL" in (
        state.article_insert_sql
    )


@pytest.mark.asyncio
async def test_finalization_atomically_prepares_social_child_and_dispatch_event():
    article_id = uuid4()
    state = _state(article_id)
    db = _FinalizationDatabase(state)
    repository = GenerationFinalizationRepository(db)

    outcome = await repository.finalize(
        task_id="task-social-finalize",
        article=_article(article_id),
        task_result={"status": "success", "cost": 0.04},
        language="fa",
        social_request={
            "title": "Durable generation",
            "topic": "durability",
            "language": "fa",
            "submitted_by_user_id": "user-123",
        },
    )

    child_params = state.child_task_insert_statement.compile().params
    assert outcome.result["social_task_id"] == child_params["task_id"]
    assert outcome.result["social_dispatch_status"] == "pending"
    assert state.child_task_inserts == 1
    assert child_params["kwargs"]["submitted_by_user_id"] == "user-123"
    assert child_params["kwargs"]["parent_task_id"] == "task-social-finalize"
    assert state.outbox_inserts == 2
    assert set(state.outbox_event_types) == {
        "article.export.requested",
        "article.social_drafts.requested",
    }
    assert db.commits == 1


@pytest.mark.asyncio
async def test_finalization_rolls_back_when_task_record_is_missing():
    article_id = uuid4()
    state = _state(article_id, task_update_rowcount=0)
    db = _FinalizationDatabase(state)
    repository = GenerationFinalizationRepository(db)

    with pytest.raises(WorkflowError, match="Task record task-missing"):
        await repository.finalize(
            task_id="task-missing",
            article=_article(article_id),
            task_result={"status": "success", "cost": 0.04},
            language="fa",
        )

    assert db.commits == 0
    assert db.rollbacks == 1
    assert state.outbox_inserts == 0


@pytest.mark.asyncio
async def test_completed_task_lookup_requires_article_and_success_result():
    article_id = uuid4()
    state = _state(article_id)
    db = _FinalizationDatabase(state)
    repository = GenerationFinalizationRepository(db)

    assert await repository.get_finalized_result("task-lookup") is None

    state.article_id = article_id
    state.task_result = {"article_id": str(article_id), "cost": 0.04}
    outcome = await repository.get_finalized_result("task-lookup")

    assert outcome is not None
    assert outcome.article_id == article_id
    assert outcome.result == state.task_result


@pytest.mark.asyncio
async def test_pending_export_hydrates_persisted_article_and_tracks_delivery_attempts():
    article_id = uuid4()
    event_id = uuid4()
    state = _state(article_id)
    state.export_row = {
        "event_id": event_id,
        "payload": {"language": "fa"},
        "attempt_count": 0,
        "id": article_id,
        "project_id": uuid4(),
        "title": "Durable export",
        "content": "<h2>Section</h2><p>Content</p>",
        "meta_description": "A durable local export handoff test article.",
        "keywords": ["durability"],
        "word_count": 900,
        "total_cost": 0.04,
        "generation_time": 4.2,
        "created_at": datetime(2026, 7, 21, 10, 0, 0),
    }
    repository = GenerationFinalizationRepository(_FinalizationDatabase(state))

    export = await repository.claim_pending_export("task-export")

    assert export is not None
    assert export.event_id == event_id
    assert export.attempt_number == 1
    assert export.language == "fa"
    assert export.article.id == article_id
    assert export.article.quality_metrics.word_count == 900
    assert state.outbox_updates == 1
    assert state.export_claim_statement._for_update_arg.skip_locked is True
    assert {"pending", "processing"}.issubset(
        set(state.export_claim_statement.compile().params.values())
    )
    assert "generation_outbox_events.available_at <=" in state.export_claim_sql
    assert "processing" in state.outbox_update_statements[0].compile().params.values()

    await repository.complete_export(event_id, export.attempt_number)
    await repository.record_export_failure(
        event_id,
        export.attempt_number,
        "temporary filesystem error",
    )

    assert state.outbox_updates == 3
    assert "completed" in state.outbox_update_statements[1].compile().params.values()
    assert "pending" in state.outbox_update_statements[2].compile().params.values()
