from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from api.routes.content import GenerateContentRequest, generate_content_async
from knowledge.article_repository import ArticleRepository
from orchestration.task_persistence import (
    SyncTaskResultRepository,
    TaskResultRepository,
    TaskStatus,
)
from security import User


class FakeAsyncResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeAsyncDatabase:
    def __init__(self, start_time=None):
        self.start_time = start_time
        self.queries = []

    async def execute(self, query, params=None):
        self.queries.append(query)
        return FakeAsyncResult(SimpleNamespace(_mapping={"id": uuid4()}))

    async def fetch_one(self, query):
        return {"start_time": self.start_time} if self.start_time else None

    async def fetch_all(self, query):
        self.queries.append(query)
        return []


class FakeSyncDatabase:
    def __init__(self, start_time=None):
        self.start_time = start_time
        self.calls = []

    def execute(self, query, params=None, fetch_one=False):
        self.calls.append((query, params, fetch_one))
        if "SELECT start_time" in query:
            return {"start_time": self.start_time}
        if "RETURNING id" in query:
            return {"id": uuid4()}
        return None


@pytest.mark.asyncio
async def test_async_task_failure_calculates_duration_for_naive_db_timestamp():
    db = FakeAsyncDatabase(start_time=datetime(2026, 1, 1, 12, 0, 0))
    repo = TaskResultRepository(db)

    assert await repo.update_task_failure("task-1", "provider failed") is True

    update_query = db.queries[-1]
    params = update_query.compile().params
    assert params["duration_seconds"] >= 0
    assert params["status"] == TaskStatus.FAILURE


@pytest.mark.asyncio
async def test_failed_task_query_normalizes_aware_timestamp_for_naive_column():
    db = FakeAsyncDatabase()
    repo = TaskResultRepository(db)

    await repo.get_failed_tasks(since=datetime(2026, 1, 1, tzinfo=timezone.utc))

    params = db.queries[-1].compile().params
    assert params["created_at_1"].tzinfo is None


def test_sync_task_failure_calculates_duration_for_naive_db_timestamp():
    db = FakeSyncDatabase(start_time=datetime(2026, 1, 1, 12, 0, 0))
    repo = SyncTaskResultRepository(sync_db=db)

    assert repo.update_task_failure("task-1", "provider failed") is True

    update_params = db.calls[-1][1]
    assert update_params["duration"] >= 0
    assert update_params["status"] == TaskStatus.FAILURE.value


class FakeTaskRepository:
    def __init__(self):
        self.events = []

    async def create_task_record(self, **kwargs):
        self.events.append(("create", kwargs))
        return uuid4()

    async def update_task_failure(self, task_id, error, traceback=None):
        self.events.append(("failure", {"task_id": task_id, "error": error}))
        return True


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.deleted = []

    async def set(self, key, value, ttl=None, ex=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def delete(self, key):
        self.deleted.append(key)
        self.values.pop(key, None)
        return True


class ExistingProjectRepository:
    async def get_by_id(self, project_id):
        return SimpleNamespace(id=project_id)


def make_user() -> User:
    return User(
        id=str(uuid4()),
        username="tester",
        email="tester@example.com",
        role="manager",
        created_at=datetime(2026, 1, 1),
    )


@pytest.mark.asyncio
async def test_generation_persists_pending_before_broker_dispatch(monkeypatch):
    task_repo = FakeTaskRepository()
    redis = FakeRedis()
    call_order = []

    def apply_async(**kwargs):
        call_order.append(("dispatch", kwargs))
        return SimpleNamespace(id=kwargs["task_id"])

    monkeypatch.setattr(
        "orchestration.tasks.generate_content_task.apply_async",
        apply_async,
    )

    original_create = task_repo.create_task_record

    async def ordered_create(**kwargs):
        call_order.append(("persist", kwargs))
        return await original_create(**kwargs)

    task_repo.create_task_record = ordered_create
    response = await generate_content_async(
        GenerateContentRequest(
            project_id=uuid4(),
            topic="Reliable generation workflow",
            language="en",
        ),
        user=make_user(),
        task_repo=task_repo,
        redis_client=redis,
        project_repo=ExistingProjectRepository(),
    )

    assert [event[0] for event in call_order] == ["persist", "dispatch"]
    assert call_order[0][1]["status"] == TaskStatus.PENDING
    assert call_order[0][1]["task_id"] == response["task_id"]
    assert call_order[1][1]["task_id"] == response["task_id"]


@pytest.mark.asyncio
async def test_broker_failure_marks_history_failed_and_releases_submission_lock(monkeypatch):
    task_repo = FakeTaskRepository()
    redis = FakeRedis()

    def fail_dispatch(**kwargs):
        raise ConnectionError("broker unavailable")

    monkeypatch.setattr(
        "orchestration.tasks.generate_content_task.apply_async",
        fail_dispatch,
    )

    with pytest.raises(HTTPException) as error:
        await generate_content_async(
            GenerateContentRequest(
                project_id=uuid4(),
                topic="Reliable generation workflow",
            ),
            user=make_user(),
            task_repo=task_repo,
            redis_client=redis,
            project_repo=ExistingProjectRepository(),
        )

    assert error.value.status_code == 503
    assert [event[0] for event in task_repo.events] == ["create", "failure"]
    assert len(redis.deleted) == 1


class FakeSession:
    def __init__(self):
        self.queries = []
        self.commits = 0

    async def execute(self, query):
        self.queries.append(query)
        return SimpleNamespace(rowcount=1)

    async def commit(self):
        self.commits += 1


class FakeArticleDatabase:
    def __init__(self):
        self.fake_session = FakeSession()

    @asynccontextmanager
    async def session(self):
        yield self.fake_session


@pytest.mark.asyncio
async def test_article_and_project_counters_commit_in_same_transaction():
    db = FakeArticleDatabase()
    repo = ArticleRepository(db)
    project_id = uuid4()
    article = SimpleNamespace(
        id=uuid4(),
        project_id=project_id,
        content_plan_id=uuid4(),
        title="Reliability",
        content="Content",
        meta_description="Description",
        keywords=["reliability"],
        quality_metrics=SimpleNamespace(
            word_count=100,
            readability_score=8.0,
            keyword_density={"reliability": 0.02},
            model_dump_json=lambda: "{}",
        ),
        total_tokens_used=200,
        total_cost_usd=0.02,
        generation_time_seconds=1.5,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
        model_dump=lambda: {},
    )

    await repo.save_generated_article(article)

    assert len(db.fake_session.queries) == 2
    assert db.fake_session.queries[0].table.name == "generated_articles"
    assert db.fake_session.queries[1].table.name == "projects"
    assert db.fake_session.commits == 1
