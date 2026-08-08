from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.routes.content import get_task_history, get_task_status
from security import User

PROJECT_ID = "project-123"


class FakeTaskRepository:
    def __init__(self, row):
        self.row = row

    async def get_task_by_id(self, task_id):
        return self.row


class FakeDatabase:
    def __init__(self, rows):
        self.rows = rows

    async def fetch_all(self, query):
        return self.rows


class FakeTaskHistoryRepository:
    def __init__(self, rows):
        self.db = FakeDatabase(rows)


def manager_user() -> User:
    return User(
        id=str(uuid4()),
        username="manager",
        email="manager@example.com",
        role="manager",
        created_at=datetime(2026, 1, 1),
    )


def fake_async_result(state: str):
    result = {"article_id": "article-1"} if state == "SUCCESS" else None
    return SimpleNamespace(state=state, info=None, result=result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("celery_state", "db_status"),
    [
        ("PENDING", "pending"),
        ("STARTED", "started"),
        ("RETRY", "retry"),
        ("FAILURE", "failure"),
        ("SUCCESS", "success"),
    ],
)
async def test_task_detail_preserves_state_and_known_project_provenance(
    monkeypatch,
    celery_state,
    db_status,
):
    import celery.result

    monkeypatch.setattr(
        celery.result,
        "AsyncResult",
        lambda *args, **kwargs: fake_async_result(celery_state),
    )
    row = {
        "status": db_status,
        "args": [PROJECT_ID, "Topic", "high", None],
        "result": {"article_id": "article-1"} if celery_state == "SUCCESS" else None,
    }
    if celery_state == "FAILURE":
        row["error"] = "Generation failed"

    response = await get_task_status(
        "task-known-project",
        user=manager_user(),
        task_repo=FakeTaskRepository(row),
    )

    assert response["state"] == celery_state
    assert response["project_id"] == PROJECT_ID


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("celery_state", "db_status"),
    [
        ("PENDING", "pending"),
        ("STARTED", "started"),
        ("RETRY", "retry"),
        ("FAILURE", "failure"),
        ("SUCCESS", "success"),
    ],
)
async def test_task_detail_does_not_fabricate_unknown_project_provenance(
    monkeypatch,
    celery_state,
    db_status,
):
    import celery.result

    monkeypatch.setattr(
        celery.result,
        "AsyncResult",
        lambda *args, **kwargs: fake_async_result(celery_state),
    )
    row = {
        "status": db_status,
        "args": [],
        "result": {"article_id": "article-1"} if celery_state == "SUCCESS" else None,
    }
    if celery_state == "FAILURE":
        row["error"] = "Generation failed"

    response = await get_task_status(
        "task-unknown-project",
        user=manager_user(),
        task_repo=FakeTaskRepository(row),
    )

    assert response["state"] == celery_state
    assert "project_id" not in response


@pytest.mark.asyncio
async def test_task_history_rows_keep_status_contract():
    rows = [
        {
            "task_id": "task-history",
            "task_name": "orchestration.tasks.generate_content_task",
            "status": "pending",
            "args": [PROJECT_ID, "Topic", "high", None],
        }
    ]

    response = await get_task_history(
        skip=0,
        limit=50,
        search=None,
        user=manager_user(),
        task_repo=FakeTaskHistoryRepository(rows),
    )

    assert response[0]["status"] == "PENDING"
    assert "state" not in response[0]
