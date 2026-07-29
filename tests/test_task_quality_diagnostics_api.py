from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.routes.content import _safe_task_error, get_task_status
from security import User


class FakeTaskRepository:
    def __init__(self, row):
        self.row = row

    async def get_task_by_id(self, task_id):
        return self.row


def manager_user() -> User:
    return User(
        id=str(uuid4()),
        username="manager",
        email="manager@example.com",
        role="manager",
        created_at=datetime(2026, 1, 1),
    )


def fake_async_result(state: str, info=None):
    return SimpleNamespace(state=state, info=info, result=None)


def test_connection_failure_is_not_classified_as_authentication_failure():
    message, error_code = _safe_task_error(
        "Cannot reach OpenAI-compatible provider. Check network access and provider status. "
        "Code: llm_unavailable"
    )

    assert error_code == "LLM_PROVIDER_UNAVAILABLE"
    assert "temporarily unreachable" in message


@pytest.mark.asyncio
async def test_failed_task_detail_returns_persisted_quality_diagnostics(monkeypatch):
    import celery.result

    diagnostics = {
        "actual_word_count": 22,
        "min_word_count": 800,
        "max_word_count": 1800,
        "headings_count": 1,
        "paragraphs_count": 1,
        "language": "fa",
        "regeneration_attempted": True,
        "findings": [{"code": "word_count_below_minimum"}],
    }
    monkeypatch.setattr(
        celery.result,
        "AsyncResult",
        lambda *args, **kwargs: fake_async_result("FAILURE"),
    )
    response = await get_task_status(
        "task-1",
        user=manager_user(),
        task_repo=FakeTaskRepository(
            {
                "status": "failure",
                "error": "Article failed release gate",
                "result": {"quality_diagnostics": diagnostics},
            }
        ),
    )

    assert response["state"] == "FAILURE"
    assert response["error"] == "Article failed release gate"
    assert response["quality_diagnostics"] == diagnostics


@pytest.mark.asyncio
async def test_success_task_detail_keeps_existing_response_shape_without_diagnostics(monkeypatch):
    import celery.result

    monkeypatch.setattr(
        celery.result,
        "AsyncResult",
        lambda *args, **kwargs: fake_async_result("SUCCESS"),
    )
    response = await get_task_status(
        "task-2",
        user=manager_user(),
        task_repo=FakeTaskRepository(
            {
                "status": "success",
                "result": {"article_id": "article-1", "project_id": "project-1"},
                "end_time": "2026-01-01T00:00:00",
            }
        ),
    )

    assert response["state"] == "SUCCESS"
    assert response["result"]["article_id"] == "article-1"
    assert "quality_diagnostics" not in response
