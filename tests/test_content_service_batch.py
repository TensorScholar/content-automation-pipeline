import sys
import types
from uuid import uuid4

import pytest

if "asyncpg" not in sys.modules:
    asyncpg_stub = types.ModuleType("asyncpg")
    asyncpg_stub.Connection = object
    sys.modules["asyncpg"] = asyncpg_stub

if "pgvector.asyncpg" not in sys.modules:
    pgvector_stub = types.ModuleType("pgvector")
    pgvector_asyncpg_stub = types.ModuleType("pgvector.asyncpg")
    pgvector_sqlalchemy_stub = types.ModuleType("pgvector.sqlalchemy")

    async def register_vector(_conn):
        return None

    def Vector(_dimensions):
        from sqlalchemy import JSON

        return JSON

    pgvector_asyncpg_stub.register_vector = register_vector
    pgvector_sqlalchemy_stub.Vector = Vector
    sys.modules["pgvector"] = pgvector_stub
    sys.modules["pgvector.asyncpg"] = pgvector_asyncpg_stub
    sys.modules["pgvector.sqlalchemy"] = pgvector_sqlalchemy_stub

from services.content_service import ContentService


class FakeArticleRepository:
    db = object()


class FakeTask:
    def __init__(self, task_id: str):
        self.id = task_id


class ExistingProjectRepository:
    async def get_by_id(self, project_id):
        return types.SimpleNamespace(id=project_id)


@pytest.mark.asyncio
async def test_batch_generation_preserves_instructions_and_priority_routing(monkeypatch):
    calls = []

    def fake_apply_async(**kwargs):
        calls.append(kwargs)
        return FakeTask(f"task-{len(calls)}")

    tasks_stub = types.ModuleType("orchestration.tasks")
    tasks_stub.generate_content_task = types.SimpleNamespace(apply_async=fake_apply_async)
    monkeypatch.setitem(sys.modules, "orchestration.tasks", tasks_stub)

    service = ContentService(
        article_repository=FakeArticleRepository(),
        content_agent=object(),
    )
    service.projects = ExistingProjectRepository()
    project_id = uuid4()

    result = await service.batch_generate_content(
        project_id=project_id,
        topics=["Launch workflow", "Monitoring workflow"],
        priority="high",
        custom_instructions="Output language must be Persian.",
        submitted_by_user_id="user-123",
    )

    assert result["task_ids"] == ["task-1", "task-2"]
    assert len(calls) == 2
    assert calls[0]["queue"] == "high"
    assert calls[0]["routing_key"] == "high"
    assert calls[0]["kwargs"] == {
        "project_id": str(project_id),
        "topic": "Launch workflow",
        "priority": "high",
        "custom_instructions": "Output language must be Persian.",
        "submitted_by_user_id": "user-123",
        "model_override": None,
        "language": "fa",
    }
    assert calls[1]["kwargs"]["topic"] == "Monitoring workflow"
    assert calls[1]["kwargs"]["custom_instructions"] == "Output language must be Persian."
    assert calls[1]["kwargs"]["submitted_by_user_id"] == "user-123"
