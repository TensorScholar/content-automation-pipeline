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


def make_article_row(article_id=None, project_id=None, **overrides):
    content = " ".join(["workflow"] * 120)
    row = {
        "id": article_id or uuid4(),
        "project_id": project_id or uuid4(),
        "content_plan_id": None,
        "title": "Workflow launch article",
        "content": content,
        "meta_description": "short",
        "keywords": ["workflow", "launch"],
        "word_count": 120,
        "readability_score": 8.5,
        "keyword_density": {"workflow": 0.04},
        "total_tokens_used": 250,
        "total_cost": 0.03,
        "generation_time": 12.5,
    }
    row.update(overrides)
    return row


class FakeDistributionRepository:
    db = object()

    def __init__(self, article):
        self.article = article
        self.updates = []

    async def get_by_id(self, article_id, include_content=True):
        return self.article if article_id == self.article["id"] else None

    async def update(self, article_id, updates):
        self.updates.append((article_id, updates))
        return {**self.article, **updates}


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
    }
    assert calls[1]["kwargs"]["topic"] == "Monitoring workflow"
    assert calls[1]["kwargs"]["custom_instructions"] == "Output language must be Persian."
    assert calls[1]["kwargs"]["submitted_by_user_id"] == "user-123"


def test_persisted_article_rebuilds_strict_generated_article_model():
    service = ContentService(article_repository=FakeArticleRepository(), content_agent=object())
    article_id = uuid4()
    project_id = uuid4()

    generated = service._article_dict_to_generated_article(
        make_article_row(article_id=article_id, project_id=project_id)
    )

    assert generated.id == article_id
    assert generated.project_id == project_id
    assert generated.content_plan_id == article_id
    assert len(generated.meta_description) >= 50
    assert generated.quality_metrics.word_count == 120
    assert generated.quality_metrics.keyword_density == {"workflow": 0.04}
    assert generated.total_cost_usd == 0.03


@pytest.mark.asyncio
async def test_distribution_skipped_channel_does_not_mark_article_distributed():
    article = make_article_row()
    repository = FakeDistributionRepository(article)
    service = ContentService(article_repository=repository, content_agent=object())

    result = await service.distribute_article(article["id"], ["telegram"])

    assert result["status"] == "skipped"
    assert result["distributed"] is False
    assert result["channels"] == []
    assert result["delivery_confirmations"]["telegram"]["status"] == "skipped"
    assert repository.updates == []


@pytest.mark.asyncio
async def test_distribution_success_marks_only_successful_channels():
    article = make_article_row()
    repository = FakeDistributionRepository(article)
    service = ContentService(article_repository=repository, content_agent=object())

    result = await service.distribute_article(article["id"], ["rss"])

    assert result["status"] == "success"
    assert result["distributed"] is True
    assert result["channels"] == ["rss"]
    assert result["delivery_confirmations"]["rss"]["status"] == "success"
    assert repository.updates[0][1]["distribution_channels"] == ["rss"]
