from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

import pytest

from knowledge.article_repository import ArticleRepository


class _FakeMappingResult:
    def __init__(self, row):
        self._row = row

    def mappings(self):
        return self

    def one(self):
        return self._row


class _FakeSession:
    def __init__(self, database):
        self.database = database

    async def execute(self, query):
        self.database.insert_count += 1
        self.database.query = query
        values = dict(query.compile().params)
        values.update(
            publish_status="not_published",
            wordpress_post_id=None,
            wordpress_post_url=None,
            wordpress_post_status=None,
            publish_attempt_count=0,
        )
        self.database.rows.append(values)
        return _FakeMappingResult(values)


class _FakeDatabase:
    def __init__(self):
        self.insert_count = 0
        self.query = None
        self.rows = []

    async def execute(self, query):
        self.insert_count += 1
        return _FakeMappingResult({})

    @asynccontextmanager
    async def session(self):
        yield _FakeSession(self)


@pytest.mark.asyncio
async def test_create_inserts_once_and_returns_article_with_safe_publish_defaults():
    database = _FakeDatabase()
    repository = ArticleRepository(database)
    article_id = uuid4()
    project_id = uuid4()
    article_data = {
        "id": article_id,
        "project_id": project_id,
        "content_plan_id": None,
        "title": "Repository create regression",
        "content": "<p>Repository create must return the inserted article without a second insert.</p>",
        "meta_description": "Regression coverage for SQLAlchemy insert result handling.",
        "keywords": ["reliability"],
        "word_count": 11,
        "readability_score": 70.0,
        "keyword_density": {"reliability": 0.05},
        "total_tokens_used": 0,
        "total_cost": 0.0,
        "generation_time": 0.0,
        "created_at": datetime(2026, 6, 14),
        "updated_at": datetime(2026, 6, 14),
    }

    created = await repository.create(article_data)

    assert database.insert_count == 1
    assert len(database.rows) == 1
    assert database.query.table.name == "generated_articles"
    assert database.query._returning
    assert created["id"] == article_id
    assert created["project_id"] == project_id
    assert created["title"] == article_data["title"]
    assert created["content"] == article_data["content"]
    assert created["publish_status"] == "not_published"
    assert created["publish_attempt_count"] == 0
    assert created["wordpress_post_id"] is None
    assert created["wordpress_post_url"] is None
    assert created["wordpress_post_status"] is None
