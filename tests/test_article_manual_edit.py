from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api.routes.content import ArticleEditRequest
from infrastructure.redis_client import RedisClient
from services.content_service import ContentService


class FakeArticleRepository:
    def __init__(self, article_id):
        self.db = None
        self.article_id = article_id
        self.calls = []

    async def update_content_with_revision(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["article_id"] != self.article_id:
            return None
        return {
            "content": kwargs["content"],
            "word_count": kwargs["word_count"],
            "review_status": "pending_review",
            "updated_at": "2026-07-24T10:00:00",
        }


@pytest.mark.asyncio
async def test_manual_edit_normalizes_content_and_delegates_atomic_revision_write():
    article_id = uuid4()
    repository = FakeArticleRepository(article_id)
    service = ContentService(repository, content_agent=object())

    result = await service.update_article_content(
        article_id,
        "  <h2>عنوان</h2><p>متن ویرایش شده مقاله</p>  ",
        "editor-1",
        "اصلاح نهایی",
    )

    assert repository.calls == [
        {
            "article_id": article_id,
            "content": "<h2>عنوان</h2><p>متن ویرایش شده مقاله</p>",
            "word_count": 5,
            "revision_note": "اصلاح نهایی",
        }
    ]
    assert result["content"] == "<h2>عنوان</h2><p>متن ویرایش شده مقاله</p>"
    assert result["word_count"] == 5
    assert result["review_status"] == "pending_review"


@pytest.mark.asyncio
async def test_manual_edit_missing_article_returns_not_found():
    repository = FakeArticleRepository(uuid4())
    service = ContentService(repository, content_agent=object())

    with pytest.raises(HTTPException) as exc_info:
        await service.update_article_content(uuid4(), "Valid replacement content", "editor-1")

    assert exc_info.value.status_code == 404


def test_manual_edit_request_rejects_whitespace_only_content():
    with pytest.raises(ValidationError):
        ArticleEditRequest(content="   ")


@pytest.mark.asyncio
async def test_cache_pattern_invalidation_uses_non_blocking_scan():
    class FakeRedisConnection:
        def __init__(self):
            self.deleted = []

        async def scan_iter(self, *, match, count):
            assert match == "article:get_by_id:*"
            assert count == 100
            for key in (b"article:get_by_id:1", b"article:get_by_id:2"):
                yield key

        async def delete(self, key):
            self.deleted.append(key)
            return 1

    connection = FakeRedisConnection()

    class FakePool:
        @asynccontextmanager
        async def get_connection(self):
            yield connection

    client = RedisClient()
    client._pool = FakePool()

    deleted = await client.delete_pattern("article:get_by_id:*")

    assert deleted == 2
    assert connection.deleted == [b"article:get_by_id:1", b"article:get_by_id:2"]
