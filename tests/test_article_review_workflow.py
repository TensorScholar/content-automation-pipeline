from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi import HTTPException

from services.content_service import ContentService


def _article() -> dict:
    paragraph = "A reliable review workflow keeps publishing decisions explicit and auditable. "
    return {
        "id": uuid4(),
        "project_id": uuid4(),
        "title": "A production-ready article review workflow",
        "content": f"<h2>Review</h2><p>{paragraph * 10}</p><h2>Publishing</h2>",
        "meta_description": "A concise description of the review workflow.",
        "keywords": ["review", "publishing"],
    }


class FakeArticleRepository:
    def __init__(self, article: dict):
        self.db = None
        self.article = article
        self.review = {
            "review_status": "pending_review",
            "review_note": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "review_updated_at": None,
            "reviewer_full_name": None,
            "reviewer_email": None,
        }
        self.writes = 0

    async def get_by_id(self, article_id, include_content=True):
        del include_content
        return self.article if article_id == self.article["id"] else None

    async def get_review_state(self, article_id):
        if article_id != self.article["id"]:
            return None
        return {"id": article_id, "project_id": self.article["project_id"], **self.review}

    async def set_review_state(self, *, article_id, review_status, reviewer_id, note):
        if article_id != self.article["id"]:
            return None
        self.writes += 1
        now = datetime.now(timezone.utc)
        self.review.update(
            review_status=review_status,
            review_note=note,
            reviewed_by=reviewer_id,
            reviewed_at=now,
            review_updated_at=now,
            reviewer_full_name="Review Manager",
        )
        return await self.get_review_state(article_id)


class FakeAgent:
    pass


def _service(article: dict | None = None):
    repository = FakeArticleRepository(article or _article())
    service = ContentService(repository, FakeAgent())
    return service, repository


@pytest.mark.asyncio
async def test_review_status_includes_compact_deterministic_checklist():
    service, repository = _service()

    result = await service.get_article_review(repository.article["id"])

    assert result["status"] == "pending_review"
    assert result["can_approve"] is True
    assert len(result["checklist"]) == 5
    assert all("label" in item and "passed" in item for item in result["checklist"])


@pytest.mark.asyncio
async def test_manager_can_approve_review_ready_article():
    service, repository = _service()
    reviewer_id = uuid4()

    result = await service.review_article(
        article_id=repository.article["id"],
        action="approve",
        reviewer_id=reviewer_id,
        note=None,
    )

    assert result["status"] == "approved"
    assert result["reviewed_by"] == str(reviewer_id)
    assert result["reviewer_name"] == "Review Manager"
    assert repository.writes == 1


@pytest.mark.asyncio
async def test_request_changes_requires_feedback():
    service, repository = _service()

    with pytest.raises(HTTPException) as exc:
        await service.review_article(
            article_id=repository.article["id"],
            action="request_changes",
            reviewer_id=uuid4(),
            note=" ",
        )

    assert exc.value.status_code == 400
    assert repository.writes == 0


@pytest.mark.asyncio
async def test_blocking_check_prevents_approval_but_allows_request_changes():
    article = _article()
    article["content"] = ""
    service, repository = _service(article)

    with pytest.raises(HTTPException) as exc:
        await service.review_article(
            article_id=article["id"],
            action="approve",
            reviewer_id=uuid4(),
            note=None,
        )
    assert exc.value.status_code == 409
    assert repository.writes == 0

    result = await service.review_article(
        article_id=article["id"],
        action="request_changes",
        reviewer_id=uuid4(),
        note="Restore the missing article body.",
    )
    assert result["status"] == "changes_requested"
    assert repository.writes == 1
