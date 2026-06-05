from datetime import datetime
from uuid import uuid4

import pytest

from services.content_memory_service import ContentMemoryService


class FakeArticleRepository:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def get_recent_project_articles(self, project_id, limit=12):
        self.calls.append((project_id, limit))
        return self.rows[:limit]


@pytest.mark.asyncio
async def test_content_memory_summarizes_recent_titles_and_keywords():
    project_id = uuid4()
    repo = FakeArticleRepository([
        {
            "title": "WordPress Launch Checklist",
            "keywords": ["wordpress", "launch"],
            "word_count": 1200,
            "created_at": datetime(2026, 1, 1, 12, 0),
        },
        {
            "title": "WordPress Security Basics",
            "keywords": ["wordpress", "security"],
            "word_count": 1000,
            "created_at": datetime(2026, 1, 2, 12, 0),
        },
    ])
    service = ContentMemoryService(repo)

    memory = await service.get_project_memory(project_id)

    assert memory["project_id"] == str(project_id)
    assert memory["article_count"] == 2
    assert memory["recent_titles"] == [
        "WordPress Launch Checklist",
        "WordPress Security Basics",
    ]
    assert memory["average_word_count"] == 1100
    assert memory["repeated_keywords"] == [{"keyword": "wordpress", "count": 2}]
    assert memory["planning_guidance"]


@pytest.mark.asyncio
async def test_content_memory_builds_duplicate_angle_guidance():
    repo = FakeArticleRepository([
        {
            "title": "WordPress Launch Checklist",
            "keywords": ["wordpress"],
            "word_count": 1200,
            "created_at": None,
        }
    ])
    service = ContentMemoryService(repo)

    guidance = await service.build_planning_guidance(
        uuid4(), "Advanced WordPress launch workflow"
    )

    assert guidance is not None
    assert "Recent titles" in guidance
    assert "WordPress Launch Checklist" in guidance
    assert "distinct angle" in guidance


@pytest.mark.asyncio
async def test_empty_content_memory_returns_first_article_guidance():
    service = ContentMemoryService(FakeArticleRepository([]))

    memory = await service.get_project_memory(uuid4())
    guidance = await service.build_planning_guidance(uuid4(), "First article")

    assert memory["article_count"] == 0
    assert memory["planning_guidance"] == [
        "No historical content patterns yet. Generate the first article normally."
    ]
    assert guidance is None
