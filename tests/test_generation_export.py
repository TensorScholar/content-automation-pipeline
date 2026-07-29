from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from execution.content_generator import ContentGenerator


@pytest.mark.asyncio
async def test_file_export_reports_success_after_writing_both_formats(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    generator = object.__new__(ContentGenerator)
    article = SimpleNamespace(
        id=uuid4(),
        project_id=uuid4(),
        title="Durable export",
        content="<h2>Section</h2><p>Accepted article content.</p>",
        meta_description="A valid meta description for the durable export test article.",
        keywords=["durability"],
        quality_metrics=SimpleNamespace(word_count=900),
        total_cost_usd=0.04,
        generation_time_seconds=4.2,
        created_at=datetime(2026, 7, 21, 10, 0, 0),
    )

    assert await generator._export_article_to_files(article, SimpleNamespace(language="fa"))
    assert len(list(tmp_path.glob("exports/**/*.html"))) == 1
    assert len(list(tmp_path.glob("exports/**/*.md"))) == 1
