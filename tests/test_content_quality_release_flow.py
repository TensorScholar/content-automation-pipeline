from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from core.exceptions import WorkflowError
from execution.content_generator import ContentGenerator
from orchestration.content_agent import ContentAgent


def _agent() -> ContentAgent:
    agent = object.__new__(ContentAgent)
    agent.current_workflow = {"settings": {}}
    agent.workflow_events = []
    agent._record_event = lambda *args, **kwargs: None
    return agent


def _article(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="article-id",
        title="عنوان",
        content=content,
        keywords=[],
        quality_metrics=SimpleNamespace(readability_score=0.0, word_count=0),
        total_tokens_used=10,
        total_cost_usd=0.0,
        generation_time_seconds=1.0,
    )


def _accepted_persian_article() -> SimpleNamespace:
    paragraph = "واژه " * 300
    return _article(
        f"<h2>مقدمه</h2><p>{paragraph}</p>"
        f"<h2>جزئیات</h2><p>{paragraph}</p>"
        f"<h2>جمع بندی</h2><p>{paragraph}</p>"
    )


def _accepted_persian_faq_article() -> SimpleNamespace:
    article = _accepted_persian_article()
    article.content += (
        "<h2>پرسش‌های متداول</h2>"
        "<h3>پرسش نخست چیست؟</h3><p>پاسخ روشن و کاربردی نخست.</p>"
        "<h3>پرسش دوم چیست؟</h3><p>پاسخ روشن و کاربردی دوم.</p>"
    )
    return article


def _workflow_agent(first_article, regenerated_article) -> tuple[ContentAgent, AsyncMock]:
    finalized = AsyncMock()
    agent = object.__new__(ContentAgent)
    agent.config = SimpleNamespace(
        default_priority="high",
        max_generation_retries=2,
    )
    agent.content_generator = SimpleNamespace(finalize_article=finalized)
    agent.metrics = SimpleNamespace(record_workflow_completion=lambda **kwargs: None)
    agent._transition_state = AsyncMock()
    agent._conduct_keyword_research = AsyncMock(return_value={"primary": [], "secondary": []})
    agent._plan_content = AsyncMock(return_value=SimpleNamespace(target_word_count=900))
    agent._generate_article = AsyncMock(return_value=first_article)
    agent._regenerate_with_feedback = AsyncMock(return_value=regenerated_article)
    agent._record_workflow_metrics = AsyncMock()
    return agent, finalized


@pytest.mark.asyncio
async def test_short_persian_article_fails_the_release_gate_before_finalization():
    result = await _agent()._validate_article_quality(
        _article("<h2>عنوان</h2><p>" + "واژه " * 22 + "</p>"),
        language="fa",
        target_word_count=900,
    )

    assert not result["passed"]
    assert result["issues"]
    assert not any("readability" in issue.lower() for issue in result["issues"])


@pytest.mark.asyncio
async def test_requested_faq_requirement_is_applied_by_agent_validation():
    result = await _agent()._validate_article_quality(
        _accepted_persian_article(),
        language="fa",
        target_word_count=900,
        require_faq=True,
    )

    assert result["passed"] is False
    assert result["release_gate"]["findings"][-1]["code"] == "missing_required_faq"


@pytest.mark.asyncio
async def test_missing_title_fails_the_release_gate():
    article = _accepted_persian_article()
    article.title = ""

    result = await _agent()._validate_article_quality(article, language="fa", target_word_count=900)

    assert not result["passed"]
    assert "Article title is required." in result["issues"]
    assert result["release_gate"]["findings"][-1]["code"] == "missing_title"


@pytest.mark.asyncio
async def test_structured_persian_article_passes_despite_advisory_readability():
    paragraph = "واژه " * 300
    content = (
        f"<h2>مقدمه</h2><p>{paragraph}</p>"
        f"<h2>جزئیات</h2><p>{paragraph}</p>"
        f"<h2>جمع بندی</h2><p>{paragraph}</p>"
    )

    result = await _agent()._validate_article_quality(
        _article(content), language="fa", target_word_count=900
    )

    assert result["passed"]
    assert result["advisories"]


@pytest.mark.asyncio
async def test_release_gate_word_count_replaces_legacy_article_metric():
    article = _accepted_persian_article()
    article.quality_metrics.word_count = 9999

    result = await _agent()._validate_article_quality(
        article,
        language="fa",
        target_word_count=900,
    )

    assert result["passed"]
    assert article.quality_metrics.word_count == result["release_gate"]["word_count"]


@pytest.mark.asyncio
async def test_finalization_saves_before_exporting_an_accepted_article():
    events = []

    class Repository:
        async def save_generated_article(self, article):
            events.append(("save", article))

    generator = object.__new__(ContentGenerator)
    generator.article_repo = Repository()

    async def export(article, plan):
        events.append(("export", article, plan))

    generator._export_article_to_files = export
    article = object()
    plan = object()

    await generator.finalize_article(article, plan)

    assert [event[0] for event in events] == ["save", "export"]


@pytest.mark.asyncio
async def test_regenerated_article_is_revalidated_before_finalization():
    agent, finalized = _workflow_agent(
        _article("<h2>عنوان</h2><p>" + "واژه " * 22 + "</p>"),
        _accepted_persian_article(),
    )
    project = SimpleNamespace(id=uuid4())

    article = await agent.create_content(
        project.id,
        "موضوع",
        project_context={"project": project},
        language="fa",
    )

    assert article is agent._regenerate_with_feedback.return_value
    assert article.total_tokens_used == 20
    assert article.generation_time_seconds == 2.0
    agent._regenerate_with_feedback.assert_awaited_once()
    finalized.assert_awaited_once_with(article, agent._plan_content.return_value)


@pytest.mark.asyncio
async def test_requested_faq_is_revalidated_after_regeneration():
    agent, finalized = _workflow_agent(
        _accepted_persian_article(),
        _accepted_persian_faq_article(),
    )
    original_validator = agent._validate_article_quality
    agent._validate_article_quality = AsyncMock(wraps=original_validator)
    project = SimpleNamespace(id=uuid4())

    article = await agent.create_content(
        project.id,
        "موضوع",
        project_context={"project": project},
        language="fa",
        include_faq=True,
    )

    assert article is agent._regenerate_with_feedback.return_value
    assert agent._validate_article_quality.await_count == 2
    assert all(
        call.kwargs["require_faq"] is True
        for call in agent._validate_article_quality.await_args_list
    )
    finalized.assert_awaited_once_with(article, agent._plan_content.return_value)


@pytest.mark.asyncio
async def test_deferred_finalization_keeps_accepted_article_side_effect_free():
    agent, finalized = _workflow_agent(
        _accepted_persian_article(),
        _accepted_persian_article(),
    )
    project = SimpleNamespace(id=uuid4())

    article = await agent.create_content(
        project.id,
        "موضوع",
        project_context={"project": project},
        language="fa",
        _defer_finalization=True,
    )

    assert article is agent._generate_article.return_value
    finalized.assert_not_awaited()


@pytest.mark.asyncio
async def test_twice_rejected_article_fails_without_finalization():
    rejected = _article("<h2>عنوان</h2><p>" + "واژه " * 22 + "</p>")
    agent, finalized = _workflow_agent(rejected, rejected)
    project = SimpleNamespace(id=uuid4())

    with pytest.raises(WorkflowError, match="Article failed release gate") as error:
        await agent.create_content(
            project.id,
            "موضوع",
            project_context={"project": project},
            language="fa",
        )

    agent._regenerate_with_feedback.assert_awaited_once()
    finalized.assert_not_awaited()
    message = str(error.value)
    assert "word_count" in message
    assert "minimum_word_count" in message
    assert "maximum_word_count" in message
    assert "heading_count" in message
    assert "paragraph_count" in message
    assert "language" in message
    assert "regeneration_attempted=True" in message
    assert error.value.context["quality_diagnostics"] == {
        "actual_word_count": 23,
        "min_word_count": 800,
        "max_word_count": 1800,
        "headings_count": 1,
        "paragraphs_count": 1,
        "language": "fa",
        "regeneration_attempted": True,
        "findings": error.value.context["release_gate"]["findings"],
    }


@pytest.mark.asyncio
async def test_regeneration_error_preserves_original_quality_diagnostics():
    rejected = _article("<h2>عنوان</h2><p>" + "واژه " * 22 + "</p>")
    agent, finalized = _workflow_agent(rejected, rejected)
    agent._regenerate_with_feedback.side_effect = RuntimeError("provider interrupted")
    project = SimpleNamespace(id=uuid4())

    with pytest.raises(WorkflowError, match="Article regeneration failed") as error:
        await agent.create_content(
            project.id,
            "موضوع",
            project_context={"project": project},
            language="fa",
        )

    finalized.assert_not_awaited()
    assert error.value.context["regeneration_attempted"] is True
    assert error.value.context["quality_diagnostics"] == {
        "actual_word_count": 23,
        "min_word_count": 800,
        "max_word_count": 1800,
        "headings_count": 1,
        "paragraphs_count": 1,
        "language": "fa",
        "regeneration_attempted": True,
        "findings": error.value.context["release_gate"]["findings"],
    }
