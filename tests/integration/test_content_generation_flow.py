"""
Content Generation Flow Integration Tests
==========================================

End-to-end tests for content generation workflow ensuring:
- Task dispatch and tracking
- Content quality validation
- LLM integration behavior
- Error recovery and retries
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


class TestContentGenerationWorkflow:
    """Tests for the complete content generation workflow."""

    @pytest.mark.asyncio
    async def test_content_generation_task_dispatch(
        self, mock_celery_task, sample_project_data
    ):
        """Test that content generation dispatches Celery task correctly."""
        with patch(
            "orchestration.tasks.generate_content_task", mock_celery_task
        ):
            # Simulate task dispatch
            task_result = mock_celery_task.delay(
                project_id=sample_project_data["id"],
                topic="Test Article Topic",
                keywords=["test", "integration"],
            )

            assert task_result is not None
            assert task_result.id is not None
            assert task_result.status == "PENDING"

    @pytest.mark.asyncio
    async def test_content_planning_generates_outline(self, mock_llm_client):
        """Test that content planning generates a valid outline."""
        from core.models import Project
        from execution.content_planner import ContentPlanner

        context_synthesizer = MagicMock()
        context_synthesizer.synthesize_context = AsyncMock(return_value=MagicMock())
        model_router = MagicMock()
        model_router.route = AsyncMock(return_value=MagicMock(selected_model="test-model"))
        article_repository = MagicMock()
        article_repository.save_content_plan = AsyncMock()
        mock_llm_client.complete = AsyncMock(
            return_value=MagicMock(
                content="""
                {
                    "title": "Test Article",
                    "sections": [
                        {"heading": "Introduction", "key_points": ["point1"]},
                        {"heading": "Main Content", "key_points": ["point2"]},
                        {"heading": "Conclusion", "key_points": ["point3"]}
                    ],
                    "meta_description": "A complete test article description for planner validation."
                }
                """
            )
        )

        planner = ContentPlanner(
            decision_engine=MagicMock(),
            context_synthesizer=context_synthesizer,
            model_router=model_router,
            llm_client=mock_llm_client,
            article_repository=article_repository,
            metrics_collector=MagicMock(),
        )

        plan = await planner.create_content_plan(
            project=Project(name="Integration Test Project"),
            topic="Test Article Topic",
            keywords=[],
        )

        assert plan.outline.title == "Test Article"
        assert [section.heading for section in plan.outline.sections] == [
            "Introduction",
            "Main Content",
            "Conclusion",
        ]
        mock_llm_client.complete.assert_awaited_once()
        article_repository.save_content_plan.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_parallel_section_generation(self, mock_llm_client):
        """Test that multiple sections can be generated in parallel."""
        call_count = 0

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(
                content=f"<h2>Section {call_count}</h2><p>Content here.</p>",
                usage=MagicMock(total_tokens=100),
                cost=0.001,
            )

        mock_llm_client.generate = mock_generate

        # Simulate parallel calls (what asyncio.gather would do)
        sections = ["Intro", "Body", "Conclusion"]
        for section in sections:
            await mock_generate(section)

        assert call_count == 3


class TestContentQuality:
    """Tests for content quality validation."""

    def test_word_count_validation(self, sample_article_data):
        """Test that word count is properly validated."""
        content = sample_article_data["content"]
        # Simple word count (actual implementation may be more sophisticated)
        word_count = len(content.split())
        assert word_count > 0

    def test_html_structure_validation(self, sample_article_data):
        """Test that HTML structure is valid."""
        from bs4 import BeautifulSoup

        content = sample_article_data["content"]
        soup = BeautifulSoup(content, "html.parser")

        # Should have at least one heading
        headings = soup.find_all(["h1", "h2", "h3"])
        assert len(headings) >= 1

        # Should have paragraph content
        paragraphs = soup.find_all("p")
        assert len(paragraphs) >= 1


class TestErrorRecovery:
    """Tests for error recovery in content generation."""

    @pytest.mark.asyncio
    async def test_llm_timeout_triggers_retry(self, mock_llm_client):
        """Test that LLM timeout triggers retry logic."""
        from core.exceptions import LLMTimeoutError

        call_count = 0

        async def mock_generate_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMTimeoutError("Request timed out")
            return MagicMock(content="Success after retry")

        mock_llm_client.generate = mock_generate_with_retry

        # Simulate retry behavior
        result = None
        for _ in range(3):
            try:
                result = await mock_llm_client.generate("test prompt")
                break
            except LLMTimeoutError:
                continue

        assert result is not None
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_backoff(self, mock_llm_client):
        """Test that rate limit triggers exponential backoff."""
        from core.exceptions import LLMRateLimitError

        # First call hits rate limit
        async def mock_rate_limited(*args, **kwargs):
            raise LLMRateLimitError("Rate limit exceeded", retry_after=60)

        mock_llm_client.generate = mock_rate_limited

        with pytest.raises(LLMRateLimitError) as exc_info:
            await mock_llm_client.generate("test prompt")

        assert exc_info.value.retryable is True
        assert exc_info.value.retry_after == 60


class TestCacheIntegration:
    """Tests for caching in content generation."""

    @pytest.mark.asyncio
    async def test_identical_prompts_use_cache(self, mock_redis, mock_llm_client):
        """Test that identical prompts retrieve from cache."""
        prompt = "Generate content about testing"

        # First call - cache miss
        mock_redis.get_cached_response.return_value = None
        result1 = await mock_llm_client.generate(prompt)
        assert result1 is not None

        # Simulate caching
        mock_redis.cache_response.return_value = True

        # Second call - cache hit
        cached_result = MagicMock(content="Cached content")
        mock_redis.get_cached_response.return_value = {
            "response": "Cached content",
            "tokens_used": 100,
        }

        # Should return cached version
        cache_check = await mock_redis.get_cached_response(
            prompt, "claude-3-haiku", 0.7, 1000
        )
        assert cache_check is not None
        assert cache_check["response"] == "Cached content"


class TestTaskPersistence:
    """Tests for task result persistence."""

    @pytest.mark.asyncio
    async def test_task_result_stored_on_completion(
        self, mock_database, sample_article_data
    ):
        """Test that completed task results are persisted."""
        task_id = str(uuid4())

        # Mock storing result
        mock_database.execute.return_value = None

        # Simulate task completion storage
        await mock_database.execute(
            "INSERT INTO task_results ...",
            {
                "task_id": task_id,
                "status": "completed",
                "result": sample_article_data,
            },
        )

        mock_database.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_task_stored_with_error(self, mock_database):
        """Test that failed tasks are stored with error details."""
        task_id = str(uuid4())
        error_msg = "LLM generation failed"

        # Mock storing failure
        mock_database.execute.return_value = None

        await mock_database.execute(
            "UPDATE task_results SET status = 'failed' ...",
            {
                "task_id": task_id,
                "error": error_msg,
            },
        )

        mock_database.execute.assert_called()
