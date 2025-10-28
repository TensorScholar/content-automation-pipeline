"""
Unit tests for ContentPlanner.

Tests the core logic for creating content plans using LLM responses
and proper JSON parsing without requiring actual LLM API calls.
"""

import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from core.models import Keyword, Project, ProjectContext
from execution.content_planner import ContentPlanner
from infrastructure.llm_client import AbstractLLMClient
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.decision_engine import DecisionEngine
from knowledge.article_repository import ArticleRepository
from optimization.model_router import ModelRouter


class TestContentPlanner:
    """Test ContentPlanner functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        mock = Mock(spec=AbstractLLMClient)
        mock.complete = AsyncMock()
        return mock

    @pytest.fixture
    def mock_context_synthesizer(self):
        """Mock ContextSynthesizer for testing."""
        mock = Mock(spec=ContextSynthesizer)
        mock.synthesize_context = AsyncMock()
        return mock

    @pytest.fixture
    def mock_decision_engine(self):
        """Mock DecisionEngine for testing."""
        mock = Mock(spec=DecisionEngine)
        return mock

    @pytest.fixture
    def mock_model_router(self):
        """Mock ModelRouter for testing."""
        mock = Mock(spec=ModelRouter)
        mock.route = AsyncMock()
        return mock

    @pytest.fixture
    def mock_article_repository(self):
        """Mock ArticleRepository for testing."""
        mock = Mock(spec=ArticleRepository)
        mock.save_content_plan = AsyncMock()
        return mock

    @pytest.fixture
    def mock_metrics_collector(self):
        """Mock MetricsCollector for testing."""
        mock = Mock(spec=MetricsCollector)
        return mock

    @pytest.fixture
    def content_planner(
        self,
        mock_decision_engine,
        mock_context_synthesizer,
        mock_model_router,
        mock_llm_client,
        mock_article_repository,
        mock_metrics_collector,
    ):
        """Create ContentPlanner instance for testing."""
        return ContentPlanner(
            decision_engine=mock_decision_engine,
            context_synthesizer=mock_context_synthesizer,
            model_router=mock_model_router,
            llm_client=mock_llm_client,
            article_repository=mock_article_repository,
            metrics_collector=mock_metrics_collector,
        )

    @pytest.fixture
    def sample_project(self):
        """Create a sample Project for testing."""
        return Project(id=uuid4(), name="Test Project", domain="https://example.com")

    @pytest.fixture
    def sample_keywords(self):
        """Create sample keywords for testing."""
        return [
            Keyword(
                phrase="test keyword",
                search_volume=1000,
                difficulty=0.5,
                intent="informational",
                embedding=[0.1] * 384,
                related_concepts=["concept1", "concept2"],
            )
        ]

    @pytest.mark.asyncio
    async def test_create_content_plan_success(
        self,
        content_planner,
        sample_project,
        sample_keywords,
        mock_llm_client,
        mock_article_repository,
    ):
        """Test successful content plan creation."""
        # Mock LLM response with valid JSON
        mock_llm_response = Mock()
        mock_llm_response.content = json.dumps(
            {
                "title": "Test Article Title",
                "meta_description": "This is a comprehensive test meta description that meets the minimum length requirement of 50 characters for validation.",
                "target_word_count": 1500,
                "target_audience": "technical professionals",
                "tone": "professional",
                "sections": [
                    {"heading": "Introduction", "prompt": "Introduce the topic"},
                    {"heading": "Main Content", "prompt": "Explain the main concepts"},
                    {"heading": "Conclusion", "prompt": "Summarize key points"},
                ],
            }
        )
        mock_llm_client.complete.return_value = mock_llm_response

        # Mock model router response
        mock_routing_decision = Mock()
        mock_routing_decision.selected_model = "gpt-4"
        content_planner.model_router.route.return_value = mock_routing_decision

        # Execute
        result = await content_planner.create_content_plan(
            project=sample_project,
            topic="Test Topic",
            keywords=sample_keywords,
            custom_instructions="Test instructions",
        )

        # Verify
        assert result is not None
        assert result.project_id == sample_project.id
        assert result.topic == "Test Topic"
        assert result.outline.title == "Test Article Title"
        assert len(result.outline.sections) == 3
        assert result.target_word_count == 1500

        # Verify that the plan was saved
        mock_article_repository.save_content_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_content_plan_invalid_json(
        self, content_planner, sample_project, sample_keywords, mock_llm_client
    ):
        """Test content plan creation with invalid JSON response."""
        # Mock LLM response with invalid JSON
        mock_llm_response = Mock()
        mock_llm_response.content = "This is not valid JSON"
        mock_llm_client.complete.return_value = mock_llm_response

        # Mock model router response
        mock_routing_decision = Mock()
        mock_routing_decision.selected_model = "gpt-4"
        content_planner.model_router.route.return_value = mock_routing_decision

        # Execute and verify exception
        with pytest.raises(Exception):  # Should raise WorkflowError
            await content_planner.create_content_plan(
                project=sample_project, topic="Test Topic", keywords=sample_keywords
            )

    @pytest.mark.asyncio
    async def test_create_content_plan_llm_error(
        self, content_planner, sample_project, sample_keywords, mock_llm_client
    ):
        """Test content plan creation when LLM call fails."""
        # Mock LLM error
        mock_llm_client.complete.side_effect = Exception("LLM API Error")

        # Mock model router response
        mock_routing_decision = Mock()
        mock_routing_decision.selected_model = "gpt-4"
        content_planner.model_router.route.return_value = mock_routing_decision

        # Execute and verify exception
        with pytest.raises(Exception):  # Should raise WorkflowError
            await content_planner.create_content_plan(
                project=sample_project, topic="Test Topic", keywords=sample_keywords
            )

    def test_build_planning_prompt(self, content_planner, sample_keywords):
        """Test that planning prompt is built correctly."""
        topic = "Test Topic"
        context = ProjectContext(
            target_audience="developers",
            tone="technical",
            style_guide="formal",
            custom_instructions="Focus on practical examples",
        )

        prompt = content_planner._build_planning_prompt(topic, sample_keywords, context)

        # Verify prompt contains required elements
        assert topic in prompt
        assert "test keyword" in prompt
        assert "developers" in prompt
        assert "technical" in prompt
        assert "formal" in prompt
        assert "Focus on practical examples" in prompt
        assert "JSON" in prompt
        assert "title" in prompt
        assert "sections" in prompt

    def test_parse_llm_json_response_valid(self, content_planner):
        """Test parsing valid JSON response."""
        json_content = '{"title": "Test", "sections": []}'

        result = content_planner._parse_llm_json_response(json_content)

        assert result["title"] == "Test"
        assert result["sections"] == []

    def test_parse_llm_json_response_with_markdown(self, content_planner):
        """Test parsing JSON response wrapped in markdown."""
        json_content = '```json\n{"title": "Test", "sections": []}\n```'

        result = content_planner._parse_llm_json_response(json_content)

        assert result["title"] == "Test"
        assert result["sections"] == []

    def test_parse_llm_json_response_invalid(self, content_planner):
        """Test parsing invalid JSON response."""
        invalid_json = "This is not JSON"

        with pytest.raises(Exception):  # Should raise WorkflowError
            content_planner._parse_llm_json_response(invalid_json)

    @pytest.mark.asyncio
    async def test_create_content_plan_calls_model_router(
        self, content_planner, sample_project, sample_keywords, mock_model_router
    ):
        """Test that ContentPlanner calls ModelRouter correctly."""
        # Mock LLM response
        mock_llm_response = Mock()
        mock_llm_response.content = json.dumps(
            {
                "title": "Test Article",
                "meta_description": "This is a comprehensive test meta description that meets the minimum length requirement of 50 characters for validation.",
                "target_word_count": 1000,
                "target_audience": "test audience",
                "tone": "test tone",
                "sections": [{"heading": "Test Section", "prompt": "Test prompt"}],
            }
        )
        content_planner.llm_client.complete.return_value = mock_llm_response

        # Mock routing decision
        mock_routing_decision = Mock()
        mock_routing_decision.selected_model = "gpt-4"
        mock_model_router.route.return_value = mock_routing_decision

        await content_planner.create_content_plan(
            project=sample_project, topic="Test Topic", keywords=sample_keywords
        )

        # Verify ModelRouter was called
        mock_model_router.route.assert_called_once()
        call_args = mock_model_router.route.call_args[0][0]
        assert call_args.capability_required.value == "reasoning"
        assert call_args.complexity.value == "complex"
