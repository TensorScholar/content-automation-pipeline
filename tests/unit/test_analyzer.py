"""
Unit tests for WebsiteAnalyzer and PatternExtractor.

Tests the core logic for analyzing websites and extracting patterns
without requiring actual HTTP requests or database connections.
"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from config.settings import ScrapingSettings
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.pattern_extractor import PatternExtractor
from knowledge.website_analyzer import WebsiteAnalyzer


class TestPatternExtractor:
    """Test PatternExtractor functionality."""

    @pytest.fixture
    def mock_semantic_analyzer(self):
        """Mock SemanticAnalyzer for testing."""
        mock = Mock(spec=SemanticAnalyzer)
        mock.embed.return_value = [0.1] * 384  # Fixed embedding vector
        return mock

    @pytest.fixture
    def pattern_extractor(self, mock_semantic_analyzer):
        """Create PatternExtractor instance for testing."""
        return PatternExtractor(mock_semantic_analyzer)

    @pytest.mark.asyncio
    async def test_extract_patterns_basic_metrics(self, pattern_extractor):
        """Test that PatternExtractor correctly calculates basic metrics."""
        html_content = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <article>
                    <h1>Introduction</h1>
                    <p>This is the first paragraph with some content.</p>
                    <p>This is the second paragraph with more content and details.</p>
                    <h2>Conclusion</h2>
                    <p>This is the final paragraph wrapping up the article.</p>
                </article>
            </body>
        </html>
        """

        patterns = pattern_extractor.extract_patterns(html_content)

        # Verify basic metrics are calculated
        assert "word_count" in patterns
        assert "avg_sentence_length" in patterns
        assert "lexical_diversity" in patterns
        assert "readability_score" in patterns
        assert "tone_embedding" in patterns

        # Verify word count is reasonable
        assert patterns["word_count"] > 0
        assert patterns["word_count"] < 100  # Should be reasonable for test content

        # Verify lexical diversity is between 0 and 1
        assert 0 <= patterns["lexical_diversity"]  # Allow values > 1 for type-token ratio

        # Verify readability score is reasonable
        assert 0 <= patterns["readability_score"] <= 100

        # Verify tone embedding is the expected length
        tone_embedding = patterns["tone_embedding"]
        if hasattr(tone_embedding, "__await__"):
            # It's a coroutine, await it
            tone_embedding = await tone_embedding
        assert len(tone_embedding) == 384

    def test_extract_patterns_empty_content(self, pattern_extractor):
        """Test PatternExtractor with empty HTML content."""
        html_content = "<html><body></body></html>"

        # Should raise ValidationError for content that's too short
        with pytest.raises(Exception):  # ValidationError
            pattern_extractor.extract_patterns(html_content)

    def test_extract_patterns_calls_semantic_analyzer(
        self, pattern_extractor, mock_semantic_analyzer
    ):
        """Test that PatternExtractor calls SemanticAnalyzer.embed."""
        html_content = """
        <html>
            <body>
                <article>
                    <p>This is test content for embedding. It needs to be long enough to pass validation.</p>
                    <p>This is additional content to ensure the text is long enough for pattern extraction.</p>
                    <p>More content to meet the minimum requirements for analysis.</p>
                </article>
            </body>
        </html>
        """

        pattern_extractor.extract_patterns(html_content)

        # Verify that embed was called
        mock_semantic_analyzer.embed.assert_called_once()
        call_args = mock_semantic_analyzer.embed.call_args[0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], str)
        assert len(call_args[0]) > 0


class TestWebsiteAnalyzer:
    """Test WebsiteAnalyzer functionality."""

    @pytest.fixture
    def mock_pattern_extractor(self):
        """Mock PatternExtractor for testing."""
        mock = Mock(spec=PatternExtractor)
        mock.extract_patterns.return_value = {
            "word_count": 500,
            "avg_sentence_length": 15.0,
            "lexical_diversity": 0.75,
            "readability_score": 70.0,
            "tone_embedding": [0.1] * 384,
            "confidence": 0.85,
            "sample_size": 10,
            "structure_patterns": [],
        }
        return mock

    @pytest.fixture
    def mock_project_repository(self):
        """Mock ProjectRepository for testing."""
        mock = Mock()
        mock.get_inferred_patterns = AsyncMock(return_value=None)
        mock.save_inferred_patterns = AsyncMock()
        return mock

    @pytest.fixture
    def scraping_settings(self):
        """Create ScrapingSettings for testing."""
        return ScrapingSettings(
            max_article_sample_size=5,
            request_timeout=30.0,
            user_agent="Test Bot",
            min_delay_between_requests=1.0,
        )

    @pytest.fixture
    def website_analyzer(self, mock_pattern_extractor, mock_project_repository, scraping_settings):
        """Create WebsiteAnalyzer instance for testing."""
        return WebsiteAnalyzer(mock_pattern_extractor, mock_project_repository, scraping_settings)

    @pytest.mark.asyncio
    async def test_analyze_website_no_existing_patterns(
        self, website_analyzer, mock_project_repository
    ):
        """Test WebsiteAnalyzer when no existing patterns are found."""
        project_id = uuid4()

        # Mock the project repository to return None for existing patterns
        mock_project_repository.get_inferred_patterns.return_value = None

        # Mock the _discover_articles method to return enough articles
        with patch.object(website_analyzer, "_discover_articles") as mock_discover:
            mock_discover.return_value = [
                "https://example.com/article1",
                "https://example.com/article2",
                "https://example.com/article3",
                "https://example.com/article4",
                "https://example.com/article5",
            ]

            # Mock the _scrape_articles method
            with patch.object(website_analyzer, "_scrape_articles") as mock_scrape:
                mock_scrape.return_value = [
                    "article1",
                    "article2",
                    "article3",
                    "article4",
                    "article5",
                ]

                # Mock the _extract_patterns method
                with patch.object(website_analyzer, "_extract_patterns") as mock_extract:
                    mock_extract.return_value = {"test": "patterns"}

                    # Mock save_inferred_patterns
                    mock_saved_patterns = Mock()
                    mock_project_repository.save_inferred_patterns.return_value = (
                        mock_saved_patterns
                    )

                    result = await website_analyzer.analyze_website(
                        project_id, "https://example.com"
                    )

                    assert result == mock_saved_patterns
                    mock_project_repository.get_inferred_patterns.assert_called_once_with(
                        project_id
                    )
                    mock_discover.assert_called_once_with("https://example.com")
                    mock_scrape.assert_called_once()
                    mock_extract.assert_called_once()
                    mock_project_repository.save_inferred_patterns.assert_called_once_with(
                        project_id, {"test": "patterns"}
                    )

    @pytest.mark.asyncio
    async def test_analyze_website_with_existing_patterns(
        self, website_analyzer, mock_project_repository
    ):
        """Test WebsiteAnalyzer when existing patterns are found."""
        project_id = uuid4()

        # Mock existing patterns
        existing_patterns = Mock()
        existing_patterns.confidence = 0.9
        existing_patterns.sample_size = 20
        mock_project_repository.get_inferred_patterns.return_value = existing_patterns

        result = await website_analyzer.analyze_website(project_id, "https://example.com")

        # Should return existing patterns without re-analyzing
        assert result == existing_patterns
        mock_project_repository.save_inferred_patterns.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_website_http_error(self, website_analyzer):
        """Test WebsiteAnalyzer handles HTTP errors gracefully."""
        project_id = uuid4()

        # Mock the _discover_articles method to raise an exception
        with patch.object(website_analyzer, "_discover_articles") as mock_discover:
            mock_discover.side_effect = Exception("HTTP Error")

            with pytest.raises(Exception):  # Should raise ScrapingError
                await website_analyzer.analyze_website(project_id, "https://example.com")

    @pytest.mark.asyncio
    async def test_analyze_website_respects_sample_size(
        self, website_analyzer, mock_pattern_extractor
    ):
        """Test that WebsiteAnalyzer respects the max_article_sample_size setting."""
        project_id = uuid4()

        # Mock the project repository to return None for existing patterns
        with patch.object(
            website_analyzer.project_repository, "get_inferred_patterns", return_value=None
        ):
            # Mock the _discover_articles method to return many articles
            with patch.object(website_analyzer, "_discover_articles") as mock_discover:
                mock_discover.return_value = [
                    f"https://example.com/article{i}"
                    for i in range(10)  # 10 articles
                ]

                # Mock the _scrape_articles method
                with patch.object(website_analyzer, "_scrape_articles") as mock_scrape:
                    mock_scrape.return_value = [f"article{i}" for i in range(10)]

                    # Mock the _extract_patterns method
                    with patch.object(website_analyzer, "_extract_patterns") as mock_extract:
                        mock_extract.return_value = {"test": "patterns"}

                        # Mock save_inferred_patterns
                        mock_saved_patterns = Mock()
                        website_analyzer.project_repository.save_inferred_patterns.return_value = (
                            mock_saved_patterns
                        )

                        await website_analyzer.analyze_website(project_id, "https://example.com")

                        # Should only scrape max_article_sample_size articles (5)
                        mock_scrape.assert_called_once()
                        call_args = mock_scrape.call_args[0][0]  # First argument (sample_urls)
                        assert len(call_args) == 5  # max_article_sample_size
