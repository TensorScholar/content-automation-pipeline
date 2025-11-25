"""
Unit tests for WordPress Distributor.

Tests the core logic for distributing articles to WordPress sites
using the REST API without requiring actual HTTP requests.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from core.exceptions import DistributionError
from core.models import GeneratedArticle, Project
from execution.distributer import Distributor


class TestWordPressDistributor:
    """Test WordPress distribution functionality."""

    @pytest.fixture
    def distributor(self):
        """Create Distributor instance for testing."""
        return Distributor()


@pytest.fixture
def sample_article(self):
    """Create a sample GeneratedArticle for testing."""
    from core.models import QualityMetrics

    return GeneratedArticle(
        id=uuid4(),
        project_id=uuid4(),
        content_plan_id=uuid4(),
        title="Test Article Title",
        content="<h1>Test Article</h1><p>This is test content that meets the minimum length requirement of 100 characters for validation purposes.</p><p>Additional content to ensure we have enough text for proper testing.</p>",
        meta_description="This is a comprehensive test meta description that meets the minimum length requirement of 50 characters for validation.",
        quality_metrics=QualityMetrics(
            readability_score=75.0,
            seo_score=80.0,
            engagement_score=70.0,
            factual_accuracy=95.0,
            coherence_score=85.0,
            word_count=500,
            lexical_diversity=0.75,
            avg_sentence_length=15.0,
            paragraph_count=3,
        ),
        total_tokens_used=650,
        total_cost_usd=0.05,
        generation_time_seconds=30.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    @pytest.fixture
    def sample_project_with_wordpress(self):
        """Create a sample Project with WordPress credentials."""
        from pydantic import SecretStr

        return Project(
            id=uuid4(),
            name="Test Project",
            domain="https://example.com",
            wordpress_url="https://test-site.com",
            wordpress_username="testuser",
            wordpress_app_password=SecretStr("test-password"),
        )

    @pytest.fixture
    def sample_project_without_wordpress(self):
        """Create a sample Project without WordPress credentials."""
        return Project(id=uuid4(), name="Test Project", domain="https://example.com")

    @pytest.mark.asyncio
    async def test_distribute_to_wordpress_success(
        self, distributor, sample_article, sample_project_with_wordpress
    ):
        """Test successful WordPress distribution."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {
                "id": 123,
                "link": "https://test-site.com/test-article-title",
            }
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await distributor.distribute_to_wordpress(
                sample_article, sample_project_with_wordpress
            )

            # Verify result
            assert result["status"] == "published"
            assert result["url"] == "https://test-site.com/test-article-title"

            # Verify HTTP request was made correctly
            mock_client.return_value.__aenter__.return_value.post.assert_called_once()
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args

            assert call_args[1]["url"] == "https://test-site.com/wp-json/wp/v2/posts"
            assert call_args[1]["timeout"] == 30.0

            # Verify JSON payload
            json_data = call_args[1]["json"]
            assert json_data["title"] == "Test Article Title"
            assert json_data["content"] == "<h1>Test Article</h1><p>This is test content.</p>"
            assert json_data["status"] == "publish"
            assert json_data["meta"]["_yoast_wpseo_metadesc"] == "Test meta description"

    @pytest.mark.asyncio
    async def test_distribute_to_wordpress_no_credentials(
        self, distributor, sample_article, sample_project_without_wordpress
    ):
        """Test WordPress distribution when credentials are not configured."""
        result = await distributor.distribute_to_wordpress(
            sample_article, sample_project_without_wordpress
        )

        # Should return skipped status
        assert result["status"] == "skipped"
        assert result["reason"] == "Not configured"

    @pytest.mark.asyncio
    async def test_distribute_to_wordpress_http_error(
        self, distributor, sample_article, sample_project_with_wordpress
    ):
        """Test WordPress distribution with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock HTTP error
            import httpx

            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"

            http_error = httpx.HTTPStatusError(
                "Unauthorized", request=Mock(), response=mock_response
            )
            mock_client.return_value.__aenter__.return_value.post.side_effect = http_error

            # Should raise DistributionError
            with pytest.raises(DistributionError, match="WordPress API error"):
                await distributor.distribute_to_wordpress(
                    sample_article, sample_project_with_wordpress
                )

    @pytest.mark.asyncio
    async def test_distribute_to_wordpress_general_error(
        self, distributor, sample_article, sample_project_with_wordpress
    ):
        """Test WordPress distribution with general error."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock general error
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception(
                "Network error"
            )

            # Should raise DistributionError
            with pytest.raises(DistributionError, match="Network error"):
                await distributor.distribute_to_wordpress(
                    sample_article, sample_project_with_wordpress
                )

    @pytest.mark.asyncio
    async def test_distribute_to_wordpress_auth_headers(
        self, distributor, sample_article, sample_project_with_wordpress
    ):
        """Test that WordPress distribution uses correct authentication."""
        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 123, "link": "https://test.com/test"}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            await distributor.distribute_to_wordpress(sample_article, sample_project_with_wordpress)

            # Verify authentication was used
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            auth = call_args[1]["auth"]

            assert auth.username == "testuser"
            assert auth.password == "test-password"

    @pytest.mark.asyncio
    async def test_distribute_to_wordpress_url_stripping(self, distributor, sample_article):
        """Test that WordPress URL is properly stripped of trailing slashes."""
        from pydantic import SecretStr

        project_with_trailing_slash = Project(
            id=uuid4(),
            name="Test Project",
            wordpress_url="https://test-site.com/",
            wordpress_username="testuser",
            wordpress_app_password=SecretStr("test-password"),
        )

        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 123, "link": "https://test.com/test"}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            await distributor.distribute_to_wordpress(sample_article, project_with_trailing_slash)

            # Verify URL was stripped
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            assert call_args[1]["url"] == "https://test-site.com/wp-json/wp/v2/posts"

    @pytest.mark.asyncio
    async def test_distribute_to_wordpress_keywords_as_tags(
        self, distributor, sample_article, sample_project_with_wordpress
    ):
        """Test that article keywords are converted to WordPress tags."""
        # Add keywords to the article (using the keywords field from GeneratedArticle)
        sample_article.keywords = ["keyword1", "keyword2", "keyword3"]

        with patch("httpx.AsyncClient") as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 201
            mock_response.json.return_value = {"id": 123, "link": "https://test.com/test"}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            await distributor.distribute_to_wordpress(sample_article, sample_project_with_wordpress)

            # Verify keywords were added as tags
            call_args = mock_client.return_value.__aenter__.return_value.post.call_args
            json_data = call_args[1]["json"]
            assert json_data["tags"] == "keyword1, keyword2, keyword3"
