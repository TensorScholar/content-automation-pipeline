"""
Comprehensive WordPress Integration Tests
=========================================

Tests the complete WordPress distribution workflow including:
- Connection validation with real WordPress REST API
- Article publishing with Gutenberg blocks
- Schema.org markup generation
- Retry logic with exponential backoff
- Error handling for various failure scenarios
- Connection pooling and timeout configuration
- Network resilience patterns

Author: Senior SRE Team
Date: December 29, 2025
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from core.exceptions import DistributionError
from core.models import GeneratedArticle, Project, QualityMetrics
from execution.distributer import Distributor, WordPressPublishError

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_article():
    """Create a sample article for testing."""
    return GeneratedArticle(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        content_plan_id=uuid.uuid4(),
        topic="تست موضوع",
        title="راهنمای کامل تست WordPress",
        content="""
        <h1>راهنمای کامل تست WordPress</h1>
        <h2>مقدمه</h2>
        <p>این یک پاراگراف تستی است.</p>
        <h2>بخش اول</h2>
        <p>محتوای بخش اول با <strong>متن برجسته</strong> و <em>تاکید</em>.</p>
        <ul>
            <li>آیتم اول</li>
            <li>آیتم دوم</li>
        </ul>
        <h3>زیربخش</h3>
        <p>محتوای زیربخش.</p>
        <img src="https://example.com/image.jpg" alt="تصویر تست" />
        <blockquote>این یک نقل قول است.</blockquote>
        """,
        meta_description="این یک توضیح متا برای تست است که باید بین 120 تا 160 کاراکتر باشد تا در نتایج جستجو به درستی نمایش داده شود.",
        keywords=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        quality_metrics=QualityMetrics(
            readability_score=75.0,
            word_count=250,
            lexical_diversity=0.72,
            keyword_density={"تست": 0.03, "wordpress": 0.02},
            avg_sentence_length=14.0,
            paragraph_count=5,
        ),
        total_cost_usd=0.05,
        total_tokens_used=500,
        generation_time_seconds=15.0
    )


@pytest.fixture
def sample_project():
    """Create a sample project with WordPress credentials."""
    return Project(
        id=uuid.uuid4(),
        name="تست پروژه",
        description="پروژه تستی برای WordPress",
        domain="https://test-blog.com",
        wordpress_url="https://test-blog.com",
        wordpress_username="testuser",
        wordpress_app_password=SecretStr("test_app_password_123"),
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def distributor():
    """Create a Distributor with remote verification isolated for legacy tests."""
    instance = Distributor(max_retries=5, initial_retry_delay=0.001)

    async def verified_post(*, post_id, expected_slug, expected_status, **kwargs):
        return {
            "id": int(post_id),
            "slug": expected_slug,
            "status": expected_status,
            "link": f"https://test-blog.com/posts/{post_id}",
        }

    instance._verify_wordpress_post = AsyncMock(side_effect=verified_post)
    return instance


# ============================================================================
# CONNECTION VALIDATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_validate_wordpress_connection_success(distributor, sample_project):
    """Test successful WordPress connection validation."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": 7,
        "capabilities": {"edit_posts": True, "publish_posts": True},
    }

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        is_valid, error_msg = await distributor.validate_wordpress_connection(sample_project)

        assert is_valid is True
        assert error_msg == ""
        mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_validate_wordpress_connection_rejects_missing_publish_permission(
    distributor,
    sample_project,
):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": 7,
        "capabilities": {"edit_posts": True, "publish_posts": False},
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        is_valid, error_msg = await distributor.validate_wordpress_connection(
            sample_project,
            required_status="publish",
        )

    assert is_valid is False
    assert "permission to publish" in error_msg


@pytest.mark.asyncio
async def test_validate_wordpress_connection_invalid_credentials(distributor, sample_project):
    """Test WordPress connection validation with invalid credentials."""
    mock_response = MagicMock()
    mock_response.status_code = 401

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        is_valid, error_msg = await distributor.validate_wordpress_connection(sample_project)

        assert is_valid is False
        assert "Invalid WordPress credentials" in error_msg


@pytest.mark.asyncio
async def test_validate_wordpress_connection_api_not_found(distributor, sample_project):
    """Test WordPress connection validation when REST API is not available."""
    mock_response = MagicMock()
    mock_response.status_code = 404

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        is_valid, error_msg = await distributor.validate_wordpress_connection(sample_project)

        assert is_valid is False
        assert "REST API not found" in error_msg


@pytest.mark.asyncio
async def test_validate_wordpress_connection_timeout(distributor, sample_project):
    """Test WordPress connection validation with timeout."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.side_effect = httpx.TimeoutException("Connection timeout")

    with patch("httpx.AsyncClient", return_value=mock_client):
        is_valid, error_msg = await distributor.validate_wordpress_connection(sample_project)

        assert is_valid is False
        assert "timeout" in error_msg.lower()


@pytest.mark.asyncio
async def test_validate_wordpress_connection_missing_credentials(distributor, sample_project):
    """Test WordPress connection validation with missing credentials."""
    sample_project.wordpress_url = None

    is_valid, error_msg = await distributor.validate_wordpress_connection(sample_project)

    assert is_valid is False
    assert "not configured" in error_msg


# ============================================================================
# ARTICLE DISTRIBUTION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_distribute_to_wordpress_success(distributor, sample_article, sample_project):
    """Test successful article distribution to WordPress."""
    # Mock validation
    mock_validation = AsyncMock(return_value=(True, ""))

    # Mock post creation response
    mock_post_response = MagicMock()
    mock_post_response.status_code = 201
    mock_post_response.json.return_value = {
        "id": 123,
        "link": "https://test-blog.com/posts/test-article"
    }
    mock_post_response.raise_for_status = MagicMock()

    # Mock schema update response
    mock_update_response = MagicMock()
    mock_update_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.side_effect = [mock_post_response, mock_update_response]

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await distributor.distribute_to_wordpress(sample_article, sample_project)

            assert result["status"] == "published"
            assert result["post_id"] == 123
            assert result["url"] == "https://test-blog.com/posts/123"
            assert result["gutenberg_formatted"] is True
            assert result["schema_generated"] is True
            assert result["attempts"] == 1


@pytest.mark.asyncio
async def test_distribute_to_wordpress_with_retry(distributor, sample_article, sample_project):
    """Test article distribution with retry on transient failure."""
    mock_validation = AsyncMock(return_value=(True, ""))

    # First attempt fails with 503, second succeeds
    mock_fail_response = MagicMock()
    mock_fail_response.status_code = 503
    mock_fail_response.text = "Service temporarily unavailable"
    mock_fail_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable",
        request=MagicMock(),
        response=mock_fail_response
    )

    mock_success_response = MagicMock()
    mock_success_response.status_code = 201
    mock_success_response.json.return_value = {"id": 456, "link": "https://test-blog.com/p/456"}
    mock_success_response.raise_for_status = MagicMock()

    mock_update_response = MagicMock()
    mock_update_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.side_effect = [
        mock_fail_response,  # First attempt fails
        mock_success_response,  # Second attempt succeeds
        mock_update_response  # Schema update
    ]

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await distributor.distribute_to_wordpress(sample_article, sample_project)

            assert result["status"] == "published"
            assert result["attempts"] == 2  # Took 2 attempts


@pytest.mark.asyncio
async def test_distribute_to_wordpress_permanent_failure(distributor, sample_article, sample_project):
    """Test article distribution with permanent 4xx error (no retry)."""
    mock_validation = AsyncMock(return_value=(True, ""))

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad request"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400 Bad Request",
        request=MagicMock(),
        response=mock_response
    )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_response

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(DistributionError) as exc_info:
                await distributor.distribute_to_wordpress(sample_article, sample_project)

            assert "WordPress API error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_distribute_to_wordpress_exhausted_retries(distributor, sample_article, sample_project):
    """Test article distribution when all retries are exhausted."""
    mock_validation = AsyncMock(return_value=(True, ""))

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service unavailable"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503 Service Unavailable",
        request=MagicMock(),
        response=mock_response
    )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_response

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(DistributionError) as exc_info:
                await distributor.distribute_to_wordpress(sample_article, sample_project)

            assert "failed after" in str(exc_info.value).lower()
            assert mock_client.post.call_count == distributor.max_retries


@pytest.mark.asyncio
async def test_distribute_to_wordpress_network_error(distributor, sample_article, sample_project):
    """Test article distribution with network error and retry."""
    mock_validation = AsyncMock(return_value=(True, ""))

    mock_success_response = MagicMock()
    mock_success_response.status_code = 201
    mock_success_response.json.return_value = {"id": 789, "link": "https://test-blog.com/p/789"}
    mock_success_response.raise_for_status = MagicMock()

    mock_update_response = MagicMock()
    mock_update_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.side_effect = [
        httpx.NetworkError("Network unreachable"),  # First attempt fails
        mock_success_response,  # Second attempt succeeds
        mock_update_response  # Schema update
    ]

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await distributor.distribute_to_wordpress(sample_article, sample_project)

            assert result["status"] == "published"
            assert result["attempts"] == 2


@pytest.mark.asyncio
async def test_distribute_to_wordpress_validation_failure(distributor, sample_article, sample_project):
    """Test article distribution when validation fails."""
    mock_validation = AsyncMock(return_value=(False, "Invalid credentials"))

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with pytest.raises(WordPressPublishError) as exc_info:
            await distributor.distribute_to_wordpress(sample_article, sample_project)

        assert exc_info.value.category == "auth_error"
        assert "Invalid credentials" in exc_info.value.safe_message


# ============================================================================
# GUTENBERG BLOCK CONVERSION TESTS
# ============================================================================

def test_convert_to_gutenberg_blocks_headings(distributor):
    """Test Gutenberg block conversion for headings."""
    html = "<h2>Heading 2</h2><h3>Heading 3</h3><h4>Heading 4</h4>"

    result = distributor.convert_to_gutenberg_blocks(html)

    assert '<!-- wp:heading {"level":2} -->' in result
    assert '<!-- wp:heading {"level":3} -->' in result
    assert '<!-- wp:heading {"level":4} -->' in result
    assert '<!-- /wp:heading -->' in result


def test_convert_to_gutenberg_blocks_paragraphs(distributor):
    """Test Gutenberg block conversion for paragraphs."""
    html = "<p>First paragraph</p><p>Second paragraph</p>"

    result = distributor.convert_to_gutenberg_blocks(html)

    assert '<!-- wp:paragraph -->' in result
    assert '<!-- /wp:paragraph -->' in result
    assert result.count('<!-- wp:paragraph -->') == 2


def test_convert_to_gutenberg_blocks_lists(distributor):
    """Test Gutenberg block conversion for lists."""
    html = "<ul><li>Item 1</li><li>Item 2</li></ul><ol><li>Step 1</li></ol>"

    result = distributor.convert_to_gutenberg_blocks(html)

    assert '<!-- wp:list -->' in result
    assert '<!-- wp:list {"ordered":true} -->' in result
    assert '<!-- /wp:list -->' in result


def test_convert_to_gutenberg_blocks_blockquotes(distributor):
    """Test Gutenberg block conversion for blockquotes."""
    html = "<blockquote>This is a quote</blockquote>"

    result = distributor.convert_to_gutenberg_blocks(html)

    assert '<!-- wp:quote -->' in result
    assert '<blockquote class="wp-block-quote">' in result
    assert '<!-- /wp:quote -->' in result


def test_convert_to_gutenberg_blocks_preserves_blockquote_classes(distributor):
    """Test Gutenberg conversion merges the canonical quote class."""
    html = '<blockquote class="featured">This is a quote</blockquote>'

    result = distributor.convert_to_gutenberg_blocks(html)

    assert '<blockquote class="featured wp-block-quote">' in result


# ============================================================================
# SCHEMA.ORG MARKUP TESTS
# ============================================================================

def test_generate_schema_markup_basic(distributor, sample_article):
    """Test basic schema.org markup generation."""
    schema = distributor.generate_schema_markup(sample_article, "https://example.com/article")

    assert '<script type="application/ld+json">' in schema
    assert '"@context": "https://schema.org"' in schema
    assert '"@type": "Article"' in schema
    assert sample_article.title in schema
    assert sample_article.meta_description in schema
    assert "https://example.com/article" in schema


def test_generate_schema_markup_does_not_infer_faq_from_heading(distributor, sample_article):
    """FAQ markup must not be claimed without structured question/answer data."""
    sample_article.content += "\n<h2>سوالات متداول</h2>\n<p>پرسش و پاسخ</p>"

    schema = distributor.generate_schema_markup(sample_article, "https://example.com/article")

    assert '"@type": "Article"' in schema
    assert "FAQPage" not in schema


def test_generate_schema_markup_keywords(distributor, sample_article):
    """Test schema.org markup includes keywords."""
    sample_article.keywords = ["تست", "wordpress", "توزیع محتوا", "SEO", "وبلاگ"]
    schema = distributor.generate_schema_markup(sample_article)

    for keyword in sample_article.keywords:
        assert keyword in schema


# ============================================================================
# RETRY LOGIC AND BACKOFF TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_exponential_backoff_timing(distributor, sample_article, sample_project):
    """Test that exponential backoff increases wait time properly."""
    mock_validation = AsyncMock(return_value=(True, ""))

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "503", request=MagicMock(), response=mock_response
    )

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post = mock_post
    sleep_delays = []

    async def record_sleep(delay):
        sleep_delays.append(delay)

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", side_effect=record_sleep):
                try:
                    await distributor.distribute_to_wordpress(sample_article, sample_project)
                except DistributionError:
                    pass  # Expected

    # Assert the requested backoff durations rather than flaky wall-clock timing.
    assert len(sleep_delays) == distributor.max_retries - 1
    assert all(later > earlier for earlier, later in zip(sleep_delays, sleep_delays[1:]))


# ============================================================================
# EDGE CASES AND ERROR SCENARIOS
# ============================================================================

def test_generated_article_rejects_empty_content(sample_article):
    """Test invalid empty content is rejected before distribution."""
    with pytest.raises(ValueError, match="at least 100 characters"):
        sample_article.content = ""


@pytest.mark.asyncio
async def test_distribute_unicode_content(distributor, sample_article, sample_project):
    """Test distribution with Unicode/Persian content."""
    sample_article.title = "راهنمای کامل محتوای فارسی 🚀"
    sample_article.content = (
        "<p>این یک تست با محتوای فارسی و ایموجی است 😊 که برای بررسی انتشار "
        "صحیح نویسه‌های یونیکد در وردپرس به اندازه کافی طولانی نوشته شده است.</p>"
    )

    mock_validation = AsyncMock(return_value=(True, ""))
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": 111, "link": "https://test-blog.com/p/111"}
    mock_response.raise_for_status = MagicMock()

    mock_update_response = MagicMock()
    mock_update_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.side_effect = [mock_response, mock_update_response]

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await distributor.distribute_to_wordpress(sample_article, sample_project)

            assert result["status"] == "published"
            # Verify Unicode was properly sent
            call_kwargs = mock_client.post.call_args_list[0][1]
            assert "فارسی" in str(call_kwargs["json"])


@pytest.mark.asyncio
async def test_distributor_custom_retry_config():
    """Test Distributor with custom retry configuration."""
    custom_distributor = Distributor(max_retries=3, initial_retry_delay=0.5)

    assert custom_distributor.max_retries == 3
    assert custom_distributor.retry_delay == 0.5


# ============================================================================
# PERFORMANCE AND RESILIENCE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_connection_pooling_limits(distributor, sample_article, sample_project):
    """Test that connection pooling is properly configured."""
    mock_validation = AsyncMock(return_value=(True, ""))

    captured_limits = None

    def capture_client(*args, **kwargs):
        nonlocal captured_limits
        if kwargs.get("limits") is not None:
            captured_limits = kwargs["limits"]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1, "link": "https://test.com/p/1"}
        mock_response.raise_for_status = MagicMock()
        schema_response = MagicMock()
        schema_response.status_code = 200
        mock_client.post.side_effect = [mock_response, schema_response]

        return mock_client

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", side_effect=capture_client):
            await distributor.distribute_to_wordpress(sample_article, sample_project)

    # Verify connection pooling configuration
    assert captured_limits is not None
    assert captured_limits.max_keepalive_connections == 5
    assert captured_limits.max_connections == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


@pytest.mark.asyncio
async def test_optional_wordpress_metadata_failure_does_not_duplicate_or_fail_core_post(
    distributor,
    sample_article,
    sample_project,
):
    """Optional plugin metadata must be a warning after a verified core post write."""
    mock_validation = AsyncMock(return_value=(True, ""))
    create_response = MagicMock(status_code=201)
    create_response.json.return_value = {"id": 321, "link": "https://test-blog.com/posts/321"}
    create_response.raise_for_status = MagicMock()
    metadata_response = MagicMock(status_code=400)

    lookup_response = MagicMock(status_code=200)
    lookup_response.json.return_value = []
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = lookup_response
    mock_client.post.side_effect = [create_response, metadata_response]

    with patch.object(distributor, "validate_wordpress_connection", mock_validation):
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await distributor.distribute_to_wordpress(
                sample_article,
                sample_project,
                idempotency_key="wp:test",
            )

    assert result["status"] == "published"
    assert result["post_id"] == 321
    assert result["schema_stored"] is False
    assert result["seo_meta_stored"] is False
    assert any(
        warning["category"] == "optional_metadata_storage_failed"
        for warning in result["warnings"]
    )
    primary_payload = mock_client.post.call_args_list[0].kwargs["json"]
    assert "meta" not in primary_payload
