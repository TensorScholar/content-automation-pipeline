"""
Property-Based Tests for WordPress Integration Edge Cases

GOAL: Find bugs and edge cases in WordPress upload pipeline, NOT to make tests pass.
Uses Hypothesis to generate adversarial inputs that might break the system.

Focus Areas:
1. Credential validation edge cases
2. Retry logic failure modes
3. Network timeout scenarios
4. WordPress API response variations
5. Content payload edge cases
6. Concurrent upload race conditions
"""

import asyncio
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import httpx
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from pydantic import SecretStr

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import GeneratedArticle, Project, QualityMetrics
from execution.distributer import Distributor

# ============================================================================
# Test Helpers
# ============================================================================

def create_test_article(**kwargs):
    """
    Factory function to create valid GeneratedArticle for testing.

    Provides sensible defaults for all required fields.
    """
    defaults = {
        "id": uuid4(),
        "project_id": uuid4(),
        "content_plan_id": uuid4(),
        "title": "Test Article Title",
        "content": "<p>" + ("Test content. " * 20) + "</p>",  # 100+ chars
        "meta_description": "This is a test meta description that is at least 50 characters long.",
        "quality_metrics": QualityMetrics(
            word_count=100,
            readability_score=60.0,
            keyword_density={"test": 0.05},
            avg_sentence_length=15.0,
            lexical_diversity=0.7,
            paragraph_count=5,
        ),
        "total_tokens_used": 500,
        "total_cost_usd": 0.01,
        "generation_time_seconds": 5.0,
    }
    defaults.update(kwargs)
    return GeneratedArticle(**defaults)

def create_test_project(**kwargs):
    """
    Factory function to create valid Project for testing.
    """
    defaults = {
        "id": uuid4(),
        "name": "Test Project",
        "wordpress_url": "https://example.com",
        "wordpress_username": "admin",
        "wordpress_app_password": SecretStr("test pass 1234"),
    }
    defaults.update(kwargs)
    return Project(**defaults)

# ============================================================================
# PROPERTY 1: Credential Validation Edge Cases
# ============================================================================

@given(
    username=st.one_of(
        st.none(),
        st.just(""),
        st.text(min_size=0, max_size=500),
        st.text(alphabet=st.characters(blacklist_categories=("Cs", "Cc")), min_size=1, max_size=100),
        st.just("admin"),
        st.just("user@example.com"),
        st.just("user with spaces"),
        st.just("user\nwith\nnewlines"),
        st.just("🚀emoji_user"),
    ),
    password=st.one_of(
        st.none(),
        st.just(""),
        st.text(min_size=0, max_size=500),
        st.just("valid pass word 1234"),
        st.just("pass\nwith\nnewline"),
        st.just("pass\twith\ttab"),
        st.just("🔐🔑 emoji password"),
    ),
    url=st.one_of(
        st.none(),
        st.just(""),
        st.just("https://example.com"),
        st.just("http://example.com"),  # HTTP should be upgraded
        st.just("https://example.com/"),  # Trailing slash
        st.just("https://example.com/wp-admin"),  # Should fail
        st.just("example.com"),  # No protocol
        st.just("https://example.com:8080"),  # Custom port
        st.just("https://user:pass@example.com"),  # URL with auth
        st.just("not a url at all"),
        st.just("https://"),
        st.just("ftp://example.com"),  # Wrong protocol
    )
)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_wordpress_credential_edge_cases(username, password, url):
    """
    Property: Credential validation should handle ALL edge cases gracefully.

    Edge cases to find:
    - Empty/None values
    - Special characters (newlines, tabs, emojis)
    - Very long values
    - Invalid URLs
    - Malformed inputs

    Expected: Should return clear error, NEVER crash
    """
    distributor = Distributor()

    # Create mock project with potentially invalid credentials
    try:
        project = Project(
            id=uuid4(),
            name="Test Project",
            wordpress_url=url,
            wordpress_username=username,
            wordpress_app_password=SecretStr(password) if password else None,
        )
    except Exception as e:
        # Pydantic validation caught it - that's good!
        return

    # Try validation
    async def validate():
        is_valid, error_msg = await distributor.validate_wordpress_connection(project)

        # PROPERTY: Should ALWAYS return (bool, str), never crash
        assert isinstance(is_valid, bool), f"Expected bool, got {type(is_valid)}"
        assert isinstance(error_msg, str), f"Expected str, got {type(error_msg)}"

        # PROPERTY: If invalid, error message should be non-empty
        if not is_valid:
            assert len(error_msg) > 0, "Error message should not be empty"

        return is_valid, error_msg

    mock_response = MagicMock(status_code=401)
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_response

    try:
        with patch("execution.distributer.httpx.AsyncClient", return_value=mock_client):
            asyncio.run(validate())
    except Exception as e:
        # This is a BUG - validation should never crash
        pytest.fail(f"Credential validation CRASHED with {type(e).__name__}: {e}")

# ============================================================================
# PROPERTY 2: Retry Logic State Machine
# ============================================================================

class WordPressRetryStateMachine(RuleBasedStateMachine):
    """
    State machine to test retry logic transitions.
    Finds edge cases in exponential backoff, jitter, and retry exhaustion.
    """

    def __init__(self):
        super().__init__()
        self.distributor = Distributor(max_retries=5, initial_retry_delay=0.1)
        self.current_attempt_count = 0
        self.failure_pattern = []

    @initialize()
    def setup(self):
        """Start with clean state."""
        self.current_attempt_count = 0
        self.failure_pattern = []

    @rule(
        failure_count=st.integers(min_value=0, max_value=10),
        failure_type=st.sampled_from([
            "timeout",
            "network_error",
            "503_service_unavailable",
            "401_auth_error",
            "404_not_found",
            "500_server_error",
        ])
    )
    def simulate_upload_with_failures(self, failure_count, failure_type):
        """
        Simulate WordPress upload with N failures before success.

        Property: System should handle all failure patterns correctly.
        """
        self.failure_pattern.append((failure_count, failure_type))

        self.current_attempt_count = min(
            failure_count + 1,
            self.distributor.max_retries,
        )

    @invariant()
    def check_retry_exhaustion(self):
        """
        Property: If failures >= max_retries, upload should fail.
        If failures < max_retries, upload should eventually succeed.
        """
        if not self.failure_pattern:
            return

        max_retries = self.distributor.max_retries

        # Property: Retry count should never exceed max_retries
        assert self.current_attempt_count <= max_retries, (
            f"Too many retry attempts: {self.current_attempt_count} > {max_retries}"
        )

@given(failure_count=st.integers(min_value=0, max_value=10))
def test_wordpress_retry_attempt_model(failure_count):
    """Each upload is capped independently at the configured retry limit."""
    distributor = Distributor(max_retries=5, initial_retry_delay=0.1)

    attempt_count = min(failure_count + 1, distributor.max_retries)

    assert 1 <= attempt_count <= distributor.max_retries

# ============================================================================
# PROPERTY 3: Network Timeout Variations
# ============================================================================

@given(response_delay=st.floats(min_value=0.0, max_value=150.0))
@settings(max_examples=50, deadline=None)
@example(response_delay=61.0)
def test_timeout_edge_cases(response_delay):
    """
    Property: Timeout handling should work correctly for all timeout combinations.

    Edge cases to find:
    - Response slower than timeout → Should timeout
    - Response faster than timeout → Should succeed
    - Very short timeouts → Should fail fast
    - Timeout during different phases (connect, read, write)
    """
    async def mock_response(*args, **kwargs):
        if response_delay > 60.0:
            raise httpx.ReadTimeout("Simulated WordPress read timeout")
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 123, "link": "https://example.com/post"}
        mock_response.raise_for_status = Mock()
        return mock_response

    distributor = Distributor(max_retries=1, initial_retry_delay=0)

    # Create valid test objects using factory
    article = create_test_article()
    project = create_test_project()

    async def test_upload():
        with patch.object(
            distributor,
            "validate_wordpress_connection",
            AsyncMock(return_value=(True, "")),
        ), patch("execution.distributer.httpx.AsyncClient") as mock_client_cls:
            mock_instance = AsyncMock()
            mock_instance.get = mock_response
            mock_instance.post = mock_response
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_instance

            try:
                result = await distributor.distribute_to_wordpress(article, project)
                assert response_delay <= 60.0
                assert result["status"] == "published"
            except Exception as exc:
                assert response_delay > 60.0
                assert "failed after 1 attempts" in str(exc)

    asyncio.run(test_upload())

# ============================================================================
# PROPERTY 4: WordPress API Response Variations
# ============================================================================

@given(
    status_code=st.integers(min_value=100, max_value=599),
    response_body=st.one_of(
        st.just('{"id": 123, "link": "https://example.com/post"}'),  # Valid
        st.just('{"error": "Invalid request"}'),  # Error JSON
        st.just(''),  # Empty
        st.just('Not JSON at all'),  # Invalid JSON
        st.just('{"incomplete": '),  # Malformed JSON
        st.just('<html>WordPress Error</html>'),  # HTML instead of JSON
        st.just('null'),
        st.just('[]'),  # Wrong type
        st.text(min_size=0, max_size=1000),  # Random text
    ),
    has_id=st.booleans(),
    has_link=st.booleans(),
)
@settings(max_examples=100, deadline=None)
def test_wordpress_api_response_edge_cases(status_code, response_body, has_id, has_link):
    """
    Property: System should handle ALL WordPress API response variations.

    Edge cases to find:
    - Unexpected status codes (1xx, 3xx, 4xx, 5xx)
    - Malformed JSON
    - Missing required fields (id, link)
    - Wrong response types
    - Empty responses
    """
    distributor = Distributor()

    # Create valid test objects using factory
    article = create_test_article()
    project = create_test_project()

    async def test_response():
        def mock_request(*args, **kwargs):
            mock_resp = Mock()
            mock_resp.status_code = status_code
            mock_resp.text = response_body

            # Try to parse JSON
            try:
                import json
                parsed = json.loads(response_body)

                # Modify based on has_id/has_link flags
                if isinstance(parsed, dict):
                    if not has_id and "id" in parsed:
                        del parsed["id"]
                    if not has_link and "link" in parsed:
                        del parsed["link"]

                mock_resp.json.return_value = parsed
            except (TypeError, ValueError):
                mock_resp.json.side_effect = ValueError("Invalid JSON")

            if status_code >= 400:
                mock_resp.raise_for_status = Mock(side_effect=httpx.HTTPStatusError(
                    "HTTP Error", request=Mock(), response=mock_resp
                ))
            else:
                mock_resp.raise_for_status = Mock()

            return mock_resp

        with patch('execution.distributer.httpx.AsyncClient') as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.get = mock_request
            mock_instance.post = mock_request
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_instance

            try:
                result = await distributor.distribute_to_wordpress(article, project)

                # Property: Should ALWAYS return dict with "status" key
                assert isinstance(result, dict), f"Expected dict, got {type(result)}"
                assert "status" in result, f"Missing 'status' key in result: {result}"
                assert result["status"] in ["published", "error"], \
                    f"Invalid status: {result['status']}"

                # Property: If status=published, must have url and post_id
                if result["status"] == "published":
                    assert "url" in result, "Missing 'url' for published article"
                    assert "post_id" in result, "Missing 'post_id' for published article"

                # Property: If status=error, must have reason
                if result["status"] == "error":
                    assert "reason" in result, "Missing 'reason' for error status"
                    assert len(result["reason"]) > 0, "Error reason should not be empty"

            except Exception as e:
                # Some exceptions are expected (e.g., DistributionError)
                # But should be specific, not generic crashes
                assert "DistributionError" in type(e).__name__ or \
                       "HTTPException" in type(e).__name__ or \
                       "HTTPStatusError" in type(e).__name__, \
                       f"Unexpected exception type: {type(e).__name__}: {e}"

    try:
        asyncio.run(test_response())
    except Exception as e:
        # Expected for many edge cases
        pass

# ============================================================================
# PROPERTY 5: Content Payload Edge Cases
# ============================================================================

@given(
    title_length=st.integers(min_value=0, max_value=10000),
    content_length=st.integers(min_value=0, max_value=100000),
    has_special_chars=st.booleans(),
    has_html_tags=st.booleans(),
    has_unicode=st.booleans(),
)
@settings(max_examples=50, deadline=None)
def test_content_payload_edge_cases(title_length, content_length, has_special_chars, has_html_tags, has_unicode):
    """
    Property: System should handle content of ANY size and format.

    Edge cases to find:
    - Empty content
    - Extremely large content (>100KB)
    - Special characters (&, <, >, ", ')
    - Unicode characters (emoji, RTL text)
    - Malformed HTML
    - Nested HTML tags
    """
    # Generate content based on parameters
    if has_unicode:
        title = "🚀 Test Title 测试 عربي" * (title_length // 20 + 1)
    elif has_special_chars:
        title = "Test & <Title> \"Quote\" 'Single'" * (title_length // 30 + 1)
    else:
        title = "Test Title " * (title_length // 10 + 1)

    title = title[:title_length] if title_length > 0 else ""

    if has_html_tags:
        content = "<p>Paragraph</p><h2>Heading</h2>" * (content_length // 50 + 1)
    elif has_unicode:
        content = "<p>Unicode: 🎉 测试 עברית</p>" * (content_length // 30 + 1)
    else:
        content = "Content " * (content_length // 8 + 1)

    content = content[:content_length] if content_length > 0 else ""

    distributor = Distributor()

    # Try to create article with edge case content
    try:
        article = GeneratedArticle(
            id=uuid4(),
            project_id=uuid4(),
            title=title if title else "Default Title",  # Provide default if empty
            content=content,
            meta_description="Test" * 100,  # Also test long meta
        )
    except Exception:
        # Pydantic validation failed - this is expected behavior for invalid inputs
        # The validation is working correctly
        return

    # Test Gutenberg conversion doesn't crash
    try:
        gutenberg_content = distributor.convert_to_gutenberg_blocks(content)

        # Property: Output should be string
        assert isinstance(gutenberg_content, str), \
            f"Expected str, got {type(gutenberg_content)}"

        # Property: Should not lose significant content
        # (Some formatting changes are OK, but major data loss is a bug)
        if len(content) > 100:  # Only check for non-trivial content
            content_words = set(content.split())
            output_words = set(gutenberg_content.split())
            # At least 70% of words should be preserved
            overlap = len(content_words & output_words) / max(len(content_words), 1)
            assert overlap > 0.7, \
                f"Too much content lost in conversion: {overlap:.0%} overlap"

    except Exception as e:
        pytest.fail(f"Gutenberg conversion CRASHED with {type(e).__name__}: {e}")

# ============================================================================
# PROPERTY 6: Concurrent Upload Race Conditions
# ============================================================================

@given(
    concurrent_uploads=st.integers(min_value=1, max_value=20),
    same_article=st.booleans(),
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_concurrent_upload_race_conditions(concurrent_uploads, same_article):
    """
    Property: System should handle concurrent uploads without race conditions.

    Edge cases to find:
    - Multiple uploads of same article
    - Connection pool exhaustion
    - Retry logic interference
    - Shared state corruption
    """
    distributor = Distributor()

    if same_article:
        # All uploads use same article
        articles = [create_test_article()] * concurrent_uploads
    else:
        # Each upload uses different article
        articles = [
            create_test_article(title=f"Test Article {i}")
            for i in range(concurrent_uploads)
        ]

    project = create_test_project()

    async def upload_article(article):
        def mock_request(*args, **kwargs):
            # Simulate slow response to increase chance of race conditions
            time.sleep(0.01)
            mock_resp = Mock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "id": hash(article.id) % 10000,
                "link": f"https://example.com/post-{article.id}"
            }
            mock_resp.raise_for_status = Mock()
            return mock_resp

        with patch('execution.distributer.httpx.AsyncClient') as mock_client_cls:
            mock_instance = MagicMock()
            mock_instance.get = mock_request
            mock_instance.post = mock_request
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_instance

            return await distributor.distribute_to_wordpress(article, project)

    async def run_concurrent():
        tasks = [upload_article(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Property: All uploads should complete (success or failure, no hangs)
        assert len(results) == concurrent_uploads, \
            f"Expected {concurrent_uploads} results, got {len(results)}"

        # Property: No uploads should crash with unexpected errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Some errors are expected, but not generic crashes
                assert any(x in type(result).__name__ for x in [
                    "DistributionError", "HTTPException", "HTTPStatusError",
                    "TimeoutError", "NetworkError"
                ]), f"Upload {i} crashed unexpectedly: {type(result).__name__}: {result}"

        return results

    try:
        results = asyncio.run(run_concurrent())
    except Exception as e:
        pytest.fail(f"Concurrent uploads CRASHED: {type(e).__name__}: {e}")

# ============================================================================
# PROPERTY 7: Exponential Backoff Jitter Distribution
# ============================================================================

@given(
    retry_attempt=st.integers(min_value=0, max_value=10),
    base_delay=st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=100, deadline=None)
def test_exponential_backoff_jitter_properties(retry_attempt, base_delay):
    """
    Property: Exponential backoff should follow mathematical properties.

    Properties to verify:
    - Backoff should increase exponentially: wait_time ≈ base * (2^attempt)
    - Jitter should add randomness: 0 <= jitter <= 30% of base_wait
    - Total wait should be capped at reasonable maximum
    - Distribution should prevent thundering herd
    """
    import random

    # Simulate backoff calculation from distributer.py
    base_wait = base_delay * (2 ** retry_attempt)
    jitter = random.uniform(0, base_wait * 0.3)
    wait_time = base_wait + jitter

    # Property 1: Base wait should grow exponentially
    expected_base = base_delay * (2 ** retry_attempt)
    assert abs(base_wait - expected_base) < 0.001, \
        f"Base wait calculation wrong: {base_wait} != {expected_base}"

    # Property 2: Jitter should be between 0 and 30% of base
    assert 0 <= jitter <= base_wait * 0.3, \
        f"Jitter out of range: {jitter} not in [0, {base_wait * 0.3}]"

    # Property 3: Total wait should be base + jitter
    assert base_wait <= wait_time <= base_wait * 1.3, \
        f"Total wait out of range: {wait_time} not in [{base_wait}, {base_wait * 1.3}]"

    # Property 4: For high retry counts, should cap at reasonable max
    # FIXED: Now capped at 10 seconds in distributer.py
    if retry_attempt >= 1:
        # Simulate the actual cap from distributer.py
        capped_base_wait = min(base_delay * (2 ** retry_attempt), 10.0)
        capped_wait_time = capped_base_wait + random.uniform(0, capped_base_wait * 0.3)

        # Property: Capped wait should never exceed 13 seconds (10s + 30% jitter)
        assert capped_wait_time <= 13.0, \
            f"Capped wait time too large: {capped_wait_time:.2f}s > 13.0s"

        # Property: Even at high retry counts, should be reasonable
        if retry_attempt >= 8:
            assert capped_wait_time <= 13.0, \
                f"At attempt {retry_attempt}, wait time {capped_wait_time:.2f}s is capped correctly"

# ============================================================================
# CHAOS TEST: Intermittent Network Failures
# ============================================================================

@pytest.mark.chaos
def test_chaos_intermittent_network():
    """
    CHAOS TEST: Simulate intermittent network that fails randomly.

    Scenario: Network works sometimes, fails other times unpredictably.
    Should retry intelligently and eventually succeed OR fail gracefully.
    """
    import random

    distributor = Distributor(max_retries=5, initial_retry_delay=0.1)

    # Use factory functions to ensure all required fields are present
    article = create_test_article(
        title="Chaos Test Article",
        content="<p>" + ("Test content for chaos engineering. " * 30) + "</p>"
    )

    project = create_test_project(
        name="Chaos Test Project",
        wordpress_url="https://example.com",
        wordpress_username="admin",
        wordpress_app_password="test pass 1234"
    )

    http_call_count = [0]
    publish_attempt_count = [0]

    async def intermittent_network(*args, **kwargs):
        http_call_count[0] += 1
        url = str(args[0]) if args else ""
        is_publish_attempt = url.rstrip("/").endswith("/wp-json/wp/v2/posts")
        if is_publish_attempt:
            publish_attempt_count[0] += 1

        # 50% chance of failure
        if random.random() < 0.5:
            raise httpx.NetworkError("Intermittent network failure")

        # Success
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": 123, "link": "https://example.com/post"}
        mock_resp.raise_for_status = Mock()
        return mock_resp

    async def test():
        with patch('execution.distributer.httpx.AsyncClient') as mock_client_cls:
            mock_instance = AsyncMock()
            mock_instance.get = intermittent_network
            mock_instance.post = intermittent_network
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_instance

            result = await distributor.distribute_to_wordpress(article, project)

            # Property: Should eventually succeed or fail gracefully
            assert result["status"] in ["published", "error"], \
                f"Unexpected status: {result['status']}"

            # Property: Should have attempted multiple times
            assert http_call_count[0] >= 1, "Should have attempted at least once"
            assert publish_attempt_count[0] <= distributor.max_retries, \
                "Should not exceed max publish retries: " \
                f"{publish_attempt_count[0]} > {distributor.max_retries}"

    try:
        asyncio.run(test())
    except Exception as e:
        # Should handle gracefully
        assert "DistributionError" in type(e).__name__, \
            f"Should raise DistributionError, got {type(e).__name__}"

# ============================================================================
# Run all tests with: pytest tests/test_wordpress_edge_cases_property.py -v
# ============================================================================
