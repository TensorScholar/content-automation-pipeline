"""
Integration Test Fixtures
==========================

Shared fixtures for integration tests providing:
- Test database setup and teardown
- Redis mock/real connection
- API client configuration
- Authentication helpers
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

# Set test environment before importing app modules
# Settings only accepts runtime environments that the application supports.
# Test isolation is provided by the remaining test-specific variables below.
os.environ["ENVIRONMENT"] = "development"
os.environ["PYTEST_CURRENT_TEST"] = "true"


@pytest.fixture
def mock_database():
    """Mock database manager for tests not requiring real DB."""
    db_mock = MagicMock()
    db_mock.initialize = AsyncMock()
    db_mock.close = AsyncMock()
    db_mock.health_check = AsyncMock(return_value=True)
    db_mock.session = MagicMock()
    db_mock.fetch_one = AsyncMock(return_value=None)
    db_mock.fetch_all = AsyncMock(return_value=[])
    db_mock.execute = AsyncMock()
    return db_mock


@pytest.fixture
def mock_redis():
    """Mock Redis client for tests not requiring real Redis."""
    redis_mock = MagicMock()
    redis_mock.initialize = AsyncMock()
    redis_mock.close = AsyncMock()
    redis_mock.health_check = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=True)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.get_cached_response = AsyncMock(return_value=None)
    redis_mock.cache_response = AsyncMock(return_value=True)
    return redis_mock


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for tests not requiring real API calls."""
    llm_mock = MagicMock()
    llm_mock.generate = AsyncMock(
        return_value=MagicMock(
            content="Generated test content",
            model="claude-3-haiku-20240307",
            usage=MagicMock(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
            ),
            cost=0.001,
        )
    )
    llm_mock.get_provider_status = MagicMock(return_value={"anthropic": True, "openai": False})
    return llm_mock


@pytest.fixture
def sample_project_data():
    """Sample project data for testing."""
    return {
        "id": str(uuid4()),
        "name": "Test Project",
        "description": "A test project for integration testing",
        "domain": "test.example.com",
        "target_audience": "developers",
        "tone": "professional",
        "primary_keywords": ["testing", "automation"],
        "user_id": str(uuid4()),
    }


@pytest.fixture
def sample_article_data():
    """Sample article data for testing."""
    return {
        "id": str(uuid4()),
        "title": "Test Article Title",
        "content": "<h1>Test Article</h1><p>This is test content.</p>",
        "word_count": 100,
        "status": "completed",
        "keywords": ["test", "article"],
        "meta_description": "A test article for integration testing",
    }


@pytest.fixture
def auth_headers():
    """Generate mock authentication headers."""
    # In real tests, this would create a valid JWT token
    return {"Authorization": "Bearer test-token-for-integration-tests"}


@pytest_asyncio.fixture
async def test_client():
    """
    Create async test client for API integration tests.

    Note: For tests requiring full FastAPI client,
    use httpx.AsyncClient with the app.
    """
    from httpx import AsyncClient

    from api.main import create_app

    app = create_app()

    # Disable rate limiting for tests
    async with AsyncClient(app=app, base_url="http://localhost") as client:
        yield client


@pytest.fixture
def mock_celery_task():
    """Mock Celery task for testing task dispatch."""
    task_mock = MagicMock()
    task_mock.delay = MagicMock(return_value=MagicMock(id=str(uuid4()), status="PENDING"))
    task_mock.apply_async = MagicMock(return_value=MagicMock(id=str(uuid4()), status="PENDING"))
    return task_mock
