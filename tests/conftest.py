"""
Pytest Configuration and Fixture Library

Comprehensive test infrastructure providing:
- Database isolation with transaction rollback
- Mock service factories
- Reusable test data generators
- Property-based testing utilities
- Async test support
- Performance benchmarking fixtures

Design Pattern: Test Data Builder + Fixture Factory
"""

import asyncio
import os
import random
import string
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, Generator, List
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio
import redis as redis_client
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Set test environment variables before importing any modules
os.environ.update(
    {
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_DATABASE": "test_db",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_DB": "15",
        "SECRET_KEY": "test-secret-key",
        "LLM_ANTHROPIC_API_KEY": "test-key",
        "LLM_OPENAI_API_KEY": "test-key",
    }
)

from core.models import (
    ContentPlan,
    GeneratedArticle,
    InferredPatterns,
    Keyword,
    Outline,
    Project,
    Rule,
    Rulebook,
    Section,
)
from infrastructure.database import DatabaseManager
from infrastructure.redis_client import RedisClient
from intelligence.semantic_analyzer import SemanticAnalyzer

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring external services"
    )
    config.addinivalue_line("markers", "slow: mark test as slow-running (>1s)")
    config.addinivalue_line("markers", "unit: mark test as unit test (no external dependencies)")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_metrics_collector():
    """Reset MetricsCollector singleton before each test."""
    from infrastructure.monitoring import MetricsCollector

    MetricsCollector.reset_singleton()
    yield
    MetricsCollector.reset_singleton()


@pytest.fixture(autouse=True)
async def clear_cache():
    """Clear cache before each test."""
    from optimization.cache_manager import cache_manager

    # Clear memory cache
    cache_manager._memory_cache.clear()
    cache_manager._warm_keys.clear()
    # Reset stats
    cache_manager.reset_statistics()
    yield
    # Clear again after test
    cache_manager._memory_cache.clear()
    cache_manager._warm_keys.clear()
    cache_manager.reset_statistics()


# ============================================================================
# DATABASE FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[DatabaseManager, None]:
    """
    Database fixture with mocked database manager.

    Uses a mock database manager to avoid requiring actual database connection.
    """
    from unittest.mock import AsyncMock, Mock

    # Create a mock database manager
    mock_db = Mock(spec=DatabaseManager)
    mock_db.execute = AsyncMock()
    mock_db.execute_raw = AsyncMock()
    mock_db.session = Mock()
    mock_db.session.return_value.__aenter__ = AsyncMock()
    mock_db.session.return_value.__aexit__ = AsyncMock()
    mock_db.disconnect = AsyncMock()

    yield mock_db


@pytest_asyncio.fixture
async def clean_db() -> AsyncGenerator[DatabaseManager, None]:
    """
    Clean database fixture with mocked database manager.

    Uses a mock database manager to avoid requiring actual database connection.
    """
    from unittest.mock import AsyncMock, Mock

    # Create a mock database manager
    mock_db = Mock(spec=DatabaseManager)
    mock_db.execute = AsyncMock()
    mock_db.execute_raw = AsyncMock()
    mock_db.session = Mock()
    mock_db.session.return_value.__aenter__ = AsyncMock()
    mock_db.session.return_value.__aexit__ = AsyncMock()
    mock_db.disconnect = AsyncMock()

    yield mock_db


# ============================================================================
# REDIS FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def redis() -> AsyncGenerator[RedisClient, None]:
    """
    Redis fixture with mocked Redis client.

    Uses a mock Redis client to avoid requiring actual Redis connection.
    """
    from unittest.mock import AsyncMock, Mock

    # Create a mock Redis client
    mock_redis = Mock(spec=RedisClient)
    mock_redis.connect = AsyncMock()
    mock_redis.disconnect = AsyncMock()
    mock_redis.flushdb = AsyncMock()
    mock_redis.get = AsyncMock()
    mock_redis.set = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.incr = AsyncMock()
    mock_redis.expire = AsyncMock()
    mock_redis.exists = AsyncMock()

    yield mock_redis


# ============================================================================
# API CLIENT FIXTURES
# ============================================================================


@pytest.fixture
def api_client():
    """
    FastAPI test client fixture.

    Provides a test client for making HTTP requests to the API
    without running an actual server.
    """
    from api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_client_with_mocked_auth(monkeypatch):
    """
    FastAPI test client with mocked auth dependencies.

    Patches container initialization at the source to prevent real DB connections.
    """
    # Patch container before app loads
    from datetime import datetime
    from unittest.mock import AsyncMock
    from uuid import uuid4

    from security import UserInDB, get_password_hash, verify_password
    from services.user_service import UserService

    # Create mock user service
    mock_user_service = AsyncMock(spec=UserService)
    created_users = {}

    async def mock_create_user(user_create):
        user_id = uuid4()
        password_hash = get_password_hash(user_create.password)
        user = UserInDB(
            id=user_id,
            username=user_create.email.split("@")[0],
            email=user_create.email,
            full_name=user_create.full_name,
            hashed_password=password_hash,
            is_active=True,
            is_superuser=False,
            created_at=datetime.utcnow(),
        )
        created_users[user.email] = user
        return user

    async def mock_authenticate_user(username, password):
        user = created_users.get(username)
        if user and verify_password(password, user.hashed_password):
            return user
        return None

    async def mock_get_user_by_email(email):
        return created_users.get(email)

    mock_user_service.create_user = AsyncMock(side_effect=mock_create_user)
    mock_user_service.authenticate_user = AsyncMock(side_effect=mock_authenticate_user)
    mock_user_service.get_user_by_email = AsyncMock(side_effect=mock_get_user_by_email)

    # Monkeypatch container methods to prevent real DB initialization
    import sys

    import container

    # Override get_user_service to return our mock
    def patched_get_user_service():
        return mock_user_service

    monkeypatch.setattr(container, "get_user_service", patched_get_user_service)

    # Reload api.main to apply the patch
    if "api.main" in sys.modules:
        from importlib import reload

        import api.main

        reload(api.main)
        from api.main import app
    else:
        from api.main import app

    with TestClient(app) as client:
        yield client, created_users


# ============================================================================
# MOCK SERVICE FIXTURES
# ============================================================================


@pytest.fixture
def mock_llm_client():
    """
    Mock LLM client with configurable responses.

    Provides realistic mock responses without actual API calls.
    """
    mock = AsyncMock()

    # Default response
    mock.complete.return_value = Mock(
        content="This is a generated article section with relevant content.",
        usage=Mock(prompt_tokens=100, completion_tokens=200),
        cost=0.015,
    )

    return mock


@pytest.fixture
def mock_semantic_analyzer():
    """Mock semantic analyzer with deterministic embeddings."""
    mock = Mock(spec=SemanticAnalyzer)

    # Generate consistent embeddings
    def generate_embedding(text: str):
        import numpy as np

        # Hash text to seed for reproducibility
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return np.random.rand(384).astype(np.float32)

    mock.generate_embedding = Mock(side_effect=generate_embedding)
    mock.compute_similarity = Mock(return_value=0.85)
    mock.compute_coherence = AsyncMock(return_value=0.78)

    return mock


@pytest.fixture
def mock_website_analyzer():
    """Mock website analyzer for testing without actual scraping."""
    mock = AsyncMock()

    mock.analyze_website.return_value = InferredPatterns(
        id=uuid4(),
        project_id=uuid4(),
        avg_sentence_length=(18.5, 2.3),
        lexical_diversity=0.72,
        readability_score=65.0,
        tone_embedding=[0.1] * 384,
        confidence=0.85,
        sample_size=15,
        analyzed_at=datetime.utcnow(),
    )

    return mock


# ============================================================================
# TEST DATA FACTORIES
# ============================================================================


class ProjectFactory:
    """Factory for creating test Project instances."""

    @staticmethod
    def create(
        name: str = None, domain: str = None, telegram_channel: str = None, **kwargs
    ) -> Project:
        """Create project with sensible defaults."""
        return Project(
            id=uuid4(),
            name=name or f"Test Project {random.randint(1000, 9999)}",
            domain=domain or f"https://test{random.randint(100, 999)}.com",
            telegram_channel=telegram_channel,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            total_articles_generated=0,
            **kwargs,
        )

    @staticmethod
    def create_batch(count: int) -> List[Project]:
        """Create multiple projects."""
        return [ProjectFactory.create() for _ in range(count)]


class RulebookFactory:
    """Factory for creating test Rulebook instances."""

    @staticmethod
    def create(
        project_id: uuid4 = None, content: str = None, rule_count: int = 5, **kwargs
    ) -> Rulebook:
        """Create rulebook with rules."""
        rules = [
            Rule(
                id=uuid4(),
                rule_type=random.choice(["tone", "structure", "style"]),
                content=f"Test rule {i}: {_random_string(50)}",
                embedding=[random.random() for _ in range(384)],
                priority=random.randint(1, 10),
                context="Test context",
            )
            for i in range(rule_count)
        ]

        return Rulebook(
            id=uuid4(),
            project_id=project_id or uuid4(),
            rules=rules,
            raw_content=content or _random_string(500),
            version=1,
            updated_at=datetime.utcnow(),
            **kwargs,
        )


class ContentPlanFactory:
    """Factory for creating test ContentPlan instances."""

    @staticmethod
    def create(
        project_id: uuid4 = None, topic: str = None, section_count: int = 5, **kwargs
    ) -> ContentPlan:
        """Create content plan with outline."""
        sections = [
            Section(
                heading=f"Section {i+1}: {_random_string(30)}",
                theme_embedding=[random.random() for _ in range(384)],
                target_keywords=[f"keyword{i}", f"term{i}"],
                estimated_words=random.randint(200, 400),
                intent="explain",
            )
            for i in range(section_count)
        ]

        outline = Outline(
            title=f"Test Article: {topic or _random_string(50)}",
            meta_description=_random_string(150),
            sections=sections,
        )

        keywords = [
            Keyword(
                phrase=f"keyword_{i}",
                search_volume=random.randint(100, 10000),
                difficulty=random.random(),
                intent="informational",
                embedding=[random.random() for _ in range(384)],
                related_concepts=[f"concept{i}", f"topic{i}"],
            )
            for i in range(10)
        ]

        return ContentPlan(
            id=uuid4(),
            project_id=project_id or uuid4(),
            topic=topic or _random_string(100),
            primary_keywords=keywords[:5],
            secondary_keywords=keywords[5:],
            outline=outline,
            target_word_count=section_count * 300,
            readability_target="10-12 grade",
            estimated_cost=random.uniform(0.1, 0.5),
            created_at=datetime.utcnow(),
            **kwargs,
        )


class ArticleFactory:
    """Factory for creating test GeneratedArticle instances."""

    @staticmethod
    def create(
        project_id: uuid4 = None, content_plan_id: uuid4 = None, word_count: int = None, **kwargs
    ) -> GeneratedArticle:
        """Create generated article with realistic data."""
        wc = word_count or random.randint(800, 2500)

        return GeneratedArticle(
            id=uuid4(),
            project_id=project_id or uuid4(),
            content_plan_id=content_plan_id or uuid4(),
            title=f"Test Article: {_random_string(60)}",
            content=_generate_article_content(wc),
            meta_description=_random_string(150),
            word_count=wc,
            readability_score=random.uniform(60.0, 80.0),
            keyword_density={"keyword1": 0.015, "keyword2": 0.008},
            total_tokens_used=int(wc * 1.3),
            total_cost=random.uniform(0.10, 0.30),
            generation_time=random.uniform(45.0, 180.0),
            distributed_at=None,
            created_at=datetime.utcnow(),
            **kwargs,
        )


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def project_factory():
    """Provide ProjectFactory to tests."""
    return ProjectFactory


@pytest.fixture
def rulebook_factory():
    """Provide RulebookFactory to tests."""
    return RulebookFactory


@pytest.fixture
def content_plan_factory():
    """Provide ContentPlanFactory to tests."""
    return ContentPlanFactory


@pytest.fixture
def article_factory():
    """Provide ArticleFactory to tests."""
    return ArticleFactory


@pytest_asyncio.fixture
async def sample_project(db: DatabaseManager) -> Project:
    """Create and persist a sample project."""
    project = ProjectFactory.create()

    await db.execute_raw(
        """
        INSERT INTO projects (id, name, domain, telegram_channel, created_at, last_active, total_articles_generated)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        project.id,
        project.name,
        project.domain,
        project.telegram_channel,
        project.created_at,
        project.last_active,
        project.total_articles_generated,
    )

    return project


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _random_string(length: int) -> str:
    """Generate random string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits + " ", k=length))


def _generate_article_content(word_count: int) -> str:
    """Generate realistic article content with specified word count."""
    words = []
    lorem = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua".split()

    while len(words) < word_count:
        words.extend(random.sample(lorem, min(len(lorem), word_count - len(words))))

    # Add structure
    content = f"# {_random_string(50)}\n\n"

    paragraphs = []
    for i in range(0, len(words), 50):
        chunk = " ".join(words[i : i + 50])
        paragraphs.append(chunk.capitalize() + ".")

    for i, para in enumerate(paragraphs):
        if i % 3 == 0:
            content += f"\n\n## {_random_string(30)}\n\n"
        content += para + "\n\n"

    return content


@pytest.fixture
def benchmark_timer():
    """Fixture for performance benchmarking."""
    import time

    class BenchmarkTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.end_time = time.time()

        @property
        def elapsed(self) -> float:
            return self.end_time - self.start_time if self.end_time else 0

    return BenchmarkTimer
