"""FastAPI Dependency Injection - Pragmatic Factory Functions.

Provides simple, explicit factory functions for dependency injection,
replacing heavy DI containers with Pythonic @lru_cache decorators.

Philosophy:
    Explicit over implicit. No magic, no auto-wiring, just functions.

Architecture:
    - Infrastructure layer: Singletons with @lru_cache
    - Service layer: Factory functions creating new instances
    - FastAPI integration: Use with Depends() for request scoping

Example:
    >>> from fastapi import Depends
    >>> from api.dependencies import get_llm_client
    >>>
    >>> @router.post("/generate")
    >>> async def generate(client: UnifiedLLMClient = Depends(get_llm_client)):
    >>>     return await client.generate(...)
"""

import os

# ============================================================================
# INFRASTRUCTURE LAYER (Singletons)
# ============================================================================
# Per-process registries for components that need event loop binding
# These avoid creating new instances on every call while still being safe for Celery workers
import threading
from functools import lru_cache
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import Settings, get_settings
from execution.content_generator import ContentGenerator
from execution.content_planner import ContentPlanner
from execution.keyword_researcher import KeywordResearcher
from infrastructure.database import DatabaseManager
from infrastructure.llm_client import UnifiedLLMClient
from infrastructure.llm_usage import LLMUsageService
from infrastructure.monitoring import MetricsCollector
from infrastructure.redis_client import RedisClient
from intelligence.best_practices_kb import BestPracticesKB
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.decision_engine import DecisionEngine
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.article_repository import ArticleRepository
from knowledge.pattern_extractor import PatternExtractor
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager
from knowledge.user_repository import UserRepository
from knowledge.website_analyzer import WebsiteAnalyzer
from optimization.cache_manager import CacheManager
from optimization.model_router import ModelRouter
from optimization.prompt_compressor import PromptCompressor
from optimization.token_budget_manager import BudgetConfig, TokenBudgetManager
from orchestration.content_agent import ContentAgent, ContentAgentConfig
from orchestration.task_persistence import TaskResultRepository
from services.content_memory_service import ContentMemoryService
from services.content_service import ContentService
from services.project_readiness_service import ProjectReadinessService
from services.project_service import ProjectService
from services.user_service import UserService

_redis_registry: dict[int, RedisClient] = {}
_redis_lock = threading.Lock()


@lru_cache
def get_settings_cached() -> Settings:
    """Retrieve cached application settings singleton.

    Returns:
        Settings instance with all configuration loaded.
    """
    return get_settings()


@lru_cache
def get_database() -> DatabaseManager:
    """Retrieve singleton database manager with connection pooling.

    Returns:
        DatabaseManager instance for all database operations.
    """
    from infrastructure.database import get_db_manager

    return get_db_manager()


def get_redis() -> RedisClient:
    """Get Redis client with per-process caching.

    Uses process ID as key to ensure Celery workers get their own instance
    while avoiding creating new connection pools on every call.

    Returns:
        RedisClient instance for the current process.
    """
    import os

    pid = os.getpid()

    with _redis_lock:
        if pid not in _redis_registry:
            _redis_registry[pid] = RedisClient()
        return _redis_registry[pid]


def clear_redis_registry():
    """Clear Redis registry (call on worker shutdown)."""
    with _redis_lock:
        _redis_registry.clear()


@lru_cache
def get_metrics() -> MetricsCollector:
    """Get singleton metrics collector."""
    return MetricsCollector()


def get_cache_manager() -> CacheManager:
    """Get cache manager.

    Note: Uses per-process Redis client for event loop safety.
    """
    return CacheManager(
        redis_client=get_redis(),
        max_memory_entries=1000,
        metrics_collector=get_metrics(),
    )


@lru_cache
def get_llm_client() -> UnifiedLLMClient:
    """Retrieve singleton UnifiedLLMClient.

    IMPORTANT: Must be cached (@lru_cache) for Circuit Breakers to persist
    across requests. The internal locks are lazy-loaded to ensure event
    loop safety, so global instantiation is safe.

    Returns:
        UnifiedLLMClient configured with caching and metrics.
    """
    return UnifiedLLMClient(
        cache_manager=get_cache_manager(),
        metrics_collector=get_metrics(),
        usage_service=LLMUsageService(get_database()),
    )


# ============================================================================
# KNOWLEDGE LAYER (Factories)
# ============================================================================


def get_project_repository() -> ProjectRepository:
    """Create new ProjectRepository instance.

    Returns:
        ProjectRepository for project data access operations.
    """
    return ProjectRepository(database_manager=get_database())


def get_article_repository() -> ArticleRepository:
    """Create ArticleRepository instance."""
    return ArticleRepository(db_manager=get_database())


def get_user_repository() -> UserRepository:
    """Create UserRepository instance."""
    return UserRepository(database_manager=get_database())


def get_pattern_extractor() -> PatternExtractor:
    """Create PatternExtractor instance."""
    return PatternExtractor(semantic_analyzer=get_semantic_analyzer())


def get_website_analyzer() -> WebsiteAnalyzer:
    """Create WebsiteAnalyzer instance."""
    settings = get_settings_cached()
    return WebsiteAnalyzer(
        pattern_extractor=get_pattern_extractor(),
        project_repository=get_project_repository(),
        scraping_settings=settings.scraping,
    )


# ============================================================================
# INTELLIGENCE LAYER (Factories)
# ============================================================================

# Track SemanticAnalyzer loading attempts for retry logic
_semantic_analyzer_instance: Optional[SemanticAnalyzer] = None
_semantic_analyzer_attempted: bool = False


def get_semantic_analyzer() -> Optional[SemanticAnalyzer]:
    """Get SemanticAnalyzer with retry capability.

    Unlike @lru_cache, this allows retry if first attempt failed.
    The model loads lazily on first embed() call.

    Returns:
        SemanticAnalyzer instance, or None if creation failed.
    """
    global _semantic_analyzer_instance, _semantic_analyzer_attempted

    # Return cached instance if available
    if _semantic_analyzer_instance is not None:
        return _semantic_analyzer_instance

    # Retry if previous attempt failed (allows recovery after transient issues)
    try:
        _semantic_analyzer_instance = SemanticAnalyzer(redis_client=get_redis())
        _semantic_analyzer_attempted = True
        return _semantic_analyzer_instance
    except Exception as e:
        import logging

        logging.error(f"Failed to create SemanticAnalyzer: {e}")
        _semantic_analyzer_attempted = True
        return None


def reset_semantic_analyzer():
    """Reset SemanticAnalyzer to allow retry (call after fixing underlying issue)."""
    global _semantic_analyzer_instance, _semantic_analyzer_attempted
    _semantic_analyzer_instance = None
    _semantic_analyzer_attempted = False


def get_best_practices_kb() -> BestPracticesKB:
    """Create BestPracticesKB instance."""
    return BestPracticesKB(
        redis_client=get_redis(),
        semantic_analyzer=get_semantic_analyzer(),
    )


# REMOVED: get_rulebook_manager() - RulebookManager requires active AsyncSession
# Instantiate directly within session context: RulebookManager(session, semantic_analyzer)


def get_decision_engine() -> DecisionEngine:
    """Create DecisionEngine instance."""
    return DecisionEngine(
        database_manager=get_database(),
        best_practices=get_best_practices_kb(),
        semantic_analyzer=get_semantic_analyzer(),
    )


def get_context_synthesizer() -> ContextSynthesizer:
    """Create ContextSynthesizer instance."""
    return ContextSynthesizer(semantic_analyzer=get_semantic_analyzer())


# ============================================================================
# OPTIMIZATION LAYER (Factories)
# ============================================================================


def get_token_budget_manager() -> TokenBudgetManager:
    """Create TokenBudgetManager instance."""
    return TokenBudgetManager(config=BudgetConfig())


def get_prompt_compressor() -> PromptCompressor:
    """Create PromptCompressor instance."""
    return PromptCompressor()


# ============================================================================
# EXECUTION LAYER (Factories)
# ============================================================================


def get_keyword_researcher() -> KeywordResearcher:
    """Create KeywordResearcher instance."""
    return KeywordResearcher(
        semantic_analyzer=get_semantic_analyzer(),
        cache_manager=get_cache_manager(),
        llm_client=get_llm_client(),
    )


def get_model_router() -> ModelRouter:
    """Create ModelRouter instance."""
    return ModelRouter()


def get_content_planner() -> ContentPlanner:
    """Create ContentPlanner instance."""
    return ContentPlanner(
        decision_engine=get_decision_engine(),
        context_synthesizer=get_context_synthesizer(),
        model_router=get_model_router(),
        llm_client=get_llm_client(),
        article_repository=get_article_repository(),
        metrics_collector=get_metrics(),
    )


def get_content_generator() -> ContentGenerator:
    """Create ContentGenerator instance."""
    return ContentGenerator(
        llm_client=get_llm_client(),
        context_synthesizer=get_context_synthesizer(),
        semantic_analyzer=get_semantic_analyzer(),
        token_budget_manager=get_token_budget_manager(),
        article_repository=get_article_repository(),
        metrics_collector=get_metrics(),
    )


# ============================================================================
# ORCHESTRATION LAYER (Factories)
# ============================================================================


def get_content_agent() -> ContentAgent:
    """Create ContentAgent instance."""
    settings = get_settings_cached()
    config = ContentAgentConfig(
        max_retries=3,
        timeout_seconds=300,
        enable_caching=True,
    )
    return ContentAgent(
        database_manager=get_database(),
        semantic_analyzer=get_semantic_analyzer(),
        keyword_researcher=get_keyword_researcher(),
        content_planner=get_content_planner(),
        content_generator=get_content_generator(),
        project_repository=get_project_repository(),
        website_analyzer=get_website_analyzer(),
        redis_client=get_redis(),
        metrics_collector=get_metrics(),
        config=config,
    )


def get_task_result_repository() -> TaskResultRepository:
    """Create TaskResultRepository instance."""
    return TaskResultRepository(db_manager=get_database())


# ============================================================================
# SERVICE LAYER (Factories)
# ============================================================================


def get_content_service() -> ContentService:
    """Create ContentService instance."""
    return ContentService(
        content_agent=get_content_agent(),
        article_repository=get_article_repository(),
        task_repository=get_task_result_repository(),
        metrics_collector=get_metrics(),
        semantic_analyzer=get_semantic_analyzer(),
    )


def get_content_memory_service() -> ContentMemoryService:
    """Create ContentMemoryService instance."""
    return ContentMemoryService(article_repository=get_article_repository())


def get_project_service() -> ProjectService:
    """Create ProjectService instance."""
    return ProjectService(
        database_manager=get_database(),
        semantic_analyzer=get_semantic_analyzer(),
        project_repository=get_project_repository(),
    )


def get_project_readiness_service() -> ProjectReadinessService:
    """Create ProjectReadinessService instance."""
    return ProjectReadinessService(
        database=get_database(),
        redis=get_redis(),
        settings=get_settings_cached(),
        semantic_analyzer=get_semantic_analyzer(),
        project_repository=get_project_repository(),
    )


def get_user_service() -> UserService:
    """Create UserService instance."""
    return UserService(user_repository=get_user_repository())


# ============================================================================
# DATABASE SESSION HELPER
# ============================================================================


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a database session for FastAPI endpoints.

    Usage:
        @router.get("/")
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    db = get_database()
    async with db.session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
