"""
Dependency Injection Container: Centralized Object Lifecycle Management

Implements a comprehensive dependency injection system using dependency-injector
to manage the entire application's object graph with proper lifecycle management,
type safety, and acyclic dependency resolution.

Architecture: Container Pattern + Dependency Injection + Singleton Registry
Mathematical Foundation: Directed Acyclic Graph (DAG) for dependency resolution
"""

import asyncio
from typing import Any, Dict, Optional, Type, TypeVar

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from loguru import logger

from config.settings import Settings, get_settings

# Type variables for generic container access
T = TypeVar("T")

# Execution layer imports
from execution.content_generator import ContentGenerator
from execution.content_planner import ContentPlanner

from execution.keyword_researcher import KeywordResearcher

# Infrastructure layer imports
from infrastructure.database import DatabaseManager
from infrastructure.llm_client import AbstractLLMClient, get_llm_client
from infrastructure.monitoring import MetricsCollector
from infrastructure.redis_client import RedisClient
from intelligence.best_practices_kb import BestPracticesKB

# Intelligence layer imports
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.decision_engine import DecisionEngine
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.article_repository import ArticleRepository
from knowledge.pattern_extractor import PatternExtractor

# Knowledge layer imports
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager
from knowledge.user_repository import UserRepository
from knowledge.website_analyzer import WebsiteAnalyzer

# Optimization layer imports
from optimization.cache_manager import CacheManager
from optimization.model_router import ModelRouter
from optimization.prompt_compressor import PromptCompressor
from optimization.token_budget_manager import BudgetConfig, TokenBudgetManager

# Orchestration layer imports
from orchestration.content_agent import ContentAgent, ContentAgentConfig

# Security imports
from security import (
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_current_active_user,
    get_current_superuser,
    get_current_user,
    get_password_hash,
    verify_password,
)
from services.content_service import ContentService

# Service layer imports
from services.project_service import ProjectService
from services.user_service import UserService



class Container(containers.DeclarativeContainer):
    """
    Central dependency injection container.

    Manages the entire application's object lifecycle with:
    - Singleton providers for infrastructure components
    - Factory providers for business logic components
    - Configuration providers for settings
    - Proper dependency resolution with type hints

    Dependency Graph (DAG):
    Settings -> Infrastructure -> Knowledge -> Intelligence -> Optimization -> Execution -> Orchestration -> Services
    """

    # Configuration providers (singletons)
    config: providers.Singleton[Settings] = providers.Singleton(get_settings)

    # Infrastructure layer providers (singletons)
    database: providers.Singleton[DatabaseManager] = providers.Singleton(DatabaseManager)

    redis: providers.Singleton[RedisClient] = providers.Singleton(RedisClient)

    metrics: providers.Singleton[MetricsCollector] = providers.Singleton(MetricsCollector)

    cache: providers.Singleton[CacheManager] = providers.Singleton(
        CacheManager,
        redis_client=redis,  # <--- ADD THIS
        max_memory_entries=1000,
        metrics_collector=metrics,
    )

    llm: providers.Singleton[AbstractLLMClient] = providers.Singleton(
        get_llm_client,
        redis_client=redis,
        cache_manager=cache,
        metrics_collector=metrics,
        settings=config,
    )

    # Alias for compatibility
    llm_client: providers.Singleton[AbstractLLMClient] = providers.Singleton(
        get_llm_client,
        redis_client=redis,
        cache_manager=cache,
        metrics_collector=metrics,
        settings=config,
    )

    # Knowledge layer providers (factories)
    project_repository: providers.Factory[ProjectRepository] = providers.Factory(
        ProjectRepository,
        database_manager=database,
    )

    article_repository: providers.Factory[ArticleRepository] = providers.Factory(
        ArticleRepository,
        db_manager=database,
    )

    # Intelligence layer providers (factories) - moved before rulebook_manager
    semantic_analyzer: providers.Factory[SemanticAnalyzer] = providers.Factory(
        SemanticAnalyzer,
        redis_client=redis,  # <--- ADD THIS
    )

    rulebook_manager: providers.Factory[RulebookManager] = providers.Factory(
        RulebookManager,
        session=database,
        embedding_model=semantic_analyzer,
    )

    pattern_extractor: providers.Factory[PatternExtractor] = providers.Factory(PatternExtractor)

    website_analyzer: providers.Factory[WebsiteAnalyzer] = providers.Factory(
        WebsiteAnalyzer,
        pattern_extractor=pattern_extractor,  # <--- Re-order for clarity
        project_repository=project_repository,  # <--- ADD THIS
        scraping_settings=config.provided.scraping,  # <--- ADD THIS
    )

    user_repository: providers.Factory[UserRepository] = providers.Factory(
        UserRepository,
        database_manager=database,
    )

    # Intelligence layer providers (factories)
    best_practices_kb: providers.Factory[BestPracticesKB] = providers.Factory(
        BestPracticesKB,
        redis_client=redis,  # <--- ADD THIS
        semantic_analyzer=semantic_analyzer,  # <--- ADD THIS
    )

    decision_engine: providers.Factory[DecisionEngine] = providers.Factory(
        DecisionEngine,
        session=database,
        rulebook_manager=rulebook_manager,
        best_practices=best_practices_kb,
    )

    context_synthesizer: providers.Factory[ContextSynthesizer] = providers.Factory(
        ContextSynthesizer
    )

    # Optimization layer providers (factories)
    token_budget_manager: providers.Factory[TokenBudgetManager] = providers.Factory(
        TokenBudgetManager,
        config=BudgetConfig(),
    )

    model_router: providers.Factory[ModelRouter] = providers.Factory(
        ModelRouter,
        budget_manager=token_budget_manager,
    )

    prompt_compressor: providers.Factory[PromptCompressor] = providers.Factory(
        PromptCompressor,
    )

    # Execution layer providers (factories)
    keyword_researcher: providers.Factory[KeywordResearcher] = providers.Factory(
        KeywordResearcher,
    )

    content_planner: providers.Factory[ContentPlanner] = providers.Factory(
        ContentPlanner,
    )

    content_generator: providers.Factory[ContentGenerator] = providers.Factory(
        ContentGenerator,
        llm_client=llm,
        context_synthesizer=context_synthesizer,
        semantic_analyzer=semantic_analyzer,
        model_router=model_router,
        token_budget_manager=token_budget_manager,
        prompt_compressor=prompt_compressor,
        metrics_collector=metrics,
    )

    # Orchestration layer providers (factories)

    content_agent: providers.Factory[ContentAgent] = providers.Factory(
        ContentAgent,
        database_manager=database,
        rulebook_manager=rulebook_manager,
        website_analyzer=website_analyzer,
        decision_engine=decision_engine,
        context_synthesizer=context_synthesizer,
        keyword_researcher=keyword_researcher,
        content_planner=content_planner,
        content_generator=content_generator,
        budget_manager=token_budget_manager,
        metrics_collector=metrics,
        config=providers.Factory(ContentAgentConfig),
    )

    # Service layer providers (factories)
    project_service: providers.Factory[ProjectService] = providers.Factory(
        ProjectService,
        database_manager=database,
    )

    content_service: providers.Factory[ContentService] = providers.Factory(
        ContentService,
        article_repository=article_repository,
        project_service=project_service,
    )

    user_service: providers.Factory[UserService] = providers.Factory(
        UserService,
        user_repository=user_repository,
    )

    # Security providers (singletons)
    password_verifier: providers.Singleton[callable] = providers.Singleton(verify_password)

    password_hasher: providers.Singleton[callable] = providers.Singleton(get_password_hash)

    token_creator: providers.Singleton[callable] = providers.Singleton(create_access_token)

    token_decoder: providers.Singleton[callable] = providers.Singleton(decode_access_token)

    user_authenticator: providers.Singleton[callable] = providers.Singleton(authenticate_user)

    current_user_provider: providers.Singleton[callable] = providers.Singleton(get_current_user)

    current_active_user_provider: providers.Singleton[callable] = providers.Singleton(
        get_current_active_user
    )

    current_superuser_provider: providers.Singleton[callable] = providers.Singleton(
        get_current_superuser
    )


# Global container instance
container = Container()


class ContainerManager:
    """
    Container lifecycle manager.

    Handles initialization, wiring, and cleanup of the dependency injection
    container with proper async resource management.
    """

    def __init__(self) -> None:
        self._container: Optional[Container] = None
        self._initialized: bool = False

    async def initialize(self) -> None:
        """
        Initialize the container and all infrastructure dependencies.

        Performs async initialization of database connections, Redis clients,
        and other infrastructure components that require async setup.

        This method uses graceful degradation - if non-critical components
        fail to initialize, the application continues with reduced functionality.

        Raises:
            RuntimeError: If critical infrastructure components fail to initialize
        """
        if self._initialized:
            logger.warning("Container already initialized - skipping re-initialization")
            return

        logger.info("Initializing dependency injection container")

        initialization_errors = []

        try:
            # Initialize database (critical component)
            try:
                await container.database.provided.initialize()
                logger.info("✓ Database initialized successfully")
            except Exception as db_error:
                error_msg = f"Database initialization failed: {db_error}"
                logger.error(error_msg)
                initialization_errors.append(("database", str(db_error)))
                # Database is critical - re-raise after cleanup
                logger.error("Database is a critical component - cannot continue without it")
                raise RuntimeError(error_msg) from db_error

            # Initialize Redis (non-critical - cache fallback to in-memory)
            try:
                await container.redis.provided.initialize()
                logger.info("✓ Redis initialized successfully")
            except Exception as redis_error:
                error_msg = f"Redis initialization failed: {redis_error}"
                logger.warning(error_msg)
                initialization_errors.append(("redis", str(redis_error)))
                logger.warning("Continuing without Redis - caching will use in-memory fallback")

            # Log initialization summary
            if initialization_errors:
                logger.warning(
                    f"Container initialized with {len(initialization_errors)} warning(s): "
                    f"{', '.join(f'{comp}' for comp, _ in initialization_errors)}"
                )
            else:
                logger.info("✓ All container components initialized successfully")

            self._initialized = True

        except Exception as e:
            logger.error(f"❌ Container initialization failed critically: {e}")
            # Attempt cleanup of partially initialized components
            await self.cleanup()
            raise RuntimeError(f"Failed to initialize dependency injection container: {e}") from e

    async def cleanup(self) -> None:
        """
        Clean up container resources.

        Properly closes database connections, Redis clients, and other
        infrastructure components that require cleanup.

        This method is idempotent and safe to call multiple times.
        Errors during cleanup are logged but don't prevent other components
        from being cleaned up.
        """
        if not self._initialized:
            logger.debug("Container not initialized - skipping cleanup")
            return

        logger.info("Cleaning up dependency injection container")

        cleanup_errors = []

        # Cleanup database connections
        try:
            if container.database.provided:
                logger.debug("Closing database connections...")
                await container.database.provided.close()
                logger.info("✓ Database connections closed")
        except Exception as db_error:
            error_msg = f"Database cleanup failed: {db_error}"
            logger.error(error_msg)
            cleanup_errors.append(("database", str(db_error)))

        # Cleanup Redis connections
        try:
            if container.redis.provided:
                logger.debug("Closing Redis connections...")
                await container.redis.provided.close()
                logger.info("✓ Redis connections closed")
        except Exception as redis_error:
            error_msg = f"Redis cleanup failed: {redis_error}"
            logger.error(error_msg)
            cleanup_errors.append(("redis", str(redis_error)))

        # Log cleanup summary
        if cleanup_errors:
            logger.warning(
                f"Container cleanup completed with {len(cleanup_errors)} error(s): "
                f"{', '.join(f'{comp}' for comp, _ in cleanup_errors)}"
            )
        else:
            logger.info("✓ Container cleanup completed successfully")

        self._initialized = False

    def get_container(self) -> Container:
        """Get the container instance."""
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        return container


# Global container manager instance
container_manager = ContainerManager()


# Convenience functions for dependency injection
@inject
def get_database(db: DatabaseManager = Provide[Container.database]) -> DatabaseManager:
    """Get database manager instance."""
    return db


@inject
def get_redis(redis: RedisClient = Provide[Container.redis]) -> RedisClient:
    """Get Redis client instance."""
    return redis


@inject
def get_metrics(metrics: MetricsCollector = Provide[Container.metrics]) -> MetricsCollector:
    """Get metrics collector instance."""
    return metrics


@inject
def get_llm(llm: AbstractLLMClient = Provide[Container.llm]) -> AbstractLLMClient:
    """Get LLM client instance."""
    return llm


@inject
def get_llm_client_dependency(
    llm_client: AbstractLLMClient = Provide[Container.llm_client],
) -> AbstractLLMClient:
    """
    Get LLM client instance for dependency injection.

    This is a FastAPI dependency function. For direct access to the LLM client
    factory, use container.llm() or container.llm_client().

    Returns:
        AbstractLLMClient: Configured LLM client instance
    """
    return llm_client


@inject
def get_cache(cache: CacheManager = Provide[Container.cache]) -> CacheManager:
    """Get cache manager instance."""
    return cache


@inject
def get_project_repository(
    repo: ProjectRepository = Provide[Container.project_repository],
) -> ProjectRepository:
    """Get project repository instance."""
    return repo


@inject
def get_article_repository(
    repo: ArticleRepository = Provide[Container.article_repository],
) -> ArticleRepository:
    """Get article repository instance."""
    return repo


@inject
def get_rulebook_manager(
    manager: RulebookManager = Provide[Container.rulebook_manager],
) -> RulebookManager:
    """Get rulebook manager instance."""
    return manager


@inject
def get_website_analyzer(
    analyzer: WebsiteAnalyzer = Provide[Container.website_analyzer],
) -> WebsiteAnalyzer:
    """Get website analyzer instance."""
    return analyzer


@inject
def get_semantic_analyzer(
    analyzer: SemanticAnalyzer = Provide[Container.semantic_analyzer],
) -> SemanticAnalyzer:
    """Get semantic analyzer instance."""
    return analyzer


@inject
def get_decision_engine(
    engine: DecisionEngine = Provide[Container.decision_engine],
) -> DecisionEngine:
    """Get decision engine instance."""
    return engine


@inject
def get_context_synthesizer(
    synthesizer: ContextSynthesizer = Provide[Container.context_synthesizer],
) -> ContextSynthesizer:
    """Get context synthesizer instance."""
    return synthesizer


@inject
def get_model_router(router: ModelRouter = Provide[Container.model_router]) -> ModelRouter:
    """Get model router instance."""
    return router


@inject
def get_token_budget_manager(
    manager: TokenBudgetManager = Provide[Container.token_budget_manager],
) -> TokenBudgetManager:
    """Get token budget manager instance."""
    return manager


@inject
def get_prompt_compressor(
    compressor: PromptCompressor = Provide[Container.prompt_compressor],
) -> PromptCompressor:
    """Get prompt compressor instance."""
    return compressor


@inject
def get_keyword_researcher(
    researcher: KeywordResearcher = Provide[Container.keyword_researcher],
) -> KeywordResearcher:
    """Get keyword researcher instance."""
    return researcher


@inject
def get_content_planner(
    planner: ContentPlanner = Provide[Container.content_planner],
) -> ContentPlanner:
    """Get content planner instance."""
    return planner


@inject
def get_content_generator(
    generator: ContentGenerator = Provide[Container.content_generator],
) -> ContentGenerator:
    """Get content generator instance."""
    return generator



@inject
def get_content_agent(agent: ContentAgent = Provide[Container.content_agent]) -> ContentAgent:
    """Get content agent instance."""
    return agent


@inject
def get_project_service(
    service: ProjectService = Provide[Container.project_service],
) -> ProjectService:
    """Get project service instance."""
    return service


@inject
def get_content_service(
    service: ContentService = Provide[Container.content_service],
) -> ContentService:
    """Get content service instance."""
    return service


@inject
def get_user_repository(
    repository: UserRepository = Provide[Container.user_repository],
) -> UserRepository:
    """Get user repository instance."""
    return repository


@inject
def get_user_service(service: UserService = Provide[Container.user_service]) -> UserService:
    """Get user service instance."""
    return service


# Security dependency injection functions
@inject
def get_password_verifier(verifier: callable = Provide[Container.password_verifier]) -> callable:
    """Get password verification function."""
    return verifier


@inject
def get_password_hasher(hasher: callable = Provide[Container.password_hasher]) -> callable:
    """Get password hashing function."""
    return hasher


@inject
def get_token_creator(creator: callable = Provide[Container.token_creator]) -> callable:
    """Get token creation function."""
    return creator


@inject
def get_token_decoder(decoder: callable = Provide[Container.token_decoder]) -> callable:
    """Get token decoding function."""
    return decoder


@inject
def get_user_authenticator(
    authenticator: callable = Provide[Container.user_authenticator],
) -> callable:
    """Get user authentication function."""
    return authenticator


@inject
def get_current_user_dependency(
    provider: callable = Provide[Container.current_user_provider],
) -> callable:
    """Get current user dependency function."""
    return provider


@inject
def get_current_active_user_dependency(
    provider: callable = Provide[Container.current_active_user_provider],
) -> callable:
    """Get current active user dependency function."""
    return provider


@inject
def get_current_superuser_dependency(
    provider: callable = Provide[Container.current_superuser_provider],
) -> callable:
    """Get current superuser dependency function."""
    return provider


# Container wiring helper
def wire_container(*modules: str) -> None:
    """
    Wire the container to specified modules.

    This enables dependency injection in the specified modules.

    Args:
        *modules: Module names to wire
    """
    container.wire(modules=modules)
    logger.info(f"Container wired to modules: {modules}")


def unwire_container(*modules: str) -> None:
    """
    Unwire the container from specified modules.

    Args:
        *modules: Module names to unwire
    """
    container.unwire(modules=modules)
    logger.info(f"Container unwired from modules: {modules}")


# Export public API
__all__ = [
    "Container",
    "ContainerManager",
    "container",
    "container_manager",
    "wire_container",
    "unwire_container",
    # Dependency injection functions
    "get_database",
    "get_redis",
    "get_metrics",
    "get_llm",
    "get_llm_client_dependency",
    "get_cache",
    "get_project_repository",
    "get_article_repository",
    "get_rulebook_manager",
    "get_website_analyzer",
    "get_semantic_analyzer",
    "get_decision_engine",
    "get_context_synthesizer",
    "get_model_router",
    "get_token_budget_manager",
    "get_prompt_compressor",
    "get_keyword_researcher",
    "get_content_planner",
    "get_content_generator",
    "get_content_agent",
    "get_project_service",
    "get_content_service",
    "get_user_repository",
    "get_user_service",
    # Security dependency injection functions
    "get_password_verifier",
    "get_password_hasher",
    "get_token_creator",
    "get_token_decoder",
    "get_user_authenticator",
    "get_current_user_dependency",
    "get_current_active_user_dependency",
    "get_current_superuser_dependency",
]
