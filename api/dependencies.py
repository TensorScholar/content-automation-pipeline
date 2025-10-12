"""
API Dependencies: FastAPI Dependency Injection Helpers

Centralized dependency injection functions for clean separation of concerns.
Implements Service Locator pattern for dependency management.
"""

from typing import Optional

from fastapi import Depends, Request

from config.settings import settings
from infrastructure.database import DatabaseManager
from infrastructure.monitoring import MetricsCollector
from infrastructure.redis_client import RedisClient
from orchestration.content_agent import ContentAgent, ContentAgentConfig
from orchestration.task_queue import TaskManager
from knowledge.project_repository import ProjectRepository


class DependencyContainer:
    """
    Application-wide dependency container implementing Service Locator pattern.

    Provides singleton lifecycle management for infrastructure components
    with lazy initialization and graceful shutdown.

    Theoretical Foundation: Dependency Inversion Principle + Factory Pattern
    """

    _instance: Optional["DependencyContainer"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        """Initialize all dependencies with proper lifecycle management."""
        if self._initialized:
            return

        # Infrastructure layer
        self.db = DatabaseManager()
        await self.db.connect()

        self.redis = RedisClient()
        await self.redis.connect()

        self.metrics = MetricsCollector()

        # Initialize all application layers
        from execution.content_generator import ContentGenerator
        from execution.content_planner import ContentPlanner
        from execution.distributer import Distributor
        from execution.keyword_researcher import KeywordResearcher
        from infrastructure.llm_client import LLMClient
        from intelligence.context_synthesizer import ContextSynthesizer
        from intelligence.decision_engine import DecisionEngine
        from intelligence.semantic_analyzer import SemanticAnalyzer
        from knowledge.project_repository import ProjectRepository
        from knowledge.rulebook_manager import RulebookManager
        from knowledge.website_analyzer import WebsiteAnalyzer
        from optimization.cache_manager import CacheManager
        from optimization.model_router import ModelRouter
        from optimization.prompt_compressor import PromptCompressor
        from optimization.token_budget_manager import TokenBudgetManager

        # Delegate layer initialization to specialized methods
        await self._initialize_knowledge_layer(locals())
        await self._initialize_intelligence_layer(locals())
        await self._initialize_optimization_layer(locals())
        await self._initialize_execution_layer(locals())
        await self._initialize_orchestration_layer(locals())

        self._initialized = True

    async def cleanup(self):
        """Graceful shutdown with resource cleanup."""
        if hasattr(self, "db"):
            await self.db.disconnect()

        if hasattr(self, "redis"):
            await self.redis.disconnect()

        self._initialized = False

    # =========================================================================
    # LAYER-SPECIFIC INITIALIZATION METHODS
    # =========================================================================

    async def _initialize_knowledge_layer(self, imports):
        """Initialize Knowledge layer components."""
        ProjectRepository = imports['ProjectRepository']
        RulebookManager = imports['RulebookManager']
        WebsiteAnalyzer = imports['WebsiteAnalyzer']

        # Knowledge layer
        self.projects = ProjectRepository(self.db)
        self.rulebook_mgr = RulebookManager(self.db)
        self.website_analyzer = WebsiteAnalyzer()

    async def _initialize_intelligence_layer(self, imports):
        """Initialize Intelligence layer components."""
        SemanticAnalyzer = imports['SemanticAnalyzer']
        DecisionEngine = imports['DecisionEngine']
        CacheManager = imports['CacheManager']
        ContextSynthesizer = imports['ContextSynthesizer']

        # Intelligence layer
        self.semantic_analyzer = SemanticAnalyzer()
        self.decision_engine = DecisionEngine(self.db, self.semantic_analyzer)
        self.cache = CacheManager(self.redis)
        self.context_synthesizer = ContextSynthesizer(
            self.projects, self.rulebook_mgr, self.decision_engine, self.cache
        )

    async def _initialize_optimization_layer(self, imports):
        """Initialize Optimization layer components."""
        LLMClient = imports['LLMClient']
        ModelRouter = imports['ModelRouter']
        TokenBudgetManager = imports['TokenBudgetManager']
        PromptCompressor = imports['PromptCompressor']

        # Optimization layer
        self.llm = LLMClient(redis_client=self.redis)
        self.model_router = ModelRouter()
        self.budget_manager = TokenBudgetManager(self.redis, self.metrics)
        self.prompt_compressor = PromptCompressor(self.semantic_analyzer)

    async def _initialize_execution_layer(self, imports):
        """Initialize Execution layer components."""
        KeywordResearcher = imports['KeywordResearcher']
        ContentPlanner = imports['ContentPlanner']
        ContentGenerator = imports['ContentGenerator']
        Distributor = imports['Distributor']

        # Execution layer
        self.keyword_researcher = KeywordResearcher(self.llm, self.semantic_analyzer, self.cache)
        self.content_planner = ContentPlanner(
            self.llm, self.decision_engine, self.context_synthesizer, self.model_router
        )
        self.content_generator = ContentGenerator(
            self.llm,
            self.context_synthesizer,
            self.semantic_analyzer,
            self.model_router,
            self.budget_manager,
            self.prompt_compressor,
            self.metrics,
        )
        self.distributor = Distributor(
            telegram_bot_token=settings.telegram.bot_token.get_secret_value() if settings.telegram.bot_token else None,
            metrics_collector=self.metrics
        )

    async def _initialize_orchestration_layer(self, imports):
        """Initialize Orchestration layer components."""
        ContentAgent = imports['ContentAgent']
        TaskManager = imports['TaskManager']

        # Orchestration layer
        self.content_agent = ContentAgent(
            project_repository=self.projects,
            rulebook_manager=self.rulebook_mgr,
            website_analyzer=self.website_analyzer,
            decision_engine=self.decision_engine,
            context_synthesizer=self.context_synthesizer,
            keyword_researcher=self.keyword_researcher,
            content_planner=self.content_planner,
            content_generator=self.content_generator,
            distributor=self.distributor,
            budget_manager=self.budget_manager,
            metrics_collector=self.metrics,
            config=ContentAgentConfig(),
        )

        self.task_manager = TaskManager()


# Dependency injection helpers
async def get_container(request: Request) -> DependencyContainer:
    """Dependency injection: Get application container."""
    return request.app.state.container


async def get_content_agent(
    container: DependencyContainer = Depends(get_container),
) -> ContentAgent:
    """Dependency injection: Get content agent."""
    return container.content_agent


async def get_task_manager(container: DependencyContainer = Depends(get_container)) -> TaskManager:
    """Dependency injection: Get task manager."""
    return container.task_manager


async def get_project_repository(container: DependencyContainer = Depends(get_container)):
    """Dependency injection: Get project repository."""
    return container.projects


async def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency injection: Get database manager."""
    return request.app.state.container.db


async def get_redis_client(request: Request) -> RedisClient:
    """Dependency injection: Get Redis client."""
    return request.app.state.container.redis
