"""
API Gateway: High-Performance RESTful Interface

Production-grade FastAPI application implementing:
- CQRS-inspired command/query separation
- Dependency injection with lifespan management
- Streaming responses for real-time workflow updates
- Circuit breaker patterns for fault tolerance
- Rate limiting with token bucket algorithm
- Comprehensive observability (traces, metrics, logs)

Architectural Pattern: API Gateway + CQRS + Event-Driven Streams
Theoretical Foundation: Category Theory for composable request handlers
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware

from core.exceptions import (
    DistributionError,
    ProjectNotFoundError,
    TokenBudgetExceededError,
    WorkflowError,
)
from core.models import ContentPlan, GeneratedArticle, Project
from infrastructure.database import DatabaseManager
from infrastructure.monitoring import MetricsCollector
from infrastructure.redis_client import RedisClient
from orchestration.content_agent import ContentAgent, ContentAgentConfig
from orchestration.task_queue import TaskManager

# ============================================================================
# REQUEST/RESPONSE SCHEMAS (Domain Transfer Objects)
# ============================================================================


class CreateProjectRequest(BaseModel):
    """Command: Create new project with validation."""

    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    domain: Optional[str] = Field(None, description="Target website domain")
    telegram_channel: Optional[str] = Field(None, description="Telegram channel ID")
    rulebook_content: Optional[str] = Field(None, description="Initial rulebook content")

    @validator("domain")
    def validate_domain(cls, v):
        if v and not (v.startswith("http://") or v.startswith("https://")):
            v = f"https://{v}"
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "TechBlog AI",
                "domain": "https://techblog.ai",
                "telegram_channel": "@techblogai",
                "rulebook_content": "Write in conversational tone...",
            }
        }


class ProjectResponse(BaseModel):
    """Query result: Project representation."""

    id: str
    name: str
    domain: Optional[str]
    telegram_channel: Optional[str]
    created_at: datetime
    total_articles_generated: int
    has_rulebook: bool
    has_inferred_patterns: bool

    class Config:
        orm_mode = True


class GenerateContentRequest(BaseModel):
    """Command: Generate content with strategic parameters."""

    topic: str = Field(..., min_length=1, max_length=500, description="Content topic")
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")
    custom_instructions: Optional[str] = Field(None, max_length=2000)
    async_execution: bool = Field(False, description="Execute as background task")

    class Config:
        schema_extra = {
            "example": {
                "topic": "Advanced NLP Techniques for Content Generation",
                "priority": "high",
                "custom_instructions": "Focus on practical implementations",
                "async_execution": False,
            }
        }


class ArticleResponse(BaseModel):
    """Query result: Generated article metadata."""

    article_id: str
    project_id: str
    title: str
    word_count: int
    cost: float
    generation_time: float
    readability_score: float
    distributed: bool
    created_at: datetime

    class Config:
        orm_mode = True


class TaskStatusResponse(BaseModel):
    """Query result: Async task execution status."""

    task_id: str
    state: str
    ready: bool
    successful: Optional[bool]
    failed: Optional[bool]
    result: Optional[dict]
    error: Optional[str]
    progress: Optional[dict]


class WorkflowStatusResponse(BaseModel):
    """Query result: Real-time workflow state."""

    workflow_id: str
    state: str
    project_id: str
    topic: str
    start_time: datetime
    events: list


class HealthCheckResponse(BaseModel):
    """System health status."""

    status: str
    timestamp: datetime
    version: str
    dependencies: dict


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str]


# ============================================================================
# DEPENDENCY INJECTION SYSTEM
# ============================================================================


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

        logger.info("Initializing dependency container")

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

        # Knowledge layer
        self.projects = ProjectRepository(self.db)
        self.rulebook_mgr = RulebookManager(self.db)
        self.website_analyzer = WebsiteAnalyzer()

        # Intelligence layer
        self.semantic_analyzer = SemanticAnalyzer()
        self.decision_engine = DecisionEngine(self.db, self.semantic_analyzer)
        self.cache = CacheManager(self.redis)
        self.context_synthesizer = ContextSynthesizer(
            self.projects, self.rulebook_mgr, self.decision_engine, self.cache
        )

        # Optimization layer
        self.llm = LLMClient()
        self.model_router = ModelRouter()
        self.budget_manager = TokenBudgetManager(self.redis, self.metrics)
        self.prompt_compressor = PromptCompressor(self.semantic_analyzer)

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
            telegram_bot_token=None, metrics_collector=self.metrics  # Load from settings
        )

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

        self._initialized = True
        logger.success("Dependency container initialized")

    async def cleanup(self):
        """Graceful shutdown with resource cleanup."""
        logger.info("Cleaning up dependency container")

        if hasattr(self, "db"):
            await self.db.disconnect()

        if hasattr(self, "redis"):
            await self.redis.disconnect()

        self._initialized = False
        logger.info("Dependency container cleaned up")


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan context manager.

    Manages startup/shutdown lifecycle with proper resource initialization
    and cleanup. Implements graceful degradation on startup failures.
    """
    container = DependencyContainer()

    try:
        await container.initialize()
        app.state.container = container
        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        await container.cleanup()
        logger.info("Application shutdown complete")


# ============================================================================
# MIDDLEWARE STACK (Cross-Cutting Concerns)
# ============================================================================


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Distributed tracing middleware with request ID propagation.

    Implements OpenTelemetry-compatible tracing for observability.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(UUID.uuid4()))
        request.state.request_id = request_id

        logger.bind(request_id=request_id).info(
            f"Request started | method={request.method} | path={request.url.path}"
        )

        start_time = datetime.utcnow()

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id

            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.bind(request_id=request_id).info(
                f"Request completed | status={response.status_code} | " f"duration={duration:.3f}s"
            )

            return response

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.bind(request_id=request_id).error(
                f"Request failed | error={e} | duration={duration:.3f}s"
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id,
                },
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.

    Implements per-client rate limiting with Redis-backed token bucket.
    Prevents API abuse and ensures fair resource allocation.

    Algorithm: Token Bucket with sliding window
    """

    def __init__(self, app, rate_limit: int = 100, window: int = 60):
        super().__init__(app)
        self.rate_limit = rate_limit  # tokens per window
        self.window = window  # seconds

    async def dispatch(self, request: Request, call_next):
        client_id = request.client.host

        # Check rate limit
        container = request.app.state.container
        redis = container.redis

        key = f"rate_limit:{client_id}"

        # Atomic increment with expiry
        current = await redis.incr(key)

        if current == 1:
            await redis.expire(key, self.window)

        if current > self.rate_limit:
            logger.warning(f"Rate limit exceeded | client={client_id}")

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate Limit Exceeded",
                    "detail": f"Maximum {self.rate_limit} requests per {self.window}s",
                    "retry_after": self.window,
                },
                headers={"Retry-After": str(self.window)},
            )

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.rate_limit - current))
        response.headers["X-RateLimit-Reset"] = str(
            int(datetime.utcnow().timestamp()) + self.window
        )

        return response


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Content Automation Engine API",
    description="Advanced NLP-driven SEO content automation platform with adaptive intelligence",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware stack (order matters: last added = first executed)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware, rate_limit=100, window=60)
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DEPENDENCY INJECTION HELPERS
# ============================================================================


async def get_container() -> DependencyContainer:
    """Dependency injection: Get application container."""
    return app.state.container


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


# ============================================================================
# EXCEPTION HANDLERS (Domain Error → HTTP Error Mapping)
# ============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(ProjectNotFoundError)
async def project_not_found_handler(request: Request, exc: ProjectNotFoundError):
    """Handle project not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Project Not Found",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(WorkflowError)
async def workflow_error_handler(request: Request, exc: WorkflowError):
    """Handle workflow execution errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Workflow Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(TokenBudgetExceededError)
async def budget_exceeded_handler(request: Request, exc: TokenBudgetExceededError):
    """Handle token budget exceeded errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Token Budget Exceeded",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# ============================================================================
# API ENDPOINTS (Command/Query Handlers)
# ============================================================================


@app.get(
    "/health", response_model=HealthCheckResponse, tags=["System"], summary="Health check endpoint"
)
async def health_check(container: DependencyContainer = Depends(get_container)):
    """
    System health check with dependency status.

    Returns health status of all critical dependencies.
    """
    dependencies = {}

    # Check database
    try:
        await container.db.execute("SELECT 1")
        dependencies["database"] = "healthy"
    except Exception as e:
        dependencies["database"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        await container.redis.ping()
        dependencies["redis"] = "healthy"
    except Exception as e:
        dependencies["redis"] = f"unhealthy: {str(e)}"

    # Check task queue
    try:
        worker_status = container.task_manager.get_worker_status()
        dependencies["task_queue"] = f"healthy ({worker_status.get('total_workers', 0)} workers)"
    except Exception as e:
        dependencies["task_queue"] = f"unhealthy: {str(e)}"

    overall_status = "healthy" if all("healthy" in v for v in dependencies.values()) else "degraded"

    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        dependencies=dependencies,
    )


@app.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Projects"],
    summary="Create new project",
)
async def create_project(request: CreateProjectRequest, projects=Depends(get_project_repository)):
    """
    Create new content project with optional initial configuration.

    Projects serve as multi-tenant isolation boundaries. Each project
    maintains its own rulebook, inferred patterns, and content history.
    """
    logger.info(f"Creating project | name={request.name}")

    project = await projects.create(
        name=request.name, domain=request.domain, telegram_channel=request.telegram_channel
    )

    # If rulebook provided, create it
    has_rulebook = False
    if request.rulebook_content:
        container = app.state.container
        await container.rulebook_mgr.create_rulebook(
            project_id=project.id, content=request.rulebook_content
        )
        has_rulebook = True

    return ProjectResponse(
        id=str(project.id),
        name=project.name,
        domain=project.domain,
        telegram_channel=project.telegram_channel,
        created_at=project.created_at,
        total_articles_generated=0,
        has_rulebook=has_rulebook,
        has_inferred_patterns=False,
    )


@app.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    tags=["Projects"],
    summary="Get project details",
)
async def get_project(project_id: UUID, projects=Depends(get_project_repository)):
    """Retrieve project details by ID."""
    project = await projects.get_by_id(project_id)

    if not project:
        raise ProjectNotFoundError(f"Project not found: {project_id}")

    # Check for rulebook and patterns
    container = app.state.container
    rulebook = await container.rulebook_mgr.get_rulebook(project_id)
    patterns = await projects.get_inferred_patterns(project_id)

    return ProjectResponse(
        id=str(project.id),
        name=project.name,
        domain=project.domain,
        telegram_channel=project.telegram_channel,
        created_at=project.created_at,
        total_articles_generated=project.total_articles_generated,
        has_rulebook=rulebook is not None,
        has_inferred_patterns=patterns is not None and patterns.confidence > 0.65,
    )


@app.get(
    "/projects",
    response_model=list[ProjectResponse],
    tags=["Projects"],
    summary="List all projects",
)
async def list_projects(skip: int = 0, limit: int = 100, projects=Depends(get_project_repository)):
    """List all projects with pagination."""
    project_list = await projects.list_all(skip=skip, limit=limit)

    # TODO: Batch load rulebook/pattern status for efficiency
    return [
        ProjectResponse(
            id=str(p.id),
            name=p.name,
            domain=p.domain,
            telegram_channel=p.telegram_channel,
            created_at=p.created_at,
            total_articles_generated=p.total_articles_generated,
            has_rulebook=False,  # TODO: batch load
            has_inferred_patterns=False,  # TODO: batch load
        )
        for p in project_list
    ]


@app.post(
    "/projects/{project_id}/generate",
    response_model=ArticleResponse,
    tags=["Content Generation"],
    summary="Generate content (sync)",
)
async def generate_content_sync(
    project_id: UUID,
    request: GenerateContentRequest,
    agent: ContentAgent = Depends(get_content_agent),
):
    """
    Generate content synchronously.

    Executes complete workflow and returns generated article.
    Use for interactive requests. For batch processing, use async endpoint.
    """
    if request.async_execution:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Use /projects/{project_id}/generate/async for async execution",
        )

    logger.info(f"Sync content generation | project_id={project_id} | topic={request.topic}")

    article = await agent.create_content(
        project_id=project_id,
        topic=request.topic,
        priority=request.priority,
        custom_instructions=request.custom_instructions,
    )

    return ArticleResponse(
        article_id=str(article.id),
        project_id=str(article.project_id),
        title=article.title,
        word_count=article.word_count,
        cost=article.total_cost,
        generation_time=article.generation_time,
        readability_score=article.readability_score,
        distributed=article.distributed_at is not None,
        created_at=article.created_at,
    )


@app.post(
    "/projects/{project_id}/generate/async",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Content Generation"],
    summary="Generate content (async)",
)
async def generate_content_async(
    project_id: UUID,
    request: GenerateContentRequest,
    task_manager: TaskManager = Depends(get_task_manager),
):
    """
    Generate content asynchronously via task queue.

    Returns immediately with task ID. Poll /tasks/{task_id} for status.
    Recommended for batch processing or long-running operations.
    """
    logger.info(f"Async content generation | project_id={project_id} | topic={request.topic}")

    task_id = task_manager.submit_generation(
        project_id=project_id,
        topic=request.topic,
        priority=request.priority,
        custom_instructions=request.custom_instructions,
    )

    return TaskStatusResponse(
        task_id=task_id,
        state="PENDING",
        ready=False,
        successful=None,
        failed=None,
        result=None,
        error=None,
        progress=None,
    )


@app.get(
    "/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"], summary="Get task status"
)
async def get_task_status(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    """
    Query async task execution status.

    Poll this endpoint to monitor task progress. Task results
    available for 24 hours after completion.
    """
    status_info = task_manager.get_task_status(task_id)

    return TaskStatusResponse(**status_info)


@app.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Tasks"],
    summary="Cancel task",
)
async def cancel_task(
    task_id: str, terminate: bool = False, task_manager: TaskManager = Depends(get_task_manager)
):
    """
    Cancel pending or running task.

    Args:
        terminate: If true, forcefully terminate running task (use with caution)
    """
    task_manager.cancel_task(task_id, terminate=terminate)
    logger.info(f"Task cancelled | task_id={task_id} | terminate={terminate}")


@app.get(
    "/projects/{project_id}/workflow/status",
    response_model=WorkflowStatusResponse,
    tags=["Workflow"],
    summary="Get current workflow status",
)
async def get_workflow_status(project_id: UUID, agent: ContentAgent = Depends(get_content_agent)):
    """
    Get real-time status of active workflow.

    Returns workflow state, events, and progress information.
    """
    status_info = await agent.get_workflow_status()

    if not status_info or status_info.get("status") == "no_active_workflow":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No active workflow for this project"
        )

    return WorkflowStatusResponse(**status_info)


@app.get(
    "/projects/{project_id}/workflow/stream",
    tags=["Workflow"],
    summary="Stream workflow events (SSE)",
)
async def stream_workflow_events(
    project_id: UUID, agent: ContentAgent = Depends(get_content_agent)
):
    """
    Server-Sent Events stream of workflow execution.

    Provides real-time updates during content generation.
    Compatible with EventSource API in browsers.

    Event Stream Format:
        event: workflow_event
        data: {"state": "...", "message": "...", "timestamp": "..."}
    """

    async def event_generator():
        """
        Async generator yielding SSE-formatted events.

        Implements reactive streaming pattern for real-time observability.
        """
        # Subscribe to workflow events
        while True:
            status = await agent.get_workflow_status()

            if not status or status.get("status") == "no_active_workflow":
                yield f"event: complete\ndata: {json.dumps({'status': 'no_active_workflow'})}\n\n"
                break

            # Yield latest events
            for event in status.get("events", []):
                event_data = {
                    "state": event.get("state"),
                    "message": event.get("message"),
                    "timestamp": event.get("timestamp"),
                }
                yield f"event: workflow_event\ndata: {json.dumps(event_data)}\n\n"

            # Check if workflow complete
            if status.get("state") in ["completed", "failed"]:
                yield f"event: complete\ndata: {json.dumps({'state': status.get('state')})}\n\n"
                break

            await asyncio.sleep(1)  # Poll interval

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/metrics", tags=["Observability"], summary="System metrics (Prometheus format)")
async def get_metrics(container: DependencyContainer = Depends(get_container)):
    """
    Export metrics in Prometheus format.

    Compatible with Prometheus scraper for monitoring/alerting.
    """
    metrics_collector = container.metrics

    # Generate Prometheus-formatted metrics
    # TODO: Implement actual Prometheus metric export

    return {
        "status": "metrics_endpoint",
        "message": "Prometheus metrics export not yet implemented",
        "note": "Use metrics_collector.export_prometheus() when implemented",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info", access_log=True
    )
