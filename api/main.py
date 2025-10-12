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
import time
import uuid
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
from starlette.responses import Response

from core.models import ContentPlan, GeneratedArticle, Project
from infrastructure.database import DatabaseManager
from infrastructure.monitoring import MetricsCollector
from infrastructure.redis_client import RedisClient
from orchestration.content_agent import ContentAgent, ContentAgentConfig
from orchestration.task_queue import TaskManager
from config.settings import settings

# Import schemas from separate module
from api.schemas import (
    ArticleResponse,
    CreateProjectRequest,
    ErrorResponse,
    GenerateContentRequest,
    HealthCheckResponse,
    ProjectResponse,
    TaskStatusResponse,
    WorkflowStatusResponse,
)


# Import dependencies from separate module
from api.dependencies import get_content_agent, get_project_repository, get_task_manager


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan context manager.

    Manages startup/shutdown lifecycle with proper resource initialization
    and cleanup. Implements graceful degradation on startup failures.
    """
    try:
        # Infrastructure layer
        db = DatabaseManager()
        await db.connect()
        app.state.db = db

        redis = RedisClient()
        await redis.connect()
        app.state.redis = redis

        metrics = MetricsCollector()
        app.state.metrics = metrics

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
        projects = ProjectRepository(db)
        app.state.projects = projects
        
        rulebook_mgr = RulebookManager(db)
        app.state.rulebook_mgr = rulebook_mgr
        
        website_analyzer = WebsiteAnalyzer()
        app.state.website_analyzer = website_analyzer

        # Intelligence layer
        semantic_analyzer = SemanticAnalyzer()
        app.state.semantic_analyzer = semantic_analyzer
        
        decision_engine = DecisionEngine(db, semantic_analyzer)
        app.state.decision_engine = decision_engine
        
        cache = CacheManager(redis)
        app.state.cache = cache
        
        context_synthesizer = ContextSynthesizer(
            projects, rulebook_mgr, decision_engine, cache
        )
        app.state.context_synthesizer = context_synthesizer

        # Optimization layer
        llm = LLMClient(redis_client=redis)
        app.state.llm = llm
        
        model_router = ModelRouter()
        app.state.model_router = model_router
        
        budget_manager = TokenBudgetManager(redis, metrics)
        app.state.budget_manager = budget_manager
        
        prompt_compressor = PromptCompressor(semantic_analyzer)
        app.state.prompt_compressor = prompt_compressor

        # Execution layer
        keyword_researcher = KeywordResearcher(llm, semantic_analyzer, cache)
        app.state.keyword_researcher = keyword_researcher
        
        content_planner = ContentPlanner(
            llm, decision_engine, context_synthesizer, model_router
        )
        app.state.content_planner = content_planner
        
        content_generator = ContentGenerator(
            llm,
            context_synthesizer,
            semantic_analyzer,
            model_router,
            budget_manager,
            prompt_compressor,
            metrics,
        )
        app.state.content_generator = content_generator
        
        distributor = Distributor(
            telegram_bot_token=settings.telegram.bot_token.get_secret_value() if settings.telegram.bot_token else None,
            metrics_collector=metrics
        )
        app.state.distributor = distributor

        # Orchestration layer
        content_agent = ContentAgent(
            project_repository=projects,
            rulebook_manager=rulebook_mgr,
            website_analyzer=website_analyzer,
            decision_engine=decision_engine,
            context_synthesizer=context_synthesizer,
            keyword_researcher=keyword_researcher,
            content_planner=content_planner,
            content_generator=content_generator,
            distributor=distributor,
            budget_manager=budget_manager,
            metrics_collector=metrics,
            config=ContentAgentConfig(),
        )
        app.state.content_agent = content_agent

        task_manager = TaskManager()
        app.state.task_manager = task_manager

        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(app.state, "db"):
            await app.state.db.disconnect()
        if hasattr(app.state, "redis"):
            await app.state.redis.disconnect()
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
        redis = request.app.state.redis

        key = f"rate_limit:{client_id}"
        now = int(time.time())  # Current Unix timestamp (seconds)
        window_start = now - self.window

        # Step 1: Remove old requests (outside the sliding window)
        await redis.zremrangebyscore(key, 0, window_start)

        # Step 2: Get current request count within the window
        current_count = await redis.zcard(key)

        if current_count >= self.rate_limit:
            logger.warning(f"Rate limit exceeded | client={client_id}")

            # Calculate Retry-After header: Find the score of the oldest request in the set
            oldest_request = await redis.zrange(key, 0, 0, withscores=True)
            
            if oldest_request:
                # Oldest request is at the window_start. The next request can be served 
                # when the window moves past this oldest request.
                # Reset time = (oldest_request timestamp + window duration) - now
                _, oldest_score = oldest_request[0]
                retry_after = int(oldest_score) + self.window - now
            else:
                retry_after = self.window  # Should not happen if current_count > 0, but fallback

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate Limit Exceeded",
                    "detail": f"Maximum {self.rate_limit} requests per {self.window}s",
                    "retry_after": max(1, retry_after),
                },
                headers={"Retry-After": str(max(1, retry_after))},
            )

        # Step 3: Record the new request (timestamp = score and member)
        # Using ZADD with NX (Not Exist) is often preferred, but here we just add the timestamp
        # to represent the log of requests. We use the current timestamp for both score and member.
        await redis.zadd(key, {str(now) + "_" + str(uuid.uuid4()): now})

        response = await call_next(request)

        # Add rate limit headers
        # The Remaining count is the limit minus the current count *before* this request was added
        remaining = max(0, self.rate_limit - current_count - 1) 
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        # Reset time is more complex with Sliding Window Log, often set to the end of the current window for simplicity
        response.headers["X-RateLimit-Reset"] = str(int(datetime.utcnow().timestamp()) + self.window)

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

# Add exception handlers
add_exception_handlers(app)

# Include route modules
app.include_router(content.router)
app.include_router(projects.router)

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


# Dependency injection helpers are now imported from api.dependencies


# Import exception handlers from separate module
from api.exceptions import add_exception_handlers

# Import route modules
from api.routes import content, projects


# ============================================================================
# API ENDPOINTS (Command/Query Handlers)
# ============================================================================


@app.get(
    "/health", response_model=HealthCheckResponse, tags=["System"], summary="Health check endpoint"
)
async def health_check(request: Request):
    """
    System health check with dependency status.

    Returns health status of all critical dependencies.
    """
    dependencies = {}

    # Check database
    try:
        await request.app.state.db.execute("SELECT 1")
        dependencies["database"] = "healthy"
    except Exception as e:
        dependencies["database"] = f"unhealthy: {str(e)}"

    # Check Redis
    try:
        await request.app.state.redis.ping()
        dependencies["redis"] = "healthy"
    except Exception as e:
        dependencies["redis"] = f"unhealthy: {str(e)}"

    # Check task queue
    try:
        worker_status = request.app.state.task_manager.get_worker_status()
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
        await app.state.rulebook_mgr.create_rulebook(
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
    rulebook = await app.state.rulebook_mgr.get_rulebook(project_id)
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
async def get_metrics(request: Request):
    """
    Export metrics in Prometheus format.

    Compatible with Prometheus scraper for monitoring/alerting.
    """
    metrics_collector = request.app.state.metrics

    # Generate Prometheus-formatted metrics
    metrics_content = metrics_collector.export_metrics()
    content_type = metrics_collector.get_content_type()

    return Response(content=metrics_content, media_type=content_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info", access_log=True
    )
