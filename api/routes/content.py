"""
Content Routes: Generation, Management, and Analytics

Comprehensive content lifecycle management:
- Single and batch generation
- Quality metrics and analysis
- Content revision and iteration
- Distribution management
- Performance analytics

Design Pattern: Command Query Responsibility Segregation (CQRS)
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from loguru import logger
from pydantic import BaseModel, Field

# Import dependency functions from container
from container import container, get_content_service
from core.exceptions import WorkflowError
from core.models import ContentPlan, GeneratedArticle
from infrastructure.database import DatabaseManager

from knowledge.article_repository import ArticleRepository
from orchestration.content_agent import ContentAgent
from security import User, get_current_active_user
from services.content_service import ContentService

router = APIRouter(prefix="/content", tags=["Content"])


# Simple dependency function for FastAPI
def get_content_service_dependency() -> ContentService:
    """Get ContentService instance for FastAPI dependency injection."""
    return container.content_service()


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================


class BatchGenerateRequest(BaseModel):
    """Command: Batch content generation."""

    project_id: UUID
    topics: List[str] = Field(..., min_length=1, max_length=50)
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")
    schedule_after: Optional[datetime] = Field(None, description="Delayed execution")


class ContentRevisionRequest(BaseModel):
    """Command: Request content revision."""

    feedback: str = Field(..., min_length=10, description="Revision feedback")
    sections_to_revise: Optional[List[str]] = Field(None, description="Specific sections")
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")


class ContentQualityMetrics(BaseModel):
    """Query result: Detailed quality metrics."""

    article_id: str
    readability_score: float
    readability_grade: str
    keyword_density: dict
    semantic_coherence: float
    structure_score: float
    seo_score: float
    overall_quality: float


class ContentHistoryResponse(BaseModel):
    """Query result: Article with revision history."""

    current_version: dict
    revisions: List[dict]
    total_revisions: int


class DistributionStatusResponse(BaseModel):
    """Query result: Distribution status."""

    article_id: str
    distributed: bool
    channels: List[str]
    distributed_at: Optional[datetime]
    delivery_confirmations: dict


class ContentAnalyticsResponse(BaseModel):
    """Query result: Content performance analytics."""

    total_articles: int
    total_cost: float
    avg_generation_time: float
    avg_quality_score: float
    cost_per_article: float
    articles_by_day: List[dict]
    quality_trend: List[dict]


# ============================================================================
# CONTENT GENERATION
# ============================================================================


class GenerateContentRequest(BaseModel):
    """Command: Generate single article asynchronously."""

    project_id: UUID
    topic: str = Field(..., min_length=3, max_length=500)
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")
    custom_instructions: Optional[str] = Field(None, description="Custom generation instructions")


@router.post(
    "/generate/async",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate content asynchronously",
)
async def generate_content_async(
    request: GenerateContentRequest,
    user: User = Depends(get_current_active_user),
):
    """
    Generate a single article asynchronously using Celery.

    Dispatches content generation task to Celery worker queue and returns
    immediately with a task ID for tracking progress.

    The task will:
    1. Execute keyword research
    2. Create content plan
    3. Generate full article
    4. Optionally distribute to configured channels

    Args:
        request: Generation parameters including project_id, topic, priority

    Returns:
        Dict with task_id for status tracking via /content/task/{task_id}

    Example:
        POST /content/generate/async
        {
            "project_id": "123e4567-e89b-12d3-a456-426614174000",
            "topic": "AI-Powered Content Automation",
            "priority": "high",
            "custom_instructions": "Focus on technical implementation"
        }

        Response:
        {
            "task_id": "abc-123-def-456",
            "status": "queued",
            "project_id": "123e4567-e89b-12d3-a456-426614174000",
            "topic": "AI-Powered Content Automation",
            "submitted_at": "2024-01-15T10:30:00Z"
        }
    """
    from orchestration.tasks import generate_content_task

    logger.info(
        f"Async content generation requested | project_id={request.project_id} | "
        f"topic={request.topic} | priority={request.priority}"
    )

    # Dispatch Celery task
    task = generate_content_task.apply_async(
        args=[
            str(request.project_id),
            request.topic,
            request.priority,
            request.custom_instructions,
        ],
        queue=request.priority,  # Route to appropriate priority queue
        routing_key=request.priority,
    )

    logger.info(
        f"Content generation task dispatched | task_id={task.id} | "
        f"project_id={request.project_id}"
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "project_id": str(request.project_id),
        "topic": request.topic,
        "priority": request.priority,
        "submitted_at": datetime.utcnow().isoformat(),
        "status_endpoint": f"/content/task/{task.id}",
    }


@router.get(
    "/task/{task_id}",
    response_model=dict,
    summary="Get task status",
)
async def get_task_status(task_id: str):
    """
    Query the status of an asynchronous content generation task.

    Args:
        task_id: The task ID returned from /generate/async

    Returns:
        Dict with task state, progress, and result (if completed)

    Task States:
        - PENDING: Task is waiting to be executed
        - STARTED: Task has been started
        - RETRY: Task is being retried
        - FAILURE: Task failed (includes error info)
        - SUCCESS: Task completed successfully (includes article data)
    """
    from celery.result import AsyncResult

    from orchestration.celery_app import app

    result = AsyncResult(task_id, app=app)

    response = {
        "task_id": task_id,
        "state": result.state,
        "ready": result.ready(),
    }

    # Add state-specific information
    if result.state == "PENDING":
        response["status"] = "Task is queued and waiting to be executed"

    elif result.state == "STARTED":
        response["status"] = "Task is currently being processed"
        response["current"] = result.info.get("current", 0) if isinstance(result.info, dict) else 0
        response["total"] = result.info.get("total", 100) if isinstance(result.info, dict) else 100

    elif result.state == "RETRY":
        response["status"] = "Task is being retried after a failure"
        response["retry_count"] = (
            result.info.get("retry_count", 0) if isinstance(result.info, dict) else 0
        )

    elif result.state == "FAILURE":
        response["status"] = "Task failed"
        response["error"] = str(result.info) if result.info else "Unknown error"
        response["traceback"] = result.traceback

    elif result.state == "SUCCESS":
        response["status"] = "Task completed successfully"
        response["result"] = result.result
        response["completed_at"] = result.date_done.isoformat() if result.date_done else None

    else:
        response["status"] = f"Unknown state: {result.state}"

    return response


@router.post(
    "/generate/batch",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch generate content",
)
async def batch_generate_content(
    request: BatchGenerateRequest,
    background_tasks: BackgroundTasks,
    content_service: ContentService = Depends(get_content_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Generate multiple articles in parallel.

    Submits batch job to task queue. All articles generated concurrently
    within resource constraints. Returns batch tracking ID.

    Args:
        schedule_after: Optional delayed execution (for scheduled content)
    """
    return await content_service.batch_generate_content(
        request.project_id, request.topics, request.priority, request.schedule_after
    )


@router.get("/batch/{batch_id}/status", response_model=dict, summary="Get batch generation status")
async def get_batch_status(
    batch_id: str, content_service: ContentService = Depends(get_content_service_dependency)
):
    """
    Query batch generation progress.

    Returns aggregated status of all articles in batch including
    completion percentage, failures, and individual task statuses.
    """
    return await content_service.get_batch_status(batch_id)


# ============================================================================
# CONTENT MANAGEMENT
# ============================================================================


@router.get("/{article_id}", response_model=dict, summary="Get article details")
async def get_article(
    article_id: UUID,
    include_content: bool = Query(True, description="Include full content"),
    content_service: ContentService = Depends(get_content_service_dependency),
):
    """
    Retrieve article by ID.

    Args:
        include_content: If false, returns metadata only (faster)
    """
    return await content_service.get_article(article_id, include_content)


@router.post(
    "/{article_id}/revise",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request article revision",
)
async def revise_article(
    article_id: UUID,
    request: ContentRevisionRequest,
    content_service: ContentService = Depends(get_content_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Request revision of existing article based on feedback.

    Creates revision task that regenerates specified sections or entire
    article incorporating feedback. Original version preserved.
    """
    return await content_service.request_article_revision(
        article_id, request.feedback, request.sections_to_revise, request.priority
    )


@router.delete("/{article_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete article")
async def delete_article(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Delete article permanently.

    This operation cannot be undone. Article is removed from all systems.
    """
    await content_service.delete_article(article_id)


# ============================================================================
# QUALITY ANALYSIS
# ============================================================================


@router.get(
    "/{article_id}/quality", response_model=ContentQualityMetrics, summary="Get quality metrics"
)
async def get_quality_metrics(
    article_id: UUID, content_service: ContentService = Depends(get_content_service_dependency)
):
    """
    Retrieve detailed quality metrics for article.

    Analyzes readability, SEO, structure, and semantic coherence.
    Returns comprehensive quality assessment.
    """
    result = await content_service.get_quality_metrics(article_id)
    return ContentQualityMetrics(**result)


# Helper function moved to ContentService


@router.post("/{article_id}/analyze", response_model=dict, summary="Trigger comprehensive analysis")
async def trigger_comprehensive_analysis(
    article_id: UUID,
    background_tasks: BackgroundTasks,
    content_service: ContentService = Depends(get_content_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Trigger deep analysis of article quality.
    Performs extensive quality checks including:
    - Plagiarism detection (semantic similarity to existing content)
    - SEO optimization analysis
    - Readability assessment
    - Fact-checking capabilities

    Runs asynchronously; results available via separate endpoint.
    """
    return await content_service.trigger_comprehensive_analysis(article_id)


# ============================================================================
# DISTRIBUTION MANAGEMENT
# ============================================================================
@router.post(
    "/{article_id}/distribute",
    response_model=DistributionStatusResponse,
    summary="Distribute article",
)
async def distribute_article(
    article_id: UUID,
    channels: List[str] = Query(..., description="Distribution channels"),
    content_service: ContentService = Depends(get_content_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
        Distribute article to specified channels.
        Supports multi-channel distribution including:
    - Telegram
    - WordPress (TODO)
    - Email (TODO)
    - Social media (TODO)
    """
    result = await content_service.distribute_article(article_id, channels)
    return DistributionStatusResponse(**result)


@router.get(
    "/{article_id}/distribution",
    response_model=DistributionStatusResponse,
    summary="Get distribution status",
)
async def get_distribution_status(
    article_id: UUID, content_service: ContentService = Depends(get_content_service_dependency)
):
    """
    Query article distribution status.
    Returns delivery confirmations and channel-specific metadata.
    """
    result = await content_service.get_distribution_status(article_id)
    return DistributionStatusResponse(**result)


# ============================================================================
# CONTENT HISTORY & VERSIONING
# ============================================================================
@router.get(
    "/{article_id}/history",
    response_model=ContentHistoryResponse,
    summary="Get article revision history",
)
async def get_article_history(
    article_id: UUID, content_service: ContentService = Depends(get_content_service_dependency)
):
    """
    Retrieve complete revision history for article.
    Returns all versions with diff information and revision metadata.
    """
    result = await content_service.get_article_history(article_id)
    return ContentHistoryResponse(**result)


# ============================================================================
# ANALYTICS & REPORTING
# ============================================================================
@router.get("/analytics", response_model=ContentAnalyticsResponse, summary="Get content analytics")
async def get_content_analytics(
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    start_date: datetime = Query(datetime.utcnow() - timedelta(days=30)),
    end_date: datetime = Query(datetime.utcnow()),
    content_service: ContentService = Depends(get_content_service_dependency),
):
    """
    Retrieve comprehensive content generation analytics.
    Returns aggregated metrics including cost analysis, quality trends,
    and production velocity over specified time period.
    """
    analytics = await content_service.get_content_analytics(project_id, start_date, end_date)
    return ContentAnalyticsResponse(**analytics)


@router.get("/export", summary="Export content data")
async def export_content(
    project_id: Optional[UUID] = Query(None),
    format: str = Query("json", pattern="^(json|csv)$"),
    start_date: datetime = Query(datetime.utcnow() - timedelta(days=30)),
    end_date: datetime = Query(datetime.utcnow()),
    content_service: ContentService = Depends(get_content_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
        Export content data in specified format.
    Supports JSON and CSV formats for integration with external systems
    or data analysis tools.
    """
    articles = await content_service.export_content(project_id, start_date, end_date)

    if format == "json":
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={
                "articles": articles,
                "total": len(articles),
                "exported_at": datetime.utcnow().isoformat(),
            }
        )

    elif format == "csv":
        import csv
        import io

        from fastapi.responses import StreamingResponse

        # Generate CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=articles[0].keys() if articles else [])
        writer.writeheader()
        writer.writerows(articles)

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=content_export_{datetime.utcnow().date()}.csv"
            },
        )


# ============================================================================
# SEARCH & FILTERING
# ============================================================================
@router.get("/search", response_model=List[dict], summary="Search articles")
async def search_articles(
    query: str = Query(..., min_length=1, description="Search query"),
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    limit: int = Query(20, ge=1, le=100),
    content_service: ContentService = Depends(get_content_service_dependency),
):
    """
        Full-text search across article titles and content.
    Uses PostgreSQL full-text search for efficient querying.
    Returns ranked results by relevance.
    """
    return await content_service.search_articles(query, project_id, limit)
