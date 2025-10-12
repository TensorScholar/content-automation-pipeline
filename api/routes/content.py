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
from pydantic import BaseModel, Field

from core.exceptions import WorkflowError
from core.models import ContentPlan, GeneratedArticle
from infrastructure.database import DatabaseManager
from orchestration.content_agent import ContentAgent
from orchestration.task_queue import TaskManager
from knowledge.article_repository import ArticleRepository
from services.content_service import ContentService

# Import dependency functions
from api.dependencies import get_db_manager, get_article_repository, get_task_manager, get_content_service

router = APIRouter(prefix="/content", tags=["Content"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================


class BatchGenerateRequest(BaseModel):
    """Command: Batch content generation."""

    project_id: UUID
    topics: List[str] = Field(..., min_items=1, max_items=50)
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


@router.post(
    "/generate/batch",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch generate content",
)
async def batch_generate_content(
    request: BatchGenerateRequest,
    background_tasks: BackgroundTasks,
    content_service: ContentService = Depends(get_content_service),
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
    batch_id: str, 
    content_service: ContentService = Depends(get_content_service)
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
    content_service: ContentService = Depends(get_content_service),
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
    content_service: ContentService = Depends(get_content_service),
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
    content_service: ContentService = Depends(get_content_service)
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
    article_id: UUID, 
    content_service: ContentService = Depends(get_content_service)
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
    content_service: ContentService = Depends(get_content_service)
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
    content_service: ContentService = Depends(get_content_service),
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
    article_id: UUID, 
    content_service: ContentService = Depends(get_content_service)
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
    article_id: UUID, 
    content_service: ContentService = Depends(get_content_service)
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
    content_service: ContentService = Depends(get_content_service),
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
    content_service: ContentService = Depends(get_content_service),
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
    content_service: ContentService = Depends(get_content_service),
):
    """
        Full-text search across article titles and content.
    Uses PostgreSQL full-text search for efficient querying.
    Returns ranked results by relevance.
    """
    return await content_service.search_articles(query, project_id, limit)
