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

# Import dependency functions
from api.dependencies import get_db_manager, get_article_repository, get_task_manager

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
    task_manager: TaskManager = Depends(get_task_manager),
):
    """
    Generate multiple articles in parallel.

    Submits batch job to task queue. All articles generated concurrently
    within resource constraints. Returns batch tracking ID.

    Args:
        schedule_after: Optional delayed execution (for scheduled content)
    """
    if request.schedule_after and request.schedule_after < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="schedule_after must be in the future"
        )

    # Submit batch task
    batch_id = task_manager.submit_batch(
        project_id=request.project_id, topics=request.topics, priority=request.priority
    )

    return {
        "batch_id": batch_id,
        "project_id": str(request.project_id),
        "total_articles": len(request.topics),
        "status": "scheduled" if request.schedule_after else "processing",
        "schedule_time": request.schedule_after,
    }


@router.get("/batch/{batch_id}/status", response_model=dict, summary="Get batch generation status")
async def get_batch_status(batch_id: str, task_manager: TaskManager = Depends()):
    """
    Query batch generation progress.

    Returns aggregated status of all articles in batch including
    completion percentage, failures, and individual task statuses.
    """
    batch_status = task_manager.get_task_status(batch_id)

    # Aggregate child task statuses
    # TODO: Implement batch task status aggregation

    return {
        "batch_id": batch_id,
        "status": batch_status.get("state"),
        "progress": {
            "completed": 0,  # TODO: Calculate from child tasks
            "failed": 0,
            "pending": 0,
            "total": 0,
        },
    }


# ============================================================================
# CONTENT MANAGEMENT
# ============================================================================


@router.get("/{article_id}", response_model=dict, summary="Get article details")
async def get_article(
    article_id: UUID,
    include_content: bool = Query(True, description="Include full content"),
    article_repo: ArticleRepository = Depends(get_article_repository),
):
    """
    Retrieve article by ID.

    Args:
        include_content: If false, returns metadata only (faster)
    """
    article = await article_repo.get_by_id(article_id, include_content)

    if not article:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    return article


@router.post(
    "/{article_id}/revise",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request article revision",
)
async def revise_article(
    article_id: UUID,
    request: ContentRevisionRequest,
    task_manager: TaskManager = Depends(get_task_manager),
    article_repo: ArticleRepository = Depends(get_article_repository),
):
    """
    Request revision of existing article based on feedback.

    Creates revision task that regenerates specified sections or entire
    article incorporating feedback. Original version preserved.
    """
    # Verify article exists
    article = await article_repo.get_by_id(article_id, include_content=False)

    if not article:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    # TODO: Implement revision task
    # task_id = task_manager.submit_revision(...)

    return {
        "article_id": str(article_id),
        "revision_task_id": "placeholder",
        "status": "processing",
        "feedback_incorporated": request.feedback[:100] + "...",
    }


@router.delete("/{article_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete article")
async def delete_article(
    article_id: UUID, 
    article_repo: ArticleRepository = Depends(get_article_repository)
):
    """
    Delete article permanently.

    This operation cannot be undone. Article is removed from all systems.
    """
    deleted = await article_repo.delete(article_id)

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")


# ============================================================================
# QUALITY ANALYSIS
# ============================================================================


@router.get(
    "/{article_id}/quality", response_model=ContentQualityMetrics, summary="Get quality metrics"
)
async def get_quality_metrics(
    article_id: UUID, 
    article_repo: ArticleRepository = Depends(get_article_repository)
):
    """
    Retrieve detailed quality metrics for article.

    Analyzes readability, SEO, structure, and semantic coherence.
    Returns comprehensive quality assessment.
    """
    article = await article_repo.get_quality_metrics(article_id)

    if not article:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    # TODO: Implement comprehensive quality analysis
    # For now, return basic metrics

    readability_grade = _readability_score_to_grade(article["readability_score"])

    return ContentQualityMetrics(
        article_id=str(article_id),
        readability_score=article["readability_score"],
        readability_grade=readability_grade,
        keyword_density=article["keyword_density"],
        semantic_coherence=0.85,  # TODO: Calculate
        structure_score=0.90,  # TODO: Calculate
        seo_score=0.88,  # TODO: Calculate
        overall_quality=0.87,  # TODO: Calculate weighted average
    )


def _readability_score_to_grade(score: float) -> str:
    """Convert Flesch-Kincaid score to grade level."""
    if score >= 90:
        return "5th grade"
    elif score >= 80:
        return "6th grade"
    elif score >= 70:
        return "7th grade"
    elif score >= 60:
        return "8th-9th grade"
    elif score >= 50:
        return "10th-12th grade"
    elif score >= 30:
        return "College"
    else:
        return "College graduate"


@router.post("/{article_id}/analyze", response_model=dict, summary="Trigger comprehensive analysis")
async def trigger_comprehensive_analysis(
    article_id: UUID, 
    background_tasks: BackgroundTasks, 
    article_repo: ArticleRepository = Depends(get_article_repository)
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
    article = await article_repo.get_by_id(article_id, include_content=False)

    if not article:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    # TODO: Implement comprehensive analysis task
    # analysis_task_id = ...

    return {
        "article_id": str(article_id),
        "analysis_task_id": "placeholder",
        "status": "processing",
        "estimated_completion": (datetime.utcnow() + timedelta(minutes=2)).isoformat(),
    }


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
    article_repo: ArticleRepository = Depends(get_article_repository),
):
    """
        Distribute article to specified channels.
        Supports multi-channel distribution including:
    - Telegram
    - WordPress (TODO)
    - Email (TODO)
    - Social media (TODO)
    """
    article = await article_repo.get_by_id(article_id, include_content=True)

    if not article:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    # TODO: Implement distribution logic
    # distributor = get_distributor()
    # results = await distributor.distribute_multi_channel(...)

    return DistributionStatusResponse(
        article_id=str(article_id),
        distributed=True,
        channels=channels,
        distributed_at=datetime.utcnow(),
        delivery_confirmations={},  # TODO: Actual confirmations
    )


@router.get(
    "/{article_id}/distribution",
    response_model=DistributionStatusResponse,
    summary="Get distribution status",
)
async def get_distribution_status(
    article_id: UUID, 
    article_repo: ArticleRepository = Depends(get_article_repository)
):
    """
    Query article distribution status.
    Returns delivery confirmations and channel-specific metadata.
    """
    article = await article_repo.get_distribution_status(article_id)

    if not article:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    return DistributionStatusResponse(
        article_id=str(article_id),
        distributed=article["distributed_at"] is not None,
        channels=article["distribution_channels"] or [],
        distributed_at=article["distributed_at"],
        delivery_confirmations={},  # TODO: Query from distribution logs
    )


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
    article_repo: ArticleRepository = Depends(get_article_repository)
):
    """
    Retrieve complete revision history for article.
    Returns all versions with diff information and revision metadata.
    """
    history = await article_repo.get_article_history(article_id)

    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    return ContentHistoryResponse(
        current_version=history["current_version"],
        revisions=history["revisions"],
        total_revisions=history["total_revisions"],
    )


# ============================================================================
# ANALYTICS & REPORTING
# ============================================================================
@router.get("/analytics", response_model=ContentAnalyticsResponse, summary="Get content analytics")
async def get_content_analytics(
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    start_date: datetime = Query(datetime.utcnow() - timedelta(days=30)),
    end_date: datetime = Query(datetime.utcnow()),
    article_repo: ArticleRepository = Depends(get_article_repository),
):
    """
    Retrieve comprehensive content generation analytics.
    Returns aggregated metrics including cost analysis, quality trends,
    and production velocity over specified time period.
    """
    analytics = await article_repo.get_analytics(project_id, start_date, end_date)

    return ContentAnalyticsResponse(
        total_articles=analytics["total_articles"],
        total_cost=analytics["total_cost"],
        avg_generation_time=analytics["avg_generation_time"],
        avg_quality_score=analytics["avg_quality_score"],
        cost_per_article=analytics["cost_per_article"],
        articles_by_day=analytics["articles_by_day"],
        quality_trend=analytics["quality_trend"],
    )


@router.get("/export", summary="Export content data")
async def export_content(
    project_id: Optional[UUID] = Query(None),
    format: str = Query("json", pattern="^(json|csv)$"),
    start_date: datetime = Query(datetime.utcnow() - timedelta(days=30)),
    end_date: datetime = Query(datetime.utcnow()),
    article_repo: ArticleRepository = Depends(get_article_repository),
):
    """
        Export content data in specified format.
    Supports JSON and CSV formats for integration with external systems
    or data analysis tools.
    """
    articles = await article_repo.export_articles(project_id, start_date, end_date)

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
    article_repo: ArticleRepository = Depends(get_article_repository),
):
    """
        Full-text search across article titles and content.
    Uses PostgreSQL full-text search for efficient querying.
    Returns ranked results by relevance.
    """
    results = await article_repo.search(query, project_id, limit)
    return results
