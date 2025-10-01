"""
Project Routes: Advanced CRUD with Relationship Management

Implements comprehensive project management operations:
- CRUD with optimistic concurrency control
- Bulk operations for batch processing
- Advanced filtering and search
- Rulebook and pattern management
- Article history and analytics

Design Pattern: Resource-Oriented Architecture with HATEOAS principles
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from core.exceptions import ProjectNotFoundError
from core.models import InferredPatterns, Project, Rulebook
from infrastructure.database import DatabaseManager
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager

router = APIRouter(prefix="/projects", tags=["Projects"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================


class UpdateProjectRequest(BaseModel):
    """Command: Update project properties."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    domain: Optional[str] = None
    telegram_channel: Optional[str] = None


class RulebookRequest(BaseModel):
    """Command: Create or update rulebook."""

    content: str = Field(..., min_length=10, description="Rulebook content")
    version_note: Optional[str] = Field(None, description="Version change notes")


class RulebookResponse(BaseModel):
    """Query result: Rulebook representation."""

    id: str
    project_id: str
    content: str
    version: int
    rule_count: int
    updated_at: datetime

    class Config:
        orm_mode = True


class InferredPatternsResponse(BaseModel):
    """Query result: Inferred patterns representation."""

    id: str
    project_id: str
    avg_sentence_length: float
    lexical_diversity: float
    readability_score: float
    confidence: float
    sample_size: int
    analyzed_at: datetime

    class Config:
        orm_mode = True


class ProjectAnalyticsResponse(BaseModel):
    """Query result: Project performance analytics."""

    project_id: str
    total_articles: int
    total_cost: float
    avg_generation_time: float
    avg_readability: float
    avg_word_count: int
    period_start: datetime
    period_end: datetime


class BulkProjectResponse(BaseModel):
    """Result: Bulk operation outcome."""

    successful: List[str]
    failed: List[dict]
    total_processed: int


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================


async def get_project_repo() -> ProjectRepository:
    """Inject project repository."""
    # In production, get from app.state.container
    db = DatabaseManager()
    return ProjectRepository(db)


async def get_rulebook_manager() -> RulebookManager:
    """Inject rulebook manager."""
    db = DatabaseManager()
    return RulebookManager(db)


# ============================================================================
# PROJECT CRUD OPERATIONS
# ============================================================================


@router.put("/{project_id}", response_model=dict, summary="Update project")
async def update_project(
    project_id: UUID,
    request: UpdateProjectRequest,
    projects: ProjectRepository = Depends(get_project_repo),
):
    """
    Update project properties.

    Implements optimistic concurrency control to prevent conflicts.
    Only non-null fields are updated (partial update semantics).
    """
    project = await projects.get_by_id(project_id)

    if not project:
        raise ProjectNotFoundError(f"Project not found: {project_id}")

    # Apply updates
    update_data = request.dict(exclude_unset=True)

    updated_project = await projects.update(project_id, update_data)

    return {
        "id": str(updated_project.id),
        "message": "Project updated successfully",
        "updated_fields": list(update_data.keys()),
    }


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete project")
async def delete_project(
    project_id: UUID,
    cascade: bool = Query(False, description="Delete associated content"),
    projects: ProjectRepository = Depends(get_project_repo),
):
    """
    Delete project.

    Args:
        cascade: If true, deletes all associated content (articles, patterns, etc.)
                 If false, fails if project has associated content.
    """
    project = await projects.get_by_id(project_id)

    if not project:
        raise ProjectNotFoundError(f"Project not found: {project_id}")

    # Check for associated content
    article_count = await projects.count_articles(project_id)

    if article_count > 0 and not cascade:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Project has {article_count} associated articles. Use cascade=true to force delete.",
        )

    await projects.delete(project_id, cascade=cascade)


@router.get(
    "/{project_id}/analytics",
    response_model=ProjectAnalyticsResponse,
    summary="Get project analytics",
)
async def get_project_analytics(
    project_id: UUID,
    start_date: Optional[datetime] = Query(None, description="Analytics period start"),
    end_date: Optional[datetime] = Query(None, description="Analytics period end"),
    projects: ProjectRepository = Depends(get_project_repo),
):
    """
    Retrieve project performance analytics.

    Returns aggregated metrics for content generation performance,
    cost, and quality over specified time period.
    """
    project = await projects.get_by_id(project_id)

    if not project:
        raise ProjectNotFoundError(f"Project not found: {project_id}")

    analytics = await projects.get_analytics(
        project_id=project_id,
        start_date=start_date or datetime(2020, 1, 1),
        end_date=end_date or datetime.utcnow(),
    )

    return ProjectAnalyticsResponse(**analytics)


# ============================================================================
# RULEBOOK MANAGEMENT
# ============================================================================


@router.post(
    "/{project_id}/rulebook",
    response_model=RulebookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update rulebook",
)
async def create_or_update_rulebook(
    project_id: UUID,
    request: RulebookRequest,
    projects: ProjectRepository = Depends(get_project_repo),
    rulebook_mgr: RulebookManager = Depends(get_rulebook_manager),
):
    """
    Create or update project rulebook.

    Rulebooks are versioned. Each update creates a new version while
    preserving history for audit and rollback purposes.
    """
    # Verify project exists
    project = await projects.get_by_id(project_id)

    if not project:
        raise ProjectNotFoundError(f"Project not found: {project_id}")

    # Check if rulebook exists
    existing_rulebook = await rulebook_mgr.get_rulebook(project_id)

    if existing_rulebook:
        # Update existing
        rulebook = await rulebook_mgr.update_rulebook(
            project_id=project_id, content=request.content, version_note=request.version_note
        )
        status_message = "updated"
    else:
        # Create new
        rulebook = await rulebook_mgr.create_rulebook(
            project_id=project_id, content=request.content
        )
        status_message = "created"

    return RulebookResponse(
        id=str(rulebook.id),
        project_id=str(rulebook.project_id),
        content=rulebook.raw_content,
        version=rulebook.version,
        rule_count=len(rulebook.rules),
        updated_at=rulebook.updated_at,
    )


@router.get(
    "/{project_id}/rulebook", response_model=RulebookResponse, summary="Get project rulebook"
)
async def get_rulebook(
    project_id: UUID,
    version: Optional[int] = Query(None, description="Specific version (default: latest)"),
    rulebook_mgr: RulebookManager = Depends(get_rulebook_manager),
):
    """
    Retrieve project rulebook.

    Returns latest version by default, or specific version if requested.
    """
    if version:
        rulebook = await rulebook_mgr.get_rulebook_version(project_id, version)
    else:
        rulebook = await rulebook_mgr.get_rulebook(project_id)

    if not rulebook:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rulebook not found")

    return RulebookResponse(
        id=str(rulebook.id),
        project_id=str(rulebook.project_id),
        content=rulebook.raw_content,
        version=rulebook.version,
        rule_count=len(rulebook.rules),
        updated_at=rulebook.updated_at,
    )


@router.get(
    "/{project_id}/rulebook/history",
    response_model=List[RulebookResponse],
    summary="Get rulebook version history",
)
async def get_rulebook_history(
    project_id: UUID, rulebook_mgr: RulebookManager = Depends(get_rulebook_manager)
):
    """
    Retrieve complete rulebook version history.

    Returns all versions ordered by version number (newest first).
    """
    history = await rulebook_mgr.get_rulebook_history(project_id)

    return [
        RulebookResponse(
            id=str(rb.id),
            project_id=str(rb.project_id),
            content=rb.raw_content,
            version=rb.version,
            rule_count=len(rb.rules),
            updated_at=rb.updated_at,
        )
        for rb in history
    ]


@router.delete(
    "/{project_id}/rulebook", status_code=status.HTTP_204_NO_CONTENT, summary="Delete rulebook"
)
async def delete_rulebook(
    project_id: UUID, rulebook_mgr: RulebookManager = Depends(get_rulebook_manager)
):
    """
    Delete project rulebook (all versions).

    This operation cannot be undone. Project will fall back to
    inferred patterns or best practices.
    """
    await rulebook_mgr.delete_rulebook(project_id)


# ============================================================================
# INFERRED PATTERNS MANAGEMENT
# ============================================================================


@router.post(
    "/{project_id}/analyze",
    response_model=InferredPatternsResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger website analysis",
)
async def trigger_website_analysis(
    project_id: UUID,
    force_refresh: bool = Query(False, description="Force re-analysis"),
    projects: ProjectRepository = Depends(get_project_repo),
):
    """
    Trigger website analysis to infer content patterns.

    Scrapes target website and extracts linguistic patterns for
    content generation guidance. Returns immediately; analysis
    runs asynchronously.

    Args:
        force_refresh: Re-analyze even if recent patterns exist
    """
    project = await projects.get_by_id(project_id)

    if not project:
        raise ProjectNotFoundError(f"Project not found: {project_id}")

    if not project.domain:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project has no domain configured for analysis",
        )

    # Check for existing patterns
    existing_patterns = await projects.get_inferred_patterns(project_id)

    if existing_patterns and not force_refresh:
        # Return existing patterns
        return InferredPatternsResponse(
            id=str(existing_patterns.id),
            project_id=str(existing_patterns.project_id),
            avg_sentence_length=existing_patterns.avg_sentence_length[0],
            lexical_diversity=existing_patterns.lexical_diversity,
            readability_score=existing_patterns.readability_score,
            confidence=existing_patterns.confidence,
            sample_size=existing_patterns.sample_size,
            analyzed_at=existing_patterns.analyzed_at,
        )

    # Trigger async analysis
    from orchestration.task_queue import celery_app

    task = celery_app.send_task(
        "content_automation.analyze_website", args=[str(project_id), project.domain]
    )

    return {
        "message": "Website analysis initiated",
        "task_id": task.id,
        "project_id": str(project_id),
    }


@router.get(
    "/{project_id}/patterns",
    response_model=InferredPatternsResponse,
    summary="Get inferred patterns",
)
async def get_inferred_patterns(
    project_id: UUID, projects: ProjectRepository = Depends(get_project_repo)
):
    """
    Retrieve inferred content patterns for project.

    Returns linguistic patterns extracted from website analysis.
    """
    patterns = await projects.get_inferred_patterns(project_id)

    if not patterns:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No inferred patterns found. Trigger analysis first.",
        )

    return InferredPatternsResponse(
        id=str(patterns.id),
        project_id=str(patterns.project_id),
        avg_sentence_length=patterns.avg_sentence_length[0],
        lexical_diversity=patterns.lexical_diversity,
        readability_score=patterns.readability_score,
        confidence=patterns.confidence,
        sample_size=patterns.sample_size,
        analyzed_at=patterns.analyzed_at,
    )


# ============================================================================
# BULK OPERATIONS
# ============================================================================


@router.post("/bulk/create", response_model=BulkProjectResponse, summary="Bulk create projects")
async def bulk_create_projects(
    projects_data: List[dict], projects: ProjectRepository = Depends(get_project_repo)
):
    """
    Create multiple projects in single operation.

    Useful for initial setup or migration scenarios.
    Returns summary of successful and failed operations.
    """
    successful = []
    failed = []

    for idx, project_data in enumerate(projects_data):
        try:
            project = await projects.create(**project_data)
            successful.append(str(project.id))
        except Exception as e:
            failed.append({"index": idx, "data": project_data, "error": str(e)})

    return BulkProjectResponse(
        successful=successful, failed=failed, total_processed=len(projects_data)
    )


# ============================================================================
# ADVANCED FILTERING & SEARCH
# ============================================================================


@router.get("/search", response_model=List[dict], summary="Search projects")
async def search_projects(
    query: str = Query(..., min_length=1, description="Search query"),
    field: str = Query("name", pattern="^(name|domain)$", description="Field to search"),
    limit: int = Query(20, ge=1, le=100),
    projects: ProjectRepository = Depends(get_project_repo),
):
    """
    Search projects by name or domain.

    Supports fuzzy matching for user-friendly search experience.
    """
    results = await projects.search(query=query, field=field, limit=limit)

    return [
        {
            "id": str(p.id),
            "name": p.name,
            "domain": p.domain,
            "match_score": p.match_score,  # Relevance score
        }
        for p in results
    ]


@router.get("/filter", response_model=List[dict], summary="Filter projects with advanced criteria")
async def filter_projects(
    has_rulebook: Optional[bool] = Query(None, description="Filter by rulebook presence"),
    has_patterns: Optional[bool] = Query(None, description="Filter by inferred patterns"),
    min_articles: Optional[int] = Query(None, ge=0, description="Minimum article count"),
    created_after: Optional[datetime] = Query(None, description="Created after date"),
    projects: ProjectRepository = Depends(get_project_repo),
):
    """
    Filter projects with multiple criteria.

    Supports complex filtering for analytics and reporting use cases.
    """
    filters = {
        "has_rulebook": has_rulebook,
        "has_patterns": has_patterns,
        "min_articles": min_articles,
        "created_after": created_after,
    }

    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    results = await projects.filter(**filters)

    return [
        {
            "id": str(p.id),
            "name": p.name,
            "domain": p.domain,
            "total_articles": p.total_articles_generated,
            "created_at": p.created_at,
        }
        for p in results
    ]
