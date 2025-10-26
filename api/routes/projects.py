"""
Project Routes: CRUD with Relationship Management

Implements project management operations:
- CRUD with optimistic concurrency control
- Bulk operations for batch processing
- Filtering and search
- Rulebook and pattern management
- Article history and analytics

Design Pattern: Resource-Oriented Architecture with HATEOAS principles
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

# Import dependency functions from container
from container import container, get_project_service
from core.exceptions import ProjectNotFoundError
from core.models import InferredPatterns, Project, Rulebook
from infrastructure.database import DatabaseManager
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager
from security import User, get_current_active_user
from services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["Projects"])


# Simple dependency function for FastAPI
def get_project_service_dependency() -> ProjectService:
    """Get ProjectService instance for FastAPI dependency injection."""
    return container.project_service()


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


# Rulebook manager is now accessed through ProjectService


# ============================================================================
# PROJECT CRUD OPERATIONS
# ============================================================================


@router.put("/{project_id}", response_model=dict, summary="Update project")
async def update_project(
    project_id: UUID,
    request: UpdateProjectRequest,
    project_service: ProjectService = Depends(get_project_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Update project properties.

    Implements optimistic concurrency control to prevent conflicts.
    Only non-null fields are updated (partial update semantics).
    """
    update_data = request.dict(exclude_unset=True)
    return await project_service.update_project(project_id, update_data)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete project")
async def delete_project(
    project_id: UUID,
    cascade: bool = Query(False, description="Delete associated content"),
    project_service: ProjectService = Depends(get_project_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Delete project.

    Args:
        cascade: If true, deletes all associated content (articles, patterns, etc.)
                 If false, fails if project has associated content.
    """
    await project_service.delete_project(project_id, cascade)


@router.get(
    "/{project_id}/analytics",
    response_model=ProjectAnalyticsResponse,
    summary="Get project analytics",
)
async def get_project_analytics(
    project_id: UUID,
    start_date: Optional[datetime] = Query(None, description="Analytics period start"),
    end_date: Optional[datetime] = Query(None, description="Analytics period end"),
    project_service: ProjectService = Depends(get_project_service_dependency),
):
    """
    Retrieve project performance analytics.

    Returns aggregated metrics for content generation performance,
    cost, and quality over specified time period.
    """
    analytics = await project_service.get_project_analytics(project_id, start_date, end_date)
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
    project_service: ProjectService = Depends(get_project_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Create or update project rulebook.

    Rulebooks are versioned. Each update creates a new version while
    preserving history for audit and rollback purposes.
    """
    result = await project_service.create_or_update_rulebook(
        project_id, request.content, request.version_note
    )
    return RulebookResponse(**result)


@router.get(
    "/{project_id}/rulebook", response_model=RulebookResponse, summary="Get project rulebook"
)
async def get_rulebook(
    project_id: UUID,
    version: Optional[int] = Query(None, description="Specific version (default: latest)"),
    project_service: ProjectService = Depends(get_project_service_dependency),
):
    """
    Retrieve project rulebook.

    Returns latest version by default, or specific version if requested.
    """
    result = await project_service.get_rulebook(project_id, version)
    return RulebookResponse(**result)


@router.get(
    "/{project_id}/rulebook/history",
    response_model=List[RulebookResponse],
    summary="Get rulebook version history",
)
async def get_rulebook_history(
    project_id: UUID, project_service: ProjectService = Depends(get_project_service_dependency)
):
    """
    Retrieve complete rulebook version history.

    Returns all versions ordered by version number (newest first).
    """
    history = await project_service.get_rulebook_history(project_id)
    return [RulebookResponse(**rb) for rb in history]


@router.delete(
    "/{project_id}/rulebook", status_code=status.HTTP_204_NO_CONTENT, summary="Delete rulebook"
)
async def delete_rulebook(
    project_id: UUID,
    project_service: ProjectService = Depends(get_project_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Delete project rulebook (all versions).

    This operation cannot be undone. Project will fall back to
    inferred patterns or best practices.
    """
    await project_service.delete_rulebook(project_id)


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
    project_service: ProjectService = Depends(get_project_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Trigger website analysis to infer content patterns.

    Scrapes target website and extracts linguistic patterns for
    content generation guidance. Returns immediately; analysis
    runs asynchronously.

    Args:
        force_refresh: Re-analyze even if recent patterns exist
    """
    result = await project_service.trigger_website_analysis(project_id, force_refresh)
    return InferredPatternsResponse(**result)


@router.get(
    "/{project_id}/patterns",
    response_model=InferredPatternsResponse,
    summary="Get inferred patterns",
)
async def get_inferred_patterns(
    project_id: UUID, project_service: ProjectService = Depends(get_project_service_dependency)
):
    """
    Retrieve inferred content patterns for project.

    Returns linguistic patterns extracted from website analysis.
    """
    result = await project_service.get_inferred_patterns(project_id)
    return InferredPatternsResponse(**result)


# ============================================================================
# BULK OPERATIONS
# ============================================================================


@router.post("/bulk/create", response_model=BulkProjectResponse, summary="Bulk create projects")
async def bulk_create_projects(
    projects_data: List[dict],
    project_service: ProjectService = Depends(get_project_service_dependency),
    user: User = Depends(get_current_active_user),
):
    """
    Create multiple projects in single operation.

    Useful for initial setup or migration scenarios.
    Returns summary of successful and failed operations.
    """
    result = await project_service.bulk_create_projects(projects_data)
    return BulkProjectResponse(**result)


# ============================================================================
# ADVANCED FILTERING & SEARCH
# ============================================================================


@router.get("/search", response_model=List[dict], summary="Search projects")
async def search_projects(
    query: str = Query(..., min_length=1, description="Search query"),
    field: str = Query("name", pattern="^(name|domain)$", description="Field to search"),
    limit: int = Query(20, ge=1, le=100),
    project_service: ProjectService = Depends(get_project_service_dependency),
):
    """
    Search projects by name or domain.

    Supports fuzzy matching for user-friendly search experience.
    """
    return await project_service.search_projects(query, field, limit)


@router.get("/filter", response_model=List[dict], summary="Filter projects with advanced criteria")
async def filter_projects(
    has_rulebook: Optional[bool] = Query(None, description="Filter by rulebook presence"),
    has_patterns: Optional[bool] = Query(None, description="Filter by inferred patterns"),
    min_articles: Optional[int] = Query(None, ge=0, description="Minimum article count"),
    created_after: Optional[datetime] = Query(None, description="Created after date"),
    project_service: ProjectService = Depends(get_project_service_dependency),
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

    return await project_service.filter_projects(**filters)
