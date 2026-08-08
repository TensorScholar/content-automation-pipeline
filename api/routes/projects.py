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

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import dependency functions from new dependencies module
from api.dependencies import (
    get_content_memory_service,
    get_performance_feedback_service,
    get_project_readiness_service,
    get_project_service,
    get_seo_intelligence_service,
)
from api.schemas import CreateProjectRequest
from core.exceptions import ProjectNotFoundError
from core.models import InferredPatterns, Project, Rulebook
from infrastructure.database import DatabaseManager
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager
from security import User, get_current_active_user, get_current_superuser, is_manager_user
from services.content_memory_service import ContentMemoryService
from services.performance_feedback_service import PerformanceFeedbackService
from services.project_readiness_service import ProjectReadinessService
from services.project_service import ProjectService
from services.seo_intelligence_service import SeoIntelligenceService

router = APIRouter(prefix="/projects", tags=["Projects"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================


class UpdateProjectRequest(BaseModel):
    """Command: Update project properties."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    domain: Optional[str] = None
    telegram_channel: Optional[str] = None
    description: Optional[str] = Field(None, max_length=500)
    wordpress_url: Optional[str] = None
    wordpress_username: Optional[str] = None
    wordpress_app_password: Optional[str] = None
    vertical: Optional[str] = Field(None, max_length=100)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("Project name cannot be empty")
        return normalized

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if normalized.startswith("https://"):
            normalized = normalized.removeprefix("https://")
        elif normalized.startswith("http://"):
            normalized = normalized.removeprefix("http://")
        return normalized.rstrip("/") or None

    @field_validator(
        "telegram_channel",
        "description",
        "wordpress_url",
        "wordpress_username",
        "wordpress_app_password",
        "vertical",
    )
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None


class RulebookRequest(BaseModel):
    """Command: Create or update rulebook."""

    content: str = Field(..., min_length=1, description="Rulebook content")
    version_note: Optional[str] = Field(None, description="Version change notes")


class RulebookResponse(BaseModel):
    """Query result: Rulebook representation."""

    id: str
    project_id: str
    content: str
    version: int
    rule_count: int
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


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

    model_config = ConfigDict(from_attributes=True)


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


class ReadinessCheckResponse(BaseModel):
    """Single project readiness check."""

    id: str
    label: str
    status: str
    severity: str
    message: str
    remediation: Optional[str] = None


class ReadinessActionResponse(BaseModel):
    """Manager action surfaced by the readiness engine."""

    id: str
    label: str
    method: str
    endpoint: Optional[str] = None
    destructive: bool = False


class ProjectReadinessResponse(BaseModel):
    """Project readiness computed from current configuration and runtime health."""

    project_id: str
    status: str
    can_generate: bool
    can_publish: bool
    blocking_items: List[ReadinessCheckResponse]
    warnings: List[ReadinessCheckResponse]
    checks: List[ReadinessCheckResponse]
    manager_actions: List[ReadinessActionResponse]
    last_checked_at: datetime


class RepeatedKeywordResponse(BaseModel):
    keyword: str
    count: int


class ProjectContentMemoryResponse(BaseModel):
    project_id: str
    article_count: int
    recent_titles: List[str]
    repeated_keywords: List[RepeatedKeywordResponse]
    average_word_count: Optional[int] = None
    last_article_at: Optional[str] = None
    planning_guidance: List[str]


class PerformanceCsvImportRequest(BaseModel):
    """Command: Import a manual performance CSV snapshot."""

    csv_text: str = Field(..., min_length=1, max_length=200_000)
    source: str = Field("manual_csv", pattern="^manual_csv$")


class DismissOpportunityResponse(BaseModel):
    id: str
    status: str


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================


# Rulebook manager is now accessed through ProjectService


# ============================================================================
# PROJECT CRUD OPERATIONS
# ============================================================================


@router.post("", response_model=dict, status_code=status.HTTP_201_CREATED, summary="Create project")
async def create_project(
    request: CreateProjectRequest,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_superuser),
):
    """Create a new project with optional initial rulebook."""
    result = await project_service.create_project_with_rulebook(
        name=request.name,
        domain=request.domain,
        telegram_channel=request.telegram_channel,
        wordpress_url=request.wordpress_url,
        wordpress_username=request.wordpress_username,
        wordpress_app_password=request.wordpress_app_password,
        rulebook_content=request.rulebook_content,
        vertical=request.vertical,
        description=request.description,
    )
    return result


@router.get("", response_model=List[dict], summary="List projects")
async def list_projects(
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
):
    """List projects (basic metadata)."""
    projects = await project_service.list_projects(limit=limit, offset=offset)
    return [
        {
            "id": str(p.id) if hasattr(p, 'id') else str(p.get('id', '')),
            "name": p.name if hasattr(p, 'name') else p.get('name', ''),
            "domain": p.domain if hasattr(p, 'domain') else p.get('domain'),
            "description": p.description if hasattr(p, 'description') else p.get('description'),
            "wordpress_url": p.wordpress_url if hasattr(p, 'wordpress_url') else p.get('wordpress_url'),
            "wordpress_username": p.wordpress_username if hasattr(p, 'wordpress_username') else p.get('wordpress_username'),
            "vertical": p.vertical if hasattr(p, 'vertical') else p.get('vertical'),
            "created_at": p.created_at if hasattr(p, 'created_at') else p.get('created_at'),
            "last_active": p.last_active if hasattr(p, 'last_active') else p.get('last_active'),
        }
        for p in projects
    ]


@router.get("/{project_id:uuid}", response_model=dict, summary="Get project")
async def get_project(
    project_id: UUID,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
):
    """
    Get a single project by ID with full details.

    Returns complete project information including metadata.
    """
    return await project_service.get_project_with_details(project_id)


@router.get(
    "/{project_id:uuid}/readiness",
    response_model=ProjectReadinessResponse,
    summary="Get project readiness",
    description="Computes whether a project can safely generate and publish content.",
)
async def get_project_readiness(
    project_id: UUID,
    readiness_service: ProjectReadinessService = Depends(get_project_readiness_service),
    user: User = Depends(get_current_active_user),
):
    """
    Compute project readiness from current project configuration and runtime dependencies.

    The check is read-only and intentionally does not run remote WordPress validation.
    Use the WordPress test endpoint when a manager wants to validate credentials.
    """
    return await readiness_service.get_project_readiness(project_id)


@router.get(
    "/{project_id:uuid}/content-memory",
    response_model=ProjectContentMemoryResponse,
    summary="Get project content memory",
    description="Summarizes recent project content for duplicate-angle prevention.",
)
async def get_project_content_memory(
    project_id: UUID,
    limit: int = Query(12, ge=1, le=50),
    memory_service: ContentMemoryService = Depends(get_content_memory_service),
    user: User = Depends(get_current_active_user),
):
    """Return read-only content memory derived from generated article metadata."""
    return await memory_service.get_project_memory(project_id, limit=limit)


@router.get(
    "/{project_id:uuid}/performance",
    response_model=dict,
    summary="Get read-only performance feedback",
)
async def get_project_performance_feedback(
    project_id: UUID,
    service: PerformanceFeedbackService = Depends(get_performance_feedback_service),
    user: User = Depends(get_current_active_user),
):
    del user
    return await service.get_project_performance(project_id)


@router.get(
    "/{project_id:uuid}/seo-intelligence",
    response_model=dict,
    summary="Get deterministic SEO portfolio intelligence",
)
async def get_project_seo_intelligence(
    project_id: UUID,
    service: SeoIntelligenceService = Depends(get_seo_intelligence_service),
    user: User = Depends(get_current_active_user),
):
    """Return explainable, read-only prioritization from first-party metrics."""
    del user
    return await service.get_project_intelligence(project_id)


@router.post(
    "/{project_id:uuid}/performance/import-csv",
    response_model=dict,
    summary="Import manual performance CSV",
)
async def import_project_performance_csv(
    project_id: UUID,
    request: PerformanceCsvImportRequest,
    service: PerformanceFeedbackService = Depends(get_performance_feedback_service),
    user: User = Depends(get_current_active_user),
):
    if not is_manager_user(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager access is required to import performance snapshots",
        )
    return await service.import_csv(
        project_id=project_id,
        csv_text=request.csv_text,
        source=request.source,
    )


@router.post(
    "/{project_id:uuid}/performance/opportunities/{opportunity_id:uuid}/dismiss",
    response_model=DismissOpportunityResponse,
    summary="Dismiss performance opportunity",
)
async def dismiss_project_performance_opportunity(
    project_id: UUID,
    opportunity_id: UUID,
    service: PerformanceFeedbackService = Depends(get_performance_feedback_service),
    user: User = Depends(get_current_active_user),
):
    if not is_manager_user(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager access is required to dismiss performance opportunities",
        )
    return await service.dismiss_opportunity(project_id=project_id, opportunity_id=opportunity_id)


@router.put("/{project_id:uuid}", response_model=dict, summary="Update project")
async def update_project(
    project_id: UUID,
    request: UpdateProjectRequest,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_superuser),
):
    """
    Update project properties.

    Implements optimistic concurrency control to prevent conflicts.
    Only non-null fields are updated (partial update semantics).
    """
    update_data = request.model_dump(exclude_unset=True)
    return await project_service.update_project(project_id, update_data)


@router.delete("/{project_id:uuid}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete project")
async def delete_project(
    project_id: UUID,
    cascade: bool = Query(False, description="Delete associated content"),
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_superuser),
):
    """
    Delete project.

    Args:
        cascade: If true, deletes all associated content (articles, patterns, etc.)
                 If false, fails if project has associated content.
    """
    await project_service.delete_project(project_id, cascade)


@router.get(
    "/{project_id:uuid}/analytics",
    response_model=ProjectAnalyticsResponse,
    summary="Get project analytics",
)
async def get_project_analytics(
    project_id: UUID,
    start_date: Optional[datetime] = Query(None, description="Analytics period start"),
    end_date: Optional[datetime] = Query(None, description="Analytics period end"),
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
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
    "/{project_id:uuid}/rulebook",
    response_model=RulebookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update rulebook",
)
async def create_or_update_rulebook(
    project_id: UUID,
    request: RulebookRequest,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_superuser),
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
    "/{project_id:uuid}/rulebook", response_model=RulebookResponse, summary="Get project rulebook"
)
async def get_rulebook(
    project_id: UUID,
    version: Optional[int] = Query(None, description="Specific version (default: latest)"),
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
):
    """
    Retrieve project rulebook.

    Returns latest version by default, or specific version if requested.
    """
    result = await project_service.get_rulebook(project_id, version)
    return RulebookResponse(**result)


@router.get(
    "/{project_id:uuid}/rulebook/history",
    response_model=List[RulebookResponse],
    summary="Get rulebook version history",
)
async def get_rulebook_history(
    project_id: UUID,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
):
    """
    Retrieve complete rulebook version history.

    Returns all versions ordered by version number (newest first).
    """
    history = await project_service.get_rulebook_history(project_id)
    return [RulebookResponse(**rb) for rb in history]


@router.delete(
    "/{project_id:uuid}/rulebook", status_code=status.HTTP_204_NO_CONTENT, summary="Delete rulebook"
)
async def delete_rulebook(
    project_id: UUID,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_superuser),
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
    "/{project_id:uuid}/analyze",
    response_model=InferredPatternsResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger website analysis",
)
async def trigger_website_analysis(
    project_id: UUID,
    force_refresh: bool = Query(False, description="Force re-analysis"),
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_superuser),
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
    "/{project_id:uuid}/patterns",
    response_model=InferredPatternsResponse,
    summary="Get inferred patterns",
)
async def get_inferred_patterns(
    project_id: UUID,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
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
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_superuser),
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
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
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
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
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


@router.post(
    "/{project_id:uuid}/wordpress/test-connection",
    response_model=dict,
    summary="Test WordPress connection",
    description="Validates WordPress credentials and returns detailed connection status"
)
async def test_wordpress_connection(
    project_id: UUID,
    project_service: ProjectService = Depends(get_project_service),
    user: User = Depends(get_current_active_user),
):
    """
    Test WordPress connection for a project.

    Returns detailed connection status including:
    - Connection success/failure
    - Specific error category (auth, network, timeout, invalid_url)
    - Actionable error message for user
    - WordPress site details if connection successful

    This endpoint should be called:
    - Before saving WordPress credentials (validation)
    - After saving credentials (verification)
    - When troubleshooting failed article uploads
    """
    from loguru import logger

    from execution.distributer import Distributor
    from knowledge.project_repository import ProjectRepository

    try:
        # Get project with WordPress configuration
        project_repo = ProjectRepository(project_service.database_manager)
        project = await project_repo.get_by_id(project_id)

        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found"
            )

        # Check if WordPress configured
        if not project.wordpress_url:
            return {
                "status": "not_configured",
                "connected": False,
                "error_category": "configuration",
                "message": "WordPress URL not configured",
                "actionable_message": "لطفاً آدرس وردپرس، نام کاربری و رمز اپلیکیشن را وارد کنید",
                "details": None
            }

        if not project.wordpress_username:
            return {
                "status": "not_configured",
                "connected": False,
                "error_category": "configuration",
                "message": "WordPress username not configured",
                "actionable_message": "لطفاً نام کاربری وردپرس را وارد کنید",
                "details": None
            }

        if not project.wordpress_app_password:
            return {
                "status": "not_configured",
                "connected": False,
                "error_category": "configuration",
                "message": "WordPress App Password not configured",
                "actionable_message": "لطفاً رمز اپلیکیشن وردپرس را وارد کنید",
                "details": None
            }

        # Test connection using Distributor's validation
        distributor = Distributor()
        is_valid, error_message = await distributor.validate_wordpress_connection(project)

        if is_valid:
            logger.info(f"WordPress connection test successful for project {project_id}")
            return {
                "status": "success",
                "connected": True,
                "error_category": None,
                "message": "Connection successful",
                "actionable_message": f"✅ اتصال به {project.wordpress_url} برقرار است",
                "details": {
                    "wordpress_url": project.wordpress_url,
                    "wordpress_username": project.wordpress_username,
                    "api_endpoint": f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2"
                }
            }
        else:
            # Categorize error for better user guidance
            error_category = "unknown"
            actionable_message = "خطای ناشناخته در اتصال"

            if "Invalid WordPress credentials" in error_message or "401" in error_message:
                error_category = "authentication"
                actionable_message = "❌ نام کاربری یا رمز اپلیکیشن اشتباه است. لطفاً مجدداً بررسی کنید"
            elif "WordPress REST API not found" in error_message or "404" in error_message:
                error_category = "invalid_url"
                actionable_message = "❌ آدرس وردپرس نادرست است. مطمئن شوید آدرس سایت را بدون /wp-admin وارد کرده‌اید"
            elif "timeout" in error_message.lower():
                error_category = "timeout"
                actionable_message = "⏱️ زمان اتصال تمام شد. ممکن است سایت کند باشد یا دسترسی محدود شده باشد"
            elif "Cannot connect" in error_message or "Connection" in error_message:
                error_category = "network"
                actionable_message = "🌐 امکان اتصال به سایت وجود ندارد. لطفاً آدرس را بررسی کنید"

            logger.warning(f"WordPress connection test failed for project {project_id}: {error_message}")

            return {
                "status": "failed",
                "connected": False,
                "error_category": error_category,
                "message": error_message,
                "actionable_message": actionable_message,
                "details": {
                    "wordpress_url": project.wordpress_url,
                    "tested_endpoint": f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2/posts"
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing WordPress connection for project {project_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test WordPress connection. Please check configuration."
        )
