"""
Project Service: Business Logic Layer for Project Management

Encapsulates all business logic for project operations including:
- Project CRUD operations with validation
- Rulebook management and versioning
- Pattern analysis and inference
- Analytics and reporting
- Bulk operations

Design Pattern: Service Layer with Repository Pattern
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from pydantic import SecretStr

from core.exceptions import ProjectNotFoundError
from core.models import InferredPatterns, Project, Rulebook
from infrastructure.database import DatabaseManager
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager


class ProjectService:
    """
    Service layer for project business logic.

    Provides high-level business operations for project management,
    abstracting away implementation details from the API layer.
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        semantic_analyzer: SemanticAnalyzer,
        project_repository: Optional[ProjectRepository] = None,
    ):
        """
        Initialize service with database manager and dependencies.

        Args:
            database_manager: Database manager for session management
            semantic_analyzer: Analyzer for rulebook processing
        """
        self.database_manager = database_manager
        self.semantic_analyzer = semantic_analyzer
        logger.debug("ProjectService initialized")
        # Lazily used repositories/managers are created per call to ensure fresh sessions
        # Provide a default repository for common operations
        self.projects = project_repository or ProjectRepository(self.database_manager)

    async def create_project_with_rulebook(
        self,
        name: str,
        domain: Optional[str] = None,
        telegram_channel: Optional[str] = None,
        wordpress_url: Optional[str] = None,
        wordpress_username: Optional[str] = None,
        wordpress_app_password: Optional[str] = None,
        rulebook_content: Optional[str] = None,
        vertical: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new project with optional rulebook in a single atomic transaction.

        This method ensures that both project and rulebook creation happen atomically,
        preventing inconsistent states if one operation fails.

        Args:
            name: Project name
            domain: Optional project domain
            telegram_channel: Optional Telegram channel for distribution
            rulebook_content: Optional initial rulebook content
            vertical: Optional business vertical
            description: Optional project description

        Returns:
            Project creation result with metadata

        Raises:
            HTTPException: If project creation fails
        """
        from fastapi import HTTPException

        try:
            async with self.database_manager.session() as session:
                # Create project object
                project = Project(
                    name=name,
                    domain=domain,
                    vertical=vertical,
                    description=description,
                    telegram_channel=telegram_channel,
                    wordpress_url=wordpress_url,
                    wordpress_username=wordpress_username,
                    wordpress_app_password=SecretStr(wordpress_app_password)
                    if wordpress_app_password
                    else None,
                )

                # Create repositories with database manager
                project_repo = ProjectRepository(self.database_manager)

                rulebook_manager = RulebookManager(session, self.semantic_analyzer)

                # Save project to database
                created_project = await project_repo.create(project)

                # Create rulebook if content provided
                has_rulebook = False
                if rulebook_content:
                    await rulebook_manager.create_rulebook(
                        project_id=created_project.id, raw_content=rulebook_content
                    )
                    has_rulebook = True

                return {
                    "id": str(created_project.id),
                    "name": created_project.name,
                    "domain": created_project.domain,
                    "telegram_channel": created_project.telegram_channel,
                    "description": created_project.description,
                    "vertical": created_project.vertical,
                    "created_at": created_project.created_at,
                    "total_articles_generated": 0,
                    "has_rulebook": has_rulebook,
                    "has_inferred_patterns": False,
                }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create project: {str(e)}",
            )

    async def get_project_with_details(self, project_id: UUID) -> Dict[str, Any]:
        """
        Get project details including rulebook and pattern status.

        Args:
            project_id: Project identifier

        Returns:
            Project details with metadata

        Raises:
            HTTPException: If project not found
        """
        from fastapi import HTTPException

        # Get project
        async with self.database_manager.session() as session:
            project_repo = ProjectRepository(self.database_manager)
            project = await project_repo.get_by_id(project_id)
            if not project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=f"Project not found: {project_id}"
                )

            # Check for rulebook and patterns
            rulebook_manager = RulebookManager(session, self.semantic_analyzer)
            rulebook = await rulebook_manager.get_latest_rulebook(project_id)

            # Get inferred patterns from repository
            patterns = None
            try:
                patterns = await project_repo.get_inferred_patterns(project_id)
            except Exception as e:
                logger.warning(f"Failed to get inferred patterns: {e}")
                pass

            return {
                "id": str(project.id),
                "name": project.name,
                "domain": project.domain,
                "telegram_channel": project.telegram_channel,
                "description": project.description,
                "vertical": project.vertical,
                "wordpress_url": project.wordpress_url,
                "wordpress_username": project.wordpress_username,
                "created_at": project.created_at,
                "total_articles_generated": project.total_articles_generated,
                "has_rulebook": rulebook is not None,
                "has_inferred_patterns": patterns is not None
                and hasattr(patterns, "confidence")
                and patterns.confidence > 0.65,
            }

    async def list_projects(
        self, limit: int = 100, offset: int = 0
    ) -> List[Project]:
        """
        List all projects with pagination.

        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip

        Returns:
            List of Project objects
        """
        return await self.projects.list_all(limit=limit, offset=offset)

    async def update_project(self, project_id: UUID, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update project with business logic validation.

        Args:
            project_id: Project identifier
            update_data: Fields to update

        Returns:
            Update result with metadata

        Raises:
            ProjectNotFoundError: If project doesn't exist
        """
        # Verify project exists
        project = await self.projects.get_by_id(project_id)
        if not project:
            raise ProjectNotFoundError(f"Project not found: {project_id}")

        # Apply updates
        updated_project = await self.projects.update(project_id, update_data)
        if not updated_project:
            raise ProjectNotFoundError(f"Project not found: {project_id}")

        return {
            "id": str(updated_project.id),
            "project": {
                "id": str(updated_project.id),
                "name": updated_project.name,
                "domain": updated_project.domain,
                "description": updated_project.description,
                "vertical": updated_project.vertical,
                "wordpress_url": updated_project.wordpress_url,
                "wordpress_username": updated_project.wordpress_username,
            },
            "message": "Project updated successfully",
            "updated_fields": list(update_data.keys()),
        }

    async def delete_project(self, project_id: UUID, cascade: bool = False) -> None:
        """
        Delete project with business logic validation.

        Args:
            project_id: Project identifier
            cascade: Whether to delete associated content

        Raises:
            ProjectNotFoundError: If project doesn't exist
            HTTPException: If project has content and cascade=False
        """
        # Verify project exists
        project = await self.projects.get_by_id(project_id)
        if not project:
            raise ProjectNotFoundError(f"Project not found: {project_id}")

        impact = await self.projects.get_deletion_impact(project_id)
        active_tasks = impact.get("active_tasks", 0)
        if active_tasks > 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Project has {active_tasks} active task(s). Wait for them to finish "
                    "or cancel them before deleting the project."
                ),
            )

        dependent_count = sum(
            impact.get(key, 0)
            for key in ("articles", "content_plans", "rulebooks", "inferred_patterns")
        )
        if dependent_count > 0 and not cascade:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Project has associated content "
                    f"({impact.get('articles', 0)} articles, "
                    f"{impact.get('content_plans', 0)} plans, "
                    f"{impact.get('rulebooks', 0)} rulebooks, "
                    f"{impact.get('inferred_patterns', 0)} inferred patterns). "
                    "Confirm permanent deletion to remove all related records."
                ),
            )

        if cascade:
            deleted = await self.projects.hard_delete(project_id)
        else:
            deleted = await self.projects.soft_delete(project_id)

        if not deleted:
            raise ProjectNotFoundError(f"Project not found: {project_id}")

    async def get_project_analytics(
        self,
        project_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get project analytics with business logic.

        Args:
            project_id: Project identifier
            start_date: Analytics period start
            end_date: Analytics period end

        Returns:
            Analytics data

        Raises:
            ProjectNotFoundError: If project doesn't exist
        """
        # Verify project exists
        project_repo = ProjectRepository(self.database_manager)
        project = await project_repo.get_by_id(project_id)
        if not project:
            raise ProjectNotFoundError(f"Project not found: {project_id}")

        # Get analytics data
        start = start_date or datetime(2020, 1, 1, tzinfo=timezone.utc)
        end = end_date or datetime.now(timezone.utc)

        stats = await project_repo.get_statistics(project_id)

        return {
            "project_id": str(project_id),
            "total_articles": stats.get("total_articles", 0),
            "total_cost": stats.get("total_cost_usd", 0.0),
            "avg_generation_time": 0.0,  # Not tracked; placeholder
            "avg_readability": stats.get("avg_readability", 0.0),
            "avg_word_count": int(stats.get("avg_word_count", 0.0)),
            "period_start": start,
            "period_end": end,
        }

    async def create_or_update_rulebook(
        self, project_id: UUID, content: str, version_note: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create or update project rulebook with business logic.

        Args:
            project_id: Project identifier
            content: Rulebook content
            version_note: Optional version change notes

        Returns:
            Rulebook data with metadata

        Raises:
            ProjectNotFoundError: If project doesn't exist
        """
        # Verify project exists
        project = await self.projects.get_by_id(project_id)
        if not project:
            raise ProjectNotFoundError(f"Project not found: {project_id}")

        # Check if rulebook exists
        async with self.database_manager.session() as session:
            rulebook_manager = RulebookManager(session, self.semantic_analyzer)
            existing_rulebook = await rulebook_manager.get_latest_rulebook(project_id)

        if existing_rulebook:
            # Update existing
            async with self.database_manager.session() as session:
                rulebook_manager = RulebookManager(session, self.semantic_analyzer)
                rulebook = await rulebook_manager.create_rulebook(
                    project_id=project_id, raw_content=content
                )
            status_message = "updated"
        else:
            # Create new
            async with self.database_manager.session() as session:
                rulebook_manager = RulebookManager(session, self.semantic_analyzer)
                rulebook = await rulebook_manager.create_rulebook(
                    project_id=project_id, raw_content=content
                )
            status_message = "created"

        return {
            "id": str(rulebook.id),
            "project_id": str(rulebook.project_id),
            "content": rulebook.raw_content,
            "version": rulebook.version,
            "rule_count": len(rulebook.rules),
            "updated_at": rulebook.updated_at,
            "status": status_message,
        }

    async def get_rulebook(self, project_id: UUID, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Get project rulebook with business logic.

        Args:
            project_id: Project identifier
            version: Specific version (default: latest)

        Returns:
            Rulebook data

        Raises:
            HTTPException: If rulebook not found
        """
        from fastapi import HTTPException

        async with self.database_manager.session() as session:
            rulebook_manager = RulebookManager(session, self.semantic_analyzer)
            if version:
                rulebook = await self._get_rulebook_by_version(
                    session, rulebook_manager, project_id, version
                )
            else:
                rulebook = await rulebook_manager.get_latest_rulebook(project_id)

        if not rulebook:
            raise HTTPException(status_code=404, detail="Rulebook not found")

        return {
            "id": str(rulebook.id),
            "project_id": str(rulebook.project_id),
            "content": rulebook.raw_content,
            "version": rulebook.version,
            "rule_count": len(rulebook.rules),
            "updated_at": rulebook.updated_at,
        }

    async def _get_rulebook_by_version(
        self, session, rulebook_manager: "RulebookManager", project_id: UUID, version: int
    ):
        """
        Retrieve a specific rulebook version for a project.

        Args:
            session: Active database session
            rulebook_manager: RulebookManager instance
            project_id: Project identifier
            version: Specific version number to retrieve

        Returns:
            Rulebook instance or None if not found
        """
        from sqlalchemy import select
        from infrastructure.schema import rulebooks_table
        from core.models import Rulebook

        query = (
            select(rulebooks_table)
            .where(rulebooks_table.c.project_id == project_id)
            .where(rulebooks_table.c.version == version)
            .limit(1)
        )
        result = await session.execute(query)
        row = result.fetchone()

        if not row:
            return None

        # Load associated rules via the manager
        rules = await rulebook_manager._load_rules(row.id)

        return Rulebook(
            id=row.id,
            project_id=row.project_id,
            raw_content=row.content,
            version=row.version,
            created_at=row.created_at,
            updated_at=row.updated_at,
            rules=rules,
        )

    async def get_rulebook_history(self, project_id: UUID) -> List[Dict[str, Any]]:
        """
        Get rulebook version history.

        Args:
            project_id: Project identifier

        Returns:
            List of rulebook versions
        """
        async with self.database_manager.session() as session:
            rulebook_manager = RulebookManager(session, self.semantic_analyzer)
            history = await rulebook_manager.get_rulebook_history(project_id)

        return [
            {
                "id": str(rb.id),
                "project_id": str(rb.project_id),
                "content": rb.raw_content,
                "version": rb.version,
                "rule_count": len(rb.rules),
                "updated_at": rb.updated_at,
            }
            for rb in history
        ]

    async def delete_rulebook(self, project_id: UUID) -> None:
        """
        Delete project rulebook.

        Args:
            project_id: Project identifier
        """
        async with self.database_manager.session() as session:
            rulebook_manager = RulebookManager(session, self.semantic_analyzer)
            await rulebook_manager.delete_rulebook(project_id)

    async def trigger_website_analysis(
        self, project_id: UUID, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Trigger website analysis with business logic validation.

        Args:
            project_id: Project identifier
            force_refresh: Force re-analysis

        Returns:
            Analysis result or task information

        Raises:
            ProjectNotFoundError: If project doesn't exist
            HTTPException: If no domain configured
        """
        from fastapi import HTTPException

        # Verify project exists
        project_repo = ProjectRepository(self.database_manager)
        project = await project_repo.get_by_id(project_id)
        if not project:
            raise ProjectNotFoundError(f"Project not found: {project_id}")

        if not project.domain:
            raise HTTPException(
                status_code=400,
                detail="Project has no domain configured for analysis",
            )

        # Check for existing patterns
        project_repo = ProjectRepository(self.database_manager)
        existing_patterns = await project_repo.get_inferred_patterns(project_id)

        if existing_patterns and not force_refresh:
            # Return existing patterns
            return {
                "id": str(existing_patterns.id),
                "project_id": str(existing_patterns.project_id),
                "avg_sentence_length": existing_patterns.avg_sentence_length[0],
                "lexical_diversity": existing_patterns.lexical_diversity,
                "readability_score": existing_patterns.readability_score,
                "confidence": existing_patterns.confidence,
                "sample_size": existing_patterns.sample_size,
                "analyzed_at": existing_patterns.analyzed_at,
            }

        # Trigger async analysis
        from orchestration.celery_app import app

        task = app.send_task(
            "content_automation.analyze_website", args=[str(project_id), project.domain]
        )

        return {
            "message": "Website analysis initiated",
            "task_id": task.id,
            "project_id": str(project_id),
        }

    async def get_inferred_patterns(self, project_id: UUID) -> Dict[str, Any]:
        """
        Get inferred patterns with business logic.

        Args:
            project_id: Project identifier

        Returns:
            Patterns data

        Raises:
            HTTPException: If no patterns found
        """
        from fastapi import HTTPException

        project_repo = ProjectRepository(self.database_manager)
        patterns = await project_repo.get_inferred_patterns(project_id)

        if not patterns:
            raise HTTPException(
                status_code=404,
                detail="No inferred patterns found. Trigger analysis first.",
            )

        return {
            "id": str(patterns.id),
            "project_id": str(patterns.project_id),
            "avg_sentence_length": patterns.avg_sentence_length[0],
            "lexical_diversity": patterns.lexical_diversity,
            "readability_score": patterns.readability_score,
            "confidence": patterns.confidence,
            "sample_size": patterns.sample_size,
            "analyzed_at": patterns.analyzed_at,
        }

    async def bulk_create_projects(self, projects_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bulk create projects with error handling.

        PERFORMANCE FIX: Uses single bulk INSERT instead of N individual INSERTs,
        reducing database round trips from O(N) to O(1).

        Args:
            projects_data: List of project data dictionaries

        Returns:
            Bulk operation result with successful/failed project IDs
        """
        from core.models import Project

        successful = []
        failed = []
        valid_projects = []

        # Phase 1: Validate and convert to Project models
        for idx, project_data in enumerate(projects_data):
            try:
                project = Project(**project_data)
                valid_projects.append((idx, project))
            except Exception as e:
                failed.append({"index": idx, "data": project_data, "error": str(e)})

        # Phase 2: Bulk insert all valid projects in single database operation
        if valid_projects:
            try:
                projects_to_create = [proj for _, proj in valid_projects]
                created_projects = await self.projects.bulk_create(projects_to_create)
                successful = [str(p.id) for p in created_projects]
            except Exception as e:
                # If bulk insert fails entirely, fall back to individual handling
                logger.error(f"Bulk insert failed, falling back to individual inserts: {e}")
                for idx, project in valid_projects:
                    try:
                        created = await self.projects.create(project)
                        successful.append(str(created.id))
                    except Exception as individual_error:
                        failed.append({
                            "index": idx,
                            "data": projects_data[idx],
                            "error": str(individual_error)
                        })

        return {"successful": successful, "failed": failed, "total_processed": len(projects_data)}

    async def search_projects(
        self, query: str, field: str = "name", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search projects with business logic.

        Args:
            query: Search query
            field: Field to search
            limit: Maximum results

        Returns:
            Search results with relevance scores
        """
        results = await self.projects.search(query=query, field=field, limit=limit)

        return [
            {
                "id": str(p.id),
                "name": p.name,
                "domain": p.domain,
                "match_score": getattr(p, "match_score", 1.0),  # Relevance score
            }
            for p in results
        ]

    async def filter_projects(self, **filters) -> List[Dict[str, Any]]:
        """
        Filter projects with business logic.

        Args:
            **filters: Filter criteria

        Returns:
            Filtered project results
        """
        # Remove None values
        clean_filters = {k: v for k, v in filters.items() if v is not None}

        results = await self.projects.filter(**clean_filters)

        return [
            {
                "id": str(p.id),
                "name": p.name,
                "domain": p.domain,
                "total_articles": getattr(p, "total_articles_generated", 0),
                "created_at": p.created_at,
            }
            for p in results
        ]
