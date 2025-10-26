"""
Project Repository - Data Access Layer
========================================

Implements repository pattern for Project entities with:
- CRUD operations with soft delete support
- Statistics aggregation and reporting
- Transaction-safe batch operations
- Type-safe query builders

Design: Single Responsibility - all project data access goes through this layer.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import and_, delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import DatabaseError, NotFoundError
from core.models import Project
from infrastructure.database import DatabaseManager


class ProjectRepository:
    """
    Repository for Project entity operations.

    Provides abstraction over database operations with:
    - Type safety via Pydantic models
    - Transaction management
    - Soft delete support
    - Query optimization
    """

    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize repository with database manager.

        Args:
            database_manager: Database manager for session management
        """
        self.database_manager = database_manager

    # =========================================================================
    # CREATE OPERATIONS
    # =========================================================================

    async def create(self, project: Project) -> Project:
        """
        Create new project in database.

        Args:
            project: Project model with populated fields

        Returns:
            Project model with generated ID and timestamps

        Raises:
            DatabaseError: On constraint violations or connection issues
        """
        try:
            async with self.database_manager.session() as session:
                # Convert Pydantic model to SQL insert
                query = """
                    INSERT INTO projects (
                        name, domain, telegram_channel, created_at, updated_at
                    ) VALUES (
                        :name, :domain, :telegram_channel, NOW(), NOW()
                    )
                    RETURNING id, name, domain, telegram_channel, created_at, 
                              updated_at, last_active, total_articles_generated,
                              total_tokens_consumed, total_cost_usd;
                """

                result = await session.execute(
                    text(query),
                    {
                        "name": project.name,
                        "domain": str(project.domain) if project.domain else None,
                        "telegram_channel": project.telegram_channel,
                    },
                )

                row = result.fetchone()

                # Map back to Pydantic model
                created_project = Project(
                    id=row[0],
                    name=row[1],
                    domain=row[2],
                    telegram_channel=row[3],
                    created_at=row[4],
                    updated_at=row[5],
                    last_active=row[6] or row[4],  # Use created_at if last_active is None
                    total_articles_generated=row[7],
                    total_tokens_consumed=row[8],
                    total_cost_usd=float(row[9]) if row[9] else 0.0,
                )

                logger.info(f"Created project: {created_project.name} (ID: {created_project.id})")
                return created_project

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise DatabaseError(f"Project creation failed: {e}")

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    async def get_by_id(self, project_id: UUID) -> Optional[Project]:
        """
        Retrieve project by ID.

        Args:
            project_id: UUID of project

        Returns:
            Project model or None if not found (or soft-deleted)
        """
        try:
            async with self.database_manager.session() as session:
                query = """
                    SELECT id, name, domain, telegram_channel, created_at, updated_at,
                           last_active, total_articles_generated, total_tokens_consumed,
                           total_cost_usd
                    FROM projects
                    WHERE id = :project_id AND deleted_at IS NULL;
                """

                result = await session.execute(text(query), {"project_id": project_id})
                row = result.fetchone()

                if not row:
                    return None

                return self._row_to_project(row)

        except Exception as e:
            logger.error(f"Failed to retrieve project {project_id}: {e}")
            return None

    async def get_by_name(self, name: str) -> Optional[Project]:
        """Retrieve project by unique name."""
        try:
            async with self.database_manager.session() as session:
                query = """
                    SELECT id, name, domain, telegram_channel, created_at, updated_at,
                           last_active, total_articles_generated, total_tokens_consumed,
                           total_cost_usd
                    FROM projects
                    WHERE name = :name AND deleted_at IS NULL;
                """

                result = await session.execute(text(query), {"name": name})
                row = result.fetchone()

                if not row:
                    return None

                return self._row_to_project(row)

        except Exception as e:
            logger.error(f"Failed to retrieve project by name '{name}': {e}")
            return None

    async def list_all(
        self, limit: int = 100, offset: int = 0, include_inactive: bool = False
    ) -> List[Project]:
        """
        List all projects with pagination.

        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip
            include_inactive: If True, include projects with no recent activity

        Returns:
            List of Project models
        """
        try:
            async with self.database_manager.session() as session:
                query = """
                    SELECT id, name, domain, telegram_channel, created_at, updated_at,
                           last_active, total_articles_generated, total_tokens_consumed,
                           total_cost_usd
                    FROM projects
                    WHERE deleted_at IS NULL
                """

                if not include_inactive:
                    query += (
                        " AND (last_active IS NULL OR last_active > NOW() - INTERVAL '90 days')"
                    )

                query += " ORDER BY last_active DESC NULLS LAST LIMIT :limit OFFSET :offset;"

                result = await session.execute(text(query), {"limit": limit, "offset": offset})
                rows = result.fetchall()

            return [self._row_to_project(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================

    async def update(self, project_id: UUID, updates: dict) -> Optional[Project]:
        """
        Update project fields.

        Args:
            project_id: UUID of project to update
            updates: Dict of field names to new values

        Returns:
            Updated Project model or None if not found
        """
        try:
            # Build dynamic UPDATE query
            set_clauses = []
            params = {"project_id": project_id}

            for field, value in updates.items():
                if field not in [
                    "id",
                    "created_at",
                    "total_articles_generated",
                    "total_tokens_consumed",
                    "total_cost_usd",
                ]:
                    set_clauses.append(f"{field} = :{field}")
                    params[field] = value

            if not set_clauses:
                return await self.get_by_id(project_id)

            set_clauses.append("updated_at = NOW()")

            query = f"""
                UPDATE projects
                SET {', '.join(set_clauses)}
                WHERE id = :project_id AND deleted_at IS NULL
                RETURNING id, name, domain, telegram_channel, created_at, updated_at,
                          last_active, total_articles_generated, total_tokens_consumed,
                          total_cost_usd;
            """

            async with self.database_manager.session() as session:
                result = await session.execute(text(query), params)
                row = result.fetchone()

                if not row:
                    return None

                logger.info(f"Updated project {project_id}: {list(updates.keys())}")
                return self._row_to_project(row)

        except Exception as e:
            logger.error(f"Failed to update project {project_id}: {e}")
            return None

    async def update_last_active(self, project_id: UUID) -> bool:
        """
        Update project's last_active timestamp (lightweight operation).

        Returns:
            True if updated successfully
        """
        try:
            async with self.database_manager.session() as session:
                query = """
                    UPDATE projects
                    SET last_active = NOW()
                    WHERE id = :project_id AND deleted_at IS NULL;
                """

                result = await session.execute(text(query), {"project_id": project_id})
                return result.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to update last_active for project {project_id}: {e}")
            return False

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    async def soft_delete(self, project_id: UUID) -> bool:
        """
        Soft delete project (sets deleted_at timestamp).

        Args:
            project_id: UUID of project to delete

        Returns:
            True if deleted successfully
        """
        try:
            async with self.database_manager.session() as session:
                query = """
                    UPDATE projects
                    SET deleted_at = NOW()
                    WHERE id = :project_id AND deleted_at IS NULL;
                """

                result = await session.execute(text(query), {"project_id": project_id})

                if result.rowcount > 0:
                    logger.warning(f"Soft deleted project: {project_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to soft delete project {project_id}: {e}")
            return False

    async def hard_delete(self, project_id: UUID) -> bool:
        """
        Permanently delete project and all associated data (DANGEROUS).

        Cascades to all related tables via foreign key constraints.

        Returns:
            True if deleted successfully
        """
        try:
            async with self.database_manager.session() as session:
                query = "DELETE FROM projects WHERE id = :project_id;"

                result = await session.execute(text(query), {"project_id": project_id})

                if result.rowcount > 0:
                    logger.warning(f"HARD DELETED project: {project_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to hard delete project {project_id}: {e}")
            return False

    # =========================================================================
    # STATISTICS & AGGREGATION
    # =========================================================================

    async def get_statistics(self, project_id: UUID) -> Optional[dict]:
        """
        Get comprehensive project statistics.

        Returns:
            Dict with statistics or None if project not found
        """
        try:
            async with self.database_manager.session() as session:
                query = """
                    SELECT 
                        p.total_articles_generated,
                        p.total_tokens_consumed,
                        p.total_cost_usd,
                        COUNT(DISTINCT ga.id) as articles_count,
                        AVG(ga.word_count) as avg_word_count,
                        AVG(ga.readability_score) as avg_readability,
                        SUM(ga.total_tokens_used) as verified_tokens,
                        SUM(ga.total_cost) as verified_cost
                    FROM projects p
                    LEFT JOIN generated_articles ga ON ga.project_id = p.id
                    WHERE p.id = :project_id AND p.deleted_at IS NULL
                    GROUP BY p.id, p.total_articles_generated, p.total_tokens_consumed, p.total_cost_usd;
                """

                result = await session.execute(text(query), {"project_id": project_id})
                row = result.fetchone()

                if not row:
                    return None

                return {
                    "total_articles": row[0],
                    "total_tokens": row[1],
                    "total_cost_usd": float(row[2]) if row[2] else 0.0,
                    "verified_articles": row[3] or 0,
                    "avg_word_count": float(row[4]) if row[4] else 0.0,
                    "avg_readability": float(row[5]) if row[5] else 0.0,
                    "verified_tokens": row[6] or 0,
                    "verified_cost_usd": float(row[7]) if row[7] else 0.0,
                }

        except Exception as e:
            logger.error(f"Failed to get statistics for project {project_id}: {e}")
            return None

    async def get_global_statistics(self) -> dict:
        """Get system-wide statistics across all projects."""
        try:
            async with self.database_manager.session() as session:
                query = """
                    SELECT 
                        COUNT(DISTINCT id) as total_projects,
                        SUM(total_articles_generated) as total_articles,
                        SUM(total_tokens_consumed) as total_tokens,
                        SUM(total_cost_usd) as total_cost,
                        AVG(total_articles_generated) as avg_articles_per_project
                    FROM projects
                    WHERE deleted_at IS NULL;
                """

                result = await session.execute(text(query))
                row = result.fetchone()

                return {
                    "total_projects": row[0] or 0,
                    "total_articles": row[1] or 0,
                    "total_tokens": row[2] or 0,
                    "total_cost_usd": float(row[3]) if row[3] else 0.0,
                    "avg_articles_per_project": float(row[4]) if row[4] else 0.0,
                }

        except Exception as e:
            logger.error(f"Failed to get global statistics: {e}")
            return {}

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _row_to_project(row) -> Project:
        """Convert database row to Project model."""
        return Project(
            id=row[0],
            name=row[1],
            domain=row[2],
            telegram_channel=row[3],
            created_at=row[4],
            updated_at=row[5],
            last_active=row[6] or row[4],  # Use created_at if last_active is None
            total_articles_generated=row[7],
            total_tokens_consumed=row[8],
            total_cost_usd=float(row[9]) if row[9] else 0.0,
        )

    async def exists(self, project_id: UUID) -> bool:
        """Check if project exists (and not soft-deleted)."""
        try:
            async with self.database_manager.session() as session:
                query = "SELECT 1 FROM projects WHERE id = :project_id AND deleted_at IS NULL;"
                result = await session.execute(text(query), {"project_id": project_id})
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check project existence: {e}")
            return False
