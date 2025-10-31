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
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import and_, delete, func, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import DatabaseError, NotFoundError
from core.models import InferredPatterns, Project
from infrastructure.database import DatabaseManager
from infrastructure.schema import generated_articles_table, inferred_patterns_table, projects_table


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
        Create new project in database using SQLAlchemy Core.

        Args:
            project: Project model with populated fields

        Returns:
            Project model with generated ID and timestamps

        Raises:
            DatabaseError: On constraint violations or connection issues
        """
        try:
            async with self.database_manager.session() as session:
                project_id = project.id or uuid4()
                query = (
                    insert(projects_table)
                    .values(
                        id=project_id,
                        name=project.name,
                        domain=str(project.domain) if project.domain else None,
                        telegram_channel=project.telegram_channel,
                        created_at=func.now(),
                        updated_at=func.now(),
                        last_active=func.now(),
                    )
                    .returning(projects_table)
                )

                result = await session.execute(query)
                row = result.fetchone()

                if not row:
                    raise DatabaseError("Failed to create project, no returning row.")

                created_project = self._row_to_project(row)
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
        Retrieve project by ID using SQLAlchemy Core.

        Args:
            project_id: UUID of project

        Returns:
            Project model or None if not found (or soft-deleted)
        """
        try:
            async with self.database_manager.session() as session:
                query = select(projects_table).where(
                    and_(
                        projects_table.c.id == project_id,
                        projects_table.c.deleted_at == None,
                    )
                )

                result = await session.execute(query)
                row = result.fetchone()

                if not row:
                    return None

                return self._row_to_project(row)

        except Exception as e:
            logger.error(f"Failed to retrieve project {project_id}: {e}")
            return None

    async def get_by_name(self, name: str) -> Optional[Project]:
        """Retrieve project by unique name using SQLAlchemy Core."""
        try:
            async with self.database_manager.session() as session:
                query = select(projects_table).where(
                    and_(
                        projects_table.c.name == name,
                        projects_table.c.deleted_at == None,
                    )
                )

                result = await session.execute(query)
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
        List all projects with pagination using SQLAlchemy Core.

        Args:
            limit: Maximum number of projects to return
            offset: Number of projects to skip
            include_inactive: If True, include projects with no recent activity

        Returns:
            List of Project models
        """
        try:
            async with self.database_manager.session() as session:
                query = select(projects_table).where(projects_table.c.deleted_at == None)

                if not include_inactive:
                    query = query.where(
                        (projects_table.c.last_active == None)
                        | (projects_table.c.last_active > func.now() - text("INTERVAL '90 days'"))
                    )

                query = (
                    query.order_by(projects_table.c.last_active.desc().nullslast())
                    .limit(limit)
                    .offset(offset)
                )

                result = await session.execute(query)
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
        Update project fields using SQLAlchemy Core.

        Args:
            project_id: UUID of project to update
            updates: Dict of field names to new values

        Returns:
            Updated Project model or None if not found
        """
        try:
            valid_updates = updates.copy()
            for field in [
                "id",
                "created_at",
                "total_articles_generated",
                "total_tokens_consumed",
                "total_cost_usd",
                "deleted_at",
            ]:
                valid_updates.pop(field, None)

            if not valid_updates:
                return await self.get_by_id(project_id)

            valid_updates["updated_at"] = func.now()

            async with self.database_manager.session() as session:
                query = (
                    update(projects_table)
                    .where(
                        and_(
                            projects_table.c.id == project_id,
                            projects_table.c.deleted_at == None,
                        )
                    )
                    .values(**valid_updates)
                    .returning(projects_table)
                )

                result = await session.execute(query)
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
        Update project's last_active timestamp using SQLAlchemy Core.

        Returns:
            True if updated successfully
        """
        try:
            async with self.database_manager.session() as session:
                query = (
                    update(projects_table)
                    .where(
                        and_(
                            projects_table.c.id == project_id,
                            projects_table.c.deleted_at == None,
                        )
                    )
                    .values(last_active=func.now())
                )

                result = await session.execute(query)
                return result.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to update last_active for project {project_id}: {e}")
            return False

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    async def soft_delete(self, project_id: UUID) -> bool:
        """
        Soft delete project using SQLAlchemy Core (sets deleted_at timestamp).

        Args:
            project_id: UUID of project to delete

        Returns:
            True if deleted successfully
        """
        try:
            async with self.database_manager.session() as session:
                query = (
                    update(projects_table)
                    .where(
                        and_(
                            projects_table.c.id == project_id,
                            projects_table.c.deleted_at == None,
                        )
                    )
                    .values(deleted_at=func.now())
                )

                result = await session.execute(query)

                if result.rowcount > 0:
                    logger.warning(f"Soft deleted project: {project_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"Failed to soft delete project {project_id}: {e}")
            return False

    async def hard_delete(self, project_id: UUID) -> bool:
        """
        Permanently delete project using SQLAlchemy Core (DANGEROUS).

        Cascades to all related tables via foreign key constraints.

        Returns:
            True if deleted successfully
        """
        try:
            async with self.database_manager.session() as session:
                query = delete(projects_table).where(projects_table.c.id == project_id)

                result = await session.execute(query)

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
        Get comprehensive project statistics using SQLAlchemy Core.

        Returns:
            Dict with statistics or None if project not found
        """
        try:
            async with self.database_manager.session() as session:
                p = projects_table
                ga = generated_articles_table

                query = (
                    select(
                        p.c.total_articles_generated,
                        p.c.total_tokens_consumed,
                        p.c.total_cost_usd,
                        func.count(ga.c.id).label("articles_count"),
                        func.avg(ga.c.word_count).label("avg_word_count"),
                        func.avg(ga.c.readability_score).label("avg_readability"),
                        func.sum(ga.c.total_tokens_used).label("verified_tokens"),
                        func.sum(ga.c.total_cost).label("verified_cost"),
                    )
                    .select_from(p.outerjoin(ga, ga.c.project_id == p.c.id))
                    .where(
                        and_(
                            p.c.id == project_id,
                            p.c.deleted_at == None,
                        )
                    )
                    .group_by(
                        p.c.id,
                        p.c.total_articles_generated,
                        p.c.total_tokens_consumed,
                        p.c.total_cost_usd,
                    )
                )

                result = await session.execute(query)
                row = result.fetchone()

                if not row:
                    return None

                return {
                    "total_articles": row.total_articles_generated,
                    "total_tokens": row.total_tokens_consumed,
                    "total_cost_usd": float(row.total_cost_usd) if row.total_cost_usd else 0.0,
                    "verified_articles": row.articles_count or 0,
                    "avg_word_count": float(row.avg_word_count) if row.avg_word_count else 0.0,
                    "avg_readability": float(row.avg_readability) if row.avg_readability else 0.0,
                    "verified_tokens": row.verified_tokens or 0,
                    "verified_cost_usd": float(row.verified_cost) if row.verified_cost else 0.0,
                }

        except Exception as e:
            logger.error(f"Failed to get statistics for project {project_id}: {e}")
            return None

    async def get_global_statistics(self) -> dict:
        """Get system-wide statistics across all projects using SQLAlchemy Core."""
        try:
            async with self.database_manager.session() as session:
                query = (
                    select(
                        func.count(projects_table.c.id).label("total_projects"),
                        func.sum(projects_table.c.total_articles_generated).label("total_articles"),
                        func.sum(projects_table.c.total_tokens_consumed).label("total_tokens"),
                        func.sum(projects_table.c.total_cost_usd).label("total_cost"),
                        func.avg(projects_table.c.total_articles_generated).label(
                            "avg_articles_per_project"
                        ),
                    )
                    .where(projects_table.c.deleted_at == None)
                )

                result = await session.execute(query)
                row = result.fetchone()

                return {
                    "total_projects": row.total_projects or 0,
                    "total_articles": row.total_articles or 0,
                    "total_tokens": row.total_tokens or 0,
                    "total_cost_usd": float(row.total_cost) if row.total_cost else 0.0,
                    "avg_articles_per_project": float(row.avg_articles_per_project)
                    if row.avg_articles_per_project
                    else 0.0,
                }

        except Exception as e:
            logger.error(f"Failed to get global statistics: {e}")
            return {}

    # =========================================================================
    # INFERRED PATTERNS OPERATIONS
    # =========================================================================

    async def save_inferred_patterns(self, project_id: UUID, patterns: Dict) -> InferredPatterns:
        """
        Save inferred patterns for a project.

        Args:
            project_id: UUID of project
            patterns: Aggregated pattern data

        Returns:
            InferredPatterns model
        """
        try:
            async with self.database_manager.session() as session:
                # Delete existing patterns for project
                await session.execute(
                    text("DELETE FROM inferred_patterns WHERE project_id = :project_id"),
                    {"project_id": project_id},
                )

                # Insert new patterns
                query = """
                    INSERT INTO inferred_patterns (
                        id, project_id, avg_sentence_length, sentence_length_std,
                        lexical_diversity, readability_score, tone_embedding,
                        structure_patterns, confidence, sample_size, analyzed_at
                    ) VALUES (
                        :id, :project_id, :avg_sentence_length, :sentence_length_std,
                        :lexical_diversity, :readability_score, :tone_embedding,
                        :structure_patterns, :confidence, :sample_size, NOW()
                    )
                    RETURNING id, project_id, avg_sentence_length, sentence_length_std,
                              lexical_diversity, readability_score, confidence, 
                              sample_size, analyzed_at;
                """

                pattern_id = uuid4()

                # Convert structure patterns to JSON
                structure_json = [
                    {
                        "pattern_type": p.pattern_type,
                        "frequency": p.frequency,
                        "typical_sections": p.typical_sections,
                        "avg_word_count": p.avg_word_count,
                    }
                    for p in patterns["structure_patterns"]
                ]

                result = await session.execute(
                    query,
                    {
                        "id": pattern_id,
                        "project_id": project_id,
                        "avg_sentence_length": patterns["avg_sentence_length"],
                        "sentence_length_std": patterns["sentence_length_std"],
                        "lexical_diversity": patterns["lexical_diversity"],
                        "readability_score": patterns["readability_score"],
                        "tone_embedding": patterns["tone_embedding"],
                        "structure_patterns": str(structure_json).replace("'", '"'),
                        "confidence": patterns["confidence"],
                        "sample_size": patterns["sample_size"],
                    },
                )

                await session.commit()

                row = result.fetchone()

                inferred = InferredPatterns(
                    id=row[0],
                    project_id=row[1],
                    avg_sentence_length=row[2],
                    sentence_length_std=row[3],
                    lexical_diversity=row[4],
                    readability_score=row[5],
                    tone_embedding=patterns["tone_embedding"],
                    structure_patterns=patterns["structure_patterns"],
                    confidence=row[6],
                    sample_size=row[7],
                    analyzed_at=row[8],
                )

                logger.info(f"Stored inferred patterns (confidence: {patterns['confidence']:.2f})")
                return inferred

        except Exception as e:
            logger.error(f"Failed to store patterns: {e}")
            raise DatabaseError(f"Pattern storage failed: {e}")

    async def get_inferred_patterns(self, project_id: UUID) -> Optional[InferredPatterns]:
        """
        Retrieve existing patterns if recent (< 30 days).

        Args:
            project_id: UUID of project

        Returns:
            InferredPatterns model or None if not found/recent
        """
        try:
            async with self.database_manager.session() as session:
                query = """
                    SELECT id, project_id, avg_sentence_length, sentence_length_std,
                           lexical_diversity, readability_score, tone_embedding, confidence, 
                           sample_size, analyzed_at
                    FROM inferred_patterns
                    WHERE project_id = :project_id
                    AND analyzed_at > NOW() - INTERVAL '30 days'
                    ORDER BY analyzed_at DESC
                    LIMIT 1;
                """

                result = await session.execute(text(query), {"project_id": project_id})
                row = result.fetchone()

                if not row:
                    return None

                # Note: Not loading full structure_patterns for efficiency
                return InferredPatterns(
                    id=row[0],
                    project_id=row[1],
                    avg_sentence_length=row[2],
                    sentence_length_std=row[3],
                    lexical_diversity=row[4],
                    readability_score=row[5],
                    tone_embedding=row[6],
                    structure_patterns=[],
                    confidence=row[7],
                    sample_size=row[8],
                    analyzed_at=row[9],
                )

        except Exception as e:
            logger.error(f"Failed to get inferred patterns: {e}")
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def _row_to_project(row) -> Project:
        """Convert database row to Project model."""
        return Project(
            id=row.id,
            name=row.name,
            domain=row.domain,
            telegram_channel=row.telegram_channel,
            created_at=row.created_at,
            updated_at=row.updated_at,
            last_active=row.last_active or row.created_at,  # Use created_at if last_active is None
            total_articles_generated=row.total_articles_generated,
            total_tokens_consumed=row.total_tokens_consumed,
            total_cost_usd=float(row.total_cost_usd) if row.total_cost_usd else 0.0,
        )

    async def exists(self, project_id: UUID) -> bool:
        """Check if project exists (and not soft-deleted) using SQLAlchemy Core."""
        try:
            async with self.database_manager.session() as session:
                query = select(1).where(
                    and_(
                        projects_table.c.id == project_id,
                        projects_table.c.deleted_at == None,
                    )
                )
                result = await session.execute(query)
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check project existence: {e}")
            return False
