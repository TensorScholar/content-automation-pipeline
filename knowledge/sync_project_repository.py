"""
Synchronous Project Repository for Celery Workers
==================================================

Provides synchronous database access for project data to avoid
async event loop conflicts in Celery workers.

Design: Mirror of ProjectRepository but using sync database manager.
"""

from typing import Optional
from uuid import UUID

from loguru import logger

from core.models import Project
from infrastructure.sync_database import get_sync_db


class SyncProjectRepository:
    """
    Synchronous repository for Project entity operations.

    Used exclusively in Celery workers to avoid async event loop conflicts.
    """

    def __init__(self):
        """Initialize with sync database manager."""
        self.db = get_sync_db()

    def get_by_id(self, project_id: UUID) -> Optional[Project]:
        """
        Retrieve project by ID using synchronous database access.

        Args:
            project_id: UUID of project

        Returns:
            Project model or None if not found
        """
        try:
            query = """
                SELECT id, name, domain, description, vertical, telegram_channel,
                       wordpress_url, wordpress_username, wordpress_app_password,
                       created_at, updated_at,
                       last_active, total_articles_generated, total_tokens_consumed,
                       total_cost_usd
                FROM projects
                WHERE id = %s AND deleted_at IS NULL
            """

            # Request results (list of dicts from RealDictCursor)
            result = self.db.execute(query, (str(project_id),), fetch_all=True)

            if not result:
                logger.warning(f"Project not found: {project_id}")
                return None

            # Result is a list of RealDictRow objects (dict-like)
            row = result[0]

            return Project(
                id=row["id"],
                name=row["name"],
                domain=row["domain"],
                description=row["description"],
                vertical=row["vertical"],
                telegram_channel=row["telegram_channel"],
                wordpress_url=row["wordpress_url"],
                wordpress_username=row["wordpress_username"],
                wordpress_app_password=row["wordpress_app_password"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                last_active=row["last_active"] or row["created_at"],
                total_articles_generated=row["total_articles_generated"],
                total_tokens_consumed=row["total_tokens_consumed"],
                total_cost_usd=float(row["total_cost_usd"]) if row["total_cost_usd"] else 0.0,
            )

        except Exception as e:
            logger.error(f"Failed to retrieve project {project_id}: {e}")
            return None

    def exists(self, project_id: UUID) -> bool:
        """
        Check if project exists (synchronously).

        Args:
            project_id: UUID of project

        Returns:
            True if project exists and not deleted
        """
        try:
            query = """
                SELECT 1
                FROM projects
                WHERE id = %s AND deleted_at IS NULL
            """

            result = self.db.execute(query, (str(project_id),), fetch_all=True)
            return len(result) > 0

        except Exception as e:
            logger.error(f"Failed to check project existence: {e}")
            return False
