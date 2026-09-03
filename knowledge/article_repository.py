"""
Article Repository: Data Access Layer for Generated Articles

Encapsulates all database operations for article management including:
- CRUD operations for articles and revisions
- Search and filtering capabilities
- Analytics and reporting queries
- Export functionality
- Redis-based query result caching

Design Pattern: Repository Pattern with SQLAlchemy Core
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, cast
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import delete, func, insert, or_, select, text, update
from sqlalchemy.engine import CursorResult

from core.models import ContentPlan, GeneratedArticle
from infrastructure.database import DatabaseManager
from infrastructure.redis_client import RedisClient
from infrastructure.schema import (
    article_revisions_table,
    content_plans_table,
    generated_articles_table,
    projects_table,
    users_table,
)
from optimization.query_cache import cached_query


class ArticleRepository:
    """
    Repository for article data access operations.

    Provides a clean interface for all database operations related to
    generated articles and their revisions, with Redis caching for
    frequently accessed queries.
    """

    def __init__(self, db_manager: DatabaseManager, redis_client: Optional[RedisClient] = None):
        """
        Initialize repository with database manager and optional Redis client.

        Args:
            db_manager: DatabaseManager instance for database operations
            redis_client: Optional Redis client for query caching
        """
        self.db = db_manager
        self.redis_client = redis_client
        logger.debug("ArticleRepository initialized")

    @staticmethod
    def _to_db_naive_utc(value: datetime) -> datetime:
        """Convert timezone-aware datetime to naive UTC for TIMESTAMP columns."""
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def _normalize_time_window(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> tuple[datetime, datetime]:
        """Normalize time window to naive UTC datetimes for database comparisons."""
        normalized_end = (
            self._to_db_naive_utc(end_date)
            if end_date
            else datetime.now(timezone.utc).replace(tzinfo=None)
        )
        normalized_start = (
            self._to_db_naive_utc(start_date)
            if start_date
            else normalized_end - timedelta(days=30)
        )
        return normalized_start, normalized_end

    @cached_query(ttl=300, key_prefix="article")
    async def get_by_id(
        self, article_id: UUID, include_content: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve article by ID with Redis caching.

        Cached for 5 minutes for faster repeated access.

        Args:
            article_id: Article identifier
            include_content: Whether to include full content in response

        Returns:
            Article data dict or None if not found
        """
        query = select(generated_articles_table).where(generated_articles_table.c.id == article_id)

        article = await self.db.fetch_one(query)

        if article and not include_content:
            # Remove content field for metadata-only response
            article_dict = dict(article)
            article_dict.pop("content", None)
            return article_dict

        return dict(article) if article else None

    async def create(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new article.

        Args:
            article_data: Article data dictionary

        Returns:
            Created article data
        """
        query = (
            generated_articles_table.insert()
            .values(article_data)
            .returning(generated_articles_table)
        )
        async with self.db.session() as session:
            result = await session.execute(query)
            created = result.mappings().one()

        return dict(created)

    async def update(self, article_id: UUID, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing article.

        Args:
            article_id: Article identifier
            updates: Dictionary of fields to update

        Returns:
            Updated article data
        """
        updates["updated_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

        query = (
            generated_articles_table.update()
            .where(generated_articles_table.c.id == article_id)
            .values(updates)
        )
        await self.db.execute(query)

        return await self.get_by_id(article_id)

    async def update_content_with_revision(
        self,
        *,
        article_id: UUID,
        content: str,
        word_count: int,
        revision_note: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Atomically snapshot the current article and apply a manual edit."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            current_result = await session.execute(
                select(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .with_for_update()
            )
            current = current_result.mappings().one_or_none()
            if current is None:
                return None

            await session.execute(
                insert(article_revisions_table).values(
                    id=uuid4(),
                    article_id=article_id,
                    title=current["title"],
                    content=current["content"],
                    revision_note=revision_note,
                    word_count=current["word_count"],
                    created_at=now,
                )
            )
            updated_result = await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    content=content,
                    word_count=word_count,
                    readability_score=None,
                    keyword_density={},
                    review_status="pending_review",
                    review_note=None,
                    reviewed_by=None,
                    reviewed_at=None,
                    review_updated_at=now,
                    updated_at=now,
                )
                .returning(generated_articles_table)
            )
            updated = updated_result.mappings().one()

        if self.redis_client:
            await self.redis_client.delete_pattern("article:get_by_id:*")
        return dict(updated)

    async def get_review_state(self, article_id: UUID) -> Optional[Dict[str, Any]]:
        """Return durable review state with a safe reviewer display label."""
        query = (
            select(
                generated_articles_table.c.id,
                generated_articles_table.c.project_id,
                generated_articles_table.c.review_status,
                generated_articles_table.c.review_note,
                generated_articles_table.c.reviewed_by,
                generated_articles_table.c.reviewed_at,
                generated_articles_table.c.review_updated_at,
                users_table.c.full_name.label("reviewer_full_name"),
                users_table.c.email.label("reviewer_email"),
            )
            .select_from(
                generated_articles_table.outerjoin(
                    users_table,
                    generated_articles_table.c.reviewed_by == users_table.c.id,
                )
            )
            .where(generated_articles_table.c.id == article_id)
        )
        row = await self.db.fetch_one(query)
        return dict(row) if row else None

    async def set_review_state(
        self,
        *,
        article_id: UUID,
        review_status: str,
        reviewer_id: UUID,
        note: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Atomically store the current review decision for an article."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            existing = await session.execute(
                select(generated_articles_table.c.id)
                .where(generated_articles_table.c.id == article_id)
                .with_for_update()
            )
            if existing.scalar_one_or_none() is None:
                return None

            await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    review_status=review_status,
                    review_note=note,
                    reviewed_by=reviewer_id,
                    reviewed_at=now,
                    review_updated_at=now,
                    updated_at=now,
                )
            )

        return await self.get_review_state(article_id)

    async def delete(self, article_id: UUID) -> bool:
        """
        Delete article permanently.

        Args:
            article_id: Article identifier

        Returns:
            True if deleted, False if not found
        """
        query = delete(generated_articles_table).where(generated_articles_table.c.id == article_id)
        result = await self.db.execute(query)

        return result != "DELETE 0"

    async def search(
        self, query_text: str, project_id: Optional[UUID] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search articles by title and content.

        Args:
            query_text: Search query string
            project_id: Optional project filter
            limit: Maximum results to return

        Returns:
            List of matching articles
        """
        search_query = select(
            generated_articles_table.c.id,
            generated_articles_table.c.project_id,
            generated_articles_table.c.title,
            generated_articles_table.c.word_count,
            generated_articles_table.c.readability_score,
            generated_articles_table.c.created_at,
        ).where(
            or_(
                generated_articles_table.c.title.ilike(f"%{query_text}%"),
                generated_articles_table.c.content.ilike(f"%{query_text}%"),
            )
        )

        if project_id:
            search_query = search_query.where(generated_articles_table.c.project_id == project_id)

        search_query = search_query.order_by(generated_articles_table.c.created_at.desc()).limit(
            limit
        )

        results = await self.db.fetch_all(search_query)
        return [dict(article) for article in results]

    async def get_recent_project_articles(
        self, project_id: UUID, limit: int = 12
    ) -> List[Dict[str, Any]]:
        """Fetch recent article metadata for project-level content memory."""
        safe_limit = max(1, min(limit, 50))
        query = (
            select(
                generated_articles_table.c.id,
                generated_articles_table.c.project_id,
                generated_articles_table.c.title,
                generated_articles_table.c.meta_description,
                generated_articles_table.c.keywords,
                generated_articles_table.c.word_count,
                generated_articles_table.c.readability_score,
                generated_articles_table.c.created_at,
            )
            .where(generated_articles_table.c.project_id == project_id)
            .order_by(generated_articles_table.c.created_at.desc())
            .limit(safe_limit)
        )

        rows = await self.db.fetch_all(query)
        return [dict(row) for row in rows]

    async def get_analytics(
        self,
        project_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get content generation analytics.

        Args:
            project_id: Optional project filter
            start_date: Start date for analytics (defaults to 30 days ago)
            end_date: End date for analytics (defaults to now)

        Returns:
            Analytics data dictionary
        """
        start_date, end_date = self._normalize_time_window(start_date, end_date)

        # Aggregate metrics
        stats_query = select(
            func.count(generated_articles_table.c.id).label("total_articles"),
            func.sum(generated_articles_table.c.total_cost).label("total_cost"),
            func.avg(generated_articles_table.c.generation_time).label("avg_generation_time"),
            func.avg(generated_articles_table.c.readability_score).label("avg_quality_score"),
        ).where(generated_articles_table.c.created_at.between(start_date, end_date))

        if project_id:
            stats_query = stats_query.where(generated_articles_table.c.project_id == project_id)

        stats = await self.db.fetch_one(stats_query)

        # Articles by day
        articles_by_day_query = select(
            func.date(generated_articles_table.c.created_at).label("date"),
            func.count(generated_articles_table.c.id).label("count"),
            func.sum(generated_articles_table.c.total_cost).label("daily_cost"),
        ).where(generated_articles_table.c.created_at.between(start_date, end_date))

        if project_id:
            articles_by_day_query = articles_by_day_query.where(
                generated_articles_table.c.project_id == project_id
            )

        articles_by_day_query = articles_by_day_query.group_by(
            func.date(generated_articles_table.c.created_at)
        ).order_by(func.date(generated_articles_table.c.created_at))

        articles_by_day = await self.db.fetch_all(articles_by_day_query)

        # Quality trend
        quality_trend_query = select(
            func.date(generated_articles_table.c.created_at).label("date"),
            func.avg(generated_articles_table.c.readability_score).label("avg_quality"),
        ).where(generated_articles_table.c.created_at.between(start_date, end_date))

        if project_id:
            quality_trend_query = quality_trend_query.where(
                generated_articles_table.c.project_id == project_id
            )

        quality_trend_query = quality_trend_query.group_by(
            func.date(generated_articles_table.c.created_at)
        ).order_by(func.date(generated_articles_table.c.created_at))

        quality_trend = await self.db.fetch_all(quality_trend_query)

        return {
            "total_articles": stats["total_articles"],
            "total_cost": float(stats["total_cost"] or 0),
            "avg_generation_time": float(stats["avg_generation_time"] or 0),
            "avg_quality_score": float(stats["avg_quality_score"] or 0),
            "cost_per_article": float(stats["total_cost"] or 0) / max(stats["total_articles"], 1),
            "articles_by_day": [
                {
                    "date": row["date"].isoformat(),
                    "count": row["count"],
                    "daily_cost": float(row["daily_cost"]),
                }
                for row in articles_by_day
            ],
            "quality_trend": [
                {"date": row["date"].isoformat(), "avg_quality": float(row["avg_quality"])}
                for row in quality_trend
            ],
        }

    async def get_quality_metrics(self, article_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get quality metrics for article.

        Args:
            article_id: Article identifier

        Returns:
            Quality metrics dict or None if not found
        """
        query = select(
            generated_articles_table.c.id,
            generated_articles_table.c.content,
            generated_articles_table.c.readability_score,
            generated_articles_table.c.keyword_density,
        ).where(generated_articles_table.c.id == article_id)

        article = await self.db.fetch_one(query)
        return dict(article) if article else None

    async def get_article_history(self, article_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get revision history for article.

        Args:
            article_id: Article identifier

        Returns:
            History data dict or None if not found
        """
        # Query current version
        current_query = select(
            generated_articles_table.c.id,
            generated_articles_table.c.title,
            generated_articles_table.c.content,
            generated_articles_table.c.created_at,
            generated_articles_table.c.word_count,
        ).where(generated_articles_table.c.id == article_id)

        current = await self.db.fetch_one(current_query)
        if not current:
            return None

        # Query revision history
        revisions_query = (
            select(
                article_revisions_table.c.id,
                article_revisions_table.c.title,
                article_revisions_table.c.content,
                article_revisions_table.c.created_at,
                article_revisions_table.c.revision_note,
                article_revisions_table.c.word_count,
            )
            .where(article_revisions_table.c.article_id == article_id)
            .order_by(article_revisions_table.c.created_at.desc())
        )

        revisions = await self.db.fetch_all(revisions_query)

        return {
            "current_version": {
                "id": str(current["id"]),
                "title": current["title"],
                "content": current["content"],
                "created_at": current["created_at"],
                "word_count": current["word_count"],
            },
            "revisions": [
                {
                    "id": str(rev["id"]),
                    "title": rev["title"],
                    "content": rev["content"],
                    "revision_note": rev["revision_note"],
                    "created_at": rev["created_at"],
                    "word_count": rev["word_count"],
                }
                for rev in revisions
            ],
            "total_revisions": len(revisions),
        }

    async def export_articles(
        self,
        project_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export articles for specified criteria.

        Args:
            project_id: Optional project filter
            start_date: Start date filter (defaults to 30 days ago)
            end_date: End date filter (defaults to now)

        Returns:
            List of article data for export
        """
        start_date, end_date = self._normalize_time_window(start_date, end_date)

        articles_query = select(
            generated_articles_table.c.id,
            generated_articles_table.c.project_id,
            generated_articles_table.c.title,
            generated_articles_table.c.word_count,
            generated_articles_table.c.total_cost,
            generated_articles_table.c.generation_time,
            generated_articles_table.c.readability_score,
            generated_articles_table.c.created_at,
        ).where(generated_articles_table.c.created_at.between(start_date, end_date))

        if project_id:
            articles_query = articles_query.where(
                generated_articles_table.c.project_id == project_id
            )

        articles_query = articles_query.order_by(generated_articles_table.c.created_at.desc())

        articles = await self.db.fetch_all(articles_query)
        return [dict(article) for article in articles]

    async def create_revision(self, revision_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create article revision.

        Args:
            revision_data: Revision data dictionary

        Returns:
            Created revision data
        """
        query = article_revisions_table.insert().values(revision_data)
        result = await self.db.execute(query)

        # Fetch the created revision
        created_id = result.get("id") or revision_data.get("id")

        revision_query = select(article_revisions_table).where(
            article_revisions_table.c.id == created_id
        )
        revision = await self.db.fetch_one(revision_query)

        if revision is None:
            raise RuntimeError(f"Article revision {created_id} was not persisted")
        return dict(revision)

    async def save_content_plan(self, plan: ContentPlan) -> None:
        """
        Saves a new content plan to the database using SQLAlchemy Core.

        Args:
            plan: The ContentPlan object to save.
        """
        try:
            # Serialize complex types for database
            plan_dict = plan.model_dump()
            plan_dict["outline_json"] = plan.outline.model_dump_json()

            async with self.db.session() as session:
                query = insert(content_plans_table).values(
                    id=plan.id,
                    project_id=plan.project_id,
                    topic=plan.topic,
                    outline_json=plan_dict["outline_json"],
                    target_word_count=plan.target_word_count,
                    readability_target=plan.readability_target,
                    estimated_cost=plan.estimated_cost_usd,
                    created_at=self._to_db_naive_utc(plan.created_at),
                )
                await session.execute(query)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save content plan: {e}")
            raise

    async def save_generated_article(self, article: GeneratedArticle) -> None:
        """
        Saves a fully generated article to the database using SQLAlchemy Core.

        Args:
            article: The GeneratedArticle object to save.
        """
        try:
            # Serialize complex types for database
            article_dict = article.model_dump()
            quality_metrics_json = article.quality_metrics.model_dump_json()

            async with self.db.session() as session:
                query = insert(generated_articles_table).values(
                    id=article.id,
                    project_id=article.project_id,
                    content_plan_id=article.content_plan_id,
                    title=article.title,
                    content=article.content,
                    meta_description=article.meta_description,
                    keywords=article.keywords or [],
                    word_count=article.quality_metrics.word_count,
                    readability_score=article.quality_metrics.readability_score,
                    keyword_density=article.quality_metrics.keyword_density,
                    total_tokens_used=article.total_tokens_used,
                    total_cost=article.total_cost_usd,
                    generation_time=article.generation_time_seconds,
                    # Ensure timestamps are naive UTC for DateTime column
                    created_at=article.created_at.astimezone(timezone.utc).replace(tzinfo=None) if article.created_at.tzinfo else article.created_at,
                    updated_at=article.updated_at.astimezone(timezone.utc).replace(tzinfo=None) if article.updated_at.tzinfo else article.updated_at,
                )
                await session.execute(query)
                activity_time = self._to_db_naive_utc(article.created_at)
                counter_update = (
                    update(projects_table)
                    .where(projects_table.c.id == article.project_id)
                    .values(
                        total_articles_generated=func.coalesce(
                            projects_table.c.total_articles_generated, 0
                        )
                        + 1,
                        total_tokens_consumed=func.coalesce(
                            projects_table.c.total_tokens_consumed, 0
                        )
                        + article.total_tokens_used,
                        total_cost_usd=func.coalesce(projects_table.c.total_cost_usd, 0)
                        + article.total_cost_usd,
                        last_active=activity_time,
                        updated_at=activity_time,
                    )
                )
                counter_result = cast(CursorResult[Any], await session.execute(counter_update))
                if counter_result.rowcount != 1:
                    raise ValueError(
                        f"Project {article.project_id} not found while saving generated article"
                    )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save generated article: {e}")
            raise

    # =========================================================================
    # CONTENT PLAN OPERATIONS
    # =========================================================================
