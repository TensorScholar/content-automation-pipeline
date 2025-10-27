"""
Article Repository: Data Access Layer for Generated Articles

Encapsulates all database operations for article management including:
- CRUD operations for articles and revisions
- Search and filtering capabilities
- Analytics and reporting queries
- Export functionality

Design Pattern: Repository Pattern with SQLAlchemy Core
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import delete, func, or_, select

from core.models import ContentPlan, GeneratedArticle
from infrastructure.database import DatabaseManager
from infrastructure.schema import article_revisions_table, generated_articles_table


class ArticleRepository:
    """
    Repository for article data access operations.

    Provides a clean interface for all database operations related to
    generated articles and their revisions, abstracting away SQLAlchemy
    implementation details from the service layer.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize repository with database manager.

        Args:
            db_manager: DatabaseManager instance for database operations
        """
        self.db = db_manager
        logger.debug("ArticleRepository initialized")

    async def get_by_id(
        self, article_id: UUID, include_content: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve article by ID.

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
        query = generated_articles_table.insert().values(article_data)
        result = await self.db.execute(query)

        # Fetch the created article
        created_id = result.get("id") or article_data.get("id")
        return await self.get_by_id(created_id)

    async def update(self, article_id: UUID, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update existing article.

        Args:
            article_id: Article identifier
            updates: Dictionary of fields to update

        Returns:
            Updated article data
        """
        updates["updated_at"] = datetime.utcnow()

        query = (
            generated_articles_table.update()
            .where(generated_articles_table.c.id == article_id)
            .values(updates)
        )
        await self.db.execute(query)

        return await self.get_by_id(article_id)

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
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

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

    async def get_distribution_status(self, article_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get distribution status for article.

        Args:
            article_id: Article identifier

        Returns:
            Distribution status dict or None if not found
        """
        query = select(
            generated_articles_table.c.id,
            generated_articles_table.c.distributed_at,
            generated_articles_table.c.distribution_channels,
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
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

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

    async def update_distribution_status(
        self, article_id: UUID, distributed_at: datetime, channels: List[str]
    ) -> bool:
        """
        Update article distribution status.

        Args:
            article_id: Article identifier
            distributed_at: Distribution timestamp
            channels: List of distribution channels

        Returns:
            True if updated successfully
        """
        updates = {
            "distributed_at": distributed_at,
            "distribution_channels": channels,
            "updated_at": datetime.utcnow(),
        }

        query = (
            generated_articles_table.update()
            .where(generated_articles_table.c.id == article_id)
            .values(updates)
        )
        result = await self.db.execute(query)

        return result != "UPDATE 0"

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

        return dict(revision) if revision else None

    async def save_content_plan(self, plan: ContentPlan) -> None:
        """
        Saves a new content plan to the database.

        Args:
            plan: The ContentPlan object to save.
        """
        # Note: 'outline' is a JSONB field.
        # 'keywords' needs to be serialized (e.g., list of strings).
        query = """
            INSERT INTO content_plans (
                id, project_id, topic, outline, 
                target_word_count, readability_target, 
                estimated_cost_usd, created_at
            ) VALUES (
                :id, :project_id, :topic, :outline,
                :target_word_count, :readability_target,
                :estimated_cost_usd, :created_at
            )
        """

        # Serialize complex types for database
        plan_dict = plan.model_dump()
        plan_dict["outline"] = plan.outline.model_dump_json()

        await self.db.execute(query=query, values=plan_dict)

    async def save_generated_article(self, article: GeneratedArticle) -> None:
        """
        Saves a fully generated article to the database.

        Args:
            article: The GeneratedArticle object to save.
        """
        # Note: 'quality_metrics' is a JSONB field.
        query = """
            INSERT INTO generated_articles (
                id, project_id, content_plan_id, title, content, 
                meta_description, total_tokens_used, total_cost_usd,
                generation_time_seconds, quality_metrics, model_used,
                status, created_at, updated_at
            ) VALUES (
                :id, :project_id, :content_plan_id, :title, :content, 
                :meta_description, :total_tokens_used, :total_cost_usd,
                :generation_time_seconds, :quality_metrics, :model_used,
                :status, :created_at, :updated_at
            )
        """

        # Serialize complex types for database
        article_dict = article.model_dump()
        article_dict["quality_metrics"] = article.quality_metrics.model_dump_json()

        await self.db.execute(query=query, values=article_dict)

    async def update_article_distribution(
        self, article_id: UUID, distributed_at: datetime, channels: list[str]
    ) -> None:
        query = """
            UPDATE generated_articles
            SET distributed_at = :distributed_at, 
                distribution_channels = :channels,
                updated_at = :updated_at
            WHERE id = :article_id
        """
        await self.db.execute(
            query=query,
            values={
                "article_id": article_id,
                "distributed_at": distributed_at,
                "channels": channels,
                "updated_at": datetime.utcnow(timezone.utc),
            },
        )

    # =========================================================================
    # CONTENT PLAN OPERATIONS
    # =========================================================================
