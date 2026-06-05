"""Content Service - Business Logic Layer.

Encapsulates business operations for content management:
- Content generation and batch processing
- Quality analysis and validation
- Content revision workflows
- Distribution management
- Analytics and reporting

Architecture:
    Service layer pattern with repository abstraction.
    Coordinates between repositories and domain logic.

Example:
    >>> service = ContentService(article_repo, project_service)
    >>> result = await service.batch_generate_content(
    ...     project_id=uuid.uuid4(),
    ...     topics=["Topic 1", "Topic 2"],
    ...     priority="high"
    ... )
"""

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from loguru import logger

from core.enums import DistributionChannel
from core.exceptions import DistributionError
from core.models import ContentGenerationRequest, GeneratedArticle, QualityMetrics
from execution.distributer import Distributor
from knowledge.article_repository import ArticleRepository
from knowledge.project_repository import ProjectRepository
from orchestration.content_agent import ContentAgent
from orchestration.task_persistence import TaskResultRepository
from orchestration.task_state import normalize_db_status, reconcile_task_state

if TYPE_CHECKING:
    from intelligence.semantic_analyzer import SemanticAnalyzer


class ContentService:
    """Business logic layer for content operations.

    Provides high-level operations that coordinate between repositories,
    implement business rules, and abstract complexity from API handlers.

    Attributes:
        articles: Article repository for data access.
        content_agent: Content agent for orchestration.
    """

    def __init__(
        self,
        article_repository: ArticleRepository,
        content_agent: ContentAgent,
        task_repository: Optional[TaskResultRepository] = None,
        metrics_collector: Any = None,
        semantic_analyzer: Optional["SemanticAnalyzer"] = None,
    ):
        """
        Initialize service with required dependencies.

        Args:
            article_repository: Repository for article data access
            content_agent: Content agent for orchestration
            task_repository: Optional repository for persistent task tracking
            metrics_collector: Optional metrics collector
            semantic_analyzer: Optional semantic analyzer for coherence analysis
        """
        self.articles = article_repository
        self.content_agent = content_agent
        self.task_repo = task_repository
        self.metrics = metrics_collector
        self.projects = ProjectRepository(article_repository.db)
        self.semantic_analyzer = semantic_analyzer
        logger.debug("ContentService initialized")

    async def generate_content_workflow(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Orchestrate end-to-end generation with optional auto-distribution."""
        article = await self.content_agent.create_content(
            project_id=request.project_id,
            topic=request.topic,
            priority=getattr(request, "priority", "high"),
            custom_instructions=getattr(request, "custom_instructions", None),
        )

        distribution_result: Dict[str, Any] | None = None
        if request.auto_distribute:
            distribution_result = await self._distribute_to_wordpress(article, request.project_id)

        return {
            "article_id": str(article.id),
            "project_id": str(article.project_id),
            "status": "distributed" if distribution_result else "completed",
            "distribution": distribution_result,
        }

    async def _distribute_to_wordpress(
        self, article: GeneratedArticle, project_id: UUID
    ) -> Dict[str, Any]:
        """Send generated article to WordPress without breaking the pipeline on failure."""
        # Project lookup should bubble up errors if it fails systemically
        project = await self.projects.get_by_id(project_id)

        if not project:
            logger.warning(f"Project {project_id} not found; skipping WordPress distribution")
            return {"status": "skipped", "reason": "project_not_found"}

        distributor = Distributor()

        try:
            result = await distributor.distribute_to_wordpress(article, project)
            if result.get("status") == "published":
                distributed_at = datetime.now(timezone.utc)
                channels = [DistributionChannel.WORDPRESS]
                article.distributed_at = distributed_at
                article.distribution_channels = channels
                await self.articles.update_distribution_status(
                    article.id, distributed_at, [channel.value for channel in channels]
                )
            logger.info(
                "WordPress distribution completed | article_id=%s | status=%s",
                article.id,
                result.get("status"),
            )
            return result
        except DistributionError as e:
            logger.error(f"WordPress distribution failed for article {article.id}: {e}")
            return {"status": "error", "reason": str(e)}
        # We removed the broad Exception catch here.
        # Unexpected errors should propagate or be handled by the caller (workflow).

    @staticmethod
    def _coerce_keywords(value: Any) -> list[str]:
        """Normalize persisted JSON/string keyword fields into a string list."""
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    @staticmethod
    def _coerce_keyword_density(value: Any) -> dict[str, float]:
        """Normalize persisted JSON/string keyword density fields."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return {}
        if not isinstance(value, dict):
            return {}

        density: dict[str, float] = {}
        for key, raw_score in value.items():
            try:
                density[str(key)] = float(raw_score)
            except (TypeError, ValueError):
                continue
        return density

    def _article_dict_to_generated_article(self, article: Dict[str, Any]) -> GeneratedArticle:
        """Rebuild the strict domain article model from a persisted article row."""
        article_id = UUID(str(article.get("id")))
        project_id = UUID(str(article.get("project_id")))
        content_plan_id = UUID(str(article.get("content_plan_id") or article_id))

        title = str(article.get("title") or "").strip()
        content = str(article.get("content") or "").strip()
        if not title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Article is missing a title and cannot be distributed",
            )
        if len(content) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Article content is too short for distribution",
            )

        meta_description = str(article.get("meta_description") or "").strip()
        if len(meta_description) < 50:
            meta_description = f"{title}. {content[:140]}".strip()
        if len(meta_description) > 160:
            meta_description = meta_description[:157].rstrip() + "..."

        tokens = re.findall(r"\w+", content, flags=re.UNICODE)
        word_count = int(article.get("word_count") or len(tokens))
        sentence_count = max(1, len(re.findall(r"[.!؟?]+", content)))
        paragraph_count = max(1, len([p for p in content.splitlines() if p.strip()]))
        unique_tokens = {token.lower() for token in tokens}

        quality_metrics = QualityMetrics(
            word_count=max(0, word_count),
            readability_score=float(article.get("readability_score") or 0.0),
            lexical_diversity=(len(unique_tokens) / len(tokens)) if tokens else 0.0,
            keyword_density=self._coerce_keyword_density(article.get("keyword_density")),
            avg_sentence_length=max(1.0, word_count / sentence_count),
            paragraph_count=paragraph_count,
        )

        created_at = article.get("created_at") or datetime.now(timezone.utc)
        updated_at = article.get("updated_at") or created_at

        return GeneratedArticle(
            id=article_id,
            project_id=project_id,
            content_plan_id=content_plan_id,
            title=title,
            content=content,
            meta_description=meta_description,
            keywords=self._coerce_keywords(article.get("keywords")),
            quality_metrics=quality_metrics,
            total_tokens_used=int(article.get("total_tokens_used") or 0),
            total_cost_usd=float(article.get("total_cost") or article.get("total_cost_usd") or 0.0),
            generation_time_seconds=float(
                article.get("generation_time") or article.get("generation_time_seconds") or 0.0
            ),
            created_at=created_at,
            updated_at=updated_at,
        )

    async def batch_generate_content(
        self,
        project_id: UUID,
        topics: List[str],
        priority: str = "high",
        schedule_after: Optional[datetime] = None,
        custom_instructions: Optional[str] = None,
        submitted_by_user_id: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit batch content generation with business logic validation.

        Args:
            project_id: Project identifier
            topics: List of topics to generate
            priority: Generation priority
            schedule_after: Optional delayed execution

        Returns:
            Batch submission result

        Raises:
            HTTPException: If schedule_after is in the past
        """
        if schedule_after and schedule_after < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="schedule_after must be in the future",
            )

        # Import Celery task here to avoid circular imports
        from orchestration.tasks import generate_content_task

        # Schedule generation for each topic via Celery
        task_records = []

        for topic in topics:
            # Dispatch to Celery worker - task handles its own persistence
            if schedule_after:
                # Schedule for future execution
                celery_task = generate_content_task.apply_async(
                    kwargs={
                        "project_id": str(project_id),
                        "topic": topic,
                        "priority": priority,
                        "custom_instructions": custom_instructions,
                        "submitted_by_user_id": submitted_by_user_id,
                        "model_override": model_override,
                    },
                    eta=schedule_after,
                    queue=priority,
                    routing_key=priority,
                )
            else:
                # Execute immediately
                celery_task = generate_content_task.apply_async(
                    kwargs={
                        "project_id": str(project_id),
                        "topic": topic,
                        "priority": priority,
                        "custom_instructions": custom_instructions,
                        "submitted_by_user_id": submitted_by_user_id,
                        "model_override": model_override,
                    },
                    queue=priority,
                    routing_key=priority,
                )

            task_records.append(celery_task.id)

        logger.info(f"Dispatched batch generation to Celery for {len(topics)} topics | project_id={project_id}")

        return {
            "message": "Batch generation started",
            "project_id": str(project_id),
            "batch_id": ",".join(task_records),
            "batch_size": len(topics),
            "topics": topics,
            "task_ids": task_records,
            "status": "processing",
            "estimated_completion_seconds": len(topics) * 120  # Rough estimate
        }

    # _safe_generate removed - logic now lives in orchestration.tasks.generate_content_task

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get batch generation status with business logic.

        The batch_id is a comma-separated list of Celery task IDs returned
        by batch_generate_content. We query each task's status from Celery
        and fall back to the task_results DB table for completed/expired tasks.

        Args:
            batch_id: Batch identifier (comma-separated task IDs)

        Returns:
            Batch status information with per-task breakdown
        """
        from celery.result import AsyncResult
        from orchestration.celery_app import app as celery_app

        task_ids = [tid.strip() for tid in batch_id.split(",") if tid.strip()]

        if not task_ids:
            return {
                "batch_id": batch_id,
                "status": "unknown",
                "progress": {"completed": 0, "failed": 0, "pending": 0, "total": 0},
                "tasks": [],
            }

        completed = 0
        failed = 0
        pending = 0
        tasks_detail = []

        for task_id in task_ids:
            # Check Celery first
            celery_result = AsyncResult(task_id, app=celery_app)
            state = celery_result.state
            db_task = None
            topic = "Unknown"

            if self.task_repo:
                try:
                    db_task = await self.task_repo.get_task_by_id(task_id)
                    if db_task:
                        db_status = normalize_db_status(str(db_task.get("status", "")))
                        state, _ = reconcile_task_state(celery_state=state, db_status=db_status)

                        raw_args = db_task.get("args")
                        parsed_args = None
                        if isinstance(raw_args, list):
                            parsed_args = raw_args
                        elif isinstance(raw_args, str):
                            try:
                                candidate = json.loads(raw_args)
                                parsed_args = candidate if isinstance(candidate, list) else None
                            except Exception:
                                parsed_args = None

                        if parsed_args and len(parsed_args) >= 2 and isinstance(parsed_args[1], str):
                            topic = parsed_args[1]
                except Exception as e:
                    logger.debug(f"DB fallback for batch task {task_id} failed: {e}")

            if state == "SUCCESS":
                completed += 1
            elif state == "FAILURE":
                failed += 1
            else:
                pending += 1

            tasks_detail.append({"task_id": task_id, "state": state, "topic": topic})

        total = len(task_ids)

        # Determine overall batch status
        if completed == total:
            overall_status = "completed"
        elif failed == total:
            overall_status = "failed"
        elif completed + failed == total:
            overall_status = "completed_with_errors"
        else:
            overall_status = "processing"

        return {
            "batch_id": batch_id,
            "status": overall_status,
            # Backward-compatible flat fields used by legacy dashboard
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "results": [
                {"task_id": item["task_id"], "topic": item["topic"], "status": item["state"].lower()}
                for item in tasks_detail
            ],
            "progress": {
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "total": total,
            },
            "tasks": tasks_detail,
        }

    async def get_article(self, article_id: UUID, include_content: bool = True) -> Dict[str, Any]:
        """
        Get article with business logic validation.

        Args:
            article_id: Article identifier
            include_content: Whether to include full content

        Returns:
            Article data

        Raises:
            HTTPException: If article not found
        """
        article = await self.articles.get_by_id(article_id, include_content)

        if not article:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        return article

    async def request_article_revision(
        self,
        article_id: UUID,
        feedback: str,
        sections_to_revise: Optional[List[str]] = None,
        priority: str = "high",
        submitted_by_user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Request article revision with business logic validation.

        Args:
            article_id: Article identifier
            feedback: Revision feedback
            sections_to_revise: Specific sections to revise
            priority: Revision priority

        Returns:
            Revision request result

        Raises:
            HTTPException: If article not found
        """
        # Verify article exists and get full content for revision snapshot
        article = await self.articles.get_by_id(article_id, include_content=True)

        if not article:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        # Save current version as a revision snapshot before regeneration
        from uuid import uuid4 as _uuid4

        try:
            await self.articles.create_revision({
                "id": _uuid4(),
                "article_id": article_id,
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "revision_note": f"Pre-revision snapshot. Feedback: {feedback[:200]}",
                "word_count": article.get("word_count", 0),
            })
        except Exception as e:
            logger.warning(f"Failed to save revision snapshot for {article_id}: {e}")

        # Dispatch regeneration task via Celery with feedback as custom instructions
        from orchestration.tasks import generate_content_task

        revision_instructions = (
            f"REVISION REQUEST for existing article titled '{article.get('title', '')}'.\n"
            f"Feedback to incorporate: {feedback}\n"
        )
        if sections_to_revise:
            revision_instructions += f"Sections to revise: {', '.join(sections_to_revise)}\n"

        project_id = article.get("project_id")
        topic = article.get("title", "Revision")

        celery_task = generate_content_task.apply_async(
            # Use kwargs (not positional args) so that future changes to
            # generate_content_task's signature cannot silently pass wrong values.
            kwargs={
                "project_id": str(project_id),
                "topic": topic,
                "priority": priority,
                "custom_instructions": revision_instructions,
                "submitted_by_user_id": submitted_by_user_id,
            },
            queue=priority,
            routing_key=priority,
        )

        logger.info(
            f"Article revision dispatched | article_id={article_id} | "
            f"revision_task_id={celery_task.id}"
        )

        return {
            "article_id": str(article_id),
            "revision_task_id": celery_task.id,
            "status": "processing",
            "feedback_incorporated": feedback[:200],
            "sections_to_revise": sections_to_revise,
            "status_endpoint": f"/content/task/{celery_task.id}",
        }

    async def delete_article(self, article_id: UUID) -> None:
        """
        Delete article with business logic validation.

        Args:
            article_id: Article identifier

        Raises:
            HTTPException: If article not found
        """
        deleted = await self.articles.delete(article_id)

        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

    async def get_quality_metrics(self, article_id: UUID) -> Dict[str, Any]:
        """
        Get comprehensive article quality metrics with business logic.

        Analyzes:
        - Readability (Flesch-Kincaid)
        - Keyword density and distribution
        - Semantic coherence (via embeddings)
        - Content structure (headings, paragraphs)
        - SEO optimization score
        - Overall weighted quality score

        Args:
            article_id: Article identifier

        Returns:
            Comprehensive quality metrics data

        Raises:
            HTTPException: If article not found
        """
        article = await self.articles.get_quality_metrics(article_id)

        if not article:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        # Fetch full article for comprehensive analysis
        full_article = await self.articles.get_by_id(article_id, include_content=True)
        content = full_article.get("content", "") if full_article else ""
        target_keywords = full_article.get("keywords", []) if full_article else []

        readability_grade = self._readability_score_to_grade(article["readability_score"])

        # Compute comprehensive metrics
        structure_score = self._analyze_structure(content)
        keyword_analysis = self._analyze_keyword_density(content, target_keywords)
        semantic_coherence = await self._analyze_semantic_coherence(content)
        seo_score = self._calculate_seo_score(
            content=content,
            keywords=target_keywords,
            readability=article["readability_score"],
            structure=structure_score,
        )

        # Calculate overall quality as weighted average
        overall_quality = self._calculate_overall_quality(
            readability=article["readability_score"],
            structure=structure_score["score"],
            seo=seo_score["score"],
            semantic_coherence=semantic_coherence.get("score") if semantic_coherence else None,
        )

        return {
            "article_id": str(article_id),
            "readability_score": article["readability_score"],
            "readability_grade": readability_grade,
            "keyword_density": keyword_analysis["overall_density"],
            "keyword_analysis": keyword_analysis,
            "semantic_coherence": semantic_coherence,
            "structure_score": structure_score,
            "seo_score": seo_score,
            "overall_quality": overall_quality,
        }

    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """
        Analyze content structure for quality assessment.

        Checks headings hierarchy, paragraph distribution, and formatting.
        """
        if not content:
            return {"score": 0.0, "details": {"error": "No content to analyze"}}

        # Count structural elements
        h1_count = len(re.findall(r'<h1[^>]*>|^#\s', content, re.MULTILINE))
        h2_count = len(re.findall(r'<h2[^>]*>|^##\s', content, re.MULTILINE))
        h3_count = len(re.findall(r'<h3[^>]*>|^###\s', content, re.MULTILINE))
        paragraphs = re.findall(r'<p[^>]*>.*?</p>|(?:^|\n\n)([^\n<#*-].+?)(?:\n\n|$)', content, re.DOTALL)
        lists = len(re.findall(r'<[ou]l[^>]*>|^[\*\-]\s|^\d+\.\s', content, re.MULTILINE))

        # Strip HTML for word count
        text_only = re.sub(r'<[^>]+>', '', content)
        word_count = len(text_only.split())

        # Calculate scores for each criterion
        scores = []
        details = {
            "word_count": word_count,
            "h1_count": h1_count,
            "h2_count": h2_count,
            "h3_count": h3_count,
            "paragraph_count": len(paragraphs),
            "list_count": lists,
        }

        # Heading hierarchy score (should have 1 H1, multiple H2s)
        if h1_count == 1:
            scores.append(1.0)
        elif h1_count == 0:
            scores.append(0.5)  # Missing H1
        else:
            scores.append(0.3)  # Multiple H1s is bad

        # H2 distribution (ideal: 3-7 for a standard article)
        if 3 <= h2_count <= 7:
            scores.append(1.0)
        elif 1 <= h2_count <= 10:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Content length score (ideal: 800-2500 words)
        if 800 <= word_count <= 2500:
            scores.append(1.0)
        elif 500 <= word_count <= 3500:
            scores.append(0.7)
        elif word_count < 300:
            scores.append(0.3)
        else:
            scores.append(0.5)

        # Paragraph count (ideal: 8-20 paragraphs)
        para_count = len(paragraphs)
        if 8 <= para_count <= 20:
            scores.append(1.0)
        elif 4 <= para_count <= 30:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Lists usage (good for scannability)
        if lists >= 1:
            scores.append(1.0)
        else:
            scores.append(0.6)

        final_score = sum(scores) / len(scores) if scores else 0.0
        details["component_scores"] = {
            "heading_hierarchy": scores[0] if len(scores) > 0 else 0,
            "h2_distribution": scores[1] if len(scores) > 1 else 0,
            "content_length": scores[2] if len(scores) > 2 else 0,
            "paragraph_structure": scores[3] if len(scores) > 3 else 0,
            "list_usage": scores[4] if len(scores) > 4 else 0,
        }

        return {"score": round(final_score, 3), "details": details}

    def _analyze_keyword_density(
        self, content: str, target_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze keyword density and distribution throughout content.

        Ideal density: 1-2% for primary keyword, 0.5-1% for secondary.
        """
        if not content:
            return {"overall_density": 0.0, "keywords": {}, "issues": ["No content"]}

        # Clean content for analysis
        text = re.sub(r'<[^>]+>', '', content).lower()
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)

        if total_words == 0:
            return {"overall_density": 0.0, "keywords": {}, "issues": ["No words found"]}

        word_freq = Counter(words)
        keyword_analysis = {}
        issues = []

        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            keyword_words = keyword_lower.split()

            if len(keyword_words) == 1:
                # Single word keyword
                count = word_freq.get(keyword_lower, 0)
            else:
                # Multi-word phrase - count occurrences
                count = len(re.findall(
                    r'\b' + re.escape(keyword_lower) + r'\b',
                    text
                ))

            density = (count / total_words) * 100 if total_words > 0 else 0

            # Determine if density is optimal
            if density < 0.5:
                status = "low"
                issues.append(f"Keyword '{keyword}' is underused ({density:.2f}%)")
            elif density > 3.0:
                status = "high"
                issues.append(f"Keyword '{keyword}' may be over-optimized ({density:.2f}%)")
            else:
                status = "optimal"

            keyword_analysis[keyword] = {
                "count": count,
                "density": round(density, 3),
                "status": status,
            }

        # Overall density (sum of all target keywords)
        overall_density = sum(
            kw["density"] for kw in keyword_analysis.values()
        )

        return {
            "overall_density": round(overall_density, 3),
            "keywords": keyword_analysis,
            "total_words": total_words,
            "issues": issues,
        }

    async def _analyze_semantic_coherence(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Analyze semantic coherence using embeddings.

        Measures how well paragraphs/sections relate to each other.
        """
        if not self.semantic_analyzer:
            return None

        if not content:
            return {"score": 0.0, "details": {"error": "No content"}}

        try:
            # Split content into sections (by headings or double newlines)
            sections = re.split(r'(?:<h[1-6][^>]*>|#{1,6}\s|(?:\n\n)+)', content)
            sections = [s.strip() for s in sections if s and len(s.strip()) > 50]

            if len(sections) < 2:
                return {"score": 0.85, "details": {"message": "Content too short for coherence analysis"}}

            # Generate embeddings for each section
            embeddings = await self.semantic_analyzer.embed(sections, normalize=True)

            # Compute embedding statistics
            stats = self.semantic_analyzer.compute_embedding_statistics(embeddings)

            # Coherence score based on centroid coherence (how similar sections are to overall theme)
            coherence_score = stats.get("centroid_coherence", 0.7)

            # Adjust score - very high similarity might indicate repetition
            if coherence_score > 0.95:
                coherence_score = 0.8  # Penalize for possible redundancy

            return {
                "score": round(coherence_score, 3),
                "details": {
                    "sections_analyzed": len(sections),
                    "mean_pairwise_similarity": round(stats.get("mean_pairwise_similarity", 0), 3),
                    "coherence_interpretation": self._interpret_coherence(coherence_score),
                },
            }

        except Exception as e:
            logger.error(f"Semantic coherence analysis failed: {e}", exc_info=True)
            return {"score": None, "details": {"error": str(e), "evaluated": False}}

    def _interpret_coherence(self, score: float) -> str:
        """Interpret coherence score for user understanding."""
        if score >= 0.85:
            return "Excellent - content is highly cohesive"
        elif score >= 0.7:
            return "Good - sections relate well to main theme"
        elif score >= 0.5:
            return "Fair - some sections may drift from main topic"
        else:
            return "Poor - content lacks thematic unity"

    def _calculate_seo_score(
        self,
        content: str,
        keywords: List[str],
        readability: float,
        structure: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate SEO optimization score.

        Considers: keyword placement, meta elements, readability, structure.
        """
        scores = []
        recommendations = []

        # 1. Title/H1 contains primary keyword
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>|^#\s+(.+)$', content, re.MULTILINE | re.IGNORECASE)
        h1_text = (h1_match.group(1) or h1_match.group(2) or "").lower() if h1_match else ""
        primary_keyword = keywords[0].lower() if keywords else ""

        if primary_keyword and primary_keyword in h1_text:
            scores.append(1.0)
        elif primary_keyword:
            scores.append(0.4)
            recommendations.append("Include primary keyword in H1/title")
        else:
            scores.append(0.7)  # No keyword specified

        # 2. Keyword in first paragraph
        first_para = re.search(r'<p[^>]*>(.*?)</p>|(?:^|\n\n)([^\n<#].+?)(?:\n|$)', content, re.DOTALL)
        first_para_text = (first_para.group(1) or first_para.group(2) or "").lower() if first_para else ""

        if primary_keyword and primary_keyword in first_para_text:
            scores.append(1.0)
        elif primary_keyword:
            scores.append(0.5)
            recommendations.append("Include primary keyword in first paragraph")
        else:
            scores.append(0.7)

        # 3. Readability score (ideal: 60-80 for web content)
        if 60 <= readability <= 80:
            scores.append(1.0)
        elif 50 <= readability <= 90:
            scores.append(0.8)
        elif readability < 50:
            scores.append(0.5)
            recommendations.append("Simplify content for better readability")
        else:
            scores.append(0.7)

        # 4. Structure score
        structure_score = structure.get("score", 0.5)
        scores.append(structure_score)
        if structure_score < 0.7:
            recommendations.append("Improve content structure with proper headings")

        # 5. Content length (ideal: 1000-2000 words for SEO)
        text_only = re.sub(r'<[^>]+>', '', content)
        word_count = len(text_only.split())

        if 1000 <= word_count <= 2500:
            scores.append(1.0)
        elif 500 <= word_count <= 3000:
            scores.append(0.7)
        else:
            scores.append(0.4)
            if word_count < 500:
                recommendations.append("Increase content length for better SEO")

        final_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "score": round(final_score, 3),
            "recommendations": recommendations,
            "component_scores": {
                "keyword_in_title": scores[0] if len(scores) > 0 else 0,
                "keyword_in_intro": scores[1] if len(scores) > 1 else 0,
                "readability": scores[2] if len(scores) > 2 else 0,
                "structure": scores[3] if len(scores) > 3 else 0,
                "content_length": scores[4] if len(scores) > 4 else 0,
            },
        }

    def _calculate_overall_quality(
        self,
        readability: float,
        structure: float,
        seo: float,
        semantic_coherence: Optional[float],
    ) -> Dict[str, Any]:
        """
        Calculate weighted overall quality score.

        Weights (when all metrics available):
        - Readability: 25%
        - Structure: 20%
        - SEO: 30%
        - Semantic Coherence: 25%

        If semantic_coherence is None (analysis failed), redistributes
        its weight proportionally among remaining metrics.
        """
        # Normalize readability to 0-1 scale (Flesch score is 0-100)
        readability_normalized = min(readability / 100, 1.0)

        if semantic_coherence is not None:
            weights = {
                "readability": 0.25,
                "structure": 0.20,
                "seo": 0.30,
                "semantic_coherence": 0.25,
            }
            weighted_score = (
                readability_normalized * weights["readability"]
                + structure * weights["structure"]
                + seo * weights["seo"]
                + semantic_coherence * weights["semantic_coherence"]
            )
        else:
            # Redistribute semantic_coherence weight proportionally
            weights = {
                "readability": 0.333,
                "structure": 0.267,
                "seo": 0.400,
                "semantic_coherence": 0.0,
            }
            weighted_score = (
                readability_normalized * weights["readability"]
                + structure * weights["structure"]
                + seo * weights["seo"]
            )

        # Determine grade
        if weighted_score >= 0.9:
            grade = "A+"
        elif weighted_score >= 0.85:
            grade = "A"
        elif weighted_score >= 0.8:
            grade = "B+"
        elif weighted_score >= 0.75:
            grade = "B"
        elif weighted_score >= 0.7:
            grade = "C+"
        elif weighted_score >= 0.65:
            grade = "C"
        elif weighted_score >= 0.6:
            grade = "D"
        else:
            grade = "F"

        coherence_breakdown = round(semantic_coherence * weights["semantic_coherence"], 3) if semantic_coherence is not None else None

        return {
            "score": round(weighted_score, 3),
            "grade": grade,
            "breakdown": {
                "readability": round(readability_normalized * weights["readability"], 3),
                "structure": round(structure * weights["structure"], 3),
                "seo": round(seo * weights["seo"], 3),
                "semantic_coherence": coherence_breakdown,
            },
        }

    def _readability_score_to_grade(self, score: float) -> str:
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

    async def trigger_comprehensive_analysis(self, article_id: UUID) -> Dict[str, Any]:
        """
        Run comprehensive article quality analysis.

        Performs inline deep analysis including:
        - Readability assessment with grade mapping
        - Content structure analysis (headings, paragraphs, lists)
        - Keyword density and distribution analysis
        - Semantic coherence analysis (if analyzer available)
        - SEO optimization scoring
        - Overall weighted quality score

        Args:
            article_id: Article identifier

        Returns:
            Complete analysis results

        Raises:
            HTTPException: If article not found
        """
        article = await self.articles.get_by_id(article_id, include_content=True)

        if not article:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        content = article.get("content", "")
        target_keywords = article.get("keywords", [])
        readability_score = article.get("readability_score", 0.0) or 0.0

        # Run all analysis components
        readability_grade = self._readability_score_to_grade(readability_score)
        structure_score = self._analyze_structure(content)
        keyword_analysis = self._analyze_keyword_density(content, target_keywords)
        semantic_coherence = await self._analyze_semantic_coherence(content)
        seo_score = self._calculate_seo_score(
            content=content,
            keywords=target_keywords,
            readability=readability_score,
            structure=structure_score,
        )

        overall_quality = self._calculate_overall_quality(
            readability=readability_score,
            structure=structure_score["score"],
            seo=seo_score["score"],
            semantic_coherence=semantic_coherence.get("score") if semantic_coherence else None,
        )

        return {
            "article_id": str(article_id),
            "status": "completed",
            "readability": {
                "score": readability_score,
                "grade": readability_grade,
            },
            "structure": structure_score,
            "keyword_analysis": keyword_analysis,
            "semantic_coherence": semantic_coherence,
            "seo_score": seo_score,
            "overall_quality": overall_quality,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def distribute_article(self, article_id: UUID, channels: List[str]) -> Dict[str, Any]:
        """
        Distribute article to specified channels with retry logic.

        Supported channels:
        - telegram: Send to configured Telegram channel
        - wordpress: Publish to WordPress site
        - email: Send via email distribution list
        - social: Post to social media accounts
        - rss: Add to RSS feed

        Args:
            article_id: Article identifier
            channels: Distribution channels (e.g., ["telegram", "wordpress"])

        Returns:
            Distribution result with per-channel status

        Raises:
            HTTPException: If article not found
        """
        article = await self.articles.get_by_id(article_id, include_content=True)

        if not article:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        generated_article = self._article_dict_to_generated_article(article)

        distributor = Distributor()
        delivery_confirmations = {}
        errors = []
        successful_channels: list[str] = []

        # Normalize channel names
        channels_normalized = [c.lower().strip() for c in channels]

        # Get project for WordPress distribution
        project = None
        if "wordpress" in channels_normalized:
            try:
                project_id = article.get("project_id")
                if project_id:
                    project = await self.projects.get_by_id(UUID(str(project_id)))
                    if not project:
                        logger.warning(f"Project {project_id} not found for WordPress distribution")
            except Exception as e:
                logger.error(f"Failed to retrieve project for WordPress distribution: {e}")

        for channel in channels_normalized:
            try:
                if channel == "telegram":
                    delivery_confirmations["telegram"] = {
                        "status": "skipped",
                        "reason": "Telegram distribution is not implemented",
                    }

                elif channel == "wordpress":
                    if project:
                        result = await distributor.distribute_to_wordpress(
                            generated_article, project
                        )
                        delivery_confirmations["wordpress"] = result
                        if result.get("status") == "published":
                            successful_channels.append("wordpress")
                    else:
                        delivery_confirmations["wordpress"] = {
                            "status": "skipped",
                            "reason": "Project not found or WordPress not configured",
                        }

                elif channel == "email":
                    # Email distribution via SMTP
                    import os
                    import smtplib
                    from email.mime.multipart import MIMEMultipart
                    from email.mime.text import MIMEText

                    smtp_host = os.getenv("SMTP_HOST", "")
                    smtp_port = int(os.getenv("SMTP_PORT", "587"))
                    smtp_user = os.getenv("SMTP_USER", "")
                    smtp_password = os.getenv("SMTP_PASSWORD", "")
                    email_recipients = os.getenv("EMAIL_DISTRIBUTION_LIST", "")

                    if not smtp_host or not email_recipients:
                        delivery_confirmations["email"] = {
                            "status": "skipped",
                            "reason": "SMTP not configured. Set SMTP_HOST and EMAIL_DISTRIBUTION_LIST env vars.",
                        }
                    else:
                        try:
                            recipients = [r.strip() for r in email_recipients.split(",") if r.strip()]
                            msg = MIMEMultipart("alternative")
                            msg["Subject"] = f"New Article: {generated_article.title}"
                            msg["From"] = smtp_user or f"noreply@{smtp_host}"
                            msg["To"] = ", ".join(recipients)

                            # Plain text version
                            text_body = f"New article published: {generated_article.title}\n\n{generated_article.meta_description}"
                            # HTML version
                            html_body = f"""
                            <html><body>
                            <h1>{generated_article.title}</h1>
                            <p><em>{generated_article.meta_description}</em></p>
                            <hr>
                            {generated_article.content[:2000]}
                            {'<p>... <em>Content truncated. View full article on the platform.</em></p>' if len(generated_article.content) > 2000 else ''}
                            </body></html>
                            """
                            msg.attach(MIMEText(text_body, "plain"))
                            msg.attach(MIMEText(html_body, "html"))

                            def _send_email() -> None:
                                with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                                    server.starttls()
                                    if smtp_user and smtp_password:
                                        server.login(smtp_user, smtp_password)
                                    server.sendmail(msg["From"], recipients, msg.as_string())

                            import asyncio

                            await asyncio.to_thread(_send_email)

                            delivery_confirmations["email"] = {
                                "status": "success",
                                "recipients": len(recipients),
                                "message": f"Sent to {len(recipients)} recipient(s)",
                            }
                            successful_channels.append("email")
                            logger.info(f"Email distribution sent to {len(recipients)} recipients")
                        except Exception as smtp_err:
                            logger.error(f"Email distribution failed: {smtp_err}")
                            delivery_confirmations["email"] = {
                                "status": "error",
                                "reason": str(smtp_err),
                            }
                            errors.append(f"email: {str(smtp_err)}")

                elif channel in ("social", "social_media"):
                    platforms = ["twitter", "linkedin"]  # Should come from project config
                    result = await distributor.distribute_to_social_media(
                        generated_article, platforms
                    )
                    delivery_confirmations["social"] = result
                    if result.get("status") == "success":
                        successful_channels.append("social")

                elif channel == "rss":
                    # Use project-specific feed URL or fallback to default
                    feed_url = getattr(project, "rss_feed_url", None) if project else None
                    if not feed_url:
                        # Fallback for when project config is missing
                        feed_url = f"https://example.com/projects/{article.get('project_id')}/feed"

                    result = await distributor.distribute_to_rss(
                        generated_article, feed_url
                    )
                    delivery_confirmations["rss"] = result
                    if result.get("status") == "success":
                        successful_channels.append("rss")

                else:
                    delivery_confirmations[channel] = {
                        "status": "error",
                        "reason": f"Unknown channel: {channel}",
                    }
                    errors.append(f"Unknown channel: {channel}")

            except Exception as e:
                logger.error(f"Distribution to {channel} failed: {e}")
                delivery_confirmations[channel] = {
                    "status": "error",
                    "reason": str(e),
                }
                errors.append(f"{channel}: {str(e)}")

        distributed_at = datetime.now(timezone.utc) if successful_channels else None
        if successful_channels:
            await self.articles.update(
                article_id,
                {
                    "distributed_at": distributed_at,
                    "distribution_channels": successful_channels,
                },
            )

        if errors:
            final_status = "partial_failure" if successful_channels else "failed"
        elif successful_channels:
            final_status = "success"
        else:
            final_status = "skipped"

        return {
            "article_id": str(article_id),
            "distributed": bool(successful_channels),
            "channels": successful_channels,
            "distributed_at": distributed_at,
            "status": final_status,
            "message": "Distribution completed" if not errors else f"Some channels failed: {', '.join(errors)}",
            "delivery_confirmations": delivery_confirmations,
            "requested_channels": channels_normalized,
        }

    async def get_distribution_status(self, article_id: UUID) -> Dict[str, Any]:
        """
        Get distribution status with business logic validation.

        Args:
            article_id: Article identifier

        Returns:
            Distribution status data

        Raises:
            HTTPException: If article not found
        """
        article = await self.articles.get_distribution_status(article_id)

        if not article:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        return {
            "article_id": str(article_id),
            "distributed": article["distributed_at"] is not None,
            "channels": article["distribution_channels"] or [],
            "distributed_at": article["distributed_at"],
            "delivery_confirmations": {},  # Distribution tracking not yet implemented
            "message": "Detailed delivery confirmations not yet implemented",
        }

    async def get_article_history(self, article_id: UUID) -> Dict[str, Any]:
        """
        Get article history with business logic validation.

        Args:
            article_id: Article identifier

        Returns:
            Article history data

        Raises:
            HTTPException: If article not found
        """
        history = await self.articles.get_article_history(article_id)

        if not history:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        return {
            "current_version": history["current_version"],
            "revisions": history["revisions"],
            "total_revisions": history["total_revisions"],
        }

    async def get_content_analytics(
        self,
        project_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get content analytics with business logic.

        Args:
            project_id: Optional project filter
            start_date: Analytics period start
            end_date: Analytics period end

        Returns:
            Analytics data
        """
        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(timezone.utc)

        analytics = await self.articles.get_analytics(project_id, start_date, end_date)

        return {
            "total_articles": analytics["total_articles"],
            "total_cost": analytics["total_cost"],
            "avg_generation_time": analytics["avg_generation_time"],
            "avg_quality_score": analytics["avg_quality_score"],
            "cost_per_article": analytics["cost_per_article"],
            "articles_by_day": analytics["articles_by_day"],
            "quality_trend": analytics["quality_trend"],
        }

    async def export_content(
        self,
        project_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export content data with business logic.

        Args:
            project_id: Optional project filter
            start_date: Export period start
            end_date: Export period end

        Returns:
            List of article data for export
        """
        if not start_date:
            start_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not end_date:
            end_date = datetime.now(timezone.utc)

        articles = await self.articles.export_articles(project_id, start_date, end_date)
        return articles

    async def search_articles(
        self, query: str, project_id: Optional[UUID] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search articles with business logic.

        Args:
            query: Search query
            project_id: Optional project filter
            limit: Maximum results

        Returns:
            Search results
        """
        results = await self.articles.search(query, project_id, limit)
        return results
