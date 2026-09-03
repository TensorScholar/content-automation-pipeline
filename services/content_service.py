"""Content Service - Business Logic Layer.

Encapsulates business operations for content management:
- Content generation and batch processing
- Quality analysis and validation
- Content revision workflows
- WordPress publication coordination
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from loguru import logger

from knowledge.article_repository import ArticleRepository
from knowledge.project_repository import ProjectRepository
from orchestration.content_agent import ContentAgent
from orchestration.task_persistence import TaskResultRepository, TaskStatus
from orchestration.task_state import normalize_db_status, reconcile_task_state

if TYPE_CHECKING:
    import numpy as np

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

    async def batch_generate_content(
        self,
        project_id: UUID,
        topics: List[str],
        priority: str = "high",
        schedule_after: Optional[datetime] = None,
        custom_instructions: Optional[str] = None,
        submitted_by_user_id: Optional[str] = None,
        model_override: Optional[str] = None,
        language: str = "fa",
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

        if not await self.projects.get_by_id(project_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Project not found",
            )

        # Import Celery task here to avoid circular imports
        from orchestration.tasks import generate_content_task

        submissions = []
        task_name = getattr(
            generate_content_task,
            "name",
            "orchestration.tasks.generate_content_task",
        )
        for topic in topics:
            task_id = str(uuid4())
            task_kwargs = {
                "project_id": str(project_id),
                "topic": topic,
                "priority": priority,
                "custom_instructions": custom_instructions,
                "submitted_by_user_id": submitted_by_user_id,
                "model_override": model_override,
                "language": language,
            }
            submissions.append((task_id, topic, task_kwargs))

        persisted_task_ids: list[str] = []
        if self.task_repo:
            try:
                for task_id, topic, task_kwargs in submissions:
                    await self.task_repo.create_task_record(
                        task_id=task_id,
                        task_name=task_name,
                        args=(str(project_id), topic, priority, custom_instructions),
                        kwargs=task_kwargs,
                        status=TaskStatus.PENDING,
                    )
                    persisted_task_ids.append(task_id)
            except Exception as exc:
                for persisted_task_id in persisted_task_ids:
                    try:
                        await self.task_repo.update_task_failure(
                            persisted_task_id,
                            f"Batch preparation failed before dispatch: {exc}",
                        )
                    except Exception:
                        pass
                logger.error(
                    f"Failed to persist batch before dispatch | project_id={project_id} | error={exc}"
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Batch could not be recorded safely. No generation tasks were dispatched.",
                ) from exc

        task_records = []
        dispatch_failures = []
        for task_id, topic, task_kwargs in submissions:
            try:
                if schedule_after:
                    celery_task = generate_content_task.apply_async(
                        task_id=task_id,
                        kwargs=task_kwargs,
                        eta=schedule_after,
                        queue=priority,
                        routing_key=priority,
                    )
                else:
                    celery_task = generate_content_task.apply_async(
                        task_id=task_id,
                        kwargs=task_kwargs,
                        queue=priority,
                        routing_key=priority,
                    )
                task_records.append(celery_task.id)
            except Exception as exc:
                dispatch_failures.append({"task_id": task_id, "topic": topic})
                if self.task_repo:
                    try:
                        await self.task_repo.update_task_failure(
                            task_id,
                            f"Broker dispatch failed: {exc}",
                        )
                    except Exception as persistence_exc:
                        logger.error(
                            f"Failed to record batch dispatch failure | "
                            f"task_id={task_id} | error={persistence_exc}"
                        )
                logger.error(
                    f"Batch task dispatch failed | task_id={task_id} | topic={topic} | error={exc}"
                )

        if not task_records and dispatch_failures:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Task queue is unavailable. All batch items were recorded as failed.",
            )

        logger.info(f"Dispatched batch generation to Celery for {len(topics)} topics | project_id={project_id}")

        return {
            "message": "Batch generation started",
            "project_id": str(project_id),
            "batch_id": ",".join(task_records),
            "batch_size": len(topics),
            "topics": topics,
            "task_ids": task_records,
            "status": "partial_failure" if dispatch_failures else "processing",
            "dispatch_failures": dispatch_failures,
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

    async def update_article_content(
        self,
        article_id: UUID,
        content: str,
        updated_by_user_id: str,
        revision_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save a manual edit while preserving the previous version."""
        normalized_content = content.strip()
        if not normalized_content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Article content cannot be empty.",
            )

        plain_text = re.sub(r"<[^>]+>", " ", normalized_content)
        word_count = len(re.findall(r"\S+", plain_text))
        normalized_note = (revision_note or "").strip() or "Manual edit"
        updated = await self.articles.update_content_with_revision(
            article_id=article_id,
            content=normalized_content,
            word_count=word_count,
            revision_note=normalized_note,
        )
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        logger.info(
            "Manual article edit saved | article_id={} | user_id={} | word_count={}",
            article_id,
            updated_by_user_id,
            word_count,
        )
        return {
            "article_id": str(article_id),
            "content": updated["content"],
            "word_count": updated["word_count"],
            "review_status": updated["review_status"],
            "updated_at": updated["updated_at"],
        }

    async def get_article_review(self, article_id: UUID) -> Dict[str, Any]:
        """Return current review state and a deterministic readiness checklist."""
        from services.draft_risk_service import DraftRiskService

        article = await self.get_article(article_id, include_content=True)
        review = await self.articles.get_review_state(article_id)
        if not review:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")

        risk = DraftRiskService().assess(article)
        content = re.sub(r"<[^>]+>", " ", str(article.get("content") or ""))
        content = re.sub(r"\s+", " ", content).strip()
        checklist = [
            {
                "id": "title",
                "label": "Title is present",
                "passed": bool(str(article.get("title") or "").strip()),
                "blocking": True,
            },
            {
                "id": "content",
                "label": "Content is ready for review",
                "passed": len(content) >= 100,
                "blocking": True,
            },
            {
                "id": "publish_risk",
                "label": "No blocking publish risks",
                "passed": not risk["blocking_issues"],
                "blocking": True,
            },
            {
                "id": "metadata",
                "label": "Search metadata is present",
                "passed": bool(str(article.get("meta_description") or "").strip()),
                "blocking": False,
            },
            {
                "id": "keywords",
                "label": "Keywords are attached",
                "passed": bool(article.get("keywords")),
                "blocking": False,
            },
        ]
        blocking_reasons = [
            item["label"] for item in checklist if item["blocking"] and not item["passed"]
        ]
        reviewer_name = review.get("reviewer_full_name") or review.get("reviewer_email")

        return {
            "article_id": str(article_id),
            "status": review.get("review_status") or "pending_review",
            "note": review.get("review_note"),
            "reviewed_by": str(review["reviewed_by"]) if review.get("reviewed_by") else None,
            "reviewer_name": reviewer_name,
            "reviewed_at": review.get("reviewed_at"),
            "updated_at": review.get("review_updated_at"),
            "can_approve": not blocking_reasons,
            "blocking_reasons": blocking_reasons,
            "checklist": checklist,
            "risk_level": risk["risk_level"],
        }

    async def review_article(
        self,
        *,
        article_id: UUID,
        action: str,
        reviewer_id: UUID,
        note: Optional[str],
    ) -> Dict[str, Any]:
        """Persist a manager review decision without triggering regeneration."""
        action_to_status = {
            "approve": "approved",
            "reject": "rejected",
            "request_changes": "changes_requested",
        }
        if action not in action_to_status:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid review action")

        normalized_note = (note or "").strip() or None
        if action in {"reject", "request_changes"} and not normalized_note:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Review feedback is required for this action",
            )

        current = await self.get_article_review(article_id)
        if action == "approve" and not current["can_approve"]:
            reasons = ", ".join(current["blocking_reasons"]) or "blocking checks"
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Article cannot be approved until these checks pass: {reasons}",
            )

        updated = await self.articles.set_review_state(
            article_id=article_id,
            review_status=action_to_status[action],
            reviewer_id=reviewer_id,
            note=normalized_note,
        )
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")
        return await self.get_article_review(article_id)

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

        # The current article payload is already represented by current_revision_id.
        # Regeneration lineage will bind the generation task to that revision in Phase 4D.

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
        details: Dict[str, Any] = {
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
        keyword_analysis: Dict[str, Dict[str, Any]] = {}
        issues: List[str] = []

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
            float(analysis["density"]) for analysis in keyword_analysis.values()
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
            embeddings = cast(
                "List[np.ndarray]",
                await self.semantic_analyzer.embed(sections, normalize=True),
            )

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
