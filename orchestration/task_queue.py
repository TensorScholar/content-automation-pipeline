"""
Task Queue: Asynchronous Content Generation Orchestration

Celery-based task management system enabling:
- Background content generation (non-blocking API)
- Scheduled content creation (cron-like scheduling)
- Distributed processing (horizontal scaling)
- Task monitoring and result tracking

Architectural Pattern: Task Queue + Worker Pool with Result Backend
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from celery import Celery, Task, chord, group
from celery.result import AsyncResult
from celery.schedules import crontab
from kombu import Queue
from loguru import logger

from core.exceptions import WorkflowError
from core.models import GeneratedArticle, Project
from infrastructure.monitoring import MetricsCollector
from orchestration.content_agent import ContentAgent, ContentAgentConfig

# Celery application configuration
celery_app = Celery(
    "content_automation", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minutes soft limit
    worker_prefetch_multiplier=1,  # Fair task distribution
    worker_max_tasks_per_child=50,  # Restart workers to prevent memory leaks
    result_expires=86400,  # 24 hours
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,
    # Queue configuration with priority support
    task_queues=(
        Queue("critical", routing_key="critical", priority=10),
        Queue("high", routing_key="high", priority=7),
        Queue("default", routing_key="default", priority=5),
        Queue("low", routing_key="low", priority=3),
    ),
    task_default_queue="default",
    task_default_routing_key="default",
)


class ContentGenerationTask(Task):
    """
    Base task class with enhanced error handling and lifecycle hooks.

    Provides:
    - Automatic dependency injection
    - Task lifecycle logging
    - Error tracking and reporting
    - Resource cleanup
    """

    _content_agent: Optional[ContentAgent] = None
    _metrics_collector: Optional[MetricsCollector] = None

    @property
    def content_agent(self) -> ContentAgent:
        """Lazy initialization of ContentAgent with dependencies."""
        if self._content_agent is None:
            # Import here to avoid circular dependencies
            from execution.content_generator import ContentGenerator
            from execution.content_planner import ContentPlanner
            from execution.distributor import Distributor
            from execution.keyword_researcher import KeywordResearcher
            from infrastructure.database import DatabaseManager
            from infrastructure.llm_client import LLMClient
            from infrastructure.redis_client import RedisClient
            from intelligence.context_synthesizer import ContextSynthesizer
            from intelligence.decision_engine import DecisionEngine
            from intelligence.semantic_analyzer import SemanticAnalyzer
            from knowledge.project_repository import ProjectRepository
            from knowledge.rulebook_manager import RulebookManager
            from knowledge.website_analyzer import WebsiteAnalyzer
            from optimization.cache_manager import CacheManager
            from optimization.model_router import ModelRouter
            from optimization.prompt_compressor import PromptCompressor
            from optimization.token_budget_manager import TokenBudgetManager

            # Initialize infrastructure
            db = DatabaseManager()
            redis = RedisClient()
            llm = LLMClient()
            cache = CacheManager(redis)
            metrics = MetricsCollector()

            # Initialize knowledge layer
            projects = ProjectRepository(db)
            rulebook_mgr = RulebookManager(db)
            website_analyzer = WebsiteAnalyzer()

            # Initialize intelligence layer
            semantic_analyzer = SemanticAnalyzer()
            decision_engine = DecisionEngine(db, semantic_analyzer)
            context_synthesizer = ContextSynthesizer(projects, rulebook_mgr, decision_engine, cache)

            # Initialize optimization layer
            model_router = ModelRouter()
            budget_manager = TokenBudgetManager(redis, metrics)
            prompt_compressor = PromptCompressor(semantic_analyzer)

            # Initialize execution layer
            keyword_researcher = KeywordResearcher(llm, semantic_analyzer, cache)
            content_planner = ContentPlanner(
                llm, decision_engine, context_synthesizer, model_router
            )
            content_generator = ContentGenerator(
                llm,
                context_synthesizer,
                semantic_analyzer,
                model_router,
                budget_manager,
                prompt_compressor,
                metrics,
            )
            distributor = Distributor(
                telegram_bot_token=None, metrics_collector=metrics  # Load from config
            )

            # Initialize content agent
            self._content_agent = ContentAgent(
                project_repository=projects,
                rulebook_manager=rulebook_mgr,
                website_analyzer=website_analyzer,
                decision_engine=decision_engine,
                context_synthesizer=context_synthesizer,
                keyword_researcher=keyword_researcher,
                content_planner=content_planner,
                content_generator=content_generator,
                distributor=distributor,
                budget_manager=budget_manager,
                metrics_collector=metrics,
                config=ContentAgentConfig(),
            )

            self._metrics_collector = metrics

            logger.info("ContentAgent initialized in task context")

        return self._content_agent

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with logging and metrics."""
        logger.error(
            f"Task failed | task_id={task_id} | task={self.name} | "
            f"error={exc} | traceback={einfo}"
        )

        if self._metrics_collector:
            asyncio.run(
                self._metrics_collector.record_task_failure(
                    task_id=task_id, task_name=self.name, error=str(exc)
                )
            )

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success with logging."""
        logger.success(f"Task completed | task_id={task_id} | task={self.name}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(f"Task retrying | task_id={task_id} | task={self.name} | error={exc}")


@celery_app.task(
    base=ContentGenerationTask,
    bind=True,
    name="content_automation.generate_article",
    max_retries=3,
    default_retry_delay=300,  # 5 minutes
    autoretry_for=(WorkflowError,),
    retry_backoff=True,
    retry_backoff_max=3600,  # 1 hour max
    retry_jitter=True,
)
def generate_article_task(
    self,
    project_id: str,
    topic: str,
    priority: str = "high",
    custom_instructions: Optional[str] = None,
) -> Dict:
    """
    Asynchronous article generation task.

    Args:
        project_id: Target project UUID (as string)
        topic: Content topic
        priority: Generation priority level
        custom_instructions: Optional generation overrides

    Returns:
        Serialized GeneratedArticle data
    """
    logger.info(
        f"Article generation task started | task_id={self.request.id} | "
        f"project_id={project_id} | topic={topic}"
    )

    try:
        # Convert string UUID to UUID object
        project_uuid = UUID(project_id)

        # Execute content generation workflow
        article = asyncio.run(
            self.content_agent.create_content(
                project_id=project_uuid,
                topic=topic,
                priority=priority,
                custom_instructions=custom_instructions,
            )
        )

        # Serialize article for result backend
        return {
            "article_id": str(article.id),
            "project_id": str(article.project_id),
            "title": article.title,
            "word_count": article.word_count,
            "cost": article.total_cost,
            "generation_time": article.generation_time,
            "readability_score": article.readability_score,
            "distributed": article.distributed_at is not None,
            "created_at": article.created_at.isoformat(),
        }

    except Exception as e:
        logger.error(f"Article generation failed | task_id={self.request.id} | error={e}")
        raise


@celery_app.task(name="content_automation.batch_generate", bind=True)
def batch_generate_task(self, project_id: str, topics: List[str], priority: str = "high") -> Dict:
    """
    Batch article generation with parallel execution.

    Creates group of generation tasks for concurrent processing.

    Args:
        project_id: Target project UUID
        topics: List of content topics
        priority: Generation priority

    Returns:
        Summary of batch operation with task IDs
    """
    logger.info(f"Batch generation initiated | project_id={project_id} | " f"topics={len(topics)}")

    # Create task group for parallel execution
    job = group(generate_article_task.s(project_id, topic, priority) for topic in topics)

    # Execute group
    result = job.apply_async()

    return {
        "batch_id": result.id,
        "project_id": project_id,
        "total_tasks": len(topics),
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
    }


@celery_app.task(name="content_automation.scheduled_generation", bind=True)
def scheduled_generation_task(self, project_id: str, topic_generator_config: Dict) -> Dict:
    """
    Scheduled content generation (for recurring content needs).

    Can be configured with Celery Beat for automated content creation.

    Args:
        project_id: Target project
        topic_generator_config: Configuration for dynamic topic generation

    Returns:
        Generated article metadata
    """
    logger.info(f"Scheduled generation triggered | project_id={project_id}")

    # TODO: Implement topic generation logic based on config
    # For now, use provided topic or generate from trending keywords

    topic = topic_generator_config.get("topic", "Generated Topic")
    priority = topic_generator_config.get("priority", "medium")

    return generate_article_task(project_id, topic, priority)


@celery_app.task(name="content_automation.cleanup_old_results")
def cleanup_old_results_task():
    """
    Periodic cleanup of old task results.

    Removes task results older than retention period to prevent
    backend storage bloat.
    """
    logger.info("Starting cleanup of old task results")

    # Celery automatically expires results based on result_expires config
    # This task can implement additional custom cleanup logic if needed

    logger.success("Task result cleanup completed")
    return {"status": "cleaned", "timestamp": datetime.utcnow().isoformat()}


# Celery Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-results": {
        "task": "content_automation.cleanup_old_results",
        "schedule": crontab(hour=3, minute=0),  # Daily at 3 AM
    },
    # Example: Scheduled content generation
    # "weekly-content-generation": {
    #     "task": "content_automation.scheduled_generation",
    #     "schedule": crontab(day_of_week=1, hour=9, minute=0),  # Every Monday at 9 AM
    #     "args": ("project-uuid-here", {"topic": "Weekly Newsletter"}),
    # },
}


class TaskManager:
    """
    High-level interface for task queue operations.

    Provides convenient methods for:
    - Submitting tasks
    - Querying task status
    - Canceling tasks
    - Retrieving results
    """

    def __init__(self):
        self.app = celery_app
        logger.info("TaskManager initialized")

    def submit_generation(
        self,
        project_id: UUID,
        topic: str,
        priority: str = "high",
        custom_instructions: Optional[str] = None,
    ) -> str:
        """
        Submit article generation task to queue.

        Returns task ID for tracking.
        """
        task = generate_article_task.apply_async(
            args=[str(project_id), topic, priority, custom_instructions],
            queue=priority,  # Route to priority queue
            routing_key=priority,
        )

        logger.info(f"Task submitted | task_id={task.id} | priority={priority}")
        return task.id

    def submit_batch(self, project_id: UUID, topics: List[str], priority: str = "high") -> str:
        """Submit batch generation task."""
        task = batch_generate_task.apply_async(
            args=[str(project_id), topics, priority], queue=priority, routing_key=priority
        )

        logger.info(f"Batch task submitted | task_id={task.id} | topics={len(topics)}")
        return task.id

    def get_task_status(self, task_id: str) -> Dict:
        """
        Retrieve task status and metadata.

        Returns task state, progress, and result (if completed).
        """
        result = AsyncResult(task_id, app=self.app)

        status_info = {
            "task_id": task_id,
            "state": result.state,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "failed": result.failed() if result.ready() else None,
        }

        # Add result if available
        if result.ready():
            if result.successful():
                status_info["result"] = result.result
            else:
                status_info["error"] = str(result.info)

        # Add progress info if available
        if result.state == "PROGRESS":
            status_info["progress"] = result.info

        return status_info

    def cancel_task(self, task_id: str, terminate: bool = False) -> bool:
        """
        Cancel pending or running task.

        Args:
            task_id: Task identifier
            terminate: If True, forcefully terminate running task

        Returns:
            True if cancellation successful
        """
        result = AsyncResult(task_id, app=self.app)

        if terminate:
            result.revoke(terminate=True, signal="SIGKILL")
            logger.warning(f"Task terminated | task_id={task_id}")
        else:
            result.revoke()
            logger.info(f"Task cancelled | task_id={task_id}")

        return True

    def get_pending_tasks(self, queue: str = "default") -> List[Dict]:
        """
        Retrieve pending tasks in specified queue.

        Useful for monitoring queue depth and system load.
        """
        # Use Celery inspect API
        inspect = self.app.control.inspect()

        active = inspect.active()
        scheduled = inspect.scheduled()
        reserved = inspect.reserved()

        return {
            "active": active or {},
            "scheduled": scheduled or {},
            "reserved": reserved or {},
        }

    def get_worker_status(self) -> Dict:
        """
        Retrieve status of all workers.

        Returns worker statistics and health information.
        """
        inspect = self.app.control.inspect()

        stats = inspect.stats()
        active_queues = inspect.active_queues()

        return {
            "stats": stats or {},
            "active_queues": active_queues or {},
            "total_workers": len(stats) if stats else 0,
        }
