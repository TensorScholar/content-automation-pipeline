"""
Celery Tasks: Asynchronous Content Generation
==============================================

Defines Celery tasks for:
- Asynchronous content generation with error handling
- Automatic retries with exponential backoff
- Task result tracking and monitoring
- Integration with content generation workflow

Design Pattern: Task Queue with Retry Logic and Late Acknowledgment
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from celery import Task
from loguru import logger

from container import container, container_manager
from core.exceptions import WorkflowError
from orchestration.celery_app import app


class ContentGenerationBaseTask(Task):
    """
    Base task class with enhanced error handling and lifecycle hooks.

    Provides:
    - Dependency injection via container
    - Task lifecycle logging
    - Error tracking and reporting
    - Resource cleanup
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure with comprehensive logging and metrics."""
        logger.error(
            f"Task failed | task_id={task_id} | task={self.name} | "
            f"error={exc} | traceback={einfo}"
        )

        # Lazy-load container and metrics_collector inside the handler
        try:
            import os
            import sys

            # Add project root to Python path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from container import container

            metrics_collector = container.metrics_collector()
        except Exception as e:
            logger.warning(f"Failed to load metrics_collector for failure logging: {e}")
            metrics_collector = None

        # Record failure metrics if available
        if metrics_collector:
            try:
                # Run async metrics recording in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    metrics_collector.record_workflow_completion(
                        project_id=kwargs.get("project_id", "unknown"),
                        workflow_type="content_generation",
                        duration_seconds=0,
                        cost=0.0,
                        success=False,
                        error_type=type(exc).__name__,
                    )
                )
                loop.close()
            except Exception as e:
                logger.warning(f"Failed to record failure metrics: {e}")

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success with logging."""
        logger.success(f"Task completed successfully | task_id={task_id} | task={self.name}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            f"Task retrying | task_id={task_id} | task={self.name} | "
            f"error={exc} | retry={self.request.retries}"
        )


@app.task(
    base=ContentGenerationBaseTask,
    bind=True,
    name="orchestration.tasks.generate_content_task",
    acks_late=True,  # Acknowledge only after successful completion
    autoretry_for=(Exception,),  # Auto-retry on any exception
    retry_kwargs={"max_retries": 3},  # Maximum 3 retry attempts
    retry_backoff=True,  # Use exponential backoff between retries
    retry_backoff_max=3600,  # Maximum backoff of 1 hour
    retry_jitter=True,  # Add random jitter to backoff
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minutes soft limit
)
def generate_content_task(
    self,
    project_id: str,
    topic: str,
    priority: str = "high",
    custom_instructions: Optional[str] = None,
) -> Dict:
    """
    Asynchronous content generation task with robust error handling.

    This task orchestrates the complete content generation workflow:
    1. Initialize content agent from DI container
    2. Execute content generation with all optimization layers
    3. Track metrics and costs
    4. Return serialized article data

    Args:
        project_id: Target project UUID (as string for JSON serialization)
        topic: Content topic/subject
        priority: Generation priority level (critical, high, medium, low)
        custom_instructions: Optional custom generation instructions

    Returns:
        Dict containing generated article metadata and content

    Raises:
        WorkflowError: On content generation failures
        Exception: On unexpected errors (triggers auto-retry)

    Configuration:
        - acks_late=True: Task only acknowledged after completion
        - autoretry_for=(Exception,): Automatically retry on failures
        - retry_kwargs={'max_retries': 3}: Up to 3 retry attempts
        - retry_backoff=True: Exponential backoff between retries
    """
    logger.info(
        f"Content generation task started | task_id={self.request.id} | "
        f"project_id={project_id} | topic={topic} | priority={priority}"
    )

    start_time = datetime.utcnow()

    try:
        # Convert string UUID to UUID object
        project_uuid = UUID(project_id)

        # --- START NEW INITIALIZATION LOGIC ---
        # Create a new event loop for this task execution to ensure
        # a clean state for async initialization.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        content_agent = None
        try:
            # Initialize the DI container and its async resources
            # This must be run within the new event loop.
            loop.run_until_complete(container_manager.initialize())

            # Get the fully-wired ContentAgent
            content_agent = container.content_agent()
            logger.info(f"ContentAgent and DI container initialized for task {self.request.id}")

            # Execute the main async content creation workflow
            article = loop.run_until_complete(
                content_agent.create_content(
                    project_id=project_uuid,
                    topic=topic,
                    priority=priority,
                    custom_instructions=custom_instructions,
                )
            )
        finally:
            # Ensure container resources (like DB pools) are cleaned up
            loop.run_until_complete(container_manager.cleanup())
            loop.close()
            logger.info(f"Container resources cleaned up for task {self.request.id}")
        # --- END NEW INITIALIZATION LOGIC ---

        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()

        logger.success(
            f"Content generation completed | task_id={self.request.id} | "
            f"article_id={article.id} | words={article.quality_metrics.word_count} | "
            f"cost=${article.total_cost_usd:.4f} | time={execution_time:.2f}s"
        )

        # Serialize article for result backend (JSON-compatible)
        result = {
            "task_id": self.request.id,
            "status": "completed",
            "article_id": str(article.id),
            "project_id": str(article.project_id),
            "title": article.title,
            "content": article.content,
            "word_count": article.quality_metrics.word_count,
            "readability_score": article.quality_metrics.readability_score,
            "keyword_density": 0.0,  # Placeholder - not implemented yet
            "total_tokens_used": article.total_tokens_used,
            "total_cost": article.total_cost_usd,
            "generation_time": article.generation_time_seconds,
            "distributed": article.distributed_at is not None,
            "distributed_at": (
                article.distributed_at.isoformat() if article.distributed_at else None
            ),
            "created_at": article.created_at.isoformat(),
            "execution_time": execution_time,
        }

        return result

    except WorkflowError as e:
        logger.error(
            f"Content generation workflow failed | task_id={self.request.id} | " f"error={str(e)}"
        )
        raise

    except Exception as e:
        logger.error(
            f"Unexpected error in content generation | task_id={self.request.id} | "
            f"error={str(e)}"
        )
        raise


@app.task(name="orchestration.tasks.cleanup_old_results")
def cleanup_old_results():
    """
    Periodic task to clean up old task results.

    Celery automatically expires results based on result_expires config,
    but this task can implement additional cleanup logic if needed.
    """
    logger.info("Running cleanup of old task results")

    # Cleanup logic here (if needed beyond Celery's automatic expiration)
    # For example: clean up associated database records, files, etc.

    logger.success("Task result cleanup completed")

    return {
        "status": "cleaned",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.task(
    name="orchestration.tasks.batch_generate_task",
    bind=True,
    acks_late=True,
)
def batch_generate_task(self, project_id: str, topics: list, priority: str = "high") -> Dict:
    """
    Batch content generation task.

    Creates multiple content generation tasks for parallel execution.

    Args:
        project_id: Target project UUID
        topics: List of topics to generate content for
        priority: Priority level for all tasks

    Returns:
        Dict with batch metadata and task IDs
    """
    from celery import group

    logger.info(
        f"Batch generation started | batch_id={self.request.id} | "
        f"project_id={project_id} | topics={len(topics)}"
    )

    # Create group of tasks for parallel execution
    job = group(generate_content_task.s(project_id, topic, priority) for topic in topics)

    # Execute group
    result = job.apply_async()

    return {
        "batch_id": self.request.id,
        "group_id": result.id,
        "project_id": project_id,
        "total_tasks": len(topics),
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
    }
