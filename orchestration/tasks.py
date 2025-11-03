"""
Celery Tasks: Asynchronous Content Generation
==============================================

Defines Celery tasks for:
- Asynchronous content generation with error handling
- Automatic retries with exponential backoff
- Task result tracking and monitoring
- Integration with content generation workflow
- Idempotency guarantees with Redis-based deduplication
- Atomic database operations with transaction boundaries

Design Pattern: Task Queue with Retry Logic, Late Acknowledgment, and Idempotency Keys
"""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID

from celery import Task
from loguru import logger

from container import container
from core.exceptions import WorkflowError
from orchestration.celery_app import app
from orchestration.task_persistence import TaskResultRepository


def route_to_dead_letter_queue(task_id: str, task_name: str, args: tuple, kwargs: dict, exc: Exception) -> None:
    """
    Route permanently failed task to dead letter queue for manual review.
    
    After max_retries is exhausted, tasks are sent to the DLQ for:
    - Manual inspection and debugging
    - Potential replay after fixes
    - Audit trail of persistent failures
    
    Args:
        task_id: Celery task ID
        task_name: Name of the failed task
        args: Task positional arguments
        kwargs: Task keyword arguments
        exc: Exception that caused the failure
    """
    try:
        # Re-queue to dead_letter queue with error context
        app.send_task(
            "orchestration.tasks.process_dead_letter",
            args=[],
            kwargs={
                "original_task_id": task_id,
                "original_task_name": task_name,
                "original_args": args,
                "original_kwargs": kwargs,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            queue="dead_letter",
            routing_key="dead_letter",
        )
        logger.warning(
            f"Task routed to dead letter queue | task_id={task_id} | "
            f"task={task_name} | error={exc}"
        )
    except Exception as e:
        logger.error(f"Failed to route task to DLQ | task_id={task_id} | error={e}")


def generate_idempotency_key(task_name: str, *args, **kwargs) -> str:
    """
    Generate deterministic idempotency key for task deduplication.
    
    Uses MD5 hash of task name + arguments to create unique key.
    Ensures same inputs always produce same key for deduplication.
    
    Args:
        task_name: Name of the task
        *args: Positional arguments
        **kwargs: Keyword arguments (excluding runtime-specific keys)
        
    Returns:
        MD5 hash as idempotency key
    """
    # Remove runtime-specific kwargs that shouldn't affect idempotency
    filtered_kwargs = {
        k: v for k, v in kwargs.items() 
        if k not in ['task_id', 'retries', 'eta', 'countdown']
    }
    
    # Create deterministic string representation
    key_data = {
        'task': task_name,
        'args': [str(arg) for arg in args],
        'kwargs': {k: str(v) for k, v in sorted(filtered_kwargs.items())}
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


async def check_idempotency(redis_client, idempotency_key: str, ttl: int = 3600) -> bool:
    """
    Check if task has already been executed using Redis atomic operation.
    
    Uses Redis SET NX EX for distributed locking without race conditions.
    
    Args:
        redis_client: Redis client instance
        idempotency_key: Unique task identifier
        ttl: Time to live for idempotency record (seconds)
        
    Returns:
        True if task is new (can proceed), False if duplicate (skip)
    """
    redis_key = f"idempotency:{idempotency_key}"
    
    # SET NX EX: Set if Not eXists with EXpiration
    # Returns 1 if key was set (new task), 0 if key exists (duplicate)
    result = await redis_client.set(redis_key, "processing", ex=ttl, nx=True)
    return bool(result)


async def mark_task_complete(redis_client, idempotency_key: str, result: Dict, ttl: int = 86400):
    """
    Mark task as completed with result for future duplicate requests.
    
    Stores result for 24 hours to return cached response to duplicates.
    
    Args:
        redis_client: Redis client instance
        idempotency_key: Unique task identifier
        result: Task execution result
        ttl: Time to keep result (seconds, default 24 hours)
    """
    redis_key = f"idempotency:{idempotency_key}"
    result_key = f"idempotency:result:{idempotency_key}"
    
    # Store result separately for retrieval
    await redis_client.setex(result_key, ttl, json.dumps(result, default=str))
    
    # Update status to completed
    await redis_client.setex(redis_key, ttl, "completed")


async def get_cached_result(redis_client, idempotency_key: str) -> Optional[Dict]:
    """
    Retrieve cached result for duplicate task request.
    
    Args:
        redis_client: Redis client instance
        idempotency_key: Unique task identifier
        
    Returns:
        Cached result dict or None if not found
    """
    result_key = f"idempotency:result:{idempotency_key}"
    cached = await redis_client.get(result_key)
    
    if cached:
        return json.loads(cached)
    return None


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
        """Handle task failure with comprehensive logging, metrics, persistence, and DLQ routing."""
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
            db_manager = container.database()
        except Exception as e:
            logger.warning(f"Failed to load container dependencies for failure logging: {e}")
            metrics_collector = None
            db_manager = None

        # Record failure metrics if available (synchronous call)
        if metrics_collector:
            try:
                metrics_collector.record_workflow_completion(
                    project_id=kwargs.get("project_id", "unknown"),
                    workflow_type="content_generation",
                    duration_seconds=0,
                    cost=0.0,
                    success=False,
                    error_type=type(exc).__name__,
                )
            except Exception as e:
                logger.warning(f"Failed to record failure metrics: {e}")
        
        # Persist task failure to database
        if db_manager:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                task_repo = TaskResultRepository(db_manager)
                loop.run_until_complete(
                    task_repo.update_task_failure(
                        task_id=task_id,
                        error=str(exc),
                        traceback=str(einfo),
                    )
                )
                loop.close()
            except Exception as e:
                logger.warning(f"Failed to persist task failure: {e}")
        
        # Route to dead letter queue if max retries exhausted
        if self.request.retries >= self.max_retries:
            logger.critical(
                f"Task permanently failed after {self.max_retries} retries | "
                f"task_id={task_id} | routing to DLQ"
            )
            route_to_dead_letter_queue(task_id, self.name, args, kwargs, exc)

    def on_success(self, retval, task_id, args, kwargs):
        """Handle successful task completion with metrics and persistence."""
        logger.info(
            f"Task succeeded | task_id={task_id} | task={self.name} | "
            f"result_size={len(str(retval))}"
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
            db_manager = container.database()
        except Exception as e:
            logger.warning(f"Failed to load container dependencies for success logging: {e}")
            metrics_collector = None
            db_manager = None

        # Record success metrics if available
        if metrics_collector:
            try:
                metrics_collector.record_workflow_completion(
                    project_id=kwargs.get("project_id", "unknown"),
                    workflow_type="content_generation",
                    duration_seconds=getattr(retval, "duration", 0),
                    cost=getattr(retval, "cost", 0.0),
                    success=True,
                )
            except Exception as e:
                logger.warning(f"Failed to record success metrics: {e}")
        
        # Persist task success to database
        if db_manager:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                task_repo = TaskResultRepository(db_manager)
                loop.run_until_complete(
                    task_repo.update_task_success(
                        task_id=task_id,
                        result=retval,
                    )
                )
                loop.close()
            except Exception as e:
                logger.warning(f"Failed to persist task success: {e}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry with structured logging and persistence."""
        logger.warning(
            f"Task retrying | task_id={task_id} | task={self.name} | "
            f"error={exc} | attempt={self.request.retries}"
        )
        
        # Persist retry event to database
        try:
            import os
            import sys

            # Add project root to Python path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from container import container

            db_manager = container.database()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            task_repo = TaskResultRepository(db_manager)
            loop.run_until_complete(
                task_repo.increment_retry_count(task_id=task_id)
            )
            loop.close()
        except Exception as e:
            logger.warning(f"Failed to persist retry event: {e}")


@app.task(
    base=ContentGenerationBaseTask,
    bind=True,
    name="orchestration.tasks.generate_content_task",
    acks_late=True,  # Acknowledge only after successful completion
    autoretry_for=(Exception,),  # Auto-retry on any exception
    retry_kwargs={"max_retries": 3},  # Maximum 3 retry attempts
    retry_backoff=True,  # Use exponential backoff between retries
    retry_backoff_max=600,  # Maximum backoff of 10 minutes (production best practice)
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
        
        # --- IDEMPOTENCY CHECK ---
        # Generate deterministic idempotency key
        idempotency_key = generate_idempotency_key(
            self.name, project_id, topic, priority, custom_instructions
        )
        
        # Get Redis client for idempotency checks
        redis_client = container.redis_client()
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Check if task already processed
        is_new_task = loop.run_until_complete(
            check_idempotency(redis_client, idempotency_key, ttl=3600)
        )
        
        if not is_new_task:
            # Task is duplicate - check for cached result
            logger.info(
                f"Duplicate task detected | task_id={self.request.id} | "
                f"idempotency_key={idempotency_key}"
            )
            
            cached_result = loop.run_until_complete(
                get_cached_result(redis_client, idempotency_key)
            )
            
            if cached_result:
                logger.info(
                    f"Returning cached result | task_id={self.request.id} | "
                    f"original_task={cached_result.get('task_id')}"
                )
                loop.close()
                return cached_result
            
            # No cached result yet, task may still be processing
            logger.warning(
                f"Duplicate task without cached result | task_id={self.request.id}"
            )
        
        # --- PERSIST TASK START TO DATABASE ---
        db_manager = container.database()
        task_repo = TaskResultRepository(db_manager)
        
        loop.run_until_complete(
            task_repo.create_task_record(
                task_id=self.request.id,
                task_name=self.name,
                args=[project_id, topic],
                kwargs={"priority": priority, "custom_instructions": custom_instructions},
                idempotency_key=idempotency_key,
            )
        )
        
        # --- TASK EXECUTION WITH ATOMIC TRANSACTION ---
        # Get the agent from the already-initialized container
        content_agent = container.content_agent()

        # Execute content generation with transaction boundary
        article = loop.run_until_complete(
            content_agent.create_content(
                project_id=project_uuid,
                topic=topic,
                priority=priority,
                custom_instructions=custom_instructions,
            )
        )

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
            "idempotency_key": idempotency_key,
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
        
        # --- CACHE RESULT FOR IDEMPOTENCY ---
        loop.run_until_complete(
            mark_task_complete(redis_client, idempotency_key, result, ttl=86400)
        )
        loop.close()

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


@app.task(
    name="content_automation.analyze_website",
    bind=True,
    acks_late=True,
)
def analyze_website_task(self, project_id: str, domain: str) -> Dict:
    """
    Asynchronous website analysis task to infer content patterns.
    """
    logger.info(
        f"Website analysis task started | task_id={self.request.id} | project_id={project_id} | domain={domain}"
    )
    try:
        from uuid import UUID

        from container import container

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        analyzer = container.website_analyzer()
        inferred = loop.run_until_complete(analyzer.analyze_website(UUID(project_id), domain))
        loop.close()

        return {
            "task_id": self.request.id,
            "status": "completed",
            "id": str(inferred.id) if inferred else None,
            "project_id": project_id,
            "avg_sentence_length": getattr(inferred, "avg_sentence_length", None),
            "lexical_diversity": getattr(inferred, "lexical_diversity", None),
            "readability_score": getattr(inferred, "readability_score", None),
            "confidence": getattr(inferred, "confidence", None),
            "sample_size": getattr(inferred, "sample_size", None),
            "analyzed_at": getattr(inferred, "analyzed_at", None).isoformat()
            if inferred and getattr(inferred, "analyzed_at", None)
            else None,
        }
    except Exception as e:
        logger.error(f"Website analysis task failed | task_id={self.request.id} | error={str(e)}")
        raise


@app.task(
    name="orchestration.tasks.process_dead_letter",
    bind=True,
    acks_late=True,
    max_retries=0,  # DLQ tasks don't retry - manual intervention required
    queue="dead_letter",
)
def process_dead_letter(
    self,
    original_task_id: str,
    original_task_name: str,
    original_args: list,
    original_kwargs: dict,
    error: str,
    error_type: str,
) -> Dict:
    """
    Process tasks in the dead letter queue.
    
    This task serves as a collection point for permanently failed tasks.
    These tasks require manual inspection and potential fixes before replay.
    
    Typical reasons for DLQ routing:
    - Exhausted all retry attempts (3+)
    - Persistent external service failures (API timeouts, rate limits)
    - Data validation errors requiring code changes
    - Resource exhaustion (memory, CPU)
    
    Manual Actions:
    1. Query task_results table: SELECT * FROM task_results WHERE task_id = '<original_task_id>'
    2. Inspect error and traceback for root cause
    3. Fix underlying issue (code bug, config, external service)
    4. Replay task: app.send_task(original_task_name, args=original_args, kwargs=original_kwargs)
    
    Args:
        original_task_id: ID of the failed task
        original_task_name: Name of the failed task
        original_args: Original task arguments
        original_kwargs: Original task keyword arguments
        error: Error message from failure
        error_type: Exception class name
        
    Returns:
        Dict with DLQ entry metadata
    """
    logger.critical(
        f"Dead letter queue entry | dlq_task_id={self.request.id} | "
        f"original_task_id={original_task_id} | task={original_task_name} | "
        f"error_type={error_type} | error={error}"
    )
    
    # Persist DLQ entry to database for manual review
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        db_manager = container.database()
        task_repo = TaskResultRepository(db_manager)
        
        # Create DLQ record with reference to original task
        loop.run_until_complete(
            task_repo.create_task_record(
                task_id=self.request.id,
                task_name="dead_letter_queue",
                args=[original_task_id, original_task_name],
                kwargs={
                    "original_args": original_args,
                    "original_kwargs": original_kwargs,
                    "error": error,
                    "error_type": error_type,
                },
                idempotency_key=f"dlq:{original_task_id}",
            )
        )
        
        # Mark as REVOKED status to indicate manual intervention needed
        loop.run_until_complete(
            task_repo.update_task_failure(
                task_id=self.request.id,
                error=f"Original task {original_task_id} permanently failed: {error}",
                traceback=f"Error type: {error_type}",
            )
        )
        
        loop.close()
    except Exception as e:
        logger.error(f"Failed to persist DLQ entry | dlq_task_id={self.request.id} | error={e}")
    
    return {
        "dlq_task_id": self.request.id,
        "original_task_id": original_task_id,
        "original_task_name": original_task_name,
        "error_type": error_type,
        "error": error,
        "status": "awaiting_manual_review",
        "created_at": datetime.utcnow().isoformat(),
        "instructions": "Query task_results table and replay after fixing root cause",
    }
