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
- Admin notifications on permanent failures

Design Pattern: Task Queue with Retry Logic, Late Acknowledgment, and Idempotency Keys
"""

import asyncio
import hashlib
import json
import os
import re
import smtplib
import sys
import traceback as tb_module
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional
from uuid import UUID

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from api.dependencies import get_database, get_metrics, get_redis
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from core.exceptions import WorkflowError
from infrastructure.llm_options import validate_model_available
from orchestration.celery_app import app
from orchestration.task_persistence import TaskResultRepository, SyncTaskResultRepository

# ============================================================================
# SMTP Configuration for Admin Notifications
# ============================================================================
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")
SMTP_ENABLED = bool(SMTP_HOST and ADMIN_EMAIL)

# ============================================================================
# Task Input Validation Schemas
# ============================================================================

class GenerateContentInput(BaseModel):
    """Validation schema for generate_content_task parameters."""

    project_id: str = Field(min_length=36, max_length=36, description="Valid UUID string")
    topic: str = Field(min_length=1, max_length=500, description="Topic between 1-500 characters")
    priority: str = Field(default="high", pattern="^(critical|high|medium|low)$")
    custom_instructions: Optional[str] = Field(None, max_length=2000)
    language: str = Field(default="fa", pattern="^(en|fa)$", description="Content language: 'fa' for Persian (default), 'en' for English")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)

    @field_validator('project_id')
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Ensure project_id is valid UUID format."""
        try:
            UUID(v)  # Validate UUID format
            return v
        except ValueError:
            raise ValueError("project_id must be a valid UUID string")

    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Sanitize topic to prevent injection attacks."""
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in v if ord(char) >= 32 or char in '\n\t')
        if not sanitized.strip():
            raise ValueError("Topic cannot be empty after sanitization")
        return sanitized.strip()

    @field_validator('custom_instructions')
    @classmethod
    def validate_custom_instructions(cls, v: Optional[str]) -> Optional[str]:
        """Sanitize custom instructions if provided."""
        if v is None:
            return None
        sanitized = ''.join(char for char in v if ord(char) >= 32 or char in '\n\t')
        return sanitized.strip() if sanitized.strip() else None


class AnalyzeWebsiteInput(BaseModel):
    """Validation schema for analyze_website_task parameters."""

    project_id: str = Field(min_length=36, max_length=36, description="Valid UUID string")
    domain: str = Field(min_length=3, max_length=255, description="Valid domain name")

    @field_validator('project_id')
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Ensure project_id is valid UUID format."""
        try:
            UUID(v)
            return v
        except ValueError:
            raise ValueError("project_id must be a valid UUID string")

    @field_validator('domain')
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Sanitize domain to prevent SSRF and injection attacks."""
        # Remove whitespace and convert to lowercase
        domain = v.strip().lower()

        # Basic domain validation regex (allows subdomains)
        domain_pattern = r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.[a-z]{2,}$'

        if not re.match(domain_pattern, domain):
            raise ValueError("Invalid domain format")

        # Prevent localhost/internal access (SSRF protection)
        # Includes IPv4 private ranges, IPv6 localhost patterns, and common localhost aliases
        forbidden_patterns = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            '10.',
            '172.16.', '172.17.', '172.18.', '172.19.',
            '172.20.', '172.21.', '172.22.', '172.23.',
            '172.24.', '172.25.', '172.26.', '172.27.',
            '172.28.', '172.29.', '172.30.', '172.31.',
            '192.168.',
            '169.254.',  # Link-local
            # IPv6 localhost and private patterns
            '::1', '[::1]', '::ffff:', '[::ffff:',
            'fe80::', '[fe80:',  # Link-local IPv6
            'fc00::', '[fc00:',  # Unique local address
            'fd', '[fd',  # Unique local address prefix
        ]

        if any(domain.startswith(f) or f in domain for f in forbidden_patterns):
            raise ValueError("Access to internal/localhost domains not allowed")

        return domain


class BatchGenerateInput(BaseModel):
    """Validation schema for batch_generate_task parameters."""

    project_id: str = Field(min_length=36, max_length=36, description="Valid UUID string")
    topics: List[str] = Field(min_length=1, max_length=100, description="List of topics (1-100 items)")
    priority: str = Field(default="high", pattern="^(critical|high|medium|low)$")

    @field_validator('project_id')
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Ensure project_id is valid UUID format."""
        try:
            UUID(v)
            return v
        except ValueError:
            raise ValueError("project_id must be a valid UUID string")

    @field_validator('topics')
    @classmethod
    def validate_topics(cls, v: List[str]) -> List[str]:
        """Validate and sanitize all topics in the list."""
        if not v:
            raise ValueError("Topics list cannot be empty")

        sanitized_topics = []
        for i, topic in enumerate(v):
            if not isinstance(topic, str):
                raise ValueError(f"Topic at index {i} must be a string")

            # Validate length
            if len(topic) < 1 or len(topic) > 500:
                raise ValueError(f"Topic at index {i} must be between 1-500 characters")

            # Remove null bytes and control characters
            sanitized = ''.join(char for char in topic if ord(char) >= 32 or char in '\n\t')
            if not sanitized.strip():
                raise ValueError(f"Topic at index {i} cannot be empty after sanitization")

            sanitized_topics.append(sanitized.strip())

        return sanitized_topics


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


def notify_admin(
    task_id: str,
    task_name: str,
    error: str,
    error_type: str,
    traceback_str: str,
    kwargs: dict,
) -> bool:
    """
    Send email notification to admin on permanent task failure.

    Called when a task fails after exhausting all retry attempts.
    Requires SMTP configuration via environment variables.

    Args:
        task_id: Celery task ID
        task_name: Name of the failed task
        error: Error message
        error_type: Exception class name
        traceback_str: Full traceback for debugging
        kwargs: Task keyword arguments (for context)

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not SMTP_ENABLED:
        logger.debug(
            f"Admin notification skipped (SMTP not configured) | task_id={task_id}"
        )
        return False

    try:
        # Build email content
        subject = f"[CRITICAL] Task Failed: {task_name} ({task_id[:8]}...)"

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <h2 style="color: #dc2626;">🚨 Task Permanently Failed</h2>

            <table style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Task ID</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><code>{task_id}</code></td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Task Name</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{task_name}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Error Type</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><code>{error_type}</code></td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Error Message</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{error}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Project ID</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{kwargs.get('project_id', 'N/A')}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Timestamp</strong></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{datetime.now(timezone.utc).isoformat()}</td>
                </tr>
            </table>

            <h3>Traceback</h3>
            <pre style="background: #f3f4f6; padding: 16px; border-radius: 4px; overflow-x: auto;">
{traceback_str}
            </pre>

            <h3>Next Steps</h3>
            <ol>
                <li>Check the dead letter queue for this task</li>
                <li>Review logs for additional context</li>
                <li>Fix the underlying issue</li>
                <li>Replay the task if needed</li>
            </ol>

            <hr style="margin-top: 24px;">
            <p style="color: #6b7280; font-size: 12px;">
                This is an automated notification from the Content Automation Pipeline.
            </p>
        </body>
        </html>
        """

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER or f"alerts@{SMTP_HOST}"
        msg["To"] = ADMIN_EMAIL

        # Attach HTML content
        msg.attach(MIMEText(html_body, "html"))

        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(msg["From"], [ADMIN_EMAIL], msg.as_string())

        logger.info(
            f"Admin notification sent | task_id={task_id} | recipient={ADMIN_EMAIL}"
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to send admin notification | task_id={task_id} | error={e}"
        )
        return False



def generate_idempotency_key(task_name: str, *args, **kwargs) -> str:
    """
    Generate deterministic idempotency key for task deduplication.

    Uses SHA-256 hash of task name + arguments to create unique key.
    Ensures same inputs always produce same key for deduplication.

    Args:
        task_name: Name of the task
        *args: Positional arguments
        **kwargs: Keyword arguments (excluding runtime-specific keys)

    Returns:
        SHA-256 hash as idempotency key
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
    return hashlib.sha256(key_string.encode()).hexdigest()


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
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Corrupt idempotency cache for {idempotency_key}: {e}")
            # Delete corrupt key to prevent repeated failures
            try:
                await redis_client.delete(result_key)
            except Exception:
                pass
            return None
    return None


def _build_social_posts(title: str, topic: str, language: str = "fa") -> Dict[str, str]:
    """
    Build platform-specific social copy from article metadata.
    """
    normalized_lang = (language or "fa").lower()
    safe_title = (title or topic or "Untitled").strip()
    safe_topic = (topic or safe_title).strip()

    if normalized_lang == "fa":
        linkedin = (
            f"{safe_title}\n\n"
            f"در این مطلب، {safe_topic} را به‌صورت کاربردی بررسی کردیم.\n"
            "نظر شما چیست؟"
        )
        twitter = f"{safe_title} | نگاهی سریع به {safe_topic}.\n#تولید_محتوا #SEO"
        instagram = (
            f"✨ {safe_title}\n"
            f"موضوع: {safe_topic}\n"
            "برای مطالعه کامل، لینک را بررسی کنید."
        )
    else:
        linkedin = (
            f"{safe_title}\n\n"
            f"We just published a practical deep dive on {safe_topic}.\n"
            "What are your thoughts?"
        )
        twitter = f"{safe_title} | Quick take on {safe_topic}. #content #seo"
        instagram = (
            f"✨ {safe_title}\n"
            f"Topic: {safe_topic}\n"
            "Read the full article via the link in bio."
        )

    return {
        "linkedin": linkedin,
        "twitter": twitter,
        "instagram": instagram,
    }


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
        """
        Handle task failure with comprehensive logging, metrics, persistence,
        admin notifications, and DLQ routing.

        Enhanced error handling includes:
        - asyncio.CancelledError detection
        - Admin email notifications on permanent failures
        - Error rate metrics updates
        """
        # Detect specific error types for better reporting
        error_type = type(exc).__name__
        is_cancelled = isinstance(exc, asyncio.CancelledError)
        is_workflow_error = isinstance(exc, WorkflowError)

        logger.error(
            f"Task failed | task_id={task_id} | task={self.name} | "
            f"error_type={error_type} | is_cancelled={is_cancelled} | "
            f"error={exc} | traceback={einfo}"
        )

        # Lazy-load container and metrics_collector inside the handler
        metrics_collector = None
        try:
            import os
            import sys

            # Add project root to Python path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            metrics_collector = get_metrics()
        except Exception as e:
            logger.warning(f"Failed to load dependencies for failure logging: {e}")

        # Record failure metrics if available (synchronous call)
        if metrics_collector:
            try:
                metrics_collector.record_workflow_completion(
                    project_id=kwargs.get("project_id", "unknown"),
                    workflow_type="content_generation",
                    duration_seconds=0,
                    cost=0.0,
                    success=False,
                    error_type=error_type,
                )
                # Update error rate metric
                metrics_collector.update_error_rate(error_type, 1.0)
            except Exception as e:
                logger.warning(f"Failed to record failure metrics: {e}")

        # Persist task failure to database (using sync repository)
        try:
            task_repo = SyncTaskResultRepository()
            task_repo.update_task_failure(
                task_id=task_id,
                error=str(exc),
                traceback=str(einfo),
            )
        except Exception as e:
            logger.warning(f"Failed to persist task failure: {e}")

        # Route to dead letter queue and notify admin if max retries exhausted
        if self.request.retries >= self.max_retries:
            logger.critical(
                f"Task permanently failed after {self.max_retries} retries | "
                f"task_id={task_id} | routing to DLQ and notifying admin"
            )
            route_to_dead_letter_queue(task_id, self.name, args, kwargs, exc)

            # Send admin notification for permanent failures
            notify_admin(
                task_id=task_id,
                task_name=self.name,
                error=str(exc),
                error_type=error_type,
                traceback_str=str(einfo),
                kwargs=kwargs,
            )

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

            metrics_collector = get_metrics()
            db_manager = get_database()
        except Exception as e:
            logger.warning(f"Failed to load dependencies for success logging: {e}")
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

        # Persist task success to database (using sync repository)
        try:
            task_repo = SyncTaskResultRepository()
            task_repo.update_task_success(
                task_id=task_id,
                result=retval,
            )
        except Exception as e:
            logger.warning(f"Failed to persist task success: {e}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry with structured logging and persistence."""
        logger.warning(
            f"Task retrying | task_id={task_id} | task={self.name} | "
            f"error={exc} | attempt={self.request.retries}"
        )

        # Persist retry event to database as RETRY (non-terminal).
        try:
            task_repo = SyncTaskResultRepository()
            task_repo.update_task_retry(
                task_id=task_id,
                error=str(exc),
                traceback=str(einfo),
                retry_count=self.request.retries,
            )
        except Exception as e:
            logger.warning(f"Failed to persist retry event: {e}")


@app.task(
    base=ContentGenerationBaseTask,
    bind=True,
    name="orchestration.tasks.generate_content_task",
    acks_late=True,  # Acknowledge only after successful completion
    autoretry_for=(
        # Only retry on transient/network errors
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ),
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
    **kwargs  # Accept extended parameters (word_count, tone, keywords, etc.)
) -> Dict:
    """
    Asynchronous content generation task with robust error handling.

    This task orchestrates the complete content generation workflow:
    1. Validate and sanitize inputs (prevent injection attacks)
    2. Initialize content agent from DI container
    3. Execute content generation with all optimization layers
    4. Track metrics and costs
    5. Return serialized article data

    Args:
        project_id: Target project UUID (as string for JSON serialization)
        topic: Content topic/subject (sanitized for security)
        priority: Generation priority level (critical, high, medium, low)
        custom_instructions: Optional custom generation instructions (sanitized)

    Returns:
        Dict containing generated article metadata and content

    Raises:
        ValueError: On input validation failures
        WorkflowError: On content generation failures
        Exception: On unexpected errors (triggers auto-retry)

    Configuration:
        - acks_late=True: Task only acknowledged after completion
        - autoretry_for=(Exception,): Automatically retry on failures
        - retry_kwargs={'max_retries': 3}: Up to 3 retry attempts
        - retry_backoff=True: Exponential backoff between retries
    """
    # ===== INPUT VALIDATION (SECURITY) =====
    try:
        # Extract language from kwargs to prevent duplicate argument error downstream
        # and ensure validation logic catches invalid language codes
        language_arg = kwargs.pop("language", None)
        submitted_by_user_id = kwargs.pop("submitted_by_user_id", None)

        validated_input = GenerateContentInput(
            project_id=project_id,
            topic=topic,
            priority=priority,
            custom_instructions=custom_instructions,
            language=language_arg or "fa",
        )
    except Exception as validation_error:
        logger.error(
            f"Input validation failed | task_id={self.request.id} | "
            f"error={validation_error}"
        )
        # Don't retry validation errors - they're permanent
        raise ValueError(f"Invalid task parameters: {validation_error}")

    logger.info(
        f"Content generation task started | task_id={self.request.id} | "
        f"project_id={validated_input.project_id} | topic={validated_input.topic[:50]} | "
        f"priority={validated_input.priority}"
    )

    valid_model, model_error = validate_model_available(kwargs.get("model_override"))
    if not valid_model:
        raise ValueError(model_error)

    start_time = datetime.now(timezone.utc)

    try:
        # Convert validated string UUID to UUID object
        project_uuid = UUID(validated_input.project_id)

        # Generate deterministic idempotency key
        idempotency_key = generate_idempotency_key(
            self.name,
            validated_input.project_id,
            validated_input.topic,
            validated_input.priority,
            validated_input.custom_instructions,
        )

        # --- ASYNC WORKFLOW WRAPPER ---
        async def _async_workflow():
            logger.info(f"DEBUG: Starting async workflow in PID {os.getpid()}")

            # Get Redis client
            redis_client = get_redis()
            logger.info(f"DEBUG: Got RedisClient: {redis_client}")

            # Initialize Redis with timeout to prevent hanging
            try:
                await asyncio.wait_for(redis_client.initialize(), timeout=30)
            except asyncio.TimeoutError:
                logger.error("Redis initialization timeout after 30s")
                raise WorkflowError("Redis connection timeout")

            try:
                # Idempotency Check
                is_new_task = await check_idempotency(redis_client, idempotency_key, ttl=3600)

                if not is_new_task:
                    logger.info(f"Duplicate task detected | task_id={self.request.id}")
                    cached = await get_cached_result(redis_client, idempotency_key)
                    if cached:
                        logger.info(f"Returning cached result | task_id={self.request.id}")
                        return cached  # Returns dict
                    logger.warning(f"Duplicate task without cached result | task_id={self.request.id}")

                # --- PERSIST TASK START (Sync) ---
                try:
                    task_repo = SyncTaskResultRepository()
                    task_repo.create_task_record(
                        task_id=self.request.id,
                        task_name=self.name,
                        args=(validated_input.project_id, validated_input.topic),
                        kwargs={
                            "priority": validated_input.priority,
                            "custom_instructions": validated_input.custom_instructions,
                            "submitted_by_user_id": submitted_by_user_id,
                            "model_override": kwargs.get("model_override"),
                        },
                        idempotency_key=idempotency_key,
                    )
                except Exception as e:
                    logger.warning(f"Failed to persist task start: {e}")

                # --- PREPARE CONTEXT (Sync) ---
                from api.dependencies import get_content_agent
                from knowledge.sync_project_repository import SyncProjectRepository

                logger.info(f"Loading project context synchronously | project_id={project_id}")
                sync_repo = SyncProjectRepository()
                project = sync_repo.get_by_id(project_uuid)

                if not project:
                    raise ValueError(f"Project not found (sync load): {project_id}")

                project_context = {
                    "project": project,
                    "rulebook": None,
                    "inferred_patterns": None,
                    "decision_strategy": "best_practices",
                }

                # --- GENERATE CONTENT (Async) ---
                # CRITICAL: Clear dependency cache to ensure fresh DatabaseManager for this loop
                from api.dependencies import get_database, get_content_agent

                # Close any existing database connections from previous event loop
                try:
                    old_db = get_database()
                    old_db._is_initialized = False  # Force reinitialization
                    await old_db.close()
                except Exception as e:
                    logger.debug(f"Error closing old database: {e}")

                get_database.cache_clear()

                # CRITICAL: Initialize the new DatabaseManager instance with timeout
                db = get_database()
                try:
                    await asyncio.wait_for(db.initialize(), timeout=30)
                    logger.debug("Database reinitialized for async workflow")
                except asyncio.TimeoutError:
                    logger.error("Database initialization timeout after 30s")
                    raise WorkflowError("Database connection timeout")

                content_agent = get_content_agent()

                try:
                    article = await content_agent.create_content(
                        project_id=project_uuid,
                        topic=validated_input.topic,
                        priority=validated_input.priority,
                        custom_instructions=validated_input.custom_instructions,
                        project_context=project_context,
                        language=validated_input.language,
                        **kwargs  # Pass extended SEO parameters
                    )

                    execution_result = {
                        "status": "success",
                        "article_id": str(article.id),
                        "project_id": str(project_uuid),
                        "title": article.title,
                        "word_count": article.quality_metrics.word_count,
                        "cost": article.total_cost_usd,
                        "distributed": len(article.distribution_channels) > 0,
                        "generated_at": article.created_at.isoformat(),
                    }

                    # Trigger social media repurposing as a follow-up Celery sub-task.
                    try:
                        social_task = generate_social_posts_task.apply_async(
                            kwargs={
                                "article_id": str(article.id),
                                "title": article.title,
                                "topic": validated_input.topic,
                                "language": validated_input.language,
                            },
                            queue="default",
                            routing_key="default",
                        )
                        execution_result["social_task_id"] = social_task.id
                        execution_result["social_status"] = "queued"
                    except Exception as social_error:
                        logger.warning(
                            "Failed to dispatch social repurposing task | "
                            f"task_id={self.request.id} | error={social_error}"
                        )

                finally:
                    # Cleanup fresh database resources
                    if hasattr(content_agent, 'database_manager'):
                        await content_agent.database_manager.close()

                # --- CACHE RESULT ---
                # Use mark_task_complete instead of cache_result
                await mark_task_complete(redis_client, idempotency_key, execution_result)

                # --- UPDATE TASK RECORD (Sync) ---
                try:
                    task_repo.update_task_success(
                        task_id=self.request.id,
                        result=execution_result
                    )
                except Exception as e:
                    logger.error(f"Failed to update task status: {e}")

                logger.info(f"Task completed successfully | article_id={article.id}")
                return execution_result

            finally:
                # Cleanup Redis if needed (connection pool handles itself usually)
                pass

        # Execute workflow using existing event loop (from worker init)
        # CRITICAL FIX: Don't use asyncio.run() which creates new loop - causes conflicts
        try:
            # Get or create event loop for this thread
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Execute with timeout to prevent indefinite hangs
            # Soft timeout: 50 minutes, Hard timeout: 60 minutes (from celery config)
            execution_result = loop.run_until_complete(
                asyncio.wait_for(_async_workflow(), timeout=3000)  # 50 min timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Task execution timeout | task_id={self.request.id}")
            raise WorkflowError("Task execution exceeded 50 minute timeout")
        except Exception as e:
            logger.error(f"Async workflow failed: {e}")
            raise e

        # Handle result type
        if isinstance(execution_result, dict):
            return execution_result

        article = execution_result

        # Calculate execution time
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

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
            "content_id": str(article.id),  # Use content_id for API consistency
            "article_id": str(article.id),  # Keep for backward compatibility
            "project_id": str(article.project_id),
            "title": article.title,
            "content": article.content,
            "word_count": article.quality_metrics.word_count,
            "readability_score": article.quality_metrics.readability_score,
            "keyword_density": 0.0,
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

        # Result is now ready - no more async operations needed with asyncio.run pattern
        return result

    except WorkflowError as e:
        logger.error(
            f"Content generation workflow failed | task_id={self.request.id} | " f"error={str(e)}"
        )
        raise

    except SoftTimeLimitExceeded:
        # Gracefully handle timeout - save progress before hard limit kills the task
        logger.warning(
            f"Task soft time limit exceeded | task_id={self.request.id} | "
            f"saving partial state before termination"
        )
        try:
            task_repo = SyncTaskResultRepository()
            task_repo.update_task_failure(
                task_id=self.request.id,
                error="Task exceeded soft time limit (50 minutes). Partial work may be lost.",
                traceback=None,
            )
        except Exception as save_error:
            logger.error(f"Failed to save timeout state: {save_error}")
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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

    Raises:
        ValueError: On input validation failures
    """
    # ===== INPUT VALIDATION =====
    try:
        validated_input = BatchGenerateInput(
            project_id=project_id,
            topics=topics,
            priority=priority,
        )
    except Exception as validation_error:
        logger.error(
            f"Batch input validation failed | batch_id={self.request.id} | "
            f"error={validation_error}"
        )
        raise ValueError(f"Invalid batch parameters: {validation_error}")

    from celery import group

    logger.info(
        f"Batch generation started | batch_id={self.request.id} | "
        f"project_id={validated_input.project_id} | topics={len(validated_input.topics)}"
    )

    # Create group of tasks for parallel execution
    job = group(
        generate_content_task.s(validated_input.project_id, topic, validated_input.priority)
        for topic in validated_input.topics
    )

    # Execute group
    result = job.apply_async()

    return {
        "batch_id": self.request.id,
        "group_id": result.id,
        "project_id": validated_input.project_id,
        "total_tasks": len(validated_input.topics),
        "status": "processing",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@app.task(
    name="orchestration.tasks.generate_social_posts_task",
    bind=True,
    acks_late=True,
    autoretry_for=(
        ConnectionError,
        TimeoutError,
    ),
    retry_kwargs={"max_retries": 2},
    retry_backoff=True,
    retry_jitter=True,
    task_time_limit=120,
)
def generate_social_posts_task(
    self,
    article_id: str,
    title: str,
    topic: str,
    language: str = "fa",
) -> Dict:
    """
    Generate platform-ready social media posts from article metadata.
    """
    logger.info(
        "Social repurposing started | social_task_id=%s | article_id=%s",
        self.request.id,
        article_id,
    )
    posts = _build_social_posts(title=title, topic=topic, language=language)
    return {
        "task_id": self.request.id,
        "article_id": article_id,
        "status": "completed",
        "posts": posts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.task(
    name="content_automation.analyze_website",
    bind=True,
    acks_late=True,
    autoretry_for=(
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ),
    retry_kwargs={"max_retries": 3},
    retry_backoff=True,
    retry_backoff_max=300,  # 5 minutes max backoff
    task_time_limit=600,  # 10 minutes hard limit
)
def analyze_website_task(self, project_id: str, domain: str) -> Dict:
    """
    Asynchronous website analysis task to infer content patterns.

    Args:
        project_id: Target project UUID
        domain: Website domain to analyze (validated for SSRF protection)

    Returns:
        Dict with analysis results

    Raises:
        ValueError: On input validation failures (invalid domain, SSRF attempt)
    """
    # ===== INPUT VALIDATION (SECURITY - SSRF PROTECTION) =====
    try:
        validated_input = AnalyzeWebsiteInput(
            project_id=project_id,
            domain=domain,
        )
    except Exception as validation_error:
        logger.error(
            f"Input validation failed | task_id={self.request.id} | "
            f"error={validation_error}"
        )
        raise ValueError(f"Invalid task parameters: {validation_error}")

    logger.info(
        f"Website analysis task started | task_id={self.request.id} | "
        f"project_id={validated_input.project_id} | domain={validated_input.domain}"
    )
    loop = None
    try:
        from uuid import UUID

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        from api.dependencies import get_website_analyzer
        analyzer = get_website_analyzer()
        inferred = loop.run_until_complete(
            analyzer.analyze_website(UUID(validated_input.project_id), validated_input.domain)
        )

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
    finally:
        if loop and not loop.is_closed():
            loop.close()


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
    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        db_manager = get_database()
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
    except Exception as e:
        logger.error(f"Failed to persist DLQ entry | dlq_task_id={self.request.id} | error={e}")
    finally:
        if loop and not loop.is_closed():
            loop.close()

    return {
        "dlq_task_id": self.request.id,
        "original_task_id": original_task_id,
        "original_task_name": original_task_name,
        "error_type": error_type,
        "error": error,
        "status": "awaiting_manual_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instructions": "Query task_results table and replay after fixing root cause",
    }
