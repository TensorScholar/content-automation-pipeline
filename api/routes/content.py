"""
Content Routes: Generation, Management, and Analytics

Comprehensive content lifecycle management:
- Single and batch generation
- Quality metrics and analysis
- Content revision and iteration
- Distribution management
- Performance analytics

Design Pattern: Command Query Responsibility Segregation (CQRS)
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

# Import dependency functions from new dependencies module
from api.dependencies import (
    get_content_service,
    get_project_repository,
    get_publishing_service,
    get_redis,
    get_task_result_repository,
)
from config.settings import settings
from core.exceptions import WorkflowError
from core.models import ContentPlan, GeneratedArticle
from infrastructure.database import DatabaseManager
from infrastructure.llm_options import validate_model_available
from infrastructure.redis_client import RedisClient
from knowledge.article_repository import ArticleRepository
from knowledge.project_repository import ProjectRepository
from orchestration.content_agent import ContentAgent
from orchestration.task_persistence import TaskResultRepository, TaskStatus
from orchestration.task_state import normalize_db_status, reconcile_task_state
from security import User, get_current_active_user, is_manager_user
from services.content_service import ContentService
from services.publishing_service import PublishingService

router = APIRouter(prefix="/content", tags=["Content"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================


class BatchGenerateRequest(BaseModel):
    """Command: Batch content generation."""

    project_id: UUID
    topics: List[str] = Field(..., min_length=1, max_length=20)  # M-8: aligned with UI BATCH_LIMIT=20
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")
    schedule_after: Optional[datetime] = Field(None, description="Delayed execution")
    # L-3/C-1 fix: shared instructions (language, keyword) passed from bulk queue UI
    custom_instructions: Optional[str] = Field(None, max_length=2000, description="Shared instructions for all topics")
    model_override: Optional[str] = Field(None, max_length=120, description="Optional LLM model override")
    language: str = Field("fa", pattern="^(en|fa)$", description="Content language")


class ContentRevisionRequest(BaseModel):
    """Command: Request content revision."""

    feedback: str = Field(..., min_length=10, description="Revision feedback")
    sections_to_revise: Optional[List[str]] = Field(None, description="Specific sections")
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")


class ContentQualityMetrics(BaseModel):
    """Query result: Detailed quality metrics."""

    article_id: str
    readability_score: float
    readability_grade: str
    keyword_density: dict
    semantic_coherence: float
    structure_score: float
    seo_score: float
    overall_quality: float


class ContentHistoryResponse(BaseModel):
    """Query result: Article with revision history."""

    current_version: dict
    revisions: List[dict]
    total_revisions: int


class DistributionStatusResponse(BaseModel):
    """Query result: Distribution status."""

    article_id: str
    distributed: bool
    channels: List[str]
    distributed_at: Optional[datetime]
    delivery_confirmations: dict


class ContentAnalyticsResponse(BaseModel):
    """Query result: Content performance analytics."""

    total_articles: int
    total_cost: float
    avg_generation_time: float
    avg_quality_score: float
    cost_per_article: float
    articles_by_day: List[dict]
    quality_trend: List[dict]


def _build_jsonld_schema(title: str, content: str) -> dict:
    """
    Build a lightweight JSON-LD payload (FAQ or HowTo) from article text.
    """
    plain = re.sub(r"<[^>]+>", " ", content or "")
    plain = re.sub(r"\s+", " ", plain).strip()
    title_safe = title or "Article"

    # FAQ extraction: simple question-based heuristic.
    questions = [q.strip() for q in re.findall(r"([^?.!]+[?؟])", plain) if len(q.strip()) > 8][:3]
    if questions:
        return {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": q,
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": f"See the article section that addresses: {q}",
                    },
                }
                for q in questions
            ],
        }

    # HowTo fallback from sentence chunks.
    chunks = [c.strip() for c in re.split(r"[.!؟?]", plain) if len(c.strip()) > 20][:5]
    if chunks:
        return {
            "@context": "https://schema.org",
            "@type": "HowTo",
            "name": title_safe,
            "step": [
                {
                    "@type": "HowToStep",
                    "position": idx + 1,
                    "name": step[:90],
                    "text": step,
                }
                for idx, step in enumerate(chunks)
            ],
        }

    return {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title_safe,
    }


def _parse_json_object(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _task_submitted_by_user_id(task: dict | None) -> str | None:
    if not task:
        return None
    kwargs = _parse_json_object(task.get("kwargs"))
    submitted_by = kwargs.get("submitted_by_user_id")
    return str(submitted_by) if submitted_by else None


def _can_access_task(task: dict | None, user: User) -> bool:
    if is_manager_user(user):
        return True
    return _task_submitted_by_user_id(task) == str(user.id)


def _assert_task_access(task: dict | None, user: User) -> None:
    if not is_manager_user(user) and (not task or not _can_access_task(task, user)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")


def _safe_task_error(raw_error: object) -> tuple[str, str | None]:
    """Return user-safe task error text plus optional machine error code."""
    message = str(raw_error or "Unknown error")
    normalized = message.lower()

    if any(marker in normalized for marker in ("credit", "billing", "quota", "resource_exhausted")):
        return (
            "AI provider quota or credits are exhausted. Ask a manager to select another configured model or refresh provider billing/quota.",
            "LLM_PROVIDER_QUOTA_EXHAUSTED",
        )
    if "model" in normalized and ("unavailable" in normalized or "not found" in normalized or "not allowed" in normalized):
        return (
            "The selected AI model is unavailable for this account. Ask a manager to choose another configured model.",
            "LLM_MODEL_UNAVAILABLE",
        )
    if any(marker in normalized for marker in ("403", "forbidden", "permission denied", "provider denied access")):
        return (
            "AI provider access is denied. Ask a manager to verify the API key, enabled API access, and network or regional restrictions.",
            "LLM_PROVIDER_ACCESS_DENIED",
        )
    if "authentication" in normalized or "api key" in normalized:
        return (
            "AI provider access is not configured correctly. Ask a manager to verify the API key.",
            "LLM_AUTHENTICATION_FAILED",
        )

    return message, None


# ============================================================================
# CONTENT GENERATION
# ============================================================================


class GenerateContentRequest(BaseModel):
    """Command: Generate single article asynchronously."""

    project_id: UUID
    topic: str = Field(..., min_length=3, max_length=500)
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")
    custom_instructions: Optional[str] = Field(None, alias="additional_instructions", description="Custom generation instructions")

    # Extended fields from dashboard
    word_count_range: Optional[str] = Field(None, description="Word count range (e.g., '1000-1500')")
    target_word_count: Optional[int] = Field(None, description="Legacy target word count")
    tone: Optional[str] = Field(None, max_length=100, description="Writing tone")
    primary_keyword: Optional[str] = Field(None, max_length=200, description="Primary SEO keyword")
    secondary_keywords: Optional[List[str]] = Field(None, max_length=20, description="Secondary keywords")
    meta_description: Optional[str] = Field(None, max_length=320, description="SEO meta description")
    target_audience: Optional[str] = Field(None, max_length=200, description="Target audience")
    content_goal: Optional[str] = Field(None, max_length=200, description="Content goal")
    content_structure: Optional[str] = Field(None, max_length=100, description="Article structure type")

    # New SEO Settings Object
    seo_settings: Optional[dict] = Field(default_factory=dict, description="Advanced SEO and generation settings")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM Creativity slider")
    model_override: Optional[str] = Field(None, max_length=120, description="Optional LLM model override")

    # Legacy flat fields (kept for backward compatibility)
    include_faq: Optional[bool] = Field(False)
    include_toc: Optional[bool] = Field(False)
    include_technical_depth: Optional[bool] = Field(False)
    include_cta: Optional[bool] = Field(False)
    language: str = Field("fa", pattern="^(en|fa)$", description="Content language: fa=Persian, en=English")


@router.post(
    "/generate/async",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate content asynchronously",
)
async def generate_content_async(
    request: GenerateContentRequest,
    user: User = Depends(get_current_active_user),
    task_repo: TaskResultRepository = Depends(get_task_result_repository),
    redis_client: RedisClient = Depends(get_redis),
    project_repo: ProjectRepository = Depends(get_project_repository),
):
    """
    Generate a single article asynchronously using Celery.
    """
    from orchestration.tasks import generate_content_task

    logger.info(
        f"Async content generation requested | project_id={request.project_id} | "
        f"topic={request.topic} | priority={request.priority}"
    )

    valid_model, model_error = validate_model_available(request.model_override)
    if not valid_model:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=model_error)

    if not await project_repo.get_by_id(request.project_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    task_id = str(uuid4())
    idem_key = None
    idempotency_reserved = False

    # Reserve the submission before broker dispatch. The value is the durable
    # task ID so operators can correlate Redis, PostgreSQL, and Celery state.
    import hashlib
    try:
        idem_key = "idem:gen:" + hashlib.sha256(
            f"{request.project_id}:{request.topic.strip().lower()}:{user.id}".encode()
        ).hexdigest()
        idempotency_reserved = await redis_client.set(idem_key, task_id, nx=True, ex=86400)
        if not idempotency_reserved:
            logger.warning(f"Duplicate submission blocked | idem_key={idem_key}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A task for this topic is already in progress. Duplicate submissions are blocked.",
            )
    except HTTPException:
        raise
    except Exception as e:
        # Redis unavailable — log and allow through (do not block on Redis failure)
        logger.warning(f"H-7: Idempotency check skipped (Redis unavailable): {e}")

    # Dispatch Celery task with extended parameters
    task_kwargs = {
        "word_count_range": request.word_count_range,
        "target_word_count": request.target_word_count,
        "tone": request.tone,
        "primary_keyword": request.primary_keyword,
        "secondary_keywords": request.secondary_keywords,
        "meta_description": request.meta_description,
        "target_audience": request.target_audience,
        "content_goal": request.content_goal,
        "content_structure": request.content_structure,
        "seo_settings": request.seo_settings,
        "include_faq": request.include_faq,
        "include_toc": request.include_toc,
        "include_technical_depth": request.include_technical_depth,
        "include_cta": request.include_cta,
        "language": request.language,
        "temperature": request.temperature,
        "model_override": request.model_override,
        "submitted_by_user_id": str(user.id),
        "submission_idempotency_key": idem_key if idempotency_reserved else None,
    }

    task_args = [
        str(request.project_id),
        request.topic,
        request.priority,
        request.custom_instructions,
    ]

    try:
        await task_repo.create_task_record(
            task_id=task_id,
            task_name=generate_content_task.name,
            args=tuple(task_args),
            kwargs=task_kwargs,
            status=TaskStatus.PENDING,
        )
    except Exception as exc:
        if idempotency_reserved and idem_key:
            await redis_client.delete(idem_key)
        logger.error(f"Failed to persist queued task | task_id={task_id} | error={exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task could not be recorded safely. Please retry.",
        ) from exc

    try:
        task = generate_content_task.apply_async(
            task_id=task_id,
            args=task_args,
            kwargs=task_kwargs,
            queue=request.priority,
            routing_key=request.priority,
        )
    except Exception as exc:
        await task_repo.update_task_failure(task_id, f"Broker dispatch failed: {exc}")
        if idempotency_reserved and idem_key:
            await redis_client.delete(idem_key)
        logger.error(f"Failed to dispatch generation task | task_id={task_id} | error={exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue is unavailable. The request was recorded as failed.",
        ) from exc

    logger.info(
        f"Content generation task dispatched | task_id={task.id} | "
        f"project_id={request.project_id}"
    )

    return {
        "task_id": task.id,
        "status": "queued",
        "project_id": str(request.project_id),
        "topic": request.topic,
        "priority": request.priority,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status_endpoint": f"/content/task/{task.id}",
    }


@router.get(
    "/task/{task_id}",
    response_model=dict,
    summary="Get task status",
)
async def get_task_status(
    task_id: str,
    user: User = Depends(get_current_active_user),
    task_repo: TaskResultRepository = Depends(get_task_result_repository),
):
    """
    Query the status of an asynchronous content generation task.

    Simple strategy: Trust Celery for current state, use DB for results.
    """
    from celery.result import AsyncResult

    from orchestration.celery_app import app

    # Get Celery state
    celery_result = AsyncResult(task_id, app=app)
    celery_state = celery_result.state
    db_task = None
    db_state = None

    try:
        db_task = await task_repo.get_task_by_id(task_id)
        if db_task:
            db_state = normalize_db_status(str(db_task.get("status", "")))
    except Exception as e:
        logger.error(f"DB state lookup failed | task_id={task_id} | error={e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task status is temporarily unavailable.",
        ) from e

    _assert_task_access(db_task, user)

    state, state_source = reconcile_task_state(celery_state=celery_state, db_status=db_state)

    db_result = None
    if db_task:
        raw_db_result = db_task.get("result")
        if isinstance(raw_db_result, dict):
            db_result = raw_db_result
        elif isinstance(raw_db_result, str):
            try:
                parsed = json.loads(raw_db_result)
                db_result = parsed if isinstance(parsed, dict) else None
            except Exception:
                db_result = None

    # Base response
    response = {
        "task_id": task_id,
        "state": state,
        "ready": state in ("SUCCESS", "FAILURE", "REVOKED"),
        "state_source": state_source,
    }

    # Handle different states
    if state == "SUCCESS":
        response["status"] = "Task completed successfully"
        if db_result is not None:
            response["result"] = db_result
            response["completed_at"] = db_task.get("end_time") if db_task else None
            response["project_id"] = db_result.get("project_id")
        else:
            response["result"] = celery_result.result
            if isinstance(celery_result.result, dict):
                response["project_id"] = celery_result.result.get("project_id")

    elif state == "FAILURE":
        response["status"] = "Task failed"
        if db_task and db_task.get("error"):
            raw_error = db_task.get("error")
        else:
            raw_error = str(celery_result.info) if celery_result.info else "Unknown error"

        safe_error, error_code = _safe_task_error(raw_error)
        response["error"] = safe_error
        if error_code:
            response["error_code"] = error_code
        if is_manager_user(user) and str(raw_error) != safe_error:
            response["manager_error_detail"] = str(raw_error)

    elif state == "RETRY":
        response["status"] = "Task is retrying after a transient error"
        response["progress"] = 60
        if db_task and db_task.get("retry_count") is not None:
            response["retry_count"] = db_task.get("retry_count")
        if db_task and db_task.get("error"):
            safe_error, error_code = _safe_task_error(db_task.get("error"))
            response["last_error"] = safe_error
            if error_code:
                response["error_code"] = error_code

    elif state == "STARTED":
        response["status"] = "Task is being processed"
        response["progress"] = 50

    elif state == "PENDING":
        response["status"] = "Task is queued"

    else:
        response["status"] = f"Task status: {state.lower()}"

    return response


@router.get(
    "/task/{task_id}/events",
    summary="Stream task state via SSE",
)
async def stream_task_status_events(
    task_id: str,
    request: Request,
    user: User = Depends(get_current_active_user),
    task_repo: TaskResultRepository = Depends(get_task_result_repository),
):
    """
    Stream near real-time task state updates using Server-Sent Events (SSE).
    """
    def _sse_data(payload: dict) -> str:
        return f"data: {json.dumps(payload, default=str, ensure_ascii=False)}\n\n"

    async def event_stream():
        last_payload = ""
        stream_started = time.monotonic()
        heartbeat_at = stream_started

        while True:
            # Stop polling when client disconnects to avoid orphaned async loops in production.
            if await request.is_disconnected():
                logger.info(f"SSE client disconnected | task_id={task_id}")
                break

            payload = await get_task_status(task_id=task_id, user=user, task_repo=task_repo)
            serialized = json.dumps(payload, default=str, ensure_ascii=False)

            if serialized != last_payload:
                yield f"data: {serialized}\n\n"
                last_payload = serialized

            now = time.monotonic()
            if payload.get("ready"):
                yield _sse_data({"event": "complete", "task_id": task_id})
                break

            if now - heartbeat_at >= 15:
                yield _sse_data({"event": "heartbeat", "task_id": task_id})
                heartbeat_at = now

            if now - stream_started > 3600:
                yield _sse_data({"event": "timeout", "task_id": task_id})
                break

            await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/tasks",
    response_model=List[dict],
    summary="Get all tasks",
)
async def get_all_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    user: User = Depends(get_current_active_user),
    task_repo: TaskResultRepository = Depends(get_task_result_repository),
):
    """List all tasks (alias for /tasks/history)."""
    return await get_task_history(skip, limit, search, user, task_repo)


@router.get(
    "/tasks/history",
    response_model=List[dict],
    summary="Get task history",
)
async def get_task_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    search: Optional[str] = Query(None),
    user: User = Depends(get_current_active_user),
    task_repo: TaskResultRepository = Depends(get_task_result_repository),
):
    """
    Retrieve task history from database.
    Returns list of all tasks with their status, timestamps, results, and extracted topic.
    """
    from sqlalchemy import or_, select

    from orchestration.task_persistence import task_results_table

    try:
        query = select(task_results_table).order_by(
            task_results_table.c.created_at.desc()
        )

        if search:
            query = query.where(
                or_(
                    task_results_table.c.task_id.ilike(f"%{search}%"),
                    task_results_table.c.task_name.ilike(f"%{search}%"),
                )
            )

        manager_user = is_manager_user(user)
        if manager_user:
            query = query.offset(skip).limit(limit)
        else:
            query = query.limit(max(200, min(1000, (skip + limit) * 5)))

        results = await task_repo.db.fetch_all(query)

        tasks = []
        for row in results:
            task_dict = dict(row)
            if not manager_user and not _can_access_task(task_dict, user):
                continue

            # Extract topic from args (stored as JSON)
            topic = "Unknown"
            if task_dict.get("args"):
                try:
                    args = json.loads(task_dict["args"]) if isinstance(task_dict["args"], str) else task_dict["args"]
                    # Args format: [project_id, topic, priority, custom_instructions]
                    if isinstance(args, list) and len(args) >= 2:
                        topic = args[1]  # Second argument is the topic
                except Exception:
                    pass

            task_dict["topic"] = topic

            # Normalize status to uppercase for consistent frontend handling
            if task_dict.get("status"):
                task_dict["status"] = str(task_dict["status"]).upper()

            # Convert UUID and datetime to strings for JSON serialization
            if task_dict.get("id"):
                task_dict["id"] = str(task_dict["id"])
            if task_dict.get("created_at"):
                task_dict["created_at"] = task_dict["created_at"].isoformat()
            if task_dict.get("updated_at"):
                task_dict["updated_at"] = task_dict["updated_at"].isoformat()
            if task_dict.get("start_time"):
                task_dict["start_time"] = task_dict["start_time"].isoformat()
            if task_dict.get("end_time"):
                task_dict["end_time"] = task_dict["end_time"].isoformat()

            tasks.append(task_dict)

        return tasks if manager_user else tasks[skip : skip + limit]
    except Exception as e:
        logger.error(f"Failed to fetch task history: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task history is temporarily unavailable.",
        ) from e


@router.delete(
    "/task/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete task",
)
async def delete_task(
    task_id: str,
    user: User = Depends(get_current_active_user),
    task_repo: TaskResultRepository = Depends(get_task_result_repository),
):
    """
    Delete a task from the database.

    C-4 fix: Also revokes the running Celery task (terminate=True) so it
    no longer consumes LLM tokens or worker capacity.
    """
    from celery.result import AsyncResult
    from sqlalchemy import delete

    from orchestration.celery_app import app
    from orchestration.task_persistence import task_results_table

    db_task = None
    try:
        db_task = await task_repo.get_task_by_id(task_id)
    except Exception as e:
        logger.debug(f"Task ownership lookup failed before delete: {e}")

    if not is_manager_user(user):
        if not db_task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        _assert_task_access(db_task, user)

    # C-4: Revoke Celery task first so workers stop processing
    try:
        celery_result = AsyncResult(task_id, app=app)
        state = celery_result.state
        if state not in ("SUCCESS", "FAILURE"):
            celery_result.revoke(terminate=True, signal="SIGTERM")
            logger.info(f"Celery task {task_id} revoked (was in state: {state})")
    except Exception as e:
        logger.warning(f"Could not revoke Celery task {task_id}: {e}")

    try:
        query = delete(task_results_table).where(
            task_results_table.c.task_id == task_id
        )
        await task_repo.db.execute(query)
        logger.info(f"Task {task_id} deleted by user {user.email}")
        return None

    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task. Please try again."
        )


@router.post(
    "/generate/batch",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch generate content",
)
async def batch_generate_content(
    request: BatchGenerateRequest,
    user: User = Depends(get_current_active_user),
    content_service: ContentService = Depends(get_content_service),
    redis_client: RedisClient = Depends(get_redis),
):
    """
    Generate multiple articles in parallel.

    H-3 fix: Per-user rate limit of 3 batch submissions per minute enforced
    via Redis SETNX counter. Prevents a single user from exhausting worker/LLM budget.
    """
    valid_model, model_error = validate_model_available(request.model_override)
    if not valid_model:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=model_error)

    # H-3: Per-user rate limiting
    try:
        rate_key = f"ratelimit:batch:{user.id}:" + str(int(time.time()) // 60)
        current = await redis_client.incr(rate_key)
        if current == 1:
            await redis_client.expire(rate_key, 90)
        if current > 3:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: max 3 batch submissions per minute per user.",
                headers={"Retry-After": "60"},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"H-3: Batch rate limit unavailable: {e}")
        if settings.is_production:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch generation is temporarily unavailable while rate limiting is degraded.",
            ) from e

    return await content_service.batch_generate_content(
        request.project_id,
        request.topics,
        request.priority,
        request.schedule_after,
        request.custom_instructions,
        str(user.id),
        request.model_override,
        request.language,
    )


@router.get("/batch/{batch_id}/status", response_model=dict, summary="Get batch generation status")
async def get_batch_status(
    batch_id: str,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
    task_repo: TaskResultRepository = Depends(get_task_result_repository),
):
    """
    Query batch generation progress.

    Returns aggregated status of all articles in batch including
    completion percentage, failures, and individual task statuses.
    """
    if not is_manager_user(user):
        task_ids = [task_id.strip() for task_id in batch_id.split(",") if task_id.strip()]
        if not task_ids:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found")
        for task_id in task_ids:
            try:
                task = await task_repo.get_task_by_id(task_id)
            except Exception as exc:
                logger.error(f"Failed batch ownership lookup | task_id={task_id} | error={exc}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Batch status is temporarily unavailable.",
                ) from exc
            _assert_task_access(task, user)

    return await content_service.get_batch_status(batch_id)


# ============================================================================
# ANALYTICS & REPORTING (registered before /{article_id} to avoid route shadowing)
# ============================================================================
@router.get("/analytics", response_model=ContentAnalyticsResponse, summary="Get content analytics")
async def get_content_analytics(
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Retrieve comprehensive content generation analytics.
    Returns aggregated metrics including cost analysis, quality trends,
    and production velocity over specified time period.
    """
    # Apply defaults if not provided
    if start_date is None:
        start_date = datetime.now(timezone.utc) - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    analytics = await content_service.get_content_analytics(project_id, start_date, end_date)
    return ContentAnalyticsResponse(**analytics)


@router.get("/export", summary="Export content data")
async def export_content(
    project_id: Optional[UUID] = Query(None),
    format: str = Query("json", pattern="^(json|csv)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
        Export content data in specified format.
    Supports JSON and CSV formats for integration with external systems
    or data analysis tools.
    """
    # Apply defaults if not provided
    if start_date is None:
        start_date = datetime.now(timezone.utc) - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    articles = await content_service.export_content(project_id, start_date, end_date)

    if format == "json":
        from fastapi.responses import JSONResponse

        payload = {
            "articles": articles,
            "total": len(articles),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        return JSONResponse(content=jsonable_encoder(payload))

    elif format == "csv":
        import csv
        import io

        from fastapi.responses import StreamingResponse

        # Generate CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=articles[0].keys() if articles else [])
        writer.writeheader()
        writer.writerows(articles)

        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=content_export_{datetime.now(timezone.utc).date()}.csv"
            },
        )


# ============================================================================
# SEARCH & FILTERING (registered before /{article_id} to avoid route shadowing)
# ============================================================================
@router.get("/search", response_model=List[dict], summary="Search articles")
async def search_articles(
    query: str = Query(..., min_length=1, description="Search query"),
    project_id: Optional[UUID] = Query(None, description="Filter by project"),
    limit: int = Query(20, ge=1, le=100),
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
        Full-text search across article titles and content.
    Uses PostgreSQL full-text search for efficient querying.
    Returns ranked results by relevance.
    """
    return await content_service.search_articles(query, project_id, limit)


# ============================================================================
# CONTENT MANAGEMENT
# ============================================================================


@router.get("/{article_id}", response_model=dict, summary="Get article details")
async def get_article(
    article_id: UUID,
    include_content: bool = Query(True, description="Include full content"),
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Retrieve article by ID.

    Args:
        include_content: If false, returns metadata only (faster)
    """
    return await content_service.get_article(article_id, include_content)


@router.get(
    "/{article_id}/schema/jsonld",
    response_model=dict,
    summary="Get JSON-LD schema for article",
)
async def get_article_jsonld_schema(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Generate JSON-LD schema payload for rich snippets (FAQ/HowTo fallback).
    """
    article = await content_service.get_article(article_id, include_content=True)
    title = str(article.get("title", "Article"))
    content = str(article.get("content", ""))
    schema = _build_jsonld_schema(title=title, content=content)
    return {
        "article_id": str(article_id),
        "schema": schema,
    }


@router.get(
    "/{article_id}/export/html",
    response_model=dict,
    summary="Export article as HTML with JSON-LD schema",
)
async def export_article_html_with_schema(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Export HTML payload with auto-injected JSON-LD script.
    """
    article = await content_service.get_article(article_id, include_content=True)
    title = str(article.get("title", "Article"))
    content = str(article.get("content", ""))
    schema = _build_jsonld_schema(title=title, content=content)
    schema_json = json.dumps(schema, ensure_ascii=False)

    html_payload = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script type="application/ld+json">{schema_json}</script>
</head>
<body>
  <article>
    <h1>{title}</h1>
    <div>{content}</div>
  </article>
</body>
</html>"""

    return {
        "article_id": str(article_id),
        "title": title,
        "schema": schema,
        "html": html_payload,
    }


@router.get(
    "/{article_id}/risk-assessment",
    response_model=dict,
    summary="Assess article publish risk",
)
async def get_article_risk_assessment(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Return a deterministic publish-risk assessment for a generated article.

    This v1 check is intentionally explainable and does not spend additional
    provider tokens. High-risk output should be reviewed before WordPress
    publishing; blocked output should not be published.
    """
    from services.draft_risk_service import DraftRiskService

    article = await content_service.get_article(article_id, include_content=True)
    assessment = DraftRiskService().assess(article)
    return {
        "article_id": str(article_id),
        **assessment,
    }


@router.post(
    "/{article_id}/revise",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request article revision",
)
async def revise_article(
    article_id: UUID,
    request: ContentRevisionRequest,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Request revision of existing article based on feedback.

    Creates revision task that regenerates specified sections or entire
    article incorporating feedback. Original version preserved.
    """
    return await content_service.request_article_revision(
        article_id,
        request.feedback,
        request.sections_to_revise,
        request.priority,
        str(user.id),
    )


@router.post(
    "/{article_id}/publish/wordpress",
    response_model=dict,
    summary="Publish article to WordPress",
)
async def publish_to_wordpress(
    article_id: UUID,
    project_id: UUID = Query(..., description="Project ID with WordPress configuration"),
    post_status: str = Query(
        "draft",
        description="WordPress post status: 'draft' (safe default), 'future', or explicit 'publish'",
        pattern="^(draft|future|publish)$",
    ),
    scheduled_at: Optional[datetime] = Query(
        None,
        description="Required future timestamp when post_status='future'",
    ),
    dry_run: bool = Query(False, description="Validate publish readiness without calling WordPress"),
    publishing_service: PublishingService = Depends(get_publishing_service),
    user: User = Depends(get_current_active_user),
):
    """
    Publish article to WordPress automatically.

    Requires WordPress credentials to be configured in the project settings.
    Returns the published post URL and status.
    """
    return await publishing_service.publish_to_wordpress(
        article_id=article_id,
        project_id=project_id,
        user_id=user.id,
        publish_status=post_status,
        scheduled_at=scheduled_at,
        dry_run=dry_run,
    )


@router.get(
    "/{article_id}/publish/wordpress/validate",
    response_model=dict,
    summary="Validate WordPress publish readiness",
)
async def validate_wordpress_publish_readiness(
    article_id: UUID,
    project_id: UUID = Query(..., description="Project ID with WordPress configuration"),
    post_status: str = Query("draft", pattern="^(draft|future|publish)$"),
    scheduled_at: Optional[datetime] = Query(None),
    publishing_service: PublishingService = Depends(get_publishing_service),
    user: User = Depends(get_current_active_user),
):
    del user
    return await publishing_service.validate_publish_readiness(
        article_id=article_id,
        project_id=project_id,
        publish_status=post_status,
        scheduled_at=scheduled_at,
    )


@router.get(
    "/{article_id}/publish/status",
    response_model=dict,
    summary="Get article publishing status",
)
async def get_article_publish_status(
    article_id: UUID,
    publishing_service: PublishingService = Depends(get_publishing_service),
    user: User = Depends(get_current_active_user),
):
    del user
    return await publishing_service.get_publish_status(article_id)


@router.delete("/{article_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete article")
async def delete_article(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Delete article permanently.

    This operation cannot be undone. Article is removed from all systems.
    """
    await content_service.delete_article(article_id)


# ============================================================================
# QUALITY ANALYSIS
# ============================================================================


@router.get(
    "/{article_id}/quality", response_model=ContentQualityMetrics, summary="Get quality metrics"
)
async def get_quality_metrics(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Retrieve detailed quality metrics for article.

    Analyzes readability, SEO, structure, and semantic coherence.
    Returns comprehensive quality assessment.
    """
    result = await content_service.get_quality_metrics(article_id)
    return ContentQualityMetrics(**result)


# Helper function moved to ContentService


@router.post("/{article_id}/analyze", response_model=dict, summary="Trigger comprehensive analysis")
async def trigger_comprehensive_analysis(
    article_id: UUID,
    background_tasks: BackgroundTasks,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Trigger deep analysis of article quality.
    Performs extensive quality checks including:
    - Plagiarism detection (semantic similarity to existing content)
    - SEO optimization analysis
    - Readability assessment
    - Fact-checking capabilities

    Runs asynchronously; results available via separate endpoint.
    """
    return await content_service.trigger_comprehensive_analysis(article_id)


# ============================================================================
# DISTRIBUTION MANAGEMENT
# ============================================================================
@router.post(
    "/{article_id}/distribute",
    response_model=DistributionStatusResponse,
    summary="Distribute article",
)
async def distribute_article(
    article_id: UUID,
    channels: List[str] = Query(..., description="Distribution channels"),
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Distribute article to specified channels.

    Supported channels:
    - telegram: Send to configured Telegram channel
    - wordpress: Publish to WordPress site (requires project configuration)
    - email: Send via email distribution list (requires SMTP setup)
    - social: Post to connected social media accounts
    - rss: Add to RSS feed
    """
    result = await content_service.distribute_article(article_id, channels)
    return DistributionStatusResponse(**result)


@router.get(
    "/{article_id}/distribution",
    response_model=DistributionStatusResponse,
    summary="Get distribution status",
)
async def get_distribution_status(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Query article distribution status.
    Returns delivery confirmations and channel-specific metadata.
    """
    result = await content_service.get_distribution_status(article_id)
    return DistributionStatusResponse(**result)


# ============================================================================
# CONTENT HISTORY & VERSIONING
# ============================================================================
@router.get(
    "/{article_id}/history",
    response_model=ContentHistoryResponse,
    summary="Get article revision history",
)
async def get_article_history(
    article_id: UUID,
    content_service: ContentService = Depends(get_content_service),
    user: User = Depends(get_current_active_user),
):
    """
    Retrieve complete revision history for article.
    Returns all versions with diff information and revision metadata.
    """
    result = await content_service.get_article_history(article_id)
    return ContentHistoryResponse(**result)
