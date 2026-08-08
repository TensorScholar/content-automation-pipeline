"""Safe WordPress publishing orchestration.

This service owns validation, article-level idempotency, durable attempts, and
safe publish status responses. The WordPress API call remains isolated in the
distributor boundary where the decrypted credential is needed.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from fastapi import HTTPException, status

from config.settings import get_settings
from core.exceptions import DistributionError
from execution.distributer import Distributor, WordPressPublishError
from knowledge.project_repository import ProjectRepository
from knowledge.publishing_repository import PublishingRepository
from services.content_service import ContentService

ALLOWED_PUBLISH_STATUSES = {"draft", "future", "publish"}
TERMINAL_SUCCESS_STATES = {
    "published_as_draft",
    "published_scheduled",
    "published_public",
}
PLACEHOLDER_PATTERNS = (
    "lorem ipsum",
    "placeholder",
    "todo:",
    "sample content",
    "dummy content",
)


@dataclass(frozen=True)
class PublishValidationResult:
    can_publish: bool
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]
    intended_publish_status: str
    target_site: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "can_publish": self.can_publish,
            "errors": self.errors,
            "warnings": self.warnings,
            "intended_publish_status": self.intended_publish_status,
            "target_site": self.target_site,
        }


class PublishingService:
    """Publish generated articles to WordPress with safety controls."""

    def __init__(
        self,
        *,
        content_service: ContentService,
        project_repository: ProjectRepository,
        publishing_repository: PublishingRepository,
        distributor: Distributor | None = None,
    ):
        self.content_service = content_service
        self.projects = project_repository
        self.publishing = publishing_repository
        # Keep fast in-process retries small; Celery provides the durable outer
        # retry boundary. This avoids retry amplification while preserving recovery.
        self.distributor = distributor or Distributor(max_retries=2, initial_retry_delay=1.0)

    @staticmethod
    def idempotency_key(
        *,
        article_id: UUID,
        project_id: UUID,
        publish_status: str,
        scheduled_at: datetime | None = None,
        content_fingerprint: str = "",
    ) -> str:
        raw = (
            f"wordpress:{project_id}:{article_id}:{publish_status}:"
            f"{scheduled_at.isoformat() if scheduled_at else ''}:{content_fingerprint}"
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"wp:{digest[:48]}"

    async def validate_publish_readiness(
        self,
        *,
        article_id: UUID,
        project_id: UUID,
        publish_status: str = "draft",
        scheduled_at: datetime | None = None,
    ) -> dict[str, Any]:
        article = await self.content_service.get_article(article_id, include_content=True)
        if str(article.get("project_id")) != str(project_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Article does not belong to the selected project",
            )

        project = await self.projects.get_by_id(project_id)
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        validation = self._validate(article, project, publish_status, scheduled_at)
        return {
            "article_id": str(article_id),
            "project_id": str(project_id),
            **validation.as_dict(),
        }

    async def _prepare_publish(
        self,
        *,
        article_id: UUID,
        project_id: UUID,
        publish_status: str,
        scheduled_at: datetime | None,
    ) -> tuple[dict[str, Any], Any, PublishValidationResult, str]:
        article = await self.content_service.get_article(article_id, include_content=True)
        if str(article.get("project_id")) != str(project_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Article does not belong to the selected project",
            )
        project = await self.projects.get_by_id(project_id)
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        validation = self._validate(article, project, publish_status, scheduled_at)
        content_fingerprint = hashlib.sha256(
            "\x1f".join(
                [
                    str(article.get("title") or ""),
                    str(article.get("content") or ""),
                    str(article.get("meta_description") or ""),
                    repr(article.get("keywords") or []),
                    str(article.get("updated_at") or article.get("created_at") or ""),
                ]
            ).encode("utf-8")
        ).hexdigest()[:32]
        key = self.idempotency_key(
            article_id=article_id,
            project_id=project_id,
            publish_status=publish_status,
            scheduled_at=scheduled_at,
            content_fingerprint=content_fingerprint,
        )
        return article, project, validation, key

    async def _reject_invalid_publish(
        self,
        *,
        article_id: UUID,
        project_id: UUID,
        user_id: UUID | None,
        publish_status: str,
        key: str,
        target_site_url: str | None,
        validation: PublishValidationResult,
    ) -> None:
        if validation.can_publish:
            return
        attempt_id = await self.publishing.record_preflight_failure(
            article_id=article_id,
            project_id=project_id,
            user_id=user_id,
            requested_publish_mode=publish_status,
            idempotency_key=key,
            target_site_url=target_site_url,
            error_category="validation_error",
            error_message="; ".join(error["message"] for error in validation.errors),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Article is not publishable",
                "attempt_id": str(attempt_id),
                **validation.as_dict(),
            },
        )

    async def queue_publish_to_wordpress(
        self,
        *,
        article_id: UUID,
        project_id: UUID,
        user_id: UUID | None,
        publish_status: str = "draft",
        scheduled_at: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Validate, durably claim, and enqueue the WordPress side effect."""
        article, project, validation, key = await self._prepare_publish(
            article_id=article_id,
            project_id=project_id,
            publish_status=publish_status,
            scheduled_at=scheduled_at,
        )
        if dry_run:
            return {
                "status": "validation_only",
                "article_id": str(article_id),
                "project_id": str(project_id),
                **validation.as_dict(),
            }
        await self._reject_invalid_publish(
            article_id=article_id,
            project_id=project_id,
            user_id=user_id,
            publish_status=publish_status,
            key=key,
            target_site_url=project.wordpress_url,
            validation=validation,
        )
        from uuid import uuid4

        task_id = str(uuid4())
        claim = await self.publishing.claim_publish(
            article_id=article_id,
            project_id=project_id,
            user_id=user_id,
            requested_publish_mode=publish_status,
            idempotency_key=key,
            target_site_url=project.wordpress_url,
            task_id=task_id,
            scheduled_at=scheduled_at,
        )
        if not claim.claimed:
            attempt = claim.attempt or {}
            if claim.reason == "article_not_found":
                raise HTTPException(status_code=404, detail="Article not found")
            return {
                "status": "success" if claim.reason == "already_succeeded" else "in_progress",
                "idempotent": True,
                "reason": claim.reason,
                "attempt_id": str(claim.attempt_id),
                "task_id": attempt.get("task_id"),
                "article_id": str(article_id),
                "project_id": str(project_id),
                "publish_status": (
                    self._article_publish_state(str(attempt.get("final_wordpress_status") or publish_status))
                    if claim.reason == "already_succeeded"
                    else (claim.article or {}).get("publish_status", "publish_queued")
                ),
                "wordpress_post_id": attempt.get("wordpress_post_id"),
                "wordpress_url": attempt.get("wordpress_post_url"),
                "idempotency_key": key,
            }
        try:
            from orchestration.celery_app import app

            app.send_task(
                "content_automation.publish_wordpress",
                kwargs={"attempt_id": str(claim.attempt_id)},
                task_id=task_id,
                queue="integrations",
            )
        except Exception as exc:
            await self.publishing.record_failure(
                article_id=article_id,
                attempt_id=claim.attempt_id,
                error_category="queue_unavailable",
                error_message="WordPress publication could not be queued",
                retry_count=0,
                task_id=task_id,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "WordPress publication queue is unavailable",
                    "category": "queue_unavailable",
                    "attempt_id": str(claim.attempt_id),
                },
            ) from exc
        return {
            "status": "queued",
            "idempotent": False,
            "attempt_id": str(claim.attempt_id),
            "task_id": task_id,
            "article_id": str(article_id),
            "project_id": str(project_id),
            "publish_status": "publish_queued",
            "wordpress_post_status": publish_status,
            "idempotency_key": key,
            "warnings": validation.warnings,
        }

    async def execute_publish_attempt(
        self,
        attempt_id: UUID,
        *,
        task_id: str,
    ) -> dict[str, Any]:
        """Execute a previously claimed attempt inside a Celery worker."""
        attempt = await self.publishing.get_attempt(attempt_id)
        if not attempt:
            raise WordPressPublishError(
                "Publishing attempt was not found",
                category="not_found",
                retryable=False,
            )
        if attempt.get("success") or attempt.get("status") == "succeeded":
            return {
                "status": "success",
                "attempt_id": str(attempt_id),
                "idempotent": True,
                "wordpress_post_id": attempt.get("wordpress_post_id"),
                "wordpress_url": attempt.get("wordpress_post_url"),
            }
        article_id = UUID(str(attempt["article_id"]))
        project_id = UUID(str(attempt["project_id"]))
        article_dict = await self.content_service.get_article(article_id, include_content=True)
        project = await self.projects.get_by_id(project_id)
        if not project:
            raise WordPressPublishError(
                "Publishing project was not found",
                category="not_found",
                retryable=False,
            )
        status_value = str(attempt["requested_publish_mode"])
        scheduled_at = article_dict.get("publish_scheduled_at")
        # Revalidate immediately before the remote side effect. An article may be
        # edited, rejected, or have credentials changed while waiting in the queue.
        validation = self._validate(article_dict, project, status_value, scheduled_at)
        if not validation.can_publish:
            raise WordPressPublishError(
                "; ".join(item["message"] for item in validation.errors),
                category="validation_error",
                retryable=False,
                status_code=400,
            )
        acquired = await self.publishing.mark_attempt_running(
            attempt_id=attempt_id,
            task_id=task_id,
        )
        if not acquired:
            current = await self.publishing.get_attempt(attempt_id)
            if current and (current.get("success") or current.get("status") == "succeeded"):
                return {
                    "status": "success",
                    "attempt_id": str(attempt_id),
                    "idempotent": True,
                    "wordpress_post_id": current.get("wordpress_post_id"),
                    "wordpress_url": current.get("wordpress_post_url"),
                }
            if current and current.get("task_id") != task_id:
                return {
                    "status": "superseded",
                    "attempt_id": str(attempt_id),
                    "idempotent": True,
                    "message": "Publishing task ownership changed during recovery",
                }
            return {
                "status": "in_progress",
                "attempt_id": str(attempt_id),
                "idempotent": True,
            }
        article = self.content_service._article_dict_to_generated_article(article_dict)
        existing_post_id = article_dict.get("wordpress_post_id")
        result = await self.distributor.distribute_to_wordpress(
            article,
            project,
            post_status=status_value,
            wordpress_post_id=existing_post_id,
            idempotency_key=str(attempt["idempotency_key"]),
            scheduled_at=scheduled_at,
        )
        wordpress_status = str(result.get("post_status") or status_value)
        remote_verified_at = result.get("remote_verified_at")
        if isinstance(remote_verified_at, str):
            remote_verified_at = datetime.fromisoformat(remote_verified_at.replace("Z", "+00:00"))
        persisted = await self.publishing.record_success(
            article_id=article_id,
            attempt_id=attempt_id,
            wordpress_status=wordpress_status,
            wordpress_post_id=result.get("post_id"),
            wordpress_post_url=result.get("url"),
            retry_count=max(int(result.get("attempts", 1)) - 1, 0),
            task_id=task_id,
            warnings=result.get("warnings") or [],
            remote_verified_at=remote_verified_at,
        )
        if persisted is False:
            return {
                "status": "superseded",
                "attempt_id": str(attempt_id),
                "article_id": str(article_id),
                "project_id": str(project_id),
                "message": "Publishing result was ignored because task ownership changed",
            }
        return {
            "status": "success",
            "attempt_id": str(attempt_id),
            "article_id": str(article_id),
            "project_id": str(project_id),
            "publish_status": self._article_publish_state(wordpress_status),
            "wordpress_post_status": wordpress_status,
            "wordpress_post_id": result.get("post_id"),
            "wordpress_url": result.get("url"),
            "idempotency_key": attempt["idempotency_key"],
            "attempts": result.get("attempts", 1),
            "remote_verified": bool(result.get("remote_verified")),
            "warnings": result.get("warnings") or [],
        }

    async def publish_to_wordpress(
        self,
        *,
        article_id: UUID,
        project_id: UUID,
        user_id: UUID | None,
        publish_status: str = "draft",
        scheduled_at: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Backward-compatible synchronous path used by focused tests and tooling."""
        article_dict, project, validation, key = await self._prepare_publish(
            article_id=article_id,
            project_id=project_id,
            publish_status=publish_status,
            scheduled_at=scheduled_at,
        )
        if dry_run:
            return {
                "status": "validation_only",
                "article_id": str(article_id),
                "project_id": str(project_id),
                **validation.as_dict(),
            }
        await self._reject_invalid_publish(
            article_id=article_id,
            project_id=project_id,
            user_id=user_id,
            publish_status=publish_status,
            key=key,
            target_site_url=project.wordpress_url,
            validation=validation,
        )
        claim = await self.publishing.claim_publish(
            article_id=article_id,
            project_id=project_id,
            user_id=user_id,
            requested_publish_mode=publish_status,
            idempotency_key=key,
            target_site_url=project.wordpress_url,
            scheduled_at=scheduled_at,
        )
        if not claim.claimed:
            if claim.reason == "already_succeeded":
                attempt = claim.attempt or {}
                return {
                    "status": "success",
                    "idempotent": True,
                    "attempt_id": str(claim.attempt_id),
                    "article_id": str(article_id),
                    "project_id": str(project_id),
                    "publish_status": self._article_publish_state(
                        str(attempt.get("final_wordpress_status") or publish_status)
                    ),
                    "wordpress_post_status": attempt.get("final_wordpress_status"),
                    "wordpress_post_id": attempt.get("wordpress_post_id"),
                    "wordpress_url": attempt.get("wordpress_post_url"),
                    "idempotency_key": key,
                }
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "Article publish is already in progress",
                    "attempt_id": str(claim.attempt_id),
                    "publish_status": (claim.article or {}).get("publish_status"),
                },
            )
        article = self.content_service._article_dict_to_generated_article(article_dict)
        existing_post_id = (claim.article or {}).get("wordpress_post_id")
        try:
            result = await self.distributor.distribute_to_wordpress(
                article,
                project,
                post_status=publish_status,
                wordpress_post_id=existing_post_id,
                idempotency_key=key,
                scheduled_at=scheduled_at,
            )
            wordpress_status = str(result.get("post_status") or publish_status)
            await self.publishing.record_success(
                article_id=article_id,
                attempt_id=claim.attempt_id,
                wordpress_status=wordpress_status,
                wordpress_post_id=result.get("post_id"),
                wordpress_post_url=result.get("url"),
                retry_count=max(int(result.get("attempts", 1)) - 1, 0),
                task_id=None,
                warnings=result.get("warnings") or [],
            )
            return {
                "status": "success",
                "attempt_id": str(claim.attempt_id),
                "article_id": str(article_id),
                "project_id": str(project_id),
                "publish_status": self._article_publish_state(wordpress_status),
                "wordpress_post_status": wordpress_status,
                "wordpress_post_id": result.get("post_id"),
                "wordpress_url": result.get("url"),
                "idempotency_key": key,
                "attempts": result.get("attempts", 1),
                "remote_verified": bool(result.get("remote_verified")),
                "warnings": result.get("warnings") or [],
            }
        except WordPressPublishError as exc:
            await self.publishing.record_failure(
                article_id=article_id,
                attempt_id=claim.attempt_id,
                error_category=exc.category,
                error_message=exc.safe_message,
                retry_count=exc.retry_count,
                task_id=None,
            )
            raise HTTPException(
                status_code=self._http_status_for_error(exc.category),
                detail={
                    "message": exc.safe_message,
                    "category": exc.category,
                    "retryable": exc.retryable,
                    "attempt_id": str(claim.attempt_id),
                },
            ) from exc
        except DistributionError as exc:
            await self.publishing.record_failure(
                article_id=article_id,
                attempt_id=claim.attempt_id,
                error_category="unknown_error",
                error_message=str(exc),
                retry_count=0,
                task_id=None,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "message": "WordPress publishing failed",
                    "category": "unknown_error",
                    "attempt_id": str(claim.attempt_id),
                },
            ) from exc

    async def reconcile_stale_attempts(self, *, limit: int = 100) -> dict[str, int]:
        """Requeue expired leases; deterministic slug lookup prevents duplicates."""
        from uuid import uuid4

        from orchestration.celery_app import app

        stale = await self.publishing.list_stale_attempts(limit=limit)
        requeued = 0
        failed = 0
        for attempt in stale:
            task_id = str(uuid4())
            claimed = await self.publishing.requeue_stale_attempt(
                attempt_id=UUID(str(attempt["id"])),
                task_id=task_id,
            )
            if not claimed:
                continue
            try:
                app.send_task(
                    "content_automation.publish_wordpress",
                    kwargs={"attempt_id": str(attempt["id"])},
                    task_id=task_id,
                    queue="integrations",
                )
                requeued += 1
            except Exception:
                await self.publishing.record_failure(
                    article_id=UUID(str(attempt["article_id"])),
                    attempt_id=UUID(str(attempt["id"])),
                    error_category="queue_unavailable",
                    error_message="Stale WordPress publication could not be requeued",
                    retry_count=int(attempt.get("retry_count") or 0),
                    task_id=task_id,
                )
                failed += 1
        return {"stale": len(stale), "requeued": requeued, "failed": failed}

    async def get_publish_status(self, article_id: UUID) -> dict[str, Any]:
        status_row = await self.publishing.get_publish_status(article_id)
        if not status_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Article not found")
        return {
            "article_id": str(status_row["id"]),
            "project_id": str(status_row["project_id"]),
            "publish_status": status_row.get("publish_status") or "not_published",
            "wordpress_post_id": status_row.get("wordpress_post_id"),
            "wordpress_post_url": status_row.get("wordpress_post_url"),
            "wordpress_post_status": status_row.get("wordpress_post_status"),
            "wordpress_published_at": status_row.get("wordpress_published_at"),
            "publish_error_category": status_row.get("publish_error_category"),
            "publish_error_message": status_row.get("publish_error_message"),
            "publish_attempt_count": status_row.get("publish_attempt_count") or 0,
            "publish_updated_at": status_row.get("publish_updated_at"),
            "recent_attempts": [
                {**attempt, "id": str(attempt["id"])}
                for attempt in status_row.get("recent_attempts", [])
            ],
        }

    def _validate(
        self,
        article: dict[str, Any],
        project,
        publish_status: str,
        scheduled_at: datetime | None,
    ) -> PublishValidationResult:
        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        if publish_status not in ALLOWED_PUBLISH_STATUSES:
            errors.append({"code": "invalid_status", "message": "Invalid WordPress publish status"})

        if publish_status in {"future", "publish"} and article.get("review_status") != "approved":
            errors.append({
                "code": "review_required",
                "message": "Manager approval is required before scheduled or public publishing",
            })

        from services.draft_risk_service import DraftRiskService

        risk = DraftRiskService().assess(article)
        if risk.get("risk_level") == "blocked":
            errors.append({
                "code": "draft_risk_blocked",
                "message": "Article is blocked by draft risk checks and cannot be published yet",
            })

        site_url = str(project.wordpress_url or "").strip()
        parsed = urlparse(site_url)
        env = get_settings().environment
        if not site_url:
            errors.append({"code": "missing_wordpress_url", "message": "WordPress URL is not configured"})
        elif parsed.scheme not in {"https", "http"} or not parsed.netloc:
            errors.append({"code": "invalid_wordpress_url", "message": "WordPress URL must be a valid HTTP(S) URL"})
        elif parsed.scheme == "http" and env == "production":
            errors.append({"code": "insecure_wordpress_url", "message": "WordPress URL must use HTTPS in production"})
        elif parsed.scheme == "http":
            warnings.append({"code": "insecure_wordpress_url", "message": "HTTP WordPress URLs are only acceptable outside production"})

        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.username or parsed.password:
            errors.append({
                "code": "embedded_wordpress_credentials",
                "message": "WordPress URL must not contain embedded credentials",
            })
        if parsed.query or parsed.fragment:
            errors.append({
                "code": "invalid_wordpress_base_url",
                "message": "WordPress base URL must not contain a query string or fragment",
            })
        if env == "production" and hostname:
            blocked_name = (
                hostname == "localhost"
                or hostname.endswith(".localhost")
                or hostname.endswith(".local")
                or hostname.endswith(".internal")
            )
            blocked_ip = False
            try:
                address = ipaddress.ip_address(hostname)
                blocked_ip = not address.is_global
            except ValueError:
                pass
            if blocked_name or blocked_ip:
                errors.append({
                    "code": "unsafe_wordpress_target",
                    "message": "Production WordPress URL must target a public host",
                })

        if not project.wordpress_username:
            errors.append({"code": "missing_wordpress_username", "message": "WordPress username is not configured"})
        if not project.wordpress_app_password:
            errors.append({"code": "missing_wordpress_password", "message": "WordPress app password is not configured"})

        title = str(article.get("title") or "").strip()
        content = str(article.get("content") or "").strip()
        if not title:
            errors.append({"code": "missing_title", "message": "Article title is required"})
        elif len(title) > 500:
            errors.append({"code": "title_too_long", "message": "Article title exceeds 500 characters"})

        plain_content = re.sub(r"<[^>]+>", " ", content)
        plain_content = re.sub(r"\s+", " ", plain_content).strip()
        if len(plain_content) < 100:
            errors.append({"code": "content_too_short", "message": "Article content is too short to publish"})
        content_lower = plain_content.lower()
        if any(pattern in content_lower for pattern in PLACEHOLDER_PATTERNS):
            errors.append({"code": "placeholder_content", "message": "Article content contains placeholder-like text"})

        if publish_status == "future":
            if scheduled_at is None:
                errors.append({"code": "missing_schedule", "message": "Scheduled publish requires a future timestamp"})
            else:
                comparable_scheduled_at = scheduled_at
                if comparable_scheduled_at.tzinfo is None:
                    comparable_scheduled_at = comparable_scheduled_at.replace(tzinfo=timezone.utc)
                else:
                    comparable_scheduled_at = comparable_scheduled_at.astimezone(timezone.utc)
                if comparable_scheduled_at <= datetime.now(timezone.utc):
                    errors.append({"code": "invalid_schedule", "message": "Scheduled publish time must be in the future"})
        elif scheduled_at is not None:
            errors.append({"code": "unexpected_schedule", "message": "Schedule time is only valid for future publishing"})

        return PublishValidationResult(
            can_publish=not errors,
            errors=errors,
            warnings=warnings,
            intended_publish_status=publish_status,
            target_site=site_url or None,
        )

    @staticmethod
    def _article_publish_state(wordpress_status: str) -> str:
        return {
            "draft": "published_as_draft",
            "future": "published_scheduled",
            "publish": "published_public",
        }.get(wordpress_status, "published_as_draft")

    @staticmethod
    def _http_status_for_error(category: str) -> int:
        return {
            "auth_error": status.HTTP_401_UNAUTHORIZED,
            "permission_error": status.HTTP_403_FORBIDDEN,
            "validation_error": status.HTTP_400_BAD_REQUEST,
            "not_found": status.HTTP_404_NOT_FOUND,
            "rate_limited": status.HTTP_429_TOO_MANY_REQUESTS,
            "timeout": status.HTTP_504_GATEWAY_TIMEOUT,
            "network_error": status.HTTP_503_SERVICE_UNAVAILABLE,
            "wordpress_4xx": status.HTTP_400_BAD_REQUEST,
            "wordpress_5xx": status.HTTP_502_BAD_GATEWAY,
            "remote_state_mismatch": status.HTTP_502_BAD_GATEWAY,
        }.get(category, status.HTTP_502_BAD_GATEWAY)
