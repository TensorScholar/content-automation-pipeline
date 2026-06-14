"""Safe WordPress publishing orchestration.

This service owns validation, article-level idempotency, durable attempts, and
safe publish status responses. The WordPress API call remains isolated in the
distributor boundary where the decrypted credential is needed.
"""

from __future__ import annotations

import hashlib
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
        self.distributor = distributor or Distributor()

    @staticmethod
    def idempotency_key(
        *,
        article_id: UUID,
        project_id: UUID,
        publish_status: str,
        scheduled_at: datetime | None = None,
    ) -> str:
        raw = f"wordpress:{project_id}:{article_id}:{publish_status}:{scheduled_at.isoformat() if scheduled_at else ''}"
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
        article_dict = await self.content_service.get_article(article_id, include_content=True)
        if str(article_dict.get("project_id")) != str(project_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Article does not belong to the selected project",
            )

        project = await self.projects.get_by_id(project_id)
        if not project:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        validation = self._validate(article_dict, project, publish_status, scheduled_at)
        if dry_run:
            return {
                "status": "validation_only",
                "article_id": str(article_id),
                "project_id": str(project_id),
                **validation.as_dict(),
            }

        key = self.idempotency_key(
            article_id=article_id,
            project_id=project_id,
            publish_status=publish_status,
            scheduled_at=scheduled_at,
        )

        if not validation.can_publish:
            attempt_id = await self.publishing.record_preflight_failure(
                article_id=article_id,
                project_id=project_id,
                user_id=user_id,
                requested_publish_mode=publish_status,
                idempotency_key=key,
                target_site_url=project.wordpress_url,
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

        claim = await self.publishing.claim_publish(
            article_id=article_id,
            project_id=project_id,
            user_id=user_id,
            requested_publish_mode=publish_status,
            idempotency_key=key,
            target_site_url=project.wordpress_url,
        )
        if not claim.claimed:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "Article publish is already in progress",
                    "attempt_id": str(claim.attempt_id),
                    "publish_status": claim.article.get("publish_status") if claim.article else None,
                },
            )

        article = self.content_service._article_dict_to_generated_article(article_dict)
        existing_post_id = claim.article.get("wordpress_post_id") if claim.article else None

        try:
            result = await self.distributor.distribute_to_wordpress(
                article,
                project,
                post_status=publish_status,
                wordpress_post_id=existing_post_id,
                idempotency_key=key,
                scheduled_at=scheduled_at,
            )
            wordpress_status = result.get("post_status") or publish_status
            await self.publishing.record_success(
                article_id=article_id,
                attempt_id=claim.attempt_id,
                wordpress_status=wordpress_status,
                wordpress_post_id=result.get("post_id"),
                wordpress_post_url=result.get("url"),
                retry_count=max(int(result.get("attempts", 1)) - 1, 0),
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
            }
        except WordPressPublishError as exc:
            await self.publishing.record_failure(
                article_id=article_id,
                attempt_id=claim.attempt_id,
                error_category=exc.category,
                error_message=exc.safe_message,
                retry_count=exc.retry_count,
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
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "message": "WordPress publishing failed",
                    "category": "unknown_error",
                    "attempt_id": str(claim.attempt_id),
                },
            ) from exc

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
        }.get(category, status.HTTP_502_BAD_GATEWAY)
