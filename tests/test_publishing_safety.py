from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from core.exceptions import DistributionError
from core.models import GeneratedArticle, Project, QualityMetrics
from execution.distributer import Distributor, WordPressPublishError
from knowledge.publishing_repository import PublishClaim
from services.publishing_service import PublishingService


def _valid_content() -> str:
    paragraph = (
        "Reliable publishing needs explicit state, conservative defaults, "
        "auditable retries, and clear operational ownership across teams. "
    )
    return (
        "<h2>Publishing safety model</h2>"
        f"<p>{paragraph * 5}</p>"
        "<h2>Operational controls</h2>"
        f"<p>{paragraph * 5}</p>"
    )


def _article_dict(project_id: UUID | None = None) -> dict:
    project_id = project_id or uuid4()
    article_id = uuid4()
    return {
        "id": article_id,
        "project_id": project_id,
        "content_plan_id": uuid4(),
        "title": "Reliable WordPress publishing for production teams",
        "content": _valid_content(),
        "meta_description": "A practical publishing safety model for production WordPress workflows.",
        "keywords": ["publishing", "wordpress"],
        "word_count": 120,
        "readability_score": 70.0,
        "keyword_density": {"publishing": 0.02},
        "total_tokens_used": 100,
        "total_cost": 0.01,
        "generation_time": 1.0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def _project(project_id: UUID) -> Project:
    return Project(
        id=project_id,
        name="Publishing Project",
        wordpress_url="https://example.com",
        wordpress_username="publisher",
        wordpress_app_password=SecretStr("wp app password"),
    )


class FakeContentService:
    def __init__(self, article: dict):
        self.article = article

    async def get_article(self, article_id: UUID, include_content: bool = True):
        del include_content
        if article_id != self.article["id"]:
            raise HTTPException(status_code=404, detail="Article not found")
        return self.article

    def _article_dict_to_generated_article(self, article: dict) -> GeneratedArticle:
        return GeneratedArticle(
            id=article["id"],
            project_id=article["project_id"],
            content_plan_id=article["content_plan_id"],
            title=article["title"],
            content=article["content"],
            meta_description=article["meta_description"],
            keywords=article["keywords"],
            quality_metrics=QualityMetrics(
                word_count=article["word_count"],
                readability_score=article["readability_score"],
                lexical_diversity=0.7,
                keyword_density=article["keyword_density"],
                avg_sentence_length=14.0,
                paragraph_count=4,
            ),
            total_tokens_used=article["total_tokens_used"],
            total_cost_usd=article["total_cost"],
            generation_time_seconds=article["generation_time"],
            created_at=article["created_at"],
            updated_at=article["updated_at"],
        )


class FakeProjectRepository:
    def __init__(self, project: Project):
        self.project = project

    async def get_by_id(self, project_id: UUID):
        return self.project if self.project.id == project_id else None


class FakePublishingRepository:
    def __init__(self):
        self.claimed = True
        self.claim_article: dict = {
            "publish_status": "not_published",
            "wordpress_post_id": None,
        }
        self.claims = []
        self.preflight_failures = []
        self.successes = []
        self.failures = []

    async def claim_publish(self, **kwargs):
        self.claims.append(kwargs)
        return PublishClaim(
            claimed=self.claimed,
            article=self.claim_article,
            attempt_id=uuid4(),
            reason=None if self.claimed else "already_publishing",
        )

    async def record_preflight_failure(self, **kwargs):
        self.preflight_failures.append(kwargs)
        return uuid4()

    async def record_success(self, **kwargs):
        self.successes.append(kwargs)

    async def record_failure(self, **kwargs):
        self.failures.append(kwargs)

    async def get_publish_status(self, article_id: UUID):
        return {
            "id": article_id,
            "project_id": uuid4(),
            "publish_status": "published_as_draft",
            "wordpress_post_id": "123",
            "wordpress_post_url": "https://example.com/p/123",
            "wordpress_post_status": "draft",
            "wordpress_published_at": datetime.now(timezone.utc),
            "publish_error_category": None,
            "publish_error_message": None,
            "publish_attempt_count": 1,
            "publish_updated_at": datetime.now(timezone.utc),
            "recent_attempts": [],
        }


class FakeDistributor:
    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    async def distribute_to_wordpress(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.error:
            raise self.error
        return {
            "status": "published",
            "post_status": kwargs.get("post_status", "draft"),
            "post_id": kwargs.get("wordpress_post_id") or 123,
            "url": "https://example.com/p/123",
            "attempts": 1,
        }


def _service(
    *,
    article: dict | None = None,
    project: Project | None = None,
    publishing_repo: FakePublishingRepository | None = None,
    distributor: FakeDistributor | None = None,
) -> tuple[PublishingService, dict, Project, FakePublishingRepository, FakeDistributor]:
    article = article or _article_dict()
    project = project or _project(article["project_id"])
    publishing_repo = publishing_repo or FakePublishingRepository()
    distributor = distributor or FakeDistributor()
    service = PublishingService(
        content_service=FakeContentService(article),
        project_repository=FakeProjectRepository(project),
        publishing_repository=publishing_repo,
        distributor=distributor,
    )
    return service, article, project, publishing_repo, distributor


@pytest.mark.asyncio
async def test_dry_run_does_not_call_wordpress_or_claim_publish():
    service, article, project, publishing_repo, distributor = _service()

    result = await service.publish_to_wordpress(
        article_id=article["id"],
        project_id=project.id,
        user_id=uuid4(),
        dry_run=True,
    )

    assert result["status"] == "validation_only"
    assert result["can_publish"] is True
    assert distributor.calls == []
    assert publishing_repo.claims == []


@pytest.mark.asyncio
async def test_default_publish_mode_is_draft_and_success_is_audited():
    service, article, project, publishing_repo, distributor = _service()

    result = await service.publish_to_wordpress(
        article_id=article["id"],
        project_id=project.id,
        user_id=uuid4(),
    )

    assert result["wordpress_post_status"] == "draft"
    assert result["publish_status"] == "published_as_draft"
    assert distributor.calls[0]["kwargs"]["post_status"] == "draft"
    assert publishing_repo.successes[0]["wordpress_post_id"] == 123


@pytest.mark.asyncio
async def test_invalid_publish_status_is_rejected_before_wordpress_call():
    service, article, project, publishing_repo, distributor = _service()

    with pytest.raises(HTTPException) as exc:
        await service.publish_to_wordpress(
            article_id=article["id"],
            project_id=project.id,
            user_id=uuid4(),
            publish_status="invalid",
        )

    assert exc.value.status_code == 400
    assert distributor.calls == []
    assert publishing_repo.preflight_failures
    assert publishing_repo.preflight_failures[0]["error_category"] == "validation_error"


@pytest.mark.asyncio
async def test_scheduled_publish_requires_future_timestamp():
    service, article, project, publishing_repo, distributor = _service()

    with pytest.raises(HTTPException):
        await service.publish_to_wordpress(
            article_id=article["id"],
            project_id=project.id,
            user_id=uuid4(),
            publish_status="future",
            scheduled_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

    assert distributor.calls == []
    assert publishing_repo.preflight_failures


@pytest.mark.asyncio
async def test_existing_wordpress_post_id_uses_update_path_not_create_path():
    publishing_repo = FakePublishingRepository()
    publishing_repo.claim_article["wordpress_post_id"] = "456"
    service, article, project, _, distributor = _service(publishing_repo=publishing_repo)

    result = await service.publish_to_wordpress(
        article_id=article["id"],
        project_id=project.id,
        user_id=uuid4(),
    )

    assert result["wordpress_post_id"] == "456"
    assert distributor.calls[0]["kwargs"]["wordpress_post_id"] == "456"


@pytest.mark.asyncio
async def test_duplicate_concurrent_publish_is_rejected_without_wordpress_call():
    publishing_repo = FakePublishingRepository()
    publishing_repo.claimed = False
    publishing_repo.claim_article["publish_status"] = "publishing"
    service, article, project, _, distributor = _service(publishing_repo=publishing_repo)

    with pytest.raises(HTTPException) as exc:
        await service.publish_to_wordpress(
            article_id=article["id"],
            project_id=project.id,
            user_id=uuid4(),
        )

    assert exc.value.status_code == 409
    assert distributor.calls == []


@pytest.mark.asyncio
async def test_publish_failure_is_audited_with_redacted_message():
    error = WordPressPublishError(
        "Authorization: Bearer very-secret-token",
        category="auth_error",
        retryable=False,
        retry_count=0,
    )
    service, article, project, publishing_repo, _ = _service(distributor=FakeDistributor(error))

    with pytest.raises(HTTPException) as exc:
        await service.publish_to_wordpress(
            article_id=article["id"],
            project_id=project.id,
            user_id=uuid4(),
        )

    assert exc.value.status_code == 401
    assert publishing_repo.failures[0]["error_category"] == "auth_error"
    assert "very-secret-token" not in publishing_repo.failures[0]["error_message"]


def test_wordpress_error_classification_is_deterministic():
    assert Distributor._classify_http_status(401) == ("auth_error", False)
    assert Distributor._classify_http_status(403) == ("permission_error", False)
    assert Distributor._classify_http_status(400) == ("validation_error", False)
    assert Distributor._classify_http_status(429) == ("rate_limited", True)
    assert Distributor._classify_http_status(503) == ("wordpress_5xx", True)


@pytest.mark.asyncio
async def test_distributor_does_not_retry_auth_errors(sample_project=None):
    article = _service()[1]
    generated = FakeContentService(article)._article_dict_to_generated_article(article)
    generated.keywords = []
    project = _project(article["project_id"])
    distributor = Distributor(max_retries=5, initial_retry_delay=0.01)

    from unittest.mock import AsyncMock, MagicMock, patch

    import httpx

    response = MagicMock()
    response.status_code = 401
    response.text = "password=secret"
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401",
        request=MagicMock(),
        response=response,
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.post.return_value = response

    with patch.object(distributor, "validate_wordpress_connection", AsyncMock(return_value=(True, ""))):
        with patch("execution.distributer.httpx.AsyncClient", return_value=client):
            with pytest.raises(WordPressPublishError) as exc:
                await distributor.distribute_to_wordpress(generated, project)

    assert exc.value.category == "auth_error"
    assert client.post.call_count == 1
    assert "secret" not in exc.value.safe_message


@pytest.mark.asyncio
async def test_distributor_updates_existing_slug_instead_of_creating_duplicate():
    article = _service()[1]
    generated = FakeContentService(article)._article_dict_to_generated_article(article)
    project = _project(article["project_id"])
    distributor = Distributor(max_retries=1, initial_retry_delay=0.01)

    from unittest.mock import AsyncMock, MagicMock, patch

    lookup_response = MagicMock()
    lookup_response.status_code = 200
    lookup_response.json.return_value = [{"id": 777, "link": "https://example.com/p/777"}]

    update_response = MagicMock()
    update_response.status_code = 200
    update_response.json.return_value = {"id": 777, "link": "https://example.com/p/777"}
    update_response.raise_for_status = MagicMock()

    schema_response = MagicMock()
    schema_response.status_code = 200

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get.return_value = lookup_response
    client.post.side_effect = [update_response, schema_response]

    with patch.object(distributor, "validate_wordpress_connection", AsyncMock(return_value=(True, ""))):
        with patch("execution.distributer.httpx.AsyncClient", return_value=client):
            result = await distributor.distribute_to_wordpress(
                generated,
                project,
                idempotency_key="wp:test",
            )

    assert result["post_id"] == 777
    assert "/wp-json/wp/v2/posts/777" in client.post.call_args_list[0][0][0]
