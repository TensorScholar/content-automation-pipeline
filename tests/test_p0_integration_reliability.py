from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from pydantic import SecretStr

from execution.distributer import Distributor, WordPressPublishError
from knowledge.search_console_repository import SearchConsoleSyncClaim
from services.publishing_service import PublishingService
from services.search_console_service import READONLY_SCOPE, SearchConsoleError, SearchConsoleService


class FakePerformanceRepository:
    async def project_exists(self, project_id: UUID) -> bool:
        return True


class FakePerformanceService:
    def __init__(self):
        self.repository = FakePerformanceRepository()
        self.imports: list[dict] = []

    async def import_records(self, **kwargs):
        self.imports.append(kwargs)
        return {"imported": len(kwargs["records"])}


class FakeSearchConsoleRepository:
    def __init__(self):
        self.connection = {
            "id": uuid4(),
            "project_id": uuid4(),
            "encrypted_refresh_token": "legacy-refresh-token",
            "status": "connected",
            "selected_site_url": "sc-domain:example.com",
            "permission_level": "siteOwner",
        }
        self.states: dict[str, dict] = {}
        self.properties: list[dict] = []
        self.runs: dict[UUID, dict] = {}
        self.failures: list[dict] = []

    async def create_oauth_state(self, **kwargs):
        self.states[kwargs["state_hash"]] = kwargs
        return uuid4()

    async def consume_oauth_state(self, *, state_hash: str):
        return self.states.pop(state_hash, None)

    async def get_connection(self, project_id: UUID):
        return self.connection if project_id == self.connection["project_id"] else None

    async def list_properties(self, project_id: UUID):
        return self.properties

    async def list_sync_runs(self, project_id: UUID, limit: int = 20):
        return list(self.runs.values())[:limit]

    async def claim_sync(self, **kwargs):
        run_id = uuid4()
        run = {
            "id": run_id,
            "connection_id": kwargs["connection_id"],
            "project_id": kwargs["project_id"],
            "site_url": kwargs["site_url"],
            "date_from": kwargs["date_from"],
            "date_to": kwargs["date_to"],
            "status": "queued",
            "task_id": kwargs["task_id"],
            "retry_count": 0,
        }
        self.runs[run_id] = run
        return SearchConsoleSyncClaim(True, run)

    async def mark_sync_failure(self, **kwargs):
        self.failures.append(kwargs)
        return True

    async def get_sync_run(self, run_id: UUID):
        return self.runs.get(run_id)

    async def mark_sync_running(self, run_id: UUID, *, task_id: str):
        run = self.runs[run_id]
        if run.get("task_id") != task_id or run["status"] not in {"queued", "retrying"}:
            return False
        run["status"] = "running"
        return True

    async def mark_sync_success(self, **kwargs):
        run = self.runs[kwargs["run_id"]]
        if run.get("task_id") != kwargs.get("task_id"):
            return False
        run.update(
            status="succeeded",
            **{k: v for k, v in kwargs.items() if k not in {"run_id", "task_id"}},
        )
        return True

    async def clear_connection_error(self, project_id: UUID, *, synced: bool = False):
        self.connection["status"] = "connected"

    async def set_connection_error(self, **kwargs):
        self.connection["last_error_category"] = kwargs["category"]
        if kwargs.get("status_value"):
            self.connection["status"] = kwargs["status_value"]


def search_console_settings(repo: FakeSearchConsoleRepository):
    repo.connection["project_id"] = uuid4()
    return SimpleNamespace(
        environment="production",
        credential_encryption_key=SecretStr(Fernet.generate_key().decode()),
        search_console=SimpleNamespace(
            configured=True,
            client_id="client-id",
            client_secret=SecretStr("client-secret"),
            redirect_uri="https://app.example.com/api/search-console/oauth/callback",
            frontend_return_url="https://app.example.com/?search_console=connected",
            oauth_state_ttl_seconds=600,
            request_timeout_seconds=5.0,
            data_lag_days=3,
            default_sync_days=28,
            row_limit=2,
            max_rows_per_sync=4,
        ),
    )


def make_search_service():
    repo = FakeSearchConsoleRepository()
    settings = search_console_settings(repo)
    service = SearchConsoleService(
        repository=repo,
        performance_service=FakePerformanceService(),
        settings=settings,
    )
    return service, repo


@pytest.mark.asyncio
async def test_search_console_authorization_is_read_only_and_uses_one_time_state():
    service, repo = make_search_service()
    project_id = repo.connection["project_id"]
    result = await service.create_authorization_url(project_id=project_id, user_id=uuid4())

    query = parse_qs(urlsplit(result["authorization_url"]).query)
    assert query["scope"] == [READONLY_SCOPE]
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    assert "include_granted_scopes" not in query
    assert len(repo.states) == 1


def test_search_console_production_oauth_urls_must_use_https():
    service, _ = make_search_service()
    service.settings.search_console.redirect_uri = "http://app.example.com/callback"
    with pytest.raises(HTTPException) as exc_info:
        service._require_configured()
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_search_console_rejects_invalid_or_replayed_oauth_state():
    service, _ = make_search_service()
    with pytest.raises(SearchConsoleError) as exc_info:
        await service.handle_oauth_callback(state_value="invalid", code="code", error=None)
    assert exc_info.value.category == "invalid_state"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_search_console_paginates_and_marks_truncation_at_configured_cap():
    service, repo = make_search_service()
    project_id = repo.connection["project_id"]
    run_id = uuid4()
    repo.runs[run_id] = {
        "id": run_id,
        "project_id": project_id,
        "connection_id": repo.connection["id"],
        "site_url": repo.connection["selected_site_url"],
        "date_from": date(2026, 6, 1),
        "date_to": date(2026, 6, 28),
        "status": "queued",
        "task_id": "task-1",
        "retry_count": 0,
    }

    responses = []
    for offset in (0, 2):
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "rows": [
                {
                    "keys": [f"https://example.com/page-{offset + i}"],
                    "clicks": i + 1,
                    "impressions": 10 + i,
                    "ctr": 0.1,
                    "position": 4.2,
                }
                for i in range(2)
            ]
        }
        responses.append(response)

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.post.side_effect = responses

    with patch.object(service, "_refresh_access_token", AsyncMock(return_value="access")):
        with patch("services.search_console_service.httpx.AsyncClient", return_value=client):
            result = await service.execute_sync(run_id, task_id="task-1")

    assert result["status"] == "succeeded"
    assert result["row_count"] == 4
    assert result["pages_fetched"] == 2
    assert result["truncated"] is True
    assert len(service.performance.imports[0]["records"]) == 4
    assert service.performance.imports[0]["source"] == "search_console_api"


@pytest.mark.asyncio
async def test_search_console_duplicate_worker_delivery_is_noop():
    service, repo = make_search_service()
    run_id = uuid4()
    repo.runs[run_id] = {
        "id": run_id,
        "project_id": repo.connection["project_id"],
        "connection_id": repo.connection["id"],
        "site_url": repo.connection["selected_site_url"],
        "date_from": date(2026, 6, 1),
        "date_to": date(2026, 6, 28),
        "status": "running",
        "task_id": "task-1",
        "retry_count": 0,
    }
    with patch.object(service, "_fetch_search_analytics", AsyncMock()) as fetch:
        result = await service.execute_sync(run_id, task_id="task-1")
    assert result["status"] == "running"
    fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_console_invalid_grant_is_nonretryable_and_requires_reconnect():
    service, repo = make_search_service()
    response = MagicMock(status_code=400)
    response.json.return_value = {"error": "invalid_grant"}
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.post.return_value = response

    with patch("services.search_console_service.httpx.AsyncClient", return_value=client):
        with pytest.raises(SearchConsoleError) as exc_info:
            await service._refresh_access_token(repo.connection)
    assert exc_info.value.category == "refresh_token_revoked"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_wordpress_duplicate_lookup_failure_never_falls_through_to_create():
    distributor = Distributor(max_retries=1, initial_retry_delay=0)
    response = MagicMock(status_code=503)
    client = AsyncMock()
    client.get.return_value = response

    with pytest.raises(WordPressPublishError) as exc_info:
        await distributor._find_existing_wordpress_post(
            client=client,
            api_url="https://example.com/wp-json/wp/v2/posts",
            auth=httpx.BasicAuth("user", "pass"),
            slug="smarlux-article",
        )
    assert exc_info.value.category == "wordpress_5xx"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_wordpress_read_after_write_rejects_slug_mismatch():
    distributor = Distributor(max_retries=1, initial_retry_delay=0)
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "id": 12,
        "slug": "unexpected-slug",
        "status": "draft",
        "link": "https://example.com/p/12",
    }
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get.return_value = response

    with patch("execution.distributer.httpx.AsyncClient", return_value=client):
        with pytest.raises(WordPressPublishError) as exc_info:
            await distributor._verify_wordpress_post(
                wordpress_url="https://example.com",
                auth=httpx.BasicAuth("user", "pass"),
                post_id=12,
                expected_slug="expected-slug",
                expected_status="draft",
            )
    assert exc_info.value.category == "remote_state_mismatch"
    assert exc_info.value.retryable is False


def test_wordpress_idempotency_changes_when_article_revision_changes():
    article_id = uuid4()
    project_id = uuid4()
    first = PublishingService.idempotency_key(
        article_id=article_id,
        project_id=project_id,
        publish_status="draft",
        content_fingerprint="revision-a",
    )
    repeated = PublishingService.idempotency_key(
        article_id=article_id,
        project_id=project_id,
        publish_status="draft",
        content_fingerprint="revision-a",
    )
    changed = PublishingService.idempotency_key(
        article_id=article_id,
        project_id=project_id,
        publish_status="draft",
        content_fingerprint="revision-b",
    )
    assert first == repeated
    assert first != changed

@pytest.mark.asyncio
async def test_search_console_invalid_analytics_json_is_classified_retryable():
    service, repo = make_search_service()
    response = MagicMock(status_code=200)
    response.json.side_effect = ValueError("invalid json")
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.post.return_value = response

    with patch.object(service, "_refresh_access_token", AsyncMock(return_value="access")):
        with patch("services.search_console_service.httpx.AsyncClient", return_value=client):
            with pytest.raises(SearchConsoleError) as exc_info:
                await service._fetch_search_analytics(
                    connection=repo.connection,
                    site_url=repo.connection["selected_site_url"],
                    date_from=date(2026, 6, 1),
                    date_to=date(2026, 6, 28),
                )
    assert exc_info.value.category == "invalid_response"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_search_console_status_rejects_unknown_project():
    service, repo = make_search_service()
    service.performance.repository.project_exists = AsyncMock(return_value=False)
    with pytest.raises(HTTPException) as exc_info:
        await service.get_status(uuid4())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_search_console_disconnect_remains_available_with_corrupt_ciphertext():
    service, repo = make_search_service()
    project_id = repo.connection["project_id"]
    repo.connection["encrypted_refresh_token"] = "enc:v1:not-valid"
    repo.disconnect = AsyncMock(return_value=True)
    result = await service.disconnect(project_id)
    assert result == {"project_id": str(project_id), "connected": False}
    repo.disconnect.assert_awaited_once_with(project_id)


@pytest.mark.asyncio
async def test_search_console_rejects_broader_granted_scope():
    service, repo = make_search_service()
    state_value = "valid-state"
    import hashlib
    repo.states[hashlib.sha256(state_value.encode("utf-8")).hexdigest()] = {
        "project_id": repo.connection["project_id"],
        "user_id": uuid4(),
    }
    token_payload = {
        "refresh_token": "refresh",
        "scope": f"{READONLY_SCOPE} https://www.googleapis.com/auth/webmasters",
    }
    with patch.object(service, "_exchange_authorization_code", AsyncMock(return_value=token_payload)):
        with pytest.raises(SearchConsoleError) as exc_info:
            await service.handle_oauth_callback(state_value=state_value, code="code", error=None)
    assert exc_info.value.category == "scope_mismatch"
    assert exc_info.value.retryable is False


def test_wordpress_production_rejects_private_or_embedded_credential_target():
    service = object.__new__(PublishingService)
    article = {
        "title": "A production-ready article title",
        "content": "<h2>Section</h2><p>" + ("reliable content " * 80) + "</p>",
        "meta_description": "A reliable description",
        "keywords": ["reliability"],
        "review_status": "approved",
    }
    private_project = SimpleNamespace(
        wordpress_url="https://127.0.0.1/wordpress",
        wordpress_username="editor",
        wordpress_app_password="credential",
    )
    embedded_project = SimpleNamespace(
        wordpress_url="https://editor:secret@example.com/wordpress",
        wordpress_username="editor",
        wordpress_app_password="credential",
    )
    with patch("services.publishing_service.get_settings", return_value=SimpleNamespace(environment="production")):
        private_result = service._validate(article, private_project, "draft", None)
        embedded_result = service._validate(article, embedded_project, "draft", None)
    assert "unsafe_wordpress_target" in {item["code"] for item in private_result.errors}
    assert "embedded_wordpress_credentials" in {item["code"] for item in embedded_result.errors}

@pytest.mark.asyncio
async def test_wordpress_production_rejects_hostname_resolving_to_private_network():
    distributor = Distributor(max_retries=1, initial_retry_delay=0)
    resolver_loop = SimpleNamespace(
        getaddrinfo=AsyncMock(
            return_value=[
                (2, 1, 6, "", ("10.20.30.40", 443)),
            ]
        )
    )
    with patch(
        "config.settings.get_settings",
        return_value=SimpleNamespace(environment="production"),
    ):
        with patch("execution.distributer.asyncio.get_running_loop", return_value=resolver_loop):
            with pytest.raises(WordPressPublishError) as exc_info:
                await distributor._validate_wordpress_network_target("https://wp.example.com")
    assert exc_info.value.category == "unsafe_target"
    assert exc_info.value.retryable is False

@pytest.mark.asyncio
async def test_search_console_stale_task_id_cannot_take_over_requeued_run():
    service, repo = make_search_service()
    run_id = uuid4()
    repo.runs[run_id] = {
        "id": run_id,
        "project_id": repo.connection["project_id"],
        "connection_id": repo.connection["id"],
        "site_url": repo.connection["selected_site_url"],
        "date_from": date(2026, 6, 1),
        "date_to": date(2026, 6, 28),
        "status": "queued",
        "task_id": "replacement-task",
        "retry_count": 1,
    }
    with patch.object(service, "_fetch_search_analytics", AsyncMock()) as fetch:
        result = await service.execute_sync(run_id, task_id="stale-task")
    assert result["status"] == "queued"
    fetch.assert_not_awaited()
