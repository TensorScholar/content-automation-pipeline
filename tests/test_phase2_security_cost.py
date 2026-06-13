"""Focused regression tests for Phase 2 security and cost controls."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr, ValidationError
from sqlalchemy.sql.dml import Insert

from api.schemas import ProjectResponse
from core.exceptions import DatabaseError, LLMError, TokenBudgetExceededError
from core.models import Project
from infrastructure.credential_encryption import (
    decrypt_credential,
    encrypt_credential,
    is_encrypted_credential,
)
from infrastructure.error_tracking import _before_send
from infrastructure.llm_client import (
    LLMResponse,
    ModelProvider,
    TokenUsage,
    UnifiedLLMClient,
)
from infrastructure.llm_usage import LLMUsageContext, LLMUsageService
from infrastructure.redaction import REDACTED, redact_secrets
from knowledge.project_repository import ProjectRepository


class _ScalarResult:
    def __init__(self, value=0):
        self.value = value

    def scalar_one(self):
        return self.value


class _FakeSession:
    def __init__(self, committed_cost=0):
        self.committed_cost = committed_cost
        self.inserted = []
        self.executions = []

    async def execute(self, statement, params=None):
        self.executions.append((statement, params))
        if isinstance(statement, Insert):
            self.inserted.append(statement.compile().params)
            return _ScalarResult()
        if "COALESCE(SUM" in str(statement):
            return _ScalarResult(self.committed_cost)
        return _ScalarResult()


class _FakeDatabase:
    def __init__(self, committed_cost=0):
        self.session_value = _FakeSession(committed_cost)

    @asynccontextmanager
    async def session(self):
        yield self.session_value


class _CaptureInsertSession:
    def __init__(self):
        self.inserted = None

    async def execute(self, statement, params=None):
        del params
        self.inserted = statement.compile().params
        raise RuntimeError("stop after capturing insert")


class _CaptureInsertDatabase:
    def __init__(self):
        self.session_value = _CaptureInsertSession()

    @asynccontextmanager
    async def session(self):
        yield self.session_value


class _FakeUsageService:
    def __init__(self, reserve_error=None):
        self.reserve_error = reserve_error
        self.reservations = []
        self.successes = []
        self.failures = []

    async def reserve(self, **kwargs):
        if self.reserve_error:
            raise self.reserve_error
        record_id = uuid4()
        self.reservations.append((record_id, kwargs))
        return record_id

    async def record_success(self, record_id, **kwargs):
        self.successes.append((record_id, kwargs))

    async def record_failure(self, record_id, error_category):
        self.failures.append((record_id, error_category))


def _settings_with_limits(daily=10, monthly=100):
    return SimpleNamespace(
        llm=SimpleNamespace(
            daily_cost_limit_usd=daily,
            monthly_cost_limit_usd=monthly,
            project_daily_cost_limit_usd=0,
            project_monthly_cost_limit_usd=0,
            user_daily_cost_limit_usd=0,
            user_monthly_cost_limit_usd=0,
        )
    )


def test_wordpress_password_is_encrypted_and_not_double_encrypted(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(credential_encryption_key=key),
    )

    stored = ProjectRepository._encrypted_wordpress_password("wp-secret-value")
    stored_again = ProjectRepository._encrypted_wordpress_password(stored)

    assert stored != "wp-secret-value"
    assert is_encrypted_credential(stored)
    assert stored_again == stored
    assert decrypt_credential(stored, key) == "wp-secret-value"
    assert decrypt_credential("legacy-plaintext", key) == "legacy-plaintext"


def test_project_response_does_not_expose_wordpress_password():
    assert "wordpress_app_password" not in ProjectResponse.model_fields


@pytest.mark.asyncio
async def test_project_repository_insert_never_persists_plaintext(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(credential_encryption_key=key),
    )
    database = _CaptureInsertDatabase()
    repository = ProjectRepository(database)

    with pytest.raises(DatabaseError):
        await repository.create(
            Project(
                name="Encrypted Project",
                wordpress_url="https://example.com",
                wordpress_username="publisher",
                wordpress_app_password="wp-plaintext-secret",
            )
        )

    stored = database.session_value.inserted["wordpress_app_password"]
    assert stored != "wp-plaintext-secret"
    assert decrypt_credential(stored, key) == "wp-plaintext-secret"


def test_missing_credential_key_fails_in_production(monkeypatch):
    from config.settings import Settings

    production_env = {
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost/db",
        "REDIS_URL": "redis://localhost:6379/0",
        "CELERY_BROKER_URL": "redis://localhost:6379/1",
        "CELERY_RESULT_BACKEND": "redis://localhost:6379/2",
        "SECRET_KEY": "s" * 40,
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-placeholder",
        "LLM_PRIMARY_MODEL": "gemini-2.5-flash-lite",
        "LLM_SECONDARY_MODEL": "gemini-2.5-flash-lite",
        "LLM_KEYWORD_MODEL": "gemini-2.5-flash-lite",
        "LLM_PLANNING_MODEL": "gemini-2.5-flash-lite",
        "LLM_WRITING_MODEL": "gemini-2.5-flash-lite",
        "LLM_VERIFICATION_MODEL": "gemini-2.5-flash-lite",
        "ALLOWED_HOSTS": "localhost",
        "CORS_ORIGINS": "https://localhost",
    }
    for name, value in production_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)

    with pytest.raises(ValidationError, match="CREDENTIAL_ENCRYPTION_KEY"):
        Settings(_env_file=None)


@pytest.mark.asyncio
async def test_usage_reservation_is_persisted_without_prompt_content():
    database = _FakeDatabase()
    service = LLMUsageService(database)
    service.settings = _settings_with_limits()
    project_id = uuid4()

    record_id = await service.reserve(
        provider="gemini",
        model="gemini-2.5-flash-lite",
        estimated_cost=0.25,
        context=LLMUsageContext(project_id=project_id, operation_type="content_generation"),
    )

    assert record_id
    inserted = database.session_value.inserted[0]
    assert inserted["provider"] == "gemini"
    assert inserted["project_id"] == project_id
    assert inserted["status"] == "reserved"
    assert not any("prompt" in key and key != "prompt_tokens" for key in inserted)
    budget_queries = [
        str(statement)
        for statement, _ in database.session_value.executions
        if "COALESCE(SUM" in str(statement)
    ]
    assert budget_queries
    assert "status = 'failure'" in budget_queries[0]


@pytest.mark.asyncio
async def test_multiple_usage_records_crossing_budget_are_blocked():
    database = _FakeDatabase(committed_cost=9.75)
    service = LLMUsageService(database)
    service.settings = _settings_with_limits(daily=10, monthly=100)

    with pytest.raises(TokenBudgetExceededError, match="daily budget"):
        await service.reserve(
            provider="openai",
            model="gpt-4o-mini",
            estimated_cost=0.50,
        )
    assert database.session_value.inserted == []


@pytest.mark.asyncio
async def test_llm_client_records_actual_usage():
    usage_service = _FakeUsageService()
    client = UnifiedLLMClient(usage_service=usage_service)

    async def fake_call(*args, **kwargs):
        return "ok", TokenUsage(10, 5, 15), "stop"

    client._call_llm = fake_call
    response = await client.generate(
        model="gemini-2.5-flash-lite",
        prompt="test prompt",
        max_tokens=10,
        _allow_fallback=False,
    )

    assert response.content == "ok"
    assert len(usage_service.reservations) == 1
    assert usage_service.successes[0][1]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_budget_failure_blocks_provider_and_fallback():
    budget_error = TokenBudgetExceededError(
        "LLM global daily budget is exhausted",
        current_cost=10,
        budget_limit=10,
    )
    usage_service = _FakeUsageService(reserve_error=budget_error)
    client = UnifiedLLMClient(usage_service=usage_service)
    provider_called = False

    async def fake_call(*args, **kwargs):
        nonlocal provider_called
        provider_called = True
        raise LLMError("provider should not be called")

    client._call_llm = fake_call
    with pytest.raises(TokenBudgetExceededError):
        await client.generate(
            model="gemini-2.5-flash-lite",
            prompt="test prompt",
            max_tokens=10,
        )
    assert provider_called is False


def test_redaction_covers_headers_keys_urls_and_sentry_events():
    payload = {
        "authorization": "Bearer visible-token",
        "OPENAI_API_KEY": "sk-proj-visiblevalue12345",
        "database_url": "postgresql://user:password@db/app",
        "message": "redis://default:redis-secret@redis:6379/0",
        "total_tokens": 42,
    }

    redacted = redact_secrets(payload)
    sentry_event = _before_send({"request": {"headers": payload}}, {})

    assert redacted["authorization"] == REDACTED
    assert redacted["OPENAI_API_KEY"] == REDACTED
    assert "password" not in redacted["database_url"]
    assert "redis-secret" not in redacted["message"]
    assert redacted["total_tokens"] == 42
    assert sentry_event["request"]["headers"]["authorization"] == REDACTED


def test_sentry_without_dsn_is_startup_safe(monkeypatch):
    from infrastructure.error_tracking import initialize_sentry

    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(
            sentry=SimpleNamespace(dsn=None),
            is_production=True,
        ),
    )
    assert initialize_sentry("api") is False


def test_sentry_initialization_installs_redaction(monkeypatch):
    import sentry_sdk

    import infrastructure.error_tracking as error_tracking

    captured = {}
    monkeypatch.setattr(
        "config.settings.get_settings",
        lambda: SimpleNamespace(
            sentry=SimpleNamespace(
                dsn=SecretStr("https://public@example.invalid/1"),
                environment="test",
                traces_sample_rate=0.0,
            ),
            environment="development",
            is_production=False,
        ),
    )
    monkeypatch.setattr(sentry_sdk, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(error_tracking, "_initialized_dsn", None)

    assert error_tracking.initialize_sentry("api") is True
    assert captured["send_default_pii"] is False
    assert captured["before_send"] is error_tracking._before_send
