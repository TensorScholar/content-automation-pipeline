import pytest
from starlette.requests import Request

from config.settings import get_settings
from infrastructure.rate_limiter import RateLimitConfig, RateLimitMiddleware, RedisRateLimiter


class BrokenRedis:
    def pipeline(self, transaction=True):
        raise RuntimeError("redis unavailable")


def set_required_settings(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SECRET_KEY", "x" * 48)
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    monkeypatch.setenv(
        "CREDENTIAL_ENCRYPTION_KEY",
        "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
    )
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("LLM_PRIMARY_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("LLM_SECONDARY_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("LLM_KEYWORD_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("LLM_PLANNING_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("LLM_WRITING_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("LLM_VERIFICATION_MODEL", "gemini-2.5-flash-lite")


def make_request(path: str, headers: dict[str, str] | None = None) -> Request:
    encoded_headers = [
        (key.lower().encode("utf-8"), value.encode("utf-8"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": encoded_headers,
        "client": ("192.0.2.10", 51423),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


def make_middleware() -> RateLimitMiddleware:
    return RateLimitMiddleware(app=lambda scope, receive, send: None, redis_client=None)


def test_auth_ip_limit_defaults_to_broad_ip_guard() -> None:
    config = RateLimitConfig(default_limit=100, auth_limit=20)

    assert config.auth_limit == 20
    assert config.auth_ip_limit == 100


def test_auth_account_identifier_is_normalized_and_hashed() -> None:
    middleware = make_middleware()
    first = make_request("/auth/token", {"X-Login-Identifier": " User@Example.COM "})
    second = make_request("/auth/token", {"X-Login-Identifier": "user@example.com"})

    first_identifier = middleware._get_auth_account_identifier(first)
    second_identifier = middleware._get_auth_account_identifier(second)

    assert first_identifier == second_identifier
    assert first_identifier is not None
    assert first_identifier.startswith("auth_account:192.0.2.10:")
    assert "user@example.com" not in first_identifier


def test_auth_account_identifier_is_missing_without_header() -> None:
    middleware = make_middleware()

    assert middleware._get_auth_account_identifier(make_request("/auth/token")) is None


@pytest.mark.asyncio
async def test_auth_identifier_uses_account_header_before_ip() -> None:
    middleware = make_middleware()
    request = make_request("/auth/token", {"X-Login-Identifier": "viewer@example.com"})

    identifier = await middleware._get_identifier(request)

    assert identifier.startswith("auth_account:192.0.2.10:")
    assert identifier != "ip:192.0.2.10"


@pytest.mark.asyncio
async def test_non_auth_identifier_falls_back_to_ip() -> None:
    middleware = make_middleware()

    identifier = await middleware._get_identifier(make_request("/projects"))

    assert identifier == "ip:192.0.2.10"


@pytest.mark.asyncio
async def test_rate_limiter_fails_closed_in_production(monkeypatch) -> None:
    set_required_settings(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "false")
    get_settings.cache_clear()

    limiter = RedisRateLimiter(BrokenRedis(), RateLimitConfig())

    allowed, info = await limiter.is_allowed("user:1", limit=10, window=60)

    assert allowed is False
    assert info["retry_after"] == 60
