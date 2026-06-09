import pytest
from starlette.requests import Request

from infrastructure.rate_limiter import RateLimitConfig, RateLimitMiddleware


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
