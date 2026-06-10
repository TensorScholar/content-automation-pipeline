#!/usr/bin/env python3
"""Production smoke checks for staging and controlled launches.

The script is intentionally lightweight: it validates configuration and service
connectivity without submitting an LLM generation job or mutating application
data.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _ok(name: str, detail: str = "ok") -> CheckResult:
    return CheckResult(name=name, ok=True, detail=detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=False, detail=detail)


def _secret_is_configured(value) -> bool:
    if value is None:
        return False
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return bool(str(value).strip())


def _http_get(name: str, url: str, timeout: float = 5.0) -> CheckResult:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if 200 <= response.status < 300:
                return _ok(name, f"HTTP {response.status} {url}")
            return _fail(name, f"HTTP {response.status} {url}")
    except urllib.error.HTTPError as exc:
        return _fail(name, f"HTTP {exc.code} {url}")
    except Exception as exc:
        return _fail(name, f"{type(exc).__name__}: {exc}")


def _run_command(name: str, args: list[str]) -> CheckResult:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return _fail(name, f"{type(exc).__name__}: {exc}")

    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode == 0:
        return _ok(name, output or "command succeeded")
    return _fail(name, output or f"exit code {completed.returncode}")


def check_settings() -> CheckResult:
    try:
        from config.settings import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        provider = settings.llm.provider
        provider_configured = {
            "anthropic": _secret_is_configured(settings.llm.anthropic_api_key),
            "openai": _secret_is_configured(settings.llm.openai_api_key),
            "gemini": _secret_is_configured(settings.llm.gemini_api_key),
            "local": bool(settings.llm.local_llm_url and settings.llm.local_llm_url.strip()),
        }.get(provider, False)
        if not provider_configured:
            return _fail(
                "settings",
                f"LLM_PROVIDER={provider} is selected but its credentials are missing",
            )
        return _ok("settings", f"environment={settings.environment}, llm_provider={provider}")
    except Exception as exc:
        return _fail("settings", f"{type(exc).__name__}: {exc}")


async def check_database() -> CheckResult:
    try:
        from infrastructure.database import DatabaseManager

        db = DatabaseManager()
        await db.initialize()
        await db.health_check(skip_vector_check=True)
        await db.close()
        return _ok("database", "health check passed")
    except Exception as exc:
        return _fail("database", f"{type(exc).__name__}: {exc}")


async def check_redis() -> CheckResult:
    try:
        from infrastructure.redis_client import RedisClient

        redis = RedisClient()
        await redis.initialize()
        healthy = await redis.ping()
        if healthy:
            return _ok("redis", "ping passed")
        return _fail("redis", "ping returned false")
    except Exception as exc:
        return _fail("redis", f"{type(exc).__name__}: {exc}")


def check_alembic_current() -> CheckResult:
    current = _run_command("alembic-current", ["alembic", "current"])
    if not current.ok:
        return current

    heads = _run_command("alembic-heads", ["alembic", "heads"])
    if not heads.ok:
        return heads

    current_output = current.detail
    head_revisions = [
        line.split()[0]
        for line in heads.detail.splitlines()
        if line.strip() and not line.startswith("INFO")
    ]
    if head_revisions and not any(revision in current_output for revision in head_revisions):
        return _fail(
            "alembic-current",
            f"database is not at head; current={current_output!r}, heads={heads.detail!r}",
        )
    return _ok("alembic-current", current_output)


def check_celery_broker() -> CheckResult:
    try:
        from orchestration.celery_app import app

        with app.connection_for_read() as connection:
            connection.ensure_connection(max_retries=1)
        return _ok("celery-broker", "broker connection established")
    except Exception as exc:
        return _fail("celery-broker", f"{type(exc).__name__}: {exc}")


def check_celery_beat_import() -> CheckResult:
    try:
        from orchestration.celery_app import app

        schedule = app.conf.beat_schedule or {}
        return _ok("celery-beat", f"{len(schedule)} scheduled task(s) importable")
    except Exception as exc:
        return _fail("celery-beat", f"{type(exc).__name__}: {exc}")


async def run_async_check(
    name: str,
    fn: Callable[[], Awaitable[CheckResult]],
) -> CheckResult:
    try:
        return await asyncio.wait_for(fn(), timeout=30)
    except Exception as exc:
        return _fail(name, f"{type(exc).__name__}: {exc}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run production smoke checks.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:3001")
    parser.add_argument("--nginx-url", default="")
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument("--skip-http", action="store_true")
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        help=(
            "Also call /system/ready. This may perform external provider health "
            "checks, so leave it disabled when using dummy LLM credentials."
        ),
    )
    args = parser.parse_args()

    checks: list[CheckResult] = [
        check_settings(),
        await run_async_check("database", check_database),
        await run_async_check("redis", check_redis),
        check_alembic_current(),
        check_celery_broker(),
        check_celery_beat_import(),
    ]

    if not args.skip_http:
        checks.append(_http_get("api-health", f"{args.api_url.rstrip('/')}/health"))
        if args.check_readiness:
            checks.append(_http_get("api-readiness", f"{args.api_url.rstrip('/')}/system/ready"))
        checks.append(_http_get("api-metrics", f"{args.api_url.rstrip('/')}/metrics"))
        if not args.skip_frontend:
            checks.append(_http_get("frontend", args.frontend_url.rstrip("/")))
        if args.nginx_url:
            checks.append(_http_get("nginx-health", f"{args.nginx_url.rstrip('/')}/health"))

    failed = [check for check in checks if not check.ok]
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    if failed:
        print(f"\n{len(failed)} smoke check(s) failed.", file=sys.stderr)
        return 1

    print("\nAll production smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
