"""Regression tests for Task 5A API startup configuration.

Proves that:
- importing ``api.main`` does not require any runtime secrets;
- building the app is deferred to ``create_app()`` so settings are resolved
  (and fail fast) at startup rather than at import time;
- production builds disable docs/OpenAPI while development builds expose them;
- startup continues to reject missing required configuration instead of
  silently continuing.

The module is self-bootstrapping (repo root inserted into ``sys.path``) and
runs the focused checks in isolated subprocesses with a scrubbed environment.
"""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _scrubbed_env(**overrides) -> dict[str, str]:
    """Minimal environment: none of the runtime secrets present."""
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
    }
    for key in ("VIRTUAL_ENV", "POETRY_ACTIVE", "LANG", "LC_ALL", "LC_CTYPE"):
        if key in os.environ:
            env[key] = os.environ[key]
    for secret in (
        "DATABASE_URL",
        "REDIS_URL",
        "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND",
        "SECRET_KEY",
        "CREDENTIAL_ENCRYPTION_KEY",
        "ENVIRONMENT",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LLM_PROVIDER",
        "LLM_PRIMARY_MODEL",
    ):
        env.pop(secret, None)
    env.update(overrides)
    return env


def _run(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        env=env if env is not None else _scrubbed_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_api_main_without_runtime_secrets():
    """Bare ``import api.main`` must not resolve settings (no secrets required)."""
    proc = _run(
        """
        import api.main
        print("IMPORT_OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IMPORT_OK" in proc.stdout


def test_create_app_is_deferred_factory_not_module_level_app():
    """Module scope must not build the app (no eager settings resolution)."""
    proc = _run(
        """
        import api.main
        assert not hasattr(api.main, "app"), "module-level app must not exist"
        assert callable(api.main.create_app)
        print("FACTORY_OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FACTORY_OK" in proc.stdout


def test_create_app_rejects_missing_database_url():
    """Startup must fail fast (not suppress) when DATABASE_URL is missing."""
    proc = _run(
        """
        from api.main import create_app
        try:
            create_app()
        except Exception as exc:
            text = str(exc)
            assert "DATABASE_URL" in text, text
            print("REJECTS_OK")
        else:
            raise AssertionError("create_app() accepted missing DATABASE_URL")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "REJECTS_OK" in proc.stdout


def test_create_app_development_builds_with_docs():
    """Valid development config builds the app and exposes /docs."""
    env = _scrubbed_env(
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        REDIS_URL="redis://localhost:6379/0",
        CELERY_BROKER_URL="redis://localhost:6379/1",
        CELERY_RESULT_BACKEND="redis://localhost:6379/1",
        SECRET_KEY=hashlib.sha256(b"startup-test-secret").hexdigest(),
        ENVIRONMENT="development",
    )
    proc = _run(
        """
        from api.main import create_app
        app = create_app()
        assert app.docs_url == "/docs", app.docs_url
        assert app.openapi_url == "/openapi.json", app.openapi_url
        paths = {route.path for route in app.routes}
        assert "/docs" in paths
        print("DEV_OK")
        """,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DEV_OK" in proc.stdout


def test_create_app_production_disables_docs():
    """Production config builds the app with docs/OpenAPI disabled."""
    env = _scrubbed_env(
        ENVIRONMENT="production",
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
        REDIS_URL="redis://localhost:6379/0",
        CELERY_BROKER_URL="redis://localhost:6379/1",
        CELERY_RESULT_BACKEND="redis://localhost:6379/1",
        SECRET_KEY=hashlib.sha256(b"startup-test-secret").hexdigest(),
        CREDENTIAL_ENCRYPTION_KEY=base64.urlsafe_b64encode(
            hashlib.sha256(b"startup-test-credential-key").digest()
        ).decode("ascii"),
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
        LLM_PRIMARY_MODEL="anthropic/claude-sonnet-4-5-20250929",
        SENTRY_DSN="",
    )
    proc = _run(
        """
        from api.main import create_app
        app = create_app()
        assert app.docs_url is None, app.docs_url
        assert app.openapi_url is None, app.openapi_url
        paths = {route.path for route in app.routes}
        assert "/docs" not in paths
        print("PROD_OK")
        """,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PROD_OK" in proc.stdout
