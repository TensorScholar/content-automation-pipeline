"""Regression tests for lazy settings loading.

Proves that:
- importing ``security`` (and pure unit modules) does not require
  DATABASE_URL or other runtime secrets;
- the settings module does not construct Settings at import time;
- explicit ``Settings()`` construction still rejects missing required
  configuration (production-grade fail-fast is preserved).

The file is self-bootstrapping (repo root inserted into ``sys.path``) so it
runs both as a standalone file and as part of the full suite. The focused
checks run in isolated subprocesses with a scrubbed environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Keep cached Settings isolated between in-process assertions.
from config.settings import get_settings  # noqa: E402  (after sys.path bootstrap)


def _scrubbed_env() -> dict[str, str]:
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
    ):
        env.pop(secret, None)
    return env


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(REPO_ROOT),
        env=_scrubbed_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_security_without_database_url():
    """security must import with no runtime secrets configured."""
    proc = _run(
        """
        import security
        from security import User, resolve_user_role, normalize_managed_user_role
        print("IMPORT_OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IMPORT_OK" in proc.stdout


def test_settings_module_import_does_not_construct_settings():
    """Importing config.settings alone must not build/validate Settings."""
    proc = _run(
        """
        import config.settings as s
        assert s.get_settings.cache_info().currsize == 0
        print("LAZY_OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "LAZY_OK" in proc.stdout


def test_explicit_settings_still_rejects_missing_required_configuration():
    """Production fail-fast is preserved: Settings() requires real config."""
    get_settings.cache_clear()
    proc = _run(
        """
        from pydantic import ValidationError
        from config.settings import Settings
        try:
            Settings()
        except ValidationError as exc:
            text = str(exc)
            assert "DATABASE_URL" in text, text
            print("REJECTS_OK")
        else:
            raise AssertionError("Settings() accepted missing DATABASE_URL")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "REJECTS_OK" in proc.stdout


def test_pure_unit_modules_import_and_pytest_collects_cleanly():
    """Commonly imported unit modules must collect without runtime env."""
    proc = _run(
        """
        import security
        from datetime import datetime
        from security import User, resolve_user_role
        user = User(
            id="00000000-0000-0000-0000-000000000001",
            username="u",
            email="u@example.com",
            created_at=datetime(2026, 1, 1),
        )
        assert resolve_user_role(user) == "user"
        print("UNIT_OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "UNIT_OK" in proc.stdout