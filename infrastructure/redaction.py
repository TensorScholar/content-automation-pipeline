"""Central secret redaction for logs and error-reporting payloads."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "id_token",
    "jwt",
    "password",
    "passwd",
    "secret",
    "secret_key",
    "client_secret",
    "wordpress_app_password",
    "database_url",
    "redis_url",
    "celery_broker_url",
    "celery_result_backend",
    "smtp_password",
    "credential_encryption_key",
    "sentry_dsn",
}

_KEY_SUFFIXES = ("_api_key", "_password", "_secret")

_PATTERNS = (
    re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"(?i)\b(authorization|api[_-]?key|password|secret)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/@\s]+:)([^@\s]+)(@)"),
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).strip().lower()
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_KEY_SUFFIXES)


def redact_text(value: str) -> str:
    """Redact common credential formats without retaining the matched value."""
    result = value
    result = _PATTERNS[0].sub(lambda match: f"{match.group(1)} {REDACTED}", result)
    result = _PATTERNS[1].sub(REDACTED, result)
    result = _PATTERNS[2].sub(REDACTED, result)
    result = _PATTERNS[3].sub(lambda match: f"{match.group(1)}={REDACTED}", result)
    result = _PATTERNS[4].sub(
        lambda match: f"{match.group(1)}{REDACTED}{match.group(3)}",
        result,
    )
    return result


def redact_secrets(value: Any, *, key: object | None = None) -> Any:
    """Recursively redact values by key and by common inline secret patterns."""
    if key is not None and _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, dict):
        return {
            item_key: redact_secrets(item_value, key=item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def structlog_redaction_processor(logger, method_name: str, event_dict: dict) -> dict:
    del logger, method_name
    return redact_secrets(event_dict)


def configure_loguru_redaction() -> None:
    """Install process-wide Loguru message/extra redaction."""
    from loguru import logger

    def patch_record(record: dict) -> None:
        record["message"] = redact_text(record["message"])
        record["extra"] = redact_secrets(record["extra"])

    logger.configure(patcher=patch_record)
