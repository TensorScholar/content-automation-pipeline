"""Optional Sentry initialization for API and Celery processes."""

from __future__ import annotations

import logging
from typing import Any, Optional

from infrastructure.redaction import redact_secrets

logger = logging.getLogger(__name__)
_initialized_dsn: Optional[str] = None


def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    del hint
    return redact_secrets(event)


def initialize_sentry(process_type: str) -> bool:
    """Initialize Sentry when configured; never require it for startup."""
    global _initialized_dsn

    from config.settings import get_settings

    settings = get_settings()
    if not settings.sentry.dsn:
        if settings.is_production:
            logger.warning("Sentry is not configured for production process %s", process_type)
        return False

    dsn = settings.sentry.dsn.get_secret_value()
    if _initialized_dsn == dsn:
        return True

    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.fastapi import FastApiIntegration

    integrations = (
        [CeleryIntegration()]
        if process_type in {"celery", "worker", "beat"}
        else [FastApiIntegration()]
    )
    sentry_sdk.init(
        dsn=dsn,
        environment=settings.sentry.environment or settings.environment,
        traces_sample_rate=settings.sentry.traces_sample_rate,
        integrations=integrations,
        send_default_pii=False,
        before_send=_before_send,
    )
    _initialized_dsn = dsn
    logger.info("Sentry initialized for process %s", process_type)
    return True
