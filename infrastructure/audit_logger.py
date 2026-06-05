"""
Audit Logging System for Security and Compliance

Tracks critical operations for:
- Security audits
- Compliance requirements
- Incident investigation
- User activity monitoring
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, JSON, String, Boolean, Text, Table, insert
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from infrastructure.database import DatabaseManager
from infrastructure.redis_client import RedisClient


class AuditEventType(str, Enum):
    """Types of auditable events."""

    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET = "auth.password.reset"

    # User management
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_ACTIVATED = "user.activated"
    USER_DEACTIVATED = "user.deactivated"
    ROLE_CHANGED = "user.role.changed"

    # Content operations
    CONTENT_GENERATED = "content.generated"
    CONTENT_DELETED = "content.deleted"
    CONTENT_DISTRIBUTED = "content.distributed"

    # Project operations
    PROJECT_CREATED = "project.created"
    PROJECT_UPDATED = "project.updated"
    PROJECT_DELETED = "project.deleted"

    # Security events
    UNAUTHORIZED_ACCESS = "security.unauthorized_access"
    RATE_LIMIT_EXCEEDED = "security.rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    API_KEY_USED = "security.api_key_used"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Structured audit event."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.INFO
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None  # e.g., "project", "article"
    resource_id: Optional[str] = None
    action: str  # Human-readable action description
    details: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class AuditLogger:
    """
    Centralized audit logging system.

    Logs critical operations to:
    1. Structured log files (via loguru)
    2. Redis (for recent event queries)
    3. Database (for long-term storage and compliance)
    """

    def __init__(
        self,
        database_manager: Optional[DatabaseManager] = None,
        redis_client: Optional[RedisClient] = None,
    ):
        """Initialize audit logger with optional persistent storage."""
        self.db = database_manager
        self.redis = redis_client
        self.recent_events_key = "audit:recent_events"
        self.max_recent_events = 1000  # Keep last 1000 events in Redis

    async def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        success: bool = True,
        error_message: Optional[str] = None,
        **details: Any,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event being logged
            action: Human-readable description of the action
            user_id: ID of the user performing the action
            user_email: Email of the user
            ip_address: IP address of the request
            resource_type: Type of resource being acted upon
            resource_id: ID of the resource
            severity: Severity level
            success: Whether the action succeeded
            error_message: Error message if action failed
            **details: Additional context data

        Returns:
            Created AuditEvent
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            success=success,
            error_message=error_message,
            details=details,
        )

        # Log to structured logger
        log_data = event.model_dump(mode="json")
        log_level = self._severity_to_log_level(severity)
        logger.bind(**log_data).log(log_level, f"AUDIT: {action}")

        # Store in Redis for recent events (async, non-blocking)
        if self.redis:
            try:
                await self.redis.lpush(
                    self.recent_events_key, json.dumps(log_data, default=str)
                )
                await self.redis.ltrim(self.recent_events_key, 0, self.max_recent_events - 1)
                await self.redis.expire(self.recent_events_key, 86400 * 30)  # 30 days
            except Exception as e:
                logger.error(f"Failed to store audit event in Redis: {e}")

        # Store in database for long-term compliance
        if self.db:
            await self._store_in_database(event)

        return event

    def _severity_to_log_level(self, severity: AuditSeverity) -> str:
        """Convert audit severity to log level."""
        mapping = {
            AuditSeverity.INFO: "INFO",
            AuditSeverity.WARNING: "WARNING",
            AuditSeverity.ERROR: "ERROR",
            AuditSeverity.CRITICAL: "CRITICAL",
        }
        return mapping.get(severity, "INFO")

    async def _store_in_database(self, event: AuditEvent) -> None:
        """Persist audit event to database for compliance and long-term storage."""
        try:
            from infrastructure.schema import metadata

            # Lazily create table reference (table is created via Alembic migration)
            audit_events_table = Table(
                "audit_events",
                metadata,
                Column("id", PG_UUID, primary_key=True, default=uuid4),
                Column("event_id", String(255), nullable=False, unique=True, index=True),
                Column("timestamp", DateTime, nullable=False),
                Column("event_type", String(100), nullable=False, index=True),
                Column("severity", String(20), nullable=False),
                Column("user_id", String(255), index=True),
                Column("user_email", String(255)),
                Column("ip_address", String(45)),
                Column("user_agent", Text),
                Column("resource_type", String(100)),
                Column("resource_id", String(255)),
                Column("action", Text, nullable=False),
                Column("details", JSON),
                Column("success", Boolean, default=True),
                Column("error_message", Text),
                extend_existing=True,
            )

            query = insert(audit_events_table).values(
                event_id=event.event_id,
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                severity=event.severity.value,
                user_id=event.user_id,
                user_email=event.user_email,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                action=event.action,
                details=event.details,
                success=event.success,
                error_message=event.error_message,
            )
            await self.db.execute(query)
            logger.debug(f"Audit event persisted to database: {event.event_id}")
        except Exception as e:
            # Never let audit persistence failure break the application
            logger.error(f"Failed to persist audit event to database: {e}")

    async def get_recent_events(self, limit: int = 100) -> list[AuditEvent]:
        """Retrieve recent audit events from Redis."""
        if not self.redis:
            return []

        try:
            events_json = await self.redis.lrange(self.recent_events_key, 0, limit - 1)
            return [AuditEvent(**json.loads(e)) for e in events_json]
        except Exception as e:
            logger.error(f"Failed to retrieve recent audit events: {e}")
            return []


# Singleton instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    database_manager: Optional[DatabaseManager] = None,
    redis_client: Optional[RedisClient] = None,
) -> AuditLogger:
    """Get or create singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(database_manager, redis_client)
    return _audit_logger
