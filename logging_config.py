"""
Structured Logging Configuration for Production
================================================

Provides environment-aware logging with:
- JSON format for production (machine-parseable)
- Human-readable format for development
- Correlation ID propagation for distributed tracing
- Performance metrics and structured context
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from infrastructure.redaction import redact_secrets, redact_text

# Determine environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
IS_PRODUCTION = ENVIRONMENT == "production"


class JSONFormatter(logging.Formatter):
    """
    Production-grade JSON log formatter.

    Output format for log aggregation systems (ELK, CloudWatch, Datadog):
    {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "level": "INFO",
        "logger": "api.main",
        "message": "Request processed",
        "correlation_id": "uuid",
        "extra": {...}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }

        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id

        # Add request ID if present
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": redact_text(str(record.exc_info[1])) if record.exc_info[1] else None,
                "traceback": redact_text(self.formatException(record.exc_info)),
            }

        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "message",
            "correlation_id",
            "request_id",
        }

        extra = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        if extra:
            log_data["extra"] = redact_secrets(extra)

        return json.dumps(log_data, default=str)


class DevelopmentFormatter(logging.Formatter):
    """
    Human-readable formatter for development.

    Output: 2024-01-01 12:00:00 | INFO     | api.main | Request processed
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname

        # Add color if terminal supports it
        if sys.stderr.isatty():
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level_str = f"{color}{level:8}{reset}"
        else:
            level_str = f"{level:8}"

        message = redact_text(record.getMessage())

        # Add correlation ID if present
        correlation = ""
        if hasattr(record, "correlation_id"):
            correlation = f" | cid={record.correlation_id[:8]}"

        return f"{timestamp} | {level_str} | {record.name} | {message}{correlation}"


def get_formatter() -> logging.Formatter:
    """Get appropriate formatter based on environment."""
    if IS_PRODUCTION:
        return JSONFormatter()
    return DevelopmentFormatter()


def configure_logging(
    level: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Force JSON format (default: auto based on environment)
    """
    log_level = level or LOG_LEVEL
    use_json = json_format if json_format is not None else IS_PRODUCTION

    # Create formatter
    formatter = JSONFormatter() if use_json else DevelopmentFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class CorrelationAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds correlation ID to all log records.

    Usage:
        logger = CorrelationAdapter(logging.getLogger(__name__), {"correlation_id": "..."})
        logger.info("Processing request")
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


# Legacy compatibility - LOGGING_CONFIG dict format
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": JSONFormatter,
        },
        "development": {
            "()": DevelopmentFormatter,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if IS_PRODUCTION else "development",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "loggers": {
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "uvicorn.access": {"level": "WARNING"},
    },
}


# Initialize logging on module import
configure_logging()
