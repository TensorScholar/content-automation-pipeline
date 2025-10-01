"""
Exception Hierarchy & Error Handling Framework
===============================================
Type-safe exception taxonomy with structured context propagation,
retry metadata, and observability integration.

Architecture: Railway-Oriented Programming + Error Algebra
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from core.enums import ErrorSeverity

# =============================================================================
# BASE EXCEPTION CLASSES
# =============================================================================


class ContentAutomationException(Exception):
    """
    Root exception for all application errors.

    Implements structured error context with:
    - Unique error ID for distributed tracing
    - Severity classification for alerting
    - Structured context dictionary
    - Retry metadata
    - Timestamp for temporal analysis
    """

    def __init__(
        self,
        message: str,
        *,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None,
        retryable: bool = False,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)

        self.error_id: UUID = uuid4()
        self.message: str = message
        self.severity: ErrorSeverity = severity
        self.context: dict[str, Any] = context or {}
        self.error_code: Optional[str] = error_code
        self.retryable: bool = retryable
        self.timestamp: datetime = datetime.utcnow()

        # Exception chaining for causal analysis
        if cause:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception for logging/telemetry."""
        return {
            "error_id": str(self.error_id),
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.name,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.__cause__) if self.__cause__ else None,
        }

    def __str__(self) -> str:
        """Human-readable error representation."""
        parts = [f"[{self.severity.name}] {self.message}"]
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        if self.context:
            parts.append(f"Context: {self.context}")
        return " | ".join(parts)


# =============================================================================
# LLM & API EXCEPTIONS
# =============================================================================


class LLMException(ContentAutomationException):
    """Base exception for LLM-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


# Aliases for backward compatibility
LLMError = LLMException
LLMProviderError = LLMException


class LLMRateLimitError(LLMException):
    """API rate limit exceeded."""

    def __init__(
        self,
        message: str = "LLM API rate limit exceeded",
        *,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"retry_after_seconds": retry_after},
            error_code="LLM_RATE_LIMIT",
            **kwargs,
        )
        self.retry_after = retry_after


class LLMTimeoutError(LLMException):
    """API request timed out."""

    def __init__(
        self,
        message: str = "LLM API request timed out",
        *,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"timeout_seconds": timeout_seconds},
            error_code="LLM_TIMEOUT",
            **kwargs,
        )


class LLMInvalidResponseError(LLMException):
    """LLM returned malformed or invalid response."""

    def __init__(
        self,
        message: str = "LLM returned invalid response",
        *,
        response_text: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,  # May succeed on retry with different generation
            context={
                "response_preview": response_text[:500] if response_text else None,
                "expected_format": expected_format,
            },
            error_code="LLM_INVALID_RESPONSE",
            **kwargs,
        )


class LLMCostExceededError(LLMException):
    """Token budget or cost limit exceeded."""

    def __init__(
        self,
        message: str = "LLM cost budget exceeded",
        *,
        current_cost: Optional[float] = None,
        budget_limit: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            retryable=False,
            context={
                "current_cost_usd": current_cost,
                "budget_limit_usd": budget_limit,
            },
            error_code="LLM_COST_EXCEEDED",
            **kwargs,
        )


class TokenBudgetExceededError(LLMCostExceededError):
    """Alias for LLMCostExceededError for backward compatibility."""

    pass


class LLMModelNotAvailableError(LLMException):
    """Requested model is not available."""

    def __init__(
        self,
        message: str = "Requested LLM model not available",
        *,
        requested_model: Optional[str] = None,
        available_models: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=False,
            context={
                "requested_model": requested_model,
                "available_models": available_models,
            },
            error_code="LLM_MODEL_UNAVAILABLE",
            **kwargs,
        )


# =============================================================================
# INFRASTRUCTURE EXCEPTIONS
# =============================================================================


class InfrastructureException(ContentAutomationException):
    """Base exception for infrastructure errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)


# Alias for backward compatibility
InfrastructureError = InfrastructureException


# =============================================================================
# DATABASE EXCEPTIONS
# =============================================================================


class DatabaseException(InfrastructureException):
    """Base exception for database errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)


# Alias for backward compatibility
DatabaseError = DatabaseException


class DatabaseConnectionError(DatabaseException):
    """Failed to establish database connection."""

    def __init__(
        self,
        message: str = "Database connection failed",
        *,
        host: Optional[str] = None,
        database: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"host": host, "database": database},
            error_code="DB_CONNECTION_FAILED",
            **kwargs,
        )


class DatabaseQueryTimeoutError(DatabaseException):
    """Database query exceeded timeout."""

    def __init__(
        self,
        message: str = "Database query timeout",
        *,
        query_preview: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={
                "query_preview": query_preview[:200] if query_preview else None,
                "timeout_seconds": timeout_seconds,
            },
            error_code="DB_QUERY_TIMEOUT",
            **kwargs,
        )


class EntityNotFoundError(DatabaseException):
    """Requested entity does not exist."""

    def __init__(self, entity_type: str, entity_id: Any, message: Optional[str] = None, **kwargs):
        message = message or f"{entity_type} with ID {entity_id} not found"
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            retryable=False,
            context={"entity_type": entity_type, "entity_id": str(entity_id)},
            error_code="ENTITY_NOT_FOUND",
            **kwargs,
        )


# Alias for backward compatibility
NotFoundError = EntityNotFoundError


# =============================================================================
# CACHE EXCEPTIONS
# =============================================================================


class CacheException(ContentAutomationException):
    """Base exception for caching errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,  # Cache failures should degrade gracefully
            **kwargs,
        )


# Alias for backward compatibility
CacheError = CacheException


class CacheMissError(CacheException):
    """Requested key not found in cache."""

    def __init__(self, message: str = "Cache miss", *, cache_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            retryable=False,
            context={"cache_key": cache_key},
            error_code="CACHE_MISS",
            **kwargs,
        )


class CacheWriteError(CacheException):
    """Failed to write to cache."""

    def __init__(
        self, message: str = "Cache write failed", *, cache_key: Optional[str] = None, **kwargs
    ):
        super().__init__(
            message,
            retryable=True,
            context={"cache_key": cache_key},
            error_code="CACHE_WRITE_FAILED",
            **kwargs,
        )


# =============================================================================
# CONTENT GENERATION EXCEPTIONS
# =============================================================================


class ContentGenerationException(ContentAutomationException):
    """Base exception for content generation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


class WorkflowError(ContentGenerationException):
    """Workflow execution error."""

    def __init__(
        self,
        message: str = "Workflow execution failed",
        *,
        workflow_step: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"workflow_step": workflow_step},
            error_code="WORKFLOW_ERROR",
            **kwargs,
        )


class GenerationError(ContentGenerationException):
    """Content generation error."""

    def __init__(
        self,
        message: str = "Content generation failed",
        *,
        generation_step: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"generation_step": generation_step},
            error_code="GENERATION_ERROR",
            **kwargs,
        )


class QualityValidationError(ContentGenerationException):
    """Content quality validation error."""

    def __init__(
        self,
        message: str = "Content quality validation failed",
        *,
        validation_issues: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"validation_issues": validation_issues},
            error_code="QUALITY_VALIDATION_ERROR",
            **kwargs,
        )


class ContentQualityError(ContentGenerationException):
    """Generated content failed quality validation."""

    def __init__(
        self,
        message: str = "Content failed quality validation",
        *,
        quality_metrics: Optional[dict[str, float]] = None,
        failed_checks: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,  # Can regenerate with adjusted parameters
            context={
                "quality_metrics": quality_metrics,
                "failed_checks": failed_checks,
            },
            error_code="CONTENT_QUALITY_FAILED",
            **kwargs,
        )


class ContentTooShortError(ContentGenerationException):
    """Generated content below minimum word count."""

    def __init__(
        self, actual_words: int, minimum_words: int, message: Optional[str] = None, **kwargs
    ):
        message = message or f"Content too short: {actual_words} words (min: {minimum_words})"
        super().__init__(
            message,
            retryable=True,
            context={
                "actual_word_count": actual_words,
                "minimum_word_count": minimum_words,
            },
            error_code="CONTENT_TOO_SHORT",
            **kwargs,
        )


class ContentTooLongError(ContentGenerationException):
    """Generated content exceeds maximum word count."""

    def __init__(
        self, actual_words: int, maximum_words: int, message: Optional[str] = None, **kwargs
    ):
        message = message or f"Content too long: {actual_words} words (max: {maximum_words})"
        super().__init__(
            message,
            retryable=True,
            context={
                "actual_word_count": actual_words,
                "maximum_word_count": maximum_words,
            },
            error_code="CONTENT_TOO_LONG",
            **kwargs,
        )


# =============================================================================
# SCRAPING EXCEPTIONS
# =============================================================================


class ScrapingException(ContentAutomationException):
    """Base exception for web scraping errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


# Alias for backward compatibility
ScrapingError = ScrapingException


class ScrapingTimeoutError(ScrapingException):
    """Web scraping request timed out."""

    def __init__(
        self,
        message: str = "Scraping request timed out",
        *,
        url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"url": url, "timeout_seconds": timeout_seconds},
            error_code="SCRAPING_TIMEOUT",
            **kwargs,
        )


class ScrapingBlockedError(ScrapingException):
    """Scraping blocked by target website (403, 429, etc.)."""

    def __init__(
        self,
        message: str = "Scraping blocked by target website",
        *,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=False,  # Likely requires manual intervention
            context={"url": url, "status_code": status_code},
            error_code="SCRAPING_BLOCKED",
            **kwargs,
        )


class InsufficientSampleError(ScrapingException):
    """Insufficient articles scraped for pattern inference."""

    def __init__(
        self, scraped_count: int, minimum_required: int, message: Optional[str] = None, **kwargs
    ):
        message = message or (
            f"Insufficient samples: {scraped_count} scraped, "
            f"{minimum_required} required for inference"
        )
        super().__init__(
            message,
            retryable=False,
            context={
                "scraped_count": scraped_count,
                "minimum_required": minimum_required,
            },
            error_code="INSUFFICIENT_SAMPLE",
            **kwargs,
        )


# =============================================================================
# PROJECT MANAGEMENT EXCEPTIONS
# =============================================================================


class ProjectException(ContentAutomationException):
    """Base exception for project-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.WARNING, **kwargs)


class ProjectNotFoundError(ProjectException):
    """Project does not exist."""

    def __init__(self, project_id: UUID, message: Optional[str] = None, **kwargs):
        message = message or f"Project {project_id} not found"
        super().__init__(
            message,
            retryable=False,
            context={"project_id": str(project_id)},
            error_code="PROJECT_NOT_FOUND",
            **kwargs,
        )


class RulebookParseError(ProjectException):
    """Failed to parse rulebook."""

    def __init__(
        self,
        message: str = "Failed to parse rulebook",
        *,
        parse_errors: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=False,
            context={"parse_errors": parse_errors},
            error_code="RULEBOOK_PARSE_FAILED",
            **kwargs,
        )


class InvalidProjectConfigError(ProjectException):
    """Project configuration is invalid."""

    def __init__(
        self,
        message: str = "Invalid project configuration",
        *,
        validation_errors: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=False,
            context={"validation_errors": validation_errors},
            error_code="INVALID_PROJECT_CONFIG",
            **kwargs,
        )


# =============================================================================
# DECISION EXCEPTIONS
# =============================================================================


class DecisionException(ContentAutomationException):
    """Base exception for decision-making errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


# Alias for backward compatibility
DecisionError = DecisionException


class PlanningError(ContentAutomationException):
    """Content planning error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


class ProcessingError(ContentAutomationException):
    """General processing error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


class InsufficientContextError(ContentAutomationException):
    """Insufficient context for operation."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.WARNING, **kwargs)


class ModelRoutingError(ContentAutomationException):
    """Model routing error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


class BudgetExceededError(ContentAutomationException):
    """Budget exceeded error."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================


class ValidationException(ContentAutomationException):
    """Base exception for validation errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.WARNING, **kwargs)


# Alias for backward compatibility
ValidationError = ValidationException


class SchemaValidationError(ValidationException):
    """Data does not conform to expected schema."""

    def __init__(
        self,
        message: str = "Schema validation failed",
        *,
        validation_errors: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=False,
            context={"validation_errors": validation_errors},
            error_code="SCHEMA_VALIDATION_FAILED",
            **kwargs,
        )


class SemanticValidationError(ValidationException):
    """Content semantically invalid despite syntactic correctness."""

    def __init__(
        self,
        message: str = "Semantic validation failed",
        *,
        issues: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"semantic_issues": issues},
            error_code="SEMANTIC_VALIDATION_FAILED",
            **kwargs,
        )


# =============================================================================
# KEYWORD RESEARCH EXCEPTIONS
# =============================================================================


class KeywordResearchException(ContentAutomationException):
    """Base exception for keyword research errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


class NoKeywordsFoundError(KeywordResearchException):
    """Keyword research yielded no results."""

    def __init__(
        self, message: str = "No keywords found for topic", *, topic: Optional[str] = None, **kwargs
    ):
        super().__init__(
            message,
            retryable=False,
            context={"topic": topic},
            error_code="NO_KEYWORDS_FOUND",
            **kwargs,
        )


class KeywordAPIError(KeywordResearchException):
    """External keyword research API error."""

    def __init__(
        self,
        message: str = "Keyword research API error",
        *,
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={"api_name": api_name, "status_code": status_code},
            error_code="KEYWORD_API_ERROR",
            **kwargs,
        )


# =============================================================================
# DISTRIBUTION EXCEPTIONS
# =============================================================================


class DistributionException(ContentAutomationException):
    """Base exception for distribution errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


# Alias for backward compatibility
DistributionError = DistributionException


class TelegramDistributionError(DistributionException):
    """Failed to distribute content via Telegram."""

    def __init__(
        self,
        message: str = "Telegram distribution failed",
        *,
        channel_id: Optional[str] = None,
        error_details: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={
                "channel_id": channel_id,
                "error_details": error_details,
            },
            error_code="TELEGRAM_DISTRIBUTION_FAILED",
            **kwargs,
        )


class ChannelNotConfiguredError(DistributionException):
    """Distribution channel not configured for project."""

    def __init__(
        self, channel_name: str, project_id: UUID, message: Optional[str] = None, **kwargs
    ):
        message = message or (f"Channel '{channel_name}' not configured for project {project_id}")
        super().__init__(
            message,
            retryable=False,
            context={
                "channel_name": channel_name,
                "project_id": str(project_id),
            },
            error_code="CHANNEL_NOT_CONFIGURED",
            **kwargs,
        )


# =============================================================================
# NLP & EMBEDDING EXCEPTIONS
# =============================================================================


class NLPException(ContentAutomationException):
    """Base exception for NLP processing errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.ERROR, **kwargs)


class EmbeddingGenerationError(NLPException):
    """Failed to generate text embeddings."""

    def __init__(
        self,
        message: str = "Embedding generation failed",
        *,
        text_preview: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            retryable=True,
            context={
                "text_preview": text_preview[:100] if text_preview else None,
                "model_name": model_name,
            },
            error_code="EMBEDDING_GENERATION_FAILED",
            **kwargs,
        )


class ModelLoadError(NLPException):
    """Failed to load NLP model."""

    def __init__(
        self,
        message: str = "Failed to load NLP model",
        *,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            retryable=False,
            context={"model_name": model_name},
            error_code="MODEL_LOAD_FAILED",
            **kwargs,
        )


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================


class ConfigurationException(ContentAutomationException):
    """Base exception for configuration errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)


class MissingConfigurationError(ConfigurationException):
    """Required configuration parameter missing."""

    def __init__(self, parameter_name: str, message: Optional[str] = None, **kwargs):
        message = message or f"Required configuration parameter missing: {parameter_name}"
        super().__init__(
            message,
            retryable=False,
            context={"parameter_name": parameter_name},
            error_code="MISSING_CONFIGURATION",
            **kwargs,
        )


class InvalidConfigurationError(ConfigurationException):
    """Configuration parameter has invalid value."""

    def __init__(
        self, parameter_name: str, invalid_value: Any, message: Optional[str] = None, **kwargs
    ):
        message = message or (f"Invalid configuration value for {parameter_name}: {invalid_value}")
        super().__init__(
            message,
            retryable=False,
            context={
                "parameter_name": parameter_name,
                "invalid_value": str(invalid_value),
            },
            error_code="INVALID_CONFIGURATION",
            **kwargs,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def handle_exception(
    exc: Exception,
    *,
    log_function: Optional[callable] = None,
    raise_new: bool = False,
    default_return: Any = None,
) -> Any:
    """
    Centralized exception handler with logging and error transformation.

    Args:
        exc: The exception to handle
        log_function: Optional logging function (e.g., logger.error)
        raise_new: If True, wraps unknown exceptions in ContentAutomationException
        default_return: Value to return if not re-raising

    Returns:
        default_return value if not raising

    Raises:
        ContentAutomationException: If raise_new=True and exc is not already one
    """
    if log_function:
        if isinstance(exc, ContentAutomationException):
            log_function(f"Application error: {exc}", extra={"error_context": exc.to_dict()})
        else:
            log_function(f"Unexpected error: {exc}", exc_info=True)

    if raise_new and not isinstance(exc, ContentAutomationException):
        raise ContentAutomationException(
            message=str(exc),
            cause=exc,
            severity=ErrorSeverity.ERROR,
        ) from exc

    return default_return


def is_retryable(exc: Exception) -> bool:
    """
    Determine if an exception is retryable.

    Args:
        exc: Exception to check

    Returns:
        True if operation should be retried, False otherwise
    """
    if isinstance(exc, ContentAutomationException):
        return exc.retryable

    # Heuristic for non-application exceptions
    retryable_types = (
        TimeoutError,
        ConnectionError,
        IOError,
    )
    return isinstance(exc, retryable_types)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base
    "ContentAutomationException",
    # LLM
    "LLMException",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMInvalidResponseError",
    "LLMCostExceededError",
    "TokenBudgetExceededError",
    "LLMModelNotAvailableError",
    # Infrastructure
    "InfrastructureException",
    "InfrastructureError",
    # Database
    "DatabaseException",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryTimeoutError",
    "EntityNotFoundError",
    "NotFoundError",
    # Cache
    "CacheException",
    "CacheError",
    "CacheMissError",
    "CacheWriteError",
    # Content
    "ContentGenerationException",
    "WorkflowError",
    "GenerationError",
    "QualityValidationError",
    "ContentQualityError",
    "ContentTooShortError",
    "ContentTooLongError",
    # Scraping
    "ScrapingException",
    "ScrapingError",
    "ScrapingTimeoutError",
    "ScrapingBlockedError",
    "InsufficientSampleError",
    # Project
    "ProjectException",
    "ProjectNotFoundError",
    "RulebookParseError",
    "InvalidProjectConfigError",
    # Decision
    "DecisionException",
    "DecisionError",
    "PlanningError",
    "ProcessingError",
    "InsufficientContextError",
    "ModelRoutingError",
    "BudgetExceededError",
    # Validation
    "ValidationException",
    "ValidationError",
    "SchemaValidationError",
    "SemanticValidationError",
    # Keywords
    "KeywordResearchException",
    "NoKeywordsFoundError",
    "KeywordAPIError",
    # Distribution
    "DistributionException",
    "DistributionError",
    "TelegramDistributionError",
    "ChannelNotConfiguredError",
    # NLP
    "NLPException",
    "EmbeddingGenerationError",
    "ModelLoadError",
    # Configuration
    "ConfigurationException",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    # Utilities
    "handle_exception",
    "is_retryable",
]
