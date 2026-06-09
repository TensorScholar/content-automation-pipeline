"""Unified LLM Client - Multi-Provider Interface with Automatic Failover.

Provides a single interface for OpenAI and Anthropic with:
- Unified API calls via litellm library
- Automatic provider routing based on model name prefixes
- Exponential backoff retries using tenacity
- Token usage tracking and cost calculation
- Optional response caching
- Circuit breaker pattern for network resilience

Philosophy:
    Pragmatic simplicity over architectural complexity. Unified routing
    via litellm, reliable retries, circuit breakers for resilience.

Example:
    >>> client = UnifiedLLMClient()
    >>> response = await client.generate(
    ...     model="gpt-4",
    ...     prompt="Explain quantum computing",
    ...     max_tokens=500
    ... )
    >>> print(f"Cost: ${response.cost:.4f}")
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Optional

import litellm
from litellm import acompletion
from loguru import logger
from openai import APITimeoutError, RateLimitError  # LiteLLM raises OpenAI-compatible exceptions
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

DEFAULT_TIMEOUT = 120.0
RETRY_MIN_WAIT = 2
RETRY_MAX_WAIT = 60
DEFAULT_MAX_RETRIES = 3

# Network resilience configuration
CIRCUIT_BREAKER_THRESHOLD = 5  # Failures before opening circuit
CIRCUIT_BREAKER_RECOVERY_TIME = 60  # Seconds before attempting recovery
DNS_RETRY_ERRORS = ["DNS resolution failed", "Name resolution failed", "getaddrinfo failed"]


def _secret_value(secret: object | None) -> str | None:
    if secret is None:
        return None
    if hasattr(secret, "get_secret_value"):
        return secret.get_secret_value()  # type: ignore[no-any-return]
    return str(secret)


def _is_provider_request_error(exc: Exception) -> bool:
    """Return true for non-transient provider/account/request failures."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "400 bad request",
            "invalid_request_error",
            "credit balance is too low",
            "insufficient credits",
            "billing",
            "quota exceeded",
            "exceeded your current quota",
            "resource_exhausted",
            "free tier",
            "403",
            "forbidden",
            "permission denied",
            "model_not_found",
            "not found",
            "not allowed",
        )
    )


def _provider_request_message(model: str, exc: Exception) -> str:
    """Build an actionable message for provider-side 4xx failures."""
    text = str(exc)
    lowered = text.lower()
    if (
        "credit balance is too low" in lowered
        or "insufficient credits" in lowered
        or "quota exceeded" in lowered
        or "exceeded your current quota" in lowered
        or "resource_exhausted" in lowered
    ):
        return (
            f"LLM provider account has insufficient credits or quota for model {model}. "
            "Add provider credits, wait for quota reset, or select another configured model."
        )
    if "model" in lowered and ("not found" in lowered or "not allowed" in lowered):
        return (
            f"LLM model {model} is unavailable for the configured provider account. "
            "Use a model ID enabled for this account."
        )
    if "403" in lowered or "forbidden" in lowered or "permission denied" in lowered:
        return (
            f"LLM provider denied access for model {model}. "
            "Verify the provider API key, enabled API access, and network or regional restrictions."
        )
    return (
        f"LLM provider rejected the request for model {model}. "
        "Check model access, billing credits, and request parameters."
    )


def _is_retryable_llm_exception(exc: BaseException) -> bool:
    """Retry only transient LLM failures.

    The previous broad `Exception` retry retried deterministic provider 4xx
    failures such as exhausted credits. That wastes latency and hides the real
    launch blocker behind misleading connection errors.
    """
    if _is_provider_request_error(exc):  # type: ignore[arg-type]
        return False
    if isinstance(exc, (LLMRateLimitError, LLMTimeoutError, LLMConnectionError)):
        return True
    retryable = getattr(exc, "retryable", None)
    if retryable is not None:
        return bool(retryable)
    return isinstance(
        exc,
        (
            RateLimitError,
            APITimeoutError,
            litellm.RateLimitError,
            litellm.Timeout,
            litellm.APIConnectionError,
            ConnectionError,
            TimeoutError,
        ),
    )


# ============================================================================
# SAFE LOGGING FOR TENACITY RETRIES
# ============================================================================

def safe_before_sleep_log(logger_instance, log_level):
    """
    Safe wrapper for tenacity's before_sleep_log that prevents Loguru formatting crashes.

    When exception messages contain curly braces (e.g., from JSON errors), Loguru tries to
    format them as f-strings, causing KeyError. This function sanitizes messages first.
    """
    def log_it(retry_state):
        if retry_state.outcome and retry_state.outcome.failed:
            exception = retry_state.outcome.exception()
            # Sanitize exception message to prevent Loguru formatting issues
            exception_str = str(exception).replace("{", "{{").replace("}", "}}")
            verb, value = "raised", f"{exception.__class__.__name__}: {exception_str}"
        else:
            verb, value = "returned", retry_state.outcome.result()

        logger_instance.log(
            log_level,
            f"Retrying {{}} in {{}} seconds as it {{}} {{}}.".replace("{{}}", "{}"),
            retry_state.fn.__name__ if hasattr(retry_state.fn, '__name__') else retry_state.fn,
            getattr(retry_state.next_action, 'sleep', 0),
            verb,
            value
        )
    return log_it


class CircuitBreaker:
    """Circuit breaker pattern for network resilience.

    Prevents cascade failures by tracking consecutive errors and temporarily
    blocking requests when a threshold is exceeded. Automatically recovers
    after a cooldown period.

    Thread-safe implementation using asyncio.Lock with lazy initialization
    to ensure the lock binds to the correct event loop (critical for Celery).
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        recovery_time: int = CIRCUIT_BREAKER_RECOVERY_TIME,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half-open
        self._lock: asyncio.Lock | None = None  # Lazy init for event loop safety

    def _get_lock(self) -> asyncio.Lock:
        """Get or create lock bound to current event loop.

        Lazy initialization ensures the lock is created in the same event loop
        where it will be used, preventing 'attached to different loop' errors
        when Celery tasks use asyncio.run().

        FIXED: Uses thread-safe double-checked locking to prevent race condition
        where multiple concurrent calls could create multiple locks.
        """
        if self._lock is None:
            import threading
            # Thread-safe initialization: use threading.Lock for synchronization
            _init_lock = threading.Lock()
            with _init_lock:
                if self._lock is None:  # Double-check inside thread lock
                    self._lock = asyncio.Lock()
        return self._lock

    async def record_success(self) -> None:
        """Record a successful call, resetting the failure count."""
        async with self._get_lock():
            self._failure_count = 0
            self._state = "closed"

    async def record_failure(self) -> None:
        """Record a failed call, potentially opening the circuit."""
        async with self._get_lock():
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(
                    f"Circuit breaker OPENED for {self.name} | "
                    f"failures={self._failure_count} threshold={self.failure_threshold}"
                )

    async def can_proceed(self) -> bool:
        """Check if a request can proceed through the circuit."""
        async with self._get_lock():
            if self._state == "closed":
                return True

            if self._state == "open":
                # Check if recovery time has passed
                if self._last_failure_time and (time.time() - self._last_failure_time) >= self.recovery_time:
                    self._state = "half-open"
                    logger.info(f"Circuit breaker HALF-OPEN for {self.name} | attempting recovery")
                    return True
                return False

            # half-open state allows one request through
            return True

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state,
            "failure_count": self._failure_count,
            "threshold": self.failure_threshold,
            "last_failure": self._last_failure_time,
        }


def is_dns_error(error: Exception) -> bool:
    """Check if an error is DNS-related and should trigger special handling."""
    error_str = str(error).lower()
    return any(dns_err.lower() in error_str for dns_err in DNS_RETRY_ERRORS)


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"


@dataclass(frozen=True)
class TokenUsage:
    """Immutable token usage metrics for a single LLM request.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated response.
        total_tokens: Sum of prompt and completion tokens.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self):
        assert self.prompt_tokens >= 0, "Prompt tokens must be non-negative"
        assert self.completion_tokens >= 0, "Completion tokens must be non-negative"
        assert self.total_tokens >= 0, "Total tokens must be non-negative"
        # Note: total_tokens may not equal sum for providers that count tokens differently


@dataclass(frozen=True)
class ModelPricing:
    """Token-based pricing configuration for LLM models.

    Attributes:
        input_cost_per_1k: Cost per 1000 input tokens in USD.
        output_cost_per_1k: Cost per 1000 output tokens in USD.
    """
    input_cost_per_1k: float
    output_cost_per_1k: float

    def calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate total request cost with high precision.

        Args:
            usage: Token usage metrics from LLM response.

        Returns:
            Total cost in USD rounded to 6 decimal places.
        """
        input_cost = (
            Decimal(str(usage.prompt_tokens))
            * Decimal(str(self.input_cost_per_1k))
            / Decimal("1000")
        )
        output_cost = (
            Decimal(str(usage.completion_tokens))
            * Decimal(str(self.output_cost_per_1k))
            / Decimal("1000")
        )
        total = (input_cost + output_cost).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        return float(total)


# Default pricing database - should match config/settings.py model_pricing
# These are validated Anthropic model identifiers
DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    # OpenAI models
    "gpt-4": ModelPricing(input_cost_per_1k=0.03, output_cost_per_1k=0.06),
    "gpt-4-turbo": ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.03),
    "gpt-4o": ModelPricing(input_cost_per_1k=0.0025, output_cost_per_1k=0.01),
    "gpt-4o-mini": ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    "gpt-3.5-turbo": ModelPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    # Anthropic models (official model IDs)
    "claude-3-opus-20240229": ModelPricing(input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    "claude-3-5-sonnet-20241022": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-3-5-haiku-20241022": ModelPricing(input_cost_per_1k=0.001, output_cost_per_1k=0.005),
    "claude-3-sonnet-20240229": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-3-haiku-20240307": ModelPricing(input_cost_per_1k=0.00025, output_cost_per_1k=0.00125),
    # Claude 4.5 models (short aliases + dated variants)
    "claude-haiku-4-5": ModelPricing(input_cost_per_1k=0.001, output_cost_per_1k=0.005),
    "claude-haiku-4-5-20251001": ModelPricing(input_cost_per_1k=0.001, output_cost_per_1k=0.005),
    "claude-sonnet-4-5": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-sonnet-4-5-20250514": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-opus-4-5": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.025),
    "claude-opus-4-5-20250630": ModelPricing(input_cost_per_1k=0.005, output_cost_per_1k=0.025),
    # Gemini paid-tier reference pricing. Free-tier usage is still reported as
    # estimated cost because provider billing state is not exposed per request.
    "gemini-2.5-flash-lite": ModelPricing(input_cost_per_1k=0.0001, output_cost_per_1k=0.0004),
    "gemini-2.5-flash": ModelPricing(input_cost_per_1k=0.0003, output_cost_per_1k=0.0025),
}


@dataclass(frozen=True)
class LLMResponse:
    """Complete LLM generation response with usage metrics and metadata.

    Attributes:
        content: Generated text content.
        model: Model identifier used for generation.
        usage: Token usage statistics.
        cost: Total cost in USD.
        latency_ms: Request duration in milliseconds.
        provider: LLM provider (OpenAI or Anthropic).
        timestamp: Response creation time (UTC).
        finish_reason: Completion reason (e.g., 'stop', 'length').
    """
    content: str
    model: str
    usage: TokenUsage
    cost: float
    latency_ms: float
    provider: ModelProvider
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finish_reason: str | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "provider": self.provider.value,
            "timestamp": self.timestamp.isoformat(),
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMResponse":
        """Create LLMResponse from dictionary."""
        return cls(
            content=data["content"],
            model=data["model"],
            usage=TokenUsage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            ),
            cost=data["cost"],
            latency_ms=data["latency_ms"],
            provider=ModelProvider(data["provider"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            finish_reason=data.get("finish_reason"),
        )


class LLMRequest(BaseModel):
    """Type-safe, validated LLM generation request.

    All parameters are validated at instantiation. Frozen to ensure immutability.
    """
    model_config = {"frozen": True}

    prompt: str = Field(..., min_length=1, max_length=100000)
    model: str = Field(..., pattern=r"^(gpt-|claude-|gemini-|local-|openai/|anthropic/|gemini/)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(..., ge=1, le=32000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop_sequences: list[str] | None = Field(default=None, max_length=4)


# ============================================================================
# UNIFIED LLM CLIENT
# ============================================================================


class UnifiedLLMClient:
    """Unified multi-provider LLM client with automatic routing and retries.

    Provides a single interface to OpenAI and Anthropic via litellm with:
    - Automatic routing from model name (gpt-* → OpenAI, claude-* → Anthropic)
    - Exponential backoff retries for rate limits and timeouts
    - Circuit breaker pattern for network resilience
    - Token usage tracking and cost calculation
    - Optional response caching

    Example:
        >>> client = UnifiedLLMClient()
        >>> response = await client.generate(
        ...     model="gpt-4-turbo",
        ...     prompt="Explain photosynthesis",
        ...     max_tokens=200
        ... )
        >>> print(f"Generated {response.usage.total_tokens} tokens for ${response.cost:.4f}")
    """

    def __init__(
        self,
        cache_manager: Optional[object] = None,
        metrics_collector: Optional[object] = None,
    ):
        """
        Initialize the unified client.

        Args:
            cache_manager: Optional cache for responses
            metrics_collector: Optional metrics tracking
        """
        self.cache_manager = cache_manager
        self.metrics = metrics_collector

        # Track local LLM configuration for special handling
        self._local_base_url: str | None = None
        self._active_provider: str | None = None  # Track which provider is active

        # Circuit breakers for each provider (network resilience)
        self._circuit_breakers = {
            ModelProvider.OPENAI: CircuitBreaker("openai"),
            ModelProvider.ANTHROPIC: CircuitBreaker("anthropic"),
            ModelProvider.GEMINI: CircuitBreaker("gemini"),
            ModelProvider.LOCAL: CircuitBreaker("local"),
        }

        # Request coalescing: prevents duplicate identical requests
        # Maps request_hash -> asyncio.Future for in-flight requests
        self._in_flight_requests: dict[str, asyncio.Future] = {}
        self._coalescing_lock: asyncio.Lock | None = None  # Lazy init for event loop safety

        self._initialize_providers()

    def _get_coalescing_lock(self) -> asyncio.Lock:
        """Get or create coalescing lock bound to current event loop."""
        if self._coalescing_lock is None:
            self._coalescing_lock = asyncio.Lock()
        return self._coalescing_lock

    def _compute_request_hash(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Compute hash for request coalescing."""
        key = f"{model}:{temperature}:{max_tokens}:{prompt}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def _initialize_providers(self) -> None:
        """Validate API keys are available (litellm uses env vars directly).

        Logs warnings for unavailable providers without raising errors,
        allowing the application to start with any subset of providers.
        """
        try:
            from config.settings import get_settings

            llm_settings = get_settings().llm
        except Exception as exc:
            logger.warning(f"Could not load LLM settings while initializing providers: {exc}")
            llm_settings = None

        # OpenAI - litellm reads OPENAI_API_KEY automatically
        openai_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_OPENAI_API_KEY")
            or _secret_value(getattr(llm_settings, "openai_api_key", None))
        )
        if openai_key:
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = openai_key
            logger.info("✓ OpenAI API key configured (litellm)")
        else:
            logger.warning("⚠ OPENAI_API_KEY not found - OpenAI models unavailable")

        # Anthropic - litellm reads ANTHROPIC_API_KEY automatically
        anthropic_key = (
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("LLM_ANTHROPIC_API_KEY")
            or _secret_value(getattr(llm_settings, "anthropic_api_key", None))
        )
        if anthropic_key:
            # Set for litellm if using alternate env var name
            if not os.getenv("ANTHROPIC_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            logger.info("✓ Anthropic API key configured (litellm)")
            if not self._active_provider:
                self._active_provider = "anthropic"
        else:
            logger.warning("⚠ ANTHROPIC_API_KEY not found - Anthropic models unavailable")

        # Gemini / Google AI Studio - LiteLLM requires GEMINI_API_KEY when using
        # the gemini/ provider route. GOOGLE_API_KEY and LLM_GEMINI_API_KEY are
        # accepted as local aliases and normalized here.
        gemini_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("LLM_GEMINI_API_KEY")
            or _secret_value(getattr(llm_settings, "gemini_api_key", None))
        )
        if gemini_key:
            if not os.getenv("GEMINI_API_KEY"):
                os.environ["GEMINI_API_KEY"] = gemini_key
            logger.info("✓ Gemini API key configured (litellm)")
            if not self._active_provider:
                self._active_provider = "gemini"
        else:
            logger.warning("⚠ GEMINI_API_KEY not found - Gemini models unavailable")

        # Local (Ollama/OpenAI-compatible) - needs special handling
        local_url = os.getenv("LOCAL_LLM_URL")
        if local_url:
            self._local_base_url = local_url.rstrip("/")
            logger.info(f"✓ Local LLM configured | base_url={self._local_base_url}")
            if not self._active_provider:
                self._active_provider = "local"
        else:
            logger.warning("⚠ LOCAL_LLM_URL not set - Local models unavailable")

        # Configure litellm settings
        litellm.drop_params = True  # Drop unsupported params instead of erroring

        # Log active provider
        if self._active_provider or openai_key:
            logger.info("✓ LiteLLM ready | providers available")
        else:
            logger.error("✗ No LLM provider available! Set GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or LOCAL_LLM_URL")

    def _provider_has_credentials(self, provider: ModelProvider) -> bool:
        """Return whether a provider can be attempted in this process."""
        try:
            from config.settings import get_settings

            llm_settings = get_settings().llm
        except Exception:
            llm_settings = None

        if provider == ModelProvider.OPENAI:
            return bool(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("LLM_OPENAI_API_KEY")
                or _secret_value(getattr(llm_settings, "openai_api_key", None))
            )
        if provider == ModelProvider.ANTHROPIC:
            return bool(
                os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("LLM_ANTHROPIC_API_KEY")
                or _secret_value(getattr(llm_settings, "anthropic_api_key", None))
            )
        if provider == ModelProvider.GEMINI:
            return bool(
                os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("LLM_GEMINI_API_KEY")
                or _secret_value(getattr(llm_settings, "gemini_api_key", None))
            )
        if provider == ModelProvider.LOCAL:
            return bool(self._local_base_url)
        return False

    def _fallback_candidates(self, model: str) -> list[str]:
        """Build ordered fallback models without adding routing infrastructure.

        Explicit `LLM_FALLBACK_MODEL` wins. Remaining configured providers and
        same-provider alternate models are considered in deterministic order.
        """
        try:
            self._determine_provider(model)
        except LLMError:
            return []

        candidates = [
            os.getenv("LLM_FALLBACK_MODEL"),
            os.getenv("LLM_OPENAI_FALLBACK_MODEL") or "gpt-4o-mini",
            os.getenv("LLM_ANTHROPIC_FALLBACK_MODEL")
            or os.getenv("LLM_ANTHROPIC_MODEL")
            or "claude-haiku-4-5",
            os.getenv("LLM_GEMINI_FALLBACK_MODEL")
            or os.getenv("LLM_GEMINI_MODEL")
            or "gemini-2.5-flash-lite",
            os.getenv("LLM_LOCAL_FALLBACK_MODEL"),
        ]

        usable: list[str] = []
        seen = {model}
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                provider = self._determine_provider(candidate)
            except LLMError:
                logger.warning(f"Skipping invalid fallback model name: {candidate}")
                continue
            if self._provider_has_credentials(provider):
                usable.append(candidate)
        return usable

    async def _try_fallbacks(
        self,
        *,
        primary_model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: list[str] | None,
        primary_error: Exception,
    ) -> LLMResponse | None:
        """Try configured fallback models after the primary provider fails."""
        for fallback_model in self._fallback_candidates(primary_model):
            provider = self._determine_provider(fallback_model)
            circuit_breaker = self._circuit_breakers.get(provider)
            if circuit_breaker and not await circuit_breaker.can_proceed():
                logger.warning(
                    f"Skipping fallback {fallback_model}: circuit breaker open for {provider.value}"
                )
                continue

            logger.warning(
                f"LLM fallback attempt | primary={primary_model} | "
                f"fallback={fallback_model} | cause={primary_error}"
            )
            try:
                return await self.generate(
                    model=fallback_model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    _allow_fallback=False,
                )
            except (LLMProviderError, LLMRateLimitError, LLMTimeoutError, LLMConnectionError, LLMError) as exc:
                logger.warning(
                    f"LLM fallback failed | model={fallback_model} | error={exc}"
                )
                continue
        return None

    def _determine_provider(self, model: str) -> ModelProvider:
        """Route model name to provider based on prefix matching.

        Args:
            model: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet').

        Returns:
            Matched provider enum value.

        Raises:
            LLMError: If model prefix doesn't match any known provider.
        """
        model_lower = model.lower()
        if model_lower.startswith("openai/"):
            return ModelProvider.OPENAI
        elif model_lower.startswith("anthropic/"):
            return ModelProvider.ANTHROPIC
        elif model_lower.startswith("gemini/"):
            return ModelProvider.GEMINI
        elif model_lower.startswith("gpt-"):
            return ModelProvider.OPENAI
        elif model_lower.startswith("claude-"):
            return ModelProvider.ANTHROPIC
        elif model_lower.startswith("gemini-"):
            return ModelProvider.GEMINI
        elif model_lower.startswith("local-"):
            return ModelProvider.LOCAL
        else:
            raise LLMError(f"Unknown model prefix: {model}")

    def _litellm_model_name(self, model: str) -> str:
        """Return the model identifier LiteLLM expects.

        Newer LiteLLM versions require an explicit provider prefix for Anthropic
        models. Keep app-level model names provider-neutral so pricing, metrics,
        task metadata, and env values do not need to carry transport details.
        """
        model_lower = model.lower()
        if "/" in model:
            return model
        if model_lower.startswith("claude-"):
            return f"anthropic/{model}"
        if model_lower.startswith("gemini-"):
            return f"gemini/{model}"
        return model

    def _pricing_key(self, model: str) -> str:
        """Normalize provider-prefixed model names for local pricing lookup."""
        if "/" in model:
            return model.split("/", 1)[1]
        return model

    @retry(
        retry=retry_if_exception(_is_retryable_llm_exception),
        stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
        before_sleep=safe_before_sleep_log(logger, "WARNING"),
    )
    async def _call_llm(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop_sequences: list[str] | None,
    ) -> tuple[str, TokenUsage, str | None]:
        """Call LLM via litellm with automatic provider routing and retry logic.

        Uses litellm.acompletion which automatically routes to the correct provider
        based on model name prefix (gpt-* → OpenAI, claude-* → Anthropic).

        Args:
            model: Model identifier (e.g., 'gpt-4', 'claude-3-sonnet-20240229').
            prompt: Input text.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            stop_sequences: Optional stop sequences.

        Returns:
            Tuple of (generated_text, token_usage, finish_reason).

        Raises:
            LLMRateLimitError: When rate limit is exceeded.
            LLMTimeoutError: When request times out.
            LLMProviderError: When provider is unavailable or authentication fails.
            LLMError: For other API errors.
        """
        # Special handling for local models
        is_local = model.lower().startswith("local-")
        if is_local:
            if not self._local_base_url:
                raise LLMProviderError(
                    "Local LLM not configured. Set LOCAL_LLM_URL and start the Ollama container"
                )
            # Strip local- prefix for local models
            actual_model = model.removeprefix("local-")
            # Use litellm with custom_llm_provider for local endpoint
            try:
                response = await acompletion(
                    model=f"openai/{actual_model}",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop_sequences,
                    api_base=self._local_base_url,
                    api_key=os.getenv("LOCAL_LLM_API_KEY", "ollama"),
                    timeout=DEFAULT_TIMEOUT,
                )
            except Exception as e:
                logger.error(f"Local LLM error via {self._local_base_url}: {e}")
                raise LLMProviderError(
                    f"Local LLM unreachable at {self._local_base_url}. Ensure the Ollama container is running."
                ) from e
        else:
            # Standard models (OpenAI/Anthropic) - litellm handles routing automatically
            try:
                # Build params dict - Anthropic models don't support both temperature and top_p
                provider = self._determine_provider(model)
                litellm_model = self._litellm_model_name(model)
                params = {
                    "model": litellm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": stop_sequences,
                    "timeout": DEFAULT_TIMEOUT,
                }

                # Only add top_p for non-Anthropic models (claude models reject both params)
                if provider != ModelProvider.ANTHROPIC:
                    params["top_p"] = top_p

                response = await acompletion(**params)
            except litellm.AuthenticationError as e:
                logger.error(f"Authentication failed for {model}: {e}")
                raise LLMProviderError(
                    f"Authentication failed for model {model}. Check provider API keys.",
                    error_code="LLM_AUTHENTICATION_FAILED",
                    retryable=False,
                    context={"model": model, "provider": provider.value},
                ) from e
            except litellm.RateLimitError as e:
                if _is_provider_request_error(e):
                    message = _provider_request_message(model, e)
                    logger.error(f"Provider request rejected for {model}: {e}")
                    raise LLMProviderError(
                        message,
                        error_code="LLM_PROVIDER_REQUEST_FAILED",
                        retryable=False,
                        context={"model": model, "provider": provider.value},
                    ) from e
                logger.warning(f"Rate limit hit for {model}: {e}")
                raise LLMRateLimitError(f"Rate limit: {e}") from e
            except (litellm.Timeout, APITimeoutError) as e:
                logger.warning(f"Timeout for {model}: {e}")
                raise LLMTimeoutError(f"Timeout: {e}") from e
            except RateLimitError as e:
                # OpenAI-compatible exception
                if _is_provider_request_error(e):
                    message = _provider_request_message(model, e)
                    logger.error(f"Provider request rejected for {model}: {e}")
                    raise LLMProviderError(
                        message,
                        error_code="LLM_PROVIDER_REQUEST_FAILED",
                        retryable=False,
                        context={"model": model, "provider": provider.value},
                    ) from e
                logger.warning(f"Rate limit hit: {e}")
                raise LLMRateLimitError(f"Rate limit: {e}") from e
            except litellm.APIConnectionError as e:
                if _is_provider_request_error(e):
                    message = _provider_request_message(model, e)
                    logger.error(f"Provider request rejected for {model}: {e}")
                    raise LLMProviderError(
                        message,
                        error_code="LLM_PROVIDER_REQUEST_FAILED",
                        retryable=False,
                        context={"model": model, "provider": provider.value},
                    ) from e
                logger.error(f"Cannot connect to LLM provider {model}: {e}")
                raise LLMConnectionError(f"Cannot reach LLM provider. Check network access.") from e
            except Exception as e:
                if _is_provider_request_error(e):
                    message = _provider_request_message(model, e)
                    logger.error(f"Provider request rejected for {model}: {e}")
                    raise LLMProviderError(
                        message,
                        error_code="LLM_PROVIDER_REQUEST_FAILED",
                        retryable=False,
                        context={"model": model, "provider": provider.value},
                    ) from e
                logger.error(f"LLM API error for {model}: {e}")
                raise LLMError(f"LLM error: {e}") from e

        # Extract response (litellm returns OpenAI-compatible format)
        try:
            content = response.choices[0].message.content or ""

            # Extract usage with fallbacks for local models
            usage_raw = getattr(response, "usage", None)
            if usage_raw:
                usage = TokenUsage(
                    prompt_tokens=getattr(usage_raw, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage_raw, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage_raw, "total_tokens", 0) or 0,
                )
            else:
                # Fallback for models without usage tracking
                usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

            finish_reason = response.choices[0].finish_reason

            return content, usage, finish_reason

        except (AttributeError, IndexError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise LLMError(f"Invalid response format: {e}") from e

    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
        _allow_fallback: bool = True,
    ) -> LLMResponse:
        """
        Generate text using the specified model.

        Routes to the appropriate provider via litellm based on model name prefix.
        Includes automatic retries, circuit breaker protection, and cost tracking.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet")
            prompt: Input text
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop_sequences: Optional stop sequences

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            LLMProviderError: If required provider is not configured
            LLMError: For other generation errors
        """
        start_time = time.perf_counter()

        # Determine provider from model name (for circuit breaker)
        provider = self._determine_provider(model)

        # Circuit breaker pre-flight check
        circuit_breaker = self._circuit_breakers.get(provider)
        if circuit_breaker and not await circuit_breaker.can_proceed():
            logger.warning(
                f"Circuit breaker OPEN for {provider.value} - request blocked | "
                f"status={circuit_breaker.get_status()}"
            )
            raise LLMProviderError(
                f"Provider {provider.value} is temporarily unavailable due to repeated failures. "
                f"Will retry automatically in {CIRCUIT_BREAKER_RECOVERY_TIME}s."
            )

        # Request coalescing: reuse in-flight requests for identical prompts
        request_hash = self._compute_request_hash(model, prompt, temperature, max_tokens)

        async with self._get_coalescing_lock():
            # Check if identical request is already in-flight
            if request_hash in self._in_flight_requests:
                logger.debug(f"Request coalescing: reusing in-flight request {request_hash[:8]}...")
                # Await the in-flight request and ensure cleanup on completion
                future = self._in_flight_requests[request_hash]

        # If we found an in-flight request, await it outside the lock
        if request_hash in self._in_flight_requests and 'future' in locals():
            try:
                result = await future
                return result
            except Exception as e:
                # If the shared request failed, remove it and let this one try again
                async with self._get_coalescing_lock():
                    self._in_flight_requests.pop(request_hash, None)
                logger.debug(f"In-flight request {request_hash[:8]} failed, retrying: {e}")

        # Check cache if available
        if self.cache_manager:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            cache_key = f"llm:{model}:{prompt_hash}:{temperature}:{max_tokens}"
            try:
                cached = await self.cache_manager.get(cache_key)
                if cached:
                    logger.debug(f"Cache hit for {model}")
                    # Ensure we return an object, not a dict
                    if isinstance(cached, dict):
                        return LLMResponse.from_dict(cached)
                    return cached
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        # Create and store task for request coalescing
        # This ensures concurrent identical requests share the same execution
        current_task = asyncio.current_task()
        async with self._get_coalescing_lock():
            if current_task:
                self._in_flight_requests[request_hash] = current_task

        # Call LLM via unified litellm interface
        try:
            content, usage, finish_reason = await self._call_llm(
                model, prompt, temperature, max_tokens, top_p, stop_sequences
            )

            # Record circuit breaker success
            if circuit_breaker:
                await circuit_breaker.record_success()

            # Calculate cost (local models are free)
            if provider == ModelProvider.LOCAL:
                cost = 0.0
            else:
                pricing = DEFAULT_MODEL_PRICING.get(self._pricing_key(model))
                if not pricing:
                    logger.warning(f"No pricing data for {model}, using default")
                    pricing = ModelPricing(input_cost_per_1k=0.001, output_cost_per_1k=0.002)
                cost = pricing.calculate_cost(usage)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Build response
            response = LLMResponse(
                content=content,
                model=model,
                usage=usage,
                cost=cost,
                latency_ms=latency_ms,
                provider=provider,
                finish_reason=finish_reason,
            )

            # Cache response if available
            if self.cache_manager:
                try:
                    # Use to_dict for JSON serialization
                    await self.cache_manager.set(cache_key, response.to_dict(), ttl=3600)
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")

            # Record metrics if available
            if self.metrics:
                try:
                    self.metrics.record_llm_api_call(
                        model=model,
                        provider=provider.value,
                        status="success",
                        tokens_used=usage.total_tokens,
                        cost=cost,
                        latency_seconds=latency_ms / 1000.0,
                    )
                except Exception as e:
                    logger.warning(f"Metrics recording error: {e}")

            logger.info(
                f"✓ {model} generated {usage.completion_tokens} tokens "
                f"in {latency_ms:.0f}ms (${cost:.4f})"
            )

            return response

        except (LLMProviderError, LLMRateLimitError, LLMTimeoutError, LLMConnectionError, LLMError) as e:
            # Record circuit breaker failure for provider errors
            if circuit_breaker:
                await circuit_breaker.record_failure()
            # Retryability controls repeated calls to the same provider. A
            # configured fallback is a separate recovery path and is useful
            # for terminal provider/account failures such as exhausted quota,
            # invalid credentials, or a model that is unavailable.
            if _allow_fallback:
                fallback_response = await self._try_fallbacks(
                    primary_model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                    primary_error=e,
                )
                if fallback_response:
                    logger.warning(
                        f"LLM fallback succeeded | primary={model} | "
                        f"used={fallback_response.model}"
                    )
                    return fallback_response
            raise
        except Exception as e:
            # Record circuit breaker failure for unexpected errors
            if circuit_breaker:
                await circuit_breaker.record_failure()
            logger.error(f"Generation failed for {model}: {e}")
            raise
        finally:
            # CRITICAL: Clean up in-flight request to prevent memory leak
            async with self._get_coalescing_lock():
                self._in_flight_requests.pop(request_hash, None)

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Backward-compatible alias for generate().

        Maintained for compatibility with legacy code. New code should use generate().

        Args:
            prompt: Input text.
            model: Model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Complete LLM response with metadata.
        """
        return await self.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


# ============================================================================
# FACTORY FUNCTION (for backward compatibility)
# ============================================================================


def get_llm_client(
    cache_manager: Optional[object] = None,
    metrics_collector: Optional[object] = None,
    **kwargs,
) -> UnifiedLLMClient:
    """Factory function for UnifiedLLMClient instantiation.

    Provides backward compatibility with legacy dependency injection container.

    Args:
        cache_manager: Optional cache for response caching.
        metrics_collector: Optional metrics tracking.
        **kwargs: Additional parameters (ignored).

    Returns:
        Configured UnifiedLLMClient instance.
    """
    return UnifiedLLMClient(
        cache_manager=cache_manager,
        metrics_collector=metrics_collector,
    )


# Type alias for backward compatibility
AbstractLLMClient = UnifiedLLMClient
LLMClient = UnifiedLLMClient  # Legacy alias for tests
