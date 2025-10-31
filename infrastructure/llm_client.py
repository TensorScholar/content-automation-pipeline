"""
Unified client for multiple LLM providers, implementing circuit breaker logic, adaptive retries, and cost tracking.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import anthropic
import httpx
from anthropic import AnthropicError, AsyncAnthropic
from loguru import logger
from openai import APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError
from pydantic import BaseModel, Field, root_validator, validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.exceptions import LLMError, LLMProviderError, LLMRateLimitError, LLMTimeoutError

# ============================================================================
# TYPE SYSTEM: Algebraic Data Types for LLM Operations
# ============================================================================


class ModelProvider(str, Enum):
    """Enumeration of supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTier(str, Enum):
    """Model capability tiers for intelligent routing."""

    NANO = "nano"  # < 1B parameters (local models)
    SMALL = "small"  # 1-10B parameters (GPT-3.5 class)
    MEDIUM = "medium"  # 10-100B parameters (GPT-4 class)
    LARGE = "large"  # > 100B parameters (GPT-4 Turbo class)


@dataclass(frozen=True)
class TokenUsage:
    """Immutable token usage record with cost calculation."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self):
        """Validate token counts satisfy invariants."""
        assert self.prompt_tokens >= 0, "Prompt tokens must be non-negative"
        assert self.completion_tokens >= 0, "Completion tokens must be non-negative"
        assert (
            self.total_tokens == self.prompt_tokens + self.completion_tokens
        ), "Total tokens must equal sum of prompt and completion"


@dataclass(frozen=True)
class ModelPricing:
    """Token pricing configuration with atomic cost calculation."""

    input_cost_per_1k: float  # USD per 1000 input tokens
    output_cost_per_1k: float  # USD per 1000 output tokens

    def calculate_cost(self, usage: TokenUsage) -> float:
        """
        Calculate request cost with floating-point precision.

        Uses Decimal internally for exact monetary arithmetic, then converts
        to float for compatibility with downstream systems.
        """
        from decimal import ROUND_HALF_UP, Decimal

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


# Default pricing database (fallback if no settings provided)
DEFAULT_MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4": ModelPricing(input_cost_per_1k=0.03, output_cost_per_1k=0.06),
    "gpt-4-turbo": ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.03),
    "gpt-3.5-turbo": ModelPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    "claude-3-opus": ModelPricing(input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    "claude-3-sonnet": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "claude-sonnet-4-20250514": ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
}


@dataclass(frozen=True)
class LLMResponse:
    """Immutable LLM response with comprehensive metadata."""

    content: str
    model: str
    usage: TokenUsage
    cost: float
    latency_ms: float
    provider: ModelProvider
    timestamp: datetime = field(default_factory=datetime.utcnow)
    finish_reason: Optional[str] = None

    def __post_init__(self):
        """Validate response invariants."""
        assert len(self.content) > 0, "Response content cannot be empty"
        assert self.cost >= 0, "Cost must be non-negative"
        assert self.latency_ms >= 0, "Latency must be non-negative"


class LLMRequest(BaseModel):
    """Validated LLM request with constraint enforcement."""

    prompt: str = Field(..., min_length=1, max_length=100000)
    model: str = Field(..., pattern=r"^(gpt-|claude-)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(..., ge=1, le=32000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop_sequences: Optional[List[str]] = Field(default=None, max_items=4)

    @validator("max_tokens")
    def validate_max_tokens(cls, v, values):
        """Ensure max_tokens doesn't exceed model context window."""
        model = values.get("model", "")
        if "gpt-4" in model and v > 8192:
            raise ValueError(f"GPT-4 max tokens cannot exceed 8192, got {v}")
        elif "gpt-3.5" in model and v > 4096:
            raise ValueError(f"GPT-3.5 max tokens cannot exceed 4096, got {v}")
        return v

    class Config:
        frozen = True  # Immutable after creation


# ============================================================================
# CIRCUIT BREAKER: Fault Isolation Pattern
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states implementing finite state machine."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure threshold exceeded
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker implementing fault isolation for external API calls.

    State Transitions:
    CLOSED → OPEN: After failure_threshold consecutive failures
    OPEN → HALF_OPEN: After recovery_timeout duration
    HALF_OPEN → CLOSED: After success_threshold consecutive successes
    HALF_OPEN → OPEN: On any failure

    Theoretical Foundation: Finite State Machine with probabilistic transitions
    Now uses Redis for distributed state management across multiple processes.
    """

    def __init__(
        self,
        redis_client: Any,
        key_prefix: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        # Redis keys for state management
        self.state_key = f"{key_prefix}:state"
        self.failure_count_key = f"{key_prefix}:failure_count"
        self.success_count_key = f"{key_prefix}:success_count"
        self.last_failure_time_key = f"{key_prefix}:last_failure_time"

    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function through circuit breaker with fault isolation.

        Uses Redis for distributed state management with atomic operations.

        Raises:
            LLMProviderError: If circuit is open and recovery timeout not elapsed
        """
        # Get current state from Redis
        current_state = await self._get_state()

        # State: OPEN - Check if recovery timeout elapsed
        if current_state == CircuitState.OPEN:
            last_failure_time = await self._get_last_failure_time()
            if (
                last_failure_time
                and (datetime.utcnow() - last_failure_time).total_seconds() > self.recovery_timeout
            ):
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                await self._set_state(CircuitState.HALF_OPEN)
                await self._reset_success_count()
            else:
                raise LLMProviderError("Circuit breaker is OPEN - service unavailable")

        try:
            result = await func(*args, **kwargs)

            # Handle success
            current_state = await self._get_state()
            if current_state == CircuitState.HALF_OPEN:
                success_count = await self._increment_success_count()
                if success_count >= self.success_threshold:
                    logger.info("Circuit breaker transitioning to CLOSED")
                    await self._set_state(CircuitState.CLOSED)
                    await self._reset_failure_count()
            elif current_state == CircuitState.CLOSED:
                await self._reset_failure_count()  # Reset on success

            return result

        except Exception as e:
            # Handle failure
            await self._increment_failure_count()
            await self._set_last_failure_time(datetime.utcnow())

            current_state = await self._get_state()
            if current_state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker transitioning to OPEN (failure in HALF_OPEN)")
                await self._set_state(CircuitState.OPEN, ttl=int(self.recovery_timeout))
                await self._reset_success_count()
            else:
                failure_count = await self._get_failure_count()
                if failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker transitioning to OPEN (failures: {failure_count})"
                    )
                    await self._set_state(CircuitState.OPEN, ttl=int(self.recovery_timeout))

            raise

    # =========================================================================
    # REDIS STATE MANAGEMENT HELPERS
    # =========================================================================

    async def _get_state(self) -> CircuitState:
        """Get current circuit breaker state from Redis."""
        try:
            state_value = await self.redis_client.get(self.state_key)
            if state_value is None:
                return CircuitState.CLOSED  # Default state
            return CircuitState(state_value)
        except Exception as e:
            logger.warning(f"Failed to get circuit breaker state: {e}")
            return CircuitState.CLOSED  # Fail-safe default

    async def _set_state(self, state: CircuitState, ttl: Optional[int] = None) -> None:
        """Set circuit breaker state in Redis with optional TTL."""
        try:
            if ttl:
                await self.redis_client.set(self.state_key, state.value, ttl=ttl)
            else:
                await self.redis_client.set(self.state_key, state.value)
        except Exception as e:
            logger.error(f"Failed to set circuit breaker state: {e}")

    async def _get_failure_count(self) -> int:
        """Get current failure count from Redis."""
        try:
            count = await self.redis_client.get(self.failure_count_key)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.warning(f"Failed to get failure count: {e}")
            return 0

    async def _increment_failure_count(self) -> int:
        """Atomically increment failure count in Redis."""
        try:
            return await self.redis_client.increment(self.failure_count_key)
        except Exception as e:
            logger.error(f"Failed to increment failure count: {e}")
            return 0

    async def _reset_failure_count(self) -> None:
        """Reset failure count in Redis."""
        try:
            await self.redis_client.delete(self.failure_count_key)
        except Exception as e:
            logger.warning(f"Failed to reset failure count: {e}")

    async def _get_success_count(self) -> int:
        """Get current success count from Redis."""
        try:
            count = await self.redis_client.get(self.success_count_key)
            return int(count) if count is not None else 0
        except Exception as e:
            logger.warning(f"Failed to get success count: {e}")
            return 0

    async def _increment_success_count(self) -> int:
        """Atomically increment success count in Redis."""
        try:
            return await self.redis_client.increment(self.success_count_key)
        except Exception as e:
            logger.error(f"Failed to increment success count: {e}")
            return 0

    async def _reset_success_count(self) -> None:
        """Reset success count in Redis."""
        try:
            await self.redis_client.delete(self.success_count_key)
        except Exception as e:
            logger.warning(f"Failed to reset success count: {e}")

    async def _get_last_failure_time(self) -> Optional[datetime]:
        """Get last failure time from Redis."""
        try:
            timestamp = await self.redis_client.get(self.last_failure_time_key)
            if timestamp is None:
                return None
            return datetime.fromisoformat(timestamp)
        except Exception as e:
            logger.warning(f"Failed to get last failure time: {e}")
            return None

    async def _set_last_failure_time(self, failure_time: datetime) -> None:
        """Set last failure time in Redis."""
        try:
            await self.redis_client.set(self.last_failure_time_key, failure_time.isoformat())
        except Exception as e:
            logger.error(f"Failed to set last failure time: {e}")


# ============================================================================
# ABSTRACT BASE CLASS: LLM Provider Interface
# ============================================================================


class AbstractLLMClient(ABC):
    """
    Abstract base class defining the common interface for all LLM providers.

    This class establishes a contract that all concrete LLM client implementations
    must follow, enabling polymorphic usage and dynamic provider selection through
    the Factory pattern.

    Design Philosophy:
    - Interface segregation: Only essential methods in the abstract interface
    - Dependency inversion: Clients depend on abstraction, not concrete implementations
    - Open/Closed principle: Open for extension (new providers), closed for modification
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion with automatic provider routing and fault tolerance.

        This is the primary interface method that all LLM clients must implement.

        Args:
            prompt: Input prompt
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            temperature: Sampling temperature [0.0, 2.0]
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            LLMResponse with content, usage, and cost

        Raises:
            LLMError: On unrecoverable errors
            LLMTimeoutError: On timeout
            LLMRateLimitError: On rate limit (after retries)
        """
        pass

    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Retrieve aggregated metrics.

        Returns:
            Dictionary with request count, token usage, and total cost
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, str]:
        """
        Verify connectivity to LLM providers.

        Returns:
            Health status for each configured provider
        """
        pass


# ============================================================================
# LLM CLIENT: Unified Provider Abstraction
# ============================================================================


class LLMClient(AbstractLLMClient):
    """
    Production-grade LLM client with fault tolerance and observability.

    Design Principles:
    1. Provider abstraction through functorial composition
    2. Fail-safe with circuit breaker and retry strategies
    3. Observable with distributed tracing and metrics
    4. Economical with precise cost tracking
    5. Type-safe with Pydantic validation

    Usage:
        client = LLMClient()
        response = await client.complete(
            prompt="Explain quantum computing",
            model="gpt-4",
            max_tokens=500
        )
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
        settings: Optional[Any] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize LLM client with provider credentials and caching.

        Args:
            redis_client: Redis client for distributed circuit breaker state
            cache_manager: Cache manager for LLM response caching
            openai_api_key: OpenAI API key (reads from env if None)
            anthropic_api_key: Anthropic API key (reads from env if None)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient failures
        """
        import os

        # Initialize providers
        self.openai_client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            timeout=httpx.Timeout(timeout),
            max_retries=0,  # We handle retries ourselves
        )

        self.anthropic_client = (
            AsyncAnthropic(
                api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
                timeout=httpx.Timeout(timeout),
                max_retries=0,
            )
            if anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            else None
        )

        # Initialize caching and metrics
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector

        # Initialize settings and pricing
        self.settings = settings
        self.model_pricing = self._initialize_model_pricing()

        # Circuit breakers per provider with Redis state management
        self.circuit_breakers: Dict[ModelProvider, CircuitBreaker] = {
            ModelProvider.OPENAI: CircuitBreaker(
                redis_client=redis_client,
                key_prefix="breaker:openai",
                failure_threshold=5,
                recovery_timeout=60.0,
            ),
            ModelProvider.ANTHROPIC: CircuitBreaker(
                redis_client=redis_client,
                key_prefix="breaker:anthropic",
                failure_threshold=5,
                recovery_timeout=60.0,
            ),
        }

        self.timeout = timeout
        self.max_retries = max_retries

        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._metrics_lock = asyncio.Lock()

        logger.info(
            f"LLMClient initialized | timeout={timeout}s | max_retries={max_retries} | redis_enabled={redis_client is not None} | cache_enabled={cache_manager is not None}"
        )

    def _initialize_model_pricing(self) -> Dict[str, ModelPricing]:
        """Initialize model pricing from settings or use defaults."""
        if (
            self.settings
            and hasattr(self.settings, "llm")
            and hasattr(self.settings.llm, "model_pricing")
        ):
            pricing_data = self.settings.llm.model_pricing
            return {
                model: ModelPricing(
                    input_cost_per_1k=pricing["input"], output_cost_per_1k=pricing["output"]
                )
                for model, pricing in pricing_data.items()
            }
        else:
            # Fallback to default pricing if no settings provided
            return DEFAULT_MODEL_PRICING

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion with automatic provider routing and fault tolerance.

        Args:
            prompt: Input prompt
            model: Model identifier (e.g., "gpt-4", "claude-3-opus")
            temperature: Sampling temperature [0.0, 2.0]
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            LLMResponse with content, usage, and cost

        Raises:
            LLMError: On unrecoverable errors
            LLMTimeoutError: On timeout
            LLMRateLimitError: On rate limit (after retries)
        """
        # Validate request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Enforce daily token budget (fail-fast before external calls)
        try:
            if self.settings and hasattr(self.settings, "llm") and self.redis_client:
                from datetime import datetime as _dt

                day_key = _dt.utcnow().strftime("%Y%m%d")
                usage_key = f"llm_usage:{day_key}"
                usage = await self.redis_client.get(usage_key) or {}
                tokens_used = int(usage.get("tokens", 0))
                daily_budget = int(getattr(self.settings.llm, "daily_token_budget", 0))
                if daily_budget and tokens_used >= daily_budget:
                    raise LLMRateLimitError(
                        f"Daily token budget exceeded ({tokens_used}/{daily_budget})"
                    )
        except Exception:
            # Fail-open on accounting errors
            pass

        # Check cache first if cache manager is available
        if self.cache_manager:
            cached_response = await self._get_cached_response(request)
            if cached_response:
                logger.info(
                    f"Cache hit for LLM request | model={model} | prompt_hash={self._compute_prompt_hash(request)}"
                )
                return cached_response

        # Determine provider
        provider = self._get_provider(model)

        # Execute through circuit breaker
        circuit_breaker = self.circuit_breakers[provider]

        try:
            response = await circuit_breaker.call(
                self._execute_with_retry,
                request,
                provider,
            )

            # Cache the response if cache manager is available
            if self.cache_manager:
                await self._cache_response(request, response)

            # Update metrics atomically
            async with self._metrics_lock:
                self.total_requests += 1
                self.total_tokens += response.usage.total_tokens
                self.total_cost += response.cost

            # Record Prometheus metrics if available
            if self.metrics_collector:
                self.metrics_collector.record_llm_api_call(
                    model=model,
                    provider=provider.value,
                    status="success",
                    tokens_used=response.usage.total_tokens,
                    cost=response.cost,
                    latency_seconds=response.latency_ms / 1000.0,
                )

            return response

        except Exception as e:
            logger.error(f"LLM completion failed | model={model} | error={e}")

            # Record failure metrics if available
            if self.metrics_collector:
                error_type = "unknown"
                if isinstance(e, (APITimeoutError, httpx.TimeoutException)):
                    error_type = "timeout"
                elif isinstance(e, RateLimitError):
                    error_type = "rate_limit"
                elif isinstance(e, (OpenAIError, AnthropicError)):
                    error_type = "api_error"

                self.metrics_collector.record_llm_api_call(
                    model=model,
                    provider=provider.value,
                    status="failure",
                    tokens_used=0,
                    cost=0.0,
                    latency_seconds=0.0,
                )

            raise

        finally:
            # Update daily usage accounting on success
            try:
                if self.settings and hasattr(self.settings, "llm") and self.redis_client:
                    from datetime import datetime as _dt

                    day_key = _dt.utcnow().strftime("%Y%m%d")
                    usage_key = f"llm_usage:{day_key}"
                    # Read existing
                    usage = await self.redis_client.get(usage_key) or {"tokens": 0, "cost": 0.0}
                    # response may not be defined on exception
                    if 'response' in locals() and response:
                        usage["tokens"] = int(usage.get("tokens", 0)) + int(response.usage.total_tokens)
                        usage["cost"] = float(usage.get("cost", 0.0)) + float(response.cost)
                        # Persist with 2-day TTL
                        await self.redis_client.set(usage_key, usage, ttl=172800)
                        # Alert threshold
                        threshold = float(getattr(self.settings.llm, "cost_alert_threshold", 0.0) or 0.0)
                        if threshold and usage["cost"] >= threshold:
                            logger.warning(
                                f"LLM daily cost crossed alert threshold: ${usage['cost']:.2f} >= ${threshold:.2f}"
                            )
            except Exception:
                # Never fail due to accounting
                pass

    async def _execute_with_retry(
        self,
        request: LLMRequest,
        provider: ModelProvider,
    ) -> LLMResponse:
        """
        Execute LLM request with adaptive retry strategy.

        Retry Strategy:
        - Exponential backoff: 2^attempt seconds with jitter
        - Max attempts: 3 (configurable)
        - Retry conditions: Timeout, rate limit, transient errors
        - No retry: Authentication, validation errors
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(
                (APITimeoutError, RateLimitError, httpx.TimeoutException)
            ),
            before_sleep=before_sleep_log(logger, "WARNING"),
        )
        async def _execute():
            start_time = time.perf_counter()

            if provider == ModelProvider.OPENAI:
                result = await self._call_openai(request)
            elif provider == ModelProvider.ANTHROPIC:
                result = await self._call_anthropic(request)
            else:
                raise LLMProviderError(f"Unsupported provider: {provider}")

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Calculate cost
            pricing = self.model_pricing.get(request.model)
            if not pricing:
                logger.warning(f"No pricing data for model: {request.model}")
                cost = 0.0
            else:
                cost = pricing.calculate_cost(result[1])  # result[1] is TokenUsage

            return LLMResponse(
                content=result[0],
                model=request.model,
                usage=result[1],
                cost=cost,
                latency_ms=latency_ms,
                provider=provider,
                finish_reason=result[2],
            )

        try:
            return await _execute()
        except RateLimitError as e:
            raise LLMRateLimitError(f"Rate limit exceeded: {str(e)}") from e
        except APITimeoutError as e:
            raise LLMTimeoutError(f"Request timeout: {str(e)}") from e
        except (OpenAIError, AnthropicError) as e:
            raise LLMProviderError(f"Provider error: {str(e)}") from e

    async def _call_openai(self, request: LLMRequest) -> tuple:
        """Call OpenAI API with request mapping."""
        response = await self.openai_client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop_sequences,
        )

        content = response.choices[0].message.content
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        finish_reason = response.choices[0].finish_reason

        return content, usage, finish_reason

    async def _call_anthropic(self, request: LLMRequest) -> tuple:
        """Call Anthropic API with request mapping."""
        if not self.anthropic_client:
            raise LLMProviderError("Anthropic API key not configured")

        response = await self.anthropic_client.messages.create(
            model=request.model,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop_sequences=request.stop_sequences,
        )

        content = response.content[0].text
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        finish_reason = response.stop_reason

        return content, usage, finish_reason

    def _get_provider(self, model: str) -> ModelProvider:
        """Determine provider from model identifier."""
        if model.startswith("gpt-"):
            return ModelProvider.OPENAI
        elif model.startswith("claude-"):
            return ModelProvider.ANTHROPIC
        else:
            raise ValueError(f"Cannot determine provider for model: {model}")

    @staticmethod
    def _compute_prompt_hash(request: LLMRequest) -> str:
        """Compute hash for caching based on request parameters."""
        import hashlib

        content = f"{request.prompt}|{request.model}|{request.temperature:.3f}|{request.max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def _get_cached_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Retrieve cached LLM response if available."""
        try:
            prompt_hash = self._compute_prompt_hash(request)
            cached_data = await self.cache_manager.get(f"llm_cache:{prompt_hash}")

            if cached_data:
                # Reconstruct LLMResponse from cached data
                return LLMResponse(
                    content=cached_data["response"],
                    model=request.model,
                    usage=TokenUsage(
                        prompt_tokens=cached_data["prompt_tokens"],
                        completion_tokens=cached_data["completion_tokens"],
                        total_tokens=cached_data["total_tokens"],
                    ),
                    cost=cached_data["cost"],
                    latency_ms=cached_data["latency_ms"],
                    provider=ModelProvider(cached_data["provider"]),
                    timestamp=cached_data["timestamp"],
                    finish_reason=cached_data.get("finish_reason"),
                )
        except Exception as e:
            logger.warning(f"Failed to retrieve cached response: {e}")

        return None

    async def _cache_response(self, request: LLMRequest, response: LLMResponse) -> None:
        """Cache LLM response for future use."""
        try:
            prompt_hash = self._compute_prompt_hash(request)
            cache_data = {
                "response": response.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": response.cost,
                "latency_ms": response.latency_ms,
                "provider": response.provider.value,
                "timestamp": response.timestamp,
                "finish_reason": response.finish_reason,
            }

            # Cache with 30-day TTL (from RedisSettings.llm_response_cache_ttl)
            await self.cache_manager.set(f"llm_cache:{prompt_hash}", cache_data)
            logger.debug(
                f"Cached LLM response | hash={prompt_hash[:16]}... | tokens={response.usage.total_tokens}"
            )

        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Retrieve aggregated metrics.

        Returns:
            Dictionary with request count, token usage, and total cost
        """
        async with self._metrics_lock:
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "avg_tokens_per_request": self.total_tokens / max(self.total_requests, 1),
                "avg_cost_per_request": self.total_cost / max(self.total_requests, 1),
            }

    async def health_check(self) -> Dict[str, str]:
        """
        Verify connectivity to LLM providers.

        Returns:
            Health status for each configured provider
        """
        health = {}

        # Check OpenAI
        try:
            await self.openai_client.models.list()
            health["openai"] = "healthy"
        except Exception as e:
            health["openai"] = f"unhealthy: {str(e)}"

        # Check Anthropic
        if self.anthropic_client:
            try:
                # Anthropic doesn't have a list models endpoint, so we use a minimal request
                model = (
                    self.settings.llm.primary_model if self.settings else "claude-3-sonnet-20240229"
                )
                await self.anthropic_client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                )
                health["anthropic"] = "healthy"
            except Exception as e:
                health["anthropic"] = f"unhealthy: {str(e)}"
        else:
            health["anthropic"] = "not_configured"

        return health

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.openai_client.close()
        if self.anthropic_client:
            await self.anthropic_client.close()


# ============================================================================
# CONCRETE PROVIDER IMPLEMENTATIONS
# ============================================================================


class OpenAIClient(AbstractLLMClient):
    """
    OpenAI-specific LLM client implementation.

    Implements the AbstractLLMClient interface for OpenAI models (GPT-3.5, GPT-4, etc.).
    Includes fault tolerance, caching, and comprehensive metrics collection.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
        settings: Optional[Any] = None,
        openai_api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI client.

        Args:
            redis_client: Redis client for distributed circuit breaker state
            cache_manager: Cache manager for LLM response caching
            metrics_collector: Metrics collector for monitoring
            settings: Application settings
            openai_api_key: OpenAI API key (reads from env if None)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient failures
        """
        import os

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            timeout=httpx.Timeout(timeout),
            max_retries=0,  # We handle retries ourselves
        )

        # Initialize caching and metrics
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        self.settings = settings
        self.model_pricing = self._initialize_model_pricing()

        # Circuit breaker with Redis state management
        self.circuit_breaker = CircuitBreaker(
            redis_client=redis_client,
            key_prefix="breaker:openai",
            failure_threshold=5,
            recovery_timeout=60.0,
        )

        self.timeout = timeout
        self.max_retries = max_retries

        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._metrics_lock = asyncio.Lock()

        logger.info(f"OpenAIClient initialized | timeout={timeout}s | max_retries={max_retries}")

    def _initialize_model_pricing(self) -> Dict[str, ModelPricing]:
        """Initialize model pricing from settings or use defaults."""
        if (
            self.settings
            and hasattr(self.settings, "llm")
            and hasattr(self.settings.llm, "model_pricing")
        ):
            pricing_data = self.settings.llm.model_pricing
            return {
                model: ModelPricing(
                    input_cost_per_1k=pricing["input"], output_cost_per_1k=pricing["output"]
                )
                for model, pricing in pricing_data.items()
                if model.startswith("gpt-")
            }
        else:
            # Fallback to default pricing for OpenAI models only
            return {k: v for k, v in DEFAULT_MODEL_PRICING.items() if k.startswith("gpt-")}

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        # Validate request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Check cache first if cache manager is available
        if self.cache_manager:
            cached_response = await self._get_cached_response(request)
            if cached_response:
                logger.info(f"Cache hit for OpenAI request | model={model}")
                return cached_response

        # Execute through circuit breaker
        try:
            response = await self.circuit_breaker.call(
                self._execute_with_retry,
                request,
            )

            # Cache the response if cache manager is available
            if self.cache_manager:
                await self._cache_response(request, response)

            # Update metrics atomically
            async with self._metrics_lock:
                self.total_requests += 1
                self.total_tokens += response.usage.total_tokens
                self.total_cost += response.cost

            # Record Prometheus metrics if available
            if self.metrics_collector:
                self.metrics_collector.record_llm_api_call(
                    model=model,
                    provider="openai",
                    status="success",
                    tokens_used=response.usage.total_tokens,
                    cost=response.cost,
                    latency_seconds=response.latency_ms / 1000.0,
                )

            return response

        except Exception as e:
            logger.error(f"OpenAI completion failed | model={model} | error={e}")

            # Record failure metrics if available
            if self.metrics_collector:
                error_type = "unknown"
                if isinstance(e, (APITimeoutError, httpx.TimeoutException)):
                    error_type = "timeout"
                elif isinstance(e, RateLimitError):
                    error_type = "rate_limit"
                elif isinstance(e, OpenAIError):
                    error_type = "api_error"

                self.metrics_collector.record_llm_api_call(
                    model=model,
                    provider="openai",
                    status="failure",
                    tokens_used=0,
                    cost=0.0,
                    latency_seconds=0.0,
                )

            raise

    async def _execute_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Execute OpenAI request with adaptive retry strategy."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(
                (APITimeoutError, RateLimitError, httpx.TimeoutException)
            ),
            before_sleep=before_sleep_log(logger, "WARNING"),
        )
        async def _execute():
            start_time = time.perf_counter()

            response = await self.openai_client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop_sequences,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            content = response.choices[0].message.content
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
            finish_reason = response.choices[0].finish_reason

            # Calculate cost
            pricing = self.model_pricing.get(request.model)
            cost = pricing.calculate_cost(usage) if pricing else 0.0

            return LLMResponse(
                content=content,
                model=request.model,
                usage=usage,
                cost=cost,
                latency_ms=latency_ms,
                provider=ModelProvider.OPENAI,
                finish_reason=finish_reason,
            )

        try:
            return await _execute()
        except RateLimitError as e:
            raise LLMRateLimitError(f"Rate limit exceeded: {str(e)}") from e
        except APITimeoutError as e:
            raise LLMTimeoutError(f"Request timeout: {str(e)}") from e
        except OpenAIError as e:
            raise LLMProviderError(f"OpenAI API error: {str(e)}") from e

    @staticmethod
    def _compute_prompt_hash(request: LLMRequest) -> str:
        """Compute hash for caching based on request parameters."""
        import hashlib

        content = f"{request.prompt}|{request.model}|{request.temperature:.3f}|{request.max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def _get_cached_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Retrieve cached LLM response if available."""
        try:
            prompt_hash = self._compute_prompt_hash(request)
            cached_data = await self.cache_manager.get(f"llm_cache:{prompt_hash}")

            if cached_data:
                return LLMResponse(
                    content=cached_data["response"],
                    model=request.model,
                    usage=TokenUsage(
                        prompt_tokens=cached_data["prompt_tokens"],
                        completion_tokens=cached_data["completion_tokens"],
                        total_tokens=cached_data["total_tokens"],
                    ),
                    cost=cached_data["cost"],
                    latency_ms=cached_data["latency_ms"],
                    provider=ModelProvider(cached_data["provider"]),
                    timestamp=cached_data["timestamp"],
                    finish_reason=cached_data.get("finish_reason"),
                )
        except Exception as e:
            logger.warning(f"Failed to retrieve cached response: {e}")

        return None

    async def _cache_response(self, request: LLMRequest, response: LLMResponse) -> None:
        """Cache LLM response for future use."""
        try:
            prompt_hash = self._compute_prompt_hash(request)
            cache_data = {
                "response": response.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": response.cost,
                "latency_ms": response.latency_ms,
                "provider": response.provider.value,
                "timestamp": response.timestamp,
                "finish_reason": response.finish_reason,
            }

            await self.cache_manager.set(f"llm_cache:{prompt_hash}", cache_data)
            logger.debug(f"Cached OpenAI response | hash={prompt_hash[:16]}...")

        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Retrieve aggregated metrics."""
        async with self._metrics_lock:
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "avg_tokens_per_request": self.total_tokens / max(self.total_requests, 1),
                "avg_cost_per_request": self.total_cost / max(self.total_requests, 1),
                "provider": "openai",
            }

    async def health_check(self) -> Dict[str, str]:
        """Verify connectivity to OpenAI API."""
        try:
            await self.openai_client.models.list()
            return {"openai": "healthy"}
        except Exception as e:
            return {"openai": f"unhealthy: {str(e)}"}

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.openai_client.close()


class AnthropicClient(AbstractLLMClient):
    """
    Anthropic-specific LLM client implementation.

    Implements the AbstractLLMClient interface for Anthropic models (Claude).
    Includes fault tolerance, caching, and comprehensive metrics collection.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
        settings: Optional[Any] = None,
        anthropic_api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize Anthropic client.

        Args:
            redis_client: Redis client for distributed circuit breaker state
            cache_manager: Cache manager for LLM response caching
            metrics_collector: Metrics collector for monitoring
            settings: Application settings
            anthropic_api_key: Anthropic API key (reads from env if None)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient failures
        """
        import os

        # Initialize Anthropic client
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError("Anthropic API key not configured")

        self.anthropic_client = AsyncAnthropic(
            api_key=api_key,
            timeout=httpx.Timeout(timeout),
            max_retries=0,  # We handle retries ourselves
        )

        # Initialize caching and metrics
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        self.settings = settings
        self.model_pricing = self._initialize_model_pricing()

        # Circuit breaker with Redis state management
        self.circuit_breaker = CircuitBreaker(
            redis_client=redis_client,
            key_prefix="breaker:anthropic",
            failure_threshold=5,
            recovery_timeout=60.0,
        )

        self.timeout = timeout
        self.max_retries = max_retries

        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._metrics_lock = asyncio.Lock()

        logger.info(f"AnthropicClient initialized | timeout={timeout}s | max_retries={max_retries}")

    def _initialize_model_pricing(self) -> Dict[str, ModelPricing]:
        """Initialize model pricing from settings or use defaults."""
        if (
            self.settings
            and hasattr(self.settings, "llm")
            and hasattr(self.settings.llm, "model_pricing")
        ):
            pricing_data = self.settings.llm.model_pricing
            return {
                model: ModelPricing(
                    input_cost_per_1k=pricing["input"], output_cost_per_1k=pricing["output"]
                )
                for model, pricing in pricing_data.items()
                if model.startswith("claude-")
            }
        else:
            # Fallback to default pricing for Anthropic models only
            return {k: v for k, v in DEFAULT_MODEL_PRICING.items() if k.startswith("claude-")}

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        # Validate request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Check cache first if cache manager is available
        if self.cache_manager:
            cached_response = await self._get_cached_response(request)
            if cached_response:
                logger.info(f"Cache hit for Anthropic request | model={model}")
                return cached_response

        # Execute through circuit breaker
        try:
            response = await self.circuit_breaker.call(
                self._execute_with_retry,
                request,
            )

            # Cache the response if cache manager is available
            if self.cache_manager:
                await self._cache_response(request, response)

            # Update metrics atomically
            async with self._metrics_lock:
                self.total_requests += 1
                self.total_tokens += response.usage.total_tokens
                self.total_cost += response.cost

            # Record Prometheus metrics if available
            if self.metrics_collector:
                self.metrics_collector.record_llm_api_call(
                    model=model,
                    provider="anthropic",
                    status="success",
                    tokens_used=response.usage.total_tokens,
                    cost=response.cost,
                    latency_seconds=response.latency_ms / 1000.0,
                )

            return response

        except Exception as e:
            logger.error(f"Anthropic completion failed | model={model} | error={e}")

            # Record failure metrics if available
            if self.metrics_collector:
                self.metrics_collector.record_llm_api_call(
                    model=model,
                    provider="anthropic",
                    status="failure",
                    tokens_used=0,
                    cost=0.0,
                    latency_seconds=0.0,
                )

            raise

    async def _execute_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Execute Anthropic request with adaptive retry strategy."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((httpx.TimeoutException, AnthropicError)),
            before_sleep=before_sleep_log(logger, "WARNING"),
        )
        async def _execute():
            start_time = time.perf_counter()

            # Try Messages API first (for newer models), fallback to Completions API
            try:
                if hasattr(self.anthropic_client, "messages"):
                    # Use Messages API for newer models
                    response = await self.anthropic_client.messages.create(
                        model=request.model,
                        messages=[{"role": "user", "content": request.prompt}],
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        stop_sequences=request.stop_sequences,
                    )
                    content = response.content[0].text
                    usage = TokenUsage(
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    )
                    finish_reason = response.stop_reason
                else:
                    raise AttributeError("Messages API not available")
            except (AttributeError, Exception) as e:
                # Fallback to Completions API for older models
                logger.warning(f"Messages API failed, using Completions API: {e}")
                response = await self.anthropic_client.completions.create(
                    model="claude-3-haiku-20240307",  # Use compatible model
                    prompt=f"{anthropic.HUMAN_PROMPT} {request.prompt} {anthropic.AI_PROMPT}",
                    temperature=request.temperature,
                    max_tokens_to_sample=request.max_tokens,
                    top_p=request.top_p,
                    stop_sequences=request.stop_sequences,
                )
                content = response.completion
                usage = TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                )
                finish_reason = response.stop_reason

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Calculate cost
            pricing = self.model_pricing.get(request.model)
            cost = pricing.calculate_cost(usage) if pricing else 0.0

            return LLMResponse(
                content=content,
                model=request.model,
                usage=usage,
                cost=cost,
                latency_ms=latency_ms,
                provider=ModelProvider.ANTHROPIC,
                finish_reason=finish_reason,
            )

        try:
            return await _execute()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Request timeout: {str(e)}") from e
        except AnthropicError as e:
            raise LLMProviderError(f"Anthropic API error: {str(e)}") from e

    @staticmethod
    def _compute_prompt_hash(request: LLMRequest) -> str:
        """Compute hash for caching based on request parameters."""
        import hashlib

        content = f"{request.prompt}|{request.model}|{request.temperature:.3f}|{request.max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def _get_cached_response(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Retrieve cached LLM response if available."""
        try:
            prompt_hash = self._compute_prompt_hash(request)
            cached_data = await self.cache_manager.get(f"llm_cache:{prompt_hash}")

            if cached_data:
                return LLMResponse(
                    content=cached_data["response"],
                    model=request.model,
                    usage=TokenUsage(
                        prompt_tokens=cached_data["prompt_tokens"],
                        completion_tokens=cached_data["completion_tokens"],
                        total_tokens=cached_data["total_tokens"],
                    ),
                    cost=cached_data["cost"],
                    latency_ms=cached_data["latency_ms"],
                    provider=ModelProvider(cached_data["provider"]),
                    timestamp=cached_data["timestamp"],
                    finish_reason=cached_data.get("finish_reason"),
                )
        except Exception as e:
            logger.warning(f"Failed to retrieve cached response: {e}")

        return None

    async def _cache_response(self, request: LLMRequest, response: LLMResponse) -> None:
        """Cache LLM response for future use."""
        try:
            prompt_hash = self._compute_prompt_hash(request)
            cache_data = {
                "response": response.content,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": response.cost,
                "latency_ms": response.latency_ms,
                "provider": response.provider.value,
                "timestamp": response.timestamp,
                "finish_reason": response.finish_reason,
            }

            await self.cache_manager.set(f"llm_cache:{prompt_hash}", cache_data)
            logger.debug(f"Cached Anthropic response | hash={prompt_hash[:16]}...")

        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Retrieve aggregated metrics."""
        async with self._metrics_lock:
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost,
                "avg_tokens_per_request": self.total_tokens / max(self.total_requests, 1),
                "avg_cost_per_request": self.total_cost / max(self.total_requests, 1),
                "provider": "anthropic",
            }

    async def health_check(self) -> Dict[str, str]:
        """Verify connectivity to Anthropic API."""
        try:
            # Anthropic doesn't have a list models endpoint, so we use a minimal request
            model = self.settings.llm.primary_model if self.settings else "claude-3-sonnet-20240229"
            await self.anthropic_client.messages.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return {"anthropic": "healthy"}
        except Exception as e:
            return {"anthropic": f"unhealthy: {str(e)}"}

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.anthropic_client.close()


# ============================================================================
# FACTORY FUNCTION: Dynamic Provider Selection
# ============================================================================


def get_llm_client(
    provider: Optional[str] = None,
    redis_client: Optional[Any] = None,
    cache_manager: Optional[Any] = None,
    metrics_collector: Optional[Any] = None,
    settings: Optional[Any] = None,
    **kwargs,
) -> AbstractLLMClient:
    """
    Factory function to create LLM client based on provider configuration.

    This function implements the Factory design pattern, dynamically selecting
    and instantiating the appropriate LLM client implementation based on the
    configured provider.

    Args:
        provider: LLM provider name ('openai' or 'anthropic'). If None, reads from settings.
        redis_client: Redis client for distributed circuit breaker
        cache_manager: Cache manager for response caching
        metrics_collector: Metrics collector for monitoring
        settings: Application settings (auto-loaded if None)
        **kwargs: Additional provider-specific parameters

    Returns:
        AbstractLLMClient: Configured LLM client instance for the specified provider

    Raises:
        ValueError: If provider is not supported or not configured

    Example:
        >>> from config.settings import get_settings
        >>> settings = get_settings()
        >>> client = get_llm_client(settings=settings)
        >>> response = await client.complete(prompt="Hello", model="gpt-4")
    """
    # Load settings if not provided
    if settings is None:
        from config.settings import get_settings

        settings = get_settings()

    # Determine provider from settings or parameter
    if provider is None:
        if hasattr(settings, "llm") and hasattr(settings.llm, "provider"):
            provider = settings.llm.provider
        else:
            # Default to Anthropic if not specified
            provider = "anthropic"
            logger.warning("No LLM provider specified, defaulting to Anthropic")

    provider = provider.lower()

    # Get API keys from settings
    openai_key = None
    anthropic_key = None

    if hasattr(settings, "llm"):
        if hasattr(settings.llm, "openai_api_key") and settings.llm.openai_api_key:
            openai_key = settings.llm.openai_api_key.get_secret_value()
        if hasattr(settings.llm, "anthropic_api_key") and settings.llm.anthropic_api_key:
            anthropic_key = settings.llm.anthropic_api_key.get_secret_value()

    # Factory logic: instantiate appropriate client
    if provider == "openai":
        logger.info("Instantiating OpenAI LLM client via factory")
        return OpenAIClient(
            redis_client=redis_client,
            cache_manager=cache_manager,
            metrics_collector=metrics_collector,
            settings=settings,
            openai_api_key=kwargs.get("openai_api_key", openai_key),
            timeout=kwargs.get("timeout", 120.0),
            max_retries=kwargs.get("max_retries", 3),
        )

    elif provider == "anthropic":
        logger.info("Instantiating Anthropic LLM client via factory")
        return AnthropicClient(
            redis_client=redis_client,
            cache_manager=cache_manager,
            metrics_collector=metrics_collector,
            settings=settings,
            anthropic_api_key=kwargs.get("anthropic_api_key", anthropic_key),
            timeout=kwargs.get("timeout", 120.0),
            max_retries=kwargs.get("max_retries", 3),
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. "
            f"Supported providers: 'openai', 'anthropic'"
        )
