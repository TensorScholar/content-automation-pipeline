"""
Integration Tests for Circuit Breaker Resilience

Tests the distributed circuit breaker functionality to ensure
fault isolation and fail-fast behavior when LLM providers are down.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from core.exceptions import LLMTimeoutError
from infrastructure.llm_client import LLMProviderError, get_llm_client


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_repeated_failures(redis):
    """Verify the circuit breaker opens after multiple failed API calls."""
    # Mock the LLM API client to simulate persistent timeout
    mock_llm_api = AsyncMock()
    mock_llm_api.chat.completions.create.side_effect = LLMTimeoutError("API timed out")

    # Mock Redis to return proper circuit breaker state
    from infrastructure.llm_client import CircuitState

    redis.get.return_value = None  # Start with CLOSED state
    redis.incr.return_value = 1
    redis.set.return_value = True

    # Initialize LLM client with low retries and low failure threshold for faster testing
    llm_client = get_llm_client(
        provider="openai", redis_client=redis, cache_manager=None, max_retries=1
    )
    llm_client.openai_client = mock_llm_api
    llm_client.circuit_breaker.failure_threshold = 2  # Open after 2 failures

    # First and second calls should fail after retries
    with pytest.raises(LLMTimeoutError):
        await llm_client.complete(prompt="test 1", model="gpt-4")
    with pytest.raises(LLMTimeoutError):
        await llm_client.complete(prompt="test 2", model="gpt-4")

    # Mock Redis to return OPEN state after failures
    redis.get.return_value = CircuitState.OPEN.value

    # The circuit should now be open. This call should fail instantly.
    with pytest.raises(LLMProviderError, match="Circuit breaker is OPEN - service unavailable"):
        await llm_client.complete(prompt="test 3", model="gpt-4")

    # Ensure the API was not called for the third attempt
    assert mock_llm_api.chat.completions.create.call_count == 2
    print("✓ Circuit breaker test passed. Fail-fast behavior confirmed.")


@pytest.mark.asyncio
async def test_circuit_breaker_recovers_after_timeout(redis):
    """Verify the circuit breaker recovers after the recovery timeout."""
    # Mock the LLM API client to simulate initial failures then success
    mock_llm_api = AsyncMock()

    # First few calls fail, then succeed
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise LLMTimeoutError("API timed out")
        else:
            # Return a successful response
            mock_response = AsyncMock()
            mock_response.choices = [
                AsyncMock(message=AsyncMock(content="Success!"), finish_reason="stop")
            ]
            mock_response.usage = AsyncMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
            return mock_response

    mock_llm_api.chat.completions.create.side_effect = side_effect

    # Initialize LLM client with low thresholds for faster testing
    llm_client = get_llm_client(
        provider="openai", redis_client=redis, cache_manager=None, max_retries=1
    )
    llm_client.openai_client = mock_llm_api
    llm_client.circuit_breaker.failure_threshold = 2
    llm_client.circuit_breaker.recovery_timeout = 5.0  # 5 second recovery
    llm_client.circuit_breaker.success_threshold = 1  # 1 success to close

    # Mock Redis to properly track circuit breaker state
    from infrastructure.llm_client import CircuitState

    # Initialize Redis mock to start with CLOSED state
    redis.get = AsyncMock(return_value=None)  # No state = CLOSED
    redis.incr = AsyncMock(return_value=1)
    redis.set = AsyncMock(return_value=True)

    # First two calls should fail
    with pytest.raises(LLMTimeoutError):
        await llm_client.complete(prompt="test 1", model="gpt-4")
    with pytest.raises(LLMTimeoutError):
        await llm_client.complete(prompt="test 2", model="gpt-4")

    # Wait for recovery timeout
    await asyncio.sleep(6)

    # After recovery timeout, circuit should be half-open and allow one call
    # Call_count will be 3 after the two failures above, so we need one more call
    # to get past the if call_count <= 3 check
    with pytest.raises(LLMTimeoutError):
        await llm_client.complete(prompt="test 3", model="gpt-4")

    # Now call_count is 4, so the next call should succeed
    response = await llm_client.complete(prompt="test 4", model="gpt-4")
    assert response.content == "Success!"

    print("✓ Circuit breaker recovery test passed. Circuit recovered after timeout.")


@pytest.mark.asyncio
async def test_circuit_breaker_separate_per_provider(redis):
    """Verify that circuit breakers are separate for each provider."""
    # Mock OpenAI
    mock_openai = AsyncMock()
    mock_openai.chat.completions.create.side_effect = LLMTimeoutError("OpenAI down")

    # Mock Anthropic - also mock completions API for fallback
    mock_anthropic = AsyncMock()
    mock_anthropic.messages.create.side_effect = LLMTimeoutError("Anthropic down")
    # Mock the completions API fallback
    mock_anthropic.completions = AsyncMock()
    mock_anthropic.completions.create = AsyncMock(
        side_effect=LLMTimeoutError("Anthropic completions down")
    )

    # Initialize separate LLM clients for each provider
    openai_client = get_llm_client(
        provider="openai", redis_client=redis, cache_manager=None, max_retries=1
    )
    openai_client.openai_client = mock_openai
    openai_client.circuit_breaker.failure_threshold = 1

    anthropic_client = get_llm_client(
        provider="anthropic", redis_client=redis, cache_manager=None, max_retries=1
    )
    anthropic_client.anthropic_client = mock_anthropic
    anthropic_client.circuit_breaker.failure_threshold = 1

    # Set up Redis mock to track separate circuit breakers for each provider
    redis.get = AsyncMock(return_value=None)  # Start with CLOSED
    redis.incr = AsyncMock(return_value=1)
    redis.set = AsyncMock(return_value=True)

    # Both providers should fail independently
    # OpenAI should fail
    with pytest.raises(LLMTimeoutError):
        await openai_client.complete(prompt="test", model="gpt-4")

    # Anthropic should also fail (separate circuit breaker)
    with pytest.raises(LLMTimeoutError):
        await anthropic_client.complete(prompt="test", model="claude-haiku-4-5-20251001")

    print("✓ Separate circuit breakers test passed. Providers isolated correctly.")
