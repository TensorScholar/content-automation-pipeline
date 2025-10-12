"""
Integration Tests for LLM Caching

Tests the semantic LLM response caching functionality to ensure
redundant API calls are avoided and cache hits work correctly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from infrastructure.llm_client import LLMClient
from optimization.cache_manager import CacheManager


@pytest.mark.asyncio
async def test_llm_caching_reduces_api_calls(redis):
    """Verify that a repeated LLM request results in a cache hit."""
    # Mock the LLM API client
    mock_llm_api = AsyncMock()
    mock_llm_api.chat.completions.create.return_value = MagicMock()
    mock_llm_api.chat.completions.create.return_value.choices = [
        MagicMock(
            message=MagicMock(content="This is a test response."),
            finish_reason="stop"
        )
    ]
    mock_llm_api.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15
    )

    # Initialize cache manager and LLM client
    cache_manager = CacheManager()
    llm_client = LLMClient(redis_client=redis, cache_manager=cache_manager)
    llm_client.openai_client = mock_llm_api  # Mock the actual client

    prompt = "This is a unique test prompt for caching."
    model = "gpt-4"

    # First call - should trigger an API call
    response1 = await llm_client.complete(prompt=prompt, model=model)
    assert response1.content == "This is a test response."
    mock_llm_api.chat.completions.create.assert_called_once()

    # Second call - should be a cache hit
    response2 = await llm_client.complete(prompt=prompt, model=model)
    assert response2.content == "This is a test response."
    # The mock should NOT be called a second time
    mock_llm_api.chat.completions.create.assert_called_once()

    print("✓ Caching test passed. API call was avoided on the second request.")


@pytest.mark.asyncio
async def test_cache_miss_with_different_parameters(redis):
    """Verify that different parameters result in cache misses."""
    # Mock the LLM API client
    mock_llm_api = AsyncMock()
    mock_llm_api.chat.completions.create.return_value = MagicMock()
    mock_llm_api.chat.completions.create.return_value.choices = [
        MagicMock(
            message=MagicMock(content="This is a test response."),
            finish_reason="stop"
        )
    ]
    mock_llm_api.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15
    )

    # Initialize cache manager and LLM client
    cache_manager = CacheManager()
    llm_client = LLMClient(redis_client=redis, cache_manager=cache_manager)
    llm_client.openai_client = mock_llm_api

    prompt = "This is a test prompt."
    model = "gpt-4"

    # First call with temperature 0.7
    response1 = await llm_client.complete(prompt=prompt, model=model, temperature=0.7)
    assert mock_llm_api.chat.completions.create.call_count == 1

    # Second call with different temperature - should be cache miss
    response2 = await llm_client.complete(prompt=prompt, model=model, temperature=0.9)
    assert mock_llm_api.chat.completions.create.call_count == 2

    print("✓ Cache miss test passed. Different parameters triggered new API calls.")


@pytest.mark.asyncio
async def test_caching_without_cache_manager(redis):
    """Verify that LLM client works without cache manager (no caching)."""
    # Mock the LLM API client
    mock_llm_api = AsyncMock()
    mock_llm_api.chat.completions.create.return_value = MagicMock()
    mock_llm_api.chat.completions.create.return_value.choices = [
        MagicMock(
            message=MagicMock(content="This is a test response."),
            finish_reason="stop"
        )
    ]
    mock_llm_api.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15
    )

    # Initialize LLM client without cache manager
    llm_client = LLMClient(redis_client=redis, cache_manager=None)
    llm_client.openai_client = mock_llm_api

    prompt = "This is a test prompt."
    model = "gpt-4"

    # Both calls should trigger API calls since no caching
    response1 = await llm_client.complete(prompt=prompt, model=model)
    response2 = await llm_client.complete(prompt=prompt, model=model)
    
    assert mock_llm_api.chat.completions.create.call_count == 2
    print("✓ No caching test passed. Both requests triggered API calls.")
