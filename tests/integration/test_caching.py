"""
Integration Tests for LLM Caching

Tests the semantic LLM response caching functionality to ensure
redundant API calls are avoided and cache hits work correctly.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from infrastructure.llm_client import get_llm_client
from optimization.cache_manager import CacheManager


@pytest.mark.asyncio
async def test_llm_caching_reduces_api_calls(redis):
    """Verify that a repeated LLM request results in a cache hit."""
    # Mock the LLM API client
    mock_llm_api = AsyncMock()
    mock_llm_api.chat.completions.create.return_value = MagicMock()
    mock_llm_api.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="This is a test response."), finish_reason="stop")
    ]
    mock_llm_api.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=5, completion_tokens=10, total_tokens=15
    )

    # Initialize cache manager and LLM client
    cache_manager = CacheManager()
    llm_client = get_llm_client(provider="openai", redis_client=redis, cache_manager=cache_manager)
    llm_client.openai_client = mock_llm_api  # Mock the actual client

    prompt = "This is a unique test prompt for caching."
    model = "gpt-4"

    # First call - should trigger an API call
    response1 = await llm_client.complete(prompt=prompt, model=model)
    assert response1.content == "This is a test response."
    first_call_count = mock_llm_api.chat.completions.create.call_count

    # Second call - should be a cache hit
    response2 = await llm_client.complete(prompt=prompt, model=model)
    assert response2.content == "This is a test response."
    # The mock should NOT be called a second time (cache hit)
    assert mock_llm_api.chat.completions.create.call_count == first_call_count

    print("✓ Caching test passed. API call was avoided on the second request.")


@pytest.mark.asyncio
async def test_cache_miss_with_different_parameters(redis):
    """Verify that different parameters result in cache misses."""
    # Mock the LLM API client
    mock_llm_api = AsyncMock()
    mock_llm_api.chat.completions.create.return_value = MagicMock()
    mock_llm_api.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="This is a test response."), finish_reason="stop")
    ]
    mock_llm_api.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=5, completion_tokens=10, total_tokens=15
    )

    # Initialize cache manager and LLM client
    cache_manager = CacheManager()
    llm_client = get_llm_client(provider="openai", redis_client=redis, cache_manager=cache_manager)
    llm_client.openai_client = mock_llm_api

    # Use unique prompts to avoid cache conflicts
    prompt1 = f"This is a unique test prompt for cache miss test {id(cache_manager)}."
    prompt2 = f"This is another unique test prompt for cache miss test {id(cache_manager)}."
    model = "gpt-4"

    # First call with temperature 0.7
    response1 = await llm_client.complete(prompt=prompt1, model=model, temperature=0.7)
    first_call_count = mock_llm_api.chat.completions.create.call_count
    assert first_call_count == 1

    # Second call with different temperature - should be cache miss
    response2 = await llm_client.complete(prompt=prompt1, model=model, temperature=0.9)
    second_call_count = mock_llm_api.chat.completions.create.call_count
    assert second_call_count == 2  # Should be called again

    print("✓ Cache miss test passed. Different parameters triggered new API calls.")


@pytest.mark.asyncio
async def test_caching_without_cache_manager(redis):
    """Verify that LLM client works without cache manager (no caching)."""
    # Mock the LLM API client
    mock_llm_api = AsyncMock()
    mock_llm_api.chat.completions.create.return_value = MagicMock()
    mock_llm_api.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="This is a test response."), finish_reason="stop")
    ]
    mock_llm_api.chat.completions.create.return_value.usage = MagicMock(
        prompt_tokens=5, completion_tokens=10, total_tokens=15
    )

    # Initialize LLM client without cache manager
    llm_client = get_llm_client(provider="openai", redis_client=redis, cache_manager=None)
    llm_client.openai_client = mock_llm_api

    prompt = "This is a test prompt."
    model = "gpt-4"

    # Both calls should trigger API calls since no caching
    response1 = await llm_client.complete(prompt=prompt, model=model)
    response2 = await llm_client.complete(prompt=prompt, model=model)

    assert mock_llm_api.chat.completions.create.call_count == 2
    print("✓ No caching test passed. Both requests triggered API calls.")


@pytest.mark.asyncio
async def test_cache_manager_cleanup_and_optimize():
    """Validate CacheManager maintenance paths (cleanup_expired, optimize)."""
    from datetime import timedelta as td
    from optimization.cache_manager import CacheManager, CacheLevel

    # Use a small memory limit to trigger eviction code paths
    cm = CacheManager(redis_client=MagicMock(), max_memory_entries=2)

    # Insert entries with short TTL to simulate expiry
    await cm.set("k1", {"v": 1}, ttl=1, levels=[CacheLevel.MEMORY])
    await cm.set("k2", {"v": 2}, ttl=1, levels=[CacheLevel.MEMORY])

    # Force an expired state
    import asyncio as aio
    await aio.sleep(1.1)

    removed = await cm.cleanup_expired()
    assert removed >= 1

    # Add 3 items to trigger LRU eviction on set
    await cm.set("a", 1, levels=[CacheLevel.MEMORY])
    await cm.set("b", 2, levels=[CacheLevel.MEMORY])
    await cm.set("c", 3, levels=[CacheLevel.MEMORY])

    stats_before = cm.get_statistics()
    assert stats_before["memory_cache"]["size"] <= 2

    result = await cm.optimize()
    assert "final_size" in result
