"""
Unit Tests for LLM Client Factory
=================================

Comprehensive unit tests for the LLM client factory function including:
- Dynamic provider selection based on configuration
- Correct instantiation of provider-specific clients
- Error handling for unsupported providers
- API key configuration validation
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.exceptions import LLMProviderError
from infrastructure.llm_client import (
    AbstractLLMClient,
    AnthropicClient,
    LLMRequest,
    LLMResponse,
    ModelPricing,
    ModelProvider,
    OpenAIClient,
    TokenUsage,
    get_llm_client,
)


class TestGetLLMClientFactory:
    """Test the get_llm_client factory function."""

    def test_factory_returns_openai_client_when_specified(self):
        """Test that factory returns OpenAIClient when provider is 'openai'."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.provider = "openai"
        mock_settings.llm.openai_api_key = MagicMock()
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "test-openai-key"

        # Mock redis client
        mock_redis = MagicMock()

        # Create client via factory
        client = get_llm_client(
            provider="openai",
            redis_client=mock_redis,
            settings=mock_settings,
        )

        # Verify it's an OpenAI client
        assert isinstance(client, OpenAIClient)
        assert isinstance(client, AbstractLLMClient)

    def test_factory_returns_anthropic_client_when_specified(self):
        """Test that factory returns AnthropicClient when provider is 'anthropic'."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.provider = "anthropic"
        mock_settings.llm.anthropic_api_key = MagicMock()
        mock_settings.llm.anthropic_api_key.get_secret_value.return_value = "test-anthropic-key"

        # Mock redis client
        mock_redis = MagicMock()

        # Create client via factory
        client = get_llm_client(
            provider="anthropic",
            redis_client=mock_redis,
            settings=mock_settings,
        )

        # Verify it's an Anthropic client
        assert isinstance(client, AnthropicClient)
        assert isinstance(client, AbstractLLMClient)

    def test_factory_reads_provider_from_settings_when_not_specified(self):
        """Test that factory reads provider from settings when not explicitly specified."""
        # Mock settings with provider set to anthropic
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.provider = "anthropic"
        mock_settings.llm.anthropic_api_key = MagicMock()
        mock_settings.llm.anthropic_api_key.get_secret_value.return_value = "test-key"

        mock_redis = MagicMock()

        # Don't specify provider, should read from settings
        client = get_llm_client(
            redis_client=mock_redis,
            settings=mock_settings,
        )

        # Should be Anthropic client based on settings
        assert isinstance(client, AnthropicClient)

    def test_factory_defaults_to_openai_when_no_provider_configured(self):
        """Test that factory defaults to OpenAI when no provider is configured."""
        # Mock settings without provider
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.provider = None
        mock_settings.llm.openai_api_key = MagicMock()
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "test-key"

        # Remove provider attribute to simulate missing config
        delattr(mock_settings.llm, "provider")

        mock_redis = MagicMock()

        # Should default to OpenAI
        client = get_llm_client(
            redis_client=mock_redis,
            settings=mock_settings,
        )

        assert isinstance(client, OpenAIClient)

    def test_factory_raises_error_for_unsupported_provider(self):
        """Test that factory raises ValueError for unsupported providers."""
        mock_settings = MagicMock()
        mock_redis = MagicMock()

        with pytest.raises(ValueError) as exc_info:
            get_llm_client(
                provider="unsupported_provider",
                redis_client=mock_redis,
                settings=mock_settings,
            )

        assert "Unsupported LLM provider" in str(exc_info.value)
        assert "unsupported_provider" in str(exc_info.value)
        assert "openai" in str(exc_info.value).lower()
        assert "anthropic" in str(exc_info.value).lower()

    def test_factory_handles_case_insensitive_provider_names(self):
        """Test that factory handles provider names case-insensitively."""
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.openai_api_key = MagicMock()
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "test-key"

        mock_redis = MagicMock()

        # Test uppercase
        client = get_llm_client(
            provider="OPENAI",
            redis_client=mock_redis,
            settings=mock_settings,
        )
        assert isinstance(client, OpenAIClient)

        # Test mixed case
        client = get_llm_client(
            provider="OpenAI",
            redis_client=mock_redis,
            settings=mock_settings,
        )
        assert isinstance(client, OpenAIClient)

    def test_factory_passes_kwargs_to_client(self):
        """Test that factory passes additional kwargs to client constructor."""
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.openai_api_key = MagicMock()
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "test-key"

        mock_redis = MagicMock()

        # Create client with custom timeout and max_retries
        client = get_llm_client(
            provider="openai",
            redis_client=mock_redis,
            settings=mock_settings,
            timeout=60.0,
            max_retries=5,
        )

        assert client.timeout == 60.0
        assert client.max_retries == 5

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"})
    def test_factory_uses_env_vars_when_no_api_key_in_settings(self):
        """Test that factory falls back to environment variables for API keys."""
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        # No API key in settings
        delattr(mock_settings.llm, "openai_api_key")

        mock_redis = MagicMock()

        # Should use API key from environment
        client = get_llm_client(
            provider="openai",
            redis_client=mock_redis,
            settings=mock_settings,
        )

        assert isinstance(client, OpenAIClient)

    def test_factory_passes_cache_manager_to_client(self):
        """Test that factory passes cache manager to client."""
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.openai_api_key = MagicMock()
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "test-key"

        mock_redis = MagicMock()
        mock_cache = MagicMock()

        client = get_llm_client(
            provider="openai",
            redis_client=mock_redis,
            cache_manager=mock_cache,
            settings=mock_settings,
        )

        assert client.cache_manager is mock_cache

    def test_factory_passes_metrics_collector_to_client(self):
        """Test that factory passes metrics collector to client."""
        mock_settings = MagicMock()
        mock_settings.llm = MagicMock()
        mock_settings.llm.openai_api_key = MagicMock()
        mock_settings.llm.openai_api_key.get_secret_value.return_value = "test-key"

        mock_redis = MagicMock()
        mock_metrics = MagicMock()

        client = get_llm_client(
            provider="openai",
            redis_client=mock_redis,
            metrics_collector=mock_metrics,
            settings=mock_settings,
        )

        assert client.metrics_collector is mock_metrics


class TestModelProvider:
    """Test ModelProvider enum."""

    def test_model_provider_enum_values(self):
        """Test that ModelProvider enum has expected values."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.ANTHROPIC.value == "anthropic"

    def test_model_provider_enum_members(self):
        """Test that ModelProvider has all required members."""
        assert hasattr(ModelProvider, "OPENAI")
        assert hasattr(ModelProvider, "ANTHROPIC")


class TestTokenUsage:
    """Test TokenUsage data class."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage instance."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_validates_non_negative_tokens(self):
        """Test that TokenUsage validates non-negative token counts."""
        with pytest.raises(AssertionError):
            TokenUsage(prompt_tokens=-10, completion_tokens=50, total_tokens=40)

        with pytest.raises(AssertionError):
            TokenUsage(prompt_tokens=100, completion_tokens=-50, total_tokens=50)

    def test_token_usage_validates_total_equals_sum(self):
        """Test that TokenUsage validates total equals sum of prompt and completion."""
        with pytest.raises(AssertionError):
            TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=200)  # Should be 150


class TestModelPricing:
    """Test ModelPricing data class."""

    def test_model_pricing_creation(self):
        """Test creating ModelPricing instance."""
        pricing = ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.03)

        assert pricing.input_cost_per_1k == 0.01
        assert pricing.output_cost_per_1k == 0.03

    def test_model_pricing_calculate_cost(self):
        """Test cost calculation for token usage."""
        pricing = ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.03)

        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

        # 1000 * 0.01 / 1000 + 500 * 0.03 / 1000 = 0.01 + 0.015 = 0.025
        cost = pricing.calculate_cost(usage)

        assert cost == pytest.approx(0.025, rel=1e-6)

    def test_model_pricing_calculates_fractional_cost(self):
        """Test cost calculation with fractional tokens."""
        pricing = ModelPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015)

        usage = TokenUsage(prompt_tokens=250, completion_tokens=100, total_tokens=350)

        # 250 * 0.0005 / 1000 + 100 * 0.0015 / 1000 = 0.000125 + 0.00015 = 0.000275
        cost = pricing.calculate_cost(usage)

        assert cost == pytest.approx(0.000275, rel=1e-6)

    def test_model_pricing_handles_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        pricing = ModelPricing(input_cost_per_1k=0.01, output_cost_per_1k=0.03)

        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        cost = pricing.calculate_cost(usage)
        assert cost == 0.0


class TestLLMRequest:
    """Test LLMRequest validation."""

    def test_llm_request_creation(self):
        """Test creating valid LLMRequest."""
        request = LLMRequest(prompt="Test prompt", model="gpt-4", temperature=0.7, max_tokens=1000)

        assert request.prompt == "Test prompt"
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 1000

    def test_llm_request_validates_model_pattern(self):
        """Test that LLMRequest validates model pattern."""
        # Valid models
        LLMRequest(prompt="test", model="gpt-4", max_tokens=100)
        LLMRequest(prompt="test", model="gpt-3.5-turbo", max_tokens=100)
        LLMRequest(prompt="test", model="claude-3-opus", max_tokens=100)

        # Invalid model
        with pytest.raises(ValueError):
            LLMRequest(prompt="test", model="invalid-model", max_tokens=100)

    def test_llm_request_validates_temperature_range(self):
        """Test that LLMRequest validates temperature range."""
        # Valid temperatures
        LLMRequest(prompt="test", model="gpt-4", temperature=0.0, max_tokens=100)
        LLMRequest(prompt="test", model="gpt-4", temperature=1.0, max_tokens=100)
        LLMRequest(prompt="test", model="gpt-4", temperature=2.0, max_tokens=100)

        # Invalid temperatures
        with pytest.raises(ValueError):
            LLMRequest(prompt="test", model="gpt-4", temperature=-0.1, max_tokens=100)

        with pytest.raises(ValueError):
            LLMRequest(prompt="test", model="gpt-4", temperature=2.1, max_tokens=100)

    def test_llm_request_validates_max_tokens_positive(self):
        """Test that LLMRequest validates max_tokens is positive."""
        # Valid max_tokens
        LLMRequest(prompt="test", model="gpt-4", max_tokens=1)
        LLMRequest(prompt="test", model="gpt-4", max_tokens=1000)

        # Invalid max_tokens
        with pytest.raises(ValueError):
            LLMRequest(prompt="test", model="gpt-4", max_tokens=0)

        with pytest.raises(ValueError):
            LLMRequest(prompt="test", model="gpt-4", max_tokens=-1)

    def test_llm_request_validates_prompt_not_empty(self):
        """Test that LLMRequest validates prompt is not empty."""
        # Valid prompt
        LLMRequest(prompt="test", model="gpt-4", max_tokens=100)

        # Empty prompt
        with pytest.raises(ValueError):
            LLMRequest(prompt="", model="gpt-4", max_tokens=100)

    def test_llm_request_immutable(self):
        """Test that LLMRequest is immutable after creation."""
        request = LLMRequest(prompt="test", model="gpt-4", max_tokens=100)

        # Should not be able to modify
        with pytest.raises((ValueError, TypeError)):
            request.prompt = "new prompt"


class TestLLMResponse:
    """Test LLMResponse data class."""

    def test_llm_response_creation(self):
        """Test creating LLMResponse instance."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            usage=usage,
            cost=0.025,
            latency_ms=1500.0,
            provider=ModelProvider.OPENAI,
            finish_reason="stop",
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage == usage
        assert response.cost == 0.025
        assert response.latency_ms == 1500.0
        assert response.provider == ModelProvider.OPENAI
        assert response.finish_reason == "stop"

    def test_llm_response_validates_non_empty_content(self):
        """Test that LLMResponse validates content is not empty."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with pytest.raises(AssertionError):
            LLMResponse(
                content="",  # Empty content
                model="gpt-4",
                usage=usage,
                cost=0.025,
                latency_ms=1500.0,
                provider=ModelProvider.OPENAI,
            )

    def test_llm_response_validates_non_negative_cost(self):
        """Test that LLMResponse validates cost is non-negative."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with pytest.raises(AssertionError):
            LLMResponse(
                content="test",
                model="gpt-4",
                usage=usage,
                cost=-0.01,  # Negative cost
                latency_ms=1500.0,
                provider=ModelProvider.OPENAI,
            )

    def test_llm_response_validates_non_negative_latency(self):
        """Test that LLMResponse validates latency is non-negative."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

        with pytest.raises(AssertionError):
            LLMResponse(
                content="test",
                model="gpt-4",
                usage=usage,
                cost=0.025,
                latency_ms=-100.0,  # Negative latency
                provider=ModelProvider.OPENAI,
            )


# Test module coverage
def test_module_imports():
    """Test that all required LLM client components are importable."""
    from infrastructure.llm_client import (
        AbstractLLMClient,
        AnthropicClient,
        CircuitBreaker,
        LLMClient,
        LLMRequest,
        LLMResponse,
        ModelPricing,
        ModelProvider,
        OpenAIClient,
        TokenUsage,
        get_llm_client,
    )

    assert callable(get_llm_client)
    assert issubclass(OpenAIClient, AbstractLLMClient)
    assert issubclass(AnthropicClient, AbstractLLMClient)
    assert issubclass(LLMClient, AbstractLLMClient)
