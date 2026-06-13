from unittest.mock import MagicMock, patch

import bleach
import pytest

from core.exceptions import ValidationError
from intelligence.semantic_analyzer import MAX_TEXT_LENGTH, SemanticAnalyzer


@pytest.mark.asyncio
async def test_semantic_input_sanitization():
    """Verify that dangerous HTML inputs are sanitized before processing."""
    mock_redis = MagicMock()
    # Mock model to avoid loading heavy weights
    with patch("intelligence.semantic_analyzer.SentenceTransformer") as MockTransformer:
        MockTransformer.return_value.get_sentence_embedding_dimension.return_value = 384
        MockTransformer.return_value.encode.return_value = [0.1] * 384

        analyzer = SemanticAnalyzer(redis_client=mock_redis)
        # Force model load
        _ = analyzer.model

        # 1. Test XSS Injection
        malicious_input = "<script>alert('xss')</script>Hello World"
        # We need to mock _embed_single caching logic to check the input that actually gets embedded
        # But simpler is to test the _sanitize_text helper if it was exposed,
        # or rely on the fact that bleach.clean strips tags.

        # Let's inspect what happens by mocking the model.encode method and checking args
        analyzer.model.encode = MagicMock(return_value=[0.1] * 384)

        await analyzer.embed(malicious_input, use_cache=False)

        # Verify the model received sanitized text
        called_args, _ = analyzer.model.encode.call_args
        passed_text = called_args[0]

        assert "<script>" not in passed_text
        assert "alert" in passed_text  # Content remains
        assert "Hello World" in passed_text
        assert passed_text == "alert('xss')Hello World"

@pytest.mark.asyncio
async def test_semantic_input_length_limit():
    """Verify that oversized inputs raise ValidationError."""
    mock_redis = MagicMock()
    analyzer = SemanticAnalyzer(redis_client=mock_redis)

    # Create huge string
    huge_input = "a" * (MAX_TEXT_LENGTH + 100)

    with pytest.raises(ValidationError) as exc:
        await analyzer.embed(huge_input)

    assert f"exceeds maximum allowed {MAX_TEXT_LENGTH}" in str(exc.value)
