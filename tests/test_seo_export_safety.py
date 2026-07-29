"""Tests for Phase 2: SEO Export Safety & Correctness.

Verifies that:
1. Persian articles export with lang="fa" dir="rtl".
2. Titles containing <script>alert(1)</script> are safely escaped in HTML output.
3. JSON-LD content containing </script> does not break the HTML document.
4. No fabricated answer text in FAQPage schema (removed FAQ/HowTo heuristics).
"""
import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from api.routes.content import _build_jsonld_schema, export_article_html_with_schema
from core.models import User

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

MOCK_ARTICLE_ID = uuid4()
MOCK_USER = User(
    id=uuid4(),
    email="test@smarlux.com",
    role="user",
    full_name="Test User",
    hashed_password="",
    is_active=True,
)

# ---------------------------------------------------------------------------
# Test 1: JSON-LD Schema (No fabricated FAQ/HowTo)
# ---------------------------------------------------------------------------

def test_jsonld_schema_is_article_only():
    """Verify that heuristic FAQ/HowTo schemas have been removed."""
    content = "Here is a question? This is an answer. Step 1. Do this. Step 2. Do that."
    schema = _build_jsonld_schema(title="Test", content=content)

    assert schema["@type"] == "Article"
    assert "mainEntity" not in schema
    assert "step" not in schema

# ---------------------------------------------------------------------------
# Test 2: HTML Export Safety (XSS, Injection, lang/dir)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_html_export_escapes_xss_and_injection():
    """Verify HTML escaping for title and JSON-LD script tag injection."""
    mock_content_service = AsyncMock()
    mock_content_service.get_article.return_value = {
        "title": "<script>alert('xss')</script>",
        "content": (
            '<p onclick="alert(1)">محتوای امن</p>'
            '<script>alert("stored")</script>'
            '<a href="javascript:alert(2)">bad link</a>'
        ),
        "language": "fa"
    }

    result = await export_article_html_with_schema(
        article_id=MOCK_ARTICLE_ID,
        content_service=mock_content_service,
        user=MOCK_USER
    )

    html_payload = result["html"]

    # Check Lang and Dir for Persian
    assert '<html lang="fa" dir="rtl">' in html_payload

    # Check Title escaping
    assert "<title>&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;</title>" in html_payload
    assert "<h1>&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;</h1>" in html_payload
    assert "<script>alert('xss')</script>" not in html_payload

    # Check JSON-LD injection prevention and stored-content sanitization.
    assert "<\\/script>" in html_payload
    assert "onclick=" not in html_payload
    assert "<script>alert(\"stored\")</script>" not in html_payload
    assert "javascript:" not in html_payload

@pytest.mark.asyncio
async def test_html_export_ltr_english():
    """Verify LTR English default."""
    mock_content_service = AsyncMock()
    mock_content_service.get_article.return_value = {
        "title": "Normal Title",
        "content": "<p>Normal content</p>",
        "language": "en"
    }

    result = await export_article_html_with_schema(
        article_id=MOCK_ARTICLE_ID,
        content_service=mock_content_service,
        user=MOCK_USER
    )

    html_payload = result["html"]
    assert '<html lang="en" dir="ltr">' in html_payload


@pytest.mark.asyncio
async def test_html_export_converts_persian_markdown_to_semantic_html():
    mock_content_service = AsyncMock()
    mock_content_service.get_article.return_value = {
        "title": "راهنمای تولید محتوا",
        "content": (
            "## برنامه‌ریزی محتوا\n\n"
            "این یک پاراگراف فارسی است.\n\n"
            "- نیاز مخاطب را بررسی کنید\n"
            "- هدف مقاله را مشخص کنید"
        ),
        "language": "fa",
    }

    result = await export_article_html_with_schema(
        article_id=MOCK_ARTICLE_ID,
        content_service=mock_content_service,
        user=MOCK_USER,
    )

    html_payload = result["html"]
    assert '<html lang="fa" dir="rtl">' in html_payload
    assert "<h2>برنامه‌ریزی محتوا</h2>" in html_payload
    assert "<p>این یک پاراگراف فارسی است.</p>" in html_payload
    assert "<ul>" in html_payload
    assert "<li>نیاز مخاطب را بررسی کنید</li>" in html_payload
    assert "## برنامه‌ریزی محتوا" not in html_payload


@pytest.mark.asyncio
async def test_html_export_converts_mixed_generator_html_and_markdown():
    mock_content_service = AsyncMock()
    mock_content_service.get_article.return_value = {
        "title": "Mixed export",
        "content": (
            "<h2>Existing section heading</h2>\n"
            "A Markdown paragraph with **important detail**.\n\n"
            "- First item\n"
            "- Second item"
        ),
        "language": "en",
    }

    result = await export_article_html_with_schema(
        article_id=MOCK_ARTICLE_ID,
        content_service=mock_content_service,
        user=MOCK_USER,
    )

    html_payload = result["html"]
    assert "<h2>Existing section heading</h2>" in html_payload
    assert "<p>A Markdown paragraph with <strong>important detail</strong>.</p>" in html_payload
    assert "<li>First item</li>" in html_payload
