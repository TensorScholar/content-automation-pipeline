"""
PHASE 1 Verification Test: Architectural Cleanup
=================================================

Verifies that:
1. The root endpoint (/) returns JSON or redirects to /docs (NOT HTML)
2. The /docs endpoint is accessible
3. No HTML/CSS/JS tags exist in api/main.py
"""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint_no_html():
    """Test that GET / does not return HTML, returns 200 JSON or 302 redirect."""
    from api.main import app

    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    # Should be either 200 with JSON or 302/307 redirect
    assert response.status_code in [200, 302, 307], (
        f"Expected status code 200 (JSON) or 302/307 (redirect), got {response.status_code}"
    )

    # If it's 200, must be JSON
    if response.status_code == 200:
        assert "application/json" in response.headers.get("content-type", ""), (
            "Root endpoint returned 200 but not JSON"
        )

    # If it's a redirect, check it's to /docs
    if response.status_code in [302, 307]:
        location = response.headers.get("location", "")
        assert "/docs" in location, f"Expected redirect to /docs, got {location}"

    # Must NOT be HTML
    content_type = response.headers.get("content-type", "")
    assert "text/html" not in content_type, (
        "Root endpoint must not return HTML"
    )


def test_docs_endpoint_accessible():
    """Test that GET /docs returns 200 (OpenAPI UI)."""
    from api.main import app

    client = TestClient(app)
    response = client.get("/docs")

    assert response.status_code == 200, (
        f"Expected /docs to return 200, got {response.status_code}"
    )

    # Should be HTML (this is the OpenAPI docs UI)
    assert "text/html" in response.headers.get("content-type", ""), (
        "/docs should return HTML (OpenAPI UI)"
    )


def test_main_py_no_html_tags():
    """Scan api/main.py to ensure no <html> or <style> tags exist."""
    # Get the project root (go up from tests/verification to project root)
    project_root = Path(__file__).parent.parent.parent
    main_py_path = project_root / "api" / "main.py"

    assert main_py_path.exists(), f"api/main.py not found at {main_py_path}"

    with open(main_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for HTML tags
    assert "<html>" not in content.lower(), (
        "Found <html> tag in api/main.py - HTML code must be removed"
    )

    assert "<style>" not in content.lower(), (
        "Found <style> tag in api/main.py - CSS code must be removed"
    )

    assert "<script>" not in content.lower(), (
        "Found <script> tag in api/main.py - JavaScript code must be removed"
    )


def test_main_py_line_count():
    """Verify api/main.py is under 500 lines (reasonable for pure backend logic)."""
    project_root = Path(__file__).parent.parent.parent
    main_py_path = project_root / "api" / "main.py"

    with open(main_py_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    line_count = len(lines)

    assert line_count < 500, (
        f"api/main.py has {line_count} lines - should be under 500 after cleanup"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
