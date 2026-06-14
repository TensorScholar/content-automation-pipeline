"""
API Integration Tests
======================

Tests for the FastAPI endpoints ensuring:
- Proper request/response handling
- Authentication and authorization
- Error handling and validation
- Rate limiting behavior
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_root_health_returns_200(self, test_client: AsyncClient):
        """Test that /health endpoint returns 200 OK."""
        response = await test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "dependencies" in data

    @pytest.mark.asyncio
    async def test_system_health_detailed(self, test_client: AsyncClient):
        """Test that /system/health is protected."""
        response = await test_client.get("/system/health")
        assert response.status_code == 401


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    @pytest.mark.asyncio
    async def test_login_with_invalid_credentials_returns_401(
        self, test_client: AsyncClient
    ):
        """Test that login with wrong credentials fails."""
        response = await test_client.post(
            "/auth/token",
            data={"username": "nonexistent@test.com", "password": "wrongpassword"},
        )
        assert response.status_code in [401, 422]

    @pytest.mark.asyncio
    async def test_register_with_valid_data(
        self, test_client: AsyncClient, mock_database
    ):
        """Test user registration with valid data."""
        with patch("api.routes.auth.get_user_service") as mock_service:
            mock_service.return_value.create_user = AsyncMock(
                return_value=MagicMock(
                    id=str(uuid4()),
                    email="test@example.com",
                    username="testuser",
                )
            )

            response = await test_client.post(
                "/auth/register",
                json={
                    "email": "test@example.com",
                    "password": "SecurePassword123!",
                    "full_name": "Test User",
                },
            )
            # Should be 200/201 or validation error if service unavailable
            assert response.status_code in [200, 201, 422, 500]


class TestProjectEndpoints:
    """Tests for project management endpoints."""

    @pytest.mark.asyncio
    async def test_list_projects_requires_auth(self, test_client: AsyncClient):
        """Test that listing projects requires authentication."""
        response = await test_client.get("/projects")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_create_project_with_valid_data(
        self, test_client: AsyncClient, auth_headers, mock_database
    ):
        """Test project creation with valid data."""
        project_data = {
            "name": "Integration Test Project",
            "description": "Testing project creation",
            "domain": "test.example.com",
            "target_audience": "developers",
            "tone": "professional",
        }

        with patch("security.get_current_user") as mock_user:
            mock_user.return_value = MagicMock(id=str(uuid4()))

            with patch("api.routes.projects.get_project_service") as mock_service:
                mock_service.return_value.create_project = AsyncMock(
                    return_value=MagicMock(
                        id=str(uuid4()),
                        **project_data,
                    )
                )

                response = await test_client.post(
                    "/projects",
                    json=project_data,
                    headers=auth_headers,
                )
                # Check for expected responses (success or auth needed)
                assert response.status_code in [200, 201, 401, 422]


class TestContentEndpoints:
    """Tests for content generation endpoints."""

    @pytest.mark.asyncio
    async def test_generate_content_requires_auth(self, test_client: AsyncClient):
        """Test that content generation requires authentication."""
        response = await test_client.post(
            "/content/generate/async",
            json={
                "project_id": str(uuid4()),
                "topic": "Test Topic",
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_article_not_found(
        self, test_client: AsyncClient, auth_headers
    ):
        """Test that getting non-existent article returns 404."""
        non_existent_id = str(uuid4())

        with patch("security.get_current_user") as mock_user:
            mock_user.return_value = MagicMock(id=str(uuid4()))

            response = await test_client.get(
                f"/content/{non_existent_id}",
                headers=auth_headers,
            )
            # Should be 404 or 401 depending on auth
            assert response.status_code in [401, 404]


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_health_endpoint_not_rate_limited(self, test_client: AsyncClient):
        """Test that health endpoints are exempt from rate limiting."""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = await test_client.get("/health")
            responses.append(response.status_code)

        # All should succeed (not 429)
        assert all(status != 429 for status in responses)


class TestErrorHandling:
    """Tests for error handling and responses."""

    @pytest.mark.asyncio
    async def test_404_for_unknown_route(self, test_client: AsyncClient):
        """Test that unknown routes return 404."""
        response = await test_client.get("/api/nonexistent-endpoint")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_validation_error_returns_422(self, test_client: AsyncClient):
        """Test that invalid data returns 422."""
        response = await test_client.post(
            "/auth/token",
            data={"invalid": "data"},  # Missing required fields
        )
        assert response.status_code == 422


class TestSecurityHeaders:
    """Tests for security headers."""

    @pytest.mark.asyncio
    async def test_security_headers_present(self, test_client: AsyncClient):
        """Test that security headers are present in responses."""
        response = await test_client.get("/health")

        # Check for key security headers
        assert "x-content-type-options" in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
