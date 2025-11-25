"""
Unit Tests for Authentication and Authorization

Refactored to use FastAPI's dependency_overrides to mock the UserService,
which is the correct way to test protected endpoints in isolation.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.auth import get_user_service_dependency
from core.models import User as PublicUser
from security import (
    TokenData,
    UserInDB,
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from services.user_service import UserService


@pytest.fixture
def mock_user_service():
    """Creates a mock UserService with predefined users."""
    mock = AsyncMock(spec=UserService)

    hashed_password = get_password_hash("password")
    admin_user_db = UserInDB(
        id="dev-admin-001",
        username="admin",
        email="admin@example.com",
        full_name="Development Admin",
        hashed_password=hashed_password,
        is_active=True,
        is_superuser=True,
        created_at=datetime.utcnow(),
    )

    inactive_user_db = UserInDB(
        id="dev-inactive-001",
        username="inactive",
        email="inactive@example.com",
        full_name="Inactive User",
        hashed_password=get_password_hash("password"),
        is_active=False,
        is_superuser=False,
        created_at=datetime.utcnow(),
    )

    async def mock_auth(username, password):
        if username == "admin@example.com" and verify_password("password", hashed_password):
            return admin_user_db
        if username == "inactive@example.com" and verify_password(
            "password", inactive_user_db.hashed_password
        ):
            return inactive_user_db
        return None

    async def mock_get_by_email(email):
        if email == "admin@example.com":
            return admin_user_db
        if email == "inactive@example.com":
            return inactive_user_db
        return None

    mock.authenticate_user = AsyncMock(side_effect=mock_auth)
    mock.get_user_by_email = AsyncMock(side_effect=mock_get_by_email)

    return mock


@pytest.fixture
def client(mock_user_service):
    """FastAPI test client with mocked UserService."""
    app.dependency_overrides[get_user_service_dependency] = lambda: mock_user_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides = {}  # Clear overrides after test


class TestAuthRoutes:
    def test_login_for_access_token_success(self, client: TestClient):
        """Test successful login and token generation."""
        response = client.post(
            "/auth/token", data={"username": "admin@example.com", "password": "password"}
        )
        assert response.status_code == 200
        token_data = response.json()
        assert "access_token" in token_data
        assert token_data["token_type"] == "bearer"

        # Verify the token
        token = token_data["access_token"]
        decoded = decode_access_token(token)
        assert decoded.username == "admin@example.com"
        assert decoded.scopes == ["read", "write"]

    def test_login_invalid_password(self, client: TestClient):
        """Test login with incorrect password."""
        response = client.post(
            "/auth/token", data={"username": "admin@example.com", "password": "wrongpassword"}
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_login_invalid_username(self, client: TestClient):
        """Test login with non-existent username."""
        response = client.post(
            "/auth/token", data={"username": "nouser@example.com", "password": "password"}
        )
        assert response.status_code == 401

    def test_login_inactive_user(self, client: TestClient):
        """Test login attempt by an inactive user."""
        response = client.post(
            "/auth/token", data={"username": "inactive@example.com", "password": "password"}
        )
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_read_users_me_success(self, client: TestClient):
        """Test /auth/me with a valid token."""
        # Create a token directly for the test
        token = create_access_token(data={"sub": "admin@example.com", "user_id": "dev-admin-001"})

        response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        user_data = response.json()
        assert user_data["email"] == "admin@example.com"
        assert user_data["is_active"] is True
        assert "hashed_password" not in user_data

    def test_read_users_me_inactive(self, client: TestClient):
        """Test /auth/me with a token for an inactive user."""
        token = create_access_token(
            data={"sub": "inactive@example.com", "user_id": "dev-inactive-001"}
        )

        response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 400
        assert "Inactive user" in response.json()["detail"]

    def test_read_users_me_no_token(self, client: TestClient):
        """Test /auth/me without a token."""
        response = client.get("/auth/me")
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

    def test_read_users_me_invalid_token(self, client: TestClient):
        """Test /auth/me with an invalid token."""
        response = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.string"})
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]

    def test_admin_endpoint_as_superuser(self, client: TestClient):
        """Test admin-only endpoint with superuser token."""
        token = create_access_token(
            data={
                "sub": "admin@example.com",
                "user_id": "dev-admin-001",
                "scopes": ["read", "write"],
            }
        )
        response = client.get("/auth/admin", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["message"] == "Admin access granted"

    def test_admin_endpoint_as_regular_user(self, client: TestClient, mock_user_service):
        """Test admin-only endpoint with regular user token."""
        # Need to add a regular user to the mock service for this test
        reg_user_db = UserInDB(
            id="dev-reg-001",
            username="user",
            email="user@example.com",
            full_name="Regular User",
            hashed_password=get_password_hash("password"),
            is_active=True,
            is_superuser=False,
            created_at=datetime.utcnow(),
        )

        async def mock_get_by_email_with_user(email):
            if email == "admin@example.com":
                return UserInDB(
                    id="dev-admin-001",
                    username="admin",
                    email="admin@example.com",
                    full_name="Development Admin",
                    hashed_password=get_password_hash("password"),
                    is_active=True,
                    is_superuser=True,
                    created_at=datetime.utcnow(),
                )
            if email == "user@example.com":
                return reg_user_db
            return None

        mock_user_service.get_user_by_email.side_effect = mock_get_by_email_with_user

        token = create_access_token(
            data={"sub": "user@example.com", "user_id": "dev-reg-001", "scopes": ["read"]}
        )
        response = client.get("/auth/admin", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 403
        assert "Not enough permissions" in response.json()["detail"]

    def test_refresh_token_uses_email_subject(self, client: TestClient, mock_user_service):
        """Ensure refresh endpoint issues tokens with email as sub to match lookup."""
        # Login to get a valid token first
        response = client.post(
            "/auth/token", data={"username": "admin@example.com", "password": "password"}
        )
        assert response.status_code == 200
        token = response.json()["access_token"]

        # Call refresh with the valid token
        refresh_response = client.post(
            "/auth/refresh", headers={"Authorization": f"Bearer {token}"}
        )
        assert refresh_response.status_code == 200
        new_token = refresh_response.json()["access_token"]

        # Decode and verify that sub is email
        decoded = decode_access_token(new_token)
        assert decoded.username == "admin@example.com"
