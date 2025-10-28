"""
Unit Tests for Authentication and Authorization
================================================

Comprehensive unit tests for security module including:
- Password hashing and verification
- JWT token creation and validation
- User authentication
- FastAPI security dependencies
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from jose import jwt

from config.settings import get_settings
from security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    TokenData,
    User,
    UserInDB,
    authenticate_user,
    create_access_token,
    decode_access_token,
    get_current_active_user,
    get_current_superuser,
    get_current_user,
    get_password_hash,
    verify_password,
)


class TestPasswordHashing:
    """Test password hashing and verification functions."""

    def test_password_hash_and_verify(self):
        """Test that password hashing and verification work correctly."""
        plain_password = "MySecurePassword123!"

        # Hash the password
        hashed = get_password_hash(plain_password)

        # Verify correct password
        assert verify_password(plain_password, hashed) is True

        # Verify incorrect password
        assert verify_password("WrongPassword", hashed) is False

    def test_different_hashes_for_same_password(self):
        """Test that hashing the same password twice produces different hashes (salt)."""
        password = "SamePassword123"

        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Hashes should be different due to salt
        assert hash1 != hash2

        # But both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True

    def test_hash_is_string(self):
        """Test that password hash returns a string."""
        hashed = get_password_hash("password")
        assert isinstance(hashed, str)
        assert len(hashed) > 0

    def test_verify_with_empty_password(self):
        """Test verification with empty password."""
        hashed = get_password_hash("realpassword")
        assert verify_password("", hashed) is False

    def test_hash_special_characters(self):
        """Test hashing passwords with special characters."""
        password = "P@$$w0rd!#%&*()[]{}|\\/<>?"
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True


class TestJWTTokens:
    """Test JWT token creation and validation."""

    def test_create_access_token(self):
        """Test JWT token creation."""
        data = {"sub": "testuser", "user_id": "123"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

        # Token should have three parts separated by dots
        parts = token.split(".")
        assert len(parts) == 3

    def test_create_token_with_custom_expiration(self):
        """Test token creation with custom expiration time."""
        data = {"sub": "testuser"}
        expires_delta = timedelta(minutes=60)

        token = create_access_token(data, expires_delta)

        # Decode without verification to check expiration
        settings = get_settings()
        payload = jwt.decode(token, settings.secret_key.get_secret_value(), algorithms=[ALGORITHM])

        exp_timestamp = payload.get("exp")
        exp_datetime = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)

        # Should expire in approximately 60 minutes
        now = datetime.now(timezone.utc)
        delta = exp_datetime - now
        assert 59 <= delta.total_seconds() / 60 <= 61

    def test_decode_valid_token(self):
        """Test decoding a valid JWT token."""
        data = {"sub": "testuser@example.com", "user_id": "user-123", "scopes": ["read", "write"]}
        token = create_access_token(data)

        token_data = decode_access_token(token)

        assert isinstance(token_data, TokenData)
        assert token_data.username == "testuser@example.com"
        assert token_data.user_id == "user-123"
        assert token_data.scopes == ["read", "write"]
        assert isinstance(token_data.expires_at, datetime)

    def test_decode_expired_token(self):
        """Test that decoding an expired token raises HTTPException."""
        data = {"sub": "testuser"}
        expires_delta = timedelta(seconds=-1)  # Already expired

        token = create_access_token(data, expires_delta)

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)

        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in str(exc_info.value.detail)

    def test_decode_invalid_token(self):
        """Test that decoding an invalid token raises HTTPException."""
        invalid_token = "invalid.token.here"

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(invalid_token)

        assert exc_info.value.status_code == 401

    def test_decode_token_without_subject(self):
        """Test that token without 'sub' field raises HTTPException."""
        # Create token manually without 'sub'
        settings = get_settings()
        payload = {"exp": datetime.utcnow() + timedelta(minutes=30), "user_id": "123"}
        token = jwt.encode(payload, settings.secret_key.get_secret_value(), algorithm=ALGORITHM)

        with pytest.raises(HTTPException) as exc_info:
            decode_access_token(token)

        assert exc_info.value.status_code == 401

    def test_token_includes_expiration(self):
        """Test that created tokens include expiration."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        token_data = decode_access_token(token)

        assert token_data.expires_at is not None
        assert token_data.expires_at > datetime.utcnow()


class TestUserAuthentication:
    """Test user authentication functions."""

    def test_authenticate_valid_user(self):
        """Test authenticating with valid credentials."""
        # Mock user exists and password is correct
        user = authenticate_user("admin", "password")

        assert user is not None
        assert isinstance(user, UserInDB)
        assert user.username == "admin"
        assert user.is_active is True

    def test_authenticate_invalid_username(self):
        """Test authentication with non-existent username."""
        user = authenticate_user("nonexistent", "password")
        assert user is None

    def test_authenticate_invalid_password(self):
        """Test authentication with incorrect password."""
        user = authenticate_user("admin", "wrongpassword")
        assert user is None

    def test_authenticate_user_properties(self):
        """Test that authenticated user has expected properties."""
        user = authenticate_user("admin", "password")

        assert user is not None
        assert hasattr(user, "id")
        assert hasattr(user, "username")
        assert hasattr(user, "email")
        assert hasattr(user, "hashed_password")
        assert hasattr(user, "is_active")
        assert hasattr(user, "is_superuser")

    def test_authenticate_regular_user(self):
        """Test authenticating a regular (non-superuser) user."""
        user = authenticate_user("user", "userpass")

        assert user is not None
        assert user.is_active is True
        assert user.is_superuser is False


class TestFastAPIDependencies:
    """Test FastAPI security dependency functions."""

    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        """Test getting current user with valid token."""
        from unittest.mock import AsyncMock

        from security import UserInDB

        # Create a valid token
        token = create_access_token({"sub": "admin", "user_id": "1"})

        # Mock user service
        mock_user_service = AsyncMock()
        mock_user_in_db = UserInDB(
            id="1",
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            hashed_password="hashed_password",
            is_active=True,
            is_superuser=True,
            created_at=datetime.utcnow(),
        )
        mock_user_service.get_user_by_email = AsyncMock(return_value=mock_user_in_db)

        # Patch get_user_service to return our mock
        with patch("container.get_user_service", return_value=mock_user_service):
            user = await get_current_user(token)

        assert isinstance(user, User)
        assert user.username == "admin"
        assert user.is_active is True

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self):
        """Test that invalid token raises HTTPException."""
        invalid_token = "invalid.jwt.token"

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(invalid_token)

        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_current_user_returns_user_without_password(self):
        """Test that returned User object doesn't contain password."""
        from unittest.mock import AsyncMock

        from security import UserInDB

        token = create_access_token({"sub": "admin", "user_id": "1"})

        # Mock user service
        mock_user_service = AsyncMock()
        mock_user_in_db = UserInDB(
            id="1",
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            hashed_password="hashed_password",
            is_active=True,
            is_superuser=True,
            created_at=datetime.utcnow(),
        )
        mock_user_service.get_user_by_email = AsyncMock(return_value=mock_user_in_db)

        # Patch get_user_service to return our mock
        with patch("container.get_user_service", return_value=mock_user_service):
            user = await get_current_user(token)

        # User object should not have hashed_password attribute
        assert not hasattr(user, "hashed_password")
        assert hasattr(user, "username")
        assert hasattr(user, "email")

    @pytest.mark.asyncio
    async def test_get_current_active_user_with_active_user(self):
        """Test getting current active user when user is active."""
        # Create active user
        current_user = User(
            id="1",
            username="testuser",
            email="test@example.com",
            is_active=True,
            is_superuser=False,
            created_at=datetime.utcnow(),
        )

        result = await get_current_active_user(current_user)

        assert result == current_user
        assert result.is_active is True

    @pytest.mark.asyncio
    async def test_get_current_active_user_with_inactive_user(self):
        """Test that inactive user raises HTTPException."""
        # Create inactive user
        current_user = User(
            id="1",
            username="testuser",
            email="test@example.com",
            is_active=False,
            is_superuser=False,
            created_at=datetime.utcnow(),
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(current_user)

        assert exc_info.value.status_code == 400
        assert "Inactive user" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_current_superuser_with_superuser(self):
        """Test getting superuser when user has superuser privileges."""
        current_user = User(
            id="1",
            username="admin",
            email="admin@example.com",
            is_active=True,
            is_superuser=True,
            created_at=datetime.utcnow(),
        )

        result = await get_current_superuser(current_user)

        assert result == current_user
        assert result.is_superuser is True

    @pytest.mark.asyncio
    async def test_get_current_superuser_with_regular_user(self):
        """Test that regular user raises HTTPException when accessing superuser endpoint."""
        current_user = User(
            id="2",
            username="user",
            email="user@example.com",
            is_active=True,
            is_superuser=False,
            created_at=datetime.utcnow(),
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_superuser(current_user)

        assert exc_info.value.status_code == 403
        assert "Not enough permissions" in str(exc_info.value.detail)


class TestTokenData:
    """Test TokenData model."""

    def test_token_data_creation(self):
        """Test creating TokenData instance."""
        token_data = TokenData(
            username="testuser",
            user_id="123",
            scopes=["read", "write"],
            expires_at=datetime.utcnow(),
        )

        assert token_data.username == "testuser"
        assert token_data.user_id == "123"
        assert token_data.scopes == ["read", "write"]
        assert isinstance(token_data.expires_at, datetime)

    def test_token_data_optional_fields(self):
        """Test TokenData with optional fields as None."""
        token_data = TokenData()

        assert token_data.username is None
        assert token_data.user_id is None
        assert token_data.scopes == []
        assert token_data.expires_at is None


class TestUserModels:
    """Test User and UserInDB models."""

    def test_user_model(self):
        """Test User model creation."""
        user = User(
            id="123",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            is_active=True,
            is_superuser=False,
            created_at=datetime.utcnow(),
        )

        assert user.id == "123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True

    def test_user_in_db_model(self):
        """Test UserInDB model with hashed password."""
        hashed_password = get_password_hash("testpass")

        user = UserInDB(
            id="123",
            username="testuser",
            email="test@example.com",
            hashed_password=hashed_password,
            is_active=True,
            is_superuser=False,
            created_at=datetime.utcnow(),
        )

        assert user.hashed_password == hashed_password
        assert verify_password("testpass", user.hashed_password) is True


# Test coverage validation
def test_module_imports():
    """Test that all required security functions are importable."""
    from security import (
        authenticate_user,
        create_access_token,
        decode_access_token,
        get_current_active_user,
        get_current_superuser,
        get_current_user,
        get_password_hash,
        oauth2_scheme,
        verify_password,
    )

    assert callable(verify_password)
    assert callable(get_password_hash)
    assert callable(create_access_token)
    assert callable(decode_access_token)
    assert callable(authenticate_user)
    assert callable(get_current_user)
    assert callable(get_current_active_user)
    assert callable(get_current_superuser)
    assert oauth2_scheme is not None
