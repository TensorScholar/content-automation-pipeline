"""
Authentication Routes: OAuth2 Token Management

Provides authentication endpoints for:
- Token generation and validation
- User authentication
- Password reset functionality
- Security utilities

Architectural Pattern: OAuth2 + JWT + Security Foundation
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from container import container, get_user_service
from core.models import UserCreate
from security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    Token,
    User,
    create_access_token,
    get_current_active_user,
    get_current_superuser,
    get_current_user,
)
from services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Custom form class for better compatibility
class LoginForm(BaseModel):
    username: str
    password: str


# Simple dependency function for FastAPI
def get_user_service_dependency() -> UserService:
    """Get UserService instance for FastAPI dependency injection."""
    return container.user_service()


@router.post(
    "/token",
    response_model=Token,
    summary="Generate access token",
    description="Authenticate user and generate JWT access token",
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service: UserService = Depends(get_user_service_dependency),
) -> Token:
    """
    Generate access token for authenticated user.

    This endpoint implements OAuth2 password flow for authentication.
    Users provide username and password to receive a JWT access token.
    """
    user = await user_service.authenticate_user(form_data.username, form_data.password)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.email,  # Use email as subject
            "user_id": str(user.id),
            "scopes": ["read", "write"] if user.is_superuser else ["read"],
        },
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        scope="read write" if user.is_superuser else "read",
    )


@router.post(
    "/register",
    response_model=User,
    summary="Register new user",
    description="Create a new user account",
)
async def register_user(
    user_create: UserCreate, user_service: UserService = Depends(get_user_service_dependency)
) -> User:
    """
    Register a new user account.

    Creates a new user with the provided information and returns
    the user data without sensitive information.

    Args:
        user_create: User creation data
        user_service: UserService for user creation

    Returns:
        User: Created user information

    Raises:
        HTTPException: If user creation fails
    """
    try:
        user = await user_service.create_user(user_create)
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account",
        )


@router.get(
    "/me",
    response_model=User,
    summary="Get current user",
    description="Get information about the currently authenticated user",
)
async def read_users_me(current_user: User = Depends(get_current_active_user)) -> User:
    """
    Get current user information.

    Returns the profile information of the currently authenticated user.
    This endpoint requires a valid JWT token.

    Args:
        current_user: Current authenticated user from dependency injection

    Returns:
        User: Current user information
    """
    return current_user


@router.get("/verify", summary="Verify token", description="Verify if the provided token is valid")
async def verify_token(current_user: User = Depends(get_current_user)) -> dict:
    """
    Verify token validity.

    This endpoint can be used to check if a JWT token is still valid
    without performing any additional operations.

    Args:
        current_user: Current user from token validation

    Returns:
        dict: Token verification result
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "expires_at": "token_expiry_info",  # TODO: Add actual expiry info
    }


@router.post(
    "/refresh",
    response_model=Token,
    summary="Refresh access token",
    description="Generate a new access token using existing valid token",
)
async def refresh_access_token(current_user: User = Depends(get_current_active_user)) -> Token:
    """
    Refresh access token.

    Generates a new access token for the currently authenticated user.
    This is useful for extending session duration without requiring
    re-authentication.

    Args:
        current_user: Current authenticated user

    Returns:
        Token: New JWT access token
    """
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": current_user.username,
            "user_id": current_user.id,
            "scopes": ["read", "write"] if current_user.is_superuser else ["read"],
        },
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        scope="read write" if current_user.is_superuser else "read",
    )


@router.get(
    "/admin",
    summary="Admin endpoint",
    description="Admin-only endpoint for testing superuser access",
)
async def admin_endpoint(current_user: User = Depends(get_current_superuser)) -> dict:
    """
    Admin-only endpoint for testing superuser access.

    This endpoint requires superuser privileges and can be used to
    test role-based access control.

    Args:
        current_user: Current superuser from dependency injection

    Returns:
        dict: Admin access confirmation
    """
    return {
        "message": "Admin access granted",
        "user_id": current_user.id,
        "username": current_user.username,
        "is_superuser": current_user.is_superuser,
    }


# Mock user endpoints for testing (remove in production)
@router.get(
    "/users",
    summary="List users (mock)",
    description="List all users - mock implementation for testing",
)
async def list_users(current_user: User = Depends(get_current_superuser)) -> list[dict]:
    """
    List all users - mock implementation.

    This is a mock endpoint for testing purposes. In a real implementation,
    this would query the user database and return actual user data.

    Args:
        current_user: Current superuser

    Returns:
        list[dict]: List of users
    """
    # Mock user data
    mock_users = [
        {
            "id": "1",
            "username": "admin",
            "email": "admin@example.com",
            "full_name": "Administrator",
            "is_active": True,
            "is_superuser": True,
        },
        {
            "id": "2",
            "username": "user",
            "email": "user@example.com",
            "full_name": "Regular User",
            "is_active": True,
            "is_superuser": False,
        },
    ]

    return mock_users
