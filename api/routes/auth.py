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
from uuid import UUID

from loguru import logger
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from api.dependencies import get_redis, get_user_service
from core.models import UserCreate
from core.exceptions import DatabaseConnectionError
from infrastructure.audit_logger import (
    AuditEventType,
    AuditSeverity,
    get_audit_logger,
)
from infrastructure.token_blacklist import get_token_blacklist
from security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    Token,
    User,
    create_access_token,
    decode_access_token,
    get_current_active_user,
    get_current_superuser,
    get_current_user,
    normalize_managed_user_role,
    oauth2_scheme,
)
from services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Custom form class for better compatibility
class LoginForm(BaseModel):
    username: str
    password: str


@router.post(
    "/logout",
    summary="Logout and invalidate token",
    description="Invalidate the current access token to securely logout",
)
async def logout(
    token: str = Depends(oauth2_scheme),
) -> dict:
    """
    Logout and invalidate the current access token.

    This endpoint adds the token's JTI to a Redis blacklist,
    preventing further use until it naturally expires.
    """
    try:
        # Decode to get JTI and expiration
        token_data = decode_access_token(token)

        if not token_data.jti:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token missing JTI claim - cannot invalidate"
            )

        # Calculate remaining TTL
        if token_data.expires_at:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            # Make expires_at timezone-aware if it isn't
            exp = token_data.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            remaining_seconds = int((exp - now).total_seconds())
        else:
            remaining_seconds = 3600  # Default 1 hour

        # Add to blacklist with graceful degradation (fail-open for UX)
        blacklist = get_token_blacklist()
        blacklist_success = False

        if blacklist:
            try:
                blacklist_success = await blacklist.add(
                    jti=token_data.jti,
                    exp_seconds=max(remaining_seconds, 1),
                    reason="user_logout"
                )
            except Exception as e:
                logger.error(f"Blacklist add failed during logout: {e}", exc_info=True)
        else:
            logger.warning("Token blacklist not available during logout")

        response = {
            "message": "Successfully logged out",
            "token_invalidated": blacklist_success,
        }

        if not blacklist_success:
            response["warning"] = "Token may remain valid until expiration. Close browser for security."
            logger.warning(f"Logout without blacklist | jti={token_data.jti} | ttl={remaining_seconds}s")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Logout error")
        return {
            "message": "Logged out (best effort)",
            "token_invalidated": False,
            "warning": "Logout processed but token may remain valid until expiration."
        }


@router.post(
    "/token",
    response_model=Token,
    summary="Generate access token",
    description="Authenticate user and generate JWT access token",
)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service = Depends(get_user_service),
) -> Token:
    """
    Generate access token for authenticated user with audit logging.
    """
    audit_logger = get_audit_logger(redis_client=get_redis())
    ip_address = request.client.host if request.client else None

    try:
        user = await user_service.authenticate_user(form_data.username, form_data.password)
    except DatabaseConnectionError as exc:
        logger.error(f"DB unreachable during login for {form_data.username}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service_unavailable",
            headers={"Retry-After": "30"},
        )

    if not user or not user.is_active:
        # AUDIT: Log failed login attempt (Fail-safe)
        try:
            await audit_logger.log_event(
                event_type=AuditEventType.LOGIN_FAILURE,
                action=f"Failed login attempt for user: {form_data.username}",
                user_email=form_data.username,
                ip_address=ip_address,
                severity=AuditSeverity.WARNING,
                success=False,
                error_message="Incorrect username or password",
            )
        except Exception as e:
            # Don't block login on audit failure
            pass

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

    # AUDIT: Log successful login (Fail-safe)
    try:
        await audit_logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action=f"User logged in successfully",
            user_id=str(user.id),
            user_email=user.email,
            ip_address=ip_address,
            severity=AuditSeverity.INFO,
            success=True,
        )
    except Exception as e:
        # Don't block login on audit failure
        pass

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
    description="Bootstrap the first manager account. Disabled after any user exists.",
)
async def register_user(
    user_create: UserCreate, user_service = Depends(get_user_service)
) -> User:
    """
    Bootstrap the first manager account only.

    Normal user creation must happen through the manager-only /auth/users
    endpoint. This keeps distributed desktop installs from becoming open
    registration surfaces.
    """
    if await user_service.count_users() > 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Registration is disabled. Ask a manager to create your account.",
        )

    # Check if user already exists
    existing_user = await user_service.get_user_by_email(user_create.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create first user and promote to manager.
    user = await user_service.create_user(user_create)
    promoted = await user_service.promote_to_superuser(UUID(str(user.id)))
    if not promoted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to bootstrap manager account",
        )
    user.is_superuser = True
    user.role = "manager"
    return user


@router.get(
    "/users/me",
    response_model=User,
    summary="Get current user",
    description="Get current authenticated user information",
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


# Alias route for backward compatibility (/auth/me -> /auth/users/me)
@router.get(
    "/me",
    response_model=User,
    summary="Get current user (alias)",
    description="Alias for /users/me endpoint",
    include_in_schema=False,  # Hide from API docs to avoid confusion
)
async def read_users_me_alias(current_user: User = Depends(get_current_active_user)) -> User:
    """Backward-compatible alias for read_users_me."""
    return current_user

@router.get("/verify", summary="Verify token", description="Verify if the provided token is valid")
async def verify_token(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Verify token validity and return expiration info.

    This endpoint can be used to check if a JWT token is still valid
    and retrieve its expiration time.

    Args:
        token: JWT token from request
        current_user: Current user from token validation

    Returns:
        dict: Token verification result with expiry info
    """
    # Decode token to get expiry info
    token_data = decode_access_token(token)

    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "expires_at": token_data.expires_at.isoformat() if token_data.expires_at else None,
        "issued_at": token_data.issued_at.isoformat() if token_data.issued_at else None,
        "token_type": token_data.token_type,
        "scopes": token_data.scopes,
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
            # Use email as canonical subject to match login and lookup semantics
            "sub": current_user.email,
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


# Admin user management endpoints


class AdminUserCreate(BaseModel):
    email: str
    password: str
    full_name: str = ""
    role: str = "user"  # "admin" | "user"


def _parse_user_id(user_id: str) -> UUID:
    try:
        return UUID(user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid user id",
        ) from exc


@router.post(
    "/users",
    summary="Create user (admin only)",
    description="Create a new user account. Admins can set the role to 'admin'.",
    status_code=201,
)
async def admin_create_user(
    body: AdminUserCreate,
    current_user: User = Depends(get_current_superuser),
    user_service = Depends(get_user_service),
) -> dict:
    """Admin-only endpoint to create a new user, optionally as superuser."""
    try:
        normalized_role = normalize_managed_user_role(body.role)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    from core.models import UserCreate as _UserCreate
    user_in = _UserCreate(
        email=body.email,
        password=body.password,
        full_name=body.full_name or None,
    )
    new_user = await user_service.create_user(user_in)
    # Promote to superuser if requested
    if normalized_role == "manager":
        promoted = await user_service.promote_to_superuser(UUID(str(new_user.id)))
        if not promoted:
            raise HTTPException(status_code=404, detail="User not found after creation")
        new_user.is_superuser = True
    return {
        "id": new_user.id,
        "email": new_user.email,
        "full_name": new_user.full_name,
        "role": "manager" if new_user.is_superuser else "user",
        "is_active": new_user.is_active,
        "is_superuser": new_user.is_superuser,
        "created_at": new_user.created_at,
    }


@router.get(
    "/users",
    summary="List users",
    description="List all users (admin only)",
)
async def list_users(
    current_user: User = Depends(get_current_superuser),
    user_service = Depends(get_user_service),
) -> list[dict]:
    users = await user_service.list_users()
    return [
        {
            "id": u.id,
            "username": (
                u.username
                if hasattr(u, "username") and u.username
                else (u.email.split("@")[0] if u.email else None)
            ),
            "email": u.email,
            "full_name": u.full_name,
            "role": "manager" if u.is_superuser else "user",
            "is_active": u.is_active,
            "is_superuser": u.is_superuser,
            "created_at": u.created_at,
        }
        for u in users
    ]


@router.post(
    "/users/{user_id}/activate",
    summary="Activate user",
    description="Activate a user account (admin only)",
)
async def activate_user(
    user_id: str,
    current_user: User = Depends(get_current_superuser),
    user_service = Depends(get_user_service),
) -> dict:
    """Activate a user account."""
    target_id = _parse_user_id(user_id)
    success = await user_service.activate_user(target_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User activated", "user_id": user_id}


@router.post(
    "/users/{user_id}/deactivate",
    summary="Deactivate user",
    description="Deactivate a user account (admin only)",
)
async def deactivate_user(
    user_id: str,
    current_user: User = Depends(get_current_superuser),
    user_service = Depends(get_user_service),
) -> dict:
    """Deactivate a user account."""
    target_id = _parse_user_id(user_id)
    if str(target_id) == str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Managers cannot deactivate their own account.",
        )

    target_user = await user_service.get_user_by_id(target_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    if target_user.is_superuser and await user_service.count_users(
        is_active=True, is_superuser=True
    ) <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate the last active manager.",
        )

    success = await user_service.deactivate_user(target_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deactivated", "user_id": user_id}
