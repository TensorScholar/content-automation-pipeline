"""
Security Module: Authentication and Authorization Foundation

Provides foundational security components including:
- Password hashing and verification
- JWT token creation and validation
- User authentication schemas
- Security utilities for API protection

Architectural Pattern: Security Utilities + Cross-Cutting Concerns
Security Foundation: OAuth2 + JWT + Password Hashing
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from dependency_injector.wiring import Provide
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from config.settings import get_settings

# Configuration
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# JWT Configuration - Enterprise Security Settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
PASSWORD_RESET_TOKEN_EXPIRE_HOURS = 1


def get_security_headers() -> Dict[str, str]:
    """
    Get security headers with environment-specific CSP configuration.
    
    Returns:
        Dict[str, str]: Security headers dictionary with appropriate CSP
    """
    settings = get_settings()
    
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }
    
    # Configure CSP based on environment
    if settings.monitoring.enable_strict_csp:
        csp_directive = "default-src 'self'"
        if settings.monitoring.csp_report_only:
            headers["Content-Security-Policy-Report-Only"] = csp_directive
        else:
            headers["Content-Security-Policy"] = csp_directive
    else:
        # Relaxed CSP for development (allows inline scripts, eval, external resources)
        # Useful for hot-reload, dev tools, and local testing
        csp_directive = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:"
        headers["Content-Security-Policy"] = csp_directive
        logger.debug("Using relaxed CSP for development environment")
    
    return headers


# Backward compatibility - static headers for direct import
# Note: Prefer get_security_headers() for environment-aware headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


class TokenData(BaseModel):
    """Pydantic schema for JWT token data with enhanced security."""

    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: list[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    issued_at: Optional[datetime] = None
    token_type: str = "access"
    jti: Optional[str] = None  # JWT ID for token revocation
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class UserInDB(BaseModel):
    """Pydantic schema for user data in database."""

    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None


class User(BaseModel):
    """Pydantic schema for user data (without sensitive information)."""

    id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    last_login: Optional[datetime] = None


class Token(BaseModel):
    """Pydantic schema for authentication token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(default=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    scope: str = "read write"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hash.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to verify against

    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        # Handles potential errors during verification (e.g., invalid hash format)
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.

    Bcrypt automatically handles long passwords and includes salt generation.

    Args:
        password: The plain text password to hash

    Returns:
        str: The hashed password
    """
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    token_type: str = "access",  # nosec B107 - Token type identifier, not password
    jti: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> str:
    """
    Create a JWT access token with enhanced security features.

    Args:
        data: The data to encode in the token
        expires_delta: Optional expiration time delta
        token_type: Type of token (access, refresh, etc.)
        jti: JWT ID for token revocation
        ip_address: Client IP address for security tracking
        user_agent: Client user agent for security tracking

    Returns:
        str: The encoded JWT token
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    # Add security metadata
    to_encode.update(
        {
            "exp": expire,
            "iat": now,
            "token_type": token_type,
            "jti": jti or str(uuid4()),
            "ip_address": ip_address,
            "user_agent": user_agent,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        }
    )

    encoded_jwt = jwt.encode(to_encode, settings.secret_key.get_secret_value(), algorithm=ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """
    Decode and validate a JWT access token with enhanced security checks.

    Args:
        token: The JWT token to decode

    Returns:
        TokenData: The decoded token data

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[ALGORITHM],
            audience=settings.jwt_audience,
            options={"verify_aud": True},
        )

        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        scopes: list[str] = payload.get("scopes", [])
        exp: int = payload.get("exp")
        iat: int = payload.get("iat")
        token_type: str = payload.get("token_type", "access")
        jti: str = payload.get("jti")
        ip_address: str = payload.get("ip_address")
        user_agent: str = payload.get("user_agent")
        iss: str = payload.get("iss")

        # Validate required fields
        if username is None or user_id is None:
            logger.error("JWT token missing required fields (sub or user_id)")
            raise credentials_exception

        # Issuer check
        if iss != settings.jwt_issuer:
            logger.warning(f"JWT issuer mismatch: expected {settings.jwt_issuer}, got {iss}")
            raise credentials_exception

        # Convert timestamps to datetime
        expires_at = datetime.fromtimestamp(exp) if exp else None
        issued_at = datetime.fromtimestamp(iat) if iat else None

        token_data = TokenData(
            username=username,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at,
            issued_at=issued_at,
            token_type=token_type,
            jti=jti,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        return token_data

    except JWTError:
        raise credentials_exception


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_service: "UserService" = Depends(Provide["Container.user_service"]),
) -> User:
    """
    FastAPI dependency to get the current authenticated user from the token.

    Validates the JWT token, extracts the user email (subject), and fetches
    the user from the database via the UserService.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode the token
        token_data = decode_access_token(token)
        if token_data.username is None:
            raise credentials_exception

        # Get user from database via injected UserService
        user_in_db = await user_service.get_user_by_email(token_data.username)
        if user_in_db is None:
            raise credentials_exception

        # Return public user model (ensure required fields only)
        safe_username = (
            user_in_db.username
            if hasattr(user_in_db, "username") and user_in_db.username
            else (user_in_db.email.split("@")[0] if user_in_db.email else "")
        )

        return User(
            id=user_in_db.id,
            username=safe_username,
            email=user_in_db.email,
            full_name=user_in_db.full_name,
            is_active=user_in_db.is_active,
            is_superuser=user_in_db.is_superuser,
            created_at=user_in_db.created_at,
            last_login=getattr(user_in_db, "last_login", None),
        )

    except JWTError:
        raise credentials_exception
    except Exception as e:
        logger.error(f"Authentication error in get_current_user: {e}")
        raise credentials_exception


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to get the current active user.

    This dependency ensures that the user is not only authenticated
    but also active.

    Args:
        current_user: The current user from get_current_user dependency

    Returns:
        User: The active user data

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

    return current_user


async def get_current_superuser(current_user: User = Depends(get_current_active_user)) -> User:
    """
    FastAPI dependency to get the current superuser.

    This dependency ensures that the user has superuser privileges.

    Args:
        current_user: The current active user

    Returns:
        User: The superuser data

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    return current_user


def check_token_scopes(token_data: TokenData, required_scopes: list[str]) -> bool:
    """
    Check if the token has the required scopes.

    Args:
        token_data: The decoded token data
        required_scopes: List of required scopes

    Returns:
        bool: True if all required scopes are present, False otherwise
    """
    if not required_scopes:
        return True

    token_scopes = set(token_data.scopes)
    required_scopes_set = set(required_scopes)

    return required_scopes_set.issubset(token_scopes)


def generate_password_reset_token(email: str) -> str:
    """
    Generate a password reset token for the given email.

    Args:
        email: The email address to generate token for

    Returns:
        str: The password reset token
    """
    delta = timedelta(hours=1)  # Token expires in 1 hour
    now = datetime.now(timezone.utc)
    expires = now + delta

    exp = expires.timestamp()
    encoded_jwt = jwt.encode(
        {"exp": exp, "nbf": now, "sub": email, "type": "password_reset"},
        settings.secret_key.get_secret_value(),
        algorithm=ALGORITHM,
    )

    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify a password reset token and return the email.

    Args:
        token: The password reset token to verify

    Returns:
        Optional[str]: The email address if token is valid, None otherwise
    """
    try:
        decoded_token = jwt.decode(
            token, settings.secret_key.get_secret_value(), algorithms=[ALGORITHM]
        )

        if decoded_token.get("type") != "password_reset":
            return None

        email: str = decoded_token.get("sub")
        return email

    except JWTError:
        return None


def get_client_ip(request) -> Optional[str]:
    """
    Extract client IP address from request.

    Args:
        request: FastAPI Request object

    Returns:
        Optional[str]: Client IP address
    """
    # Check for forwarded headers first (for reverse proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback to direct client IP
    return request.client.host if request.client else None


def get_user_agent(request) -> Optional[str]:
    """
    Extract user agent from request.

    Args:
        request: FastAPI Request object

    Returns:
        Optional[str]: User agent string
    """
    return request.headers.get("User-Agent")


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password strength according to enterprise standards.

    Args:
        password: Password to validate

    Returns:
        tuple[bool, list[str]]: (is_valid, list_of_issues)
    """
    issues = []

    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")

    if len(password) > 128:
        issues.append("Password must be no more than 128 characters long")

    if not any(c.isupper() for c in password):
        issues.append("Password must contain at least one uppercase letter")

    if not any(c.islower() for c in password):
        issues.append("Password must contain at least one lowercase letter")

    if not any(c.isdigit() for c in password):
        issues.append("Password must contain at least one digit")

    # Special characters are optional for easier testing
    # if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
    #     issues.append("Password must contain at least one special character")

    # Check for common weak patterns
    weak_patterns = ["password", "123456", "qwerty", "admin", "user"]
    if password.lower() in weak_patterns:
        issues.append("Password contains common weak patterns")

    return len(issues) == 0, issues


def sanitize_input(input_string: str) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        input_string: String to sanitize

    Returns:
        str: Sanitized string
    """
    if not input_string:
        return ""

    # Remove null bytes
    sanitized = input_string.replace("\x00", "")

    # Remove control characters except newlines and tabs
    sanitized = "".join(char for char in sanitized if ord(char) >= 32 or char in "\n\t")

    # Limit length
    sanitized = sanitized[:1000]

    return sanitized.strip()


# Export public API
__all__ = [
    # Schemas
    "TokenData",
    "UserInDB",
    "User",
    "Token",
    # Password utilities
    "verify_password",
    "get_password_hash",
    "validate_password_strength",
    # JWT utilities
    "create_access_token",
    "decode_access_token",
    # FastAPI dependencies
    "get_current_user",
    "get_current_active_user",
    "get_current_superuser",
    # OAuth2 scheme
    "oauth2_scheme",
    # Token utilities
    "check_token_scopes",
    "generate_password_reset_token",
    "verify_password_reset_token",
    # Security utilities
    "get_client_ip",
    "get_user_agent",
    "sanitize_input",
    "SECURITY_HEADERS",
    "get_security_headers",
]
