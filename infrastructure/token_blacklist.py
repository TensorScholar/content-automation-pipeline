"""Redis-backed JWT Token Blacklist for Secure Logout.

Provides a distributed token blacklist using Redis with automatic expiration.
Tokens are stored by their JTI (JWT ID) claim until their natural expiration.

Usage:
    blacklist = TokenBlacklist(redis_client)
    await blacklist.add(jti="token-id-123", exp_seconds=3600)
    is_revoked = await blacklist.is_blacklisted("token-id-123")
"""

from datetime import datetime, timezone
from typing import Optional

from infrastructure.monitoring import get_logger

logger = get_logger(__name__)


class TokenBlacklist:
    """Redis-backed token blacklist for JWT invalidation.

    Supports secure logout by tracking revoked tokens until they naturally expire.
    Uses Redis with automatic key expiration for memory efficiency.
    """

    PREFIX = "auth:token:blacklist:"

    def __init__(self, redis_client):
        """Initialize with Redis client.

        Args:
            redis_client: RedisClient instance for storage operations.
        """
        self._redis = redis_client

    async def add(
        self,
        jti: str,
        exp_seconds: int,
        reason: str = "logout"
    ) -> bool:
        """Add token JTI to blacklist with automatic expiry.

        Args:
            jti: JWT ID (unique token identifier from 'jti' claim)
            exp_seconds: Seconds until token natural expiration
            reason: Reason for blacklisting (logout, password_change, etc.)

        Returns:
            True if successfully added, False otherwise.
        """
        if not jti or exp_seconds <= 0:
            logger.warning("Invalid blacklist parameters", jti=jti, exp_seconds=exp_seconds)
            return False

        key = f"{self.PREFIX}{jti}"
        value = {
            "blacklisted_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }

        try:
            # Store with TTL matching token expiration
            # No need to keep blacklist entry after token would expire anyway
            await self._redis.set(key, value, ex=exp_seconds)
            logger.info(
                "Token blacklisted",
                jti=jti,
                reason=reason,
                expires_in_seconds=exp_seconds
            )
            return True
        except Exception as e:
            logger.error("Failed to blacklist token", jti=jti, error=str(e))
            return False

    async def is_blacklisted(self, jti: str) -> bool:
        """Check if token JTI is in the blacklist.

        Args:
            jti: JWT ID to check.

        Returns:
            True if token is blacklisted (revoked), False otherwise.
        """
        if not jti:
            return False

        key = f"{self.PREFIX}{jti}"

        try:
            exists = await self._redis.exists(key)
            if exists:
                logger.debug("Blacklisted token access attempted", jti=jti)
            return exists
        except Exception as e:
            logger.error("Failed to check token blacklist", jti=jti, error=str(e))
            # Fail open for availability, but log for monitoring
            # In high-security scenarios, you might want to fail closed
            return False

    async def remove(self, jti: str) -> bool:
        """Remove token from blacklist (for admin recovery).

        Args:
            jti: JWT ID to remove from blacklist.

        Returns:
            True if removed, False if not found or error.
        """
        key = f"{self.PREFIX}{jti}"

        try:
            deleted = await self._redis.delete(key)
            if deleted:
                logger.info("Token removed from blacklist", jti=jti)
            return deleted
        except Exception as e:
            logger.error("Failed to remove token from blacklist", jti=jti, error=str(e))
            return False

    async def get_blacklist_info(self, jti: str) -> Optional[dict]:
        """Get blacklist entry details.

        Args:
            jti: JWT ID to look up.

        Returns:
            Dict with blacklist details or None if not found.
        """
        key = f"{self.PREFIX}{jti}"

        try:
            return await self._redis.get(key)
        except Exception:
            return None


# Global singleton instance
_token_blacklist: Optional[TokenBlacklist] = None


def get_token_blacklist() -> Optional[TokenBlacklist]:
    """Get the global token blacklist instance.

    Returns:
        TokenBlacklist instance or None if not initialized.
    """
    return _token_blacklist


def init_token_blacklist(redis_client) -> TokenBlacklist:
    """Initialize the global token blacklist.

    Args:
        redis_client: RedisClient instance.

    Returns:
        Initialized TokenBlacklist instance.
    """
    global _token_blacklist
    _token_blacklist = TokenBlacklist(redis_client)
    logger.info("Token blacklist initialized")
    return _token_blacklist
