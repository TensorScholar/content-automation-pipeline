"""
User Service: Business Logic Layer for User Management
=====================================================

Orchestrates user-related business operations and coordinates between
the repository layer and external services.

Architecture: Service Layer Pattern + Business Logic Encapsulation
"""

from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger

from core.models import UserCreate, UserInDB, UserUpdate
from knowledge.user_repository import UserRepository
from security import User, get_password_hash, verify_password


class UserService:
    """
    Service for user business logic operations.

    Encapsulates user-related business rules and coordinates
    between repository and security layers.
    """

    def __init__(self, user_repository: UserRepository):
        """
        Initialize service with user repository.

        Args:
            user_repository: Repository for user data access
        """
        self.users = user_repository
        logger.debug("UserService initialized")

    async def create_user(self, user_in: UserCreate) -> User:
        """
        Create a new user with hashed password.

        Args:
            user_in: User creation data

        Returns:
            Created User instance (public model)

        Raises:
            HTTPException: If user creation fails
        """
        try:
            # Hash the password
            hashed_password = get_password_hash(user_in.password)

            # Create user in database with hashed password
            user_in_db = await self.users.create(user_in, hashed_password)

            # Fetch complete user data
            complete_user = await self.users.get_by_id(user_in_db.id)
            if not complete_user:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve created user",
                )

            # Convert to public model
            user = User(
                id=str(complete_user.id),
                username=complete_user.email.split("@")[0],  # Derive username from email
                email=complete_user.email,
                full_name=complete_user.full_name,
                is_active=complete_user.is_active,
                is_superuser=complete_user.is_superuser,
                created_at=complete_user.created_at,
                last_login=None,
            )

            logger.info(f"User created successfully: {user.email}")
            return user

        except ValueError as e:
            logger.warning(f"User creation failed - validation error: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user"
            )

    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Get user by ID (public model).

        Args:
            user_id: Unique identifier for the user

        Returns:
            User instance if found, None otherwise
        """
        try:
            user_in_db = await self.users.get_by_id(user_id)
            if not user_in_db:
                return None

            return User(
                id=user_in_db.id,
                email=user_in_db.email,
                full_name=user_in_db.full_name,
                is_active=user_in_db.is_active,
                is_superuser=user_in_db.is_superuser,
                created_at=user_in_db.created_at,
                updated_at=user_in_db.updated_at,
            )

        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve user"
            )

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Get user by email (internal model with hashed password).

        Args:
            email: User's email address

        Returns:
            UserInDB instance if found, None otherwise
        """
        try:
            return await self.users.get_by_email(email)

        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve user"
            )

    async def update_user(self, user_id: UUID, user_update: UserUpdate) -> Optional[User]:
        """
        Update user information.

        Args:
            user_id: Unique identifier for the user
            user_update: Update data

        Returns:
            Updated User instance if found, None otherwise

        Raises:
            HTTPException: If update fails
        """
        try:
            updated_user_in_db = await self.users.update(user_id, user_update)
            if not updated_user_in_db:
                return None

            return User(
                id=updated_user_in_db.id,
                email=updated_user_in_db.email,
                full_name=updated_user_in_db.full_name,
                is_active=updated_user_in_db.is_active,
                is_superuser=updated_user_in_db.is_superuser,
                created_at=updated_user_in_db.created_at,
                updated_at=updated_user_in_db.updated_at,
            )

        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update user"
            )

    async def delete_user(self, user_id: UUID) -> bool:
        """
        Delete user by ID.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if user was deleted, False if not found
        """
        try:
            return await self.users.delete(user_id)

        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete user"
            )

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        is_active: Optional[bool] = None,
        is_superuser: Optional[bool] = None,
    ) -> List[User]:
        """
        List users with optional filtering (public models).

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            is_active: Filter by active status
            is_superuser: Filter by superuser status

        Returns:
            List of User instances
        """
        try:
            users_in_db = await self.users.list_users(
                skip=skip, limit=limit, is_active=is_active, is_superuser=is_superuser
            )

            return [
                User(
                    id=user.id,
                    email=user.email,
                    full_name=user.full_name,
                    is_active=user.is_active,
                    is_superuser=user.is_superuser,
                    created_at=user.created_at,
                    updated_at=user.updated_at,
                )
                for user in users_in_db
            ]

        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list users"
            )

    async def authenticate_user(self, username_or_email: str, password: str) -> Optional[UserInDB]:
        """
        Authenticate user with username or email and password.

        Args:
            username_or_email: User's username or email address
            password: Plain text password

        Returns:
            UserInDB instance if authentication succeeds, None otherwise
        """
        try:
            # Try to find user by email first
            user = await self.get_user_by_email(username_or_email)
            if not user:
                # If not found by email and it looks like a username, try to find by email prefix
                if "@" not in username_or_email:
                    # This looks like a username, try to find by email prefix
                    # We'll need to query the database directly for this
                    user = await self.users.get_by_username_prefix(username_or_email)

            if not user:
                logger.debug(f"Authentication failed - user not found: {username_or_email}")
                return None

            if not user.is_active:
                logger.debug(f"Authentication failed - user inactive: {username_or_email}")
                return None

            if not verify_password(password, user.hashed_password):
                logger.debug(f"Authentication failed - invalid password: {username_or_email}")
                return None

            logger.info(f"User authenticated successfully: {username_or_email}")
            return user

        except Exception as e:
            logger.error(f"Authentication failed for {username_or_email}: {e}")
            return None

    async def change_password(self, user_id: UUID, old_password: str, new_password: str) -> bool:
        """
        Change user's password.

        Args:
            user_id: Unique identifier for the user
            old_password: Current password for verification
            new_password: New password

        Returns:
            True if password was changed, False if old password is incorrect

        Raises:
            HTTPException: If user not found or change fails
        """
        try:
            # Get user and verify old password
            user = await self.users.get_by_id(user_id)
            if not user:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

            if not verify_password(old_password, user.hashed_password):
                logger.debug(f"Password change failed - invalid old password: {user_id}")
                return False

            # Hash new password and update
            new_hashed_password = get_password_hash(new_password)
            success = await self.users.update_password(user_id, new_hashed_password)

            if success:
                logger.info(f"Password changed successfully: {user.email}")

            return success

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to change password for user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change password",
            )

    async def deactivate_user(self, user_id: UUID) -> bool:
        """
        Deactivate user account.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if user was deactivated, False if not found
        """
        try:
            user_update = UserUpdate(is_active=False)
            updated_user = await self.update_user(user_id, user_update)

            if updated_user:
                logger.info(f"User deactivated: {updated_user.email}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to deactivate user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to deactivate user",
            )

    async def activate_user(self, user_id: UUID) -> bool:
        """
        Activate user account.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if user was activated, False if not found
        """
        try:
            user_update = UserUpdate(is_active=True)
            updated_user = await self.update_user(user_id, user_update)

            if updated_user:
                logger.info(f"User activated: {updated_user.email}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to activate user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to activate user"
            )

    async def promote_to_superuser(self, user_id: UUID) -> bool:
        """
        Promote user to superuser.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if user was promoted, False if not found
        """
        try:
            user_update = UserUpdate(is_superuser=True)
            updated_user = await self.update_user(user_id, user_update)

            if updated_user:
                logger.info(f"User promoted to superuser: {updated_user.email}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to promote user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to promote user"
            )

    async def demote_from_superuser(self, user_id: UUID) -> bool:
        """
        Demote user from superuser.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if user was demoted, False if not found
        """
        try:
            user_update = UserUpdate(is_superuser=False)
            updated_user = await self.update_user(user_id, user_update)

            if updated_user:
                logger.info(f"User demoted from superuser: {updated_user.email}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to demote user {user_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to demote user"
            )
