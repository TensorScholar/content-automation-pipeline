"""
User Repository: Database Access Layer for User Management
=========================================================

Provides type-safe database operations for user entities using SQLAlchemy Core.
Implements the Repository pattern for clean separation of concerns.

Architecture: Repository Pattern + SQLAlchemy Core
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import User, UserCreate, UserInDB, UserUpdate
from infrastructure.database import DatabaseManager
from infrastructure.schema import users_table


class UserRepository:
    """
    Repository for user data access operations.

    Provides async methods for CRUD operations on user entities.
    All methods are type-safe and use SQLAlchemy Core for performance.
    """

    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize repository with database manager.

        Args:
            database_manager: Database manager for session management
        """
        self.database_manager = database_manager
        logger.debug("UserRepository initialized")

    async def get_by_id(self, user_id: UUID) -> Optional[UserInDB]:
        """
        Retrieve user by ID.

        Args:
            user_id: Unique identifier for the user

        Returns:
            UserInDB instance if found, None otherwise
        """
        try:
            async with self.database_manager.session() as session:
                query = select(users_table).where(users_table.c.id == user_id)
                result = await session.execute(query)
                row = result.fetchone()

                if row is None:
                    logger.debug(f"User not found: {user_id}")
                    return None

                # Convert asyncpg UUID to Python UUID for proper serialization
                user_data = row._asdict()
                if "id" in user_data:
                    from uuid import UUID

                    user_data["id"] = UUID(str(user_data["id"]))
                return UserInDB(**user_data)

        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            raise

    async def get_by_username_prefix(self, username_prefix: str) -> Optional[UserInDB]:
        """
        Retrieve user by username prefix (email prefix).

        Args:
            username_prefix: Username prefix to search for

        Returns:
            UserInDB instance if found, None otherwise
        """
        try:
            async with self.database_manager.session() as session:
                # Search for users where email starts with username_prefix@
                query = select(users_table).where(users_table.c.email.like(f"{username_prefix}@%"))
                result = await session.execute(query)
                row = result.fetchone()

                if row is None:
                    logger.debug(f"User not found with username prefix: {username_prefix}")
                    return None

                # Convert asyncpg UUID to Python UUID for proper serialization
                user_data = row._asdict()
                if "id" in user_data:
                    from uuid import UUID

                    user_data["id"] = UUID(str(user_data["id"]))
                return UserInDB(**user_data)

        except Exception as e:
            logger.error(f"Failed to get user by username prefix {username_prefix}: {e}")
            raise

    async def get_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Retrieve user by email address.

        Args:
            email: User's email address

        Returns:
            UserInDB instance if found, None otherwise
        """
        try:
            async with self.database_manager.session() as session:
                query = select(users_table).where(users_table.c.email == email.lower().strip())
                result = await session.execute(query)
                row = result.fetchone()

                if row is None:
                    logger.debug(f"User not found: {email}")
                    return None

                # Convert asyncpg UUID to Python UUID for proper serialization
                user_data = row._asdict()
                if "id" in user_data:
                    from uuid import UUID

                    user_data["id"] = UUID(str(user_data["id"]))
                return UserInDB(**user_data)

        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            raise

    async def create(self, user_in: UserCreate, hashed_password: str) -> UserInDB:
        """
        Create a new user.

        Args:
            user_in: User creation data
            hashed_password: Pre-hashed password

        Returns:
            Created UserInDB instance

        Raises:
            ValueError: If email already exists
        """
        try:
            # Check if user already exists
            existing_user = await self.get_by_email(user_in.email)
            if existing_user:
                raise ValueError(f"User with email {user_in.email} already exists")

            async with self.database_manager.session() as session:
                # Prepare user data
                user_data = {
                    "email": user_in.email.lower().strip(),
                    "hashed_password": hashed_password,
                    "full_name": user_in.full_name,
                    "is_active": True,
                    "is_superuser": False,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }

                # Insert user with hashed password
                query = insert(users_table).values(**user_data).returning(users_table)
                result = await session.execute(query)
                await session.commit()
                row = result.fetchone()

                # Convert to UserInDB
                user_in_db = UserInDB(**row._asdict())
                logger.info(f"User created: {user_in_db.email}")

                return user_in_db

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to create user {user_in.email}: {e}")
            raise

    async def update(self, user_id: UUID, user_update: UserUpdate) -> Optional[UserInDB]:
        """
        Update user information.

        Args:
            user_id: Unique identifier for the user
            user_update: Update data

        Returns:
            Updated UserInDB instance if found, None otherwise
        """
        try:
            # Build update data from non-None fields
            update_data = {}
            if user_update.email is not None:
                update_data["email"] = user_update.email.lower().strip()
            if user_update.full_name is not None:
                update_data["full_name"] = user_update.full_name
            if user_update.is_active is not None:
                update_data["is_active"] = user_update.is_active
            if user_update.is_superuser is not None:
                update_data["is_superuser"] = user_update.is_superuser

            if not update_data:
                logger.debug(f"No update data provided for user {user_id}")
                return await self.get_by_id(user_id)

            # Add timestamp
            update_data["updated_at"] = datetime.utcnow()

            async with self.database_manager.session() as session:
                # Perform update
                query = (
                    update(users_table)
                    .where(users_table.c.id == user_id)
                    .values(**update_data)
                    .returning(users_table)
                )
                result = await session.execute(query)
                await session.commit()
                row = result.fetchone()

                if row is None:
                    logger.debug(f"User not found for update: {user_id}")
                    return None

                updated_user = UserInDB(**row._asdict())
                logger.info(f"User updated: {updated_user.email}")

                return updated_user

        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            raise

    async def delete(self, user_id: UUID) -> bool:
        """
        Delete user by ID.

        Args:
            user_id: Unique identifier for the user

        Returns:
            True if user was deleted, False if not found
        """
        try:
            async with self.database_manager.session() as session:
                query = delete(users_table).where(users_table.c.id == user_id)
                result = await session.execute(query)
                await session.commit()

                if result.rowcount == 0:
                    logger.debug(f"User not found for deletion: {user_id}")
                    return False

                logger.info(f"User deleted: {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            raise

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        is_active: Optional[bool] = None,
        is_superuser: Optional[bool] = None,
    ) -> List[UserInDB]:
        """
        List users with optional filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            is_active: Filter by active status
            is_superuser: Filter by superuser status

        Returns:
            List of UserInDB instances
        """
        try:
            async with self.database_manager.session() as session:
                query = select(users_table)

                # Apply filters
                if is_active is not None:
                    query = query.where(users_table.c.is_active == is_active)
                if is_superuser is not None:
                    query = query.where(users_table.c.is_superuser == is_superuser)

                # Apply pagination
                query = query.offset(skip).limit(limit)

                # Order by creation date
                query = query.order_by(users_table.c.created_at.desc())

                result = await session.execute(query)
                rows = result.fetchall()

                users = [UserInDB(**row._asdict()) for row in rows]
                logger.debug(f"Listed {len(users)} users")

                return users

        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            raise

    async def count_users(
        self, is_active: Optional[bool] = None, is_superuser: Optional[bool] = None
    ) -> int:
        """
        Count users with optional filtering.

        Args:
            is_active: Filter by active status
            is_superuser: Filter by superuser status

        Returns:
            Number of users matching criteria
        """
        try:
            from sqlalchemy import func

            async with self.database_manager.session() as session:
                query = select(func.count(users_table.c.id))

                # Apply filters
                if is_active is not None:
                    query = query.where(users_table.c.is_active == is_active)
                if is_superuser is not None:
                    query = query.where(users_table.c.is_superuser == is_superuser)

                result = await session.execute(query)
                count = result.scalar()

                logger.debug(f"User count: {count}")
                return count

        except Exception as e:
            logger.error(f"Failed to count users: {e}")
            raise

    async def update_password(self, user_id: UUID, hashed_password: str) -> bool:
        """
        Update user's hashed password.

        Args:
            user_id: Unique identifier for the user
            hashed_password: New hashed password

        Returns:
            True if password was updated, False if user not found
        """
        try:
            async with self.database_manager.session() as session:
                query = (
                    update(users_table)
                    .where(users_table.c.id == user_id)
                    .values(hashed_password=hashed_password, updated_at=datetime.utcnow())
                )
                result = await session.execute(query)
                await session.commit()

                if result.rowcount == 0:
                    logger.debug(f"User not found for password update: {user_id}")
                    return False

                logger.info(f"Password updated for user: {user_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to update password for user {user_id}: {e}")
            raise
