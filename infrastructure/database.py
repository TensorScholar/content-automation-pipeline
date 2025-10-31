"""
Database Infrastructure & Connection Management
================================================
Async PostgreSQL client with:
- Connection pooling (SQLAlchemy + asyncpg)
- Health monitoring and automatic reconnection
- pgvector extension integration
- Transaction context managers
- Query performance tracking

Architecture: Repository Pattern + Unit of Work
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.pool.impl import AsyncAdaptedQueuePool

from config.settings import get_settings
from core.exceptions import DatabaseConnectionError, DatabaseQueryTimeoutError

# Initialize logger
logger = logging.getLogger(__name__)

# SQLAlchemy declarative base
Base = declarative_base()


class DatabaseManager:
    """
    Centralized database connection and session management.

    Implements singleton pattern for engine lifecycle management
    with automatic health monitoring and reconnection logic.
    """

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._is_initialized: bool = False
        self._settings = get_settings()

    async def initialize(self) -> None:
        """
        Initialize database engine and session factory.

        Creates connection pool and registers event listeners.
        Must be called during application startup.
        """
        if self._is_initialized:
            logger.warning("Database already initialized")
            return

        try:
            # Create async engine with optimized pooling
            self._engine = create_async_engine(
                self._settings.database.async_url,
                echo=self._settings.database.echo_sql,
                pool_size=self._settings.database.pool_size,
                max_overflow=self._settings.database.max_overflow,
                pool_timeout=self._settings.database.pool_timeout,
                pool_recycle=self._settings.database.pool_recycle,
                pool_pre_ping=True,  # Verify connections before use
                poolclass=AsyncAdaptedQueuePool,
            )

            # Register connection event listeners
            self._register_events()

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Prevent lazy loading after commit
            )

            # Verify connection
            await self.health_check()

            self._is_initialized = True
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseConnectionError(
                "Failed to initialize database connection",
                host=self._settings.database.host,
                database=self._settings.database.database,
                cause=e,
            ) from e

    async def close(self) -> None:
        """
        Close database connections and dispose engine.

        Should be called during application shutdown.
        """
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._is_initialized = False
            logger.info("Database connections closed")

    def _register_events(self) -> None:
        """Register SQLAlchemy event listeners for monitoring."""
        if not self._engine:
            return

        @event.listens_for(self._engine.sync_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new connections."""
            logger.debug("New database connection established")

        @event.listens_for(self._engine.sync_engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Track connection checkouts from pool."""
            logger.debug("Connection checked out from pool")

        @event.listens_for(self._engine.sync_engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Track connection returns to pool."""
            logger.debug("Connection returned to pool")

    async def health_check(self) -> bool:
        """
        Verify database connectivity and pgvector extension.

        Returns:
            True if healthy, raises exception otherwise
        """
        if not self._engine:
            raise DatabaseConnectionError("Database engine not initialized")

        try:
            async with self._engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

                # Verify pgvector extension
                result = await conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1")
                )
                has_pgvector = result.scalar() == 1

                if not has_pgvector:
                    logger.warning("pgvector extension not installed")
                else:
                    logger.debug("pgvector extension verified")

                return True

        except (OperationalError, DBAPIError) as e:
            logger.error(f"Database health check failed: {e}")
            raise DatabaseConnectionError("Database health check failed", cause=e) from e

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide async database session with automatic cleanup.

        Usage:
            async with db_manager.session() as session:
                result = await session.execute(query)
                await session.commit()

        Yields:
            AsyncSession: Database session
        """
        if not self._session_factory:
            raise DatabaseConnectionError("Database not initialized")

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Session error, rolled back: {e}")
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide transactional context with automatic rollback on error.

        Similar to session() but emphasizes transactional semantics.
        """
        async with self.session() as session:
            try:
                yield session
            except Exception:
                # Rollback handled by session context manager
                raise

    async def execute_raw(self, query: str, params: Optional[dict] = None) -> Any:
        """
        Execute raw SQL query with parameter binding.

        Args:
            query: SQL query string
            params: Optional parameter dictionary

        Returns:
            Query result
        """
        async with self.session() as session:
            try:
                result = await session.execute(text(query), params or {})
                return result
            except Exception as e:
                logger.error(f"Raw query execution failed: {e}")
                raise DatabaseQueryTimeoutError(
                    "Query execution failed", query_preview=query[:200], cause=e
                ) from e

    @property
    def engine(self) -> AsyncEngine:
        """Get database engine (raises if not initialized)."""
        if not self._engine:
            raise DatabaseConnectionError("Database engine not initialized")
        return self._engine


# =============================================================================
# GLOBAL DATABASE MANAGER INSTANCE
# =============================================================================

# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get or create global database manager instance.

    Returns:
        DatabaseManager: Singleton instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def init_database() -> None:
    """
    Initialize database connection pool.

    Should be called once during application startup.
    """
    manager = get_db_manager()
    await manager.initialize()


async def close_database() -> None:
    """
    Close database connections.

    Should be called during application shutdown.
    """
    manager = get_db_manager()
    await manager.close()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Convenience function for getting database session.

    Usage:
        async with get_session() as session:
            # Use session
    """
    manager = get_db_manager()
    async with manager.session() as session:
        yield session


async def ensure_pgvector_extension() -> None:
    """
    Ensure pgvector extension is installed.

    Should be called during database setup/migration.
    """
    manager = get_db_manager()

    try:
        async with manager.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector extension ensured")
    except Exception as e:
        logger.error(f"Failed to create pgvector extension: {e}")
        raise DatabaseConnectionError("Failed to ensure pgvector extension", cause=e) from e


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "Base",
    "DatabaseManager",
    "get_db_manager",
    "init_database",
    "close_database",
    "get_session",
    "ensure_pgvector_extension",
]
