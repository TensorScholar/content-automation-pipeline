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

import asyncpg
from pgvector.asyncpg import register_vector
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings
from core.exceptions import DatabaseConnectionError, DatabaseQueryTimeoutError

# Connection pool monitoring and timeout enforcement
from contextlib import asynccontextmanager
import time

# Initialize logger
logger = logging.getLogger(__name__)

# SQLAlchemy declarative base
Base = declarative_base()


async def setup_connection(conn: asyncpg.Connection):
    """Register the pgvector type handler for asyncpg."""
    await register_vector(conn)


class DatabaseManager:
    """
    Centralized database connection and session management.

    Implements singleton pattern for engine lifecycle management
    with automatic health monitoring and reconnection logic.

    Supports separate writer (primary) and reader (replica) connection pools
    for horizontal read scalability.
    """

    def __init__(self):
        self._writer_engine: Optional[AsyncEngine] = None
        self._reader_engine: Optional[AsyncEngine] = None
        self._writer_session_factory: Optional[async_sessionmaker] = None
        self._reader_session_factory: Optional[async_sessionmaker] = None
        self._is_initialized: bool = False
        self._settings = get_settings()
        self._has_separate_replica: bool = False

    async def initialize(self) -> None:
        """
        Initialize database engines and session factories.

        Creates separate connection pools for writer (primary) and reader (replica).
        If no replica URL is configured, reader falls back to primary.
        Must be called during application startup.
        """
        if self._is_initialized:
            logger.warning("Database already initialized")
            return

        try:
            # Determine if we have a separate replica configured
            self._has_separate_replica = (
                self._settings.database.replica_url is not None
                and str(self._settings.database.replica_url) != str(self._settings.database.url)
            )

            # Create writer engine (primary database)
            self._writer_engine = create_async_engine(
                self._settings.database.async_url,
                echo=self._settings.database.echo_sql,
                pool_size=self._settings.database.pool_size,
                max_overflow=self._settings.database.max_overflow,
                pool_timeout=self._settings.database.pool_timeout,
                pool_recycle=self._settings.database.pool_recycle,
                pool_pre_ping=True,  # Verify connections before use
                poolclass=AsyncAdaptedQueuePool,
                # Disable asyncpg prepared statement cache to prevent stale schema errors
                # Global statement timeout (60s) as safety net for runaway queries
                connect_args={
                    "prepared_statement_cache_size": 0,
                    "server_settings": {"statement_timeout": "60000"},
                    "timeout": 10,
                },
            )

            # Create reader engine (replica or fallback to primary)
            if self._has_separate_replica:
                self._reader_engine = create_async_engine(
                    self._settings.database.async_replica_url,
                    echo=self._settings.database.echo_sql,
                    pool_size=self._settings.database.pool_size,
                    max_overflow=self._settings.database.max_overflow,
                    pool_timeout=self._settings.database.pool_timeout,
                    pool_recycle=self._settings.database.pool_recycle,
                    pool_pre_ping=True,
                    poolclass=AsyncAdaptedQueuePool,
                    connect_args={
                        "prepared_statement_cache_size": 0,
                        "server_settings": {"statement_timeout": "60000"},
                        "timeout": 10,
                    },
                )
                logger.info("Read replica configured - using separate connection pool")
            else:
                # Fallback: use writer engine for reads
                self._reader_engine = self._writer_engine
                logger.info("No read replica configured - using primary for all operations")

            # Register connection event listeners for both engines
            self._register_events(self._writer_engine, "writer")
            if self._has_separate_replica:
                self._register_events(self._reader_engine, "reader")

            # Create session factories
            self._writer_session_factory = async_sessionmaker(
                self._writer_engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Prevent lazy loading after commit
            )
            self._reader_session_factory = async_sessionmaker(
                self._reader_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Verify connections
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
        Close database connections and dispose engines.

        Should be called during application shutdown.
        """
        if self._writer_engine:
            await self._writer_engine.dispose()
            self._writer_engine = None

        if self._has_separate_replica and self._reader_engine:
            await self._reader_engine.dispose()
        self._reader_engine = None

        self._writer_session_factory = None
        self._reader_session_factory = None
        self._is_initialized = False
        logger.info("Database connections closed")

    def _register_events(self, engine: AsyncEngine, pool_name: str = "default") -> None:
        """Register SQLAlchemy event listeners for monitoring."""
        if not engine:
            return

        @event.listens_for(engine.sync_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new connections."""
            logger.debug(f"New database connection established ({pool_name})")

        @event.listens_for(engine.sync_engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Track connection checkouts from pool."""
            logger.debug(f"Connection checked out from pool ({pool_name})")

        @event.listens_for(engine.sync_engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Track connection returns to pool."""
            logger.debug(f"Connection returned to pool ({pool_name})")

    async def get_pool_stats(self) -> dict[str, any]:
        """Get current connection pool statistics for monitoring."""
        if not self._writer_engine:
            return {"error": "Engine not initialized"}

        pool = self._writer_engine.pool
        return {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.size() - pool.checkedout(),
            "total_capacity": pool.size() + self._settings.database.max_overflow,
        }

    @asynccontextmanager
    async def session_with_timeout(
        self,
        readonly: bool = False,
        timeout_seconds: float = 30.0
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager for sessions with enforced timeout.

        Prevents connection pool exhaustion from hanging queries.
        Automatically rolls back on timeout or error.

        Args:
            readonly: If True, use read replica (if configured)
            timeout_seconds: Maximum time to hold connection

        Yields:
            AsyncSession with timeout enforcement
        """
        if not self._is_initialized:
            raise DatabaseConnectionError("Database not initialized")

        factory = self._reader_session_factory if readonly else self._writer_session_factory
        if not factory:
            raise DatabaseConnectionError("Session factory not initialized")

        session = factory()
        start_time = time.time()

        try:
            # Set statement timeout at session level
            await session.execute(text(f"SET LOCAL statement_timeout = '{int(timeout_seconds * 1000)}'"))
            yield session

            # Check if we exceeded our timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    f"Session exceeded timeout: {elapsed:.2f}s > {timeout_seconds}s",
                    extra={"elapsed": elapsed, "timeout": timeout_seconds}
                )

            await session.commit()

        except asyncio.TimeoutError as e:
            await session.rollback()
            logger.error(f"Session timeout after {time.time() - start_time:.2f}s")
            raise DatabaseQueryTimeoutError(
                f"Database query timeout after {timeout_seconds}s",
                timeout=timeout_seconds
            ) from e
        except Exception as e:
            await session.rollback()
            logger.error(f"Session error after {time.time() - start_time:.2f}s: {e}")
            raise
        finally:
            await session.close()

    async def health_check(self, skip_vector_check: bool = False) -> bool:
        """
        Verify database connectivity and optionally pgvector extension.

        Checks both writer (primary) and reader (replica) connections.

        Args:
            skip_vector_check: If True, skip pgvector extension check (faster for startup)

        Returns:
            True if healthy, raises exception otherwise
        """
        if not self._writer_engine:
            raise DatabaseConnectionError("Database writer engine not initialized")

        try:
            # Check writer (primary) connection
            async with self._writer_engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1

                # Verify pgvector extension (optional - skip during startup)
                if not skip_vector_check:
                    result = await conn.execute(
                        text("SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1")
                    )
                    has_pgvector = result.scalar() == 1

                    if not has_pgvector:
                        logger.warning("pgvector extension not installed")
                    else:
                        logger.debug("pgvector extension verified on primary")

            # Check reader (replica) connection if separate
            if self._has_separate_replica and self._reader_engine:
                async with self._reader_engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    assert result.scalar() == 1
                    logger.debug("Read replica connection verified")

            return True

        except (OperationalError, DBAPIError) as e:
            logger.error(f"Database health check failed: {e}")
            raise DatabaseConnectionError("Database health check failed", cause=e) from e

    def get_pool_status(self) -> dict:
        """Get connection pool status for monitoring."""
        if not self._writer_engine:
            return {"status": "not_initialized"}

        def _get_pool_stats(pool) -> dict:
            return {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total_connections": pool.size() + pool.overflow(),
            }

        writer_pool = self._writer_engine.pool
        result = {
            "writer": _get_pool_stats(writer_pool),
            "has_separate_replica": self._has_separate_replica,
        }

        if self._has_separate_replica and self._reader_engine:
            reader_pool = self._reader_engine.pool
            result["reader"] = _get_pool_stats(reader_pool)
        else:
            result["reader"] = result["writer"]  # Same pool

        return result

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide async database session for write operations with automatic cleanup.

        Uses the writer (primary) connection pool for transactional operations.

        Usage:
            async with db_manager.session() as session:
                result = await session.execute(query)
                await session.commit()

        Yields:
            AsyncSession: Database session bound to writer engine
        """
        if not self._writer_session_factory:
            raise DatabaseConnectionError("Database not initialized")

        session = self._writer_session_factory()
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
    async def read_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide async database session for read-only operations.

        Uses the reader (replica) connection pool to offload read queries
        from the primary database. Falls back to primary if no replica configured.

        Usage:
            async with db_manager.read_session() as session:
                result = await session.execute(query)
                # No commit needed for read-only operations

        Yields:
            AsyncSession: Database session bound to reader engine
        """
        if not self._reader_session_factory:
            raise DatabaseConnectionError("Database not initialized")

        session = self._reader_session_factory()
        try:
            yield session
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
                # Sanitize query preview to avoid leaking sensitive data
                query_preview = query[:200].replace("'", "<redacted>") if "'" in query else query[:200]
                raise DatabaseQueryTimeoutError(
                    "Query execution failed", query_preview=query_preview, cause=e
                ) from e

    # -------------------------------------------------------------------------
    # Convenience helpers expected by repositories
    # -------------------------------------------------------------------------
    async def execute(self, query, params: Optional[dict] = None):
        """
        Execute a SQLAlchemy Core query and return the execution result.

        Note: For INSERT/UPDATE without RETURNING, the result will not contain
        row data. Callers that need created IDs should set them explicitly or
        use RETURNING.
        """
        async with self.session() as session:
            try:
                result = await session.execute(query, params or {})
                return result
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseQueryTimeoutError(
                    "Query execution failed", query_preview=str(query)[:200], cause=e
                ) from e

    async def fetch_one(self, query):
        """
        Execute a SELECT and return a single row mapping or None.

        Uses read replica connection pool by default for read scalability.
        """
        async with self.read_session() as session:
            try:
                result = await session.execute(query)
                row = result.fetchone()
                return row._mapping if row else None
            except Exception as e:
                logger.error(f"fetch_one failed: {e}")
                raise DatabaseQueryTimeoutError(
                    "Query execution failed", query_preview=str(query)[:200], cause=e
                ) from e

    async def fetch_all(self, query):
        """
        Execute a SELECT and return a list of row mappings.

        Uses read replica connection pool by default for read scalability.
        """
        async with self.read_session() as session:
            try:
                result = await session.execute(query)
                rows = result.fetchall()
                return [row._mapping for row in rows]
            except Exception as e:
                logger.error(f"fetch_all failed: {e}")
                raise DatabaseQueryTimeoutError(
                    "Query execution failed", query_preview=str(query)[:200], cause=e
                ) from e

    @property
    def engine(self) -> AsyncEngine:
        """
        Get database engine (raises if not initialized).

        Backward compatibility: returns writer engine.
        Prefer using writer_engine or reader_engine explicitly.
        """
        return self.writer_engine

    @property
    def writer_engine(self) -> AsyncEngine:
        """Get writer (primary) database engine."""
        if not self._writer_engine:
            raise DatabaseConnectionError("Database writer engine not initialized")
        return self._writer_engine

    @property
    def reader_engine(self) -> AsyncEngine:
        """Get reader (replica) database engine."""
        if not self._reader_engine:
            raise DatabaseConnectionError("Database reader engine not initialized")
        return self._reader_engine


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


def reset_db_manager() -> None:
    """Clear the process-local database manager after its engines are disposed."""
    global _db_manager
    _db_manager = None


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
    "reset_db_manager",
    "init_database",
    "close_database",
    "get_session",
    "ensure_pgvector_extension",
]
