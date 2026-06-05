"""
Synchronous Database Manager for Celery Workers
================================================

Provides synchronous database operations for Celery tasks running in
multiprocessing mode. This avoids async event loop conflicts that occur
when using the async DatabaseManager in forked worker processes.

Key Features:
- Synchronous connection pooling using psycopg2
- Thread-safe operations for concurrent task execution
- Compatible with Celery's prefork worker pool
- Mirrors essential operations from async DatabaseManager

Usage:
    from infrastructure.sync_database import SyncDatabaseManager

    db = SyncDatabaseManager()
    db.initialize()

    with db.session() as session:
        result = session.execute("SELECT * FROM projects")
"""

import contextlib
import threading
from typing import Any, Dict, Generator, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import psycopg2
import psycopg2.pool
from loguru import logger
from psycopg2.extras import RealDictCursor

from config.settings import get_settings


class SyncDatabaseManager:
    """
    Synchronous database manager for Celery workers.

    Uses psycopg2 connection pooling to provide thread-safe database
    access without async/await complications.
    """

    def __init__(self):
        """Initialize the sync database manager."""
        self.settings = get_settings()
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._initialized = False

    def initialize(self, min_connections: int = 2, max_connections: int = 10) -> None:
        """
        Initialize the connection pool.

        Args:
            min_connections: Minimum number of connections to maintain
            max_connections: Maximum number of connections allowed
        """
        if self._initialized:
            logger.warning("SyncDatabaseManager already initialized")
            return

        try:
            # Convert async database URL to sync format
            # postgresql+asyncpg://... -> postgresql://...
            db_url = str(self.settings.database.url)
            if db_url.startswith("postgresql+asyncpg://"):
                db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

            # Clean up query parameters (remove 'ssl' which is asyncpg specific)
            # psycopg2 uses 'sslmode' instead
            try:
                parsed = urlparse(db_url)
                query_params = parse_qs(parsed.query)

                if "ssl" in query_params:
                    ssl_val = query_params.pop("ssl")[0]
                    # Map asyncpg 'ssl' to psycopg2 'sslmode' if needed
                    if "sslmode" not in query_params:
                        if ssl_val in ("disable", "false", "0"):
                            query_params["sslmode"] = ["disable"]
                        elif ssl_val in ("require", "true", "1"):
                            query_params["sslmode"] = ["require"]

                # Reconstruct URL
                new_query = urlencode(query_params, doseq=True)
                parsed = parsed._replace(query=new_query)
                sync_db_url = urlunparse(parsed)
            except Exception as parse_error:
                logger.warning(f"Failed to parse/clean DB URL, using raw replacement: {parse_error}")
                sync_db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

            # Create threaded connection pool
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                dsn=sync_db_url,
                cursor_factory=RealDictCursor,  # Return dicts instead of tuples
            )

            self._initialized = True
            logger.info(
                f"Sync database pool initialized | min={min_connections} max={max_connections}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize sync database pool: {e}")
            raise

    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._initialized = False
            logger.info("Sync database pool closed")

    @contextlib.contextmanager
    def get_connection(self) -> Generator:
        """
        Get a connection from the pool.

        Yields:
            psycopg2 connection with RealDictCursor

        Example:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT * FROM projects")
                    results = cursor.fetchall()
        """
        if not self._initialized:
            raise RuntimeError("SyncDatabaseManager not initialized")

        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise
        finally:
            self._pool.putconn(conn)

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
    ) -> Optional[Any]:
        """
        Execute a query and optionally fetch results.

        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: If True, fetch and return one result
            fetch_all: If True, fetch and return all results

        Returns:
            Query results if fetch_one or fetch_all is True, otherwise None

        Example:
            # Insert
            db.execute(
                "INSERT INTO projects (name, domain) VALUES (%(name)s, %(domain)s)",
                {"name": "Test", "domain": "example.com"}
            )

            # Select one
            project = db.execute(
                "SELECT * FROM projects WHERE id = %(id)s",
                {"id": project_id},
                fetch_one=True
            )

            # Select all
            projects = db.execute(
                "SELECT * FROM projects",
                fetch_all=True
            )
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or {})

                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                else:
                    return None

    def execute_many(
        self, query: str, params_list: list[Dict[str, Any]]
    ) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query to execute
            params_list: List of parameter dictionaries

        Example:
            db.execute_many(
                "INSERT INTO task_results (task_id, status) VALUES (%(task_id)s, %(status)s)",
                [
                    {"task_id": "task1", "status": "pending"},
                    {"task_id": "task2", "status": "pending"},
                ]
            )
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                for params in params_list:
                    cursor.execute(query, params)

    def ping(self) -> bool:
        """
        Check if database connection is alive.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            return False


# Global instance for Celery workers
_sync_db_manager: Optional[SyncDatabaseManager] = None
_sync_db_lock = threading.Lock()


def get_sync_db() -> SyncDatabaseManager:
    """
    Get the global sync database manager instance (thread-safe).

    Returns:
        Initialized SyncDatabaseManager instance

    Example:
        from infrastructure.sync_database import get_sync_db

        db = get_sync_db()
        projects = db.execute("SELECT * FROM projects", fetch_all=True)
    """
    global _sync_db_manager

    if _sync_db_manager is None:
        with _sync_db_lock:
            # Double-checked locking to prevent duplicate initialization
            if _sync_db_manager is None:
                _sync_db_manager = SyncDatabaseManager()
                _sync_db_manager.initialize()

    return _sync_db_manager
