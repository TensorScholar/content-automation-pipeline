"""
Task Result Persistence

Stores Celery task execution results in database for audit trail,
debugging, and result querying. Enables long-term task history beyond
Redis/result backend expiration.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from infrastructure.database import DatabaseManager
from infrastructure.schema import Table, metadata


class TaskStatus(str, Enum):
    """Task execution status states."""
    
    PENDING = "pending"
    STARTED = "started"
    RETRY = "retry"
    SUCCESS = "success"
    FAILURE = "failure"
    REVOKED = "revoked"


# Task Results Table
task_results_table = Table(
    "task_results",
    metadata,
    Column("id", PG_UUID, primary_key=True, default=uuid4),
    Column("task_id", String(255), unique=True, nullable=False, index=True),
    Column("task_name", String(255), nullable=False, index=True),
    Column("idempotency_key", String(255), index=True),
    Column("status", String(50), nullable=False, index=True),
    Column("args", JSON),
    Column("kwargs", JSON),
    Column("result", JSON),
    Column("error", Text),
    Column("traceback", Text),
    Column("start_time", DateTime),
    Column("end_time", DateTime),
    Column("duration_seconds", Float),
    Column("retry_count", Integer, default=0),
    Column("worker_name", String(255)),
    Column("created_at", DateTime, default=lambda: datetime.now(timezone.utc)),
    Column("updated_at", DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)),
)


class TaskResultRepository:
    """
    Repository for persisting and querying task execution results.
    
    Provides audit trail and debugging capabilities for async tasks.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize repository with database manager.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    async def create_task_record(
        self,
        task_id: str,
        task_name: str,
        args: tuple,
        kwargs: dict,
        idempotency_key: Optional[str] = None,
    ) -> UUID:
        """
        Create initial task record when task starts.
        
        Args:
            task_id: Celery task ID
            task_name: Task name
            args: Task positional arguments
            kwargs: Task keyword arguments
            idempotency_key: Optional idempotency key
            
        Returns:
            Database record UUID
        """
        from sqlalchemy import insert
        
        query = insert(task_results_table).values(
            task_id=task_id,
            task_name=task_name,
            idempotency_key=idempotency_key,
            status=TaskStatus.STARTED,
            args=list(args),
            kwargs=kwargs,
            start_time=datetime.now(timezone.utc),
            retry_count=0,
        ).returning(task_results_table.c.id)
        
        result = await self.db.execute(query)
        return result.get("id")
    
    async def update_task_success(
        self,
        task_id: str,
        result: Dict[str, Any],
    ) -> bool:
        """
        Update task record on successful completion.
        
        Args:
            task_id: Celery task ID
            result: Task result data
            
        Returns:
            True if updated successfully
        """
        from sqlalchemy import update
        
        end_time = datetime.now(timezone.utc)
        
        query = (
            update(task_results_table)
            .where(task_results_table.c.task_id == task_id)
            .values(
                status=TaskStatus.SUCCESS,
                result=result,
                end_time=end_time,
                updated_at=end_time,
            )
        )
        
        # Calculate duration if start_time exists
        from sqlalchemy import select
        start_query = select(task_results_table.c.start_time).where(
            task_results_table.c.task_id == task_id
        )
        start_result = await self.db.fetch_one(start_query)
        
        if start_result and start_result.get("start_time"):
            duration = (end_time - start_result["start_time"]).total_seconds()
            query = query.values(duration_seconds=duration)
        
        await self.db.execute(query)
        return True
    
    async def update_task_failure(
        self,
        task_id: str,
        error: str,
        traceback: Optional[str] = None,
    ) -> bool:
        """
        Update task record on failure.
        
        Args:
            task_id: Celery task ID
            error: Error message
            traceback: Optional exception traceback
            
        Returns:
            True if updated successfully
        """
        from sqlalchemy import update
        
        end_time = datetime.now(timezone.utc)
        
        query = (
            update(task_results_table)
            .where(task_results_table.c.task_id == task_id)
            .values(
                status=TaskStatus.FAILURE,
                error=error,
                traceback=traceback,
                end_time=end_time,
                updated_at=end_time,
            )
        )
        
        await self.db.execute(query)
        return True
    
    async def increment_retry_count(self, task_id: str) -> int:
        """
        Increment retry count for task.
        
        Args:
            task_id: Celery task ID
            
        Returns:
            New retry count
        """
        from sqlalchemy import select, update
        
        # Get current count
        query = select(task_results_table.c.retry_count).where(
            task_results_table.c.task_id == task_id
        )
        result = await self.db.fetch_one(query)
        current_count = result.get("retry_count", 0) if result else 0
        
        new_count = current_count + 1
        
        # Update count and status
        update_query = (
            update(task_results_table)
            .where(task_results_table.c.task_id == task_id)
            .values(
                retry_count=new_count,
                status=TaskStatus.RETRY,
                updated_at=datetime.now(timezone.utc),
            )
        )
        
        await self.db.execute(update_query)
        return new_count
    
    async def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task record by task ID.
        
        Args:
            task_id: Celery task ID
            
        Returns:
            Task record dict or None
        """
        from sqlalchemy import select
        
        query = select(task_results_table).where(
            task_results_table.c.task_id == task_id
        )
        
        result = await self.db.fetch_one(query)
        return dict(result) if result else None
    
    async def get_tasks_by_status(
        self,
        status: TaskStatus,
        limit: int = 100,
    ) -> list[Dict[str, Any]]:
        """
        Retrieve tasks by status.
        
        Args:
            status: Task status to filter by
            limit: Maximum results to return
            
        Returns:
            List of task records
        """
        from sqlalchemy import select
        
        query = (
            select(task_results_table)
            .where(task_results_table.c.status == status)
            .order_by(task_results_table.c.created_at.desc())
            .limit(limit)
        )
        
        results = await self.db.fetch_all(query)
        return [dict(row) for row in results]
    
    async def get_failed_tasks(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Dict[str, Any]]:
        """
        Retrieve failed tasks for debugging.
        
        Args:
            since: Optional datetime to filter from
            limit: Maximum results
            
        Returns:
            List of failed task records
        """
        from sqlalchemy import select
        
        query = (
            select(task_results_table)
            .where(task_results_table.c.status == TaskStatus.FAILURE)
        )
        
        if since:
            query = query.where(task_results_table.c.created_at >= since)
        
        query = query.order_by(task_results_table.c.created_at.desc()).limit(limit)
        
        results = await self.db.fetch_all(query)
        return [dict(row) for row in results]
