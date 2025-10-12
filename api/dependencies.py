"""
API Dependencies: FastAPI Dependency Injection Helpers

Centralized dependency injection functions for clean separation of concerns.
Uses FastAPI's native dependency injection system with app.state.
"""

from fastapi import Depends, Request

from infrastructure.database import DatabaseManager
from infrastructure.redis_client import RedisClient
from orchestration.content_agent import ContentAgent
from orchestration.task_queue import TaskManager
from knowledge.project_repository import ProjectRepository


async def get_db_manager(request: Request) -> DatabaseManager:
    """Dependency injection: Get database manager."""
    return request.app.state.db


async def get_redis_client(request: Request) -> RedisClient:
    """Dependency injection: Get Redis client."""
    return request.app.state.redis


async def get_content_agent(request: Request) -> ContentAgent:
    """Dependency injection: Get content agent."""
    return request.app.state.content_agent


async def get_task_manager(request: Request) -> TaskManager:
    """Dependency injection: Get task manager."""
    return request.app.state.task_manager


async def get_project_repository(request: Request) -> ProjectRepository:
    """Dependency injection: Get project repository."""
    return request.app.state.projects
