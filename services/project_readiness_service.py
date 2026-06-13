"""
Project readiness checks.

This service computes whether a project is operationally ready for generation
and publishing without persisting new state. Checks are deliberately isolated so
one degraded dependency cannot hide the rest of the readiness report.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from core.models import Project

if TYPE_CHECKING:
    from config.settings import Settings
    from infrastructure.database import DatabaseManager
    from infrastructure.redis_client import RedisClient
    from knowledge.project_repository import ProjectRepository


ReadinessSeverity = str
ReadinessStatus = str


@dataclass(frozen=True)
class ReadinessCheck:
    id: str
    label: str
    status: ReadinessStatus
    severity: ReadinessSeverity
    message: str
    remediation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "severity": self.severity,
            "message": self.message,
            "remediation": self.remediation,
        }


class ProjectReadinessService:
    """Compute project readiness from existing project and system state."""

    def __init__(
        self,
        database: DatabaseManager,
        redis: RedisClient,
        settings: Settings,
        semantic_analyzer: Any = None,
        project_repository: ProjectRepository | None = None,
    ):
        self.database = database
        self.redis = redis
        self.settings = settings
        self.semantic_analyzer = semantic_analyzer
        if project_repository is None:
            from knowledge.project_repository import ProjectRepository

            project_repository = ProjectRepository(database)
        self.projects = project_repository

    async def get_project_readiness(self, project_id: UUID) -> dict[str, Any]:
        project = await self.projects.get_by_id(project_id)
        if not project:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Project not found")

        checks = await asyncio.gather(
            self._check_project_profile(project),
            self._check_rulebook(project),
            self._check_wordpress(project),
            self._check_provider(),
            self._check_redis(),
            self._check_celery_workers(),
            self._check_daily_budget(),
            self._check_recent_task_failures(project),
        )

        flat_checks = [check for group in checks for check in group]
        blocking = [c for c in flat_checks if c.severity == "blocking" and c.status != "pass"]
        warnings = [c for c in flat_checks if c.severity == "warning" and c.status != "pass"]

        can_generate = not any(c.id in {"provider", "redis", "celery"} for c in blocking)
        can_publish = can_generate and not any(c.id == "wordpress" for c in blocking)

        if blocking:
            status = "blocked"
        elif warnings:
            status = "warning"
        else:
            status = "ready"

        return {
            "project_id": str(project.id),
            "status": status,
            "can_generate": can_generate,
            "can_publish": can_publish,
            "blocking_items": [c.as_dict() for c in blocking],
            "warnings": [c.as_dict() for c in warnings],
            "checks": [c.as_dict() for c in flat_checks],
            "manager_actions": self._manager_actions(project, flat_checks),
            "last_checked_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _check_project_profile(self, project: Project) -> list[ReadinessCheck]:
        if not project.name.strip():
            return [
                ReadinessCheck(
                    id="project_profile",
                    label="Project profile",
                    status="fail",
                    severity="blocking",
                    message="Project name is missing.",
                    remediation="Set a project name before generation.",
                )
            ]

        if not project.domain:
            return [
                ReadinessCheck(
                    id="project_profile",
                    label="Project profile",
                    status="warn",
                    severity="warning",
                    message="Project domain is missing.",
                    remediation="Add a domain to improve analysis, SEO context, and publishing readiness.",
                )
            ]

        return [
            ReadinessCheck(
                id="project_profile",
                label="Project profile",
                status="pass",
                severity="info",
                message="Project identity is configured.",
            )
        ]

    async def _check_rulebook(self, project: Project) -> list[ReadinessCheck]:
        try:
            async with self.database.read_session() as session:
                from knowledge.rulebook_manager import RulebookManager

                manager = RulebookManager(session, self.semantic_analyzer)
                rulebook = await manager.get_latest_rulebook(project.id)

            if not rulebook:
                return [
                    ReadinessCheck(
                        id="rulebook",
                        label="Content rules",
                        status="warn",
                        severity="warning",
                        message="No content rulebook is configured.",
                        remediation="Add brand voice, SEO, and editorial rules for more consistent output.",
                    )
                ]

            if len(rulebook.raw_content.strip()) < 80:
                return [
                    ReadinessCheck(
                        id="rulebook",
                        label="Content rules",
                        status="warn",
                        severity="warning",
                        message="The content rulebook is very short.",
                        remediation="Add concrete tone, structure, forbidden terms, and SEO preferences.",
                    )
                ]

            return [
                ReadinessCheck(
                    id="rulebook",
                    label="Content rules",
                    status="pass",
                    severity="info",
                    message="Content rulebook is configured.",
                )
            ]
        except Exception as e:
            logger.warning(f"Project readiness rulebook check failed: {e}")
            return [
                ReadinessCheck(
                    id="rulebook",
                    label="Content rules",
                    status="warn",
                    severity="warning",
                    message="Could not verify content rulebook state.",
                    remediation="Open the rulebook tab and refresh the project.",
                )
            ]

    async def _check_wordpress(self, project: Project) -> list[ReadinessCheck]:
        missing = []
        if not project.wordpress_url:
            missing.append("URL")
        if not project.wordpress_username:
            missing.append("username")
        if not project.wordpress_app_password:
            missing.append("application password")

        if missing:
            return [
                ReadinessCheck(
                    id="wordpress",
                    label="WordPress publishing",
                    status="fail",
                    severity="blocking",
                    message=f"WordPress is missing: {', '.join(missing)}.",
                    remediation="Complete the WordPress integration before publishing.",
                )
            ]

        return [
            ReadinessCheck(
                id="wordpress",
                label="WordPress publishing",
                status="pass",
                severity="info",
                message="WordPress credentials are configured. Run a connection test before first publish.",
            )
        ]

    async def _check_provider(self) -> list[ReadinessCheck]:
        llm = self.settings.llm
        has_anthropic = llm.anthropic_api_key is not None
        has_openai = llm.openai_api_key is not None

        if not has_anthropic and not has_openai:
            return [
                ReadinessCheck(
                    id="provider",
                    label="AI provider",
                    status="fail",
                    severity="blocking",
                    message="No AI provider API key is configured.",
                    remediation="Configure ANTHROPIC_API_KEY or OPENAI_API_KEY on the backend.",
                )
            ]

        provider = llm.provider.lower()
        if provider == "anthropic" and not has_anthropic and has_openai:
            message = "Configured provider is Anthropic, but only OpenAI credentials are available."
        elif provider == "openai" and not has_openai and has_anthropic:
            message = "Configured provider is OpenAI, but only Anthropic credentials are available."
        else:
            return [
                ReadinessCheck(
                    id="provider",
                    label="AI provider",
                    status="pass",
                    severity="info",
                    message="AI provider credentials are configured.",
                )
            ]

        return [
            ReadinessCheck(
                id="provider",
                label="AI provider",
                status="warn",
                severity="warning",
                message=message,
                remediation="Align LLM_PROVIDER with the available API credentials.",
            )
        ]

    async def _check_redis(self) -> list[ReadinessCheck]:
        try:
            ok = await asyncio.wait_for(self.redis.ping(), timeout=2.0)
        except Exception as e:
            logger.warning(f"Project readiness Redis check failed: {e}")
            ok = False

        if ok:
            return [
                ReadinessCheck(
                    id="redis",
                    label="Redis broker/cache",
                    status="pass",
                    severity="info",
                    message="Redis is reachable.",
                )
            ]

        return [
            ReadinessCheck(
                id="redis",
                label="Redis broker/cache",
                status="fail",
                severity="blocking",
                message="Redis is unavailable.",
                remediation="Restore Redis before dispatching generation tasks.",
            )
        ]

    async def _check_celery_workers(self) -> list[ReadinessCheck]:
        from orchestration.celery_app import app as celery_app

        def inspect_workers() -> Any:
            return celery_app.control.inspect(timeout=1.5).active()

        try:
            loop = asyncio.get_event_loop()
            active_workers = await asyncio.wait_for(
                loop.run_in_executor(None, inspect_workers),
                timeout=2.5,
            )
        except Exception as e:
            logger.warning(f"Project readiness Celery check failed: {e}")
            active_workers = None

        if active_workers:
            return [
                ReadinessCheck(
                    id="celery",
                    label="Worker queue",
                    status="pass",
                    severity="info",
                    message=f"{len(active_workers)} worker(s) are available.",
                )
            ]

        return [
            ReadinessCheck(
                id="celery",
                label="Worker queue",
                status="fail",
                severity="blocking",
                message="No active Celery workers were detected.",
                remediation="Start at least one worker before generation.",
            )
        ]

    async def _check_daily_budget(self) -> list[ReadinessCheck]:
        try:
            from infrastructure.performance_monitor import PerformanceMonitor

            daily_costs = await PerformanceMonitor(self.database, self.redis).check_daily_costs()
        except Exception as e:
            logger.warning(f"Project readiness budget check failed: {e}")
            return [
                ReadinessCheck(
                    id="budget",
                    label="Daily budget",
                    status="warn",
                    severity="warning",
                    message="Could not verify daily budget usage.",
                    remediation="Check monitoring before running a large batch.",
                )
            ]

        if daily_costs.get("status") == "warning":
            return [
                ReadinessCheck(
                    id="budget",
                    label="Daily budget",
                    status="warn",
                    severity="warning",
                    message=(
                        f"Daily AI cost is ${daily_costs.get('total_cost_usd', 0)} "
                        f"against a ${daily_costs.get('threshold_usd', 0)} warning threshold."
                    ),
                    remediation="Review budget before starting another batch.",
                )
            ]

        if daily_costs.get("status") == "error":
            return [
                ReadinessCheck(
                    id="budget",
                    label="Daily budget",
                    status="warn",
                    severity="warning",
                    message="Daily budget check failed.",
                    remediation="Open monitoring to verify cost tracking.",
                )
            ]

        return [
            ReadinessCheck(
                id="budget",
                label="Daily budget",
                status="pass",
                severity="info",
                message="Daily cost is within the configured threshold.",
            )
        ]

    async def _check_recent_task_failures(self, project: Project) -> list[ReadinessCheck]:
        try:
            since = (datetime.now(timezone.utc) - timedelta(hours=24)).replace(tzinfo=None)
            from orchestration.task_persistence import TaskStatus, task_results_table

            query = (
                select(task_results_table)
                .where(task_results_table.c.status == TaskStatus.FAILURE.value)
                .where(task_results_table.c.created_at >= since)
                .order_by(task_results_table.c.created_at.desc())
                .limit(100)
            )
            rows = await self.database.fetch_all(query)
            failures = [
                row for row in rows
                if self._task_row_belongs_to_project(dict(row), project.id)
            ]
        except Exception as e:
            logger.warning(f"Project readiness task failure check failed: {e}")
            return [
                ReadinessCheck(
                    id="recent_failures",
                    label="Recent failures",
                    status="warn",
                    severity="warning",
                    message="Could not verify recent task failures.",
                    remediation="Open task history if generation behavior looks degraded.",
                )
            ]

        if failures:
            return [
                ReadinessCheck(
                    id="recent_failures",
                    label="Recent failures",
                    status="warn",
                    severity="warning",
                    message=f"{len(failures)} failed task(s) found for this project in the last 24 hours.",
                    remediation="Review task history before starting a large batch.",
                )
            ]

        return [
            ReadinessCheck(
                id="recent_failures",
                label="Recent failures",
                status="pass",
                severity="info",
                message="No recent project task failures detected.",
            )
        ]

    def _manager_actions(
        self,
        project: Project,
        checks: list[ReadinessCheck],
    ) -> list[dict[str, Any]]:
        failing_ids = {check.id for check in checks if check.status != "pass"}
        actions: list[dict[str, Any]] = []

        if "wordpress" in failing_ids or project.wordpress_url:
            actions.append(
                {
                    "id": "test_wordpress_connection",
                    "label": "Test WordPress connection",
                    "method": "POST",
                    "endpoint": f"/projects/{project.id}/wordpress/test-connection",
                    "destructive": False,
                }
            )

        if "rulebook" in failing_ids:
            actions.append(
                {
                    "id": "open_rulebook",
                    "label": "Open content rules",
                    "method": "UI",
                    "endpoint": None,
                    "destructive": False,
                }
            )

        return actions

    @staticmethod
    def _task_row_belongs_to_project(row: dict[str, Any], project_id: UUID) -> bool:
        project_id_text = str(project_id)
        args = row.get("args")
        if isinstance(args, list) and args:
            return str(args[0]) == project_id_text
        result = row.get("result")
        if isinstance(result, dict):
            return str(result.get("project_id")) == project_id_text
        return False
