from uuid import uuid4

import pytest

from core.models import Project
from services.project_readiness_service import ProjectReadinessService, ReadinessCheck


class FakeProjectRepository:
    def __init__(self, project):
        self.project = project

    async def get_by_id(self, project_id):
        return self.project if self.project.id == project_id else None


class StubReadinessService(ProjectReadinessService):
    def __init__(self, project, groups):
        super().__init__(
            database=None,
            redis=None,
            settings=None,
            project_repository=FakeProjectRepository(project),
        )
        self.groups = groups

    async def _check_project_profile(self, project):
        return self.groups.get("project_profile", [])

    async def _check_rulebook(self, project):
        return self.groups.get("rulebook", [])

    async def _check_wordpress(self, project):
        return self.groups.get("wordpress", [])

    async def _check_provider(self):
        return self.groups.get("provider", [])

    async def _check_redis(self):
        return self.groups.get("redis", [])

    async def _check_celery_workers(self):
        return self.groups.get("celery", [])

    async def _check_daily_budget(self):
        return self.groups.get("budget", [])

    async def _check_recent_task_failures(self, project):
        return self.groups.get("recent_failures", [])


def check(check_id, status="pass", severity="info"):
    return ReadinessCheck(
        id=check_id,
        label=check_id,
        status=status,
        severity=severity,
        message=f"{check_id} {status}",
    )


@pytest.mark.asyncio
async def test_wordpress_blocker_only_blocks_publishing():
    project = Project(id=uuid4(), name="Test Project", domain="example.com")
    service = StubReadinessService(
        project,
        {
            "project_profile": [check("project_profile")],
            "rulebook": [check("rulebook")],
            "wordpress": [check("wordpress", status="fail", severity="blocking")],
            "provider": [check("provider")],
            "redis": [check("redis")],
            "celery": [check("celery")],
            "budget": [check("budget")],
            "recent_failures": [check("recent_failures")],
        },
    )

    readiness = await service.get_project_readiness(project.id)

    assert readiness["status"] == "blocked"
    assert readiness["can_generate"] is True
    assert readiness["can_publish"] is False
    assert [item["id"] for item in readiness["blocking_items"]] == ["wordpress"]


@pytest.mark.asyncio
async def test_runtime_blocker_blocks_generation_and_publishing():
    project = Project(id=uuid4(), name="Test Project", domain="example.com")
    service = StubReadinessService(
        project,
        {
            "project_profile": [check("project_profile")],
            "rulebook": [check("rulebook")],
            "wordpress": [check("wordpress")],
            "provider": [check("provider")],
            "redis": [check("redis", status="fail", severity="blocking")],
            "celery": [check("celery")],
            "budget": [check("budget")],
            "recent_failures": [check("recent_failures")],
        },
    )

    readiness = await service.get_project_readiness(project.id)

    assert readiness["status"] == "blocked"
    assert readiness["can_generate"] is False
    assert readiness["can_publish"] is False
    assert [item["id"] for item in readiness["blocking_items"]] == ["redis"]


@pytest.mark.asyncio
async def test_warnings_do_not_block_capabilities():
    project = Project(id=uuid4(), name="Test Project", domain="example.com")
    service = StubReadinessService(
        project,
        {
            "project_profile": [check("project_profile")],
            "rulebook": [check("rulebook", status="warn", severity="warning")],
            "wordpress": [check("wordpress")],
            "provider": [check("provider")],
            "redis": [check("redis")],
            "celery": [check("celery")],
            "budget": [check("budget")],
            "recent_failures": [check("recent_failures")],
        },
    )

    readiness = await service.get_project_readiness(project.id)

    assert readiness["status"] == "warning"
    assert readiness["can_generate"] is True
    assert readiness["can_publish"] is True
    assert [item["id"] for item in readiness["warnings"]] == ["rulebook"]
