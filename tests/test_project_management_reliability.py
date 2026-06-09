from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from api.routes.projects import UpdateProjectRequest
from core.exceptions import ProjectNotFoundError
from core.models import Project
from services.project_service import ProjectService


class FakeProjectRepository:
    def __init__(self, project=None, impact=None):
        self.project = project
        self.impact = impact or {
            "articles": 0,
            "content_plans": 0,
            "rulebooks": 0,
            "inferred_patterns": 0,
            "active_tasks": 0,
        }
        self.updated = None
        self.hard_deleted = False
        self.soft_deleted = False

    async def get_by_id(self, project_id):
        if self.project and self.project.id == project_id:
            return self.project
        return None

    async def update(self, project_id, updates):
        self.updated = updates
        if not self.project or self.project.id != project_id:
            return None
        return self.project.model_copy(update=updates)

    async def get_deletion_impact(self, project_id):
        return self.impact

    async def hard_delete(self, project_id):
        self.hard_deleted = True
        return True

    async def soft_delete(self, project_id):
        self.soft_deleted = True
        return True


def make_service(repository):
    return ProjectService(
        database_manager=SimpleNamespace(),
        semantic_analyzer=SimpleNamespace(),
        project_repository=repository,
    )


def test_update_request_normalizes_project_fields():
    request = UpdateProjectRequest(
        name="  Updated Project  ",
        domain="https://example.com/",
        description="  Description  ",
        wordpress_url="  ",
    )

    assert request.name == "Updated Project"
    assert request.domain == "example.com"
    assert request.description == "Description"
    assert request.wordpress_url is None


@pytest.mark.asyncio
async def test_update_project_uses_injected_repository_and_returns_updated_project():
    project = Project(id=uuid4(), name="Before", domain="before.example")
    repository = FakeProjectRepository(project)
    service = make_service(repository)

    result = await service.update_project(
        project.id,
        {"name": "After", "domain": "after.example"},
    )

    assert repository.updated == {"name": "After", "domain": "after.example"}
    assert result["project"]["name"] == "After"
    assert result["project"]["domain"] == "after.example"


@pytest.mark.asyncio
async def test_update_project_rejects_missing_project():
    service = make_service(FakeProjectRepository())

    with pytest.raises(ProjectNotFoundError):
        await service.update_project(uuid4(), {"name": "Missing"})


@pytest.mark.asyncio
async def test_delete_project_blocks_active_generation_tasks():
    project = Project(id=uuid4(), name="Busy")
    repository = FakeProjectRepository(
        project,
        impact={
            "articles": 0,
            "content_plans": 0,
            "rulebooks": 0,
            "inferred_patterns": 0,
            "active_tasks": 1,
        },
    )
    service = make_service(repository)

    with pytest.raises(HTTPException) as error:
        await service.delete_project(project.id, cascade=True)

    assert error.value.status_code == 409
    assert repository.hard_deleted is False


@pytest.mark.asyncio
async def test_delete_project_requires_confirmation_for_related_content():
    project = Project(id=uuid4(), name="Has content")
    repository = FakeProjectRepository(
        project,
        impact={
            "articles": 2,
            "content_plans": 1,
            "rulebooks": 1,
            "inferred_patterns": 0,
            "active_tasks": 0,
        },
    )
    service = make_service(repository)

    with pytest.raises(HTTPException) as error:
        await service.delete_project(project.id, cascade=False)

    assert error.value.status_code == 409
    assert repository.soft_deleted is False


@pytest.mark.asyncio
async def test_confirmed_delete_removes_project_and_dependencies():
    project = Project(id=uuid4(), name="Delete me")
    repository = FakeProjectRepository(
        project,
        impact={
            "articles": 2,
            "content_plans": 1,
            "rulebooks": 1,
            "inferred_patterns": 1,
            "active_tasks": 0,
        },
    )
    service = make_service(repository)

    await service.delete_project(project.id, cascade=True)

    assert repository.hard_deleted is True
    assert repository.soft_deleted is False
