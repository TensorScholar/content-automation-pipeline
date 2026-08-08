"""Google Search Console read-only OAuth and synchronization endpoints."""

from __future__ import annotations

from datetime import date
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from api.dependencies import get_search_console_service
from security import User, get_current_active_user, is_manager_user
from services.search_console_service import SearchConsoleError, SearchConsoleService

router = APIRouter(tags=["Search Console"])


class SearchConsolePropertySelection(BaseModel):
    site_url: str = Field(..., min_length=4, max_length=1000)


class SearchConsoleSyncRequest(BaseModel):
    date_from: date | None = None
    date_to: date | None = None


def _require_manager(user: User) -> None:
    if not is_manager_user(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Manager access is required for Search Console configuration",
        )


def _callback_redirect(base_url: str, **params: str) -> RedirectResponse:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({key: value for key, value in params.items() if value is not None})
    target = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    return RedirectResponse(target, status_code=status.HTTP_303_SEE_OTHER)


@router.get("/search-console/oauth/callback", include_in_schema=True)
async def search_console_oauth_callback(
    state: str = Query(""),
    code: str | None = Query(None),
    error: str | None = Query(None),
    service: SearchConsoleService = Depends(get_search_console_service),
):
    """Consume a one-time OAuth state and return the browser to the frontend."""
    return_url = service.settings.search_console.frontend_return_url
    try:
        result = await service.handle_oauth_callback(state_value=state, code=code, error=error)
        return _callback_redirect(
            return_url,
            search_console="connected",
            project_id=str(result["project_id"]),
        )
    except SearchConsoleError as exc:
        return _callback_redirect(
            return_url,
            search_console="error",
            category=exc.category,
            message=exc.safe_message,
        )


@router.post("/projects/{project_id:uuid}/search-console/connect")
async def connect_search_console(
    project_id: UUID,
    service: SearchConsoleService = Depends(get_search_console_service),
    user: User = Depends(get_current_active_user),
):
    _require_manager(user)
    return await service.create_authorization_url(project_id=project_id, user_id=UUID(user.id))


@router.get("/projects/{project_id:uuid}/search-console/status")
async def get_search_console_status(
    project_id: UUID,
    service: SearchConsoleService = Depends(get_search_console_service),
    user: User = Depends(get_current_active_user),
):
    del user
    return await service.get_status(project_id)


@router.post("/projects/{project_id:uuid}/search-console/properties/refresh")
async def refresh_search_console_properties(
    project_id: UUID,
    service: SearchConsoleService = Depends(get_search_console_service),
    user: User = Depends(get_current_active_user),
):
    _require_manager(user)
    try:
        properties = await service.refresh_properties(project_id)
        return {"properties": properties, "count": len(properties)}
    except SearchConsoleError as exc:
        raise HTTPException(status_code=exc.status_code, detail={
            "message": exc.safe_message,
            "category": exc.category,
            "retryable": exc.retryable,
        }) from exc


@router.put("/projects/{project_id:uuid}/search-console/property")
async def select_search_console_property(
    project_id: UUID,
    request: SearchConsolePropertySelection,
    service: SearchConsoleService = Depends(get_search_console_service),
    user: User = Depends(get_current_active_user),
):
    _require_manager(user)
    return await service.select_property(project_id=project_id, site_url=request.site_url.strip())


@router.post(
    "/projects/{project_id:uuid}/search-console/sync",
    status_code=status.HTTP_202_ACCEPTED,
)
async def queue_search_console_sync(
    project_id: UUID,
    request: SearchConsoleSyncRequest,
    service: SearchConsoleService = Depends(get_search_console_service),
    user: User = Depends(get_current_active_user),
):
    _require_manager(user)
    return await service.queue_sync(
        project_id=project_id,
        date_from=request.date_from,
        date_to=request.date_to,
    )


@router.get("/projects/{project_id:uuid}/search-console/sync-runs")
async def list_search_console_sync_runs(
    project_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    service: SearchConsoleService = Depends(get_search_console_service),
    user: User = Depends(get_current_active_user),
):
    del user
    rows = await service.list_sync_runs(project_id, limit=limit)
    return {"items": rows, "count": len(rows)}


@router.post("/projects/{project_id:uuid}/search-console/disconnect")
async def disconnect_search_console(
    project_id: UUID,
    service: SearchConsoleService = Depends(get_search_console_service),
    user: User = Depends(get_current_active_user),
):
    _require_manager(user)
    return await service.disconnect(project_id)
